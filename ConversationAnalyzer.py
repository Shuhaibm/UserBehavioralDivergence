import asyncio
import json
import logging
import os

from json_repair import repair_json
import httpx
from loguru import logger
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential, before_sleep_log
from tqdm import tqdm

from prompts import (
    REQUESTS_FACET_PROMPT_TEMPLATE,
    RESPONSES_FACET_PROMPT_TEMPLATE,
    CONTEXT_FACET_PROMPT_TEMPLATE,
    COMMUNICATION_STYLE_FACET_PROMPT_TEMPLATE,
    DAMSL_FACET_PROMPT_TEMPLATE,
    SGD_FACET_PROMPT_TEMPLATE,
)


class ConversationAnalyzer:
    def __init__(
        self,
        model_name: str,
        api_base: str,
        api_key: str,
        max_tokens: int = 8192,
        max_concurrent: int = 4096,
        max_connections: int = 4096,
    ) -> None:
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.sem = asyncio.Semaphore(max_concurrent)

        models_to_sampling_params = {
            "Qwen/Qwen3.5-122B-A10B-FP8": {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "min_p": 0.0,
                "presence_penalty": 1.5,
                "repetition_penalty": 1.0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            "Qwen/Qwen3.5-9B": {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "min_p": 0.0,
                "presence_penalty": 1.5,
                "repetition_penalty": 1.0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        }

        if model_name not in models_to_sampling_params:
            raise ValueError(f"Unsupported model: {model_name}")
        self.sampling_params = models_to_sampling_params[model_name]

        transport = httpx.AsyncHTTPTransport(
            retries=0,
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
                keepalive_expiry=600.0,
            ),
            http2=True,
        )
        http_client = httpx.AsyncClient(
            transport=transport,
            timeout=httpx.Timeout(None),
            headers={"Connection": "keep-alive"},
        )
        self.client = AsyncOpenAI(
            base_url=api_base,
            api_key=api_key,
            http_client=http_client,
        )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(multiplier=1, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _call_llm(self, prompt: str) -> dict:
        async with self.sem:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                extra_body=self.sampling_params,
            )
            msg = response.choices[0].message
            reasoning = msg.model_extra.get('reasoning') if msg.model_extra else None
            return {
                "reasoning": reasoning,
                "content": msg.content
            }

    async def _analyze_per_utterance(self, conversation: dict, prompt_template: str, field_name: str) -> dict:
        messages = conversation["messages"]
        user_turn_indices = [i for i, m in enumerate(messages) if m["role"] == "user"]

        async def _process_turn(i: int) -> dict:
            prompt = prompt_template.format(
                conversation_history="\n".join(f"{m['role']}: {m['content']}" for m in messages[:i]),
                target_user_utterance=f"{messages[i]['role']}: {messages[i]['content']}",
            )
            return await self._call_llm(prompt)

        results = await asyncio.gather(*[_process_turn(i) for i in user_turn_indices])
        conversation[field_name] = results
        return conversation

    async def analyze_DAMSL(self, conversation: dict) -> dict:
        return await self._analyze_per_utterance(conversation, DAMSL_FACET_PROMPT_TEMPLATE, "DAMSL_facet")

    async def analyze_SGD(self, conversation: dict) -> dict:
        return await self._analyze_per_utterance(conversation, SGD_FACET_PROMPT_TEMPLATE, "SGD_facet")

    async def _analyze_conversation_level(
        self,
        conversation: dict,
        prompt_template: str,
        field_name: str,
        user_utterances_only: bool,
    ) -> dict:
        if user_utterances_only:
            conversation_history = "\n".join(f"{m['role']}: {m['content']}" for m in conversation["messages"] if m["role"] == "user")
        else:
            conversation_history = "\n".join(f"{m['role']}: {m['content']}" for m in conversation["messages"])

        prompt = prompt_template.format(
            user_goal=conversation["intent"],
            conversation_history=conversation_history,
        )
        result = await self._call_llm(prompt)
        processed_result = repair_json(result["content"], return_objects=True)

        def flatten_values(obj):
            if isinstance(obj, dict):
                for v in obj.values():
                    yield from flatten_values(v)
            elif isinstance(obj, list):
                for item in obj:
                    yield from flatten_values(item)
            else:
                yield str(obj)

        stripped_result = "\n".join(flatten_values(processed_result))
        conversation[field_name] = stripped_result
        return conversation

    async def analyze_requests(self, conversation: dict) -> dict:
        return await self._analyze_conversation_level(conversation, REQUESTS_FACET_PROMPT_TEMPLATE, "requests_facet", user_utterances_only=True)

    async def analyze_responses(self, conversation: dict) -> dict:
        return await self._analyze_conversation_level(conversation, RESPONSES_FACET_PROMPT_TEMPLATE, "responses_facet", user_utterances_only=False)

    async def analyze_context(self, conversation: dict) -> dict:
        return await self._analyze_conversation_level(conversation, CONTEXT_FACET_PROMPT_TEMPLATE, "context_facet", user_utterances_only=True)

    async def analyze_communication_style(self, conversation: dict) -> dict:
        return await self._analyze_conversation_level(conversation, COMMUNICATION_STYLE_FACET_PROMPT_TEMPLATE, "communication_style_facet", user_utterances_only=True)

    async def process_conversations(
        self,
        conversations: list[dict],
        output_file: str,
        facet: str,
        max_pending: int = 2048,
    ) -> list[dict]:
        
        if facet == "requests_facet":
            process_fn = self.analyze_requests
        elif facet == "responses_facet":
            process_fn = self.analyze_responses
        elif facet == "context_facet":
            process_fn = self.analyze_context
        elif facet == "communication_style_facet":
            process_fn = self.analyze_communication_style
        elif facet == "DAMSL_facet":
            process_fn = self.analyze_DAMSL
        elif facet == "SGD_facet":
            process_fn = self.analyze_SGD
        else:
            raise ValueError(f"Invalid facet: {facet}")

        seen_ids = set()
        completed_conversations = []
        expected_ids = {c["id"] for c in conversations}
        if os.path.exists(output_file):
            logger.info("Loading checkpoint...")
            with open(output_file, "r") as f:
                for line in f:
                    seen_conversation = json.loads(line.strip())
                    if seen_conversation["id"] not in expected_ids:
                        continue
                    completed_conversations.append(seen_conversation)
                    seen_ids.add(seen_conversation["id"])
            logger.info(f"Continuing from {len(seen_ids)} seen conversations")

        unseen_conversations = [c for c in conversations if c["id"] not in seen_ids]
        if not unseen_conversations:
            return completed_conversations

        logger.info(f"Processing {len(unseen_conversations)} unseen conversations (facet={facet})")
        pending: set[asyncio.Task] = set()
        errors = 0
        it = iter(unseen_conversations)
        pbar = tqdm(total=len(unseen_conversations), desc=f"Analyzing ({facet})")
        f = open(output_file, "a")

        def launch_next() -> bool:
            try:
                conversation = next(it)
            except StopIteration:
                return False

            async def _run(conversation):
                result = await process_fn(conversation)
                f.write(json.dumps(result) + "\n")
                f.flush()
                completed_conversations.append(result)
                pbar.update(1)

            pending.add(asyncio.create_task(_run(conversation)))
            return True

        for _ in range(min(max_pending, len(unseen_conversations))):
            if not launch_next():
                break

        while pending:
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                pending.discard(t)
                try:
                    t.result()
                except Exception as e:
                    errors += 1
                    logger.warning(f"Task failed: {e}")
                launch_next()

        f.close()
        pbar.close()
        await self.client.close()

        logger.info(f"Done. completed={len(completed_conversations)} errors={errors}")
        return completed_conversations
