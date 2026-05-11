import json
import argparse
import signal
from pathlib import Path
import gc
import torch

import numpy as np
from loguru import logger
from vllm import LLM
from tqdm import tqdm

from representations import ALL_REPRESENTATIONS


class Embedder:
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        chunk_size: int = 1000,
        embed_timeout: int = 500,
        dtype: str = "float16",
    ):
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.chunk_size = chunk_size
        self.embed_timeout = embed_timeout
        self.dtype = dtype
        self._model = None
        self._init_model()

    def _init_model(self):
        if self._model is not None:
            self.unload_model()
        self._model = LLM(
            model=self.model_name,
            runner="pooling",
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            trust_remote_code=True,
        )

    def unload_model(self):
        try:
            if hasattr(self._model.llm_engine, "engine_core"):
                self._model.llm_engine.engine_core.shutdown()
        except Exception as e:
            logger.warning(f"Engine shutdown error (non-fatal): {e}")
        del self._model
        self._model = None
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Model unloaded and GPU memory freed")
    
    @staticmethod
    def _discover_generated_descriptions(representations_dir: str) -> dict[str, dict[str, str]]:
        """Scan for generated description JSONL files.

        Expected layout: <dir>/<mode>/<source>/[annotator_dir/]<mode>_<source>.jsonl
        Returns: {mode: {source_key: filepath}}
        """
        rdir = Path(representations_dir)
        known_modes = sorted(ALL_REPRESENTATIONS.keys(), key=len, reverse=True)
        discovered = {}

        for mode in known_modes:
            mode_dir = rdir / mode
            if not mode_dir.is_dir():
                continue
            for f in sorted(mode_dir.glob(f"**/{mode}_*.jsonl")):
                rel_dir = str(f.parent.relative_to(mode_dir))
                if rel_dir:
                    discovered.setdefault(mode, {})[rel_dir] = str(f)

        return discovered

    def _save(self, output_path: Path, ids: list[str], embed_map: dict[str, np.ndarray]) -> None:
        available_ids = [id_ for id_ in ids if id_ in embed_map]
        embeddings = np.array([embed_map[id_] for id_ in available_ids])
        np.savez(output_path, ids=np.array(available_ids), embeddings=embeddings)

    def _embed_chunk(self, chunk: list[str]) -> list | None:
        """Embed a chunk with a timeout. Returns outputs or None if frozen."""
        def _handler(signum, frame):
            raise TimeoutError()

        old_handler = signal.signal(signal.SIGALRM, _handler)
        timeout = max(self.embed_timeout, len(chunk) // 100)
        signal.alarm(timeout)
        try:
            outputs = self._model.embed(
                chunk,
                use_tqdm=True,
            )
            signal.alarm(0)
            return outputs
        except TimeoutError:
            logger.warning(f"Embed call timed out after {timeout}s, chunk has a bad sample")
            signal.alarm(0)
            self.unload_model()
            return None
        finally:
            signal.signal(signal.SIGALRM, old_handler)

    def _pre_filter_bad(self, chunk: list[str], c_ids: list[str]) -> tuple[list[str], list[str], list[str]]:
        """Use the tokenizer to filter out samples that will cause the engine to hang."""
        tokenizer = self._model.get_tokenizer()
        good_texts = []
        good_ids = []
        skipped = []
        for text, sid in zip(chunk, c_ids):
            try:
                tokens = tokenizer.encode(text, truncation=True)
                if len(tokens) == 0:
                    logger.warning(f"Skipping empty-tokenization sample: id={sid} chars={len(text)}")
                    skipped.append(sid)
                else:
                    # Decode truncated tokens back to text to avoid vLLM hanging on very long inputs
                    truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)
                    good_texts.append(truncated_text)
                    good_ids.append(sid)
            except Exception as e:
                logger.warning(f"Skipping bad sample (tokenization failed): id={sid} err={e}")
                skipped.append(sid)
        return good_texts, good_ids, skipped

    def _find_and_skip_bad(self, chunk: list[str], c_ids: list[str]) -> tuple[list, list[str]]:
        """Probe samples individually to find bad ones, reiniting the model only once."""
        self._init_model()

        chunk, c_ids, skipped = self._pre_filter_bad(chunk, c_ids)
        if not chunk:
            return [], skipped

        good_outputs = []
        probe_batch_size = 500
        i = 0
        while i < len(chunk):
            batch = chunk[i : i + probe_batch_size]
            batch_ids = c_ids[i : i + probe_batch_size]
            result = self._embed_chunk(batch)
            if result is not None:
                good_outputs.extend(result)
                i += probe_batch_size
            elif len(batch) == 1:
                logger.warning(f"Skipping bad sample: id={batch_ids[0]} chars={len(batch[0])}")
                skipped.append(batch_ids[0])
                self._init_model()
                i += 1
            else:
                self._init_model()
                for text, sid in zip(batch, batch_ids):
                    single_result = self._embed_chunk([text])
                    if single_result is not None:
                        good_outputs.extend(single_result)
                    else:
                        logger.warning(f"Skipping bad sample: id={sid} chars={len(text)}")
                        skipped.append(sid)
                        self._init_model()
                i += probe_batch_size

        return good_outputs, skipped

    def embed(self, texts: dict[str, str], output_path: Path) -> np.ndarray:
        output_path = Path(output_path)
        logger.info(f"Starting embedding for {output_path} ({len(texts)} texts)")
        ids = list(texts.keys())

        cached_map: dict[str, np.ndarray] = {}
        if output_path.exists():
            try:
                data = np.load(output_path)
                cached_ids = list(data["ids"])
                if cached_ids == ids:
                    logger.info(f"Loaded cached embeddings {data['embeddings'].shape} from {output_path}")
                    return data["embeddings"]
                cached_map = dict(zip(cached_ids, data["embeddings"]))
                logger.info(f"Found {len(cached_map)} cached embeddings")
            except Exception as e:
                logger.warning(f"Corrupted cache file {output_path}, starting fresh: {e}")
                output_path.unlink()
                cached_map = {}

        new_ids = [id_ for id_ in ids if id_ not in cached_map]
        logger.info(f"{len(ids)} total, {len(ids) - len(new_ids)} cached, {len(new_ids)} to embed")

        if new_ids:
            sequences = [texts[id_] for id_ in new_ids]

            output_path.parent.mkdir(parents=True, exist_ok=True)
            all_skipped = []
            if self._model is None:
                self._init_model()

            chunks = [sequences[i : i + self.chunk_size] for i in range(0, len(sequences), self.chunk_size)]
            chunk_ids = [new_ids[i : i + self.chunk_size] for i in range(0, len(new_ids), self.chunk_size)]

            for chunk, c_ids in tqdm(zip(chunks, chunk_ids), total=len(chunks), desc="Embedding chunks"):
                outputs = self._embed_chunk(chunk)
                if outputs is not None:
                    chunk_embeddings = np.array([o.outputs.embedding for o in outputs[:len(c_ids)]], dtype=np.float16)
                    cached_map.update(zip(c_ids, chunk_embeddings))
                else:
                    # Binary search for the bad sample(s), reinit engine, embed the rest
                    good_outputs, skipped = self._find_and_skip_bad(chunk, c_ids)
                    all_skipped.extend(skipped)
                    # Map good outputs back to IDs (excluding skipped)
                    good_ids = [id_ for id_ in c_ids if id_ not in set(skipped)]
                    chunk_embeddings = np.array([o.outputs.embedding for o in good_outputs], dtype=np.float16)
                    cached_map.update(zip(good_ids, chunk_embeddings))

                self._save(output_path, ids, cached_map)
                logger.info(f"Checkpoint: {len(cached_map)}/{len(ids)} embeddings saved")

            if all_skipped:
                logger.warning(f"Skipped {len(all_skipped)} bad samples total: {all_skipped}")

        embeddings = np.array([cached_map.get(id_, np.zeros(list(cached_map.values())[0].shape, dtype=np.float16)) for id_ in ids])
        logger.info(f"Final embeddings {embeddings.shape}")
        return embeddings

    def discover_and_embed(self, representations_dir: str) -> None:
        discovered = self._discover_generated_descriptions(representations_dir)
        if not discovered:
            logger.warning(f"No generated description files found in {representations_dir}")
            return

        logger.info(f"Discovered {len(discovered)} generated description files in {representations_dir}")

        for mode, generated_description_files in discovered.items():
            repr_fn = ALL_REPRESENTATIONS[mode]

            groups = {}
            for key, filepath in generated_description_files.items():
                parts = key.split("/", 1)
                source_name = parts[0]
                annotator_key = parts[1] if len(parts) > 1 else ""
                groups.setdefault(annotator_key, {})[source_name] = filepath

            for annotator_key, generated_description_files in groups.items():
                for source_name, filepath in generated_description_files.items():
                    entries = {}
                    with open(filepath) as f:
                        for line in f:
                            if line.strip():
                                entry = json.loads(line)
                                entries[entry["id"]] = entry

                    texts = {eid: repr_fn(e) for eid, e in entries.items() if repr_fn(e)}
                    if not texts:
                        continue

                    source_dir = Path(filepath).parent
                    prefix = "real" if source_name == "real" else "sim"
                    emb_path = source_dir / f"{prefix}_embeddings.npz"
                    logger.info(f"  {mode}/{source_name}: {len(texts)} texts -> {emb_path}")
                    self.embed(texts, emb_path)


def main():
    parser = argparse.ArgumentParser(description="Embed user behavior representations.")
    parser.add_argument("--model_name", type=str, required=True, help="Embedding model name.")
    parser.add_argument("--representations_dir", type=str, required=True, help="Directory with representation files.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--dtype", type=str, default="float16")
    args = parser.parse_args()

    embedder = Embedder(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        chunk_size=args.chunk_size,
        dtype=args.dtype,
    )
    embedder.discover_and_embed(args.representations_dir)
    embedder.unload_model()


if __name__ == "__main__":
    main()
