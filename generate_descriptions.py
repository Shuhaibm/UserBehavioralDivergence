import json
import asyncio
import argparse
from pathlib import Path

from loguru import logger

from ConversationAnalyzer import ConversationAnalyzer
from representations import BUILTIN_REPRESENTATIONS


USER_BEHAVIOR_FACETS = ["requests_facet", "responses_facet", "context_facet", "communication_style_facet", "DAMSL_facet", "SGD_facet"]


def load_conversations(input_file: str, source_type: str) -> list[dict]:
    """Load conversations from a JSONL file.

    source_type="real" uses the original 'messages' field.
    source_type="simulated" swaps 'simulated_conversation' into 'messages'.
    """
    conversations = []
    with open(input_file) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if source_type == "simulated" and "simulated_conversation" in entry:
                entry = {**entry, "messages": entry["simulated_conversation"]}
            conversations.append(entry)
    logger.info(f"Loaded {len(conversations)} conversations (type={source_type})")
    return conversations


def run_builtin(conversations: list[dict], output_file: str, facet: str) -> None:
    repr_fn = BUILTIN_REPRESENTATIONS[facet]
    written = 0
    with open(output_file, "w") as f:
        for entry in conversations:
            text = repr_fn(entry)
            if not text:
                continue
            out = {"id": entry["id"], "messages": entry["messages"], "intent": entry.get("intent", "")}
            f.write(json.dumps(out) + "\n")
            written += 1
    logger.info(f"Wrote {written} entries to {output_file}")


def run_combined(output_dir: Path, dir_name: str, annotator_subdir: str | None) -> list[dict]:
    """Combine representations from all six facets into a single string per conversation."""
    all_facet_data: dict[str, dict[str, str]] = {}
    id_to_base: dict[str, dict] = {}

    for facet in USER_BEHAVIOR_FACETS:
        facet_dir = output_dir / facet / dir_name
        if annotator_subdir:
            facet_dir = facet_dir / annotator_subdir
        facet_file = facet_dir / f"{facet}_{dir_name}.jsonl"

        if not facet_file.exists():
            raise FileNotFoundError(f"Required file not found: {facet_file}\nGenerate descriptions for facet '{facet}' first.")

        logger.info(f"Loading {facet} from {facet_file}")
        with open(facet_file) as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                cid = entry["id"]
                if cid not in id_to_base:
                    id_to_base[cid] = {"id": cid, "messages": entry.get("messages", []), "intent": entry.get("intent", "")}
                    all_facet_data[cid] = {}
                all_facet_data[cid][facet] = entry.get(facet)

    results = []
    for cid, base in id_to_base.items():
        facet_texts = all_facet_data.get(cid, {})
        if any(facet_texts.get(f) is None for f in USER_BEHAVIOR_FACETS):
            continue
        parts = [f"[{f}]\n{facet_texts[f]}" for f in USER_BEHAVIOR_FACETS if facet_texts.get(f)]
        base["combined"] = "\n\n".join(parts)
        results.append(base)
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate user behavior representations.")
    parser.add_argument("--facet", type=str, required=True, choices=list(BUILTIN_REPRESENTATIONS) + USER_BEHAVIOR_FACETS + ["combined"], help="User behavior facet to generate representations for.")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL conversation file.")
    parser.add_argument("--simulator_name", type=str, required=True, help="Name of the simulator (used as output subdirectory).")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--max_pending", type=int, default=4096, help="Max concurrent conversations.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    is_builtin = args.facet in BUILTIN_REPRESENTATIONS

    for source_type, dir_name in [("real", "real"), ("simulated", args.simulator_name)]:
        if is_builtin:
            source_dir = output_dir / args.facet / dir_name
            annotator_subdir = None
        else:
            short_model = args.model_name.split("/")[-1]
            annotator_subdir = f"annotator_{short_model}"
            source_dir = output_dir / args.facet / dir_name / annotator_subdir
        source_dir.mkdir(parents=True, exist_ok=True)
        output_file = str(source_dir / f"{args.facet}_{dir_name}.jsonl")

        logger.info(f"FACET={args.facet} | SOURCE={source_type} | OUTPUT={output_file}")

        if args.facet == "combined":
            results = run_combined(output_dir, dir_name, annotator_subdir)
            with open(output_file, "w") as f:
                for entry in results:
                    f.write(json.dumps(entry) + "\n")
            logger.info(f"Wrote {len(results)} combined entries to {output_file}")
        elif is_builtin:
            conversations = load_conversations(args.input_file, source_type)
            run_builtin(conversations, output_file, args.facet)
        else:
            conversations = load_conversations(args.input_file, source_type)
            analyzer = ConversationAnalyzer(
                model_name=args.model_name,
                api_base=args.api_base,
                api_key=args.api_key,
            )
            asyncio.run(analyzer.process_conversations(
                conversations=conversations,
                output_file=output_file,
                facet=args.facet,
                max_pending=args.max_pending,
            ))

    logger.info(f"Completed generation of user behavior representations descriptions for facet '{args.facet}'")


if __name__ == "__main__":
    main()
