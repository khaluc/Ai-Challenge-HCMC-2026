from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Sequence

try:
    from dotenv import load_dotenv

    load_dotenv()  # picks up DASHSCOPE_API_KEY / ANTHROPIC_API_KEY from a local .env
except ImportError:
    pass

from retrieval.clip_store import Phase1ClipStore
from retrieval.clip_encoder import (
    DEFAULT_CACHE_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    HFCLIPTextEncoder,
)
from retrieval.hybrid_store import Phase1HybridStore
from kis.hybrid_cli import _add_hybrid_arguments, _add_model_arguments, _config_from_args, _positive_int
from retrieval.hybrid_search import HybridTextualKIS
from llm.expansion_retrieval import ExpandedHybridSearch
from llm.query_expansion import QwenQueryUnderstanding

from .answer_scoring import ClipTextEmbedder, score_answers
from .candidates import CandidateConfig, QACandidateSearch
from .io import write_qa_results
from .pipeline import KISVideoQA
from .query_split import QwenQuestionSplitter
from vlm.image_qa import QwenVLAnswerer


DEFAULT_PHASE1_ARTIFACTS = Path("indexes")
DEFAULT_PHASE5_ARTIFACTS = Path("experiments/results/qna")
DEFAULT_DATA_ROOT = Path("data")


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, OSError):
            pass


def _print_json(value: object) -> None:
    _configure_stdout()
    print(json.dumps(value, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qna",
        description="Hybrid-retrieval-narrowed VLM Q&A: question -> top videos -> top frames -> VLM -> rerank.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ask = commands.add_parser("ask", help="Answer one visual question about the dataset")
    ask.add_argument("question")
    ask.add_argument("--query-id", default="qa")
    ask.add_argument("--top-k", type=_positive_int, default=5, help="how many ranked answers to return")
    ask.add_argument("--video-limit", type=_positive_int, default=15)
    ask.add_argument("--frame-limit", type=_positive_int, default=30)
    ask.add_argument("--search-top-k", type=_positive_int, default=100)
    ask.add_argument("--artifacts", type=Path, default=DEFAULT_PHASE1_ARTIFACTS)
    ask.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ask.add_argument("--vlm-model", default="qwen3.8-max")
    ask.add_argument("--vlm-base-url", default=None)
    ask.add_argument(
        "--use-expansion",
        action="store_true",
        help="use Phase 4 LLM query expansion (Qwen) for retrieval instead of plain Phase 3 hybrid search",
    )
    ask.add_argument("--expansion-llm-model", default="qwen3.8-max")
    ask.add_argument(
        "--no-split",
        action="store_true",
        help="skip the LLM scene/question split and use the raw question for both retrieval and the VLM",
    )
    ask.add_argument("--output", type=Path)
    _add_model_arguments(ask)
    _add_hybrid_arguments(ask)

    eval_answers = commands.add_parser(
        "eval-answers",
        help="Semantic answer matching: score predicted vs. ground-truth answers by embedding similarity",
    )
    eval_answers.add_argument(
        "input_csv",
        type=Path,
        help="CSV with columns: question,predicted,ground_truth",
    )
    eval_answers.add_argument("--threshold", type=float, default=0.85)
    eval_answers.add_argument("--output", type=Path)
    _add_model_arguments(eval_answers)

    return parser


def _build_encoder(args: argparse.Namespace) -> HFCLIPTextEncoder:
    model_source = args.model
    revision = args.revision
    if (
        args.allow_download
        and str(model_source) == str(DEFAULT_MODEL_DIR)
        and not DEFAULT_MODEL_DIR.exists()
    ):
        model_source = DEFAULT_MODEL_ID
        revision = revision or DEFAULT_MODEL_REVISION
    return HFCLIPTextEncoder(
        model_id=model_source,
        cache_dir=args.cache_dir,
        revision=revision,
        device=args.device,
        local_files_only=not args.allow_download,
        batch_size=args.batch_size,
    )


def _load_qa(args: argparse.Namespace) -> KISVideoQA:
    config = _config_from_args(args)
    encoder = _build_encoder(args)
    root = Path(args.artifacts)
    clip_store = Phase1ClipStore(
        root / "clip" / "faiss.index",
        root / "catalog" / "frame_mapping.parquet",
        expected_dimension=encoder.dimension,
        metadata_path=root / "clip" / "clip_index_meta.json",
    )
    hybrid_store = Phase1HybridStore(root, clip_store)
    hybrid = HybridTextualKIS(hybrid_store, encoder, config=config)

    if args.use_expansion:
        understanding = QwenQueryUnderstanding(model=args.expansion_llm_model)
        searcher = ExpandedHybridSearch(hybrid, understanding)
    else:
        searcher = hybrid

    candidate_search = QACandidateSearch(
        searcher,
        config=CandidateConfig(
            search_top_k=args.search_top_k,
            video_limit=args.video_limit,
            frame_limit=args.frame_limit,
        ),
    )
    vlm = QwenVLAnswerer(model=args.vlm_model, base_url=args.vlm_base_url)
    splitter = None if args.no_split else QwenQuestionSplitter(model=args.vlm_model, base_url=args.vlm_base_url)
    return KISVideoQA(candidate_search, vlm, data_root=args.data_root, question_splitter=splitter)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "ask":
        started = time.perf_counter()
        qa = _load_qa(args)
        results = qa.ask(args.question, query_id=args.query_id, top_k=args.top_k)
        output_path = None
        if args.output:
            output_path = write_qa_results(args.output, results)
        _print_json(
            {
                "query_id": args.query_id,
                "question": args.question,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "output": str(output_path.resolve()) if output_path else None,
                "results": [result.as_dict() for result in results],
            }
        )
        return
    if args.command == "eval-answers":
        with args.input_csv.open("r", encoding="utf-8", newline="") as stream:
            rows_raw = list(csv.DictReader(stream))
        for missing in ("predicted", "ground_truth"):
            if rows_raw and missing not in rows_raw[0]:
                raise SystemExit(f"{args.input_csv} is missing required column {missing!r}")
        questions = [row.get("question", "") for row in rows_raw]
        pairs = [(row["predicted"], row["ground_truth"]) for row in rows_raw]

        encoder = _build_encoder(args)
        embedder = ClipTextEmbedder(encoder)
        matches = score_answers(pairs, embedder=embedder, threshold=args.threshold)

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(["question", "predicted", "ground_truth", "similarity", "matched"])
                for question, match in zip(questions, matches):
                    writer.writerow(
                        [question, match.predicted, match.ground_truth, match.similarity, match.matched]
                    )

        accuracy = sum(1 for match in matches if match.matched) / len(matches) if matches else 0.0
        _print_json(
            {
                "input": str(args.input_csv.resolve()),
                "threshold": args.threshold,
                "total": len(matches),
                "matched": sum(1 for match in matches if match.matched),
                "accuracy": accuracy,
                "output": str(args.output.resolve()) if args.output else None,
                "results": [match.as_dict() for match in matches],
            }
        )
        return
    raise AssertionError(f"Unhandled command: {args.command}")


__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    main()
