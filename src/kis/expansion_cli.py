from __future__ import annotations

import argparse
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
from retrieval.evaluator import evaluate_files
from retrieval.io import load_queries, plan_btc_submission_paths
from retrieval.schemas import Query
from retrieval.hybrid_store import Phase1HybridStore, load_object_labels
from kis.hybrid_cli import (
    _add_hybrid_arguments,
    _add_model_arguments,
    _config_from_args,
    _official_top_k,
    _positive_int,
    _require_distinct_paths,
    _retrieval_paths,
    _temporary_path,
    _write_json,
)
from retrieval.processing import RuleBasedObjectParser
from retrieval.hybrid_search import HybridTextualKIS

from .expansion_io import write_expanded_predictions, write_expanded_submission
from llm.expansion_retrieval import ExpandedHybridSearch, ExpandedSearchResult, ExpansionConfig
from llm.query_expansion import (
    AnthropicQueryUnderstanding,
    QueryUnderstandingProtocol,
    QwenQueryUnderstanding,
    RuleBasedQueryUnderstanding,
)


DEFAULT_PHASE1_ARTIFACTS = Path("indexes")
DEFAULT_PHASE4_ARTIFACTS = Path("experiments/results/kis_expansion")


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, OSError):
            pass


def _print_json(value: object) -> None:
    _configure_stdout()
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _add_expansion_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-expansions", type=_positive_int, default=4)
    parser.add_argument(
        "--candidates-per-expansion",
        type=_official_top_k,
        default=100,
        help="top-k requested from the Phase 3 hybrid pipeline per expansion (<=100)",
    )
    parser.add_argument("--expansion-rrf-k", type=_positive_int, default=60)
    parser.add_argument(
        "--llm-backend",
        choices=("rule", "anthropic", "qwen"),
        default="rule",
        help="query understanding backend (default: rule-based, no API key needed)",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="override the model id for --llm-backend anthropic/qwen",
    )
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="override the OpenAI-compatible base URL for --llm-backend qwen",
    )


def _expansion_config_from_args(args: argparse.Namespace) -> ExpansionConfig:
    return ExpansionConfig(
        max_expansions=args.max_expansions,
        candidates_per_expansion=args.candidates_per_expansion,
        rrf_k=args.expansion_rrf_k,
    )


def _build_understanding(
    args: argparse.Namespace, object_parser: RuleBasedObjectParser
) -> QueryUnderstandingProtocol:
    if args.llm_backend == "anthropic":
        return AnthropicQueryUnderstanding(
            model=args.llm_model or "claude-opus-5", max_expansions=args.max_expansions
        )
    if args.llm_backend == "qwen":
        return QwenQueryUnderstanding(
            model=args.llm_model or "qwen3.8-max",
            base_url=args.llm_base_url,
            max_expansions=args.max_expansions,
        )
    return RuleBasedQueryUnderstanding(object_parser, max_expansions=args.max_expansions)


def _load_engine(
    args: argparse.Namespace,
) -> tuple[HFCLIPTextEncoder, Phase1HybridStore, ExpandedHybridSearch]:
    config = _config_from_args(args)
    model_source = args.model
    revision = args.revision
    if (
        args.allow_download
        and str(model_source) == str(DEFAULT_MODEL_DIR)
        and not DEFAULT_MODEL_DIR.exists()
    ):
        model_source = DEFAULT_MODEL_ID
        revision = revision or DEFAULT_MODEL_REVISION
    encoder = HFCLIPTextEncoder(
        model_id=model_source,
        cache_dir=args.cache_dir,
        revision=revision,
        device=args.device,
        local_files_only=not args.allow_download,
        batch_size=args.batch_size,
    )
    root = Path(args.artifacts)
    clip_store = Phase1ClipStore(
        root / "clip" / "faiss.index",
        root / "catalog" / "frame_mapping.parquet",
        expected_dimension=encoder.dimension,
        metadata_path=root / "clip" / "clip_index_meta.json",
    )
    hybrid_store = Phase1HybridStore(root, clip_store)
    hybrid = HybridTextualKIS(hybrid_store, encoder, config=config)
    object_parser = RuleBasedObjectParser(hybrid_store.object_labels)
    understanding = _build_understanding(args, object_parser)
    engine = ExpandedHybridSearch(
        hybrid, understanding, config=_expansion_config_from_args(args)
    )
    return encoder, hybrid_store, engine


def _manifest(
    *,
    encoder: HFCLIPTextEncoder,
    store: Phase1HybridStore,
    engine: ExpandedHybridSearch,
    results: Sequence[ExpandedSearchResult],
    top_k: int,
    elapsed: float,
) -> dict[str, object]:
    return {
        "pipeline": "LLM/rule query understanding -> expansions -> hybrid search -> RRF -> Top-100",
        "expansion_config": {
            "max_expansions": engine.config.max_expansions,
            "candidates_per_expansion": engine.config.candidates_per_expansion,
            "rrf_k": engine.config.rrf_k,
        },
        "hybrid_config": engine.hybrid.config.as_dict(),
        "encoder": encoder.describe(),
        "artifacts": {
            "root": str(store.root.resolve()),
            "vectors": store.clip_store.size,
            "dimension": store.clip_store.dimension,
            "metadata_documents": store.metadata_documents,
            "object_classes": len(store.object_labels),
            "object_index_min_confidence": store.object_index_min_confidence,
        },
        "queries": len(results),
        "hits": sum(len(result.hits) for result in results),
        "top_k": top_k,
        "elapsed_seconds": round(elapsed, 3),
        "per_query": [result.as_summary() for result in results],
        "submission_frame": "Video_Frame_ID from the canonical Phase 1 mapping",
        "deduplication": "(video_id, Video_Frame_ID)",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kis-expansion",
        description="LLM query understanding, expansion and fusion on top of the Phase 3 hybrid pipeline.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    analyze = commands.add_parser(
        "analyze-query",
        help="Show structured objects/attributes/relation and expansions without loading CLIP",
    )
    analyze.add_argument("query")
    analyze.add_argument("--artifacts", type=Path, default=DEFAULT_PHASE1_ARTIFACTS)
    analyze.add_argument("--max-expansions", type=_positive_int, default=4)
    analyze.add_argument("--llm-backend", choices=("rule", "anthropic", "qwen"), default="rule")
    analyze.add_argument("--llm-model", default=None)
    analyze.add_argument("--llm-base-url", default=None)

    search = commands.add_parser("search", help="Run one expanded hybrid textual KIS query")
    search.add_argument("query")
    search.add_argument("--query-id", default="query")
    search.add_argument("--top-k", type=_official_top_k, default=100)
    search.add_argument("--artifacts", type=Path, default=DEFAULT_PHASE1_ARTIFACTS)
    search.add_argument("--predictions-output", type=Path)
    search.add_argument("--submission-dir", type=Path)
    _add_model_arguments(search)
    _add_hybrid_arguments(search)
    _add_expansion_arguments(search)

    retrieve = commands.add_parser(
        "retrieve", help="Run expanded hybrid retrieval for CSV/JSONL queries"
    )
    retrieve.add_argument("--queries", type=Path, required=True)
    retrieve.add_argument("--top-k", type=_official_top_k, default=100)
    retrieve.add_argument("--artifacts", type=Path, default=DEFAULT_PHASE1_ARTIFACTS)
    retrieve.add_argument(
        "--output", type=Path, default=DEFAULT_PHASE4_ARTIFACTS / "predictions.csv"
    )
    retrieve.add_argument("--submission-dir", type=Path)
    retrieve.add_argument("--submission-zip", type=Path)
    retrieve.add_argument(
        "--manifest", type=Path, default=DEFAULT_PHASE4_ARTIFACTS / "retrieval_manifest.json"
    )
    _add_model_arguments(retrieve)
    _add_hybrid_arguments(retrieve)
    _add_expansion_arguments(retrieve)

    evaluate = commands.add_parser("evaluate", help="Reuse the strict R@K evaluator")
    evaluate.add_argument("--ground-truth", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument(
        "--output", type=Path, default=DEFAULT_PHASE4_ARTIFACTS / "evaluation.json"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "analyze-query":
        database = args.artifacts / "objects" / "objects.sqlite"
        object_parser = RuleBasedObjectParser(load_object_labels(database))
        understanding = _build_understanding(args, object_parser)
        result = understanding.understand(args.query)
        _print_json(
            {
                **result.as_dict(),
                "note": "Color, gender and spatial relations stay soft signals; only 'objects' feeds the Faster R-CNN branch.",
            }
        )
        return
    if args.command == "search":
        started = time.perf_counter()
        query = Query(args.query_id, args.query)
        if args.predictions_output and args.submission_dir:
            planned = plan_btc_submission_paths(args.submission_dir, [query.query_id])
            submission_path = planned[query.query_id]
            _require_distinct_paths(
                {
                    "predictions": args.predictions_output,
                    "temporary predictions": _temporary_path(args.predictions_output),
                    "submission directory": args.submission_dir,
                    "submission CSV": submission_path,
                    "temporary submission CSV": _temporary_path(submission_path),
                }
            )
        encoder, store, engine = _load_engine(args)
        result = engine.search_detailed_batch([query], top_k=args.top_k)[0]
        if args.predictions_output:
            write_expanded_predictions(args.predictions_output, result.hits)
        submission_files = []
        if args.submission_dir:
            submission_files = write_expanded_submission(args.submission_dir, result.hits)
        _print_json(
            {
                "manifest": _manifest(
                    encoder=encoder,
                    store=store,
                    engine=engine,
                    results=[result],
                    top_k=args.top_k,
                    elapsed=time.perf_counter() - started,
                ),
                "predictions_output": (
                    str(args.predictions_output.resolve()) if args.predictions_output else None
                ),
                "submission_files": [str(path.resolve()) for path in submission_files],
                "results": [hit.as_dict() for hit in result.hits],
            }
        )
        return
    if args.command == "retrieve":
        started = time.perf_counter()
        queries = load_queries(args.queries)
        if not queries:
            raise ValueError(f"Query file contains no queries: {args.queries}")
        submission_dir, paths = _retrieval_paths(args, queries)
        _require_distinct_paths(paths)
        encoder, store, engine = _load_engine(args)
        results = engine.search_detailed_batch(queries, top_k=args.top_k)
        hits = [hit for result in results for hit in result.hits]
        predictions = write_expanded_predictions(args.output, hits)
        submission_files = []
        if submission_dir is not None:
            submission_files = write_expanded_submission(
                submission_dir, hits, zip_path=args.submission_zip
            )
        manifest = _manifest(
            encoder=encoder,
            store=store,
            engine=engine,
            results=results,
            top_k=args.top_k,
            elapsed=time.perf_counter() - started,
        )
        manifest.update(
            {
                "query_file": str(args.queries.resolve()),
                "predictions_file": str(predictions.resolve()),
                "submission_files": [str(path.resolve()) for path in submission_files],
                "submission_zip": (
                    str(args.submission_zip.resolve()) if args.submission_zip else None
                ),
            }
        )
        report = _write_json(args.manifest, manifest)
        manifest["manifest"] = str(report.resolve())
        _print_json(manifest)
        return
    if args.command == "evaluate":
        _require_distinct_paths(
            {
                "ground truth": args.ground_truth,
                "predictions": args.predictions,
                "output": args.output,
                "temporary output": _temporary_path(args.output),
            }
        )
        result = evaluate_files(args.ground_truth, args.predictions)
        report = result.as_dict()
        report["ground_truth"] = str(args.ground_truth.resolve())
        report["predictions"] = str(args.predictions.resolve())
        output = _write_json(args.output, report)
        report["report"] = str(output.resolve())
        _print_json(report)
        return
    raise AssertionError(f"Unhandled command: {args.command}")


__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    main()
