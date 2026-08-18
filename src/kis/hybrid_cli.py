from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Sequence

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
from .hybrid_io import write_hybrid_predictions, write_hybrid_submission
from retrieval.processing import RuleBasedObjectParser, tokenize_for_metadata
from retrieval.hybrid_search import HybridConfig, HybridSearchResult, HybridTextualKIS


DEFAULT_PHASE1_ARTIFACTS = Path("indexes")
DEFAULT_PHASE3_ARTIFACTS = Path("experiments/results/kis_hybrid")


def _official_top_k(value: str) -> int:
    parsed = _positive_int(value)
    if parsed > 100:
        raise argparse.ArgumentTypeError("top-k must be between 1 and 100 for BTC KIS")
    return parsed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _unit_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be numeric") from exc
    if not math.isfinite(parsed) or not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be numeric") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return parsed


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, OSError):
            pass


def _print_json(value: object) -> None:
    _configure_stdout()
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _temporary_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".tmp")


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _require_distinct_paths(named_paths: dict[str, Path | None]) -> None:
    seen: dict[Path, str] = {}
    for name, path in named_paths.items():
        if path is None:
            continue
        resolved = path.resolve()
        previous = seen.get(resolved)
        if previous is not None:
            raise ValueError(f"{name} and {previous} must not share path: {resolved}")
        seen[resolved] = name


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument("--allow-download", action="store_true")


def _add_hybrid_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fusion", choices=("rrf", "weighted"), default="rrf")
    parser.add_argument("--rrf-k", type=_positive_int, default=60)
    parser.add_argument("--semantic-candidates", type=_positive_int, default=500)
    parser.add_argument("--metadata-videos", type=_positive_int, default=30)
    parser.add_argument("--metadata-frames", type=_positive_int, default=3)
    parser.add_argument("--object-candidates", type=_positive_int, default=300)
    parser.add_argument("--object-min-confidence", type=_unit_float, default=0.30)
    parser.add_argument("--semantic-weight", type=_nonnegative_float, default=1.0)
    parser.add_argument("--metadata-weight", type=_nonnegative_float, default=0.65)
    parser.add_argument("--object-weight", type=_nonnegative_float, default=0.80)
    parser.add_argument("--no-metadata", action="store_true")
    parser.add_argument("--no-objects", action="store_true")


def _config_from_args(args: argparse.Namespace) -> HybridConfig:
    return HybridConfig(
        semantic_candidates=args.semantic_candidates,
        metadata_video_candidates=args.metadata_videos,
        metadata_frames_per_video=args.metadata_frames,
        object_candidates=args.object_candidates,
        object_min_confidence=args.object_min_confidence,
        fusion_method=args.fusion,
        rrf_k=args.rrf_k,
        semantic_weight=args.semantic_weight,
        metadata_weight=args.metadata_weight,
        object_weight=args.object_weight,
        metadata_enabled=not args.no_metadata,
        objects_enabled=not args.no_objects,
    )


def _load_engine(
    args: argparse.Namespace,
) -> tuple[HFCLIPTextEncoder, Phase1HybridStore, HybridTextualKIS]:
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
    engine = HybridTextualKIS(
        hybrid_store,
        encoder,
        config=config,
    )
    return encoder, hybrid_store, engine


def _manifest(
    *,
    encoder: HFCLIPTextEncoder,
    store: Phase1HybridStore,
    engine: HybridTextualKIS,
    results: Sequence[HybridSearchResult],
    top_k: int,
    elapsed: float,
) -> dict[str, object]:
    active_branches = ["semantic"]
    if engine.config.metadata_enabled and engine.config.metadata_weight > 0:
        active_branches.append("metadata")
    if engine.config.objects_enabled and engine.config.object_weight > 0:
        active_branches.append("objects")
    return {
        "pipeline": " + ".join(active_branches) + " -> late fusion -> Top-100",
        "active_branches": active_branches,
        "fusion": engine.config.as_dict(),
        "encoder": encoder.describe(),
        "artifacts": {
            "root": str(store.root.resolve()),
            "vectors": store.clip_store.size,
            "dimension": store.clip_store.dimension,
            "metadata_documents": store.metadata_documents,
            "asr_or_transcript_available": False,
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


def _retrieval_paths(
    args: argparse.Namespace,
    queries: Sequence[Query],
) -> tuple[Path | None, dict[str, Path | None]]:
    submission_dir = None
    if args.submission_dir or args.submission_zip:
        submission_dir = args.submission_dir or (args.output.parent / "submission")
    paths: dict[str, Path | None] = {
        "query input": args.queries,
        "predictions": args.output,
        "temporary predictions": _temporary_path(args.output),
        "manifest": args.manifest,
        "temporary manifest": _temporary_path(args.manifest),
        "submission directory": submission_dir,
        "submission ZIP": args.submission_zip,
        "temporary submission ZIP": (
            _temporary_path(args.submission_zip) if args.submission_zip else None
        ),
    }
    if submission_dir is not None:
        planned = plan_btc_submission_paths(
            submission_dir, (query.query_id for query in queries)
        )
        for query_id, path in planned.items():
            paths[f"submission CSV {query_id!r}"] = path
            paths[f"temporary submission CSV {query_id!r}"] = _temporary_path(path)
    return submission_dir, paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kis-hybrid",
        description="Hybrid CLIP, metadata BM25 and object retrieval for textual KIS.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    analyze = commands.add_parser(
        "analyze-query", help="Show deterministic metadata terms and object concepts"
    )
    analyze.add_argument("query")
    analyze.add_argument("--artifacts", type=Path, default=DEFAULT_PHASE1_ARTIFACTS)

    search = commands.add_parser("search", help="Run one hybrid textual KIS query")
    search.add_argument("query")
    search.add_argument("--query-id", default="query")
    search.add_argument("--top-k", type=_official_top_k, default=100)
    search.add_argument("--artifacts", type=Path, default=DEFAULT_PHASE1_ARTIFACTS)
    search.add_argument("--predictions-output", type=Path)
    search.add_argument("--submission-dir", type=Path)
    _add_model_arguments(search)
    _add_hybrid_arguments(search)

    retrieve = commands.add_parser("retrieve", help="Run hybrid retrieval for CSV/JSONL queries")
    retrieve.add_argument("--queries", type=Path, required=True)
    retrieve.add_argument("--top-k", type=_official_top_k, default=100)
    retrieve.add_argument("--artifacts", type=Path, default=DEFAULT_PHASE1_ARTIFACTS)
    retrieve.add_argument(
        "--output", type=Path, default=DEFAULT_PHASE3_ARTIFACTS / "predictions.csv"
    )
    retrieve.add_argument("--submission-dir", type=Path)
    retrieve.add_argument("--submission-zip", type=Path)
    retrieve.add_argument(
        "--manifest", type=Path, default=DEFAULT_PHASE3_ARTIFACTS / "retrieval_manifest.json"
    )
    _add_model_arguments(retrieve)
    _add_hybrid_arguments(retrieve)

    evaluate = commands.add_parser("evaluate", help="Reuse the strict R@K evaluator")
    evaluate.add_argument("--ground-truth", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument(
        "--output", type=Path, default=DEFAULT_PHASE3_ARTIFACTS / "evaluation.json"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "analyze-query":
        database = args.artifacts / "objects" / "objects.sqlite"
        parser = RuleBasedObjectParser(load_object_labels(database))
        parsed = parser.parse(args.query)
        _print_json(
            {
                "text": args.query,
                "metadata_terms": list(tokenize_for_metadata(args.query)),
                "object_concepts": list(parsed.concepts),
                "detector_labels": {
                    concept: list(parsed.labels_by_concept[concept])
                    for concept in parsed.concepts
                },
                "note": "Color and spatial relations are intentionally not hard filters.",
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
            write_hybrid_predictions(args.predictions_output, result.hits)
        submission_files = []
        if args.submission_dir:
            submission_files = write_hybrid_submission(args.submission_dir, result.hits)
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
        predictions = write_hybrid_predictions(args.output, hits)
        submission_files = []
        if submission_dir is not None:
            submission_files = write_hybrid_submission(
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
