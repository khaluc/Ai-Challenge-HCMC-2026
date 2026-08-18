from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

from retrieval.clip_store import Phase1ClipStore
from retrieval.compatibility import verify_image_feature_compatibility, write_compatibility_report
from retrieval.clip_encoder import (
    DEFAULT_CACHE_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    HFCLIPTextEncoder,
)
from retrieval.evaluator import evaluate_files
from retrieval.io import (
    load_queries,
    plan_btc_submission_paths,
    write_btc_submission,
    write_predictions_csv,
)
from retrieval.model_setup import prepare_model
from retrieval.clip_search import TextualKIS
from retrieval.schemas import Query, RetrievalHit


DEFAULT_PHASE1_ARTIFACTS = Path("indexes")
DEFAULT_PHASE2_ARTIFACTS = Path("experiments/results/kis_baseline")


def _official_top_k(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("top-k must be an integer") from exc
    if not 1 <= parsed <= 100:
        raise argparse.ArgumentTypeError("top-k must be between 1 and 100 for BTC KIS")
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


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def _require_distinct_paths(named_paths: dict[str, Path | None]) -> None:
    seen: dict[Path, str] = {}
    for name, value in named_paths.items():
        if value is None:
            continue
        resolved = value.resolve()
        previous = seen.get(resolved)
        if previous is not None:
            raise ValueError(f"{name} and {previous} must not use the same path: {resolved}")
        seen[resolved] = name


def _temporary_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".tmp")


def _retrieval_output_paths(
    args: argparse.Namespace,
    queries: Sequence[Query],
) -> tuple[Path | None, dict[str, Path | None]]:
    submission_dir = None
    if args.submission_dir or args.submission_zip:
        submission_dir = args.submission_dir or (args.output.parent / "submission")

    paths: dict[str, Path | None] = {
        "queries": args.queries,
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
            paths[f"submission CSV for {query_id!r}"] = path
            paths[f"temporary submission CSV for {query_id!r}"] = _temporary_path(path)
    return submission_dir, paths


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Transformers to access Hugging Face if the checkpoint is not local",
    )


def _load_engine(args: argparse.Namespace) -> tuple[HFCLIPTextEncoder, Phase1ClipStore, TextualKIS]:
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
    store = Phase1ClipStore(
        index_path=root / "clip" / "faiss.index",
        mapping_path=root / "catalog" / "frame_mapping.parquet",
        metadata_path=root / "clip" / "clip_index_meta.json",
        expected_dimension=encoder.dimension,
    )
    return encoder, store, TextualKIS(store, encoder)


def _retrieval_manifest(
    *,
    encoder: HFCLIPTextEncoder,
    store: Phase1ClipStore,
    queries: int,
    hits: int,
    top_k: int,
    seconds: float,
) -> dict:
    return {
        "pipeline": "text -> CLIP text projection -> L2 normalize -> FAISS IndexFlatIP -> FAISS_Index mapping",
        "encoder": encoder.describe(),
        "index": {
            "path": str(store.index_path.resolve()),
            "mapping": str(store.mapping_path.resolve()),
            "vectors": store.size,
            "dimension": store.dimension,
        },
        "queries": queries,
        "hits": hits,
        "top_k": top_k,
        "elapsed_seconds": round(seconds, 3),
        "deduplication": "(video_id, Video_Frame_ID), preserving the highest-ranked hit",
        "submission_frame": "Video_Frame_ID from the Phase 1 canonical mapping",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kis-baseline",
        description="Textual KIS CLIP baseline, BTC submission writer and Recall@K evaluator.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-model", help="Download the pinned OpenAI CLIP checkpoint")
    prepare.add_argument("--repo-id", default=DEFAULT_MODEL_ID)
    prepare.add_argument("--output", type=Path, default=DEFAULT_MODEL_DIR)
    prepare.add_argument("--revision", default=DEFAULT_MODEL_REVISION)

    verify = commands.add_parser(
        "verify-encoder", help="Compare candidate image embeddings with BTC-provided vectors"
    )
    verify.add_argument("--data-root", type=Path, default=Path("data"))
    verify.add_argument("--model", default=str(DEFAULT_MODEL_DIR))
    verify.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    verify.add_argument("--revision")
    verify.add_argument("--device", default="cpu")
    verify.add_argument("--allow-download", action="store_true")
    verify.add_argument("--samples", type=int, default=5)
    verify.add_argument("--min-cosine", type=float, default=0.99)
    verify.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PHASE2_ARTIFACTS / "encoder_compatibility.json",
    )

    search = commands.add_parser("search", help="Run one end-to-end textual KIS query")
    search.add_argument("query")
    search.add_argument("--query-id", default="query")
    search.add_argument("--top-k", type=_official_top_k, default=100)
    search.add_argument("--artifacts", type=Path, default=DEFAULT_PHASE1_ARTIFACTS)
    search.add_argument("--predictions-output", type=Path)
    search.add_argument("--submission-dir", type=Path)
    _add_model_arguments(search)

    retrieve = commands.add_parser("retrieve", help="Retrieve a CSV/JSONL query batch")
    retrieve.add_argument("--queries", type=Path, required=True)
    retrieve.add_argument("--top-k", type=_official_top_k, default=100)
    retrieve.add_argument("--artifacts", type=Path, default=DEFAULT_PHASE1_ARTIFACTS)
    retrieve.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PHASE2_ARTIFACTS / "predictions.csv",
    )
    retrieve.add_argument("--submission-dir", type=Path)
    retrieve.add_argument("--submission-zip", type=Path)
    retrieve.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_PHASE2_ARTIFACTS / "retrieval_manifest.json",
    )
    _add_model_arguments(retrieve)

    evaluate = commands.add_parser("evaluate", help="Compute R@K and Final Score")
    evaluate.add_argument("--ground-truth", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PHASE2_ARTIFACTS / "evaluation.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-model":
        _print_json(
            prepare_model(repo_id=args.repo_id, output_dir=args.output, revision=args.revision)
        )
        return
    if args.command == "verify-encoder":
        report = verify_image_feature_compatibility(
            args.data_root,
            model_id=args.model,
            cache_dir=args.cache_dir,
            revision=args.revision,
            local_files_only=not args.allow_download,
            samples=args.samples,
            min_cosine=args.min_cosine,
            device=args.device,
        )
        report_path = write_compatibility_report(report, args.output)
        report["report"] = str(report_path.resolve())
        _print_json(report)
        if not report["compatible"]:
            raise SystemExit(2)
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
        hits = engine.search(query, top_k=args.top_k)
        if args.predictions_output:
            write_predictions_csv(args.predictions_output, hits)
        submission_files = []
        if args.submission_dir:
            submission_files = write_btc_submission(args.submission_dir, hits)
        _print_json(
            {
                "manifest": _retrieval_manifest(
                    encoder=encoder,
                    store=store,
                    queries=1,
                    hits=len(hits),
                    top_k=args.top_k,
                    seconds=time.perf_counter() - started,
                ),
                "predictions_output": (
                    str(args.predictions_output.resolve()) if args.predictions_output else None
                ),
                "submission_files": [str(path.resolve()) for path in submission_files],
                "results": [hit.as_dict() for hit in hits],
            }
        )
        return
    if args.command == "retrieve":
        started = time.perf_counter()
        queries = load_queries(args.queries)
        if not queries:
            raise ValueError(f"Query file contains no queries: {args.queries}")
        submission_dir, output_paths = _retrieval_output_paths(args, queries)
        _require_distinct_paths(output_paths)
        encoder, store, engine = _load_engine(args)
        hits = engine.search_batch(queries, top_k=args.top_k)
        predictions_path = write_predictions_csv(args.output, hits)
        submission_files = []
        if args.submission_dir or args.submission_zip:
            assert submission_dir is not None
            submission_files = write_btc_submission(
                submission_dir,
                hits,
                zip_path=args.submission_zip,
            )
        manifest = _retrieval_manifest(
            encoder=encoder,
            store=store,
            queries=len(queries),
            hits=len(hits),
            top_k=args.top_k,
            seconds=time.perf_counter() - started,
        )
        manifest.update(
            {
                "query_file": str(args.queries.resolve()),
                "predictions_file": str(predictions_path.resolve()),
                "submission_files": [str(path.resolve()) for path in submission_files],
                "submission_zip": (
                    str(args.submission_zip.resolve()) if args.submission_zip else None
                ),
            }
        )
        manifest_path = _write_json(args.manifest, manifest)
        manifest["manifest"] = str(manifest_path.resolve())
        _print_json(manifest)
        return
    if args.command == "evaluate":
        _require_distinct_paths(
            {
                "ground_truth": args.ground_truth,
                "predictions": args.predictions,
                "output": args.output,
                "temporary output": _temporary_path(args.output),
            }
        )
        result = evaluate_files(args.ground_truth, args.predictions)
        report = result.as_dict()
        report["ground_truth"] = str(args.ground_truth.resolve())
        report["predictions"] = str(args.predictions.resolve())
        report_path = _write_json(args.output, report)
        report["report"] = str(report_path.resolve())
        _print_json(report)
        return
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
