from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

try:
    from dotenv import load_dotenv

    load_dotenv()
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
from kis.hybrid_cli import _add_model_arguments, _positive_int
from retrieval.hybrid_search import HybridConfig, HybridTextualKIS
from llm.event_parser import QwenEventDecomposer

from .aggregation import VideoRetrievalConfig
from .video_retrieval import TRAKEVideoRetrieval


DEFAULT_PHASE1_ARTIFACTS = Path("indexes")


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
        prog="trake-videos",
        description="TRAKE video-level candidate retrieval: full query + events + "
        "expansions, aggregated into Top 3-5 candidate videos.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    find = commands.add_parser("find-videos", help="Find candidate videos for one TRAKE query")
    find.add_argument("query")
    find.add_argument("--query-id", default="trake")
    find.add_argument("--video-limit", type=_positive_int, default=5)
    find.add_argument("--per-query-top-k", type=_positive_int, default=100)
    find.add_argument("--metadata-video-limit", type=_positive_int, default=50)
    find.add_argument("--artifacts", type=Path, default=DEFAULT_PHASE1_ARTIFACTS)
    find.add_argument("--llm-model", default="qwen3.8-max")
    find.add_argument("--llm-base-url", default=None)
    _add_model_arguments(find)

    return parser


def _load_retrieval(args: argparse.Namespace) -> TRAKEVideoRetrieval:
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
    hybrid = HybridTextualKIS(hybrid_store, encoder, config=HybridConfig())
    decomposer = QwenEventDecomposer(model=args.llm_model, base_url=args.llm_base_url)
    config = VideoRetrievalConfig(
        per_query_top_k=args.per_query_top_k,
        video_limit=args.video_limit,
        metadata_video_limit=args.metadata_video_limit,
    )
    return TRAKEVideoRetrieval(hybrid, hybrid_store, decomposer, config=config)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "find-videos":
        started = time.perf_counter()
        retrieval = _load_retrieval(args)
        result = retrieval.find_candidate_videos(args.query, query_id=args.query_id)
        payload = result.as_dict()
        payload["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        _print_json(payload)
        return
    raise AssertionError(f"Unhandled command: {args.command}")


__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    main()
