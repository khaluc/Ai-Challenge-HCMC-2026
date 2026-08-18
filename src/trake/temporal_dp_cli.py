from __future__ import annotations

import argparse
import json
import sys
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
from kis.hybrid_cli import _add_model_arguments, _nonnegative_float, _positive_int

from .temporal_dp import AlignmentConfig, CoarseTemporalAligner


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
        prog="trake-coarse",
        description="TRAKE coarse temporal alignment: DP over precomputed keyframe similarity.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    align = commands.add_parser("align", help="Align an ordered event list inside one candidate video")
    align.add_argument("video_id")
    align.add_argument("events", nargs="+", help="ordered event texts, e.g. from llm.event_parser_cli decompose")
    align.add_argument("--artifacts", type=Path, default=DEFAULT_PHASE1_ARTIFACTS)
    align.add_argument("--transition-penalty-weight", type=_nonnegative_float, default=0.0)
    align.add_argument("--min-frame-gap", type=_positive_int, default=1)
    align.add_argument("--max-frames-per-video", type=_positive_int, default=5000)
    _add_model_arguments(align)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "align":
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
        config = AlignmentConfig(
            transition_penalty_weight=args.transition_penalty_weight,
            min_frame_gap=args.min_frame_gap,
            max_frames_per_video=args.max_frames_per_video,
        )
        aligner = CoarseTemporalAligner(hybrid_store, encoder, config=config)
        result = aligner.align(args.video_id, args.events)
        _print_json(result.as_dict())
        return
    raise AssertionError(f"Unhandled command: {args.command}")


__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    main()
