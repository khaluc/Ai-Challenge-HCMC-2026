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

from .frame_refinement import FineAlignmentConfig, FineFrameAligner
from .video_catalog import VideoCatalog
from vlm.frame_verifier import QwenFrameEventScorer


DEFAULT_PHASE1_ARTIFACTS = Path("indexes")
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


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be numeric") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be numeric") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _add_clip_model_arguments(parser: argparse.ArgumentParser) -> None:
    # Local copy of kis.hybrid_cli._add_model_arguments: importing that
    # module (even just for its argparse helpers) pulls in retrieval.clip_encoder
    # and therefore torch at module load time, which the `refine` command
    # never needs. `build_parser()` always builds both subparsers, so even
    # the default value here must avoid importing retrieval.clip_encoder -
    # duplicate the constant instead (checked against retrieval.clip_encoder.
    # DEFAULT_CACHE_DIR) so `refine` stays torch-free end to end.
    default_cache_dir = Path("models/huggingface")

    parser.add_argument("--model", default=None)
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir)
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument("--allow-download", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trake-fine",
        description="TRAKE fine frame alignment: decode a small window around a coarse "
        "timestamp and let a VLM pick the semantic keyframe.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    refine = commands.add_parser("refine", help="Fine-align one event around a coarse timestamp")
    refine.add_argument("video_id")
    refine.add_argument("event_text")
    refine.add_argument("coarse_timestamp", type=float)
    refine.add_argument("--artifacts", type=Path, default=DEFAULT_PHASE1_ARTIFACTS)
    refine.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    refine.add_argument("--window-seconds", type=_positive_float, default=2.5)
    refine.add_argument("--step-seconds", type=_positive_float, default=0.5)
    refine.add_argument("--max-candidate-frames", type=_positive_int, default=12)
    refine.add_argument("--vlm-model", default="qwen3.8-max")
    refine.add_argument("--vlm-base-url", default=None)

    align = commands.add_parser(
        "align", help="Coarse (Stage 9) then fine (Stage 10) alignment for an ordered event list"
    )
    align.add_argument("video_id")
    align.add_argument("events", nargs="+")
    align.add_argument("--artifacts", type=Path, default=DEFAULT_PHASE1_ARTIFACTS)
    align.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    align.add_argument("--transition-penalty-weight", type=_nonnegative_float, default=0.0)
    align.add_argument("--min-frame-gap", type=_positive_int, default=1)
    align.add_argument("--window-seconds", type=_positive_float, default=2.5)
    align.add_argument("--step-seconds", type=_positive_float, default=0.5)
    align.add_argument("--max-candidate-frames", type=_positive_int, default=12)
    align.add_argument("--vlm-model", default="qwen3.8-max")
    align.add_argument("--vlm-base-url", default=None)
    _add_clip_model_arguments(align)

    return parser


def _load_clip_encoder(args: argparse.Namespace):
    from retrieval.clip_encoder import (
        DEFAULT_MODEL_DIR,
        DEFAULT_MODEL_ID,
        DEFAULT_MODEL_REVISION,
        HFCLIPTextEncoder,
    )

    model_source = Path(args.model) if args.model else DEFAULT_MODEL_DIR
    revision = args.revision
    if args.allow_download and model_source == DEFAULT_MODEL_DIR and not DEFAULT_MODEL_DIR.exists():
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


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = Path(args.artifacts)

    if args.command == "refine":
        catalog = VideoCatalog(root / "catalog" / "frame_mapping.parquet")
        scorer = QwenFrameEventScorer(model=args.vlm_model, base_url=args.vlm_base_url)
        aligner = FineFrameAligner(
            catalog,
            scorer,
            data_root=str(args.data_root),
            config=FineAlignmentConfig(
                window_seconds=args.window_seconds,
                step_seconds=args.step_seconds,
                max_candidate_frames=args.max_candidate_frames,
            ),
        )
        result = aligner.refine(args.video_id, args.event_text, args.coarse_timestamp)
        _print_json(vars(result))
        return

    if args.command == "align":
        from retrieval.clip_store import Phase1ClipStore
        from retrieval.hybrid_store import Phase1HybridStore
        from trake.temporal_dp import AlignmentConfig, CoarseTemporalAligner

        encoder = _load_clip_encoder(args)
        clip_store = Phase1ClipStore(
            root / "clip" / "faiss.index",
            root / "catalog" / "frame_mapping.parquet",
            expected_dimension=encoder.dimension,
            metadata_path=root / "clip" / "clip_index_meta.json",
        )
        hybrid_store = Phase1HybridStore(root, clip_store)
        coarse_aligner = CoarseTemporalAligner(
            hybrid_store,
            encoder,
            config=AlignmentConfig(
                transition_penalty_weight=args.transition_penalty_weight,
                min_frame_gap=args.min_frame_gap,
            ),
        )
        coarse = coarse_aligner.align(args.video_id, args.events)
        if not coarse.feasible:
            _print_json({"video_id": args.video_id, "feasible": False, "events": args.events})
            return

        catalog = VideoCatalog(root / "catalog" / "frame_mapping.parquet")
        scorer = QwenFrameEventScorer(model=args.vlm_model, base_url=args.vlm_base_url)
        fine_aligner = FineFrameAligner(
            catalog,
            scorer,
            data_root=str(args.data_root),
            config=FineAlignmentConfig(
                window_seconds=args.window_seconds,
                step_seconds=args.step_seconds,
                max_candidate_frames=args.max_candidate_frames,
            ),
        )
        refined = [
            fine_aligner.refine(args.video_id, item.event_text, item.frame.timestamp)
            for item in coarse.assignments
        ]
        _print_json(
            {
                "video_id": args.video_id,
                "coarse": coarse.as_dict(),
                "fine": [vars(item) for item in refined],
            }
        )
        return

    raise AssertionError(f"Unhandled command: {args.command}")


__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    main()
