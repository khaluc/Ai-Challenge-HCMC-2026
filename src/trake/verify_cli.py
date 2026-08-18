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
from kis.hybrid_cli import _add_model_arguments, _positive_int
from trake.temporal_dp import CoarseTemporalAligner
from trake.dense_frame_search import decode_dense_window
from trake.frame_refinement import FineAlignmentConfig
from trake.video_catalog import VideoCatalog
from vlm.frame_verifier import QwenFrameEventScorer

from .frame_verification_schemas import OriginalEventFrame, VerificationCandidate
from .frame_verification import TRAKEVerifier


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


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be numeric") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trake-verify",
        description="TRAKE VLM verification among near-boundary candidate frames, "
        "plus a final f1 < f2 < ... < fn order check.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    verify = commands.add_parser(
        "verify", help="Coarse-align (Stage 9) then VLM-verify near each assigned frame"
    )
    verify.add_argument("video_id")
    verify.add_argument("events", nargs="+")
    verify.add_argument("--artifacts", type=Path, default=DEFAULT_PHASE1_ARTIFACTS)
    verify.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    verify.add_argument("--window-seconds", type=_positive_float, default=0.75)
    verify.add_argument("--step-seconds", type=_positive_float, default=0.25)
    verify.add_argument("--max-candidate-frames", type=_positive_int, default=6)
    verify.add_argument("--vlm-model", default="qwen3.8-max")
    verify.add_argument("--vlm-base-url", default=None)
    _add_model_arguments(verify)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command != "verify":
        raise AssertionError(f"Unhandled command: {args.command}")

    root = Path(args.artifacts)
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
    clip_store = Phase1ClipStore(
        root / "clip" / "faiss.index",
        root / "catalog" / "frame_mapping.parquet",
        expected_dimension=encoder.dimension,
        metadata_path=root / "clip" / "clip_index_meta.json",
    )
    hybrid_store = Phase1HybridStore(root, clip_store)
    coarse = CoarseTemporalAligner(hybrid_store, encoder).align(args.video_id, args.events)
    if not coarse.feasible:
        _print_json({"video_id": args.video_id, "feasible": False})
        return

    catalog = VideoCatalog(root / "catalog" / "frame_mapping.parquet")
    video_info = catalog.get(args.video_id)
    fine_config = FineAlignmentConfig(
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
        max_candidate_frames=args.max_candidate_frames,
    )

    originals = []
    candidates_by_event = {}
    for item in coarse.assignments:
        originals.append(
            OriginalEventFrame(
                event_index=item.event_index,
                event_text=item.event_text,
                frame_id=item.frame.video_frame_id,
                timestamp=item.frame.timestamp,
            )
        )
        dense = decode_dense_window(
            video_info,
            center_seconds=item.frame.timestamp,
            window_seconds=fine_config.window_seconds,
            step_seconds=fine_config.step_seconds,
            data_root=str(args.data_root),
        )
        candidates_by_event[item.event_index] = [
            VerificationCandidate(
                frame_id=int(round(frame.timestamp * video_info.fps)),
                timestamp=frame.timestamp,
                image_bytes=frame.image_bytes,
            )
            for frame in dense
        ]

    scorer = QwenFrameEventScorer(model=args.vlm_model, base_url=args.vlm_base_url)
    verifier = TRAKEVerifier(scorer)
    result = verifier.verify_sequence(args.video_id, originals, candidates_by_event)
    _print_json(result.as_dict())


__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    main()
