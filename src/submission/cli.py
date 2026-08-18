from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .io import rerank_predictions_csv


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be numeric") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="submission-rerank",
        description="Rerank an existing Top-100 predictions CSV with tiered "
        "confidence-vs-diversity ranking (rank 1-5 pure relevance, diversity "
        "ramping up toward rank 51-100).",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    rerank = commands.add_parser("rerank", help="Rerank a phase3/4-style predictions CSV")
    rerank.add_argument("--predictions", type=Path, required=True)
    rerank.add_argument("--output", type=Path, required=True)
    rerank.add_argument("--near-duplicate-seconds", type=_nonnegative_float, default=5.0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "rerank":
        output = rerank_predictions_csv(
            args.predictions, args.output, near_duplicate_seconds=args.near_duplicate_seconds
        )
        print(f"Reranked predictions written to {output.resolve()}")
        return
    raise AssertionError(f"Unhandled command: {args.command}")


__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    main()
