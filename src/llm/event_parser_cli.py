from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from .event_parser import QwenEventDecomposer


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
        prog="llm-events",
        description="LLM event decomposition for TRAKE multi-event queries.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    decompose = commands.add_parser("decompose", help="Decompose one TRAKE query into ordered events")
    decompose.add_argument("query")
    decompose.add_argument("--model", default="qwen3.8-max")
    decompose.add_argument("--base-url", default=None)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "decompose":
        decomposer = QwenEventDecomposer(model=args.model, base_url=args.base_url)
        result = decomposer.decompose(args.query)
        _print_json(result.as_dict())
        return
    raise AssertionError(f"Unhandled command: {args.command}")


__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    main()
