from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .inspect_dataset import audit_dataset, write_audit_report
from .build import build_all
from .layout import DatasetLayout
from .query import search_metadata, search_objects, search_similar_keyframe
from .validate import validate_artifacts


def _print_json(value: object) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, OSError):
            pass
    print(json.dumps(value, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="data-processing",
        description="Audit BTC/AIC data and build Phase 1 retrieval artifacts directly from ZIP files.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit", help="Audit source ZIPs and cross-modal IDs")
    audit_parser.add_argument("--data-root", type=Path, default=Path("data"))
    audit_parser.add_argument("--output", type=Path, default=Path("experiments/results/audit"))
    audit_parser.add_argument(
        "--deep-crc", action="store_true", help="Decompress every ZIP member and verify CRC"
    )

    build_parser = subparsers.add_parser("build", help="Build mapping, CLIP, metadata and objects")
    build_parser.add_argument("--data-root", type=Path, default=Path("data"))
    build_parser.add_argument("--output", type=Path, default=Path("indexes"))
    build_parser.add_argument(
        "--components",
        nargs="+",
        choices=("mapping", "clip", "metadata", "objects"),
        default=("mapping", "clip", "metadata", "objects"),
    )
    build_parser.add_argument("--object-index-min-confidence", type=float, default=0.2)

    validate_parser = subparsers.add_parser("validate", help="Validate built artifacts")
    validate_parser.add_argument("--artifacts", type=Path, default=Path("indexes"))

    metadata_parser = subparsers.add_parser("search-metadata", help="Search metadata with BM25")
    metadata_parser.add_argument("query")
    metadata_parser.add_argument("--artifacts", type=Path, default=Path("indexes"))
    metadata_parser.add_argument("--limit", type=int, default=20)

    object_parser = subparsers.add_parser("search-objects", help="Search detected object classes")
    object_parser.add_argument("object_class")
    object_parser.add_argument("--artifacts", type=Path, default=Path("indexes"))
    object_parser.add_argument("--min-confidence", type=float, default=0.2)
    object_parser.add_argument("--limit", type=int, default=50)
    object_parser.add_argument("--contains", action="store_true")

    similar_parser = subparsers.add_parser(
        "search-similar", help="Sanity-check FAISS with a known keyframe vector"
    )
    similar_parser.add_argument("video_id")
    similar_parser.add_argument("keyframe_index", type=int)
    similar_parser.add_argument("--artifacts", type=Path, default=Path("indexes"))
    similar_parser.add_argument("--limit", type=int, default=20)
    similar_parser.add_argument("--include-self", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        layout = DatasetLayout.discover(args.data_root)
        report = audit_dataset(layout, deep_crc=args.deep_crc)
        paths = write_audit_report(report, args.output)
        _print_json(
            {
                "status": report["status"],
                "summary": report["summary"],
                "errors": report["errors"],
                "warnings": report["warnings"],
                "reports": [str(path.resolve()) for path in paths],
            }
        )
        if not report["status"]["core_retrieval_data_complete"]:
            raise SystemExit(2)
    elif args.command == "build":
        layout = DatasetLayout.discover(args.data_root)
        result = build_all(
            layout,
            args.output,
            components=args.components,
            object_index_min_confidence=args.object_index_min_confidence,
        )
        _print_json(result)
    elif args.command == "validate":
        report = validate_artifacts(args.artifacts)
        _print_json(report)
        if not report["valid"]:
            raise SystemExit(2)
    elif args.command == "search-metadata":
        _print_json(
            search_metadata(
                args.artifacts / "metadata" / "metadata.sqlite", args.query, args.limit
            )
        )
    elif args.command == "search-objects":
        _print_json(
            search_objects(
                args.artifacts / "objects" / "objects.sqlite",
                args.object_class,
                min_confidence=args.min_confidence,
                limit=args.limit,
                contains=args.contains,
            )
        )
    elif args.command == "search-similar":
        _print_json(
            search_similar_keyframe(
                args.artifacts,
                args.video_id,
                args.keyframe_index,
                limit=args.limit,
                exclude_self=not args.include_self,
            )
        )
    else:
        raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
