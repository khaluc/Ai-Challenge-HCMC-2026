from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

from .schemas import QAResult


QA_RESULT_FIELDS = (
    "query_id",
    "rank",
    "route",
    "video_id",
    "frame_id",
    "answer",
    "confidence",
    "note",
)


def write_qa_results(path: str | Path, results: Iterable[QAResult]) -> Path:
    """Diagnostic CSV — no official BTC Q&A submission format is available yet."""

    values = list(results)
    for result in values:
        if not isinstance(result, QAResult):
            raise TypeError(f"Expected QAResult, got {type(result).__name__}")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=QA_RESULT_FIELDS, lineterminator="\n")
            writer.writeheader()
            for result in values:
                writer.writerow({field: getattr(result, field) for field in QA_RESULT_FIELDS})
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


__all__ = ["QA_RESULT_FIELDS", "write_qa_results"]
