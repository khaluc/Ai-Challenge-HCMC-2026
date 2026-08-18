from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from .ranking_optimizer import DEFAULT_TIERS, RankableHit, RankingTier, rerank_with_tiered_diversity

REQUIRED_COLUMNS = ("query_id", "rank", "video_id", "frame_id", "score", "timestamp")


def rerank_predictions_csv(
    input_path: str | Path,
    output_path: str | Path,
    *,
    tiers: Sequence[RankingTier] = DEFAULT_TIERS,
    near_duplicate_seconds: float = 5.0,
) -> Path:
    """Rerank a phase3/4-style diagnostic predictions CSV, query by query,
    preserving every original column but recomputing `rank`."""

    input_path = Path(input_path)
    with input_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"{input_path} has no header row")
        missing = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
        if missing:
            raise ValueError(f"{input_path} is missing required column(s): {missing}")
        rows_by_query: dict[str, list[dict]] = defaultdict(list)
        for row in reader:
            rows_by_query[row["query_id"]].append(row)

    output_rows: list[dict] = []
    for rows in rows_by_query.values():
        rows.sort(key=lambda row: int(row["rank"]))
        hits = [
            RankableHit(
                video_id=row["video_id"],
                frame_id=int(row["frame_id"]),
                timestamp=float(row["timestamp"]),
                score=float(row["score"]),
                extra=row,
            )
            for row in rows
        ]
        reranked = rerank_with_tiered_diversity(
            hits, tiers=tiers, near_duplicate_seconds=near_duplicate_seconds
        )
        for new_rank, hit in enumerate(reranked, start=1):
            row = dict(hit.extra)
            row["rank"] = str(new_rank)
            output_rows.append(row)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(output_rows)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


__all__ = ["REQUIRED_COLUMNS", "rerank_predictions_csv"]
