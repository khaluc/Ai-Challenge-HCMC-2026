from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from retrieval.io import write_btc_submission

from retrieval.hybrid_schemas import HybridHit


HYBRID_PREDICTION_FIELDS = (
    "query_id",
    "rank",
    "video_id",
    "frame_id",
    "score",
    "faiss_index",
    "keyframe_index",
    "timestamp",
    "keyframe_path",
    "keyframe_available",
    "fusion_method",
    "semantic_rank",
    "semantic_score",
    "metadata_rank",
    "metadata_score",
    "metadata_video_rank",
    "metadata_bm25_score",
    "metadata_match_mode",
    "object_rank",
    "object_score",
    "object_mean_confidence",
    "matched_objects",
)


def _validated_hits(hits: Iterable[HybridHit]) -> list[HybridHit]:
    values = list(hits)
    grouped: dict[str, list[HybridHit]] = defaultdict(list)
    for hit in values:
        if not isinstance(hit, HybridHit):
            raise TypeError(f"Expected HybridHit, got {type(hit).__name__}")
        grouped[hit.query_id].append(hit)
    for query_id, query_hits in grouped.items():
        ranks = [hit.rank for hit in query_hits]
        if ranks != list(range(1, len(query_hits) + 1)):
            raise ValueError(f"query {query_id!r} must have dense ordered ranks")
        if len(query_hits) > 100:
            raise ValueError(f"query {query_id!r} exceeds the BTC Top-100 limit")
        pairs = [(hit.video_id, hit.frame_id) for hit in query_hits]
        if len(pairs) != len(set(pairs)):
            raise ValueError(f"query {query_id!r} contains duplicate submit pairs")
    return values


def write_hybrid_predictions(path: str | Path, hits: Iterable[HybridHit]) -> Path:
    values = _validated_hits(hits)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=HYBRID_PREDICTION_FIELDS,
                lineterminator="\n",
            )
            writer.writeheader()
            for hit in values:
                row = {field: getattr(hit, field) for field in HYBRID_PREDICTION_FIELDS}
                row["matched_objects"] = json.dumps(
                    list(hit.matched_objects), ensure_ascii=False
                )
                writer.writerow(row)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def write_hybrid_submission(
    output_dir: str | Path,
    hits: Iterable[HybridHit],
    *,
    zip_path: str | Path | None = None,
) -> list[Path]:
    values = _validated_hits(hits)
    return write_btc_submission(
        output_dir,
        (hit.to_retrieval_hit() for hit in values),
        zip_path=zip_path,
    )


__all__ = [
    "HYBRID_PREDICTION_FIELDS",
    "write_hybrid_predictions",
    "write_hybrid_submission",
]
