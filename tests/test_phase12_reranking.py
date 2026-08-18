from __future__ import annotations

import csv
from pathlib import Path

import pytest

from submission.io import rerank_predictions_csv
from submission.ranking_optimizer import (
    RankableHit,
    RankingTier,
    rerank_with_tiered_diversity,
)


def _hit(video_id: str, frame_id: int, timestamp: float, score: float) -> RankableHit:
    return RankableHit(video_id=video_id, frame_id=frame_id, timestamp=timestamp, score=score)


def test_rerank_keeps_pure_relevance_order_in_top_tier() -> None:
    candidates = [_hit("V1", 100 + i, 10.0 + i * 0.1, 1.0 - i * 0.01) for i in range(5)]

    reranked = rerank_with_tiered_diversity(candidates)

    assert [item.frame_id for item in reranked] == [item.frame_id for item in candidates]


def test_rerank_prefers_diversity_in_tail_tier() -> None:
    v1_cluster = [_hit("V1", 100 + i, 10.0 + i * 0.5, 0.9 - i * 0.001) for i in range(10)]
    v2_alt = _hit("V2", 500, 50.0, 0.85)
    candidates = v1_cluster + [v2_alt]

    tiers = (RankingTier(1, 3, 0.0), RankingTier(4, 11, 0.9))
    reranked = rerank_with_tiered_diversity(candidates, tiers=tiers, near_duplicate_seconds=5.0)

    assert all(item.video_id == "V1" for item in reranked[:3])
    v2_position = next(i for i, item in enumerate(reranked) if item.video_id == "V2")
    assert v2_position <= 5


def test_rerank_returns_empty_for_empty_input() -> None:
    assert rerank_with_tiered_diversity([]) == []


def test_rerank_preserves_all_candidates() -> None:
    candidates = [_hit("V1", i, float(i), 1.0 - i * 0.01) for i in range(30)]
    reranked = rerank_with_tiered_diversity(candidates)
    assert {item.frame_id for item in reranked} == {item.frame_id for item in candidates}
    assert len(reranked) == len(candidates)


def test_ranking_tier_rejects_invalid_range() -> None:
    with pytest.raises(ValueError):
        RankingTier(5, 1, 0.0)


def test_ranking_tier_rejects_out_of_range_weight() -> None:
    with pytest.raises(ValueError):
        RankingTier(1, 5, 1.5)


def test_rerank_predictions_csv_round_trips(tmp_path: Path) -> None:
    input_path = tmp_path / "predictions.csv"
    rows = [
        {
            "query_id": "q1",
            "rank": "1",
            "video_id": "V1",
            "frame_id": "100",
            "score": "0.99",
            "timestamp": "10.0",
        },
        {
            "query_id": "q1",
            "rank": "2",
            "video_id": "V1",
            "frame_id": "101",
            "score": "0.98",
            "timestamp": "10.5",
        },
        {
            "query_id": "q1",
            "rank": "3",
            "video_id": "V2",
            "frame_id": "200",
            "score": "0.5",
            "timestamp": "50.0",
        },
    ]
    with input_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    output_path = tmp_path / "reranked.csv"
    rerank_predictions_csv(input_path, output_path)

    with output_path.open("r", encoding="utf-8") as stream:
        out_rows = list(csv.DictReader(stream))

    assert [row["rank"] for row in out_rows] == ["1", "2", "3"]
    assert {row["video_id"] for row in out_rows} == {"V1", "V2"}


def test_rerank_predictions_csv_rejects_missing_columns(tmp_path: Path) -> None:
    input_path = tmp_path / "bad.csv"
    with input_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["query_id", "video_id"])
        writer.writeheader()
        writer.writerow({"query_id": "q1", "video_id": "V1"})

    with pytest.raises(ValueError):
        rerank_predictions_csv(input_path, tmp_path / "out.csv")
