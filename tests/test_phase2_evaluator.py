from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from retrieval.evaluator import (
    Prediction,
    evaluate,
    evaluate_files,
    load_ground_truth,
    load_predictions,
)
from retrieval.schemas import GroundTruthRange


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_golden_four_query_score_ranges_and_missing_predictions(tmp_path: Path) -> None:
    ground_truth_path = tmp_path / "ground_truth.jsonl"
    targets = [
        {"query_id": "q1", "video_id": "v1", "start_frame_id": 10, "end_frame_id": 20},
        # A query may have more than one acceptable video/frame interval.
        {"query_id": "q1", "video_id": "v2", "start_frame_id": 100, "end_frame_id": 105},
        {"query_id": "q2", "video_id": "v3", "frame_id": 300},
        {"query_id": "q3", "video_id": "v4", "start_frame_id": 400, "end_frame_id": 410},
        {"query_id": "q4", "video_id": "v5", "frame_id": 500},
    ]
    ground_truth_path.write_text(
        "".join(json.dumps(row) + "\n" for row in targets), encoding="utf-8"
    )

    prediction_path = tmp_path / "predictions.csv"
    rows: list[dict[str, object]] = [
        # Endpoints are inclusive; this hits q1 through its second range at rank 1.
        {"query_id": "q1", "rank": 1, "video_id": "v2", "frame_id": 105, "score": 0.99},
    ]
    rows.extend(
        {
            "query_id": "q2",
            "rank": rank,
            "video_id": "v3",
            "frame_id": 300 if rank == 5 else 1000 + rank,
            "score": 1.0 - rank / 100,
        }
        for rank in range(1, 6)
    )
    rows.extend(
        {
            "query_id": "q3",
            "rank": rank,
            "video_id": "v4",
            "frame_id": 400 if rank == 50 else 2000 + rank,
            "score": 1.0 - rank / 100,
        }
        for rank in range(1, 51)
    )
    # q4 is deliberately absent: a query with no predictions is a miss.
    _write_csv(
        prediction_path,
        ["query_id", "rank", "video_id", "frame_id", "score"],
        rows,
    )

    result = evaluate_files(ground_truth_path, prediction_path)

    assert result.num_queries == 4
    assert result.recalls == {1: 0.25, 5: 0.5, 20: 0.5, 50: 0.75, 100: 0.75}
    assert result.recall_sum == pytest.approx(2.75)
    assert result.query_score_sum == pytest.approx(2.2)
    assert result.final_score == pytest.approx(0.55)
    assert [row.first_relevant_rank for row in result.per_query] == [1, 5, 50, None]
    assert result.per_query[0].first_relevant_score == pytest.approx(0.99)
    assert [row.query_score for row in result.per_query] == [1.0, 0.8, 0.4, 0.0]
    # The public representation must be directly writable as JSON.
    json.dumps(result.as_dict())


def test_load_ground_truth_csv_supports_exact_frame_alias(tmp_path: Path) -> None:
    path = tmp_path / "ground_truth.csv"
    _write_csv(
        path,
        ["query_id", "video_id", "frame_id"],
        [
            {"query_id": "q1", "video_id": "L21_V001", "frame_id": 0},
            {"query_id": "q2", "video_id": "L21_V002", "frame_id": 42},
        ],
    )

    targets = load_ground_truth(path)

    assert targets == [
        GroundTruthRange("q1", "L21_V001", 0, 0),
        GroundTruthRange("q2", "L21_V002", 42, 42),
    ]


def test_prediction_csv_score_is_optional_and_rows_are_sorted_by_rank(
    tmp_path: Path,
) -> None:
    path = tmp_path / "predictions.csv"
    _write_csv(
        path,
        ["query_id", "rank", "video_id", "frame_id"],
        [
            {"query_id": "q1", "rank": 2, "video_id": "v", "frame_id": 2},
            {"query_id": "q1", "rank": 1, "video_id": "v", "frame_id": 1},
        ],
    )

    loaded = load_predictions(path, known_query_ids={"q1"})
    # Loading preserves the file; evaluation is rank-driven rather than row-order-driven.
    assert [row.rank for row in loaded] == [2, 1]
    result = evaluate(
        [GroundTruthRange("q1", "v", 1, 1)],
        loaded,
    )
    assert result.per_query[0].first_relevant_rank == 1
    assert result.per_query[0].first_relevant_score is None


@pytest.mark.parametrize(
    ("predictions", "message"),
    [
        (
            [Prediction("unknown", 1, "v", 1)],
            "unknown query_id",
        ),
        (
            [Prediction("q1", 1, "v", 1), Prediction("q1", 3, "v", 3)],
            "ranks must be dense",
        ),
        (
            [Prediction("q1", 1, "v", 1), Prediction("q1", 2, "v", 1)],
            "duplicate submission pair",
        ),
        (
            [Prediction("q1", 1, "v", 1), Prediction("q1", 1, "v", 2)],
            "duplicate rank",
        ),
    ],
)
def test_invalid_predictions_are_rejected(
    predictions: list[Prediction], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate([GroundTruthRange("q1", "target", 10, 20)], predictions)


def test_prediction_rank_over_100_is_rejected() -> None:
    with pytest.raises(ValueError, match="between 1 and 100"):
        Prediction("q1", 101, "v", 1)


@pytest.mark.parametrize(
    ("first_relevant_rank", "expected_score"),
    [
        (1, 1.0),
        (2, 0.8),
        (5, 0.8),
        (6, 0.6),
        (20, 0.6),
        (21, 0.4),
        (50, 0.4),
        (51, 0.2),
        (100, 0.2),
    ],
)
def test_first_relevant_rank_cutoff_boundaries(
    first_relevant_rank: int, expected_score: float
) -> None:
    predictions = [
        Prediction(
            "q1",
            rank,
            "target" if rank == first_relevant_rank else "wrong",
            15 if rank == first_relevant_rank else rank,
            1.0 / rank,
        )
        for rank in range(1, first_relevant_rank + 1)
    ]

    result = evaluate([GroundTruthRange("q1", "target", 10, 20)], predictions)

    assert result.final_score == pytest.approx(expected_score)
    assert result.per_query[0].first_relevant_rank == first_relevant_rank


def test_invalid_ground_truth_range_is_rejected_with_location(tmp_path: Path) -> None:
    path = tmp_path / "ground_truth.csv"
    _write_csv(
        path,
        ["query_id", "video_id", "start_frame_id", "end_frame_id"],
        [
            {
                "query_id": "q1",
                "video_id": "v1",
                "start_frame_id": 20,
                "end_frame_id": 10,
            }
        ],
    )

    with pytest.raises(ValueError, match=r"ground_truth\.csv:2.*exceeds"):
        load_ground_truth(path)
