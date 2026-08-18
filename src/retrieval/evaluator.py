from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schemas import GroundTruthRange, RetrievalHit


DEFAULT_CUTOFFS = (1, 5, 20, 50, 100)
MAX_PREDICTIONS_PER_QUERY = 100


@dataclass(frozen=True)
class Prediction:
    """One row in the internal, ranked prediction format."""

    query_id: str
    rank: int
    video_id: str
    frame_id: int
    score: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or not self.query_id.strip():
            raise ValueError("Prediction query_id must not be blank")
        if not isinstance(self.video_id, str) or not self.video_id.strip():
            raise ValueError(f"Prediction {self.query_id!r} has blank video_id")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            raise ValueError(f"Prediction {self.query_id!r} rank must be an integer")
        if not 1 <= self.rank <= MAX_PREDICTIONS_PER_QUERY:
            raise ValueError(
                f"Prediction {self.query_id!r} has rank {self.rank}; "
                f"rank must be between 1 and {MAX_PREDICTIONS_PER_QUERY}"
            )
        if isinstance(self.frame_id, bool) or not isinstance(self.frame_id, int):
            raise ValueError(
                f"Prediction {self.query_id!r} rank {self.rank} frame_id must be an integer"
            )
        if self.frame_id < 0:
            raise ValueError(
                f"Prediction {self.query_id!r} rank {self.rank} frame_id must be non-negative"
            )
        if self.score is not None:
            if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
                raise ValueError(
                    f"Prediction {self.query_id!r} rank {self.rank} score must be numeric"
                )
            if not math.isfinite(self.score):
                raise ValueError(
                    f"Prediction {self.query_id!r} rank {self.rank} has a non-finite score"
                )


@dataclass(frozen=True)
class QueryEvaluation:
    query_id: str
    first_relevant_rank: int | None
    first_relevant_score: float | None
    cutoff_hits: Mapping[int, bool]
    query_score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "first_relevant_rank": self.first_relevant_rank,
            "first_relevant_score": self.first_relevant_score,
            "cutoff_hits": {
                f"R@{cutoff}": bool(self.cutoff_hits[cutoff])
                for cutoff in DEFAULT_CUTOFFS
            },
            "query_score": self.query_score,
        }


@dataclass(frozen=True)
class EvaluationResult:
    num_queries: int
    recalls: Mapping[int, float]
    recall_sum: float
    query_score_sum: float
    final_score: float
    per_query: tuple[QueryEvaluation, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_queries": self.num_queries,
            "recalls": {
                f"R@{cutoff}": self.recalls[cutoff] for cutoff in DEFAULT_CUTOFFS
            },
            "recall_sum": self.recall_sum,
            "query_score_sum": self.query_score_sum,
            "final_score": self.final_score,
            "per_query": [row.as_dict() for row in self.per_query],
        }


def _required_columns(
    fieldnames: Sequence[str] | None,
    *,
    path: Path,
    kind: str,
) -> set[str]:
    if fieldnames is None:
        raise ValueError(f"{kind} file {path} is empty or has no CSV header")
    if len(set(fieldnames)) != len(fieldnames):
        raise ValueError(f"{kind} file {path} has duplicate CSV column names")
    return set(fieldnames)


def _parse_identifier(value: Any, *, field: str, location: str) -> str:
    if value is None or isinstance(value, (dict, list, tuple, bool)):
        raise ValueError(f"{location}: {field} must be a non-blank scalar")
    parsed = str(value).strip()
    if not parsed:
        raise ValueError(f"{location}: {field} must not be blank")
    return parsed


def _parse_int(value: Any, *, field: str, location: str) -> int:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{location}: {field} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer() and math.isfinite(value):
            return int(value)
        raise ValueError(f"{location}: {field} must be an integer, got {value!r}")
    text = str(value).strip()
    try:
        return int(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{location}: {field} must be an integer, got {value!r}"
        ) from exc


def _parse_optional_score(value: Any, *, location: str) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise ValueError(f"{location}: score must be a finite number")
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location}: score must be a finite number") from exc
    if not math.isfinite(score):
        raise ValueError(f"{location}: score must be a finite number")
    return score


def _ground_truth_from_mapping(
    row: Mapping[str, Any], *, location: str
) -> GroundTruthRange:
    query_id = _parse_identifier(row.get("query_id"), field="query_id", location=location)
    video_id = _parse_identifier(row.get("video_id"), field="video_id", location=location)

    frame_value = row.get("frame_id")
    has_frame = frame_value is not None and str(frame_value).strip() != ""
    start_value = row.get("start_frame_id")
    end_value = row.get("end_frame_id")
    has_start = start_value is not None and str(start_value).strip() != ""
    has_end = end_value is not None and str(end_value).strip() != ""

    if has_frame and (has_start or has_end):
        raise ValueError(
            f"{location}: use either frame_id or start_frame_id/end_frame_id, not both"
        )
    if has_frame:
        frame_id = _parse_int(frame_value, field="frame_id", location=location)
        start_frame_id = frame_id
        end_frame_id = frame_id
    elif has_start and has_end:
        start_frame_id = _parse_int(
            start_value, field="start_frame_id", location=location
        )
        end_frame_id = _parse_int(end_value, field="end_frame_id", location=location)
    else:
        raise ValueError(
            f"{location}: provide frame_id or both start_frame_id and end_frame_id"
        )

    try:
        return GroundTruthRange(
            query_id=query_id,
            video_id=video_id,
            start_frame_id=start_frame_id,
            end_frame_id=end_frame_id,
        )
    except ValueError as exc:
        raise ValueError(f"{location}: {exc}") from exc


def load_ground_truth(path: str | Path) -> list[GroundTruthRange]:
    """Load canonical ground truth from CSV or JSON Lines.

    Range rows use ``start_frame_id`` and ``end_frame_id`` (inclusive). An exact
    target may instead use the ``frame_id`` alias.
    """

    source = Path(path)
    suffix = source.suffix.lower()
    rows: list[GroundTruthRange] = []

    if suffix == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = _required_columns(
                reader.fieldnames, path=source, kind="Ground-truth"
            )
            missing_ids = {"query_id", "video_id"} - columns
            has_range = {"start_frame_id", "end_frame_id"} <= columns
            has_exact = "frame_id" in columns
            if missing_ids or not (has_range or has_exact):
                expected = (
                    "query_id, video_id, start_frame_id, end_frame_id "
                    "(or query_id, video_id, frame_id)"
                )
                raise ValueError(
                    f"Ground-truth file {source} must contain {expected}"
                )
            for line_number, row in enumerate(reader, start=2):
                rows.append(
                    _ground_truth_from_mapping(
                        row, location=f"{source}:{line_number}"
                    )
                )
    elif suffix in {".jsonl", ".ndjson"}:
        with source.open("r", encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                location = f"{source}:{line_number}"
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{location}: invalid JSON: {exc.msg}") from exc
                if not isinstance(payload, dict):
                    raise ValueError(f"{location}: each JSONL row must be an object")
                rows.append(_ground_truth_from_mapping(payload, location=location))
    else:
        raise ValueError(
            f"Unsupported ground-truth format {source.suffix!r}; use .csv or .jsonl"
        )

    if not rows:
        raise ValueError(f"Ground-truth file {source} contains no target rows")
    return rows


def _prediction_from_mapping(row: Mapping[str, Any], *, location: str) -> Prediction:
    query_id = _parse_identifier(row.get("query_id"), field="query_id", location=location)
    video_id = _parse_identifier(row.get("video_id"), field="video_id", location=location)
    rank = _parse_int(row.get("rank"), field="rank", location=location)
    frame_id = _parse_int(row.get("frame_id"), field="frame_id", location=location)
    score = _parse_optional_score(row.get("score"), location=location)
    try:
        return Prediction(
            query_id=query_id,
            rank=rank,
            video_id=video_id,
            frame_id=frame_id,
            score=score,
        )
    except ValueError as exc:
        raise ValueError(f"{location}: {exc}") from exc


def load_predictions(
    path: str | Path,
    known_query_ids: Iterable[str] | None = None,
) -> list[Prediction]:
    """Load and validate the internal ranked-prediction CSV."""

    source = Path(path)
    predictions: list[Prediction] = []
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = _required_columns(reader.fieldnames, path=source, kind="Prediction")
        required = {"query_id", "rank", "video_id", "frame_id"}
        missing = required - columns
        if missing:
            raise ValueError(
                f"Prediction file {source} is missing columns: {', '.join(sorted(missing))}"
            )
        for line_number, row in enumerate(reader, start=2):
            predictions.append(
                _prediction_from_mapping(row, location=f"{source}:{line_number}")
            )

    _validate_predictions(predictions, known_query_ids=known_query_ids)
    return predictions


def _coerce_prediction(value: Any, *, position: int) -> Prediction:
    if isinstance(value, Prediction):
        return value
    if isinstance(value, RetrievalHit):
        return Prediction(
            query_id=value.query_id,
            rank=value.rank,
            video_id=value.video_id,
            frame_id=value.frame_id,
            score=value.score,
        )
    if isinstance(value, Mapping):
        return _prediction_from_mapping(value, location=f"prediction row {position}")
    raise TypeError(
        f"prediction row {position} must be Prediction, RetrievalHit, or a mapping"
    )


def _validate_predictions(
    predictions: Iterable[Prediction],
    *,
    known_query_ids: Iterable[str] | None,
) -> dict[str, list[Prediction]]:
    known = set(known_query_ids) if known_query_ids is not None else None
    grouped: dict[str, list[Prediction]] = defaultdict(list)
    seen_ranks: dict[str, set[int]] = defaultdict(set)
    seen_pairs: dict[str, set[tuple[str, int]]] = defaultdict(set)

    for prediction in predictions:
        query_id = prediction.query_id
        if known is not None and query_id not in known:
            raise ValueError(f"Prediction contains unknown query_id {query_id!r}")
        if prediction.rank in seen_ranks[query_id]:
            raise ValueError(
                f"Query {query_id!r} contains duplicate rank {prediction.rank}"
            )
        pair = (prediction.video_id, prediction.frame_id)
        if pair in seen_pairs[query_id]:
            raise ValueError(
                f"Query {query_id!r} contains duplicate submission pair {pair!r}"
            )
        seen_ranks[query_id].add(prediction.rank)
        seen_pairs[query_id].add(pair)
        grouped[query_id].append(prediction)

    for query_id, query_predictions in grouped.items():
        if len(query_predictions) > MAX_PREDICTIONS_PER_QUERY:
            raise ValueError(
                f"Query {query_id!r} has {len(query_predictions)} predictions; "
                f"maximum is {MAX_PREDICTIONS_PER_QUERY}"
            )
        actual_ranks = sorted(seen_ranks[query_id])
        expected_ranks = list(range(1, len(query_predictions) + 1))
        if actual_ranks != expected_ranks:
            raise ValueError(
                f"Query {query_id!r} ranks must be dense 1..{len(query_predictions)}; "
                f"got {actual_ranks}"
            )
        query_predictions.sort(key=lambda row: row.rank)
    return dict(grouped)


def evaluate(
    ground_truth: Iterable[GroundTruthRange],
    predictions: Iterable[Prediction | RetrievalHit | Mapping[str, Any]],
) -> EvaluationResult:
    """Evaluate ranked keyframes with macro Recall@K over ground-truth queries."""

    targets_by_query: dict[str, list[GroundTruthRange]] = {}
    query_order: list[str] = []
    for position, target in enumerate(ground_truth, start=1):
        if not isinstance(target, GroundTruthRange):
            raise TypeError(
                f"ground-truth row {position} must be a GroundTruthRange instance"
            )
        if target.query_id not in targets_by_query:
            targets_by_query[target.query_id] = []
            query_order.append(target.query_id)
        targets_by_query[target.query_id].append(target)
    if not targets_by_query:
        raise ValueError("Ground truth contains no queries")

    normalized_predictions = [
        _coerce_prediction(value, position=position)
        for position, value in enumerate(predictions, start=1)
    ]
    predictions_by_query = _validate_predictions(
        normalized_predictions, known_query_ids=targets_by_query
    )

    hit_counts = {cutoff: 0 for cutoff in DEFAULT_CUTOFFS}
    per_query: list[QueryEvaluation] = []
    for query_id in query_order:
        first_relevant: Prediction | None = None
        targets = targets_by_query[query_id]
        for prediction in predictions_by_query.get(query_id, ()):
            if any(
                target.accepts(prediction.video_id, prediction.frame_id)
                for target in targets
            ):
                first_relevant = prediction
                break

        first_rank = first_relevant.rank if first_relevant is not None else None
        cutoff_hits = {
            cutoff: first_rank is not None and first_rank <= cutoff
            for cutoff in DEFAULT_CUTOFFS
        }
        for cutoff, hit in cutoff_hits.items():
            hit_counts[cutoff] += int(hit)
        per_query.append(
            QueryEvaluation(
                query_id=query_id,
                first_relevant_rank=first_rank,
                first_relevant_score=(
                    first_relevant.score if first_relevant is not None else None
                ),
                cutoff_hits=cutoff_hits,
                query_score=sum(cutoff_hits.values()) / len(DEFAULT_CUTOFFS),
            )
        )

    num_queries = len(query_order)
    recalls = {
        cutoff: hit_counts[cutoff] / num_queries for cutoff in DEFAULT_CUTOFFS
    }
    recall_sum = sum(recalls.values())
    final_score = recall_sum / len(DEFAULT_CUTOFFS)
    query_score_sum = sum(row.query_score for row in per_query)
    return EvaluationResult(
        num_queries=num_queries,
        recalls=recalls,
        recall_sum=recall_sum,
        query_score_sum=query_score_sum,
        final_score=final_score,
        per_query=tuple(per_query),
    )


def evaluate_files(
    ground_truth_path: str | Path, prediction_path: str | Path
) -> EvaluationResult:
    ground_truth = load_ground_truth(ground_truth_path)
    known_query_ids = {target.query_id for target in ground_truth}
    predictions = load_predictions(
        prediction_path, known_query_ids=known_query_ids
    )
    return evaluate(ground_truth, predictions)


__all__ = [
    "DEFAULT_CUTOFFS",
    "MAX_PREDICTIONS_PER_QUERY",
    "EvaluationResult",
    "Prediction",
    "QueryEvaluation",
    "evaluate",
    "evaluate_files",
    "load_ground_truth",
    "load_predictions",
]
