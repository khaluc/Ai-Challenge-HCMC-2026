from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any

from llm.event_schemas import EventSequence


@dataclass(frozen=True)
class VideoCandidateScore:
    """One candidate video's aggregated relevance for a TRAKE query.

    ``total_score = w1*global_similarity + w2*event_coverage + w3*bm25_score
    + w4*multi_query_vote`` (weights from `VideoRetrievalConfig`), each
    component independently normalized to [0, 1].
    """

    video_id: str
    rank: int
    total_score: float
    global_similarity: float
    event_coverage: float
    bm25_score: float
    multi_query_vote: float
    matched_events: tuple[str, ...]
    best_frame_id: int | None

    def __post_init__(self) -> None:
        if not self.video_id.strip():
            raise ValueError("VideoCandidateScore.video_id must not be blank")
        if isinstance(self.rank, bool) or not isinstance(self.rank, numbers.Integral):
            raise TypeError("VideoCandidateScore.rank must be an integer")
        if int(self.rank) < 1:
            raise ValueError("VideoCandidateScore.rank must be positive")
        for name in ("event_coverage", "bm25_score", "multi_query_vote"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError(f"VideoCandidateScore.{name} must be numeric")
            if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
                raise ValueError(f"VideoCandidateScore.{name} must be between 0 and 1")
        if isinstance(self.global_similarity, bool) or not isinstance(
            self.global_similarity, numbers.Real
        ):
            raise TypeError("VideoCandidateScore.global_similarity must be numeric")
        if not math.isfinite(float(self.global_similarity)) or not 0 <= float(
            self.global_similarity
        ) <= 1:
            raise ValueError("VideoCandidateScore.global_similarity must be between 0 and 1")
        if isinstance(self.total_score, bool) or not isinstance(self.total_score, numbers.Real):
            raise TypeError("VideoCandidateScore.total_score must be numeric")
        if not math.isfinite(float(self.total_score)):
            raise ValueError("VideoCandidateScore.total_score must be finite")

    def as_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "rank": self.rank,
            "total_score": self.total_score,
            "global_similarity": self.global_similarity,
            "event_coverage": self.event_coverage,
            "bm25_score": self.bm25_score,
            "multi_query_vote": self.multi_query_vote,
            "matched_events": list(self.matched_events),
            "best_frame_id": self.best_frame_id,
        }


@dataclass(frozen=True)
class TRAKERetrievalResult:
    """Top candidate videos for one TRAKE query — never a single video,
    since a wrong video zeroes the whole R-Score."""

    query_id: str
    query: str
    event_sequence: EventSequence
    candidates: tuple[VideoCandidateScore, ...]

    def __post_init__(self) -> None:
        if not self.query_id.strip():
            raise ValueError("TRAKERetrievalResult.query_id must not be blank")
        ranks = [candidate.rank for candidate in self.candidates]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("TRAKERetrievalResult.candidates must have dense ranks from 1")

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "event_sequence": self.event_sequence.as_dict(),
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }


__all__ = ["TRAKERetrievalResult", "VideoCandidateScore"]
