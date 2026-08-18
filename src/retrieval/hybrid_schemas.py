from __future__ import annotations

import math
import numbers
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from retrieval.clip_store import FrameRecord
from retrieval.schemas import RetrievalHit


BRANCH_NAMES = ("semantic", "metadata", "objects")


@dataclass(frozen=True)
class QueryAnalysis:
    """Deterministic query signals used by the non-neural branches."""

    text: str
    metadata_terms: tuple[str, ...]
    object_concepts: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "metadata_terms": list(self.metadata_terms),
            "object_concepts": list(self.object_concepts),
        }


@dataclass(frozen=True)
class BranchHit:
    """One ranked frame from a single retrieval branch."""

    branch: str
    rank: int
    raw_score: float
    frame: FrameRecord
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.branch not in BRANCH_NAMES:
            raise ValueError(f"Unknown retrieval branch: {self.branch!r}")
        if isinstance(self.rank, bool) or not isinstance(self.rank, numbers.Integral):
            raise TypeError("BranchHit rank must be an integer")
        if int(self.rank) < 1:
            raise ValueError("BranchHit rank must be positive")
        if isinstance(self.raw_score, bool) or not isinstance(
            self.raw_score, numbers.Real
        ):
            raise TypeError("BranchHit raw_score must be numeric")
        if not math.isfinite(float(self.raw_score)):
            raise ValueError("BranchHit raw_score must be finite")
        if not isinstance(self.frame, FrameRecord):
            raise TypeError("BranchHit frame must be a FrameRecord")

    @property
    def submit_key(self) -> tuple[str, int]:
        return (self.frame.video_id, self.frame.video_frame_id)


@dataclass(frozen=True)
class BranchEvidence:
    rank: int
    raw_score: float
    contribution: float
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FusedFrame:
    score: float
    frame: FrameRecord
    evidence: Mapping[str, BranchEvidence]


@dataclass(frozen=True)
class HybridHit:
    """Submission-ready hybrid result plus branch-level diagnostics."""

    query_id: str
    rank: int
    video_id: str
    frame_id: int
    score: float
    faiss_index: int
    keyframe_index: int
    timestamp: float
    keyframe_path: str
    keyframe_available: bool
    fusion_method: str
    semantic_rank: int | None = None
    semantic_score: float | None = None
    metadata_rank: int | None = None
    metadata_score: float | None = None
    metadata_video_rank: int | None = None
    metadata_bm25_score: float | None = None
    metadata_match_mode: str | None = None
    object_rank: int | None = None
    object_score: float | None = None
    object_mean_confidence: float | None = None
    matched_objects: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Reuse the strict Phase 2 submission contract for the shared fields.
        self.to_retrieval_hit()
        if self.fusion_method not in {"rrf", "weighted"}:
            raise ValueError("HybridHit fusion_method must be 'rrf' or 'weighted'")
        for name in (
            "semantic_rank",
            "metadata_rank",
            "metadata_video_rank",
            "object_rank",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, numbers.Integral)
                or value < 1
            ):
                raise ValueError(f"HybridHit {name} must be a positive integer or null")
        for name in (
            "semantic_score",
            "metadata_score",
            "metadata_bm25_score",
            "object_score",
            "object_mean_confidence",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, numbers.Real)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"HybridHit {name} must be finite or null")
        if self.metadata_match_mode not in {None, "and", "or"}:
            raise ValueError("HybridHit metadata_match_mode must be 'and', 'or', or null")
        if not isinstance(self.matched_objects, tuple) or any(
            not isinstance(value, str) or not value for value in self.matched_objects
        ):
            raise ValueError("HybridHit matched_objects must be a tuple of nonblank strings")

    def to_retrieval_hit(self) -> RetrievalHit:
        return RetrievalHit(
            query_id=self.query_id,
            rank=self.rank,
            video_id=self.video_id,
            frame_id=self.frame_id,
            score=self.score,
            faiss_index=self.faiss_index,
            keyframe_index=self.keyframe_index,
            timestamp=self.timestamp,
            keyframe_path=self.keyframe_path,
            keyframe_available=self.keyframe_available,
        )

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["matched_objects"] = list(self.matched_objects)
        return value


__all__ = [
    "BRANCH_NAMES",
    "BranchEvidence",
    "BranchHit",
    "FusedFrame",
    "HybridHit",
    "QueryAnalysis",
]
