from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import numbers
from typing import Any


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str

    def __post_init__(self) -> None:
        canonical_id = self.query_id.strip()
        if not canonical_id:
            raise ValueError("query_id must not be blank")
        object.__setattr__(self, "query_id", canonical_id)
        if not self.text.strip():
            raise ValueError(f"Query {self.query_id!r} has blank text")


@dataclass(frozen=True)
class RetrievalHit:
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

    def __post_init__(self) -> None:
        canonical_query_id = self.query_id.strip()
        canonical_video_id = self.video_id.strip()
        if not canonical_query_id:
            raise ValueError("RetrievalHit query_id must not be blank")
        object.__setattr__(self, "query_id", canonical_query_id)
        if isinstance(self.rank, bool) or not isinstance(self.rank, numbers.Integral) or self.rank < 1:
            raise ValueError("RetrievalHit rank must be a positive integer")
        if not canonical_video_id:
            raise ValueError("RetrievalHit video_id must not be blank")
        object.__setattr__(self, "video_id", canonical_video_id)
        for name, value, minimum in (
            ("frame_id", self.frame_id, 0),
            ("faiss_index", self.faiss_index, 0),
            ("keyframe_index", self.keyframe_index, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
                raise ValueError(f"RetrievalHit {name} must be an integer >= {minimum}")
        if not isinstance(self.score, numbers.Real) or not math.isfinite(float(self.score)):
            raise ValueError("RetrievalHit score must be finite")
        if not isinstance(self.timestamp, numbers.Real) or not math.isfinite(float(self.timestamp)):
            raise ValueError("RetrievalHit timestamp must be finite")
        if not self.keyframe_path:
            raise ValueError("RetrievalHit keyframe_path must not be blank")
        if not isinstance(self.keyframe_available, bool):
            raise ValueError("RetrievalHit keyframe_available must be boolean")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GroundTruthRange:
    query_id: str
    video_id: str
    start_frame_id: int
    end_frame_id: int

    def __post_init__(self) -> None:
        canonical_query_id = self.query_id.strip()
        canonical_video_id = self.video_id.strip()
        if not canonical_query_id:
            raise ValueError("Ground-truth query_id must not be blank")
        object.__setattr__(self, "query_id", canonical_query_id)
        if not canonical_video_id:
            raise ValueError(f"Ground truth {self.query_id!r} has blank video_id")
        object.__setattr__(self, "video_id", canonical_video_id)
        if self.start_frame_id > self.end_frame_id:
            raise ValueError(
                f"Ground truth {self.query_id!r}: start_frame_id exceeds end_frame_id"
            )
        if self.start_frame_id < 0:
            raise ValueError(f"Ground truth {self.query_id!r}: frame ids must be non-negative")

    def accepts(self, video_id: str, frame_id: int) -> bool:
        return (
            video_id == self.video_id
            and self.start_frame_id <= frame_id <= self.end_frame_id
        )
