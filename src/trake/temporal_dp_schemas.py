from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any

from retrieval.clip_store import FrameRecord


@dataclass(frozen=True)
class EventFrameAssignment:
    """One event bound to one keyframe by the coarse DP alignment."""

    event_index: int
    event_text: str
    frame: FrameRecord
    similarity: float

    def __post_init__(self) -> None:
        if isinstance(self.event_index, bool) or not isinstance(self.event_index, numbers.Integral):
            raise TypeError("EventFrameAssignment.event_index must be an integer")
        if int(self.event_index) < 0:
            raise ValueError("EventFrameAssignment.event_index must be non-negative")
        if not self.event_text.strip():
            raise ValueError("EventFrameAssignment.event_text must not be blank")
        if not isinstance(self.frame, FrameRecord):
            raise TypeError("EventFrameAssignment.frame must be a FrameRecord")
        if isinstance(self.similarity, bool) or not isinstance(self.similarity, numbers.Real):
            raise TypeError("EventFrameAssignment.similarity must be numeric")
        if not math.isfinite(float(self.similarity)):
            raise ValueError("EventFrameAssignment.similarity must be finite")

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_index": self.event_index,
            "event_text": self.event_text,
            "video_id": self.frame.video_id,
            "frame_id": self.frame.video_frame_id,
            "keyframe_index": self.frame.keyframe_index,
            "timestamp": self.frame.timestamp,
            "similarity": self.similarity,
        }


@dataclass(frozen=True)
class CoarseAlignment:
    """Best f1 < f2 < ... < fn assignment maximizing sum S(Ei, fi) inside one video."""

    video_id: str
    events: tuple[str, ...]
    assignments: tuple[EventFrameAssignment, ...]
    total_score: float
    feasible: bool

    def __post_init__(self) -> None:
        if not self.video_id.strip():
            raise ValueError("CoarseAlignment.video_id must not be blank")
        if not self.events:
            raise ValueError("CoarseAlignment.events must not be empty")
        if isinstance(self.total_score, bool) or not isinstance(self.total_score, numbers.Real):
            raise TypeError("CoarseAlignment.total_score must be numeric")
        if self.feasible:
            if len(self.assignments) != len(self.events):
                raise ValueError(
                    "CoarseAlignment.assignments must have one entry per event when feasible"
                )
            indices = [item.event_index for item in self.assignments]
            if indices != list(range(len(self.events))):
                raise ValueError("CoarseAlignment.assignments must be ordered by event_index 0..N-1")
            keyframe_indices = [item.frame.keyframe_index for item in self.assignments]
            if keyframe_indices != sorted(keyframe_indices) or len(set(keyframe_indices)) != len(
                keyframe_indices
            ):
                raise ValueError(
                    "CoarseAlignment.assignments must be strictly increasing by keyframe_index"
                )
            if not math.isfinite(float(self.total_score)):
                raise ValueError("CoarseAlignment.total_score must be finite when feasible")
        elif self.assignments:
            raise ValueError("CoarseAlignment.assignments must be empty when not feasible")

    def as_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "events": list(self.events),
            "feasible": self.feasible,
            "total_score": self.total_score,
            "assignments": [item.as_dict() for item in self.assignments],
        }


__all__ = ["CoarseAlignment", "EventFrameAssignment"]
