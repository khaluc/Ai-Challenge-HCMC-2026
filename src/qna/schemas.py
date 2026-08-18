from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any

from vlm.schemas import VLMAnswer

__all__ = ["QACandidate", "QAResult", "VLMAnswer"]


@dataclass(frozen=True)
class QACandidate:
    """One keyframe narrowed down by hybrid retrieval, ready for the VLM."""

    video_id: str
    frame_id: int
    keyframe_path: str
    keyframe_available: bool
    faiss_index: int
    keyframe_index: int
    timestamp: float
    retrieval_rank: int
    retrieval_score: float

    def __post_init__(self) -> None:
        if not self.video_id.strip():
            raise ValueError("QACandidate.video_id must not be blank")
        if isinstance(self.frame_id, bool) or not isinstance(self.frame_id, numbers.Integral):
            raise TypeError("QACandidate.frame_id must be an integer")
        if isinstance(self.retrieval_rank, bool) or not isinstance(
            self.retrieval_rank, numbers.Integral
        ):
            raise TypeError("QACandidate.retrieval_rank must be an integer")
        if int(self.retrieval_rank) < 1:
            raise ValueError("QACandidate.retrieval_rank must be positive")

    @property
    def submit_key(self) -> tuple[str, int]:
        return (self.video_id, self.frame_id)


def _validate_unit_or_none(value: float | None, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"QAResult.{name} must be numeric or null")
    if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
        raise ValueError(f"QAResult.{name} must be between 0 and 1")


@dataclass(frozen=True)
class QAResult:
    """One ranked answer, ready for submission or diagnostics.

    `confidence` is the VLM's own self-reported answer confidence, kept
    unchanged for backward compatibility. `joint_confidence` (when the visual
    route ran with retrieval scores available) is the video/frame/answer
    product used to actually rank candidates — see `qna.pipeline`.
    """

    query_id: str
    question: str
    route: str
    rank: int
    video_id: str | None
    frame_id: int | None
    answer: str | None
    confidence: float
    note: str | None = None
    scene_description: str | None = None
    question_type: str | None = None
    video_score: float | None = None
    frame_score: float | None = None
    joint_confidence: float | None = None
    timestamp: float | None = None
    faiss_index: int | None = None
    keyframe_index: int | None = None
    keyframe_available: bool | None = None

    def __post_init__(self) -> None:
        if not self.query_id.strip():
            raise ValueError("QAResult.query_id must not be blank")
        if not self.question.strip():
            raise ValueError("QAResult.question must not be blank")
        if self.route not in {"visual", "transcript"}:
            raise ValueError("QAResult.route must be 'visual' or 'transcript'")
        if isinstance(self.rank, bool) or not isinstance(self.rank, numbers.Integral):
            raise TypeError("QAResult.rank must be an integer")
        if int(self.rank) < 1:
            raise ValueError("QAResult.rank must be positive")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, numbers.Real):
            raise TypeError("QAResult.confidence must be numeric")
        if not math.isfinite(float(self.confidence)) or not 0 <= float(self.confidence) <= 1:
            raise ValueError("QAResult.confidence must be between 0 and 1")
        if (self.video_id is None) != (self.frame_id is None):
            raise ValueError("QAResult.video_id and frame_id must both be set or both be null")
        if self.question_type is not None and self.question_type not in {"closed_set", "open_ended"}:
            raise ValueError("QAResult.question_type must be 'closed_set', 'open_ended', or null")
        _validate_unit_or_none(self.video_score, "video_score")
        _validate_unit_or_none(self.frame_score, "frame_score")
        _validate_unit_or_none(self.joint_confidence, "joint_confidence")
        if self.timestamp is not None:
            if isinstance(self.timestamp, bool) or not isinstance(self.timestamp, numbers.Real):
                raise TypeError("QAResult.timestamp must be numeric or null")
            if not math.isfinite(float(self.timestamp)) or float(self.timestamp) < 0:
                raise ValueError("QAResult.timestamp must be non-negative")
        if self.faiss_index is not None and (
            isinstance(self.faiss_index, bool) or not isinstance(self.faiss_index, numbers.Integral)
        ):
            raise TypeError("QAResult.faiss_index must be an integer or null")
        if self.keyframe_index is not None and (
            isinstance(self.keyframe_index, bool) or not isinstance(self.keyframe_index, numbers.Integral)
        ):
            raise TypeError("QAResult.keyframe_index must be an integer or null")

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "question": self.question,
            "route": self.route,
            "rank": self.rank,
            "video_id": self.video_id,
            "frame_id": self.frame_id,
            "answer": self.answer,
            "confidence": self.confidence,
            "note": self.note,
            "scene_description": self.scene_description,
            "question_type": self.question_type,
            "video_score": self.video_score,
            "frame_score": self.frame_score,
            "joint_confidence": self.joint_confidence,
            "timestamp": self.timestamp,
            "faiss_index": self.faiss_index,
            "keyframe_index": self.keyframe_index,
            "keyframe_available": self.keyframe_available,
        }
