from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VerificationCandidate:
    """One nearby frame the VLM will compare against its neighbors."""

    frame_id: int
    timestamp: float
    image_bytes: bytes

    def __post_init__(self) -> None:
        if isinstance(self.frame_id, bool) or not isinstance(self.frame_id, numbers.Integral):
            raise TypeError("VerificationCandidate.frame_id must be an integer")
        if not isinstance(self.image_bytes, (bytes, bytearray)) or not self.image_bytes:
            raise ValueError("VerificationCandidate.image_bytes must be nonempty bytes")


@dataclass(frozen=True)
class OriginalEventFrame:
    """Pre-verification assignment for one event — the safe fallback if
    verification would break the final f1 < f2 < ... < fn order."""

    event_index: int
    event_text: str
    frame_id: int
    timestamp: float

    def __post_init__(self) -> None:
        if isinstance(self.event_index, bool) or not isinstance(self.event_index, numbers.Integral):
            raise TypeError("OriginalEventFrame.event_index must be an integer")
        if not self.event_text.strip():
            raise ValueError("OriginalEventFrame.event_text must not be blank")


@dataclass(frozen=True)
class VerifiedEvent:
    event_index: int
    event_text: str
    video_id: str
    frame_id: int
    timestamp: float
    confidence: float
    reason: str
    candidates_considered: int
    verified: bool

    def __post_init__(self) -> None:
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, numbers.Real):
            raise TypeError("VerifiedEvent.confidence must be numeric")
        if not math.isfinite(float(self.confidence)) or not 0 <= float(self.confidence) <= 1:
            raise ValueError("VerifiedEvent.confidence must be between 0 and 1")

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_index": self.event_index,
            "event_text": self.event_text,
            "video_id": self.video_id,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "reason": self.reason,
            "candidates_considered": self.candidates_considered,
            "verified": self.verified,
        }


@dataclass(frozen=True)
class TRAKEVerificationResult:
    video_id: str
    events: tuple[VerifiedEvent, ...]
    monotonic: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "monotonic": self.monotonic,
            "events": [event.as_dict() for event in self.events],
        }


__all__ = [
    "OriginalEventFrame",
    "TRAKEVerificationResult",
    "VerificationCandidate",
    "VerifiedEvent",
]
