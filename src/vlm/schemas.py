from __future__ import annotations

import math
import numbers
from dataclasses import dataclass


@dataclass(frozen=True)
class VLMAnswer:
    """One VLM response to a (frame, question) pair."""

    answer: str
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.answer, str) or not self.answer.strip():
            raise ValueError("VLMAnswer.answer must be a nonblank string")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, numbers.Real):
            raise TypeError("VLMAnswer.confidence must be numeric")
        if not math.isfinite(float(self.confidence)) or not 0 <= float(self.confidence) <= 1:
            raise ValueError("VLMAnswer.confidence must be between 0 and 1")


__all__ = ["VLMAnswer"]
