from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EventSequence:
    """Ordered visual events a TRAKE query decomposes into, plus optional
    whole-query paraphrases (LLM query expansions) for video-level search."""

    query: str
    events: tuple[str, ...]
    expansions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("EventSequence.query must not be blank")
        if not self.events:
            raise ValueError("EventSequence.events must contain at least one event")
        if any(not isinstance(value, str) or not value.strip() for value in self.events):
            raise ValueError("EventSequence.events must be nonblank strings")
        if len(set(self.events)) != len(self.events):
            raise ValueError("EventSequence.events must not repeat an event")
        if any(not isinstance(value, str) or not value.strip() for value in self.expansions):
            raise ValueError("EventSequence.expansions must be nonblank strings")

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "events": list(self.events),
            "expansions": list(self.expansions),
        }


__all__ = ["EventSequence"]
