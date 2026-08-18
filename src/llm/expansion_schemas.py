from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from typing import Any, Mapping

from retrieval.schemas import RetrievalHit
from retrieval.hybrid_schemas import HybridHit


@dataclass(frozen=True)
class QueryStructure:
    """LLM- or rule-derived object/attribute/relation breakdown of a query."""

    objects: tuple[str, ...] = ()
    attributes: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    relation: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.objects, tuple) or any(
            not isinstance(value, str) or not value for value in self.objects
        ):
            raise ValueError("QueryStructure.objects must be a tuple of nonblank strings")
        if len(set(self.objects)) != len(self.objects):
            raise ValueError("QueryStructure.objects must not contain duplicates")
        if not isinstance(self.attributes, Mapping):
            raise TypeError("QueryStructure.attributes must be a mapping")
        for key, values in self.attributes.items():
            if not isinstance(key, str) or not key:
                raise ValueError("QueryStructure.attributes keys must be nonblank strings")
            if not isinstance(values, tuple) or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise ValueError(
                    f"QueryStructure.attributes[{key!r}] must be a tuple of nonblank strings"
                )
        if self.relation is not None and not self.relation:
            raise ValueError("QueryStructure.relation must be nonblank or null")

    def as_dict(self) -> dict[str, Any]:
        return {
            "objects": list(self.objects),
            "attributes": {key: list(values) for key, values in self.attributes.items()},
            "relation": self.relation,
        }


@dataclass(frozen=True)
class QueryUnderstanding:
    """Structured breakdown plus the query expansions to search."""

    text: str
    structure: QueryStructure
    expansions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("QueryUnderstanding.text must not be blank")
        if not isinstance(self.structure, QueryStructure):
            raise TypeError("QueryUnderstanding.structure must be a QueryStructure")
        if not self.expansions:
            raise ValueError("QueryUnderstanding.expansions must contain at least one query")
        if any(not isinstance(value, str) or not value.strip() for value in self.expansions):
            raise ValueError("QueryUnderstanding.expansions must be nonblank strings")
        if len(set(self.expansions)) != len(self.expansions):
            raise ValueError("QueryUnderstanding.expansions must not repeat a query")

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "structure": self.structure.as_dict(),
            "expansions": list(self.expansions),
        }


@dataclass(frozen=True)
class ExpansionEvidence:
    """One expansion's contribution to a fused frame."""

    expansion_id: str
    expansion_text: str
    rank: int
    raw_score: float
    hit: HybridHit

    def __post_init__(self) -> None:
        if not self.expansion_id:
            raise ValueError("ExpansionEvidence.expansion_id must not be blank")
        if not self.expansion_text.strip():
            raise ValueError("ExpansionEvidence.expansion_text must not be blank")
        if isinstance(self.rank, bool) or not isinstance(self.rank, numbers.Integral):
            raise TypeError("ExpansionEvidence.rank must be an integer")
        if int(self.rank) < 1:
            raise ValueError("ExpansionEvidence.rank must be positive")
        if isinstance(self.raw_score, bool) or not isinstance(self.raw_score, numbers.Real):
            raise TypeError("ExpansionEvidence.raw_score must be numeric")
        if not math.isfinite(float(self.raw_score)):
            raise ValueError("ExpansionEvidence.raw_score must be finite")
        if not isinstance(self.hit, HybridHit):
            raise TypeError("ExpansionEvidence.hit must be a HybridHit")


@dataclass(frozen=True)
class FusedExpansion:
    """One frame fused across every expansion ranking that surfaced it."""

    score: float
    hit: HybridHit
    evidence: tuple[ExpansionEvidence, ...]

    def __post_init__(self) -> None:
        if isinstance(self.score, bool) or not isinstance(self.score, numbers.Real):
            raise TypeError("FusedExpansion.score must be numeric")
        if not math.isfinite(float(self.score)):
            raise ValueError("FusedExpansion.score must be finite")
        if not self.evidence:
            raise ValueError("FusedExpansion.evidence must not be empty")
        ids = [item.expansion_id for item in self.evidence]
        if len(set(ids)) != len(ids):
            raise ValueError("FusedExpansion.evidence must not repeat an expansion_id")


@dataclass(frozen=True)
class ExpandedHit:
    """Submission-ready row plus which query expansions contributed."""

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
    understanding: QueryUnderstanding
    contributing_expansions: tuple[ExpansionEvidence, ...]

    def __post_init__(self) -> None:
        self.to_retrieval_hit()
        if not isinstance(self.understanding, QueryUnderstanding):
            raise TypeError("ExpandedHit.understanding must be a QueryUnderstanding")
        if not self.contributing_expansions:
            raise ValueError("ExpandedHit must have at least one contributing expansion")
        ids = [evidence.expansion_id for evidence in self.contributing_expansions]
        if len(set(ids)) != len(ids):
            raise ValueError("ExpandedHit contributing expansions must have unique expansion_id")

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

    @property
    def best_expansion(self) -> ExpansionEvidence:
        return min(self.contributing_expansions, key=lambda evidence: evidence.rank)

    def as_dict(self) -> dict[str, Any]:
        best = self.best_expansion
        return {
            "query_id": self.query_id,
            "rank": self.rank,
            "video_id": self.video_id,
            "frame_id": self.frame_id,
            "score": self.score,
            "faiss_index": self.faiss_index,
            "keyframe_index": self.keyframe_index,
            "timestamp": self.timestamp,
            "keyframe_path": self.keyframe_path,
            "keyframe_available": self.keyframe_available,
            "num_expansions_matched": len(self.contributing_expansions),
            "best_expansion_id": best.expansion_id,
            "best_expansion_text": best.expansion_text,
            "best_expansion_rank": best.rank,
            "matched_objects": list(best.hit.matched_objects),
            "object_concepts": list(self.understanding.structure.objects),
            "attributes": self.understanding.structure.as_dict()["attributes"],
            "relation": self.understanding.structure.relation,
        }


__all__ = [
    "ExpandedHit",
    "ExpansionEvidence",
    "FusedExpansion",
    "QueryStructure",
    "QueryUnderstanding",
]
