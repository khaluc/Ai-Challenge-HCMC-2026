from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from retrieval.schemas import Query

from .schemas import QACandidate


class RetrievalHitLike(Protocol):
    video_id: str
    frame_id: int
    score: float
    rank: int
    faiss_index: int
    keyframe_index: int
    timestamp: float
    keyframe_path: str
    keyframe_available: bool


class HitSearcherProtocol(Protocol):
    def search(
        self, query: Query | str, *, top_k: int = 100, query_id: str | None = None
    ) -> Sequence[Any]:
        """Return a ranked sequence of hits with the RetrievalHitLike fields."""


@dataclass(frozen=True)
class CandidateConfig:
    """Funnel widths for narrowing candidates before the VLM ever runs."""

    search_top_k: int = 100
    video_limit: int = 15
    frame_limit: int = 30

    def __post_init__(self) -> None:
        for name in ("search_top_k", "video_limit", "frame_limit"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 1 <= self.search_top_k <= 100:
            raise ValueError("search_top_k must be between 1 and 100 for BTC KIS")


def narrow_candidates(
    hits: Sequence[Any], config: CandidateConfig
) -> list[QACandidate]:
    """Top `video_limit` distinct videos by best rank, then top `frame_limit`
    keyframes drawn only from those videos (already best-score-first)."""

    candidate_videos: list[str] = []
    for hit in hits:
        if hit.video_id not in candidate_videos:
            if len(candidate_videos) == config.video_limit:
                continue
            candidate_videos.append(hit.video_id)

    allowed = set(candidate_videos)
    candidates: list[QACandidate] = []
    for hit in hits:
        if hit.video_id not in allowed or not hit.keyframe_available:
            continue
        candidates.append(
            QACandidate(
                video_id=hit.video_id,
                frame_id=hit.frame_id,
                keyframe_path=hit.keyframe_path,
                keyframe_available=hit.keyframe_available,
                faiss_index=hit.faiss_index,
                keyframe_index=hit.keyframe_index,
                timestamp=hit.timestamp,
                retrieval_rank=hit.rank,
                retrieval_score=hit.score,
            )
        )
        if len(candidates) == config.frame_limit:
            break
    return candidates


class QACandidateSearch:
    """Question -> Top-N videos -> Top-M keyframes, via any KIS searcher.

    Works with either `retrieval.hybrid_search.HybridTextualKIS` or
    `llm.expansion_retrieval.ExpandedHybridSearch` (duck-typed: both expose the
    same `.search(query, top_k=..., query_id=...)` -> ranked hits shape).
    """

    def __init__(self, searcher: HitSearcherProtocol, *, config: CandidateConfig | None = None) -> None:
        self.searcher = searcher
        self.config = config or CandidateConfig()

    def find_candidates(
        self, question: str, *, query_id: str | None = None
    ) -> list[QACandidate]:
        hits = self.searcher.search(
            question, top_k=self.config.search_top_k, query_id=query_id or "qa"
        )
        return narrow_candidates(hits, self.config)


__all__ = ["CandidateConfig", "HitSearcherProtocol", "QACandidateSearch", "narrow_candidates"]
