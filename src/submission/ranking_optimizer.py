from __future__ import annotations

import numbers
from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass(frozen=True)
class RankingTier:
    """[start_rank, end_rank] (1-based, inclusive) at a fixed MMR diversity
    weight: 0.0 = pure relevance, 1.0 = pure diversity."""

    start_rank: int
    end_rank: int
    diversity_weight: float

    def __post_init__(self) -> None:
        for name in ("start_rank", "end_rank"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.end_rank < self.start_rank:
            raise ValueError("end_rank must be >= start_rank")
        if isinstance(self.diversity_weight, bool) or not isinstance(
            self.diversity_weight, numbers.Real
        ):
            raise TypeError("diversity_weight must be numeric")
        if not 0 <= self.diversity_weight <= 1:
            raise ValueError("diversity_weight must be between 0 and 1")


# Rank 1-5 = pure confidence (Final Score is dominated by hitting Top-1/Top-5,
# see PHASE12.md); diversity ramps up only toward the tail, matching the
# "diversity chỉ nên mạnh ở cuối danh sách" guidance.
DEFAULT_TIERS: tuple[RankingTier, ...] = (
    RankingTier(1, 5, 0.0),
    RankingTier(6, 20, 0.15),
    RankingTier(21, 50, 0.4),
    RankingTier(51, 100, 0.7),
)


@dataclass(frozen=True)
class RankableHit:
    """Minimal shape needed to rerank — matches the fields shared by
    `retrieval.hybrid_schemas.HybridHit` and `llm.expansion_schemas.ExpandedHit`."""

    video_id: str
    frame_id: int
    timestamp: float
    score: float
    extra: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.video_id.strip():
            raise ValueError("RankableHit.video_id must not be blank")
        if isinstance(self.score, bool) or not isinstance(self.score, numbers.Real):
            raise TypeError("RankableHit.score must be numeric")


def _tier_for_rank(rank: int, tiers: Sequence[RankingTier]) -> RankingTier:
    for tier in tiers:
        if tier.start_rank <= rank <= tier.end_rank:
            return tier
    return tiers[-1]


def _similarity(a: RankableHit, b: RankableHit, *, near_duplicate_seconds: float) -> float:
    if a.video_id != b.video_id:
        return 0.0
    gap = abs(a.timestamp - b.timestamp)
    if near_duplicate_seconds <= 0:
        return 1.0 if gap == 0 else 0.0
    return max(0.0, 1.0 - gap / near_duplicate_seconds)


def rerank_with_tiered_diversity(
    candidates: Sequence[RankableHit],
    *,
    tiers: Sequence[RankingTier] = DEFAULT_TIERS,
    near_duplicate_seconds: float = 5.0,
) -> list[RankableHit]:
    """Greedy MMR-style reordering: relevance-only at the top ranks, more
    diversity-weighted toward the tail, per `tiers`.

    Never reaches for diversity if it would cost the leading ranks: rank 1
    is always the single best-scored candidate (tier 1's weight is 0.0), and
    every earlier tier is filled before a later, more diverse tier is
    consulted, since selection proceeds rank-by-rank in order.
    """

    if not candidates:
        return []
    if near_duplicate_seconds < 0:
        raise ValueError("near_duplicate_seconds must be non-negative")

    pool = list(candidates)
    scores = [float(item.score) for item in pool]
    low, high = min(scores), max(scores)
    span = high - low
    normalized = {
        id(item): (1.0 if span <= 0 else (score - low) / span) for item, score in zip(pool, scores)
    }

    remaining = list(pool)
    selected: list[RankableHit] = []
    for rank in range(1, len(pool) + 1):
        tier = _tier_for_rank(rank, tiers)
        best_item: RankableHit | None = None
        best_mmr = float("-inf")
        for item in remaining:
            relevance = normalized[id(item)]
            max_similarity = (
                max(
                    _similarity(item, other, near_duplicate_seconds=near_duplicate_seconds)
                    for other in selected
                )
                if selected
                else 0.0
            )
            mmr = (1.0 - tier.diversity_weight) * relevance - tier.diversity_weight * max_similarity
            if mmr > best_mmr:
                best_mmr = mmr
                best_item = item
        assert best_item is not None
        selected.append(best_item)
        remaining.remove(best_item)
    return selected


__all__ = ["DEFAULT_TIERS", "RankableHit", "RankingTier", "rerank_with_tiered_diversity"]
