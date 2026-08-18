from __future__ import annotations

import numbers
from collections.abc import Mapping, Sequence

from retrieval.hybrid_schemas import HybridHit

from .expansion_schemas import ExpansionEvidence, FusedExpansion


def _validate_expansion_ranking(
    expansion_id: str, hits: Sequence[HybridHit]
) -> tuple[HybridHit, ...]:
    values = tuple(hits)
    if any(not isinstance(hit, HybridHit) for hit in values):
        raise TypeError(f"Expansion {expansion_id!r} ranking must contain HybridHit instances")
    expected = list(range(1, len(values) + 1))
    actual = [int(hit.rank) for hit in values]
    if actual != expected:
        raise ValueError(
            f"Expansion {expansion_id!r} must have dense ranks from 1; got {actual}"
        )
    keys = [(hit.video_id, hit.frame_id) for hit in values]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Expansion {expansion_id!r} contains duplicate submit pairs")
    return values


def fuse_expansions(
    rankings: Mapping[str, Sequence[HybridHit]],
    expansion_texts: Mapping[str, str],
    *,
    rrf_k: int = 60,
    limit: int = 100,
) -> list[FusedExpansion]:
    """Reciprocal-rank-fuse per-expansion hybrid rankings into one list."""

    if not rankings:
        raise ValueError("fuse_expansions requires at least one expansion ranking")
    if set(rankings) != set(expansion_texts):
        raise ValueError("rankings and expansion_texts must share the same expansion ids")
    if isinstance(limit, bool) or not isinstance(limit, numbers.Integral) or limit < 1:
        raise ValueError("fuse_expansions limit must be a positive integer")
    if isinstance(rrf_k, bool) or not isinstance(rrf_k, numbers.Integral) or rrf_k < 1:
        raise ValueError("fuse_expansions rrf_k must be a positive integer")

    validated = {
        expansion_id: _validate_expansion_ranking(expansion_id, hits)
        for expansion_id, hits in rankings.items()
    }

    accumulators: dict[tuple[str, int], dict[str, object]] = {}
    for expansion_id, hits in validated.items():
        text = expansion_texts[expansion_id]
        for hit in hits:
            key = (hit.video_id, hit.frame_id)
            contribution = 1.0 / (int(rrf_k) + int(hit.rank))
            current = accumulators.setdefault(key, {"score": 0.0, "evidence": []})
            current["score"] = float(current["score"]) + contribution
            current["evidence"].append(
                ExpansionEvidence(
                    expansion_id=expansion_id,
                    expansion_text=text,
                    rank=int(hit.rank),
                    raw_score=float(hit.score),
                    hit=hit,
                )
            )

    fused: list[FusedExpansion] = []
    for current in accumulators.values():
        evidence = tuple(
            sorted(current["evidence"], key=lambda item: (item.rank, item.expansion_id))
        )
        fused.append(
            FusedExpansion(score=float(current["score"]), hit=evidence[0].hit, evidence=evidence)
        )

    fused.sort(
        key=lambda item: (
            -item.score,
            item.evidence[0].rank,
            item.hit.video_id,
            item.hit.frame_id,
        )
    )
    return fused[: int(limit)]


__all__ = ["fuse_expansions"]
