from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence

from .hybrid_schemas import BRANCH_NAMES, BranchEvidence, BranchHit, FusedFrame


def _validated_weights(weights: Mapping[str, float] | None) -> dict[str, float]:
    resolved = {name: 1.0 for name in BRANCH_NAMES}
    if weights is not None:
        unknown = set(weights) - set(BRANCH_NAMES)
        if unknown:
            raise ValueError(f"Unknown fusion weight(s): {', '.join(sorted(unknown))}")
        resolved.update(weights)
    for name, value in resolved.items():
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"Fusion weight for {name} must be numeric")
        resolved[name] = float(value)
        if not math.isfinite(resolved[name]) or resolved[name] < 0:
            raise ValueError(f"Fusion weight for {name} must be finite and non-negative")
    if not any(value > 0 for value in resolved.values()):
        raise ValueError("At least one fusion weight must be positive")
    return resolved


def _validate_rankings(
    rankings: Mapping[str, Sequence[BranchHit]],
) -> dict[str, tuple[BranchHit, ...]]:
    unknown = set(rankings) - set(BRANCH_NAMES)
    if unknown:
        raise ValueError(f"Unknown ranking branch(es): {', '.join(sorted(unknown))}")
    validated: dict[str, tuple[BranchHit, ...]] = {}
    for branch in BRANCH_NAMES:
        values = tuple(rankings.get(branch, ()))
        if any(not isinstance(hit, BranchHit) for hit in values):
            raise TypeError(f"Ranking {branch!r} must contain BranchHit instances")
        expected = list(range(1, len(values) + 1))
        actual = [int(hit.rank) for hit in values]
        if any(hit.branch != branch for hit in values):
            raise ValueError(f"Ranking {branch!r} contains a hit from another branch")
        if actual != expected:
            raise ValueError(
                f"Ranking {branch!r} must have dense ranks from 1; got {actual}"
            )
        keys = [hit.submit_key for hit in values]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Ranking {branch!r} contains duplicate submit pairs")
        validated[branch] = values
    return validated


def _weighted_normalized_scores(
    hits: Sequence[BranchHit],
) -> dict[tuple[str, int], float]:
    if not hits:
        return {}
    values = [float(hit.raw_score) for hit in hits]
    low = min(values)
    high = max(values)
    if math.isclose(low, high, rel_tol=1e-12, abs_tol=1e-12):
        return {hit.submit_key: 1.0 for hit in hits}
    scale = high - low
    return {hit.submit_key: (float(hit.raw_score) - low) / scale for hit in hits}


def fuse_rankings(
    rankings: Mapping[str, Sequence[BranchHit]],
    *,
    method: str = "rrf",
    weights: Mapping[str, float] | None = None,
    rrf_k: int = 60,
    limit: int = 100,
) -> list[FusedFrame]:
    """Fuse frame rankings by submit key with deterministic tie-breaking."""

    method = method.casefold().strip()
    if method not in {"rrf", "weighted"}:
        raise ValueError("fusion method must be 'rrf' or 'weighted'")
    if isinstance(limit, bool) or not isinstance(limit, numbers.Integral) or limit < 1:
        raise ValueError("fusion limit must be a positive integer")
    if isinstance(rrf_k, bool) or not isinstance(rrf_k, numbers.Integral) or rrf_k < 1:
        raise ValueError("rrf_k must be a positive integer")

    branch_weights = _validated_weights(weights)
    branch_hits = _validate_rankings(rankings)
    weighted_denominator = sum(branch_weights.values())
    normalized = {
        branch: _weighted_normalized_scores(hits)
        for branch, hits in branch_hits.items()
    }

    accumulators: dict[tuple[str, int], dict[str, object]] = {}
    for branch in BRANCH_NAMES:
        weight = branch_weights[branch]
        if weight == 0:
            continue
        for hit in branch_hits[branch]:
            key = hit.submit_key
            if method == "rrf":
                contribution = weight / (int(rrf_k) + int(hit.rank))
            else:
                contribution = (
                    weight * normalized[branch][key] / weighted_denominator
                )
            current = accumulators.setdefault(
                key,
                {"score": 0.0, "hits": {}, "frames": []},
            )
            current["score"] = float(current["score"]) + contribution
            current["hits"][branch] = BranchEvidence(
                rank=int(hit.rank),
                raw_score=float(hit.raw_score),
                contribution=float(contribution),
                details=hit.details,
            )
            current["frames"].append((branch, hit.frame))

    fused: list[FusedFrame] = []
    for current in accumulators.values():
        frames = current["frames"]
        # Prefer the semantic representative for diagnostics, then use the
        # lowest FAISS id so duplicate source frame ids stay deterministic.
        frame = min(
            frames,
            key=lambda item: (
                0 if item[0] == "semantic" else 1,
                item[1].faiss_index,
            ),
        )[1]
        fused.append(
            FusedFrame(
                score=float(current["score"]),
                frame=frame,
                evidence=dict(current["hits"]),
            )
        )

    missing_rank = 10**12
    fused.sort(
        key=lambda item: (
            -item.score,
            item.evidence.get("semantic", BranchEvidence(missing_rank, 0, 0)).rank,
            item.evidence.get("metadata", BranchEvidence(missing_rank, 0, 0)).rank,
            item.evidence.get("objects", BranchEvidence(missing_rank, 0, 0)).rank,
            item.frame.video_id,
            item.frame.video_frame_id,
            item.frame.faiss_index,
        )
    )
    return fused[: int(limit)]


__all__ = ["fuse_rankings"]
