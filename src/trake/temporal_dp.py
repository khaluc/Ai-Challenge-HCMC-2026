from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from retrieval.clip_store import FrameRecord
from retrieval.clip_search import TextEncoderProtocol
from retrieval.hybrid_store import Phase1HybridStore

from .temporal_dp_schemas import CoarseAlignment, EventFrameAssignment


@dataclass(frozen=True)
class AlignmentConfig:
    """DP alignment knobs. Transition penalty is optional (spec: "không bắt
    buộc"); off by default (`transition_penalty_weight=0`)."""

    transition_penalty_weight: float = 0.0
    min_frame_gap: int = 1
    max_frames_per_video: int = 5000

    def __post_init__(self) -> None:
        if isinstance(self.transition_penalty_weight, bool) or not isinstance(
            self.transition_penalty_weight, numbers.Real
        ):
            raise TypeError("transition_penalty_weight must be numeric")
        if self.transition_penalty_weight < 0:
            raise ValueError("transition_penalty_weight must be non-negative")
        for name in ("min_frame_gap", "max_frames_per_video"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


def align_events(
    event_texts: Sequence[str],
    frames: Sequence[FrameRecord],
    similarity: Sequence[Sequence[float]],
    *,
    video_id: str,
    config: AlignmentConfig | None = None,
) -> CoarseAlignment:
    """Dynamic-programming baseline for maximize sum S(Ei, fi) s.t. f1<...<fn.

    ``frames`` must be sorted ascending by ``keyframe_index`` and
    ``similarity[i][j]`` is S(event i, frames[j]). O(N*M^2); fine for the
    typical per-video keyframe counts here (tens to a few hundred) — DTW is
    explicitly not required by the spec, so this stays a simple baseline
    rather than a more elaborate/faster formulation.
    """

    config = config or AlignmentConfig()
    n = len(event_texts)
    m = len(frames)
    if n == 0:
        raise ValueError("event_texts must not be empty")
    if len(similarity) != n:
        raise ValueError("similarity must have exactly one row per event")
    if any(len(row) != m for row in similarity):
        raise ValueError("each similarity row must have length == len(frames)")
    if not video_id.strip():
        raise ValueError("video_id must not be blank")

    if m < n:
        return CoarseAlignment(
            video_id=video_id, events=tuple(event_texts), assignments=(), total_score=0.0, feasible=False
        )

    neg_inf = float("-inf")
    dp: list[list[float]] = [[neg_inf] * m for _ in range(n)]
    parent: list[list[int]] = [[-1] * m for _ in range(n)]

    for j in range(m):
        dp[0][j] = float(similarity[0][j])

    for i in range(1, n):
        for j in range(m):
            best_score = neg_inf
            best_prev = -1
            for j_prev in range(j):
                prev = dp[i - 1][j_prev]
                if prev == neg_inf:
                    continue
                if config.transition_penalty_weight > 0:
                    gap = j - j_prev
                    prev -= config.transition_penalty_weight * max(0, config.min_frame_gap - gap)
                if prev > best_score:
                    best_score = prev
                    best_prev = j_prev
            if best_prev == -1:
                continue
            dp[i][j] = best_score + float(similarity[i][j])
            parent[i][j] = best_prev

    best_end = -1
    best_total = neg_inf
    for j in range(m):
        if dp[n - 1][j] > best_total:
            best_total = dp[n - 1][j]
            best_end = j
    if best_end == -1:
        return CoarseAlignment(
            video_id=video_id, events=tuple(event_texts), assignments=(), total_score=0.0, feasible=False
        )

    path = [best_end]
    i, j = n - 1, best_end
    while i > 0:
        j = parent[i][j]
        i -= 1
        path.append(j)
    path.reverse()

    assignments = tuple(
        EventFrameAssignment(
            event_index=i,
            event_text=event_texts[i],
            frame=frames[path[i]],
            similarity=float(similarity[i][path[i]]),
        )
        for i in range(n)
    )
    return CoarseAlignment(
        video_id=video_id,
        events=tuple(event_texts),
        assignments=assignments,
        total_score=best_total,
        feasible=True,
    )


class CoarseTemporalAligner:
    """Real CLIP text-image similarity (Stage 1's precomputed keyframe
    vectors) fed into `align_events`."""

    def __init__(
        self,
        store: Phase1HybridStore,
        encoder: TextEncoderProtocol,
        *,
        config: AlignmentConfig | None = None,
    ) -> None:
        self.store = store
        self.encoder = encoder
        self.config = config or AlignmentConfig()

    def align(self, video_id: str, event_texts: Sequence[str]) -> CoarseAlignment:
        if not video_id.strip():
            raise ValueError("video_id must not be blank")
        events = list(event_texts)
        if not events:
            raise ValueError("event_texts must not be empty")

        vectors = np.asarray(self.encoder.encode(events))
        per_event: list[dict[int, tuple[FrameRecord, float]]] = []
        for vector in vectors:
            results = self.store.best_frames_in_video(
                video_id, vector, limit=self.config.max_frames_per_video
            )
            per_event.append({frame.keyframe_index: (frame, score) for frame, score in results})

        if not per_event or not per_event[0]:
            return CoarseAlignment(
                video_id=video_id, events=tuple(events), assignments=(), total_score=0.0, feasible=False
            )

        ordered = sorted(per_event[0].values(), key=lambda item: item[0].keyframe_index)
        frames = [item[0] for item in ordered]
        similarity = [
            [per_event[i][frame.keyframe_index][1] for frame in frames] for i in range(len(events))
        ]
        return align_events(events, frames, similarity, video_id=video_id, config=self.config)


__all__ = ["AlignmentConfig", "CoarseTemporalAligner", "align_events"]
