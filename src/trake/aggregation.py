from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Sequence

from retrieval.hybrid_store import MetadataVideoHit
from retrieval.hybrid_search import HybridSearchResult
from retrieval.hybrid_schemas import HybridHit

from .video_retrieval_schemas import VideoCandidateScore


@dataclass(frozen=True)
class VideoRetrievalConfig:
    """Funnel width and per-video-score-component weights.

    Weights are an intentionally conservative baseline (equal, 1.0 each),
    matching Stage 3/4's stance: tune against ground truth, not by eyeballing
    a few queries.
    """

    per_query_top_k: int = 100
    video_limit: int = 5
    metadata_video_limit: int = 50
    global_similarity_weight: float = 1.0
    event_coverage_weight: float = 1.0
    bm25_weight: float = 1.0
    vote_weight: float = 1.0

    def __post_init__(self) -> None:
        for name in ("per_query_top_k", "video_limit", "metadata_video_limit"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 1 <= self.per_query_top_k <= 100:
            raise ValueError("per_query_top_k must be between 1 and 100 for BTC KIS")
        for name in (
            "global_similarity_weight",
            "event_coverage_weight",
            "bm25_weight",
            "vote_weight",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError(f"{name} must be numeric")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if not any(
            weight > 0
            for weight in (
                self.global_similarity_weight,
                self.event_coverage_weight,
                self.bm25_weight,
                self.vote_weight,
            )
        ):
            raise ValueError("At least one score component weight must be positive")


def _collapse_to_best_per_video(hits: Sequence[HybridHit]) -> dict[str, HybridHit]:
    best: dict[str, HybridHit] = {}
    for hit in hits:
        best.setdefault(hit.video_id, hit)
    return best


def aggregate_video_candidates(
    *,
    full_result: HybridSearchResult,
    event_results: Sequence[tuple[str, HybridSearchResult]],
    expansion_results: Sequence[tuple[str, HybridSearchResult]],
    metadata_hits: Sequence[MetadataVideoHit],
    config: VideoRetrievalConfig,
) -> list[VideoCandidateScore]:
    """video_score = global_similarity + event_coverage + BM25 + multi_query_vote.

    ``full_result`` is the full-query search (its per-video best hit gives
    ``global_similarity`` and the anchor ``best_frame_id``). ``event_results``
    are (event_text, search_result) pairs whose video coverage drives
    ``event_coverage``. ``expansion_results`` are additional LLM paraphrase
    searches that only feed ``multi_query_vote``. ``metadata_hits`` are raw
    BM25 hits for the full query text, independent of the semantic branch.
    """

    full_by_video = _collapse_to_best_per_video(full_result.hits)
    event_video_maps = [(text, _collapse_to_best_per_video(result.hits)) for text, result in event_results]
    expansion_video_maps = [
        (text, _collapse_to_best_per_video(result.hits)) for text, result in expansion_results
    ]

    all_subquery_video_maps = (
        [full_by_video] + [video_map for _, video_map in event_video_maps] + [
            video_map for _, video_map in expansion_video_maps
        ]
    )
    total_subqueries = len(all_subquery_video_maps)
    total_events = len(event_video_maps)

    bm25_raw = {hit.video_id: hit.bm25_score for hit in metadata_hits}
    # SQLite FTS5 bm25(): lower is more relevant. Normalize (min-max, inverted).
    bm25_low = min(bm25_raw.values()) if bm25_raw else 0.0
    bm25_high = max(bm25_raw.values()) if bm25_raw else 0.0
    bm25_span = bm25_high - bm25_low

    candidate_video_ids: set[str] = set(bm25_raw)
    for video_map in all_subquery_video_maps:
        candidate_video_ids.update(video_map)

    scored: list[VideoCandidateScore] = []
    for video_id in candidate_video_ids:
        full_hit = full_by_video.get(video_id)
        global_similarity = 0.0
        if full_hit is not None and full_hit.semantic_score is not None:
            global_similarity = min(1.0, max(0.0, (float(full_hit.semantic_score) + 1.0) / 2.0))

        matched_events = tuple(
            text for text, video_map in event_video_maps if video_id in video_map
        )
        event_coverage = len(matched_events) / total_events if total_events else 0.0

        if video_id in bm25_raw:
            bm25_score = 1.0 if bm25_span <= 0 else 1.0 - (bm25_raw[video_id] - bm25_low) / bm25_span
        else:
            bm25_score = 0.0

        votes = sum(1 for video_map in all_subquery_video_maps if video_id in video_map)
        multi_query_vote = votes / total_subqueries if total_subqueries else 0.0

        total_score = (
            config.global_similarity_weight * global_similarity
            + config.event_coverage_weight * event_coverage
            + config.bm25_weight * bm25_score
            + config.vote_weight * multi_query_vote
        )
        scored.append(
            VideoCandidateScore(
                video_id=video_id,
                rank=1,  # placeholder, replaced below after sorting
                total_score=total_score,
                global_similarity=global_similarity,
                event_coverage=event_coverage,
                bm25_score=bm25_score,
                multi_query_vote=multi_query_vote,
                matched_events=matched_events,
                best_frame_id=full_hit.frame_id if full_hit else None,
            )
        )

    scored.sort(key=lambda item: (-item.total_score, item.video_id))
    return [
        VideoCandidateScore(
            video_id=item.video_id,
            rank=rank,
            total_score=item.total_score,
            global_similarity=item.global_similarity,
            event_coverage=item.event_coverage,
            bm25_score=item.bm25_score,
            multi_query_vote=item.multi_query_vote,
            matched_events=item.matched_events,
            best_frame_id=item.best_frame_id,
        )
        for rank, item in enumerate(scored, start=1)
    ]


__all__ = ["VideoRetrievalConfig", "aggregate_video_candidates"]
