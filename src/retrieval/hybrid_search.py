from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

import numpy as np

from retrieval.clip_search import TextEncoderProtocol
from retrieval.schemas import Query

from .hybrid_store import MetadataVideoHit, Phase1HybridStore
from .fusion import fuse_rankings
from .processing import ParsedObjectQuery, RuleBasedObjectParser, tokenize_for_metadata
from .hybrid_schemas import BranchHit, HybridHit, QueryAnalysis


class ObjectParserProtocol(Protocol):
    def parse(self, text: str) -> ParsedObjectQuery:
        """Return detector-backed concepts, or an empty result to abstain."""


@dataclass(frozen=True)
class HybridConfig:
    semantic_candidates: int = 500
    metadata_video_candidates: int = 30
    metadata_frames_per_video: int = 3
    object_candidates: int = 300
    object_min_confidence: float = 0.30
    fusion_method: str = "rrf"
    rrf_k: int = 60
    semantic_weight: float = 1.0
    metadata_weight: float = 0.65
    object_weight: float = 0.80
    metadata_enabled: bool = True
    objects_enabled: bool = True

    def __post_init__(self) -> None:
        for name in (
            "semantic_candidates",
            "metadata_video_candidates",
            "metadata_frames_per_video",
            "object_candidates",
            "rrf_k",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.fusion_method not in {"rrf", "weighted"}:
            raise ValueError("fusion_method must be 'rrf' or 'weighted'")
        if (
            isinstance(self.object_min_confidence, bool)
            or not isinstance(self.object_min_confidence, numbers.Real)
            or not 0 <= self.object_min_confidence <= 1
        ):
            raise ValueError("object_min_confidence must be between 0 and 1")
        if not isinstance(self.metadata_enabled, bool) or not isinstance(
            self.objects_enabled, bool
        ):
            raise TypeError("metadata_enabled and objects_enabled must be boolean")
        for name in ("semantic_weight", "metadata_weight", "object_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.semantic_weight <= 0:
            raise ValueError("semantic_weight must be positive for the CLIP baseline branch")

    @property
    def weights(self) -> dict[str, float]:
        return {
            "semantic": float(self.semantic_weight),
            "metadata": float(self.metadata_weight) if self.metadata_enabled else 0.0,
            "objects": float(self.object_weight) if self.objects_enabled else 0.0,
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "semantic_candidates": self.semantic_candidates,
            "metadata_video_candidates": self.metadata_video_candidates,
            "metadata_frames_per_video": self.metadata_frames_per_video,
            "object_candidates": self.object_candidates,
            "object_min_confidence": self.object_min_confidence,
            "fusion_method": self.fusion_method,
            "rrf_k": self.rrf_k,
            "weights": self.weights,
            "metadata_enabled": self.metadata_enabled,
            "objects_enabled": self.objects_enabled,
        }


@dataclass(frozen=True)
class HybridSearchResult:
    query: Query
    analysis: QueryAnalysis
    hits: tuple[HybridHit, ...]
    branch_counts: Mapping[str, int] = field(default_factory=dict)

    def as_summary(self) -> dict[str, object]:
        return {
            "query_id": self.query.query_id,
            "analysis": self.analysis.as_dict(),
            "branch_counts": dict(self.branch_counts),
            "hits": len(self.hits),
        }


class HybridTextualKIS:
    """CLIP + metadata BM25 + object co-occurrence retrieval with late fusion."""

    def __init__(
        self,
        store: Phase1HybridStore,
        encoder: TextEncoderProtocol,
        *,
        object_parser: ObjectParserProtocol | None = None,
        config: HybridConfig | None = None,
    ) -> None:
        self.store = store
        self.encoder = encoder
        self.object_parser = object_parser or RuleBasedObjectParser(store.object_labels)
        self.config = config or HybridConfig()
        if (
            self.config.objects_enabled
            and self.config.object_weight > 0
            and self.config.object_min_confidence < store.object_index_min_confidence
        ):
            raise ValueError(
                "Hybrid object threshold is below the Phase 1 index floor: "
                f"threshold={self.config.object_min_confidence}, "
                f"floor={store.object_index_min_confidence}"
            )

    def search(
        self,
        query: Query | str,
        *,
        top_k: int = 100,
        query_id: str | None = None,
    ) -> list[HybridHit]:
        if isinstance(query, str):
            value = Query(query_id=query_id or "query", text=query)
        elif isinstance(query, Query):
            if query_id is not None and query_id != query.query_id:
                raise ValueError("query_id cannot override an existing Query")
            value = query
        else:
            raise TypeError("query must be a Query or string")
        return list(self.search_detailed_batch([value], top_k=top_k)[0].hits)

    def search_batch(
        self,
        queries: Sequence[Query],
        *,
        top_k: int = 100,
    ) -> list[HybridHit]:
        return [
            hit
            for result in self.search_detailed_batch(queries, top_k=top_k)
            for hit in result.hits
        ]

    def search_detailed_batch(
        self,
        queries: Sequence[Query],
        *,
        top_k: int = 100,
    ) -> list[HybridSearchResult]:
        if isinstance(top_k, bool) or not isinstance(top_k, numbers.Integral):
            raise TypeError("top_k must be an integer")
        if not 1 <= int(top_k) <= 100:
            raise ValueError("top_k must be between 1 and 100 for BTC KIS")
        top_k = int(top_k)
        query_list = list(queries)
        seen_query_ids: set[str] = set()
        for position, query in enumerate(query_list):
            if not isinstance(query, Query):
                raise TypeError(f"queries[{position}] must be a Query")
            if query.query_id in seen_query_ids:
                raise ValueError(f"duplicate query_id in batch: {query.query_id!r}")
            seen_query_ids.add(query.query_id)
        if not query_list:
            return []

        vectors = np.asarray(self.encoder.encode([query.text for query in query_list]))
        expected_shape = (len(query_list), self.store.clip_store.dimension)
        if vectors.shape != expected_shape:
            raise ValueError(
                f"Text encoder output shape mismatch: expected {expected_shape}, got {vectors.shape}"
            )
        if not np.issubdtype(vectors.dtype, np.number) or not np.all(np.isfinite(vectors)):
            raise ValueError("Text encoder returned non-finite or non-numeric vectors")

        return [
            self._search_encoded(query, vector, top_k=top_k)
            for query, vector in zip(query_list, vectors)
        ]

    def _search_encoded(
        self,
        query: Query,
        vector: np.ndarray,
        *,
        top_k: int,
    ) -> HybridSearchResult:
        metadata_terms = tokenize_for_metadata(query.text)
        parsed_objects = self.object_parser.parse(query.text)
        analysis = QueryAnalysis(
            text=query.text,
            metadata_terms=metadata_terms,
            object_concepts=parsed_objects.concepts,
        )

        semantic_limit = max(top_k, self.config.semantic_candidates)
        semantic_hits = [
            BranchHit(
                branch="semantic",
                rank=rank,
                raw_score=score,
                frame=frame,
                details={"cosine_similarity": score},
            )
            for rank, (frame, score) in enumerate(
                self.store.semantic_hits(vector, semantic_limit), start=1
            )
        ]

        metadata_hits: list[BranchHit] = []
        if (
            self.config.metadata_enabled
            and self.config.metadata_weight > 0
            and metadata_terms
        ):
            video_hits = self.store.search_metadata(
                metadata_terms,
                limit=self.config.metadata_video_candidates,
            )
            metadata_hits = self._metadata_frame_hits(video_hits, vector)

        object_hits: list[BranchHit] = []
        if (
            self.config.objects_enabled
            and self.config.object_weight > 0
            and parsed_objects.concepts
        ):
            detected = self.store.search_objects(
                parsed_objects,
                min_confidence=self.config.object_min_confidence,
                limit=self.config.object_candidates,
            )
            object_hits = [
                BranchHit(
                    branch="objects",
                    rank=hit.rank,
                    raw_score=hit.min_confidence,
                    frame=hit.frame,
                    details={
                        "matched_objects": list(hit.matched_concepts),
                        "min_confidence": hit.min_confidence,
                        "mean_confidence": hit.mean_confidence,
                    },
                )
                for hit in detected
            ]

        rankings = {
            "semantic": semantic_hits,
            "metadata": metadata_hits,
            "objects": object_hits,
        }
        fused = fuse_rankings(
            rankings,
            method=self.config.fusion_method,
            weights=self.config.weights,
            rrf_k=self.config.rrf_k,
            limit=top_k,
        )
        hybrid_hits: list[HybridHit] = []
        for rank, item in enumerate(fused, start=1):
            frame = item.frame
            semantic = item.evidence.get("semantic")
            metadata = item.evidence.get("metadata")
            objects = item.evidence.get("objects")
            matched_objects = ()
            if objects is not None:
                matched_objects = tuple(objects.details.get("matched_objects", ()))
            hybrid_hits.append(
                HybridHit(
                    query_id=query.query_id,
                    rank=rank,
                    video_id=frame.video_id,
                    frame_id=frame.video_frame_id,
                    score=item.score,
                    faiss_index=frame.faiss_index,
                    keyframe_index=frame.keyframe_index,
                    timestamp=frame.timestamp,
                    keyframe_path=frame.keyframe_path,
                    keyframe_available=frame.keyframe_available,
                    fusion_method=self.config.fusion_method,
                    semantic_rank=semantic.rank if semantic else None,
                    semantic_score=semantic.raw_score if semantic else None,
                    metadata_rank=metadata.rank if metadata else None,
                    metadata_score=metadata.raw_score if metadata else None,
                    metadata_video_rank=(
                        int(metadata.details["metadata_video_rank"])
                        if metadata
                        else None
                    ),
                    metadata_bm25_score=(
                        float(metadata.details["bm25_score"]) if metadata else None
                    ),
                    metadata_match_mode=(
                        str(metadata.details["match_mode"]) if metadata else None
                    ),
                    object_rank=objects.rank if objects else None,
                    object_score=objects.raw_score if objects else None,
                    object_mean_confidence=(
                        float(objects.details["mean_confidence"]) if objects else None
                    ),
                    matched_objects=matched_objects,
                )
            )
        return HybridSearchResult(
            query=query,
            analysis=analysis,
            hits=tuple(hybrid_hits),
            branch_counts={branch: len(values) for branch, values in rankings.items()},
        )

    def _metadata_frame_hits(
        self,
        videos: Sequence[MetadataVideoHit],
        vector: np.ndarray,
    ) -> list[BranchHit]:
        candidates: list[tuple[float, int, int, object, float, MetadataVideoHit]] = []
        for video in videos:
            frames = self.store.best_frames_in_video(
                video.video_id,
                vector,
                limit=self.config.metadata_frames_per_video,
            )
            for within_rank, (frame, clip_score) in enumerate(frames, start=1):
                clip_unit = min(1.0, max(0.0, (clip_score + 1.0) / 2.0))
                raw_score = (
                    (1.0 / video.rank)
                    * (0.75 + 0.25 * clip_unit)
                    / math.sqrt(within_rank)
                )
                candidates.append(
                    (raw_score, video.rank, within_rank, frame, clip_score, video)
                )
        candidates.sort(
            key=lambda item: (
                -item[0],
                item[1],
                item[2],
                item[3].faiss_index,
            )
        )
        return [
            BranchHit(
                branch="metadata",
                rank=rank,
                raw_score=raw_score,
                frame=frame,
                details={
                    "metadata_video_rank": video.rank,
                    "bm25_score": video.bm25_score,
                    "match_mode": video.match_mode,
                    "within_video_clip_score": clip_score,
                    "title": video.title,
                    "author": video.author,
                    "publish_date": video.publish_date,
                    "watch_url": video.watch_url,
                },
            )
            for rank, (raw_score, _, _, frame, clip_score, video) in enumerate(
                candidates, start=1
            )
        ]


__all__ = [
    "HybridConfig",
    "HybridSearchResult",
    "HybridTextualKIS",
    "ObjectParserProtocol",
]
