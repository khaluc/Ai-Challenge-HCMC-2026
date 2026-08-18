from __future__ import annotations

import numbers
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from retrieval.schemas import Query
from retrieval.hybrid_search import HybridTextualKIS
from retrieval.hybrid_schemas import HybridHit

from .expansion_fusion import fuse_expansions
from .expansion_schemas import ExpandedHit, QueryUnderstanding
from .query_expansion import QueryUnderstandingProtocol


@dataclass(frozen=True)
class ExpansionConfig:
    max_expansions: int = 4
    candidates_per_expansion: int = 100
    rrf_k: int = 60

    def __post_init__(self) -> None:
        for name in ("max_expansions", "candidates_per_expansion", "rrf_k"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 1 <= self.candidates_per_expansion <= 100:
            raise ValueError("candidates_per_expansion must be between 1 and 100 for BTC KIS")


@dataclass(frozen=True)
class ExpandedSearchResult:
    query: Query
    understanding: QueryUnderstanding
    hits: tuple[ExpandedHit, ...]
    expansion_counts: Mapping[str, int] = field(default_factory=dict)

    def as_summary(self) -> dict[str, object]:
        return {
            "query_id": self.query.query_id,
            "understanding": self.understanding.as_dict(),
            "expansion_counts": dict(self.expansion_counts),
            "hits": len(self.hits),
        }


class ExpandedHybridSearch:
    """LLM/rule query understanding, per-expansion hybrid search, RRF fusion.

    Each query is expanded into up to ``config.max_expansions`` phrasings
    (the original text is always one of them). Every expansion is searched
    independently through the existing Phase 3 hybrid pipeline (CLIP +
    metadata BM25 + objects, already fused internally), and the resulting
    per-expansion rankings are fused again via RRF across expansions.
    """

    def __init__(
        self,
        hybrid: HybridTextualKIS,
        understanding: QueryUnderstandingProtocol,
        *,
        config: ExpansionConfig | None = None,
    ) -> None:
        self.hybrid = hybrid
        self.understanding = understanding
        self.config = config or ExpansionConfig()

    def search(
        self,
        query: Query | str,
        *,
        top_k: int = 100,
        query_id: str | None = None,
    ) -> list[ExpandedHit]:
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
    ) -> list[ExpandedHit]:
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
    ) -> list[ExpandedSearchResult]:
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

        understandings = [self.understanding.understand(query.text) for query in query_list]
        expansion_lists = [
            self._resolve_expansions(query, qu) for query, qu in zip(query_list, understandings)
        ]

        flat_queries: list[Query] = []
        for query, expansions in zip(query_list, expansion_lists):
            for expansion_index, expansion_text in enumerate(expansions):
                flat_queries.append(
                    Query(f"{query.query_id}::e{expansion_index}", expansion_text)
                )

        flat_hits = self.hybrid.search_batch(
            flat_queries, top_k=self.config.candidates_per_expansion
        )
        grouped_hits: dict[str, list[HybridHit]] = {flat.query_id: [] for flat in flat_queries}
        for hit in flat_hits:
            grouped_hits[hit.query_id].append(hit)

        results: list[ExpandedSearchResult] = []
        for query, expansions, understanding in zip(query_list, expansion_lists, understandings):
            rankings: dict[str, list[HybridHit]] = {}
            expansion_texts: dict[str, str] = {}
            for expansion_index, expansion_text in enumerate(expansions):
                expansion_id = f"e{expansion_index}"
                flat_query_id = f"{query.query_id}::{expansion_id}"
                rankings[expansion_id] = grouped_hits[flat_query_id]
                expansion_texts[expansion_id] = expansion_text

            fused = fuse_expansions(
                rankings,
                expansion_texts,
                rrf_k=self.config.rrf_k,
                limit=top_k,
            )
            hits = tuple(
                ExpandedHit(
                    query_id=query.query_id,
                    rank=rank,
                    video_id=item.hit.video_id,
                    frame_id=item.hit.frame_id,
                    score=item.score,
                    faiss_index=item.hit.faiss_index,
                    keyframe_index=item.hit.keyframe_index,
                    timestamp=item.hit.timestamp,
                    keyframe_path=item.hit.keyframe_path,
                    keyframe_available=item.hit.keyframe_available,
                    understanding=understanding,
                    contributing_expansions=item.evidence,
                )
                for rank, item in enumerate(fused, start=1)
            )
            results.append(
                ExpandedSearchResult(
                    query=query,
                    understanding=understanding,
                    hits=hits,
                    expansion_counts={
                        expansion_id: len(expansion_hits)
                        for expansion_id, expansion_hits in rankings.items()
                    },
                )
            )
        return results

    def _resolve_expansions(
        self, query: Query, understanding: QueryUnderstanding
    ) -> tuple[str, ...]:
        candidates = list(understanding.expansions)
        text_key = query.text.strip().casefold()
        if not any(value.strip().casefold() == text_key for value in candidates):
            candidates.insert(0, query.text)
        seen: set[str] = set()
        deduped: list[str] = []
        for value in candidates:
            key = value.strip().casefold()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(value)
            if len(deduped) == self.config.max_expansions:
                break
        return tuple(deduped)


__all__ = ["ExpandedHybridSearch", "ExpandedSearchResult", "ExpansionConfig"]
