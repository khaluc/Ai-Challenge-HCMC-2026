from __future__ import annotations

from retrieval.schemas import Query
from retrieval.hybrid_store import Phase1HybridStore
from retrieval.processing import tokenize_for_metadata
from retrieval.hybrid_search import HybridTextualKIS
from llm.event_parser import EventDecompositionProtocol

from .aggregation import VideoRetrievalConfig, aggregate_video_candidates
from .video_retrieval_schemas import TRAKERetrievalResult


class TRAKEVideoRetrieval:
    """Full query + every event + LLM expansions -> per-video score aggregation.

    A wrong video zeroes the whole TRAKE R-Score, so this never commits to a
    single video: it keeps the Top `video_limit` (3-5) candidates for the
    next stage (per-event frame localization inside the chosen video) to
    disambiguate.
    """

    def __init__(
        self,
        hybrid: HybridTextualKIS,
        store: Phase1HybridStore,
        decomposer: EventDecompositionProtocol,
        *,
        config: VideoRetrievalConfig | None = None,
    ) -> None:
        self.hybrid = hybrid
        self.store = store
        self.decomposer = decomposer
        self.config = config or VideoRetrievalConfig()

    def find_candidate_videos(
        self, query: str, *, query_id: str = "trake"
    ) -> TRAKERetrievalResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonblank string")

        event_sequence = self.decomposer.decompose(query)

        subquery_texts = [query, *event_sequence.events, *event_sequence.expansions]
        subqueries = [
            Query(f"{query_id}::s{index}", text) for index, text in enumerate(subquery_texts)
        ]
        results = self.hybrid.search_detailed_batch(
            subqueries, top_k=self.config.per_query_top_k
        )

        full_result = results[0]
        n_events = len(event_sequence.events)
        event_results = list(zip(event_sequence.events, results[1 : 1 + n_events]))
        expansion_results = list(zip(event_sequence.expansions, results[1 + n_events :]))

        metadata_terms = tokenize_for_metadata(query)
        metadata_hits = (
            self.store.search_metadata(metadata_terms, limit=self.config.metadata_video_limit)
            if metadata_terms
            else []
        )

        candidates = aggregate_video_candidates(
            full_result=full_result,
            event_results=event_results,
            expansion_results=expansion_results,
            metadata_hits=metadata_hits,
            config=self.config,
        )[: self.config.video_limit]

        return TRAKERetrievalResult(
            query_id=query_id,
            query=query,
            event_sequence=event_sequence,
            candidates=tuple(candidates),
        )


__all__ = ["TRAKEVideoRetrieval"]
