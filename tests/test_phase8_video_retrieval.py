from __future__ import annotations

import pytest

from retrieval.schemas import Query
from retrieval.hybrid_store import MetadataVideoHit
from retrieval.hybrid_search import HybridSearchResult
from retrieval.hybrid_schemas import HybridHit, QueryAnalysis
from llm.event_schemas import EventSequence
from trake.aggregation import VideoRetrievalConfig, aggregate_video_candidates
from trake.video_retrieval import TRAKEVideoRetrieval


def _hit(query_id: str, rank: int, video_id: str, frame_id: int, semantic_score: float) -> HybridHit:
    return HybridHit(
        query_id=query_id,
        rank=rank,
        video_id=video_id,
        frame_id=frame_id,
        score=1.0 / rank,
        faiss_index=rank,
        keyframe_index=rank,
        timestamp=float(rank),
        keyframe_path=f"keyframes/{rank:03d}.jpg",
        keyframe_available=True,
        fusion_method="rrf",
        semantic_rank=rank,
        semantic_score=semantic_score,
    )


def _result(query_id: str, query_text: str, hits: list[HybridHit]) -> HybridSearchResult:
    query = Query(query_id, query_text)
    analysis = QueryAnalysis(text=query_text, metadata_terms=(), object_concepts=())
    return HybridSearchResult(query=query, analysis=analysis, hits=tuple(hits), branch_counts={})


def _metadata_hit(video_id: str, rank: int, bm25_score: float) -> MetadataVideoHit:
    return MetadataVideoHit(
        rank=rank,
        video_id=video_id,
        bm25_score=bm25_score,
        title="t",
        author="a",
        publish_date="d",
        watch_url="u",
        match_mode="and",
    )


def test_aggregate_video_candidates_combines_all_components() -> None:
    full_result = _result(
        "q::s0",
        "full query",
        [_hit("q::s0", 1, "L21_V001", 10, 0.9), _hit("q::s0", 2, "L21_V002", 20, 0.5)],
    )
    event1_result = _result("q::s1", "event1", [_hit("q::s1", 1, "L21_V001", 10, 0.8)])
    event2_result = _result("q::s2", "event2", [_hit("q::s2", 1, "L21_V003", 30, 0.7)])
    expansion_result = _result("q::s3", "expansion", [_hit("q::s3", 1, "L21_V001", 10, 0.85)])
    metadata_hits = [_metadata_hit("L21_V001", 1, -5.0), _metadata_hit("L21_V002", 2, -2.0)]

    candidates = aggregate_video_candidates(
        full_result=full_result,
        event_results=[("event1", event1_result), ("event2", event2_result)],
        expansion_results=[("expansion text", expansion_result)],
        metadata_hits=metadata_hits,
        config=VideoRetrievalConfig(video_limit=5),
    )

    by_video = {candidate.video_id: candidate for candidate in candidates}
    assert candidates[0].video_id == "L21_V001"
    assert [candidate.rank for candidate in candidates] == list(range(1, len(candidates) + 1))
    assert by_video["L21_V001"].event_coverage == pytest.approx(0.5)
    assert by_video["L21_V003"].event_coverage == pytest.approx(0.5)
    assert by_video["L21_V002"].event_coverage == pytest.approx(0.0)
    assert by_video["L21_V001"].global_similarity == pytest.approx((0.9 + 1.0) / 2.0)
    assert by_video["L21_V001"].bm25_score == pytest.approx(1.0)
    assert by_video["L21_V002"].bm25_score == pytest.approx(0.0)
    # total subqueries = full + 2 events + 1 expansion = 4; L21_V001 appears in 3 of them
    assert by_video["L21_V001"].multi_query_vote == pytest.approx(3 / 4)
    assert by_video["L21_V001"].best_frame_id == 10


def test_aggregate_video_candidates_respects_video_limit_via_caller() -> None:
    full_result = _result(
        "q::s0",
        "full query",
        [_hit("q::s0", 1, "L21_V001", 10, 0.9), _hit("q::s0", 2, "L21_V002", 20, 0.5)],
    )
    candidates = aggregate_video_candidates(
        full_result=full_result,
        event_results=[],
        expansion_results=[],
        metadata_hits=[],
        config=VideoRetrievalConfig(video_limit=1),
    )[:1]
    assert len(candidates) == 1
    assert candidates[0].video_id == "L21_V001"


def test_video_retrieval_config_rejects_all_zero_weights() -> None:
    with pytest.raises(ValueError):
        VideoRetrievalConfig(
            global_similarity_weight=0,
            event_coverage_weight=0,
            bm25_weight=0,
            vote_weight=0,
        )


def test_video_retrieval_config_rejects_top_k_above_100() -> None:
    with pytest.raises(ValueError):
        VideoRetrievalConfig(per_query_top_k=101)


class _FakeHybrid:
    def __init__(self, results_by_query_id: dict[str, HybridSearchResult]) -> None:
        self._results_by_query_id = results_by_query_id
        self.last_call: dict | None = None

    def search_detailed_batch(self, queries, *, top_k):
        self.last_call = {"queries": list(queries), "top_k": top_k}
        return [self._results_by_query_id[query.query_id] for query in queries]


class _FakeStore:
    def __init__(self, metadata_hits: list[MetadataVideoHit]) -> None:
        self._metadata_hits = metadata_hits
        self.last_call: dict | None = None

    def search_metadata(self, terms, *, limit):
        self.last_call = {"terms": terms, "limit": limit}
        return self._metadata_hits


class _FakeDecomposer:
    def __init__(self, event_sequence: EventSequence) -> None:
        self._event_sequence = event_sequence

    def decompose(self, query: str) -> EventSequence:
        return self._event_sequence


def test_trake_video_retrieval_batches_full_query_events_and_expansions() -> None:
    event_sequence = EventSequence(
        query="full query text",
        events=("event a", "event b"),
        expansions=("expansion a",),
    )
    results_by_id = {
        "trake::s0": _result("trake::s0", "full query text", [_hit("trake::s0", 1, "L21_V001", 10, 0.9)]),
        "trake::s1": _result("trake::s1", "event a", [_hit("trake::s1", 1, "L21_V001", 10, 0.8)]),
        "trake::s2": _result("trake::s2", "event b", [_hit("trake::s2", 1, "L21_V002", 20, 0.7)]),
        "trake::s3": _result("trake::s3", "expansion a", [_hit("trake::s3", 1, "L21_V001", 10, 0.85)]),
    }
    hybrid = _FakeHybrid(results_by_id)
    store = _FakeStore([_metadata_hit("L21_V001", 1, -3.0)])
    retrieval = TRAKEVideoRetrieval(
        hybrid, store, _FakeDecomposer(event_sequence), config=VideoRetrievalConfig(video_limit=2)
    )

    result = retrieval.find_candidate_videos("full query text", query_id="trake")

    assert hybrid.last_call["top_k"] == 100
    assert [q.query_id for q in hybrid.last_call["queries"]] == [
        "trake::s0",
        "trake::s1",
        "trake::s2",
        "trake::s3",
    ]
    assert [q.text for q in hybrid.last_call["queries"]] == [
        "full query text",
        "event a",
        "event b",
        "expansion a",
    ]
    assert result.event_sequence is event_sequence
    assert len(result.candidates) == 2
    assert result.candidates[0].video_id == "L21_V001"
    assert [c.rank for c in result.candidates] == [1, 2]


def test_trake_video_retrieval_rejects_blank_query() -> None:
    hybrid = _FakeHybrid({})
    store = _FakeStore([])
    decomposer = _FakeDecomposer(EventSequence(query="x", events=("x",)))
    retrieval = TRAKEVideoRetrieval(hybrid, store, decomposer)
    with pytest.raises(ValueError):
        retrieval.find_candidate_videos("   ")
