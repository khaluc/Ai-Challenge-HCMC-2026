from __future__ import annotations

from pathlib import Path

import numpy as np

from data_processing.build import (
    BuildContext,
    build_clip_index,
    build_mapping,
    build_metadata_index,
    build_object_store,
)
from data_processing.layout import DatasetLayout
from retrieval.clip_store import Phase1ClipStore
from retrieval.schemas import Query
from retrieval.hybrid_store import Phase1HybridStore
from retrieval.processing import RuleBasedObjectParser
from retrieval.hybrid_search import HybridConfig, HybridTextualKIS
from kis.expansion_io import write_expanded_predictions, write_expanded_submission
from llm.expansion_retrieval import ExpandedHybridSearch, ExpansionConfig
from llm.query_expansion import RuleBasedQueryUnderstanding


class FakeEncoder:
    def encode(self, texts: list[str]) -> np.ndarray:
        assert texts
        return np.repeat(
            np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            len(texts),
            axis=0,
        )


def _build_hybrid_store(layout: DatasetLayout, output: Path) -> Phase1HybridStore:
    context = BuildContext.create(layout, output)
    build_mapping(context)
    build_clip_index(context, batch_size=2)
    build_metadata_index(context)
    build_object_store(context, index_min_confidence=0.2, frame_batch_size=2)
    clip = Phase1ClipStore(
        output / "clip" / "faiss.index",
        output / "catalog" / "frame_mapping.parquet",
        expected_dimension=3,
    )
    return Phase1HybridStore(output, clip)


def test_expanded_search_generates_multiple_expansions_and_fuses_across_them(
    synthetic_layout: DatasetLayout,
    tmp_path: Path,
) -> None:
    output = tmp_path / "phase1"
    store = _build_hybrid_store(synthetic_layout, output)
    hybrid = HybridTextualKIS(
        store,
        FakeEncoder(),
        config=HybridConfig(
            semantic_candidates=4,
            metadata_video_candidates=2,
            metadata_frames_per_video=2,
            object_candidates=4,
            object_min_confidence=0.3,
        ),
    )
    object_parser = RuleBasedObjectParser(store.object_labels)
    understanding = RuleBasedQueryUnderstanding(object_parser, max_expansions=3)
    engine = ExpandedHybridSearch(
        hybrid,
        understanding,
        config=ExpansionConfig(max_expansions=3, candidates_per_expansion=4, rrf_k=60),
    )

    result = engine.search_detailed_batch(
        [Query("expanded-q", "một người cạnh ô tô")],
        top_k=4,
    )[0]

    assert result.understanding.structure.objects == ("person", "car")
    assert result.understanding.structure.relation == "next_to"
    assert result.understanding.expansions == (
        "một người cạnh ô tô",
        "person next to car",
        "person beside car",
    )
    assert set(result.expansion_counts) == {"e0", "e1", "e2"}

    assert [hit.rank for hit in result.hits] == list(range(1, len(result.hits) + 1))
    assert len({(hit.video_id, hit.frame_id) for hit in result.hits}) == len(result.hits)
    best = result.hits[0]
    assert best.understanding is result.understanding
    assert len(best.contributing_expansions) >= 1
    assert best.best_expansion.rank == 1

    predictions = write_expanded_predictions(tmp_path / "expanded.csv", result.hits)
    lines = predictions.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(result.hits) + 1

    submission = write_expanded_submission(
        tmp_path / "submission", result.hits, zip_path=tmp_path / "submission.zip"
    )
    assert len(submission) == 1
    assert len(submission[0].read_text(encoding="utf-8").splitlines()) == len(result.hits)


def test_expanded_search_single_query_convenience_method(
    synthetic_layout: DatasetLayout,
    tmp_path: Path,
) -> None:
    output = tmp_path / "phase1-single"
    store = _build_hybrid_store(synthetic_layout, output)
    hybrid = HybridTextualKIS(store, FakeEncoder())
    object_parser = RuleBasedObjectParser(store.object_labels)
    understanding = RuleBasedQueryUnderstanding(object_parser)
    engine = ExpandedHybridSearch(hybrid, understanding)

    hits = engine.search("một người cạnh ô tô", query_id="single", top_k=3)

    assert all(hit.query_id == "single" for hit in hits)
    assert [hit.rank for hit in hits] == list(range(1, len(hits) + 1))
