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
from retrieval.evaluator import load_predictions
from retrieval.schemas import Query
from retrieval.hybrid_store import Phase1HybridStore
from kis.hybrid_io import write_hybrid_predictions, write_hybrid_submission
from retrieval.hybrid_search import HybridConfig, HybridTextualKIS


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


def test_hybrid_retrieval_combines_all_branches_and_exports_evaluator_csv(
    synthetic_layout: DatasetLayout,
    tmp_path: Path,
) -> None:
    output = tmp_path / "phase1"
    store = _build_hybrid_store(synthetic_layout, output)
    engine = HybridTextualKIS(
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

    result = engine.search_detailed_batch(
        [Query("hybrid-q", "Đường phố Hà Nội, một người cạnh ô tô")],
        top_k=4,
    )[0]

    assert result.analysis.object_concepts == ("person", "car")
    assert result.branch_counts["semantic"] == 4
    assert result.branch_counts["metadata"] >= 1
    assert result.branch_counts["objects"] == 1
    assert len(result.hits) == 4
    assert [hit.rank for hit in result.hits] == [1, 2, 3, 4]
    assert len({(hit.video_id, hit.frame_id) for hit in result.hits}) == 4
    best = result.hits[0]
    assert (best.video_id, best.frame_id) == ("L21_V001", 0)
    assert best.semantic_rank == 1
    assert best.metadata_rank is not None
    assert best.object_rank == 1
    assert best.matched_objects == ("person", "car")

    predictions = write_hybrid_predictions(tmp_path / "hybrid.csv", result.hits)
    loaded = load_predictions(predictions, known_query_ids={"hybrid-q"})
    assert [(row.rank, row.video_id, row.frame_id) for row in loaded] == [
        (hit.rank, hit.video_id, hit.frame_id) for hit in result.hits
    ]
    submission = write_hybrid_submission(
        tmp_path / "submission",
        result.hits,
        zip_path=tmp_path / "submission.zip",
    )
    assert len(submission) == 1
    assert len(submission[0].read_text(encoding="utf-8").splitlines()) == 4


def test_hybrid_rejects_object_threshold_below_index_floor(
    synthetic_layout: DatasetLayout,
    tmp_path: Path,
) -> None:
    store = _build_hybrid_store(synthetic_layout, tmp_path / "phase1-floor")
    try:
        HybridTextualKIS(
            store,
            FakeEncoder(),
            config=HybridConfig(object_min_confidence=0.19),
        )
    except ValueError as exc:
        assert "below the Phase 1 index floor" in str(exc)
    else:
        raise AssertionError("Expected the object index floor to be enforced")

    # An irrelevant threshold must not block a deliberate object-branch ablation.
    HybridTextualKIS(
        store,
        FakeEncoder(),
        config=HybridConfig(
            object_min_confidence=0.0,
            objects_enabled=False,
        ),
    )
