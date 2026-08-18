from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data_processing.build import (
    BuildContext,
    build_clip_index,
    build_mapping,
    build_metadata_index,
    build_object_store,
)
from data_processing.layout import DatasetLayout
from retrieval.clip_store import FrameRecord, Phase1ClipStore
from retrieval.hybrid_store import Phase1HybridStore
from trake.temporal_dp import AlignmentConfig, CoarseTemporalAligner, align_events
from trake.temporal_dp_schemas import CoarseAlignment, EventFrameAssignment


def _frame(keyframe_index: int, video_frame_id: int) -> FrameRecord:
    return FrameRecord(
        faiss_index=keyframe_index,
        video_id="L21_V001",
        keyframe_index=keyframe_index,
        video_frame_id=video_frame_id,
        timestamp=float(keyframe_index),
        keyframe_path=f"keyframes/{keyframe_index:03d}.jpg",
        keyframe_available=True,
    )


def test_align_events_respects_monotonic_order_over_greedy_argmax() -> None:
    # Frame 2 is individually the best match for BOTH events. Greedy argmax
    # would assign both events to frame 2 (invalid). The DP must instead
    # find the best *jointly increasing* assignment.
    frames = [_frame(1, 10), _frame(2, 20), _frame(3, 30)]
    similarity = [
        [0.1, 0.9, 0.5],  # event 0: best at frame index 1 (0.9)
        [0.2, 0.95, 0.6],  # event 1: best at frame index 1 too (0.95), but that's taken
    ]

    result = align_events(
        ["event a", "event b"], frames, similarity, video_id="L21_V001"
    )

    assert result.feasible
    assert [item.frame.keyframe_index for item in result.assignments] == [2, 3]
    # 0.9 (event0 @ frame idx1) + 0.6 (event1 @ frame idx2) = 1.5, beats any
    # other strictly-increasing pairing.
    assert result.total_score == pytest.approx(1.5)


def test_align_events_infeasible_when_fewer_frames_than_events() -> None:
    frames = [_frame(1, 10)]
    similarity = [[0.5], [0.5], [0.5]]

    result = align_events(["e1", "e2", "e3"], frames, similarity, video_id="L21_V001")

    assert not result.feasible
    assert result.assignments == ()
    assert result.total_score == 0.0


def test_align_events_transition_penalty_discourages_adjacent_frames() -> None:
    frames = [_frame(1, 10), _frame(2, 20), _frame(3, 30), _frame(4, 40)]
    # Event 1's best two options are frame idx1 (adjacent to event0's best,
    # frame idx0) and frame idx3 (farther away, slightly lower score).
    similarity = [
        [0.9, 0.1, 0.1, 0.1],
        [0.1, 0.85, 0.1, 0.80],
    ]

    no_penalty = align_events(
        ["e0", "e1"], frames, similarity, video_id="L21_V001",
        config=AlignmentConfig(transition_penalty_weight=0.0),
    )
    assert [item.frame.keyframe_index for item in no_penalty.assignments] == [1, 2]

    with_penalty = align_events(
        ["e0", "e1"], frames, similarity, video_id="L21_V001",
        config=AlignmentConfig(transition_penalty_weight=1.0, min_frame_gap=3),
    )
    # Adjacent gap (1) is penalized by (3-1)*1.0=2.0, making the farther
    # option (gap 3, no penalty) the better total.
    assert [item.frame.keyframe_index for item in with_penalty.assignments] == [1, 4]


def test_align_events_rejects_mismatched_similarity_shape() -> None:
    with pytest.raises(ValueError):
        align_events(["e0", "e1"], [_frame(1, 10)], [[0.5]], video_id="L21_V001")


def test_coarse_alignment_rejects_non_monotonic_assignments() -> None:
    with pytest.raises(ValueError):
        CoarseAlignment(
            video_id="L21_V001",
            events=("e0", "e1"),
            assignments=(
                EventFrameAssignment(0, "e0", _frame(2, 20), 0.5),
                EventFrameAssignment(1, "e1", _frame(1, 10), 0.5),
            ),
            total_score=1.0,
            feasible=True,
        )


class _EventVectorEncoder:
    """Maps known event strings to fixed CLIP-space vectors for a deterministic test."""

    _VECTORS = {
        "matches-first-frame": [1.0, 0.0, 0.0],
        "matches-second-frame": [0.0, 1.0, 0.0],
    }

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray([self._VECTORS[text] for text in texts], dtype=np.float32)


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


def test_coarse_temporal_aligner_wires_real_per_video_vectors(
    synthetic_layout: DatasetLayout, tmp_path: Path
) -> None:
    store = _build_hybrid_store(synthetic_layout, tmp_path / "phase1")
    aligner = CoarseTemporalAligner(store, _EventVectorEncoder())

    result = aligner.align("L21_V001", ["matches-first-frame", "matches-second-frame"])

    assert result.feasible
    assert len(result.assignments) == 2
    assert result.assignments[0].frame.keyframe_index < result.assignments[1].frame.keyframe_index
    # keyframe 1 -> raw feature [3,0,0] (normalizes to [1,0,0]) should score
    # near-perfect cosine similarity against "matches-first-frame".
    assert result.assignments[0].similarity > 0.99


def test_coarse_temporal_aligner_raises_on_empty_events(
    synthetic_layout: DatasetLayout, tmp_path: Path
) -> None:
    store = _build_hybrid_store(synthetic_layout, tmp_path / "phase1-empty")
    aligner = CoarseTemporalAligner(store, _EventVectorEncoder())
    with pytest.raises(ValueError):
        aligner.align("L21_V001", [])
