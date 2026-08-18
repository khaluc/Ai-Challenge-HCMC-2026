from __future__ import annotations

import pytest

from retrieval.clip_store import FrameRecord
from retrieval.fusion import fuse_rankings
from retrieval.processing import RuleBasedObjectParser, fold_text, tokenize_for_metadata
from retrieval.hybrid_schemas import BranchHit


def _frame(faiss_index: int, video_id: str, frame_id: int) -> FrameRecord:
    return FrameRecord(
        faiss_index=faiss_index,
        video_id=video_id,
        keyframe_index=faiss_index + 1,
        video_frame_id=frame_id,
        timestamp=float(faiss_index),
        keyframe_path=f"keyframes/{faiss_index + 1:03d}.jpg",
        keyframe_available=True,
    )


def test_vietnamese_processing_extracts_supported_objects_only() -> None:
    parser = RuleBasedObjectParser(
        [
            "Person",
            "Man",
            "Woman",
            "Boy",
            "Girl",
            "Car",
            "Vehicle",
            "Dog",
            "Bowl",
            "Food",
            "Table",
            "Orange",
        ]
    )

    parsed = parser.parse("một người mặc áo đỏ đứng cạnh ô tô")
    assert parsed.concepts == ("person", "car")
    assert parsed.labels_by_concept["person"] == (
        "Person",
        "Man",
        "Woman",
        "Boy",
        "Girl",
    )
    assert parsed.labels_by_concept["car"] == ("Car", "Vehicle")
    # A common function word must not be interpreted as the animal "chó";
    # "thức ăn" is a genuine food alias, so both concepts are expected.
    assert parser.parse("thức ăn cho người dân").concepts == ("food", "person")
    # Detector synonyms from the same concept must not become two artificial
    # requirements that one generic Vehicle detection can satisfy twice.
    assert parser.parse("a car and a vehicle").concepts == ("car",)
    assert parser.parse("person car").concepts == ("person", "car")
    assert parser.parse("bowl food table").concepts == ("table", "bowl", "food")
    assert parser.parse("orange car").concepts == ("car",)

    assert fold_text("Đường phố Hà Nội") == "duong pho ha noi"
    assert tokenize_for_metadata("một người đứng cạnh ô tô tại Hà Nội") == (
        "nguoi",
        "ha",
        "noi",
    )


def test_rrf_boosts_cross_branch_hit_and_keeps_deterministic_submit_mapping() -> None:
    common_semantic = _frame(8, "L21_V001", 100)
    # Same submit frame, different keyframe/FAISS id in another branch.
    common_object = _frame(3, "L21_V001", 100)
    semantic_only = _frame(1, "L21_V002", 200)
    object_only = _frame(2, "L21_V003", 300)
    rankings = {
        "semantic": [
            BranchHit("semantic", 1, 0.9, common_semantic),
            BranchHit("semantic", 2, 0.8, semantic_only),
        ],
        "metadata": [],
        "objects": [
            BranchHit("objects", 1, 0.95, common_object),
            BranchHit("objects", 2, 0.80, object_only),
        ],
    }

    fused = fuse_rankings(
        rankings,
        method="rrf",
        weights={"semantic": 1.0, "metadata": 0.0, "objects": 1.0},
        rrf_k=60,
        limit=3,
    )

    assert [(hit.frame.video_id, hit.frame.video_frame_id) for hit in fused] == [
        ("L21_V001", 100),
        ("L21_V002", 200),
        ("L21_V003", 300),
    ]
    # Semantic evidence is the authoritative diagnostic representative.
    assert fused[0].frame.faiss_index == 8
    assert set(fused[0].evidence) == {"semantic", "objects"}


def test_weighted_fusion_normalizes_each_branch_and_rejects_duplicate_pairs() -> None:
    first = _frame(0, "L21_V001", 0)
    second = _frame(1, "L21_V002", 1)
    rankings = {
        "semantic": [
            BranchHit("semantic", 1, 10.0, first),
            BranchHit("semantic", 2, 0.0, second),
        ],
        "metadata": [
            BranchHit("metadata", 1, 2.0, second),
            BranchHit("metadata", 2, 1.0, first),
        ],
        "objects": [],
    }
    fused = fuse_rankings(
        rankings,
        method="weighted",
        weights={"semantic": 1.0, "metadata": 2.0, "objects": 0.0},
        limit=2,
    )
    assert fused[0].frame.video_id == "L21_V002"
    assert fused[0].score == pytest.approx(2 / 3)
    assert fused[1].score == pytest.approx(1 / 3)

    duplicate = {
        "semantic": [
            BranchHit("semantic", 1, 1.0, first),
            BranchHit("semantic", 2, 0.9, _frame(9, "L21_V001", 0)),
        ]
    }
    with pytest.raises(ValueError, match="duplicate submit pairs"):
        fuse_rankings(duplicate)
