from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pyarrow.parquet as pq
import pytest

from data_processing.build import BuildContext, build_clip_index, build_mapping
from data_processing.layout import DatasetLayout


def test_mapping_keeps_one_based_keyframes_and_exact_submit_frame_ids(
    synthetic_layout: DatasetLayout, tmp_path: Path
) -> None:
    output = tmp_path / "mapping-output"
    context = BuildContext.create(synthetic_layout, output)

    result = build_mapping(context)
    rows = pq.read_table(result["parquet"]).to_pylist()

    assert result["rows"] == 4
    assert [
        (
            row["Video_ID"],
            row["Keyframe_Index"],
            row["CLIP_Index"],
            row["FAISS_Index"],
            row["Video_Frame_ID"],
        )
        for row in rows
    ] == [
        ("L21_V001", 1, 0, 0, 0),
        ("L21_V001", 2, 1, 1, 31),
        ("L21_V002", 1, 0, 2, 100),
        ("L21_V002", 2, 1, 3, 203),
    ]
    assert rows[0]["Keyframe_Path"].endswith("::Keyframes_L21/L21_V001/001.jpg")
    assert rows[0]["Object_Path"].endswith("::objects/L21_V001/001.json")


def test_mapping_rejects_a_zero_based_keyframe_sequence(
    dataset_factory, tmp_path: Path
) -> None:
    layout = dataset_factory(zero_based_first_map=True)
    context = BuildContext.create(layout, tmp_path / "bad-mapping-output")

    with pytest.raises(ValueError, match=r"L21_V001: keyframe n=0 at CLIP row 0"):
        build_mapping(context)


def test_faiss_index_order_normalization_and_search_join(
    synthetic_layout: DatasetLayout, tmp_path: Path
) -> None:
    output = tmp_path / "clip-output"
    context = BuildContext.create(synthetic_layout, output)
    mapping_result = build_mapping(context)

    result = build_clip_index(context, batch_size=1)

    vectors = np.load(output / "clip" / "clip_vectors.f32.npy")
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(
        vectors,
        np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
            ],
            dtype=np.float32,
        ),
        atol=1e-6,
    )
    offsets = pq.read_table(output / "clip" / "video_offsets.parquet").to_pylist()
    assert offsets == [
        {"Video_ID": "L21_V001", "FAISS_Start": 0, "FAISS_Stop": 2, "Rows": 2},
        {"Video_ID": "L21_V002", "FAISS_Start": 2, "FAISS_Stop": 4, "Rows": 2},
    ]

    index = faiss.read_index(str(output / "clip" / "faiss.index"))
    scores, positions = index.search(
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), 4
    )
    assert result["vectors"] == index.ntotal == 4
    assert positions[0].tolist() == [0, 3, 1, 2]
    np.testing.assert_allclose(
        scores[0], [1.0, 1 / np.sqrt(3), 0.0, -1.0], atol=1e-6
    )

    mapping = pq.read_table(mapping_result["parquet"]).to_pylist()
    best = mapping[int(positions[0, 0])]
    runner_up = mapping[int(positions[0, 1])]
    assert (best["Video_ID"], best["Keyframe_Index"], best["Video_Frame_ID"]) == (
        "L21_V001",
        1,
        0,
    )
    assert (
        runner_up["Video_ID"],
        runner_up["Keyframe_Index"],
        runner_up["Video_Frame_ID"],
    ) == ("L21_V002", 2, 203)
