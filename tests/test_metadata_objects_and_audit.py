from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from data_processing.inspect_dataset import audit_dataset
from data_processing.build import BuildContext, build_metadata_index, build_object_store
from data_processing.layout import DatasetLayout


def _bm25_search(database: Path, query: str) -> list[tuple[str, float]]:
    with sqlite3.connect(database) as connection:
        return connection.execute(
            """
            SELECT video_id, bm25(metadata_fts) AS score
            FROM metadata_fts
            WHERE metadata_fts MATCH ?
            ORDER BY score, video_id
            """,
            (query,),
        ).fetchall()


def test_metadata_bm25_matches_vietnamese_with_and_without_diacritics(
    synthetic_layout: DatasetLayout, tmp_path: Path
) -> None:
    output = tmp_path / "metadata-output"
    context = BuildContext.create(synthetic_layout, output)

    result = build_metadata_index(context)

    accented = _bm25_search(Path(result["database"]), '"Đường phố Hà Nội"')
    folded = _bm25_search(Path(result["database"]), '"duong pho ha noi"')
    assert [row[0] for row in accented] == ["L21_V001"]
    assert [row[0] for row in folded] == ["L21_V001"]
    assert all(math.isfinite(row[1]) for row in accented + folded)

    records = pq.read_table(result["parquet"]).to_pylist()
    first = next(row for row in records if row["video_id"] == "L21_V001")
    assert "duong pho ha noi" in first["text_folded"]
    assert first["keywords_json"] == '["Thủ đô", "Việt Nam"]'


def test_object_parser_threshold_and_video_frame_id_join(
    synthetic_layout: DatasetLayout, tmp_path: Path
) -> None:
    output = tmp_path / "objects-output"
    context = BuildContext.create(synthetic_layout, output)

    result = build_object_store(
        context, index_min_confidence=0.50, frame_batch_size=1
    )

    assert result["frames"] == 4
    assert result["raw_detections"] == 5
    assert result["indexed_detections"] == 3

    raw_rows = pq.read_table(result["raw_lossless_parquet"]).to_pylist()
    assert len(raw_rows) == 4
    first_raw = next(
        row
        for row in raw_rows
        if row["Video_ID"] == "L21_V001" and row["Keyframe_Index"] == 1
    )
    assert first_raw["Video_Frame_ID"] == 0
    assert first_raw["Object_Class"] == ["person", "car", "laptop"]
    assert first_raw["Confidence"] == pytest.approx([0.95, 0.49, 0.50])

    flat_rows = pq.read_table(result["filtered_flat_parquet"]).to_pylist()
    assert [
        (
            row["Video_ID"],
            row["Keyframe_Index"],
            row["Video_Frame_ID"],
            row["Detection_Rank"],
            row["Object_Class"],
            row["Class_MID"],
        )
        for row in flat_rows
    ] == [
        ("L21_V001", 1, 0, 1, "person", "/m/person"),
        ("L21_V001", 1, 0, 3, "laptop", "/m/laptop"),
        ("L21_V002", 1, 100, 1, "car", "/m/car"),
    ]
    laptop = next(row for row in flat_rows if row["Object_Class"] == "laptop")
    assert laptop["Confidence"] == pytest.approx(0.50)
    assert [laptop[name] for name in ("YMin", "XMin", "YMax", "XMax")] == pytest.approx(
        [0.4, 0.1, 0.8, 0.6]
    )

    with sqlite3.connect(result["query_database"]) as connection:
        sqlite_rows = connection.execute(
            """
            SELECT video_id, keyframe_index, video_frame_id, detection_rank,
                   object_class, confidence
            FROM detections
            ORDER BY video_id, keyframe_index, detection_rank
            """
        ).fetchall()
    assert [(row[0], row[1], row[2], row[3], row[4]) for row in sqlite_rows] == [
        ("L21_V001", 1, 0, 1, "person"),
        ("L21_V001", 1, 0, 3, "laptop"),
        ("L21_V002", 1, 100, 1, "car"),
    ]
    assert [row[5] for row in sqlite_rows] == pytest.approx([0.95, 0.50, 0.80])


def test_audit_counts_an_individual_missing_keyframe(
    dataset_factory, tmp_path: Path
) -> None:
    layout = dataset_factory(missing_keyframes={("L21_V001", 2)})

    report = audit_dataset(layout, deep_crc=True)

    assert report["status"] == {
        "core_retrieval_data_complete": True,
        "keyframe_images_complete": False,
        "original_videos_complete": True,
        "deep_crc_checked": True,
    }
    assert report["summary"]["mapped_keyframes"] == 4
    assert report["summary"]["available_keyframe_images"] == 3
    assert report["summary"]["missing_keyframe_images"] == 1
    assert report["summary"]["missing_keyframes_by_level"] == {"L21": 1}
    # One image is absent, but the video still has another keyframe in its archive.
    assert report["summary"]["missing_keyframe_videos"] == 0
    assert report["missing_video_ids"]["keyframes"] == []
    assert report["errors"] == []
