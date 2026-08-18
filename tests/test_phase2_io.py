from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from retrieval.io import (
    PREDICTION_FIELDS,
    load_queries,
    write_btc_submission,
    write_predictions_csv,
)
from retrieval.schemas import Query, RetrievalHit


def _hit(
    query_id: str,
    rank: int,
    *,
    video_id: str | None = None,
    frame_id: int | None = None,
) -> RetrievalHit:
    return RetrievalHit(
        query_id=query_id,
        rank=rank,
        video_id=video_id or f"L21_V{rank:03d}.mp4",
        frame_id=rank if frame_id is None else frame_id,
        score=1.0 / rank,
        faiss_index=rank - 1,
        keyframe_index=rank,
        timestamp=rank / 25,
        keyframe_path=f"keyframes/{rank:03d}.jpg",
        keyframe_available=rank % 2 == 0,
    )


def test_load_queries_from_exact_csv_and_jsonl_formats(tmp_path: Path) -> None:
    csv_path = tmp_path / "queries.csv"
    csv_path.write_bytes(
        "\ufeffquery_id,text\nq1,Một người đi xe đạp\nq2,\"Hồ nước, ban đêm\"\n".encode(
            "utf-8"
        )
    )
    assert load_queries(csv_path) == [
        Query("q1", "Một người đi xe đạp"),
        Query("q2", "Hồ nước, ban đêm"),
    ]

    jsonl_path = tmp_path / "queries.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps({"query_id": "a", "text": "first"}),
                "",
                json.dumps({"query_id": "b", "text": "second", "note": 1}),
            ]
        ),
        encoding="utf-8",
    )
    assert load_queries(jsonl_path) == [Query("a", "first"), Query("b", "second")]


def test_load_queries_rejects_bad_header_and_duplicate_ids(tmp_path: Path) -> None:
    wrong_header = tmp_path / "wrong.csv"
    wrong_header.write_text("text,query_id\nhello,q1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected CSV header"):
        load_queries(wrong_header)

    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(
        '{"query_id":"q1","text":"one"}\n'
        '{"query_id":"q1","text":"two"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate query_id 'q1'"):
        load_queries(duplicate)


def test_internal_predictions_csv_has_all_fields_and_exact_bytes(tmp_path: Path) -> None:
    output = tmp_path / "predictions.csv"
    hit = RetrievalHit(
        query_id="q1",
        rank=1,
        video_id="L21_V001",
        frame_id=31,
        score=0.75,
        faiss_index=42,
        keyframe_index=2,
        timestamp=1.24,
        keyframe_path="archive.zip::Keyframes/L21_V001/002.jpg",
        keyframe_available=False,
    )

    assert write_predictions_csv(output, [hit]) == output
    expected = (
        ",".join(PREDICTION_FIELDS)
        + "\nq1,1,L21_V001,31,0.75,42,2,1.24,"
        "archive.zip::Keyframes/L21_V001/002.jpg,False\n"
    ).encode("utf-8")
    assert output.read_bytes() == expected


def test_official_submission_is_headerless_ranked_top100_and_direct_zip(
    tmp_path: Path,
) -> None:
    # Reverse input proves that rank, not iterable order, controls the output.
    hits = [_hit("query 01", rank) for rank in range(105, 0, -1)]
    output_dir = tmp_path / "submission"
    archive = tmp_path / "submission.zip"

    files = write_btc_submission(
        output_dir, hits, max_results=999, zip_path=archive
    )

    assert files == [output_dir / "query_01.csv"]
    expected = "".join(
        f"L21_V{rank:03d},{rank}\n" for rank in range(1, 101)
    ).encode("utf-8")
    assert files[0].read_bytes() == expected
    assert b"video_id" not in expected
    assert b".mp4" not in expected
    with ZipFile(archive) as handle:
        assert handle.namelist() == ["query_01.csv"]
        assert handle.read("query_01.csv") == expected


def test_official_submission_sanitizes_names_and_rejects_unsafe_or_colliding_ids(
    tmp_path: Path,
) -> None:
    safe_dir = tmp_path / "safe"
    files = write_btc_submission(safe_dir, [_hit("Câu hỏi: 1", 1)])
    assert files == [safe_dir / "Câu_hỏi_1.csv"]

    traversal_dir = tmp_path / "traversal"
    with pytest.raises(ValueError, match="path traversal"):
        write_btc_submission(traversal_dir, [_hit("../outside", 1)])
    assert not traversal_dir.exists()
    assert not (tmp_path / "outside.csv").exists()

    collision_dir = tmp_path / "collision"
    with pytest.raises(ValueError, match="filename collision"):
        write_btc_submission(
            collision_dir,
            [_hit("query:1", 1), _hit("query?1", 1)],
        )
    assert not collision_dir.exists()


def test_official_submission_rejects_zip_csv_path_collision_without_mutation(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "submission"
    csv_path = output_dir / "q.csv"

    with pytest.raises(ValueError, match="conflicts with submission CSV"):
        write_btc_submission(
            output_dir,
            [_hit("q", 1)],
            zip_path=csv_path,
        )

    assert not output_dir.exists()
    assert not csv_path.exists()


def test_internal_writer_validates_before_replacing_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "predictions.csv"
    original = b"previous valid output\n"
    output.write_bytes(original)

    with pytest.raises(TypeError, match="RetrievalHit"):
        write_predictions_csv(output, [_hit("q", 1), object()])

    assert output.read_bytes() == original
    assert not output.with_suffix(".csv.tmp").exists()
