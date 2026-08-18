from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from retrieval.clip_store import ArtifactValidationError, Phase1ClipStore
from retrieval.clip_search import TextualKIS
from retrieval.schemas import Query, RetrievalHit


def _write_store(
    tmp_path: Path,
    *,
    faiss_ids: list[int] | None = None,
) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    vectors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.05, 0.0],
            [0.8, 0.6, 0.0],
            [0.8, 0.6, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(3)
    index.add(vectors)
    index_path = tmp_path / "faiss.index"
    faiss.write_index(index, str(index_path))

    # Physical row order is intentionally unrelated to FAISS_Index. IDs 0 and
    # 1 intentionally map to the same BTC submission pair.
    records_by_id = {
        0: {
            "Video_ID": "L21_V001",
            "Keyframe_Index": 7,
            "Video_Frame_ID": 900,
            "Timestamp": 30.0,
            "Keyframe_Path": "missing.zip::L21_V001/007.jpg",
            "Keyframe_Available": False,
        },
        1: {
            "Video_ID": "L21_V001",
            "Keyframe_Index": 8,
            "Video_Frame_ID": 900,
            "Timestamp": 31.0,
            "Keyframe_Path": "present.zip::L21_V001/008.jpg",
            "Keyframe_Available": True,
        },
        2: {
            "Video_ID": "L21_V002",
            "Keyframe_Index": 4,
            "Video_Frame_ID": 777,
            "Timestamp": 20.0,
            "Keyframe_Path": "present.zip::L21_V002/004.jpg",
            "Keyframe_Available": True,
        },
        3: {
            "Video_ID": "L21_V003",
            "Keyframe_Index": 99,
            "Video_Frame_ID": 888,
            "Timestamp": 21.0,
            "Keyframe_Path": "present.zip::L21_V003/099.jpg",
            "Keyframe_Available": True,
        },
        4: {
            "Video_ID": "L21_V004",
            "Keyframe_Index": 1,
            "Video_Frame_ID": 42,
            "Timestamp": 1.0,
            "Keyframe_Path": "present.zip::L21_V004/001.jpg",
            "Keyframe_Available": True,
        },
    }
    physical_order = [3, 0, 4, 1, 2]
    ids = faiss_ids if faiss_ids is not None else physical_order
    rows = [
        dict(records_by_id[source_id], FAISS_Index=stored_id)
        for source_id, stored_id in zip(physical_order, ids)
    ]
    mapping_path = tmp_path / "frame_mapping.parquet"
    pq.write_table(pa.Table.from_pylist(rows), mapping_path)
    return index_path, mapping_path


class _Encoder:
    dimension = 3

    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls.append(texts)
        return self.vectors


def test_explicit_faiss_join_deduplicates_and_backfills_without_image_filter(
    tmp_path: Path,
) -> None:
    index_path, mapping_path = _write_store(tmp_path)
    store = Phase1ClipStore(
        index_path=index_path,
        mapping_path=mapping_path,
        expected_dimension=3,
        overfetch_factor=1,
    )
    encoder = _Encoder(np.asarray([[2.0, 0.0, 0.0]], dtype=np.float64))
    kis = TextualKIS(store, encoder)

    hits = kis.search(Query("q-1", "a scene"), top_k=3)

    assert encoder.calls == [["a scene"]]
    assert all(isinstance(hit, RetrievalHit) for hit in hits)
    # FAISS id 1 is a higher-scoring duplicate of id 0 and must not consume a
    # rank. The result is backfilled with ids 2 and 3.
    assert [hit.faiss_index for hit in hits] == [0, 2, 3]
    assert [(hit.video_id, hit.frame_id) for hit in hits] == [
        ("L21_V001", 900),
        ("L21_V002", 777),
        ("L21_V003", 888),
    ]
    # Exact Video_Frame_ID is used; neither Keyframe_Index nor row number leaks
    # into submission results.
    assert hits[0].frame_id == 900
    assert hits[0].keyframe_index == 7
    # Unavailable images are useful retrieval records and must remain ranked.
    assert hits[0].keyframe_available is False
    assert [hit.rank for hit in hits] == [1, 2, 3]
    # Equal scores use the stable FAISS_Index secondary ordering.
    assert hits[1].score == hits[2].score


def test_search_batch_encodes_once_and_returns_flat_query_tagged_hits(
    tmp_path: Path,
) -> None:
    index_path, mapping_path = _write_store(tmp_path)
    store = Phase1ClipStore(index_path, mapping_path)
    encoder = _Encoder(
        np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
            dtype=np.float32,
        )
    )
    kis = TextualKIS(store, encoder)

    hits = kis.search_batch(
        [Query("q-a", "first"), Query("q-b", "second")], top_k=2
    )

    assert encoder.calls == [["first", "second"]]
    assert [(hit.query_id, hit.rank) for hit in hits] == [
        ("q-a", 1),
        ("q-a", 2),
        ("q-b", 1),
        ("q-b", 2),
    ]


@pytest.mark.parametrize(
    "bad_vector, message",
    [
        (np.zeros(3, dtype=np.float32), "non-zero"),
        (np.asarray([1.0, np.nan, 0.0]), "NaN or infinite"),
        (np.asarray([1.0, 0.0]), "dimension mismatch"),
        (np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), "must have shape"),
    ],
)
def test_store_rejects_bad_query_vectors(
    tmp_path: Path, bad_vector: np.ndarray, message: str
) -> None:
    index_path, mapping_path = _write_store(tmp_path)
    store = Phase1ClipStore(index_path, mapping_path)

    with pytest.raises(ValueError, match=message):
        store.search(bad_vector, top_k=2)


def test_encoder_output_shape_is_validated(tmp_path: Path) -> None:
    index_path, mapping_path = _write_store(tmp_path)
    store = Phase1ClipStore(index_path, mapping_path)
    kis = TextualKIS(
        store,
        _Encoder(np.asarray([1.0, 0.0, 0.0], dtype=np.float32)),
    )

    with pytest.raises(ValueError, match="output shape mismatch"):
        kis.search(Query("q", "text"))


def test_store_validates_count_dimension_and_unique_faiss_ids(tmp_path: Path) -> None:
    index_path, mapping_path = _write_store(tmp_path, faiss_ids=[3, 0, 4, 0, 2])
    with pytest.raises(ArtifactValidationError, match="Duplicate FAISS_Index 0"):
        Phase1ClipStore(index_path, mapping_path)

    valid_index, valid_mapping = _write_store(tmp_path / "valid")
    with pytest.raises(ArtifactValidationError, match="dimension mismatch"):
        Phase1ClipStore(valid_index, valid_mapping, expected_dimension=512)

    table = pq.read_table(valid_mapping).slice(0, 4)
    short_mapping = tmp_path / "short.parquet"
    pq.write_table(table, short_mapping)
    with pytest.raises(ArtifactValidationError, match="row-count mismatch"):
        Phase1ClipStore(valid_index, short_mapping)
