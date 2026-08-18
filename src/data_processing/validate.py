from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import faiss
import numpy as np
import pyarrow.parquet as pq

from .build import MAPPING_COLUMNS


def validate_artifacts(artifact_dir: Path | str) -> dict:
    root = Path(artifact_dir).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, object] = {}

    mapping_path = root / "catalog" / "frame_mapping.parquet"
    mapping = pq.read_table(mapping_path)
    checks["mapping_rows"] = mapping.num_rows
    missing_columns = [name for name in MAPPING_COLUMNS if name not in mapping.column_names]
    if missing_columns:
        errors.append(f"Mapping missing required columns: {missing_columns}")
    records = mapping.select(
        ["Video_ID", "Keyframe_Index", "CLIP_Index", "FAISS_Index", "Video_Frame_ID"]
    ).to_pylist()
    keys = [(row["Video_ID"], row["Keyframe_Index"]) for row in records]
    if len(keys) != len(set(keys)):
        errors.append("Mapping primary key (Video_ID, Keyframe_Index) is not unique")
    if keys != sorted(keys):
        errors.append("Mapping is not sorted by Video_ID, Keyframe_Index")
    faiss_ids = [row["FAISS_Index"] for row in records]
    if faiss_ids != list(range(len(records))):
        errors.append("FAISS_Index is not contiguous 0..N-1")
    expected_local: dict[str, int] = {}
    for row in records:
        expected = expected_local.get(row["Video_ID"], 0)
        if row["CLIP_Index"] != expected:
            errors.append(
                f"{row['Video_ID']}: CLIP_Index {row['CLIP_Index']} != expected {expected}"
            )
            break
        expected_local[row["Video_ID"]] = expected + 1
    duplicate_submit_keys = len(records) - len(
        {(row["Video_ID"], row["Video_Frame_ID"]) for row in records}
    )
    checks["duplicate_video_frame_id_extra_rows"] = duplicate_submit_keys
    if duplicate_submit_keys:
        warnings.append(
            f"{duplicate_submit_keys} duplicate (Video_ID, Video_Frame_ID) rows are expected; "
            "use (Video_ID, Keyframe_Index) as the catalog key"
        )

    vector_path = root / "clip" / "clip_vectors.f32.npy"
    index_path = root / "clip" / "faiss.index"
    vectors = np.load(vector_path, mmap_mode="r")
    index = faiss.read_index(str(index_path))
    checks["clip_vector_shape"] = list(vectors.shape)
    checks["faiss_ntotal"] = int(index.ntotal)
    checks["faiss_dimension"] = int(index.d)
    if vectors.ndim != 2 or vectors.shape[0] != mapping.num_rows:
        errors.append(f"CLIP vector shape {vectors.shape} does not match mapping")
    if index.ntotal != mapping.num_rows or index.d != vectors.shape[1]:
        errors.append("FAISS dimensions/count do not match mapping and vector matrix")
    sample_indices = np.linspace(0, max(0, len(vectors) - 1), min(1000, len(vectors)), dtype=int)
    if len(sample_indices):
        norms = np.linalg.norm(np.asarray(vectors[sample_indices]), axis=1)
        checks["sample_norm_min"] = float(norms.min())
        checks["sample_norm_max"] = float(norms.max())
        if not np.allclose(norms, 1.0, atol=1e-5):
            errors.append("Stored CLIP vectors are not L2 normalized")
        reconstruction_failures = []
        for sample_index in sample_indices[:: max(1, len(sample_indices) // 25)]:
            reconstructed = index.reconstruct(int(sample_index))
            if not np.allclose(
                reconstructed, np.asarray(vectors[int(sample_index)]), atol=1e-6
            ):
                reconstruction_failures.append(int(sample_index))
        checks["faiss_reconstruction_samples"] = min(26, len(sample_indices))
        if reconstruction_failures:
            errors.append(
                f"FAISS reconstruction differs from stored vectors at {reconstruction_failures[:10]}"
            )

    offsets_path = root / "clip" / "video_offsets.parquet"
    offsets = pq.read_table(offsets_path).to_pylist()
    expected_cursor = 0
    offset_errors = []
    per_video_counts: dict[str, int] = {}
    for row in records:
        per_video_counts[row["Video_ID"]] = per_video_counts.get(row["Video_ID"], 0) + 1
    for offset in offsets:
        rows = per_video_counts.get(offset["Video_ID"])
        if (
            rows != offset["Rows"]
            or offset["FAISS_Start"] != expected_cursor
            or offset["FAISS_Stop"] != expected_cursor + (rows or 0)
        ):
            offset_errors.append(offset["Video_ID"])
        expected_cursor = offset["FAISS_Stop"]
    checks["clip_offset_videos"] = len(offsets)
    if offset_errors or expected_cursor != mapping.num_rows:
        errors.append(f"CLIP video offsets mismatch at {offset_errors[:10]}")

    metadata_db = root / "metadata" / "metadata.sqlite"
    connection = sqlite3.connect(f"file:{metadata_db}?mode=ro", uri=True)
    try:
        metadata_rows = connection.execute("SELECT count(*) FROM metadata").fetchone()[0]
        fts_rows = connection.execute("SELECT count(*) FROM metadata_fts").fetchone()[0]
    finally:
        connection.close()
    checks["metadata_rows"] = metadata_rows
    checks["metadata_fts_rows"] = fts_rows
    if metadata_rows != fts_rows:
        errors.append("Metadata and FTS5 document counts differ")

    object_raw = pq.ParquetFile(root / "objects" / "objects_raw_nested.parquet")
    object_flat = pq.ParquetFile(root / "objects" / "objects_index.parquet")
    object_db = root / "objects" / "objects.sqlite"
    connection = sqlite3.connect(f"file:{object_db}?mode=ro", uri=True)
    try:
        object_db_rows = connection.execute("SELECT count(*) FROM detections").fetchone()[0]
    finally:
        connection.close()
    checks["object_frames"] = object_raw.metadata.num_rows
    checks["object_flat_rows"] = object_flat.metadata.num_rows
    checks["object_sqlite_rows"] = object_db_rows
    if object_raw.metadata.num_rows != mapping.num_rows:
        errors.append("Raw object frame rows do not match canonical mapping")
    if object_flat.metadata.num_rows != object_db_rows:
        errors.append("Filtered object Parquet and SQLite row counts differ")

    report = {
        "valid": not errors,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
    }
    (root / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
