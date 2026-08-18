from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import faiss
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .build import fold_text


TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _fts_query(text: str) -> str:
    tokens = TOKEN_RE.findall(fold_text(text))
    if not tokens:
        raise ValueError("Metadata query must contain at least one letter or digit")
    # Quoting every token avoids exposing FTS5 operators from user input.
    return " AND ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)


def search_metadata(database: Path | str, query: str, limit: int = 20) -> list[dict]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    connection = sqlite3.connect(f"file:{Path(database).resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT
                m.video_id,
                m.title,
                m.author,
                m.publish_date,
                m.watch_url,
                bm25(metadata_fts, 0.0, 8.0, 3.0, 5.0, 2.0, 1.0) AS bm25_score
            FROM metadata_fts
            JOIN metadata AS m ON m.video_id = metadata_fts.video_id
            WHERE metadata_fts MATCH ?
            ORDER BY bm25_score ASC, m.video_id ASC
            LIMIT ?
            """,
            (_fts_query(query), int(limit)),
        ).fetchall()
    finally:
        connection.close()
    return [dict(row) for row in rows]


def search_objects(
    database: Path | str,
    object_class: str,
    min_confidence: float = 0.2,
    limit: int = 50,
    contains: bool = False,
) -> list[dict]:
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between 0 and 1")
    if limit <= 0:
        raise ValueError("limit must be positive")
    operator = "LIKE" if contains else "="
    value = f"%{object_class}%" if contains else object_class
    connection = sqlite3.connect(f"file:{Path(database).resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            f"""
            SELECT
                video_id,
                keyframe_index,
                video_frame_id,
                timestamp,
                object_class,
                MAX(confidence) AS confidence,
                COUNT(*) AS matching_detections,
                ymin,
                xmin,
                ymax,
                xmax,
                object_path
            FROM detections
            WHERE object_class {operator} ? COLLATE NOCASE
              AND confidence >= ?
            GROUP BY video_id, keyframe_index, object_class
            ORDER BY confidence DESC, video_id, keyframe_index
            LIMIT ?
            """,
            (value, float(min_confidence), int(limit)),
        ).fetchall()
    finally:
        connection.close()
    return [dict(row) for row in rows]


def search_similar_keyframe(
    artifact_dir: Path | str,
    video_id: str,
    keyframe_index: int,
    limit: int = 20,
    exclude_self: bool = True,
) -> list[dict]:
    root = Path(artifact_dir)
    mapping_path = root / "catalog" / "frame_mapping.parquet"
    vector_path = root / "clip" / "clip_vectors.f32.npy"
    index_path = root / "clip" / "faiss.index"
    mapping = pq.read_table(
        mapping_path,
        columns=[
            "Video_ID",
            "Keyframe_Index",
            "Video_Frame_ID",
            "Timestamp",
            "FAISS_Index",
            "Keyframe_Path",
            "Keyframe_Available",
        ],
    )
    mask = pc.and_(
        pc.equal(mapping["Video_ID"], video_id.upper()),
        pc.equal(mapping["Keyframe_Index"], int(keyframe_index)),
    )
    selected = mapping.filter(mask)
    if selected.num_rows != 1:
        raise KeyError(f"Expected one mapping row for {video_id}/{keyframe_index}, got {selected.num_rows}")
    query_index = int(selected["FAISS_Index"][0].as_py())
    vectors = np.load(vector_path, mmap_mode="r")
    index = faiss.read_index(str(index_path))
    requested = min(index.ntotal, limit + (1 if exclude_self else 0))
    scores, ids = index.search(np.asarray(vectors[query_index : query_index + 1]), requested)

    valid_pairs = [
        (int(faiss_id), float(score))
        for faiss_id, score in zip(ids[0], scores[0])
        if faiss_id >= 0 and not (exclude_self and int(faiss_id) == query_index)
    ][:limit]
    rows = mapping.to_pylist()
    by_faiss = {int(row["FAISS_Index"]): row for row in rows}
    result = []
    for faiss_id, score in valid_pairs:
        row = dict(by_faiss[faiss_id])
        row["Cosine_Similarity"] = score
        result.append(row)
    return result

