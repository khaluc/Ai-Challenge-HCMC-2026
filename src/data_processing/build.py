from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
from zipfile import ZipFile

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from .layout import (
    DatasetLayout,
    archive_ref,
    indexed_members,
    logical_keyframe_member,
    logical_object_member,
    logical_video_member,
    video_members,
)
from .source import (
    FrameMapRow,
    load_all_frame_maps,
    load_all_metadata,
    load_clip_array,
    zip_members_by_video,
)


MAPPING_COLUMNS = (
    "Video_ID",
    "Keyframe_Index",
    "Keyframe_Path",
    "Video_Frame_ID",
    "Timestamp",
    "CLIP_Index",
    "Object_Path",
)


@dataclass
class BuildContext:
    layout: DatasetLayout
    output_dir: Path
    frame_maps: dict[str, list[FrameMapRow]]
    map_members: dict[str, str]
    clip_members: dict[str, str]
    metadata_members: dict[str, str]
    object_members: dict[str, dict[int, tuple[Path, str]]]
    keyframe_members: dict[str, dict[int, tuple[Path, str]]]
    source_videos: dict[str, tuple[Path, str]]
    frame_lookup: dict[tuple[str, int], FrameMapRow]

    @classmethod
    def create(cls, layout: DatasetLayout, output_dir: Path | str) -> "BuildContext":
        frame_maps, map_members = load_all_frame_maps(layout)
        clip_members = zip_members_by_video(layout.clip_archive, ".npy")
        metadata_members = zip_members_by_video(layout.metadata_archive, ".json")
        object_members = indexed_members((layout.objects_archive,), ".json")
        keyframe_members = indexed_members(layout.keyframe_archives, ".jpg")
        source_videos = video_members(layout.video_archives)
        core_ids = set(frame_maps)
        source_ids = {
            "CLIP": set(clip_members),
            "metadata": set(metadata_members),
            "objects": set(object_members),
        }
        for label, ids in source_ids.items():
            missing = sorted(core_ids - ids)
            extra = sorted(ids - core_ids)
            if missing or extra:
                raise ValueError(
                    f"{label} video IDs do not match maps: missing={missing[:10]}, extra={extra[:10]}"
                )
        for video_id, rows in frame_maps.items():
            expected = {row.keyframe_index for row in rows}
            actual = set(object_members[video_id])
            if expected != actual:
                raise ValueError(
                    f"{video_id}: object frame IDs do not match map; "
                    f"missing={sorted(expected-actual)[:10]}, extra={sorted(actual-expected)[:10]}"
                )
        frame_lookup = {
            (video_id, row.keyframe_index): row
            for video_id, rows in frame_maps.items()
            for row in rows
        }
        return cls(
            layout=layout,
            output_dir=Path(output_dir).resolve(),
            frame_maps=frame_maps,
            map_members=map_members,
            clip_members=clip_members,
            metadata_members=metadata_members,
            object_members=object_members,
            keyframe_members=keyframe_members,
            source_videos=source_videos,
            frame_lookup=frame_lookup,
        )

    @property
    def ordered_video_ids(self) -> list[str]:
        return sorted(self.frame_maps)

    @property
    def total_frames(self) -> int:
        return sum(len(rows) for rows in self.frame_maps.values())


def _replace_file(tmp_path: Path, final_path: Path) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp_path, final_path)


def _remove_stale_tmp(path: Path) -> None:
    if path.exists():
        path.unlink()


def _write_json_atomic(path: Path, value: object) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    _remove_stale_tmp(tmp_path)
    tmp_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _replace_file(tmp_path, path)


def _arrow_mapping_table(context: BuildContext) -> pa.Table:
    columns: dict[str, list] = {
        "Video_ID": [],
        "Keyframe_Index": [],
        "Keyframe_Path": [],
        "Video_Frame_ID": [],
        "Timestamp": [],
        "CLIP_Index": [],
        "Object_Path": [],
        "FAISS_Index": [],
        "FPS": [],
        "Keyframe_Available": [],
        "Object_Available": [],
        "Video_Available": [],
        "Keyframe_Archive": [],
        "Object_Archive": [],
        "Video_Path": [],
        "CLIP_Path": [],
        "Map_Path": [],
        "Metadata_Path": [],
    }
    faiss_index = 0
    for video_id in context.ordered_video_ids:
        clip_member = context.clip_members.get(video_id)
        if clip_member is None:
            raise ValueError(f"Missing CLIP feature file for {video_id}")
        metadata_member = context.metadata_members.get(video_id)
        if metadata_member is None:
            raise ValueError(f"Missing metadata file for {video_id}")
        map_member = context.map_members[video_id]
        video_source = context.source_videos.get(video_id)
        for local_index, row in enumerate(context.frame_maps[video_id]):
            if row.keyframe_index != local_index + 1:
                raise ValueError(
                    f"{video_id}: keyframe n={row.keyframe_index} at CLIP row {local_index}"
                )
            keyframe_source = context.keyframe_members.get(video_id, {}).get(row.keyframe_index)
            object_source = context.object_members.get(video_id, {}).get(row.keyframe_index)
            expected_keyframe = logical_keyframe_member(video_id, row.keyframe_index)
            expected_object = logical_object_member(video_id, row.keyframe_index)

            columns["Video_ID"].append(video_id)
            columns["Keyframe_Index"].append(row.keyframe_index)
            columns["Keyframe_Path"].append(
                archive_ref(
                    keyframe_source[0] if keyframe_source else None,
                    keyframe_source[1] if keyframe_source else expected_keyframe,
                )
            )
            columns["Video_Frame_ID"].append(row.video_frame_id)
            columns["Timestamp"].append(row.timestamp)
            columns["CLIP_Index"].append(local_index)
            columns["Object_Path"].append(
                archive_ref(
                    object_source[0] if object_source else None,
                    object_source[1] if object_source else expected_object,
                )
            )
            columns["FAISS_Index"].append(faiss_index)
            columns["FPS"].append(row.fps)
            columns["Keyframe_Available"].append(keyframe_source is not None)
            columns["Object_Available"].append(object_source is not None)
            columns["Video_Available"].append(video_source is not None)
            columns["Keyframe_Archive"].append(
                keyframe_source[0].name if keyframe_source else None
            )
            columns["Object_Archive"].append(
                object_source[0].name if object_source else None
            )
            columns["Video_Path"].append(
                archive_ref(
                    video_source[0] if video_source else None,
                    video_source[1] if video_source else logical_video_member(video_id),
                )
            )
            columns["CLIP_Path"].append(
                archive_ref(context.layout.clip_archive, clip_member)
            )
            columns["Map_Path"].append(
                archive_ref(context.layout.map_archive, map_member)
            )
            columns["Metadata_Path"].append(
                archive_ref(context.layout.metadata_archive, metadata_member)
            )
            faiss_index += 1

    schema = pa.schema(
        [
            ("Video_ID", pa.string()),
            ("Keyframe_Index", pa.int32()),
            ("Keyframe_Path", pa.string()),
            ("Video_Frame_ID", pa.int64()),
            ("Timestamp", pa.float64()),
            ("CLIP_Index", pa.int32()),
            ("Object_Path", pa.string()),
            ("FAISS_Index", pa.int64()),
            ("FPS", pa.float32()),
            ("Keyframe_Available", pa.bool_()),
            ("Object_Available", pa.bool_()),
            ("Video_Available", pa.bool_()),
            ("Keyframe_Archive", pa.string()),
            ("Object_Archive", pa.string()),
            ("Video_Path", pa.string()),
            ("CLIP_Path", pa.string()),
            ("Map_Path", pa.string()),
            ("Metadata_Path", pa.string()),
        ]
    )
    return pa.Table.from_pydict(columns, schema=schema)


def build_mapping(context: BuildContext) -> dict:
    target_dir = context.output_dir / "catalog"
    target_dir.mkdir(parents=True, exist_ok=True)
    table = _arrow_mapping_table(context)
    parquet_path = target_dir / "frame_mapping.parquet"
    csv_path = target_dir / "frame_mapping.csv"
    parquet_tmp = parquet_path.with_suffix(".parquet.tmp")
    csv_tmp = csv_path.with_suffix(".csv.tmp")
    _remove_stale_tmp(parquet_tmp)
    _remove_stale_tmp(csv_tmp)
    pq.write_table(table, parquet_tmp, compression="zstd", use_dictionary=True)
    pacsv.write_csv(table, csv_tmp)
    _replace_file(parquet_tmp, parquet_path)
    _replace_file(csv_tmp, csv_path)
    required = list(MAPPING_COLUMNS)
    if table.column_names[: len(required)] != required:
        raise AssertionError("Canonical mapping columns are not in the required order")
    return {
        "rows": table.num_rows,
        "videos": len(context.frame_maps),
        "parquet": str(parquet_path),
        "csv": str(csv_path),
        "keyframes_available": int(
            pa.compute.sum(table.column("Keyframe_Available")).as_py() or 0
        ),
        "videos_available": len(context.source_videos),
    }


def build_clip_index(context: BuildContext, batch_size: int = 16_384) -> dict:
    target_dir = context.output_dir / "clip"
    target_dir.mkdir(parents=True, exist_ok=True)
    vector_path = target_dir / "clip_vectors.f32.npy"
    vector_tmp = target_dir / "clip_vectors.f32.tmp.npy"
    index_path = target_dir / "faiss.index"
    index_tmp = target_dir / "faiss.index.tmp"
    offsets_path = target_dir / "video_offsets.parquet"
    offsets_tmp = target_dir / "video_offsets.parquet.tmp"
    for path in (vector_tmp, index_tmp, offsets_tmp):
        _remove_stale_tmp(path)

    dimension: int | None = None
    source_dtypes: set[str] = set()
    offsets: dict[str, list] = {
        "Video_ID": [],
        "FAISS_Start": [],
        "FAISS_Stop": [],
        "Rows": [],
    }
    vectors: np.memmap | None = None
    index: faiss.Index | None = None
    cursor = 0
    with ZipFile(context.layout.clip_archive) as handle:
        for number, video_id in enumerate(context.ordered_video_ids, start=1):
            member = context.clip_members.get(video_id)
            if member is None:
                raise ValueError(f"Missing CLIP feature file for {video_id}")
            source_raw = load_clip_array(handle, member)
            source_dtypes.add(str(source_raw.dtype))
            source = source_raw.astype(np.float32, copy=False)
            expected_rows = len(context.frame_maps[video_id])
            if len(source) != expected_rows:
                raise ValueError(
                    f"{video_id}: feature rows {len(source)} != map rows {expected_rows}"
                )
            if dimension is None:
                dimension = int(source.shape[1])
                vectors = np.lib.format.open_memmap(
                    vector_tmp,
                    mode="w+",
                    dtype=np.float32,
                    shape=(context.total_frames, dimension),
                )
                index = faiss.IndexFlatIP(dimension)
            elif source.shape[1] != dimension:
                raise ValueError(
                    f"{video_id}: feature dimension {source.shape[1]} != {dimension}"
                )
            if not np.isfinite(source).all():
                raise ValueError(f"{video_id}: CLIP feature contains NaN or Inf")
            norms = np.linalg.norm(source, axis=1, keepdims=True)
            if np.any(norms <= 0):
                raise ValueError(f"{video_id}: CLIP feature contains a zero vector")
            source /= norms
            assert vectors is not None and index is not None
            stop = cursor + len(source)
            vectors[cursor:stop] = source
            for batch_start in range(0, len(source), batch_size):
                index.add(source[batch_start : batch_start + batch_size])
            offsets["Video_ID"].append(video_id)
            offsets["FAISS_Start"].append(cursor)
            offsets["FAISS_Stop"].append(stop)
            offsets["Rows"].append(len(source))
            cursor = stop
            if number % 100 == 0 or number == len(context.frame_maps):
                print(
                    f"CLIP: {number}/{len(context.frame_maps)} videos, {cursor:,} vectors",
                    flush=True,
                )

    if vectors is None or index is None or dimension is None:
        raise ValueError("No CLIP features found")
    vectors.flush()
    del vectors
    if cursor != context.total_frames or index.ntotal != context.total_frames:
        raise AssertionError(
            f"CLIP total mismatch: cursor={cursor}, faiss={index.ntotal}, maps={context.total_frames}"
        )
    faiss.write_index(index, str(index_tmp))
    offset_table = pa.Table.from_pydict(
        offsets,
        schema=pa.schema(
            [
                ("Video_ID", pa.string()),
                ("FAISS_Start", pa.int64()),
                ("FAISS_Stop", pa.int64()),
                ("Rows", pa.int32()),
            ]
        ),
    )
    pq.write_table(offset_table, offsets_tmp, compression="zstd")
    _replace_file(vector_tmp, vector_path)
    _replace_file(index_tmp, index_path)
    _replace_file(offsets_tmp, offsets_path)
    meta = {
        "index_type": "IndexFlatIP",
        "similarity": "cosine (inner product over L2-normalized float32 vectors)",
        "dimension": dimension,
        "vectors": cursor,
        "ordering": "frame_mapping.parquet FAISS_Index",
        "source_dtypes": sorted(source_dtypes),
        "stored_dtype": "float32",
        "source_encoder": "clip-ViT-B-32 (from competition feature archive)",
        "vector_file": vector_path.name,
        "index_file": index_path.name,
    }
    _write_json_atomic(target_dir / "clip_index_meta.json", meta)
    return meta


def fold_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    without_marks = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return without_marks.replace("Đ", "D").replace("đ", "d").casefold()


def _metadata_record(video_id: str, item: dict) -> dict:
    raw_keywords = item.get("keywords") or []
    if isinstance(raw_keywords, str):
        keyword_list = [raw_keywords]
    else:
        keyword_list = [str(value) for value in raw_keywords]
    keywords = " ".join(keyword_list)
    title = str(item.get("title") or "")
    description = str(item.get("description") or "")
    author = str(item.get("author") or "")
    folded = fold_text(" ".join((title, description, keywords, author)))
    length = item.get("length")
    return {
        "video_id": video_id,
        "title": title,
        "description": description,
        "keywords": keywords,
        "keywords_json": json.dumps(keyword_list, ensure_ascii=False),
        "author": author,
        "channel_id": str(item.get("channel_id") or ""),
        "channel_url": str(item.get("channel_url") or ""),
        "length_seconds": int(length) if length is not None else None,
        "publish_date": str(item.get("publish_date") or ""),
        "thumbnail_url": str(item.get("thumbnail_url") or ""),
        "watch_url": str(item.get("watch_url") or ""),
        "text_folded": folded,
    }


def build_metadata_index(context: BuildContext) -> dict:
    target_dir = context.output_dir / "metadata"
    target_dir.mkdir(parents=True, exist_ok=True)
    database_path = target_dir / "metadata.sqlite"
    database_tmp = target_dir / "metadata.sqlite.tmp"
    parquet_path = target_dir / "metadata.parquet"
    parquet_tmp = target_dir / "metadata.parquet.tmp"
    for path in (database_tmp, parquet_tmp):
        _remove_stale_tmp(path)

    metadata, _ = load_all_metadata(context.layout)
    records = [_metadata_record(video_id, metadata[video_id]) for video_id in sorted(metadata)]
    metadata_schema = pa.schema(
        [
            ("video_id", pa.string()),
            ("title", pa.string()),
            ("description", pa.string()),
            ("keywords", pa.string()),
            ("keywords_json", pa.string()),
            ("author", pa.string()),
            ("channel_id", pa.string()),
            ("channel_url", pa.string()),
            ("length_seconds", pa.int64()),
            ("publish_date", pa.string()),
            ("thumbnail_url", pa.string()),
            ("watch_url", pa.string()),
            ("text_folded", pa.string()),
        ]
    )
    table = pa.Table.from_pylist(records, schema=metadata_schema)
    pq.write_table(table, parquet_tmp, compression="zstd", use_dictionary=True)

    connection = sqlite3.connect(database_tmp)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE metadata (
                video_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                keywords TEXT NOT NULL,
                keywords_json TEXT NOT NULL,
                author TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                channel_url TEXT NOT NULL,
                length_seconds INTEGER,
                publish_date TEXT NOT NULL,
                thumbnail_url TEXT NOT NULL,
                watch_url TEXT NOT NULL,
                text_folded TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE metadata_fts USING fts5(
                video_id UNINDEXED,
                title,
                description,
                keywords,
                author,
                text_folded,
                tokenize='unicode61 remove_diacritics 2',
                prefix='2 3 4'
            );
            """
        )
        columns = tuple(records[0]) if records else ()
        placeholders = ",".join("?" for _ in columns)
        connection.executemany(
            f"INSERT INTO metadata ({','.join(columns)}) VALUES ({placeholders})",
            ([record[column] for column in columns] for record in records),
        )
        connection.executemany(
            """
            INSERT INTO metadata_fts
                (video_id, title, description, keywords, author, text_folded)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    record["video_id"],
                    record["title"],
                    record["description"],
                    record["keywords"],
                    record["author"],
                    record["text_folded"],
                )
                for record in records
            ),
        )
        connection.commit()
        fts_rows = connection.execute("SELECT count(*) FROM metadata_fts").fetchone()[0]
    finally:
        connection.close()
    if fts_rows != len(records):
        raise AssertionError(f"FTS documents {fts_rows} != metadata documents {len(records)}")
    _replace_file(database_tmp, database_path)
    _replace_file(parquet_tmp, parquet_path)
    meta = {
        "documents": len(records),
        "engine": "SQLite FTS5",
        "ranking": "bm25; lower scores rank first",
        "database": str(database_path),
        "parquet": str(parquet_path),
        "transcript_or_asr_available": False,
        "empty_descriptions": sum(not record["description"].strip() for record in records),
        "empty_keywords": sum(not record["keywords"].strip() for record in records),
    }
    _write_json_atomic(target_dir / "metadata_index_meta.json", meta)
    return meta


RAW_OBJECT_SCHEMA = pa.schema(
    [
        ("Video_ID", pa.string()),
        ("Keyframe_Index", pa.int32()),
        ("Video_Frame_ID", pa.int64()),
        ("Timestamp", pa.float64()),
        ("Object_Path", pa.string()),
        ("Object_Class", pa.list_(pa.string())),
        ("Class_MID", pa.list_(pa.string())),
        ("Class_Label", pa.list_(pa.int16())),
        ("Confidence", pa.list_(pa.float32())),
        ("Bounding_Boxes_YMin_XMin_YMax_XMax", pa.list_(pa.list_(pa.float32(), 4))),
    ]
)


FLAT_OBJECT_SCHEMA = pa.schema(
    [
        ("Video_ID", pa.string()),
        ("Keyframe_Index", pa.int32()),
        ("Video_Frame_ID", pa.int64()),
        ("Timestamp", pa.float64()),
        ("Detection_Rank", pa.int16()),
        ("Object_Class", pa.string()),
        ("Class_MID", pa.string()),
        ("Class_Label", pa.int16()),
        ("Confidence", pa.float32()),
        ("YMin", pa.float32()),
        ("XMin", pa.float32()),
        ("YMax", pa.float32()),
        ("XMax", pa.float32()),
        ("Object_Path", pa.string()),
    ]
)


def _empty_object_batch() -> dict[str, list]:
    return {field.name: [] for field in RAW_OBJECT_SCHEMA}


def _empty_flat_batch() -> dict[str, list]:
    return {field.name: [] for field in FLAT_OBJECT_SCHEMA}


def _write_batch(writer: pq.ParquetWriter, values: dict[str, list], schema: pa.Schema) -> None:
    if not values[schema.names[0]]:
        return
    writer.write_table(pa.Table.from_pydict(values, schema=schema))


def build_object_store(
    context: BuildContext,
    index_min_confidence: float = 0.2,
    frame_batch_size: int = 1_000,
) -> dict:
    if not 0.0 <= index_min_confidence <= 1.0:
        raise ValueError("index_min_confidence must be between 0 and 1")
    if frame_batch_size <= 0:
        raise ValueError("frame_batch_size must be positive")
    target_dir = context.output_dir / "objects"
    target_dir.mkdir(parents=True, exist_ok=True)
    raw_path = target_dir / "objects_raw_nested.parquet"
    flat_path = target_dir / "objects_index.parquet"
    database_path = target_dir / "objects.sqlite"
    raw_tmp = target_dir / "objects_raw_nested.parquet.tmp"
    flat_tmp = target_dir / "objects_index.parquet.tmp"
    database_tmp = target_dir / "objects.sqlite.tmp"
    for path in (raw_tmp, flat_tmp, database_tmp):
        _remove_stale_tmp(path)

    raw_writer: pq.ParquetWriter | None = None
    flat_writer: pq.ParquetWriter | None = None
    connection: sqlite3.Connection | None = None
    raw_batch = _empty_object_batch()
    flat_batch = _empty_flat_batch()
    sqlite_batch: list[tuple] = []
    raw_detection_count = 0
    indexed_detection_count = 0
    processed_frames = 0
    expected_frames = context.total_frames
    ordered_members = [
        (video_id, keyframe_index, source)
        for video_id in context.ordered_video_ids
        for keyframe_index, source in sorted(context.object_members.get(video_id, {}).items())
    ]
    try:
        raw_writer = pq.ParquetWriter(
            raw_tmp, RAW_OBJECT_SCHEMA, compression="zstd", use_dictionary=True
        )
        flat_writer = pq.ParquetWriter(
            flat_tmp, FLAT_OBJECT_SCHEMA, compression="zstd", use_dictionary=True
        )
        connection = sqlite3.connect(database_tmp)
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE detections (
                video_id TEXT NOT NULL,
                keyframe_index INTEGER NOT NULL,
                video_frame_id INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                detection_rank INTEGER NOT NULL,
                object_class TEXT COLLATE NOCASE NOT NULL,
                class_mid TEXT NOT NULL,
                class_label INTEGER NOT NULL,
                confidence REAL NOT NULL,
                ymin REAL NOT NULL,
                xmin REAL NOT NULL,
                ymax REAL NOT NULL,
                xmax REAL NOT NULL,
                object_path TEXT NOT NULL
            );
            """
        )
        with ZipFile(context.layout.objects_archive) as handle:
            for video_id, keyframe_index, (archive, member) in ordered_members:
                if archive != context.layout.objects_archive:
                    raise ValueError(f"Unexpected object archive for {member}: {archive}")
                map_row = context.frame_lookup.get((video_id, keyframe_index))
                if map_row is None:
                    raise ValueError(f"Object file has no frame mapping: {member}")
                payload = json.loads(handle.read(member))
                scores_raw = payload.get("detection_scores")
                mids_raw = payload.get("detection_class_names")
                classes_raw = payload.get("detection_class_entities")
                boxes_raw = payload.get("detection_boxes")
                labels_raw = payload.get("detection_class_labels")
                arrays = (scores_raw, mids_raw, classes_raw, boxes_raw, labels_raw)
                if not all(isinstance(value, list) for value in arrays):
                    raise ValueError(f"{member}: missing or non-list detection arrays")
                lengths = {len(value) for value in arrays}
                if len(lengths) != 1:
                    raise ValueError(f"{member}: parallel object arrays have lengths {lengths}")
                scores = [float(value) for value in scores_raw]
                mids = [str(value) for value in mids_raw]
                classes = [str(value) for value in classes_raw]
                labels = [int(value) for value in labels_raw]
                boxes = [[float(coord) for coord in box] for box in boxes_raw]
                if any(len(box) != 4 or not all(math.isfinite(value) for value in box) for box in boxes):
                    raise ValueError(f"{member}: invalid bounding box; expected four finite values")
                if any(not math.isfinite(score) or not 0.0 <= score <= 1.0 for score in scores):
                    raise ValueError(f"{member}: confidence must be finite and in [0, 1]")
                if any(
                    not (
                        0.0 <= box[0] <= box[2] <= 1.0
                        and 0.0 <= box[1] <= box[3] <= 1.0
                    )
                    for box in boxes
                ):
                    raise ValueError(
                        f"{member}: bounding box must be normalized [ymin,xmin,ymax,xmax]"
                    )
                object_path = archive_ref(archive, member.replace("\\", "/"))

                raw_batch["Video_ID"].append(video_id)
                raw_batch["Keyframe_Index"].append(keyframe_index)
                raw_batch["Video_Frame_ID"].append(map_row.video_frame_id)
                raw_batch["Timestamp"].append(map_row.timestamp)
                raw_batch["Object_Path"].append(object_path)
                raw_batch["Object_Class"].append(classes)
                raw_batch["Class_MID"].append(mids)
                raw_batch["Class_Label"].append(labels)
                raw_batch["Confidence"].append(scores)
                raw_batch["Bounding_Boxes_YMin_XMin_YMax_XMax"].append(boxes)
                raw_detection_count += len(scores)

                for rank, (score, mid, object_class, label, box) in enumerate(
                    zip(scores, mids, classes, labels, boxes), start=1
                ):
                    if score < index_min_confidence:
                        continue
                    values = (
                        video_id,
                        keyframe_index,
                        map_row.video_frame_id,
                        map_row.timestamp,
                        rank,
                        object_class,
                        mid,
                        label,
                        score,
                        box[0],
                        box[1],
                        box[2],
                        box[3],
                        object_path,
                    )
                    for column, value in zip(FLAT_OBJECT_SCHEMA.names, values):
                        flat_batch[column].append(value)
                    sqlite_batch.append(values)
                    indexed_detection_count += 1

                processed_frames += 1
                if len(raw_batch["Video_ID"]) >= frame_batch_size:
                    _write_batch(raw_writer, raw_batch, RAW_OBJECT_SCHEMA)
                    raw_batch = _empty_object_batch()
                if len(flat_batch["Video_ID"]) >= frame_batch_size * 4:
                    _write_batch(flat_writer, flat_batch, FLAT_OBJECT_SCHEMA)
                    flat_batch = _empty_flat_batch()
                if len(sqlite_batch) >= frame_batch_size * 4:
                    connection.executemany(
                        "INSERT INTO detections VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        sqlite_batch,
                    )
                    sqlite_batch.clear()
                if processed_frames % 10_000 == 0 or processed_frames == len(ordered_members):
                    print(
                        f"Objects: {processed_frames:,}/{len(ordered_members):,} frames, "
                        f"{raw_detection_count:,} raw, {indexed_detection_count:,} indexed",
                        flush=True,
                    )
        _write_batch(raw_writer, raw_batch, RAW_OBJECT_SCHEMA)
        _write_batch(flat_writer, flat_batch, FLAT_OBJECT_SCHEMA)
        if sqlite_batch:
            connection.executemany(
                "INSERT INTO detections VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", sqlite_batch
            )
        connection.commit()
        connection.executescript(
            """
            CREATE INDEX idx_detections_class_confidence
                ON detections(object_class, confidence DESC);
            CREATE INDEX idx_detections_frame
                ON detections(video_id, keyframe_index);
            CREATE INDEX idx_detections_submit_frame
                ON detections(video_id, video_frame_id);
            """
        )
        connection.commit()
        sqlite_rows = connection.execute("SELECT count(*) FROM detections").fetchone()[0]
    finally:
        close_errors: list[Exception] = []
        for resource in (raw_writer, flat_writer, connection):
            if resource is None:
                continue
            try:
                resource.close()
            except Exception as exc:  # keep closing the remaining Windows handles
                close_errors.append(exc)
        if close_errors:
            raise close_errors[0]

    if processed_frames != expected_frames:
        raise ValueError(
            f"Object files {processed_frames:,} != mapped keyframes {expected_frames:,}"
        )
    if sqlite_rows != indexed_detection_count:
        raise AssertionError(
            f"Object SQLite rows {sqlite_rows} != indexed detections {indexed_detection_count}"
        )
    _replace_file(raw_tmp, raw_path)
    _replace_file(flat_tmp, flat_path)
    _replace_file(database_tmp, database_path)
    meta = {
        "frames": processed_frames,
        "raw_detections": raw_detection_count,
        "indexed_detections": indexed_detection_count,
        "index_min_confidence": index_min_confidence,
        "raw_lossless_parquet": str(raw_path),
        "filtered_flat_parquet": str(flat_path),
        "query_database": str(database_path),
        "bounding_box_order": ["ymin", "xmin", "ymax", "xmax"],
        "coordinates": "normalized to [0, 1] as supplied by the detector",
        "note": "Raw nested Parquet retains every supplied detection; flat Parquet and SQLite apply index_min_confidence.",
    }
    _write_json_atomic(target_dir / "object_store_meta.json", meta)
    return meta


def build_all(
    layout: DatasetLayout,
    output_dir: Path | str,
    components: Sequence[str] = ("mapping", "clip", "metadata", "objects"),
    object_index_min_confidence: float = 0.2,
) -> dict:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    context = BuildContext.create(layout, output)
    unknown = set(components) - {"mapping", "clip", "metadata", "objects"}
    if unknown:
        raise ValueError(f"Unknown build components: {sorted(unknown)}")
    started = time.perf_counter()
    result: dict[str, dict] = {}
    if "mapping" in components:
        print("Building canonical frame mapping...", flush=True)
        result["mapping"] = build_mapping(context)
    if "clip" in components:
        print("Building normalized global CLIP/FAISS index...", flush=True)
        result["clip"] = build_clip_index(context)
    if "metadata" in components:
        print("Building metadata Parquet + SQLite FTS5/BM25 index...", flush=True)
        result["metadata"] = build_metadata_index(context)
    if "objects" in components:
        print("Building lossless object store + filtered query index...", flush=True)
        result["objects"] = build_object_store(
            context, index_min_confidence=object_index_min_confidence
        )
    elapsed_seconds = round(time.perf_counter() - started, 3)
    manifest_path = output / "build_manifest.json"
    merged_components: dict[str, dict] = {}
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                previous.get("source_root") == str(layout.root)
                and previous.get("output_root") == str(output)
            ):
                merged_components.update(
                    {
                        name: value
                        for name, value in previous.get("components", {}).items()
                        if name in {"mapping", "clip", "metadata", "objects"}
                        and isinstance(value, dict)
                    }
                )
        except (OSError, ValueError, TypeError):
            pass
    merged_components.update(result)
    manifest = {
        "version": 1,
        "source_root": str(layout.root),
        "output_root": str(output),
        "components": merged_components,
        "last_run_components": list(components),
        "last_run_elapsed_seconds": elapsed_seconds,
        "canonical_order": "Video_ID ascending, then Keyframe_Index ascending",
        "submission_rule": "Use Video_Frame_ID from catalog/frame_mapping.parquet, never Keyframe_Index.",
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest
