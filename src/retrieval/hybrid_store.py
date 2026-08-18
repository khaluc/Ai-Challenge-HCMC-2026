from __future__ import annotations

import json
import math
import numbers
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from retrieval.clip_store import FrameRecord, Phase1ClipStore

from .processing import ParsedObjectQuery


@dataclass(frozen=True)
class MetadataVideoHit:
    rank: int
    video_id: str
    bm25_score: float
    title: str
    author: str
    publish_date: str
    watch_url: str
    match_mode: str


@dataclass(frozen=True)
class ObjectFrameHit:
    rank: int
    frame: FrameRecord
    min_confidence: float
    mean_confidence: float
    matched_concepts: tuple[str, ...]


def _readonly_connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def load_object_labels(database: str | Path) -> tuple[str, ...]:
    path = Path(database)
    if not path.is_file():
        raise FileNotFoundError(f"Object database not found: {path}")
    with _readonly_connection(path) as connection:
        return tuple(
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT object_class FROM detections ORDER BY object_class"
            )
        )


def _normalize_query_vector(vector: np.ndarray, dimension: int) -> np.ndarray:
    values = np.asarray(vector)
    if values.ndim == 2 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 1 or values.shape[0] != dimension:
        raise ValueError(
            f"Query vector must have shape ({dimension},), got {values.shape}"
        )
    if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
        values.dtype, np.complexfloating
    ):
        raise TypeError("Query vector must contain real numbers")
    values = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError("Query vector contains non-finite values")
    norm = float(np.linalg.norm(values.astype(np.float64)))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Query vector must have a finite non-zero norm")
    return np.ascontiguousarray(values / np.float32(norm), dtype=np.float32)


class Phase1HybridStore:
    """Read-only metadata/object/vector view sharing the validated CLIP mapping."""

    def __init__(self, artifact_root: str | Path, clip_store: Phase1ClipStore) -> None:
        self.root = Path(artifact_root)
        self.clip_store = clip_store
        self.vector_path = self.root / "clip" / "clip_vectors.f32.npy"
        self.metadata_database = self.root / "metadata" / "metadata.sqlite"
        self.object_database = self.root / "objects" / "objects.sqlite"
        self.object_metadata = self.root / "objects" / "object_store_meta.json"
        for label, path in (
            ("CLIP vectors", self.vector_path),
            ("metadata database", self.metadata_database),
            ("object database", self.object_database),
            ("object store metadata", self.object_metadata),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Phase 1 {label} not found: {path}")

        try:
            object_meta = json.loads(self.object_metadata.read_text(encoding="utf-8"))
            self.object_index_min_confidence = float(
                object_meta["index_min_confidence"]
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid object store metadata: {self.object_metadata}"
            ) from exc
        if not 0 <= self.object_index_min_confidence <= 1:
            raise ValueError("Object index confidence floor must be in [0, 1]")

        self._vectors = np.load(self.vector_path, mmap_mode="r")
        if self._vectors.shape != (clip_store.size, clip_store.dimension):
            raise ValueError(
                "CLIP vector/index shape mismatch: "
                f"vectors={self._vectors.shape}, "
                f"index=({clip_store.size}, {clip_store.dimension})"
            )
        if not np.issubdtype(self._vectors.dtype, np.floating):
            raise ValueError(f"CLIP vectors must be floating point, got {self._vectors.dtype}")

        by_video: dict[str, list[int]] = {}
        by_keyframe: dict[tuple[str, int], FrameRecord] = {}
        for faiss_index in range(clip_store.size):
            frame = clip_store.frame_for_faiss_index(faiss_index)
            by_video.setdefault(frame.video_id, []).append(faiss_index)
            key = (frame.video_id, frame.keyframe_index)
            if key in by_keyframe:
                raise ValueError(f"Duplicate frame mapping key: {key}")
            by_keyframe[key] = frame
        self._faiss_by_video = {
            video_id: np.asarray(indices, dtype=np.int64)
            for video_id, indices in by_video.items()
        }
        self._frame_by_keyframe = by_keyframe

        self.object_labels = load_object_labels(self.object_database)
        with _readonly_connection(self.metadata_database) as connection:
            self.metadata_documents = int(
                connection.execute("SELECT count(*) FROM metadata").fetchone()[0]
            )
        if self.metadata_documents != len(self._faiss_by_video):
            raise ValueError(
                "Metadata/catalog video-count mismatch: "
                f"metadata={self.metadata_documents}, catalog={len(self._faiss_by_video)}"
            )

    def semantic_hits(self, vector: np.ndarray, limit: int) -> list[tuple[FrameRecord, float]]:
        return [
            (hit.frame, float(hit.score))
            for hit in self.clip_store.search(vector, top_k=limit)
        ]

    def best_frames_in_video(
        self,
        video_id: str,
        vector: np.ndarray,
        *,
        limit: int,
    ) -> list[tuple[FrameRecord, float]]:
        if isinstance(limit, bool) or not isinstance(limit, numbers.Integral) or limit < 1:
            raise ValueError("per-video frame limit must be a positive integer")
        indices = self._faiss_by_video.get(video_id)
        if indices is None or len(indices) == 0:
            return []
        query = _normalize_query_vector(vector, self.clip_store.dimension)
        matrix = np.asarray(self._vectors[indices], dtype=np.float32)
        scores = matrix @ query
        order = np.lexsort((indices, -scores))
        selected: list[tuple[FrameRecord, float]] = []
        seen: set[tuple[str, int]] = set()
        for local_position in order:
            faiss_index = int(indices[int(local_position)])
            frame = self.clip_store.frame_for_faiss_index(faiss_index)
            key = (frame.video_id, frame.video_frame_id)
            if key in seen:
                continue
            seen.add(key)
            selected.append((frame, float(scores[int(local_position)])))
            if len(selected) == int(limit):
                break
        return selected

    def search_metadata(
        self,
        terms: Sequence[str],
        *,
        limit: int,
    ) -> list[MetadataVideoHit]:
        if isinstance(limit, bool) or not isinstance(limit, numbers.Integral) or limit < 1:
            raise ValueError("metadata limit must be a positive integer")
        clean_terms = tuple(term for term in terms if term)
        if not clean_terms:
            return []
        quoted = tuple(
            f'"{term.replace(chr(34), chr(34) * 2)}"' for term in clean_terms
        )
        expressions = [("and", " AND ".join(quoted))]
        if len(quoted) > 1:
            expressions.append(("or", " OR ".join(quoted)))
        combined_rows: list[tuple[sqlite3.Row, str]] = []
        seen_video_ids: set[str] = set()
        with _readonly_connection(self.metadata_database) as connection:
            for match_mode, expression in expressions:
                rows = connection.execute(
                    """
                    SELECT
                        m.video_id,
                        m.title,
                        m.author,
                        m.publish_date,
                        m.watch_url,
                        bm25(metadata_fts, 0.0, 8.0, 3.0, 5.0, 2.0, 0.5)
                            AS bm25_score
                    FROM metadata_fts
                    JOIN metadata AS m ON m.video_id = metadata_fts.video_id
                    WHERE metadata_fts MATCH ?
                    ORDER BY bm25_score ASC, m.video_id ASC
                    LIMIT ?
                    """,
                    (expression, int(limit)),
                ).fetchall()
                for row in rows:
                    video_id = str(row["video_id"])
                    if video_id in seen_video_ids:
                        continue
                    seen_video_ids.add(video_id)
                    combined_rows.append((row, match_mode))
                    if len(combined_rows) == int(limit):
                        break
                if len(combined_rows) == int(limit):
                    break
        result: list[MetadataVideoHit] = []
        for rank, (row, match_mode) in enumerate(combined_rows, start=1):
            score = float(row["bm25_score"])
            if not math.isfinite(score):
                raise ValueError(f"Metadata BM25 returned non-finite score for {row['video_id']}")
            result.append(
                MetadataVideoHit(
                    rank=rank,
                    video_id=str(row["video_id"]),
                    bm25_score=score,
                    title=str(row["title"]),
                    author=str(row["author"]),
                    publish_date=str(row["publish_date"]),
                    watch_url=str(row["watch_url"]),
                    match_mode=match_mode,
                )
            )
        return result

    def search_objects(
        self,
        parsed: ParsedObjectQuery,
        *,
        min_confidence: float,
        limit: int,
    ) -> list[ObjectFrameHit]:
        if not parsed.concepts:
            return []
        if (
            isinstance(min_confidence, bool)
            or not isinstance(min_confidence, numbers.Real)
            or not 0 <= min_confidence <= 1
        ):
            raise ValueError("object min_confidence must be between 0 and 1")
        if min_confidence < self.object_index_min_confidence:
            raise ValueError(
                "object min_confidence is below the indexed floor: "
                f"requested={min_confidence}, floor={self.object_index_min_confidence}"
            )
        if isinstance(limit, bool) or not isinstance(limit, numbers.Integral) or limit < 1:
            raise ValueError("object limit must be a positive integer")

        wanted_values: list[str] = []
        wanted_parameters: list[Any] = []
        for group_id, concept in enumerate(parsed.concepts, start=1):
            labels = tuple(parsed.labels_by_concept[concept])
            if not labels:
                continue
            for label in labels:
                wanted_values.append("(?, ?)")
                wanted_parameters.extend((group_id, label))
        if not wanted_values:
            return []

        query = f"""
            WITH wanted(group_id, object_class) AS (
                VALUES {','.join(wanted_values)}
            ),
            per_group AS (
                SELECT
                    d.video_id,
                    d.keyframe_index,
                    d.video_frame_id,
                    d.timestamp,
                    w.group_id,
                    MAX(d.confidence) AS group_confidence
                FROM wanted AS w
                JOIN detections AS d
                  ON d.object_class = w.object_class COLLATE NOCASE
                WHERE d.confidence >= ?
                GROUP BY d.video_id, d.keyframe_index, d.video_frame_id,
                         d.timestamp, w.group_id
            )
            SELECT
                video_id,
                keyframe_index,
                video_frame_id,
                timestamp,
                MIN(group_confidence) AS min_confidence,
                AVG(group_confidence) AS mean_confidence,
                COUNT(*) AS matched_concepts
            FROM per_group
            GROUP BY video_id, keyframe_index, video_frame_id, timestamp
            HAVING COUNT(*) = ?
            ORDER BY min_confidence DESC, mean_confidence DESC,
                     video_id ASC, keyframe_index ASC
            LIMIT ?
        """
        parameters = (
            wanted_parameters
            + [float(min_confidence)]
            + [len(parsed.concepts), int(limit) * 2]
        )
        with _readonly_connection(self.object_database) as connection:
            rows = connection.execute(query, parameters).fetchall()

        result: list[ObjectFrameHit] = []
        seen: set[tuple[str, int]] = set()
        for row in rows:
            mapping_key = (str(row["video_id"]), int(row["keyframe_index"]))
            frame = self._frame_by_keyframe.get(mapping_key)
            if frame is None:
                raise KeyError(f"Object detection has no canonical mapping: {mapping_key}")
            submit_key = (frame.video_id, frame.video_frame_id)
            if submit_key in seen:
                continue
            seen.add(submit_key)
            result.append(
                ObjectFrameHit(
                    rank=len(result) + 1,
                    frame=frame,
                    min_confidence=float(row["min_confidence"]),
                    mean_confidence=float(row["mean_confidence"]),
                    matched_concepts=parsed.concepts,
                )
            )
            if len(result) == int(limit):
                break
        return result


__all__ = [
    "MetadataVideoHit",
    "ObjectFrameHit",
    "Phase1HybridStore",
    "load_object_labels",
]
