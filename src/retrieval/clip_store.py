from __future__ import annotations

import json
import math
import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pyarrow.parquet as pq


class ArtifactValidationError(ValueError):
    """Raised when Phase 1 artifacts cannot be joined safely."""


@dataclass(frozen=True)
class FrameRecord:
    """The submission-relevant part of one Phase 1 mapping row."""

    faiss_index: int
    video_id: str
    keyframe_index: int
    video_frame_id: int
    timestamp: float
    keyframe_path: str
    keyframe_available: bool


@dataclass(frozen=True)
class ClipSearchHit:
    """A mapped FAISS result before a query id and rank are attached."""

    score: float
    frame: FrameRecord


class Phase1ClipStore:
    """In-memory view of the Phase 1 FAISS index and frame catalog.

    Mapping rows are keyed by their explicit ``FAISS_Index`` value.  Their
    physical Parquet row order is deliberately ignored: relying on it can
    silently produce a valid-looking but incorrect BTC ``frame_id``.
    """

    _REQUIRED_COLUMNS = (
        "FAISS_Index",
        "Video_ID",
        "Keyframe_Index",
        "Video_Frame_ID",
        "Timestamp",
        "Keyframe_Path",
    )

    def __init__(
        self,
        index_path: str | Path,
        mapping_path: str | Path,
        expected_dimension: int | None = None,
        *,
        metadata_path: str | Path | None = None,
        overfetch_factor: int = 2,
    ) -> None:
        self.index_path = Path(index_path)
        self.mapping_path = Path(mapping_path)

        if not self.index_path.is_file():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if not self.mapping_path.is_file():
            raise FileNotFoundError(f"Frame mapping not found: {self.mapping_path}")
        if (
            isinstance(overfetch_factor, bool)
            or not isinstance(overfetch_factor, numbers.Integral)
            or overfetch_factor < 1
        ):
            raise ValueError("overfetch_factor must be a positive integer")
        self._overfetch_factor = int(overfetch_factor)

        # Both artifacts are loaded exactly once. Searches below only use these
        # in-memory objects.
        self._index = faiss.read_index(str(self.index_path))
        self._dimension = int(self._index.d)
        self._size = int(self._index.ntotal)

        if self._dimension <= 0:
            raise ArtifactValidationError(
                f"FAISS index has invalid dimension {self._dimension}"
            )
        if not self._index.is_trained:
            raise ArtifactValidationError("FAISS index is not trained")
        if expected_dimension is not None:
            if isinstance(expected_dimension, bool) or not isinstance(
                expected_dimension, numbers.Integral
            ):
                raise TypeError("expected_dimension must be an integer")
            if int(expected_dimension) != self._dimension:
                raise ArtifactValidationError(
                    "CLIP encoder/index dimension mismatch: "
                    f"encoder={int(expected_dimension)}, index={self._dimension}"
                )

        table_schema = pq.read_schema(self.mapping_path)
        missing = [
            name for name in self._REQUIRED_COLUMNS if name not in table_schema.names
        ]
        if missing:
            raise ArtifactValidationError(
                f"Frame mapping is missing required columns: {', '.join(missing)}"
            )
        columns = list(self._REQUIRED_COLUMNS)
        has_availability = "Keyframe_Available" in table_schema.names
        if has_availability:
            columns.append("Keyframe_Available")
        rows = pq.read_table(self.mapping_path, columns=columns).to_pylist()

        if len(rows) != self._size:
            raise ArtifactValidationError(
                "FAISS/mapping row-count mismatch: "
                f"index={self._size}, mapping={len(rows)}"
            )

        records_by_id: dict[int, FrameRecord] = {}
        for row_number, row in enumerate(rows):
            faiss_index = self._required_int(
                row.get("FAISS_Index"), "FAISS_Index", row_number
            )
            if faiss_index in records_by_id:
                raise ArtifactValidationError(
                    f"Duplicate FAISS_Index {faiss_index} in frame mapping"
                )
            video_id = str(row.get("Video_ID") or "").strip()
            if not video_id:
                raise ArtifactValidationError(
                    f"Blank Video_ID in mapping row {row_number}"
                )
            timestamp = self._required_float(
                row.get("Timestamp"), "Timestamp", row_number
            )
            keyframe_path_value = row.get("Keyframe_Path")
            if keyframe_path_value is None:
                raise ArtifactValidationError(
                    f"Null Keyframe_Path in mapping row {row_number}"
                )
            available_value = row.get("Keyframe_Available", True)
            if available_value is None:
                raise ArtifactValidationError(
                    f"Null Keyframe_Available in mapping row {row_number}"
                )

            records_by_id[faiss_index] = FrameRecord(
                faiss_index=faiss_index,
                video_id=video_id,
                keyframe_index=self._required_int(
                    row.get("Keyframe_Index"), "Keyframe_Index", row_number
                ),
                video_frame_id=self._required_int(
                    row.get("Video_Frame_ID"), "Video_Frame_ID", row_number
                ),
                timestamp=timestamp,
                keyframe_path=str(keyframe_path_value),
                # Keep unavailable keyframes in retrieval. This flag is for UI
                # and diagnostics, not a search-time filter.
                keyframe_available=bool(available_value),
            )

        expected_ids = set(range(self._size))
        actual_ids = set(records_by_id)
        if actual_ids != expected_ids:
            missing_ids = sorted(expected_ids - actual_ids)[:5]
            unexpected_ids = sorted(actual_ids - expected_ids)[:5]
            raise ArtifactValidationError(
                "FAISS_Index values must cover every index position exactly once; "
                f"missing={missing_ids}, unexpected={unexpected_ids}"
            )

        # A tuple makes the explicit id join cheap without reverting to Parquet
        # row order. It is assembled from the validated FAISS_Index dictionary.
        self._records = tuple(records_by_id[position] for position in range(self._size))

        resolved_metadata_path: Path | None
        if metadata_path is not None:
            resolved_metadata_path = Path(metadata_path)
            if not resolved_metadata_path.is_file():
                raise FileNotFoundError(
                    f"CLIP index metadata not found: {resolved_metadata_path}"
                )
        else:
            automatic = self.index_path.with_name("clip_index_meta.json")
            resolved_metadata_path = automatic if automatic.is_file() else None
        self.metadata_path = resolved_metadata_path
        if resolved_metadata_path is not None:
            self._validate_metadata(resolved_metadata_path)

    @staticmethod
    def _required_int(value: Any, field: str, row_number: int) -> int:
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise ArtifactValidationError(
                f"{field} in mapping row {row_number} must be an integer, got {value!r}"
            )
        return int(value)

    @staticmethod
    def _required_float(value: Any, field: str, row_number: int) -> float:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ArtifactValidationError(
                f"{field} in mapping row {row_number} must be numeric, got {value!r}"
            )
        result = float(value)
        if not math.isfinite(result):
            raise ArtifactValidationError(
                f"{field} in mapping row {row_number} must be finite, got {value!r}"
            )
        return result

    def _validate_metadata(self, metadata_path: Path) -> None:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ArtifactValidationError(
                f"Cannot read CLIP index metadata {metadata_path}: {exc}"
            ) from exc
        if not isinstance(metadata, dict):
            raise ArtifactValidationError(
                f"CLIP index metadata {metadata_path} must contain a JSON object"
            )
        if "dimension" in metadata and metadata["dimension"] != self._dimension:
            raise ArtifactValidationError(
                "FAISS/metadata dimension mismatch: "
                f"index={self._dimension}, metadata={metadata['dimension']!r}"
            )
        if "vectors" in metadata and metadata["vectors"] != self._size:
            raise ArtifactValidationError(
                "FAISS/metadata vector-count mismatch: "
                f"index={self._size}, metadata={metadata['vectors']!r}"
            )

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def size(self) -> int:
        return self._size

    @property
    def ntotal(self) -> int:
        """FAISS-compatible alias for the number of indexed vectors."""

        return self._size

    def frame_for_faiss_index(self, faiss_index: int) -> FrameRecord:
        if isinstance(faiss_index, bool) or not isinstance(faiss_index, numbers.Integral):
            raise TypeError("faiss_index must be an integer")
        position = int(faiss_index)
        if position < 0 or position >= self._size:
            raise IndexError(f"FAISS index out of range: {position}")
        return self._records[position]

    def _normalize_query(self, vector: np.ndarray) -> np.ndarray:
        original = np.asarray(vector)
        if not np.issubdtype(original.dtype, np.number) or np.issubdtype(
            original.dtype, np.complexfloating
        ):
            raise ValueError("CLIP query vector must contain real numeric values")
        if original.ndim == 2 and original.shape[0] == 1:
            original = original[0]
        if original.ndim != 1:
            raise ValueError(
                f"CLIP query vector must have shape ({self._dimension},), "
                f"got {original.shape}"
            )
        if original.shape[0] != self._dimension:
            raise ValueError(
                "CLIP query dimension mismatch: "
                f"expected {self._dimension}, got {original.shape[0]}"
            )
        query = np.asarray(original, dtype=np.float32)
        if not np.all(np.isfinite(query)):
            raise ValueError("CLIP query vector contains NaN or infinite values")
        norm = float(np.linalg.norm(query.astype(np.float64)))
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError("CLIP query vector must have a non-zero finite norm")
        query = np.ascontiguousarray(query / np.float32(norm), dtype=np.float32)
        return query.reshape(1, self._dimension)

    def search(self, vector: np.ndarray, top_k: int = 100) -> list[ClipSearchHit]:
        """Search, map, and deduplicate results by the exact BTC submit pair.

        The search adaptively requests more FAISS neighbors when duplicate
        ``(Video_ID, Video_Frame_ID)`` pairs consume slots. It also expands a
        score tie at the current boundary so the final secondary ordering by
        ``FAISS_Index`` is deterministic.
        """

        if isinstance(top_k, bool) or not isinstance(top_k, numbers.Integral):
            raise TypeError("top_k must be an integer")
        top_k = int(top_k)
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        query = self._normalize_query(vector)
        if self._size == 0:
            return []

        requested = min(
            self._size,
            max(top_k + 1, top_k * self._overfetch_factor),
        )
        while True:
            raw_scores, raw_indices = self._index.search(query, requested)
            candidates: list[ClipSearchHit] = []
            for score_value, index_value in zip(raw_scores[0], raw_indices[0]):
                faiss_index = int(index_value)
                if faiss_index < 0:
                    continue
                if faiss_index >= self._size:
                    raise ArtifactValidationError(
                        f"FAISS returned unmapped index id {faiss_index}"
                    )
                score = float(score_value)
                if not math.isfinite(score):
                    raise ArtifactValidationError(
                        f"FAISS returned a non-finite score for index {faiss_index}"
                    )
                candidates.append(
                    ClipSearchHit(score=score, frame=self._records[faiss_index])
                )

            candidates.sort(key=lambda item: (-item.score, item.frame.faiss_index))
            unique: list[ClipSearchHit] = []
            seen_submit_pairs: set[tuple[str, int]] = set()
            for candidate in candidates:
                pair = (
                    candidate.frame.video_id,
                    candidate.frame.video_frame_id,
                )
                if pair in seen_submit_pairs:
                    continue
                seen_submit_pairs.add(pair)
                unique.append(candidate)

            need_more_unique = len(unique) < top_k
            boundary_tie_may_be_truncated = False
            if len(unique) >= top_k and candidates:
                cutoff_score = unique[top_k - 1].score
                boundary_tie_may_be_truncated = (
                    candidates[-1].score == cutoff_score
                )

            if requested < self._size and (
                need_more_unique or boundary_tie_may_be_truncated
            ):
                requested = min(
                    self._size,
                    max(requested + 1, requested * 2),
                )
                continue
            return unique[:top_k]


__all__ = [
    "ArtifactValidationError",
    "ClipSearchHit",
    "FrameRecord",
    "Phase1ClipStore",
]
