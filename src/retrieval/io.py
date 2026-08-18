from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import fields
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from .schemas import Query, RetrievalHit


QUERY_FIELDS = ("query_id", "text")
PREDICTION_FIELDS = tuple(field.name for field in fields(RetrievalHit))
_UNSAFE_FILENAME_RUN = re.compile(r"[^\w.-]+", flags=re.UNICODE)
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}
_MAX_SUBMISSION_BASENAME_LENGTH = 180


def load_queries(path: Path | str) -> list[Query]:
    """Load textual KIS queries from a CSV or JSON Lines file.

    CSV files must have exactly the header ``query_id,text``. JSONL/NDJSON
    records must be objects containing string ``query_id`` and ``text`` fields.
    Query identifiers are required to be unique within the input file.
    """

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        queries = _load_queries_csv(source)
    elif suffix in {".jsonl", ".ndjson"}:
        queries = _load_queries_jsonl(source)
    else:
        raise ValueError(
            f"{source}: unsupported query format {source.suffix!r}; "
            "expected .csv, .jsonl, or .ndjson"
        )

    first_seen: dict[str, int] = {}
    for position, query in enumerate(queries, start=1):
        previous = first_seen.get(query.query_id)
        if previous is not None:
            raise ValueError(
                f"{source}: duplicate query_id {query.query_id!r} "
                f"at records {previous} and {position}"
            )
        first_seen[query.query_id] = position
    return queries


def _load_queries_csv(path: Path) -> list[Query]:
    queries: list[Query] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(QUERY_FIELDS):
            raise ValueError(
                f"{path}: expected CSV header {','.join(QUERY_FIELDS)!r}, "
                f"got {reader.fieldnames!r}"
            )

        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"{path}:{line_number}: too many CSV fields")
            try:
                queries.append(Query(query_id=row["query_id"], text=row["text"]))
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return queries


def _load_queries_jsonl(path: Path) -> list[Query]:
    queries: list[Query] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")

            missing = [name for name in QUERY_FIELDS if name not in record]
            if missing:
                raise ValueError(
                    f"{path}:{line_number}: missing field(s): {', '.join(missing)}"
                )
            query_id = record["query_id"]
            text = record["text"]
            if not isinstance(query_id, str) or not isinstance(text, str):
                raise ValueError(
                    f"{path}:{line_number}: query_id and text must both be strings"
                )
            try:
                queries.append(Query(query_id=query_id, text=text))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return queries


def write_predictions_csv(
    path: Path | str, hits: Iterable[RetrievalHit]
) -> Path:
    """Write the lossless, evaluator-facing retrieval result format."""

    materialized = list(hits)
    for hit in materialized:
        if not isinstance(hit, RetrievalHit):
            raise TypeError(
                "hits must contain RetrievalHit instances, "
                f"got {type(hit).__name__}"
            )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(PREDICTION_FIELDS)
            for hit in materialized:
                writer.writerow(tuple(getattr(hit, name) for name in PREDICTION_FIELDS))
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def write_btc_submission(
    output_dir: Path | str,
    hits: Iterable[RetrievalHit],
    *,
    max_results: int = 100,
    zip_path: Path | str | None = None,
) -> list[Path]:
    """Write one official, headerless BTC CSV per query.

    Rows are ordered by the internal ``rank`` field and contain only
    ``video_id,frame_id``. Even if a larger value is supplied, at most 100
    results are emitted for a query. ZIP members, when requested, are stored at
    the archive root rather than below the output directory name.
    """

    if isinstance(max_results, bool) or not isinstance(max_results, int):
        raise TypeError("max_results must be an integer")
    if max_results < 1:
        raise ValueError("max_results must be at least 1")
    limit = min(max_results, 100)

    grouped: dict[str, list[tuple[int, RetrievalHit]]] = defaultdict(list)
    for input_position, hit in enumerate(hits):
        if not isinstance(hit, RetrievalHit):
            raise TypeError(
                "hits must contain RetrievalHit instances, "
                f"got {type(hit).__name__}"
            )
        grouped[hit.query_id].append((input_position, hit))

    # Resolve and collision-check every name before creating any output. The
    # case-folded comparison also protects callers on case-insensitive filesystems.
    planned_paths = plan_btc_submission_paths(output_dir, grouped)

    prepared: dict[str, list[tuple[str, int]]] = {}
    for query_id, query_hits in grouped.items():
        # input_position makes equal ranks stable without changing their order.
        ordered = sorted(query_hits, key=lambda item: (item[1].rank, item[0]))
        selected = ordered[:limit]
        selected_ranks = [item[1].rank for item in selected]
        expected_ranks = list(range(1, len(selected) + 1))
        if selected_ranks != expected_ranks:
            raise ValueError(
                f"query {query_id!r} ranks must be unique and dense from 1; "
                f"got {selected_ranks}"
            )
        submit_pairs = [
            (_without_mp4_suffix(item[1].video_id), item[1].frame_id) for item in selected
        ]
        if len(submit_pairs) != len(set(submit_pairs)):
            raise ValueError(f"query {query_id!r} contains duplicate BTC submission pairs")
        prepared[query_id] = submit_pairs

    destination_dir = Path(output_dir)
    written = [planned_paths[query_id] for query_id in prepared]
    temporary_files = [path.with_suffix(path.suffix + ".tmp") for path in written]
    archive_path = Path(zip_path) if zip_path is not None else None
    archive_tmp = (
        archive_path.with_suffix(archive_path.suffix + ".tmp")
        if archive_path is not None
        else None
    )
    if archive_path is not None:
        reserved = {
            path.resolve(): f"submission CSV {path.name!r}" for path in written
        }
        reserved.update(
            {
                path.resolve(): f"temporary submission CSV {path.name!r}"
                for path in temporary_files
            }
        )
        for label, path in (
            ("submission ZIP", archive_path),
            ("temporary submission ZIP", archive_tmp),
        ):
            if path is None:
                continue
            conflict = reserved.get(path.resolve())
            if conflict is not None:
                raise ValueError(f"{label} path conflicts with {conflict}: {path.resolve()}")
        if archive_path.resolve() == destination_dir.resolve():
            raise ValueError(
                "submission ZIP path must not be the submission directory: "
                f"{archive_path.resolve()}"
            )

    destination_dir.mkdir(parents=True, exist_ok=True)
    try:
        for (query_id, rows), temporary in zip(prepared.items(), temporary_files):
            with temporary.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerows(rows)

        if archive_path is not None and archive_tmp is not None:
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            with ZipFile(archive_tmp, "w", compression=ZIP_DEFLATED) as archive:
                for final_path, temporary in zip(written, temporary_files):
                    archive.write(temporary, arcname=final_path.name)

        for final_path, temporary in zip(written, temporary_files):
            temporary.replace(final_path)
        if archive_path is not None and archive_tmp is not None:
            archive_tmp.replace(archive_path)
    finally:
        for temporary in temporary_files:
            if temporary.exists():
                temporary.unlink()
        if archive_tmp is not None and archive_tmp.exists():
            archive_tmp.unlink()

    return written


def plan_btc_submission_paths(
    output_dir: Path | str,
    query_ids: Iterable[str],
) -> dict[str, Path]:
    """Return collision-checked official CSV paths without touching the filesystem."""

    filenames: dict[str, str] = {}
    owners: dict[str, str] = {}
    for query_id in query_ids:
        filename = _submission_filename(query_id)
        collision_key = filename.casefold()
        previous = owners.get(collision_key)
        if previous is not None and previous != query_id:
            raise ValueError(
                "query_id filename collision: "
                f"{previous!r} and {query_id!r} both map to {filename!r}"
            )
        filenames[query_id] = filename
        owners[collision_key] = query_id

    destination_dir = Path(output_dir)
    return {
        query_id: destination_dir / filenames[query_id]
        for query_id in filenames
    }


def _submission_filename(query_id: str) -> str:
    if not isinstance(query_id, str):
        raise TypeError("query_id must be a string")

    normalized = unicodedata.normalize("NFKC", query_id.strip())
    if not normalized:
        raise ValueError("query_id must not be blank")
    if "\x00" in normalized or "/" in normalized or "\\" in normalized:
        raise ValueError(f"unsafe query_id path traversal: {query_id!r}")
    if normalized in {".", ".."}:
        raise ValueError(f"unsafe query_id path traversal: {query_id!r}")

    basename = _UNSAFE_FILENAME_RUN.sub("_", normalized).strip(" .")
    if not basename or basename in {".", ".."}:
        raise ValueError(f"query_id {query_id!r} has no safe filename characters")
    if basename.upper() in _WINDOWS_RESERVED_NAMES:
        basename = f"_{basename}"
    basename = basename[:_MAX_SUBMISSION_BASENAME_LENGTH].rstrip(" .")
    if not basename:
        raise ValueError(f"query_id {query_id!r} has no safe filename characters")
    return f"{basename}.csv"


def _without_mp4_suffix(video_id: Any) -> str:
    if not isinstance(video_id, str):
        raise TypeError("video_id must be a string")
    normalized = video_id.strip()
    if normalized.lower().endswith(".mp4"):
        normalized = normalized[:-4]
    if not normalized:
        raise ValueError("video_id must not be blank")
    return normalized
