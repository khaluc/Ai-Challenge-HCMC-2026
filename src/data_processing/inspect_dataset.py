from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZipFile

import numpy as np

from .layout import DatasetLayout, indexed_members, video_members
from .source import (
    load_all_frame_maps,
    load_all_metadata,
    load_clip_array,
    zip_members_by_video,
)


def _level_counts(video_ids: set[str] | list[str]) -> dict[str, int]:
    counts = Counter(video_id.split("_", 1)[0] for video_id in video_ids)
    return dict(sorted(counts.items()))


def _zip_safety_and_stats(path: Path, deep_crc: bool) -> dict[str, Any]:
    started = time.perf_counter()
    with ZipFile(path) as handle:
        infos = handle.infolist()
        normalized = [info.filename.replace("\\", "/") for info in infos]
        unsafe = [
            name
            for name in normalized
            if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
        ]
        duplicates = [name for name, count in Counter(normalized).items() if count > 1]
        case_duplicates = [
            name
            for name, count in Counter(item.casefold() for item in normalized).items()
            if count > 1
        ]
        encrypted = [
            info.filename for info in infos if bool(info.flag_bits & 0x1) and not info.is_dir()
        ]
        bad_crc = handle.testzip() if deep_crc else None
        uncompressed = sum(info.file_size for info in infos if not info.is_dir())
    return {
        "archive": path.name,
        "compressed_bytes": path.stat().st_size,
        "uncompressed_bytes": uncompressed,
        "entries": len(infos),
        "unsafe_paths": unsafe,
        "duplicate_paths": duplicates,
        "case_duplicate_paths": case_duplicates,
        "encrypted_entries": encrypted,
        "crc_checked": deep_crc,
        "first_bad_crc_entry": bad_crc,
        "seconds": round(time.perf_counter() - started, 3),
    }


def audit_dataset(layout: DatasetLayout, deep_crc: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    frame_maps, map_members = load_all_frame_maps(layout)
    clip_members = zip_members_by_video(layout.clip_archive, ".npy")
    metadata, metadata_members = load_all_metadata(layout)
    object_frames = indexed_members((layout.objects_archive,), ".json")
    keyframe_frames = indexed_members(layout.keyframe_archives, ".jpg")
    videos = video_members(layout.video_archives)

    ids_by_source = {
        "maps": set(frame_maps),
        "clip": set(clip_members),
        "metadata": set(metadata_members),
        "objects": set(object_frames),
        "keyframes": set(keyframe_frames),
        "videos": set(videos),
    }
    core_ids = ids_by_source["maps"]
    errors: list[str] = []
    warnings: list[str] = []
    video_issues: dict[str, list[str]] = defaultdict(list)

    for source in ("clip", "metadata", "objects"):
        missing = sorted(core_ids - ids_by_source[source])
        extra = sorted(ids_by_source[source] - core_ids)
        if missing:
            errors.append(f"{source}: missing {len(missing)} videos present in maps")
        if extra:
            errors.append(f"{source}: has {len(extra)} videos absent from maps")

    missing_keyframe_videos = sorted(core_ids - ids_by_source["keyframes"])
    missing_video_files = sorted(core_ids - ids_by_source["videos"])
    if missing_keyframe_videos:
        warnings.append(f"Keyframe images missing for {len(missing_keyframe_videos)} videos")
    if missing_video_files:
        warnings.append(f"Original videos missing for {len(missing_video_files)} videos")

    clip_rows = 0
    clip_dimension: int | None = None
    clip_dtypes: set[str] = set()
    norm_min = math.inf
    norm_max = -math.inf
    feature_nonfinite = 0
    feature_zero_vectors = 0
    with ZipFile(layout.clip_archive) as handle:
        for video_id in sorted(clip_members):
            array = load_clip_array(handle, clip_members[video_id])
            clip_rows += len(array)
            clip_dtypes.add(str(array.dtype))
            if clip_dimension is None:
                clip_dimension = int(array.shape[1])
            elif array.shape[1] != clip_dimension:
                errors.append(
                    f"{video_id}: CLIP dimension {array.shape[1]} != {clip_dimension}"
                )
            if video_id in frame_maps and len(array) != len(frame_maps[video_id]):
                video_issues[video_id].append(
                    f"CLIP rows {len(array)} != map rows {len(frame_maps[video_id])}"
                )
            finite_rows = np.isfinite(array).all(axis=1)
            feature_nonfinite += int((~finite_rows).sum())
            if finite_rows.any():
                norms = np.linalg.norm(array[finite_rows].astype(np.float32), axis=1)
                feature_zero_vectors += int((norms == 0).sum())
                norm_min = min(norm_min, float(norms.min()))
                norm_max = max(norm_max, float(norms.max()))

    total_map_rows = 0
    duplicate_video_frame_rows = 0
    videos_with_duplicate_frame_ids = 0
    missing_keyframe_rows = 0
    missing_object_rows = 0
    missing_rows_by_level: Counter[str] = Counter()
    for video_id, rows in sorted(frame_maps.items()):
        total_map_rows += len(rows)
        expected_indices = list(range(1, len(rows) + 1))
        actual_indices = [row.keyframe_index for row in rows]
        if actual_indices != expected_indices:
            video_issues[video_id].append("map n is not contiguous 1-based order")
        timestamps = [row.timestamp for row in rows]
        if any(right < left for left, right in zip(timestamps, timestamps[1:])):
            video_issues[video_id].append("timestamps are not monotonic")
        frame_ids = [row.video_frame_id for row in rows]
        if any(right < left for left, right in zip(frame_ids, frame_ids[1:])):
            video_issues[video_id].append("video frame ids are not monotonic")
        duplicated = len(frame_ids) - len(set(frame_ids))
        if duplicated:
            videos_with_duplicate_frame_ids += 1
            duplicate_video_frame_rows += duplicated

        expected_set = set(expected_indices)
        object_set = set(object_frames.get(video_id, {}))
        keyframe_set = set(keyframe_frames.get(video_id, {}))
        missing_objects = expected_set - object_set
        if missing_objects:
            missing_object_rows += len(missing_objects)
            video_issues[video_id].append(f"missing {len(missing_objects)} object JSON files")
        missing_images = expected_set - keyframe_set
        if missing_images:
            missing_keyframe_rows += len(missing_images)
            missing_rows_by_level[video_id.split("_", 1)[0]] += len(missing_images)

    if feature_nonfinite:
        errors.append(f"CLIP contains {feature_nonfinite} non-finite vectors")
    if feature_zero_vectors:
        errors.append(f"CLIP contains {feature_zero_vectors} zero vectors")
    if missing_object_rows:
        errors.append(f"Objects missing for {missing_object_rows} mapped keyframes")
    for video_id, issues in video_issues.items():
        if any("CLIP" in issue or "object" in issue or "map n" in issue for issue in issues):
            errors.extend(f"{video_id}: {issue}" for issue in issues)

    empty_descriptions = sum(not str(item.get("description", "")).strip() for item in metadata.values())
    empty_keywords = sum(not item.get("keywords") for item in metadata.values())
    all_metadata_fields = sorted({key for item in metadata.values() for key in item})

    archive_reports = [
        _zip_safety_and_stats(path, deep_crc=deep_crc) for path in layout.all_archives
    ]
    for item in archive_reports:
        if item["unsafe_paths"] or item["duplicate_paths"] or item["case_duplicate_paths"]:
            errors.append(f"{item['archive']}: unsafe or duplicate ZIP member names")
        if item["encrypted_entries"]:
            errors.append(f"{item['archive']}: contains encrypted entries")
        if item["first_bad_crc_entry"]:
            errors.append(
                f"{item['archive']}: CRC failed at {item['first_bad_crc_entry']}"
            )

    core_sources_match = all(ids_by_source[name] == core_ids for name in ("clip", "metadata", "objects"))
    report: dict[str, Any] = {
        "status": {
            "core_retrieval_data_complete": bool(
                core_sources_match
                and not missing_object_rows
                and not feature_nonfinite
                and not feature_zero_vectors
                and total_map_rows == clip_rows
                and not errors
            ),
            "keyframe_images_complete": missing_keyframe_rows == 0,
            "original_videos_complete": len(missing_video_files) == 0,
            "deep_crc_checked": deep_crc,
        },
        "summary": {
            "videos_in_core": len(core_ids),
            "videos_by_level": _level_counts(core_ids),
            "mapped_keyframes": total_map_rows,
            "clip_vectors": clip_rows,
            "clip_dimension": clip_dimension,
            "object_json_files": sum(len(items) for items in object_frames.values()),
            "available_keyframe_images": sum(len(items) for items in keyframe_frames.values()),
            "available_original_videos": len(videos),
            "missing_keyframe_images": missing_keyframe_rows,
            "missing_keyframes_by_level": dict(sorted(missing_rows_by_level.items())),
            "missing_keyframe_videos": len(missing_keyframe_videos),
            "missing_original_videos": len(missing_video_files),
        },
        "clip": {
            "dtypes": sorted(clip_dtypes),
            "norm_min": None if math.isinf(norm_min) else norm_min,
            "norm_max": None if math.isinf(norm_max) else norm_max,
            "nonfinite_vectors": feature_nonfinite,
            "zero_vectors": feature_zero_vectors,
        },
        "frame_mapping": {
            "videos_with_duplicate_video_frame_id": videos_with_duplicate_frame_ids,
            "duplicate_video_frame_id_extra_rows": duplicate_video_frame_rows,
            "note": "Duplicate video_frame_id values are retained; submit frame_idx from the map CSV.",
        },
        "metadata": {
            "documents": len(metadata),
            "fields": all_metadata_fields,
            "empty_descriptions": empty_descriptions,
            "empty_keywords": empty_keywords,
            "has_transcript_or_asr": any(
                key.lower() in {"transcript", "asr", "subtitle", "subtitles"}
                for key in all_metadata_fields
            ),
        },
        "source_video_counts": {
            source: {"total": len(ids), "by_level": _level_counts(ids)}
            for source, ids in ids_by_source.items()
        },
        "missing_video_ids": {
            "keyframes": missing_keyframe_videos,
            "videos": missing_video_files,
        },
        "video_issues": dict(sorted(video_issues.items())),
        "archives": archive_reports,
        "errors": errors,
        "warnings": warnings,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    return report


def write_audit_report(report: dict[str, Any], output_dir: Path | str) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "data_audit.json"
    md_path = output / "data_audit.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    status = report["status"]
    summary = report["summary"]
    lines = [
        "# BTC data audit",
        "",
        f"- Core retrieval data complete: **{status['core_retrieval_data_complete']}**",
        f"- Keyframe images complete: **{status['keyframe_images_complete']}**",
        f"- Original videos complete: **{status['original_videos_complete']}**",
        f"- Deep ZIP CRC checked: **{status['deep_crc_checked']}**",
        "",
        "## Counts",
        "",
        f"- Core videos: {summary['videos_in_core']:,}",
        f"- Mapping / CLIP vectors: {summary['mapped_keyframes']:,} / {summary['clip_vectors']:,}",
        f"- Object JSON files: {summary['object_json_files']:,}",
        f"- Available / missing keyframe images: {summary['available_keyframe_images']:,} / {summary['missing_keyframe_images']:,}",
        f"- Available / missing original videos: {summary['available_original_videos']:,} / {summary['missing_original_videos']:,}",
        "",
        "## Missing keyframes by level",
        "",
    ]
    missing = summary["missing_keyframes_by_level"]
    lines.extend(f"- {level}: {count:,}" for level, count in missing.items())
    lines.extend(["", "## Errors", ""])
    if report["errors"]:
        lines.extend(f"- {item}" for item in report["errors"])
    else:
        lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    if report["warnings"]:
        lines.extend(f"- {item}" for item in report["warnings"])
    else:
        lines.append("- None")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
