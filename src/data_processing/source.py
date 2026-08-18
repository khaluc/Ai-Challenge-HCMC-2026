from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .layout import DatasetLayout, video_id_from_name


@dataclass(frozen=True)
class FrameMapRow:
    keyframe_index: int
    timestamp: float
    fps: float
    video_frame_id: int


def zip_members_by_video(archive: Path, suffix: str) -> dict[str, str]:
    with ZipFile(archive) as handle:
        result: dict[str, str] = {}
        for info in handle.infolist():
            if info.is_dir() or not info.filename.lower().endswith(suffix.lower()):
                continue
            video_id = video_id_from_name(info.filename)
            member = info.filename.replace("\\", "/")
            if video_id in result:
                raise ValueError(
                    f"{archive.name}: multiple {suffix} members for {video_id}: "
                    f"{result[video_id]} and {member}"
                )
            result[video_id] = member
    return dict(sorted(result.items()))


def load_all_frame_maps(layout: DatasetLayout) -> tuple[dict[str, list[FrameMapRow]], dict[str, str]]:
    members = zip_members_by_video(layout.map_archive, ".csv")
    maps: dict[str, list[FrameMapRow]] = {}
    with ZipFile(layout.map_archive) as handle:
        for video_id, member in members.items():
            text = io.TextIOWrapper(handle.open(member), encoding="utf-8-sig", newline="")
            try:
                reader = csv.DictReader(text)
                expected = {"n", "pts_time", "fps", "frame_idx"}
                if reader.fieldnames is None or not expected.issubset(reader.fieldnames):
                    raise ValueError(
                        f"{member}: expected columns {sorted(expected)}, got {reader.fieldnames}"
                    )
                rows = [
                    FrameMapRow(
                        keyframe_index=int(row["n"]),
                        timestamp=float(row["pts_time"]),
                        fps=float(row["fps"]),
                        video_frame_id=int(row["frame_idx"]),
                    )
                    for row in reader
                ]
            finally:
                text.close()
            maps[video_id] = rows
    return maps, members


def load_clip_array(handle: ZipFile, member: str) -> np.ndarray:
    payload = io.BytesIO(handle.read(member))
    array = np.load(payload, allow_pickle=False)
    if array.ndim != 2:
        raise ValueError(f"{member}: expected a 2-D CLIP array, got {array.shape}")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{member}: expected floating-point CLIP values, got {array.dtype}")
    return array


def load_all_metadata(layout: DatasetLayout) -> tuple[dict[str, dict], dict[str, str]]:
    members = zip_members_by_video(layout.metadata_archive, ".json")
    metadata: dict[str, dict] = {}
    with ZipFile(layout.metadata_archive) as handle:
        for video_id, member in members.items():
            metadata[video_id] = json.loads(handle.read(member))
    return metadata, members
