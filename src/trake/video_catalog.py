from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq


@dataclass(frozen=True)
class VideoInfo:
    video_id: str
    video_path: str
    fps: float
    video_available: bool


class VideoCatalog:
    """Video_Path/FPS lookup per video_id, read once from frame_mapping.parquet.

    `Video_Path` follows the same `Archive.zip::member` shape as
    `Keyframe_Path` (see `data_processing.layout.archive_ref`).
    """

    def __init__(self, frame_mapping_path: str | Path) -> None:
        path = Path(frame_mapping_path)
        if not path.is_file():
            raise FileNotFoundError(f"frame_mapping.parquet not found: {path}")
        table = pq.read_table(
            path, columns=["Video_ID", "Video_Path", "FPS", "Video_Available"]
        )
        by_video: dict[str, VideoInfo] = {}
        for video_id, video_path, fps, available in zip(
            table.column("Video_ID").to_pylist(),
            table.column("Video_Path").to_pylist(),
            table.column("FPS").to_pylist(),
            table.column("Video_Available").to_pylist(),
        ):
            by_video.setdefault(
                video_id,
                VideoInfo(
                    video_id=video_id,
                    video_path=video_path,
                    fps=float(fps),
                    video_available=bool(available),
                ),
            )
        self._by_video = by_video

    def get(self, video_id: str) -> VideoInfo:
        info = self._by_video.get(video_id)
        if info is None:
            raise KeyError(f"Unknown video_id: {video_id!r}")
        if not info.video_available:
            raise FileNotFoundError(
                f"Video {video_id!r} has no source video available (Video_Available=False)"
            )
        return info


__all__ = ["VideoCatalog", "VideoInfo"]
