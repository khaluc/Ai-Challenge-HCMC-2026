from __future__ import annotations

import shutil
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from data_processing.layout import resolve_archive

from .video_catalog import VideoInfo


@dataclass(frozen=True)
class DenseFrame:
    video_id: str
    timestamp: float
    image_bytes: bytes


@contextmanager
def _extracted_video(video_path: str, *, data_root: str | Path) -> Iterator[Path]:
    if "::" not in video_path:
        raise ValueError(f"video_path is not archive-qualified: {video_path!r}")
    archive_name, member = video_path.split("::", 1)
    archive_path = resolve_archive(data_root, archive_name)
    with tempfile.TemporaryDirectory(prefix="phase10_") as tmpdir:
        target = Path(tmpdir) / Path(member).name
        with zipfile.ZipFile(archive_path) as archive, archive.open(member) as source:
            with target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
        yield target


def extract_video_to_cache(
    video_info: VideoInfo, *, data_root: str | Path, cache_dir: str | Path
) -> Path:
    """Extract a video from its ZIP archive into a persistent cache directory.

    Unlike `_extracted_video` (a temp dir removed after one `with` block),
    this is for serving the same video repeatedly over HTTP with Range
    requests — the file must stay on disk across many separate requests,
    not just one decode call.
    """
    if "::" not in video_info.video_path:
        raise ValueError(f"video_path is not archive-qualified: {video_info.video_path!r}")
    archive_name, member = video_info.video_path.split("::", 1)
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    target = cache_root / f"{video_info.video_id}.mp4"
    if target.is_file() and target.stat().st_size > 0:
        return target
    archive_path = resolve_archive(data_root, archive_name)
    tmp_target = target.with_suffix(".mp4.tmp")
    with zipfile.ZipFile(archive_path) as archive, archive.open(member) as source:
        with tmp_target.open("wb") as destination:
            shutil.copyfileobj(source, destination)
    tmp_target.replace(target)
    return target


def decode_dense_window(
    video_info: VideoInfo,
    *,
    center_seconds: float,
    window_seconds: float = 2.5,
    step_seconds: float = 0.5,
    data_root: str | Path = ".",
) -> list[DenseFrame]:
    """Decode only a small window around a coarse timestamp, not the whole video.

    Uses OpenCV's millisecond seek (`CAP_PROP_POS_MSEC`); this is the common
    "no local GPU, no ffmpeg binary available" baseline. Seek accuracy can
    land on the nearest decodable frame for some codecs rather than the
    exact millisecond, which is an accepted tradeoff for a coarse-to-fine
    baseline — fine alignment re-scores every decoded candidate anyway.
    """

    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if step_seconds <= 0:
        raise ValueError("step_seconds must be positive")
    if center_seconds < 0:
        raise ValueError("center_seconds must be non-negative")

    import cv2  # local import: optional dependency, only needed for Stage 10

    start = max(0.0, center_seconds - window_seconds)
    end = center_seconds + window_seconds
    frames: list[DenseFrame] = []
    with _extracted_video(video_info.video_path, data_root=data_root) as video_file:
        capture = cv2.VideoCapture(str(video_file))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open decoded video: {video_file}")
        try:
            timestamp = start
            while timestamp <= end:
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
                ok, frame = capture.read()
                if ok:
                    success, encoded = cv2.imencode(".jpg", frame)
                    if success:
                        frames.append(
                            DenseFrame(
                                video_id=video_info.video_id,
                                timestamp=timestamp,
                                image_bytes=encoded.tobytes(),
                            )
                        )
                timestamp += step_seconds
        finally:
            capture.release()
    return frames


__all__ = ["DenseFrame", "decode_dense_window", "extract_video_to_cache"]
