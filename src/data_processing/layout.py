from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable
from zipfile import ZipFile


VIDEO_ID_RE = re.compile(r"(?P<video_id>L\d{2}_V\d{3})", re.IGNORECASE)
INDEXED_FRAME_RE = re.compile(
    r"(?P<video_id>L\d{2}_V\d{3})/(?P<index>\d+)\.(?P<ext>jpg|json)$",
    re.IGNORECASE,
)


class DataLayoutError(RuntimeError):
    """Raised when required source archives cannot be located unambiguously."""


def video_id_from_name(name: str) -> str:
    match = VIDEO_ID_RE.search(name.replace("\\", "/"))
    if not match:
        raise ValueError(f"No video id in path: {name}")
    return match.group("video_id").upper()


def frame_from_name(name: str) -> tuple[str, int]:
    normalized = name.replace("\\", "/")
    match = INDEXED_FRAME_RE.search(normalized)
    if not match:
        raise ValueError(f"No video/keyframe index in path: {name}")
    return match.group("video_id").upper(), int(match.group("index"))


def archive_ref(archive: Path | None, member: str) -> str:
    """Return a portable logical path, optionally qualified by its ZIP archive."""
    if archive is None:
        return member
    return f"{archive.name}::{member}"


def resolve_archive(data_root: str | Path, archive_name: str) -> Path:
    """Locate a source archive by filename under a data root.

    `Keyframe_Path`/`Video_Path`/`Object_Path` values only encode the bare
    archive filename (e.g. `"Keyframes_L21.zip"`), not its subfolder, so this
    checks the flat root first (synthetic/legacy layout), then
    `root/batch_*/<Subfolder>/archive_name` (the batch layout used by the
    real dataset).
    """
    root_path = Path(data_root)
    candidates = [root_path / archive_name, *sorted(root_path.glob(f"batch_*/*/{archive_name}"))]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Archive {archive_name!r} not found under {root_path} "
        f"(checked flat root and batch_*/*/ subfolders)"
    )


@dataclass(frozen=True)
class DatasetLayout:
    root: Path
    clip_archive: Path
    map_archive: Path
    metadata_archive: Path
    objects_archive: Path
    keyframe_archives: tuple[Path, ...]
    video_archives: tuple[Path, ...]

    @classmethod
    def discover(cls, root: Path | str) -> "DatasetLayout":
        """Locate source archives under ``root``.

        Supports two layouts, searched together: a flat root (all ZIPs
        directly under ``root``, used by synthetic test fixtures) and the
        batch layout (``root/batch_N/<Subfolder>/*.zip``, used by the real
        BTC/AIC dataset). Multiple ``batch_*`` directories are aggregated.
        """
        root_path = Path(root).resolve()

        def candidate_dirs(subfolder: str) -> list[Path]:
            dirs = [root_path, *sorted(root_path.glob(f"batch_*/{subfolder}"))]
            return [directory for directory in dirs if directory.is_dir()]

        def glob_in(subfolder: str, pattern: str) -> list[Path]:
            matches: list[Path] = []
            for directory in candidate_dirs(subfolder):
                matches.extend(directory.glob(pattern))
            return sorted(matches)

        def one(subfolder: str, pattern: str, label: str) -> Path:
            matches = glob_in(subfolder, pattern)
            if len(matches) != 1:
                names = ", ".join(path.name for path in matches) or "none"
                raise DataLayoutError(
                    f"Expected exactly one {label} archive matching {pattern!r}; found {names}"
                )
            return matches[0]

        keyframes = tuple(glob_in("Keyframes", "Keyframes_*.zip"))
        videos = tuple(glob_in("Videos", "Videos_*.zip"))
        return cls(
            root=root_path,
            clip_archive=one("CLIP_features", "clip-features*.zip", "CLIP feature"),
            map_archive=one("Keyframes", "map-keyframes*.zip", "keyframe map"),
            metadata_archive=one("Metadata", "media-info*.zip", "metadata"),
            objects_archive=one("Objects", "objects*.zip", "objects"),
            keyframe_archives=keyframes,
            video_archives=videos,
        )

    @property
    def all_archives(self) -> tuple[Path, ...]:
        return (
            self.clip_archive,
            self.map_archive,
            self.metadata_archive,
            self.objects_archive,
            *self.keyframe_archives,
            *self.video_archives,
        )


def members_with_suffix(archive: Path, suffix: str) -> list[str]:
    with ZipFile(archive) as handle:
        return sorted(
            info.filename.replace("\\", "/")
            for info in handle.infolist()
            if not info.is_dir() and info.filename.lower().endswith(suffix.lower())
        )


def members_by_video(archives: Iterable[Path], suffix: str) -> dict[str, tuple[Path, list[str]]]:
    grouped: dict[str, tuple[Path, list[str]]] = {}
    for archive in archives:
        with ZipFile(archive) as handle:
            for info in handle.infolist():
                if info.is_dir() or not info.filename.lower().endswith(suffix.lower()):
                    continue
                video_id = video_id_from_name(info.filename)
                if video_id not in grouped:
                    grouped[video_id] = (archive, [])
                previous_archive, names = grouped[video_id]
                if previous_archive != archive:
                    raise DataLayoutError(
                        f"Video {video_id} occurs in both {previous_archive.name} and {archive.name}"
                    )
                names.append(info.filename.replace("\\", "/"))
    for _, names in grouped.values():
        names.sort()
    return grouped


def indexed_members(
    archives: Iterable[Path], suffix: str
) -> dict[str, dict[int, tuple[Path, str]]]:
    """Index frame-like ZIP members by video id and their numeric filename."""
    result: dict[str, dict[int, tuple[Path, str]]] = {}
    for archive in archives:
        with ZipFile(archive) as handle:
            for info in handle.infolist():
                if info.is_dir() or not info.filename.lower().endswith(suffix.lower()):
                    continue
                video_id, frame_index = frame_from_name(info.filename)
                by_index = result.setdefault(video_id, {})
                if frame_index in by_index:
                    old_archive, old_member = by_index[frame_index]
                    raise DataLayoutError(
                        f"Duplicate {video_id}/{frame_index}: "
                        f"{old_archive.name}::{old_member} and {archive.name}::{info.filename}"
                    )
                by_index[frame_index] = (archive, info.filename.replace("\\", "/"))
    return result


def video_members(archives: Iterable[Path]) -> dict[str, tuple[Path, str]]:
    result: dict[str, tuple[Path, str]] = {}
    for archive in archives:
        with ZipFile(archive) as handle:
            for info in handle.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".mp4"):
                    continue
                video_id = video_id_from_name(info.filename)
                if video_id in result:
                    old_archive, old_member = result[video_id]
                    raise DataLayoutError(
                        f"Duplicate video {video_id}: {old_archive.name}::{old_member} "
                        f"and {archive.name}::{info.filename}"
                    )
                result[video_id] = (archive, info.filename.replace("\\", "/"))
    return result


def logical_keyframe_member(video_id: str, keyframe_index: int) -> str:
    return str(PurePosixPath("keyframes") / video_id / f"{keyframe_index:03d}.jpg")


def logical_object_member(video_id: str, keyframe_index: int) -> str:
    return str(PurePosixPath("objects") / video_id / f"{keyframe_index:03d}.json")


def logical_video_member(video_id: str) -> str:
    return str(PurePosixPath("video") / f"{video_id}.mp4")
