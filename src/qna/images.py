from __future__ import annotations

import zipfile
from pathlib import Path

from data_processing.layout import resolve_archive


def load_keyframe_bytes(keyframe_path: str, *, data_root: str | Path) -> bytes:
    """Read raw JPEG bytes for a `Keyframe_Path` value ("Archive.zip::member").

    `Keyframe_Path` is produced by `data_processing.layout.archive_ref` and always
    has this shape when a keyframe is available (`Keyframe_Available=True`).
    """

    if "::" not in keyframe_path:
        raise ValueError(f"keyframe_path is not archive-qualified: {keyframe_path!r}")
    archive_name, member = keyframe_path.split("::", 1)
    archive_path = resolve_archive(data_root, archive_name)
    with zipfile.ZipFile(archive_path) as archive:
        return archive.read(member)


__all__ = ["load_keyframe_bytes"]
