from __future__ import annotations

import csv
import io
import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

from data_processing.layout import DatasetLayout


VIDEO_IDS = ("L21_V001", "L21_V002")


def _write_zip(path: Path, members: Mapping[str, bytes | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for member, payload in members.items():
            if isinstance(payload, str):
                payload = payload.encode("utf-8")
            archive.writestr(member, payload)


def _npy_bytes(values: Iterable[Iterable[float]]) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(values, dtype=np.float16), allow_pickle=False)
    return buffer.getvalue()


def _map_csv(rows: Iterable[tuple[int, float, float, int]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(("n", "pts_time", "fps", "frame_idx"))
    writer.writerows(rows)
    return buffer.getvalue()


def _object_payload(
    detections: Iterable[tuple[float, str, str, int, list[float]]],
) -> str:
    rows = list(detections)
    return json.dumps(
        {
            "detection_scores": [row[0] for row in rows],
            "detection_class_names": [row[1] for row in rows],
            "detection_class_entities": [row[2] for row in rows],
            "detection_class_labels": [row[3] for row in rows],
            "detection_boxes": [row[4] for row in rows],
        }
    )


@pytest.fixture
def dataset_factory(
    tmp_path: Path,
) -> Callable[..., DatasetLayout]:
    counter = 0

    def create(
        *,
        missing_keyframes: set[tuple[str, int]] | None = None,
        zero_based_first_map: bool = False,
    ) -> DatasetLayout:
        nonlocal counter
        counter += 1
        root = tmp_path / f"dataset_{counter}"
        root.mkdir()
        missing = missing_keyframes or set()

        first_map = (
            [(0, 0.0, 25.0, 0), (1, 1.24, 25.0, 31)]
            if zero_based_first_map
            else [(1, 0.0, 25.0, 0), (2, 1.24, 25.0, 31)]
        )
        _write_zip(
            root / "map-keyframes-aic25-b1.zip",
            {
                "map-keyframes/L21_V001.csv": _map_csv(first_map),
                "map-keyframes/L21_V002.csv": _map_csv(
                    [(1, 4.0, 25.0, 100), (2, 8.12, 25.0, 203)]
                ),
            },
        )
        _write_zip(
            root / "clip-features-32-aic25-b1.zip",
            {
                "clip-features/L21_V001.npy": _npy_bytes(
                    [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]
                ),
                "clip-features/L21_V002.npy": _npy_bytes(
                    [[-5.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
                ),
            },
        )
        _write_zip(
            root / "media-info-aic25-b1.zip",
            {
                "media-info/L21_V001.json": json.dumps(
                    {
                        "title": "Đường phố Hà Nội",
                        "description": "Khám phá hồ Hoàn Kiếm",
                        "keywords": ["Thủ đô", "Việt Nam"],
                        "author": "Đài Truyền Hình",
                        "length": 12,
                    },
                    ensure_ascii=False,
                ),
                "media-info/L21_V002.json": json.dumps(
                    {
                        "title": "Nông trại miền Tây",
                        "description": "Thu hoạch trái cây bên dòng sông",
                        "keywords": "nông nghiệp",
                        "author": "Kênh Miền Tây",
                        "length": 20,
                    },
                    ensure_ascii=False,
                ),
            },
        )

        object_members = {
            "objects/L21_V001/001.json": _object_payload(
                [
                    (0.95, "/m/person", "person", 1, [0.1, 0.2, 0.3, 0.4]),
                    (0.49, "/m/car", "car", 3, [0.2, 0.3, 0.5, 0.7]),
                    (0.50, "/m/laptop", "laptop", 5, [0.4, 0.1, 0.8, 0.6]),
                ]
            ),
            "objects/L21_V001/002.json": _object_payload(
                [(0.20, "/m/bottle", "bottle", 7, [0.0, 0.1, 0.2, 0.3])]
            ),
            "objects/L21_V002/001.json": _object_payload(
                [(0.80, "/m/car", "car", 3, [0.25, 0.5, 0.75, 1.0])]
            ),
            "objects/L21_V002/002.json": _object_payload([]),
        }
        if zero_based_first_map:
            object_members["objects/L21_V001/000.json"] = object_members.pop(
                "objects/L21_V001/002.json"
            )
        _write_zip(root / "objects-aic25-b1.zip", object_members)

        keyframe_members: dict[str, bytes] = {}
        for video_id in VIDEO_IDS:
            for keyframe_index in (1, 2):
                if (video_id, keyframe_index) not in missing:
                    keyframe_members[
                        f"Keyframes_L21/{video_id}/{keyframe_index:03d}.jpg"
                    ] = b"synthetic-jpeg"
        _write_zip(root / "Keyframes_L21.zip", keyframe_members)
        _write_zip(
            root / "Videos_L21_a.zip",
            {
                "video/L21_V001.mp4": b"synthetic-mp4-1",
                "video/L21_V002.mp4": b"synthetic-mp4-2",
            },
        )
        return DatasetLayout.discover(root)

    return create


@pytest.fixture
def synthetic_layout(dataset_factory: Callable[..., DatasetLayout]) -> DatasetLayout:
    return dataset_factory()
