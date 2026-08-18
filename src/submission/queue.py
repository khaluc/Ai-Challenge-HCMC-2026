from __future__ import annotations

import csv
import itertools
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SubmissionItem:
    id: int
    video_id: str
    frame_id: int
    route: str = "kis"
    answer: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "video_id": self.video_id,
            "frame_id": self.frame_id,
            "route": self.route,
            "answer": self.answer,
        }


class SubmissionQueue:
    """In-memory, manually curated pick list for the web UI (spec section 17).

    A single-process, single-user competition tool: a plain list guarded by
    a lock is enough — no database, no persistence across restarts.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: list[SubmissionItem] = []
        self._next_id = itertools.count(1)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [item.as_dict() for item in self._items]

    def add(
        self,
        *,
        video_id: str,
        frame_id: int,
        route: str = "kis",
        answer: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(video_id, str) or not video_id.strip():
            raise ValueError("video_id must be a nonblank string")
        if isinstance(frame_id, bool) or not isinstance(frame_id, int):
            raise ValueError("frame_id must be an integer")
        item = SubmissionItem(
            id=next(self._next_id),
            video_id=video_id,
            frame_id=frame_id,
            route=route,
            answer=answer,
        )
        with self._lock:
            self._items.append(item)
        return item.as_dict()

    def remove(self, item_id: int) -> bool:
        with self._lock:
            before = len(self._items)
            self._items = [item for item in self._items if item.id != item_id]
            return len(self._items) != before

    def move(self, item_id: int, direction: str) -> bool:
        if direction not in ("up", "down"):
            raise ValueError("direction must be 'up' or 'down'")
        with self._lock:
            index = next(
                (position for position, item in enumerate(self._items) if item.id == item_id),
                None,
            )
            if index is None:
                return False
            target = index - 1 if direction == "up" else index + 1
            if target < 0 or target >= len(self._items):
                return False
            self._items[index], self._items[target] = self._items[target], self._items[index]
            return True

    def export_csv(self, output_path: str | Path, *, max_results: int = 100) -> Path:
        with self._lock:
            rows = [(item.video_id, item.frame_id) for item in self._items[:max_results]]
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerows(rows)
        return path


__all__ = ["SubmissionItem", "SubmissionQueue"]
