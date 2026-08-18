from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

cv2 = pytest.importorskip("cv2")

from trake import frame_refinement as fine_alignment_module
from trake.dense_frame_search import DenseFrame, decode_dense_window
from trake.frame_refinement import FineAlignmentConfig, FineFrameAligner
from trake.video_catalog import VideoCatalog, VideoInfo
from vlm.frame_verifier import FrameEventScore, QwenFrameEventScorer


def _write_synthetic_video(path: Path, *, fps: float = 10.0, seconds: float = 2.0) -> None:
    size = (64, 48)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    total_frames = int(fps * seconds)
    for index in range(total_frames):
        frame = np.full((size[1], size[0], 3), fill_value=(index * 5) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_video_catalog_lookup(tmp_path: Path) -> None:
    table = pa.table(
        {
            "Video_ID": ["L21_V001", "L21_V002"],
            "Video_Path": [
                "Videos_L21.zip::video/L21_V001.mp4",
                "Videos_L21.zip::video/L21_V002.mp4",
            ],
            "FPS": [30.0, 25.0],
            "Video_Available": [True, False],
        }
    )
    parquet_path = tmp_path / "frame_mapping.parquet"
    pq.write_table(table, parquet_path)

    catalog = VideoCatalog(parquet_path)
    info = catalog.get("L21_V001")
    assert info.video_path == "Videos_L21.zip::video/L21_V001.mp4"
    assert info.fps == 30.0

    with pytest.raises(FileNotFoundError):
        catalog.get("L21_V002")

    with pytest.raises(KeyError):
        catalog.get("L21_V999")


def test_decode_dense_window_extracts_from_zip_and_decodes_frames(tmp_path: Path) -> None:
    video_file = tmp_path / "clip.mp4"
    _write_synthetic_video(video_file, fps=10.0, seconds=2.0)

    archive_path = tmp_path / "Videos_TEST.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(video_file, "video/TEST_V001.mp4")

    info = VideoInfo(
        video_id="TEST_V001",
        video_path="Videos_TEST.zip::video/TEST_V001.mp4",
        fps=10.0,
        video_available=True,
    )

    frames = decode_dense_window(
        info, center_seconds=1.0, window_seconds=0.5, step_seconds=0.25, data_root=tmp_path
    )

    assert len(frames) >= 3
    assert all(frame.video_id == "TEST_V001" for frame in frames)
    assert all(len(frame.image_bytes) > 0 for frame in frames)
    assert all(0.4 <= frame.timestamp <= 1.6 for frame in frames)


def test_decode_dense_window_rejects_unqualified_path(tmp_path: Path) -> None:
    info = VideoInfo(video_id="X", video_path="no-archive-marker.mp4", fps=25.0, video_available=True)
    with pytest.raises(ValueError):
        decode_dense_window(info, center_seconds=1.0, data_root=tmp_path)


class _FakeScorer:
    def __init__(self, scores_by_timestamp: dict[float, FrameEventScore]) -> None:
        self._scores = scores_by_timestamp
        self.calls: list[tuple[float, str]] = []

    def score(self, image_bytes: bytes, event_text: str) -> FrameEventScore:
        timestamp = float(image_bytes.decode("utf-8"))
        self.calls.append((timestamp, event_text))
        return self._scores[timestamp]


class _FakeCatalog:
    def get(self, video_id: str) -> VideoInfo:
        return VideoInfo(video_id=video_id, video_path="fake::x.mp4", fps=25.0, video_available=True)


def test_fine_frame_aligner_picks_highest_confidence_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_frames = [
        DenseFrame(video_id="V1", timestamp=t, image_bytes=str(t).encode("utf-8"))
        for t in (0.5, 0.75, 1.0, 1.25, 1.5)
    ]
    monkeypatch.setattr(fine_alignment_module, "decode_dense_window", lambda *a, **k: fake_frames)

    scores = {
        0.5: FrameEventScore(False, 0.1, "too early"),
        0.75: FrameEventScore(True, 0.6, "close"),
        1.0: FrameEventScore(True, 0.95, "exact moment"),
        1.25: FrameEventScore(True, 0.7, "slightly late"),
        1.5: FrameEventScore(False, 0.2, "too late"),
    }
    scorer = _FakeScorer(scores)

    aligner = FineFrameAligner(
        _FakeCatalog(),
        scorer,
        config=FineAlignmentConfig(window_seconds=0.5, step_seconds=0.25, max_candidate_frames=10),
    )
    result = aligner.refine("V1", "athlete takes off", coarse_timestamp=1.0)

    assert result.timestamp == 1.0
    assert result.confidence == pytest.approx(0.95)
    assert result.matches is True
    assert len(scorer.calls) == 5


def test_fine_frame_aligner_subsamples_when_over_the_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_frames = [
        DenseFrame(video_id="V1", timestamp=float(i), image_bytes=str(float(i)).encode("utf-8"))
        for i in range(20)
    ]
    monkeypatch.setattr(fine_alignment_module, "decode_dense_window", lambda *a, **k: fake_frames)
    scorer = _FakeScorer({float(i): FrameEventScore(True, 0.5, "r") for i in range(20)})

    aligner = FineFrameAligner(
        _FakeCatalog(), scorer, config=FineAlignmentConfig(max_candidate_frames=5)
    )
    result = aligner.refine("V1", "event", coarse_timestamp=10.0)

    assert result.candidates_scored <= 5
    assert len(scorer.calls) <= 5


class _FakeChatMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _FakeChatChoice:
    def __init__(self, content: str | None, finish_reason: str) -> None:
        self.message = _FakeChatMessage(content)
        self.finish_reason = finish_reason


class _FakeChatCompletion:
    def __init__(self, content: str | None, finish_reason: str) -> None:
        self.choices = [_FakeChatChoice(content, finish_reason)]


class _FakeChatCompletions:
    def __init__(self, content: str | None, finish_reason: str) -> None:
        self._content = content
        self._finish_reason = finish_reason
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeChatCompletion(self._content, self._finish_reason)


class _FakeChat:
    def __init__(self, content: str | None, finish_reason: str) -> None:
        self.completions = _FakeChatCompletions(content, finish_reason)


class _FakeOpenAICompatibleClient:
    def __init__(self, content: str | None, finish_reason: str = "stop") -> None:
        self.chat = _FakeChat(content, finish_reason)


def test_qwen_frame_event_scorer_parses_response() -> None:
    payload = {"matches": True, "confidence": 0.87, "reason": "foot leaving the ground"}
    client = _FakeOpenAICompatibleClient(json.dumps(payload))
    scorer = QwenFrameEventScorer(client=client, model="qwen3.8-max")

    result = scorer.score(b"fake-bytes", "athlete taking off")

    assert result.matches is True
    assert result.confidence == pytest.approx(0.87)
    assert result.reason == "foot leaving the ground"
    assert client.chat.completions.last_kwargs["model"] == "qwen3.8-max"


def test_qwen_frame_event_scorer_raises_on_content_filter() -> None:
    client = _FakeOpenAICompatibleClient(None, finish_reason="content_filter")
    scorer = QwenFrameEventScorer(client=client)
    with pytest.raises(RuntimeError):
        scorer.score(b"fake-bytes", "event")


def test_qwen_frame_event_scorer_requires_api_key_without_injected_client() -> None:
    with pytest.raises(ValueError):
        QwenFrameEventScorer(api_key=None)
