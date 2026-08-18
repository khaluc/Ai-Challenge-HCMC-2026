from __future__ import annotations

import numbers
from dataclasses import dataclass

from .dense_frame_search import DenseFrame, decode_dense_window
from .video_catalog import VideoCatalog
from vlm.frame_verifier import FrameEventScorerProtocol


@dataclass(frozen=True)
class FineAlignmentConfig:
    """Only decode a small window, not the whole video — ground-truth TRAKE
    windows can be under 10 frames wide."""

    window_seconds: float = 2.5
    step_seconds: float = 0.5
    max_candidate_frames: int = 12

    def __post_init__(self) -> None:
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if self.step_seconds <= 0:
            raise ValueError("step_seconds must be positive")
        if isinstance(self.max_candidate_frames, bool) or not isinstance(
            self.max_candidate_frames, numbers.Integral
        ):
            raise TypeError("max_candidate_frames must be an integer")
        if self.max_candidate_frames < 1:
            raise ValueError("max_candidate_frames must be a positive integer")


@dataclass(frozen=True)
class SemanticKeyframe:
    video_id: str
    event_text: str
    coarse_timestamp: float
    timestamp: float
    confidence: float
    matches: bool
    reason: str
    candidates_scored: int


def _subsample(frames: list[DenseFrame], limit: int) -> list[DenseFrame]:
    if len(frames) <= limit:
        return frames
    step = len(frames) / limit
    indices = sorted({int(i * step) for i in range(limit)})
    return [frames[i] for i in indices]


class FineFrameAligner:
    """Coarse timestamp -> decode ±window -> VLM-score each dense candidate
    -> best-matching semantic keyframe."""

    def __init__(
        self,
        video_catalog: VideoCatalog,
        scorer: FrameEventScorerProtocol,
        *,
        data_root: str = ".",
        config: FineAlignmentConfig | None = None,
    ) -> None:
        self.video_catalog = video_catalog
        self.scorer = scorer
        self.data_root = data_root
        self.config = config or FineAlignmentConfig()

    def refine(self, video_id: str, event_text: str, coarse_timestamp: float) -> SemanticKeyframe:
        if not isinstance(event_text, str) or not event_text.strip():
            raise ValueError("event_text must be a nonblank string")
        if coarse_timestamp < 0:
            raise ValueError("coarse_timestamp must be non-negative")

        info = self.video_catalog.get(video_id)
        dense_frames = decode_dense_window(
            info,
            center_seconds=coarse_timestamp,
            window_seconds=self.config.window_seconds,
            step_seconds=self.config.step_seconds,
            data_root=self.data_root,
        )
        if not dense_frames:
            raise RuntimeError(
                f"No frames could be decoded for {video_id!r} around t={coarse_timestamp}s"
            )
        dense_frames = _subsample(dense_frames, self.config.max_candidate_frames)

        scored = [
            (frame, self.scorer.score(frame.image_bytes, event_text)) for frame in dense_frames
        ]
        scored.sort(key=lambda item: -item[1].confidence)
        best_frame, best_score = scored[0]

        return SemanticKeyframe(
            video_id=video_id,
            event_text=event_text,
            coarse_timestamp=coarse_timestamp,
            timestamp=best_frame.timestamp,
            confidence=best_score.confidence,
            matches=best_score.matches,
            reason=best_score.reason,
            candidates_scored=len(scored),
        )


__all__ = ["FineAlignmentConfig", "FineFrameAligner", "SemanticKeyframe"]
