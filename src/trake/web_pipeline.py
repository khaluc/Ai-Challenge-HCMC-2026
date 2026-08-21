from __future__ import annotations

from pathlib import Path
from typing import Any

from llm.event_parser import QwenEventDecomposer
from retrieval.clip_encoder import HFCLIPTextEncoder
from retrieval.hybrid_search import HybridTextualKIS
from retrieval.hybrid_store import Phase1HybridStore
from vlm.frame_verifier import QwenFrameEventScorer

from .aggregation import VideoRetrievalConfig
from .dense_frame_search import DenseFrame, decode_dense_window
from .frame_refinement import FineAlignmentConfig, FineFrameAligner, SemanticKeyframe
from .temporal_dp import CoarseTemporalAligner
from .video_catalog import VideoCatalog
from .video_retrieval import TRAKEVideoRetrieval


class TRAKEWebPipeline:
    """Bundles Stage 7-10 TRAKE services for the web UI.

    `search()` runs decompose -> video candidates -> coarse align on the top
    candidate in one call, so the Event Panel (spec section 11) can render
    immediately. `dense_frames()`/`verify()` are called on demand per event
    when the user presses Refine / Verify with VLM (sections 13-14).
    """

    def __init__(
        self,
        *,
        hybrid_engine: HybridTextualKIS,
        hybrid_store: Phase1HybridStore,
        encoder: HFCLIPTextEncoder,
        video_catalog: VideoCatalog,
        data_root: str | Path,
        llm_model: str = "qwen3.8-max",
        vlm_model: str | None = None,
        llm_base_url: str | None = None,
        video_retrieval_config: VideoRetrievalConfig | None = None,
    ) -> None:
        self.data_root = str(data_root)
        self.decomposer = QwenEventDecomposer(model=llm_model, base_url=llm_base_url)
        self.video_retrieval = TRAKEVideoRetrieval(
            hybrid_engine, hybrid_store, self.decomposer, config=video_retrieval_config
        )
        self.coarse_aligner = CoarseTemporalAligner(hybrid_store, encoder)
        self.video_catalog = video_catalog
        # `QwenFrameEventScorer` looks at images (Verify with VLM), so it needs
        # a vision-capable model — falls back to `llm_model` only if the
        # caller doesn't have a separate vision model configured.
        self.scorer = QwenFrameEventScorer(model=vlm_model or llm_model, base_url=llm_base_url)

    def search(self, query: str, *, query_id: str = "trake") -> dict[str, Any]:
        retrieval = self.video_retrieval.find_candidate_videos(query, query_id=query_id)
        top_alignment = None
        if retrieval.candidates:
            top_video_id = retrieval.candidates[0].video_id
            coarse = self.coarse_aligner.align(
                top_video_id, list(retrieval.event_sequence.events)
            )
            top_alignment = coarse.as_dict()
        return {
            "query_id": retrieval.query_id,
            "query": retrieval.query,
            "event_sequence": retrieval.event_sequence.as_dict(),
            "candidates": [candidate.as_dict() for candidate in retrieval.candidates],
            "top_alignment": top_alignment,
        }

    def align(self, video_id: str, events: list[str]) -> dict[str, Any]:
        return self.coarse_aligner.align(video_id, events).as_dict()

    def dense_frames(
        self,
        video_id: str,
        coarse_timestamp: float,
        *,
        window_seconds: float = 2.5,
        step_seconds: float = 0.5,
    ) -> list[DenseFrame]:
        info = self.video_catalog.get(video_id)
        return decode_dense_window(
            info,
            center_seconds=coarse_timestamp,
            window_seconds=window_seconds,
            step_seconds=step_seconds,
            data_root=self.data_root,
        )

    def verify(
        self,
        video_id: str,
        event_text: str,
        coarse_timestamp: float,
        *,
        window_seconds: float = 2.5,
        step_seconds: float = 0.5,
        max_candidate_frames: int = 12,
    ) -> SemanticKeyframe:
        aligner = FineFrameAligner(
            self.video_catalog,
            self.scorer,
            data_root=self.data_root,
            config=FineAlignmentConfig(
                window_seconds=window_seconds,
                step_seconds=step_seconds,
                max_candidate_frames=max_candidate_frames,
            ),
        )
        return aligner.refine(video_id, event_text, coarse_timestamp)


__all__ = ["TRAKEWebPipeline"]
