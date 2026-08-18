"""TRAKE: video retrieval, coarse DP alignment, fine VLM alignment, and verification."""

from .aggregation import VideoRetrievalConfig, aggregate_video_candidates
from .dense_frame_search import DenseFrame, decode_dense_window, extract_video_to_cache
from .frame_refinement import FineAlignmentConfig, FineFrameAligner, SemanticKeyframe
from .frame_verification import TRAKEVerifier
from .frame_verification_schemas import (
    OriginalEventFrame,
    TRAKEVerificationResult,
    VerificationCandidate,
    VerifiedEvent,
)
from .temporal_dp import AlignmentConfig, CoarseTemporalAligner, align_events
from .temporal_dp_schemas import CoarseAlignment, EventFrameAssignment
from .video_catalog import VideoCatalog, VideoInfo
from .video_retrieval import TRAKEVideoRetrieval
from .video_retrieval_schemas import TRAKERetrievalResult, VideoCandidateScore
from .web_pipeline import TRAKEWebPipeline

__all__ = [
    "AlignmentConfig",
    "CoarseAlignment",
    "CoarseTemporalAligner",
    "DenseFrame",
    "EventFrameAssignment",
    "FineAlignmentConfig",
    "FineFrameAligner",
    "OriginalEventFrame",
    "SemanticKeyframe",
    "TRAKERetrievalResult",
    "TRAKEVerificationResult",
    "TRAKEVerifier",
    "TRAKEVideoRetrieval",
    "TRAKEWebPipeline",
    "VerificationCandidate",
    "VerifiedEvent",
    "VideoCandidateScore",
    "VideoCatalog",
    "VideoInfo",
    "VideoRetrievalConfig",
    "aggregate_video_candidates",
    "align_events",
    "decode_dense_window",
    "extract_video_to_cache",
]
