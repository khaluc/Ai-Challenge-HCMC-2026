"""Candidate-narrowed VLM Q&A on top of the hybrid KIS pipeline (Stage 5)."""

from .candidates import CandidateConfig, QACandidateSearch, narrow_candidates
from .pipeline import KISVideoQA
from .routing import TranscriptQAProtocol, is_speech_question
from .schemas import QACandidate, QAResult, VLMAnswer

__all__ = [
    "CandidateConfig",
    "KISVideoQA",
    "QACandidate",
    "QACandidateSearch",
    "QAResult",
    "TranscriptQAProtocol",
    "VLMAnswer",
    "is_speech_question",
    "narrow_candidates",
]
