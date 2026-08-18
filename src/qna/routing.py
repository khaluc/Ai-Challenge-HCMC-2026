from __future__ import annotations

from typing import Protocol

from retrieval.processing import TOKEN_RE, fold_text

from .schemas import QAResult


class TranscriptQAProtocol(Protocol):
    def answer(self, question: str, *, query_id: str) -> QAResult:
        """Answer a question using ASR/transcript text plus an LLM."""


# Vietnamese/English phrases that mark a question about what was *said*
# rather than what is visible. Deliberately conservative: false negatives
# (routing a speech question to the VLM) just get a visual best guess;
# false positives (routing a visual question to transcript) get an honest
# "no transcript available" note instead of an answer.
SPEECH_QUESTION_PHRASES: tuple[str, ...] = (
    "dien gia noi",
    "nguoi noi",
    "noi ve",
    "nhac den",
    "de cap",
    "phat bieu",
    "phat ngon",
    "trong loi noi",
    "gioi thieu ve",
    "what did he say",
    "what did she say",
    "what did they say",
    "what did the speaker say",
    "what does the speaker say",
    "what is the speaker talking about",
    "speaker mention",
    "speaker say",
    "speaker talk",
    "according to the speaker",
    "mentioned in the speech",
    "mentioned during the talk",
)


def is_speech_question(question: str) -> bool:
    padded = f" {' '.join(TOKEN_RE.findall(fold_text(question)))} "
    for phrase in SPEECH_QUESTION_PHRASES:
        normalized = " ".join(TOKEN_RE.findall(fold_text(phrase)))
        if normalized and f" {normalized} " in padded:
            return True
    return False


__all__ = ["SPEECH_QUESTION_PHRASES", "TranscriptQAProtocol", "is_speech_question"]
