from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence


class TextEmbedderProtocol(Protocol):
    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one embedding vector per input text, in the same order."""


@dataclass(frozen=True)
class AnswerMatchResult:
    """One (predicted, ground_truth) pair scored by embedding similarity."""

    predicted: str
    ground_truth: str
    similarity: float
    threshold: float
    matched: bool

    def __post_init__(self) -> None:
        if not self.predicted.strip():
            raise ValueError("AnswerMatchResult.predicted must not be blank")
        if not self.ground_truth.strip():
            raise ValueError("AnswerMatchResult.ground_truth must not be blank")
        if not math.isfinite(self.similarity):
            raise ValueError("AnswerMatchResult.similarity must be finite")
        if not 0 <= self.threshold <= 1:
            raise ValueError("AnswerMatchResult.threshold must be between 0 and 1")

    def as_dict(self) -> dict[str, object]:
        return {
            "predicted": self.predicted,
            "ground_truth": self.ground_truth,
            "similarity": self.similarity,
            "threshold": self.threshold,
            "matched": self.matched,
        }


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def score_answers(
    rows: Sequence[tuple[str, str]],
    *,
    embedder: TextEmbedderProtocol,
    threshold: float = 0.85,
) -> list[AnswerMatchResult]:
    """Semantic answer matching (spec: "khớp ngữ nghĩa", similarity > 0.85).

    Batches every predicted/ground-truth string through the embedder in one
    call (cheaper than embedding pair-by-pair), then scores each row by
    cosine similarity against the given threshold.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if not rows:
        return []
    for index, (predicted, ground_truth) in enumerate(rows):
        if not predicted.strip() or not ground_truth.strip():
            raise ValueError(f"rows[{index}] must have nonblank predicted and ground_truth text")

    flat_texts = [text for pair in rows for text in pair]
    vectors = list(embedder.embed_texts(flat_texts))
    if len(vectors) != len(flat_texts):
        raise RuntimeError(
            f"embedder returned {len(vectors)} vectors for {len(flat_texts)} input texts"
        )

    results: list[AnswerMatchResult] = []
    for index, (predicted, ground_truth) in enumerate(rows):
        similarity = _cosine(vectors[2 * index], vectors[2 * index + 1])
        results.append(
            AnswerMatchResult(
                predicted=predicted,
                ground_truth=ground_truth,
                similarity=similarity,
                threshold=threshold,
                matched=similarity >= threshold,
            )
        )
    return results


class ClipTextEmbedder:
    """Adapts `retrieval.clip_encoder.HFCLIPTextEncoder` to `TextEmbedderProtocol`.

    The brief named Sentence-BERT for this step, but this project already
    loads a CLIP text tower locally for retrieval and `model.allow_download:
    false` (config.yaml) blocks fetching a new sentence-transformers
    checkpoint at runtime — so semantic answer matching reuses the encoder
    already resident in memory instead of adding a new model download.
    `encode()` already L2-normalizes, so cosine similarity here is exact.
    """

    def __init__(self, encoder: object) -> None:
        self._encoder = encoder

    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return self._encoder.encode(list(texts)).tolist()  # type: ignore[attr-defined]


__all__ = [
    "AnswerMatchResult",
    "ClipTextEmbedder",
    "TextEmbedderProtocol",
    "score_answers",
]
