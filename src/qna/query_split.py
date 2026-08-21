from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class QuerySplit:
    """A VQA query broken into its retrieval half and its question half."""

    original: str
    scene_description: str
    question: str

    def __post_init__(self) -> None:
        if not self.original.strip():
            raise ValueError("QuerySplit.original must not be blank")
        if not self.scene_description.strip():
            raise ValueError("QuerySplit.scene_description must not be blank")
        if not self.question.strip():
            raise ValueError("QuerySplit.question must not be blank")

    def as_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "scene_description": self.scene_description,
            "question": self.question,
        }


class QuestionSplitterProtocol(Protocol):
    def split(self, text: str) -> QuerySplit:
        """Split a VQA query into a scene description (for retrieval) and question (for the VLM)."""


SPLIT_RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scene_description": {"type": "string"},
        "question": {"type": "string"},
    },
    "required": ["scene_description", "question"],
    "additionalProperties": False,
}

SPLIT_SYSTEM_PROMPT = (
    "You split one Video Question Answering query (Vietnamese or English) into "
    'two parts. Return a JSON object only: {"scene_description": string, '
    '"question": string}. "scene_description" is a short English sentence '
    "describing the visual scene the query is about — concrete objects, "
    "actions, and setting only, no question words, phrased as a plain "
    "description suitable for a CLIP-style image search (e.g. 'an awards "
    "ceremony with people standing on a stage'). \"question\" is the original "
    "question, copied verbatim, unchanged, in its original language — this is "
    "what gets asked to the answering model, never rewritten, translated, or "
    "summarized. If the query already reads as a plain question with no "
    "separable scene description, derive a minimal scene description yourself "
    "from the question's own content (never leave it blank)."
)


def _build_split(text: str, payload: dict[str, Any]) -> QuerySplit:
    scene_description = str(payload.get("scene_description") or "").strip()
    question = str(payload.get("question") or "").strip() or text.strip()
    if not scene_description:
        scene_description = text.strip()
    return QuerySplit(original=text, scene_description=scene_description, question=question)


class QwenQuestionSplitter:
    """LLM-backed query splitter via Qwen (OpenAI-compatible endpoint).

    No rule-based fallback: telling "what the scene looks like" apart from
    "what is being asked" needs real language understanding — same reasoning
    TRAKE's `llm.event_parser.QwenEventDecomposer` documents for why event
    decomposition has no regex/keyword rule that generalizes.
    """

    DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str = "qwen3.8-max",
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        if client is None:
            import openai  # local import: optional dependency

            resolved_key = (
                api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
            )
            if not resolved_key:
                raise ValueError(
                    "No Qwen/DashScope API key found: pass api_key= or set "
                    "DASHSCOPE_API_KEY (or QWEN_API_KEY) in the environment"
                )
            client = openai.OpenAI(
                api_key=resolved_key,
                base_url=base_url or os.environ.get("DASHSCOPE_BASE_URL") or self.DEFAULT_BASE_URL,
            )
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    def split(self, text: str) -> QuerySplit:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a nonblank string")
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            response_format={"type": "json_object"},
            # Reasoning models can spend the whole max_tokens budget on
            # hidden chain-of-thought and return empty content otherwise —
            # not needed for this short structured-output task.
            extra_body={"enable_thinking": False},
            messages=[
                {"role": "system", "content": SPLIT_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        )
        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "content_filter":
            raise RuntimeError(f"Qwen declined query split for text={text!r}")
        payload_text = choice.message.content
        if not payload_text or not payload_text.strip():
            raise RuntimeError(f"Qwen returned an empty response for text={text!r}")
        payload = json.loads(payload_text)
        return _build_split(text, payload)


__all__ = [
    "QuerySplit",
    "QuestionSplitterProtocol",
    "QwenQuestionSplitter",
    "SPLIT_RESPONSE_JSON_SCHEMA",
    "SPLIT_SYSTEM_PROMPT",
]
