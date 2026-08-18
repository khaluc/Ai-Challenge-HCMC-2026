from __future__ import annotations

import base64
import json
import numbers
import os
from typing import Any, Protocol

from .schemas import VLMAnswer


class VLMAnswererProtocol(Protocol):
    def answer(self, image_bytes: bytes, question: str, *, question_type: str = "open_ended") -> VLMAnswer:
        """Answer a question about one keyframe image."""


VLM_RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["answer", "confidence"],
    "additionalProperties": False,
}

# Two prompt variants, selected by `qna.question_type.classify_question_type`.
# The brief that asked for this routing named separate models per bucket
# (LLaVA/Qwen-VL/InternVL2 fine-tuned for closed-set; GPT-4V/Claude/Gemini for
# open-ended) — this project only has a Qwen/DashScope API key configured, so
# routing here adapts the *prompt* for the one model actually available
# instead of pretending to call APIs with no credentials in this environment.
OPEN_ENDED_SYSTEM_PROMPT = (
    "You answer a short Video Question Answering query about exactly one "
    "keyframe image from a video. Look only at the image provided. Return a "
    'JSON object only: {"answer": string, "confidence": number}. "answer" is '
    "the shortest correct answer to the question (a word or short phrase, "
    "not a sentence), in the same language as the question. If the image "
    "does not contain enough information to answer confidently, still give "
    "your best guess in \"answer\" but set \"confidence\" low. "
    '"confidence" is your own calibrated probability, from 0.0 (pure guess) '
    "to 1.0 (certain), that \"answer\" is correct for this exact image."
)

CLOSED_SET_SYSTEM_PROMPT = (
    "You answer a short Video Question Answering query about exactly one "
    "keyframe image from a video. Look only at the image provided. This is a "
    "CLOSED-SET question — counting, color, or yes/no. Return a JSON object "
    'only: {"answer": string, "confidence": number}. "answer" must be the '
    "shortest exact value only: a bare number for counting, a single color "
    "word for color, or 'Yes'/'No' (translated to the question's language) "
    "for yes/no — never a sentence, never extra words. If the image does not "
    "contain enough information to answer confidently, still give your best "
    "guess in this exact format but set \"confidence\" low. \"confidence\" is "
    "your own calibrated probability, from 0.0 (pure guess) to 1.0 (certain), "
    "that \"answer\" is correct for this exact image."
)

# Kept as the default export name so existing imports (`VLM_SYSTEM_PROMPT`)
# keep working; it is the open-ended variant.
VLM_SYSTEM_PROMPT = OPEN_ENDED_SYSTEM_PROMPT


class QwenVLAnswerer:
    """VLM adapter over a Qwen vision-language model via an OpenAI-compatible
    endpoint (Alibaba Cloud Model Studio / DashScope by default).

    Requires the optional `openai` package and API credentials; a `client`
    can be injected for testing without any network access or dependency.
    """

    DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str = "qwen3.8-max",
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 512,
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

    def answer(self, image_bytes: bytes, question: str, *, question_type: str = "open_ended") -> VLMAnswer:
        if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
            raise ValueError("image_bytes must be nonempty bytes")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a nonblank string")
        if question_type not in ("open_ended", "closed_set"):
            raise ValueError("question_type must be 'open_ended' or 'closed_set'")
        system_prompt = CLOSED_SET_SYSTEM_PROMPT if question_type == "closed_set" else OPEN_ENDED_SYSTEM_PROMPT
        encoded = base64.b64encode(bytes(image_bytes)).decode("ascii")
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                        },
                        {"type": "text", "text": question},
                    ],
                },
            ],
        )
        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "content_filter":
            raise RuntimeError(f"Qwen VLM declined to answer question={question!r}")
        payload_text = choice.message.content
        if not payload_text or not payload_text.strip():
            raise RuntimeError(f"Qwen VLM returned an empty response for question={question!r}")
        payload = json.loads(payload_text)
        confidence = payload.get("confidence", 0.0)
        if isinstance(confidence, bool) or not isinstance(confidence, numbers.Real):
            confidence = 0.0
        confidence = min(1.0, max(0.0, float(confidence)))
        return VLMAnswer(answer=str(payload["answer"]), confidence=confidence)


__all__ = [
    "CLOSED_SET_SYSTEM_PROMPT",
    "OPEN_ENDED_SYSTEM_PROMPT",
    "QwenVLAnswerer",
    "VLMAnswererProtocol",
    "VLM_RESPONSE_JSON_SCHEMA",
    "VLM_SYSTEM_PROMPT",
]
