from __future__ import annotations

import base64
import json
import numbers
import os
from typing import Any, Protocol


class FrameEventScorerProtocol(Protocol):
    def score(self, image_bytes: bytes, event_text: str) -> "FrameEventScore":
        """Rate how well one frame depicts one specific event/moment."""


class FrameEventScore:
    __slots__ = ("matches", "confidence", "reason")

    def __init__(self, matches: bool, confidence: float, reason: str) -> None:
        if not isinstance(matches, bool):
            raise TypeError("FrameEventScore.matches must be a bool")
        if isinstance(confidence, bool) or not isinstance(confidence, numbers.Real):
            raise TypeError("FrameEventScore.confidence must be numeric")
        confidence = float(confidence)
        if not 0 <= confidence <= 1:
            raise ValueError("FrameEventScore.confidence must be between 0 and 1")
        if not isinstance(reason, str):
            raise TypeError("FrameEventScore.reason must be a string")
        self.matches = matches
        self.confidence = confidence
        self.reason = reason

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return f"FrameEventScore(matches={self.matches!r}, confidence={self.confidence!r}, reason={self.reason!r})"


FRAME_EVENT_RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "matches": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["matches", "confidence", "reason"],
    "additionalProperties": False,
}

FRAME_EVENT_SYSTEM_PROMPT = (
    "You judge whether exactly one video keyframe depicts one specific "
    "moment (event) from a described action sequence, for aligning a "
    "sequence of events to individual frames. Look only at the image "
    'provided. Return a JSON object only: {"matches": boolean, '
    '"confidence": number, "reason": string}. "matches" is true only if '
    "this exact frame visually shows that specific moment — not a moment "
    "clearly before or after it in the same action. \"confidence\" is your "
    "calibrated probability (0.0 to 1.0) that this is a strong instance of "
    "that moment, independent of other candidate frames you have not seen. "
    '"reason" is one short phrase (not a full sentence) naming the visual '
    "evidence you used."
)


class QwenFrameEventScorer:
    """VLM adapter used by both Stage 10 (fine alignment) and Stage 11
    (verification): rate one frame against one event's semantic definition.

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
        max_tokens: int = 256,
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

    def score(self, image_bytes: bytes, event_text: str) -> FrameEventScore:
        if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
            raise ValueError("image_bytes must be nonempty bytes")
        if not isinstance(event_text, str) or not event_text.strip():
            raise ValueError("event_text must be a nonblank string")
        encoded = base64.b64encode(bytes(image_bytes)).decode("ascii")
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": FRAME_EVENT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                        },
                        {"type": "text", "text": f"Event: {event_text}"},
                    ],
                },
            ],
        )
        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "content_filter":
            raise RuntimeError(f"Qwen VLM declined to score event={event_text!r}")
        payload_text = choice.message.content
        if not payload_text or not payload_text.strip():
            raise RuntimeError(f"Qwen VLM returned an empty response for event={event_text!r}")
        payload = json.loads(payload_text)
        confidence = payload.get("confidence", 0.0)
        if isinstance(confidence, bool) or not isinstance(confidence, numbers.Real):
            confidence = 0.0
        confidence = min(1.0, max(0.0, float(confidence)))
        return FrameEventScore(
            matches=bool(payload.get("matches", False)),
            confidence=confidence,
            reason=str(payload.get("reason", "")),
        )


__all__ = [
    "FRAME_EVENT_RESPONSE_JSON_SCHEMA",
    "FRAME_EVENT_SYSTEM_PROMPT",
    "FrameEventScore",
    "FrameEventScorerProtocol",
    "QwenFrameEventScorer",
]
