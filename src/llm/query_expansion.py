from __future__ import annotations

import json
import numbers
from typing import Any, Iterable, Mapping, Protocol

from retrieval.processing import TOKEN_RE, RuleBasedObjectParser, fold_text

from .expansion_schemas import QueryStructure, QueryUnderstanding


class QueryUnderstandingProtocol(Protocol):
    def understand(self, text: str) -> QueryUnderstanding:
        """Return the object/attribute/relation structure and query expansions."""


# Canonical English attribute -> Vietnamese/English trigger phrases (folded, no
# diacritics). Multi-word phrases only for anything that collides with common
# short syllables when accents are stripped (e.g. bare "nam" is both "male"
# and the folded form of "nam"/"nam" as in "Vietnam" or "nam" the year unit).
ATTRIBUTE_LEXICON: dict[str, tuple[str, ...]] = {
    "red": ("do", "mau do", "red"),
    "white": ("trang", "mau trang", "white"),
    "black": ("den", "mau den", "black"),
    "blue": ("xanh duong", "xanh nuoc bien", "blue"),
    "green": ("xanh la", "xanh la cay", "green"),
    "yellow": ("vang", "mau vang", "yellow"),
    "orange": ("mau cam", "orange"),
    "brown": ("nau", "mau nau", "brown"),
    "gray": ("xam", "mau xam", "gray", "grey"),
    "pink": ("hong", "mau hong", "pink"),
    "male": ("nguoi dan ong", "dan ong", "nam gioi", "male", "man"),
    "female": ("nguoi phu nu", "phu nu", "nu gioi", "female", "woman"),
    "child": ("tre em", "be trai", "be gai", "child", "kid"),
    "elderly": ("nguoi gia", "lon tuoi", "elderly", "old"),
}

# Canonical relation -> Vietnamese/English trigger phrases (folded), and the
# English gloss phrases used when generating CLIP-friendly expansions.
RELATION_LEXICON: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("next_to", ("ben canh", "canh", "ke ben", "gan", "next to", "beside", "near")),
    ("in_front_of", ("phia truoc", "truoc", "in front of")),
    ("behind", ("phia sau", "sau", "behind")),
    ("on", ("tren", "on top of")),
    ("inside", ("ben trong", "trong", "inside")),
    ("holding", ("dang cam", "cam", "holding", "carrying")),
    ("riding", ("dang lai", "cuoi", "riding", "driving")),
)

RELATION_ENGLISH_PHRASES: dict[str, tuple[str, ...]] = {
    "next_to": ("next to", "beside", "close to"),
    "in_front_of": ("in front of",),
    "behind": ("behind",),
    "on": ("on",),
    "inside": ("inside",),
    "holding": ("holding", "carrying"),
    "riding": ("riding", "driving"),
}


def _normalize_phrase(phrase: str) -> str:
    return " ".join(TOKEN_RE.findall(fold_text(phrase)))


def _find_span(padded: str, phrases: Iterable[str]) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    for phrase in phrases:
        normalized = _normalize_phrase(phrase)
        if not normalized:
            continue
        pattern = f" {normalized} "
        index = padded.find(pattern)
        if index < 0:
            continue
        span = (index + 1, index + len(pattern) - 1)
        if best is None or span[0] < best[0] or (span[0] == best[0] and span[1] > best[1]):
            best = span
    return best


def _detect_relation(padded: str) -> tuple[str | None, tuple[int, int] | None]:
    best_relation: str | None = None
    best_span: tuple[int, int] | None = None
    for canonical, phrases in RELATION_LEXICON:
        span = _find_span(padded, phrases)
        if span is None:
            continue
        if (
            best_span is None
            or span[0] < best_span[0]
            or (span[0] == best_span[0] and (span[1] - span[0]) > (best_span[1] - best_span[0]))
        ):
            best_relation, best_span = canonical, span
    return best_relation, best_span


def _detect_attributes(segment: str) -> tuple[str, ...]:
    padded = f" {segment} "
    found: list[str] = []
    for canonical, phrases in ATTRIBUTE_LEXICON.items():
        if _find_span(padded, phrases) is not None:
            found.append(canonical)
    return tuple(found)


class RuleBasedQueryUnderstanding:
    """Deterministic object/attribute/relation parser and expansion generator.

    Splits the query around the first relation keyword found (if any), runs
    the existing object concept parser on each side, and binds attributes to
    a side only when that side names exactly one concept. Expansions are
    template phrases built from concepts (already English detector-concept
    names) plus attribute/relation glosses; a query with no detected concepts
    abstains and expands to just the original text.
    """

    def __init__(
        self,
        object_parser: RuleBasedObjectParser,
        *,
        max_expansions: int = 4,
    ) -> None:
        if isinstance(max_expansions, bool) or not isinstance(max_expansions, numbers.Integral):
            raise TypeError("max_expansions must be an integer")
        if int(max_expansions) < 1:
            raise ValueError("max_expansions must be a positive integer")
        self._object_parser = object_parser
        self._max_expansions = int(max_expansions)

    def understand(self, text: str) -> QueryUnderstanding:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a nonblank string")
        padded = f" {fold_text(text)} "
        relation, span = _detect_relation(padded)

        if relation is None:
            left_segment = padded.strip()
            right_segment = ""
        else:
            start, end = span  # type: ignore[misc]
            left_segment = padded[:start].strip()
            right_segment = padded[end:].strip()

        objects: list[str] = []
        attributes: dict[str, tuple[str, ...]] = {}
        for segment in (left_segment, right_segment):
            if not segment:
                continue
            parsed = self._object_parser.parse(segment)
            for concept in parsed.concepts:
                if concept not in objects:
                    objects.append(concept)
            if len(parsed.concepts) == 1:
                found_attributes = _detect_attributes(segment)
                if found_attributes:
                    attributes[parsed.concepts[0]] = found_attributes

        structure = QueryStructure(
            objects=tuple(objects),
            attributes=attributes,
            relation=relation if len(objects) >= 2 else None,
        )
        expansions = self._expand(text, structure)
        return QueryUnderstanding(text=text, structure=structure, expansions=expansions)

    def _expand(self, text: str, structure: QueryStructure) -> tuple[str, ...]:
        variants: list[str] = []

        def phrase_for(concept: str) -> str:
            attrs = structure.attributes.get(concept, ())
            return f"{' '.join(attrs)} {concept}".strip()

        if structure.relation is not None and len(structure.objects) >= 2:
            first, second = structure.objects[0], structure.objects[1]
            relation_phrases = RELATION_ENGLISH_PHRASES.get(structure.relation, ())
            for relation_phrase in relation_phrases:
                variants.append(f"{phrase_for(first)} {relation_phrase} {phrase_for(second)}")
        elif len(structure.objects) == 1:
            concept = structure.objects[0]
            attrs = structure.attributes.get(concept, ())
            if attrs:
                variants.append(phrase_for(concept))

        ordered: list[str] = [text]
        seen = {text.strip().casefold()}
        for variant in variants:
            key = variant.strip().casefold()
            if not variant.strip() or key in seen:
                continue
            seen.add(key)
            ordered.append(variant.strip())
            if len(ordered) == self._max_expansions:
                break
        return tuple(ordered)


# Shared by every LLM-backed adapter: same JSON contract, same system prompt,
# same payload-to-QueryUnderstanding conversion, so every backend is a thin
# wrapper around "call the model, get this JSON back".
LLM_RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "objects": {"type": "array", "items": {"type": "string"}},
        "attributes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string"},
                    "values": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["concept", "values"],
                "additionalProperties": False,
            },
        },
        "relation": {"type": ["string", "null"]},
        "expansions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["objects", "attributes", "relation", "expansions"],
    "additionalProperties": False,
}

LLM_SYSTEM_PROMPT = (
    "You analyze a Known-Item Search query (Vietnamese or English) about a "
    "video keyframe. Return a JSON object only, matching this schema: "
    '{"objects": [string], "attributes": [{"concept": string, "values": '
    '[string]}], "relation": string|null, "expansions": [string]}. '
    '"objects" is a list of concrete, physical object nouns mentioned in '
    'the query, in English, singular (e.g. "person", "car"), limited to '
    "objects a generic object detector could plausibly localize. "
    '"attributes" binds visual attributes (colors, gender, age) to one of '
    'the objects already listed in "objects". Each attributes[i].concept '
    'value MUST be copied verbatim from one of the strings in "objects" — '
    'never a category name such as "color" or "gender". attributes[i].'
    '"values" is the list of attribute words for that object (e.g. "red", '
    '"male"). Omit an object from "attributes" entirely if you are not '
    "confident about any of its attributes. "
    '"relation" is a short snake_case spatial or interaction relation '
    'between two of the objects (e.g. "next_to", "holding", "in_front_of"), '
    "or null if there is no clear two-object relation. "
    '"expansions" is 2 to 4 alternate English phrasings of the query '
    "suitable for a CLIP-style image/text similarity search; do not "
    "invent objects or attributes not present in the query.\n"
    'Example — for "a man in a red shirt standing next to a white car", a '
    'correct response is {"objects": ["person", "car"], "attributes": '
    '[{"concept": "person", "values": ["male", "red"]}, {"concept": "car", '
    '"values": ["white"]}], "relation": "next_to", "expansions": '
    '["man in red shirt beside white car", "person in red clothing next '
    'to a white vehicle"]}.'
)


def _build_understanding_from_payload(
    text: str, payload: Mapping[str, Any], *, max_expansions: int
) -> QueryUnderstanding:
    objects = tuple(dict.fromkeys(str(value) for value in payload.get("objects", ())))
    attributes: dict[str, tuple[str, ...]] = {}
    for entry in payload.get("attributes", ()):
        concept = str(entry["concept"])
        if concept not in objects:
            continue
        values = tuple(dict.fromkeys(str(value) for value in entry.get("values", ())))
        if values:
            attributes[concept] = values
    relation = payload.get("relation")
    relation = str(relation) if relation else None
    structure = QueryStructure(
        objects=objects,
        attributes=attributes,
        relation=relation if len(objects) >= 2 else None,
    )

    expansions_raw = [str(value) for value in payload.get("expansions", ()) if str(value).strip()]
    ordered: list[str] = [text]
    seen = {text.strip().casefold()}
    for expansion in expansions_raw:
        key = expansion.strip().casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(expansion.strip())
        if len(ordered) == max_expansions:
            break
    return QueryUnderstanding(text=text, structure=structure, expansions=tuple(ordered))


class AnthropicQueryUnderstanding:
    """LLM-backed adapter implementing QueryUnderstandingProtocol via Claude.

    Requires the optional `anthropic` package and API credentials (see the
    Anthropic Python SDK docs for credential resolution order). A `client`
    can be injected for testing without any network access or dependency.
    """

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str = "claude-opus-5",
        max_tokens: int = 1024,
        max_expansions: int = 4,
    ) -> None:
        if isinstance(max_expansions, bool) or not isinstance(max_expansions, numbers.Integral):
            raise TypeError("max_expansions must be an integer")
        if int(max_expansions) < 1:
            raise ValueError("max_expansions must be a positive integer")
        if client is None:
            import anthropic  # local import: optional dependency

            client = anthropic.Anthropic()
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._max_expansions = int(max_expansions)

    def understand(self, text: str) -> QueryUnderstanding:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a nonblank string")
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=LLM_SYSTEM_PROMPT,
            output_config={
                "format": {"type": "json_schema", "schema": LLM_RESPONSE_JSON_SCHEMA}
            },
            messages=[{"role": "user", "content": text}],
        )
        if getattr(response, "stop_reason", None) == "refusal":
            raise RuntimeError(f"Claude declined query understanding for text={text!r}")
        payload_text = next(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        payload = json.loads(payload_text)
        return _build_understanding_from_payload(
            text, payload, max_expansions=self._max_expansions
        )


class QwenQueryUnderstanding:
    """LLM-backed adapter implementing QueryUnderstandingProtocol via Qwen.

    Calls Qwen through an OpenAI-compatible endpoint (Alibaba Cloud
    DashScope's compatible-mode API by default) using the `openai` package.
    Requires the optional `openai` package and API credentials; a `client`
    can be injected for testing without any network access or dependency.
    """

    #: DashScope's international OpenAI-compatible endpoint. Override via
    #: `base_url=` or `DASHSCOPE_BASE_URL` for the mainland-China endpoint
    #: (https://dashscope.aliyuncs.com/compatible-mode/v1) or another
    #: OpenAI-compatible host serving the same model id.
    DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str = "qwen3.8-max",
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 1024,
        max_expansions: int = 4,
    ) -> None:
        if isinstance(max_expansions, bool) or not isinstance(max_expansions, numbers.Integral):
            raise TypeError("max_expansions must be an integer")
        if int(max_expansions) < 1:
            raise ValueError("max_expansions must be a positive integer")
        if client is None:
            import os

            import openai  # local import: optional dependency

            resolved_key = api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get(
                "QWEN_API_KEY"
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
        self._max_expansions = int(max_expansions)

    def understand(self, text: str) -> QueryUnderstanding:
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
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        )
        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "content_filter":
            raise RuntimeError(f"Qwen declined query understanding for text={text!r}")
        payload_text = choice.message.content
        if not payload_text or not payload_text.strip():
            raise RuntimeError(f"Qwen returned an empty response for text={text!r}")
        payload = json.loads(payload_text)
        return _build_understanding_from_payload(
            text, payload, max_expansions=self._max_expansions
        )


__all__ = [
    "ATTRIBUTE_LEXICON",
    "AnthropicQueryUnderstanding",
    "LLM_RESPONSE_JSON_SCHEMA",
    "LLM_SYSTEM_PROMPT",
    "QueryUnderstandingProtocol",
    "QwenQueryUnderstanding",
    "RELATION_LEXICON",
    "RuleBasedQueryUnderstanding",
]
