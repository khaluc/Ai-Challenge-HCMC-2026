from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .hybrid_schemas import QueryAnalysis


TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)


VIETNAMESE_STOPWORDS = frozenset(
    {
        "bi",
        "ben",
        "cac",
        "canh",
        "cho",
        "co",
        "cua",
        "dang",
        "duoc",
        "dung",
        "giua",
        "la",
        "mot",
        "nhung",
        "o",
        "tai",
        "theo",
        "tren",
        "trong",
        "tu",
        "va",
        "voi",
    }
)
ENGLISH_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "at",
        "by",
        "for",
        "from",
        "in",
        "is",
        "near",
        "next",
        "of",
        "on",
        "standing",
        "the",
        "to",
        "with",
    }
)
STOPWORDS = VIETNAMESE_STOPWORDS | ENGLISH_STOPWORDS

# Open Images contains Orange as the fruit class, but in free-form KIS queries
# the bare English word is overwhelmingly likely to be a color attribute. It
# therefore stays in the CLIP text and is not an automatic hard object concept.
AUTO_LABEL_BLOCKLIST = frozenset({"orange"})


# Canonical concepts map to one or more Open Images detector labels. Groups are
# intentionally conservative: they improve recall without pretending to infer
# color, attributes, or spatial relationships.
DEFAULT_CONCEPT_LABELS: dict[str, tuple[str, ...]] = {
    "person": ("Person", "Man", "Woman", "Boy", "Girl"),
    "car": ("Car", "Land vehicle", "Vehicle"),
    "motorcycle": ("Motorcycle",),
    "bicycle": ("Bicycle",),
    "bus": ("Bus",),
    "truck": ("Truck",),
    "train": ("Train",),
    "boat": ("Boat",),
    "airplane": ("Airplane", "Aircraft"),
    "mobile phone": ("Mobile phone", "Telephone", "Corded phone"),
    "laptop": ("Laptop",),
    "television": ("Television",),
    "microphone": ("Microphone",),
    "chair": ("Chair",),
    "table": ("Table",),
    "bottle": ("Bottle",),
    "cup": ("Coffee cup", "Measuring cup"),
    "traffic light": ("Traffic light",),
    "traffic sign": ("Traffic sign",),
    "tree": ("Tree",),
    "dog": ("Dog",),
    "cat": ("Cat",),
    "food": ("Food",),
}


DEFAULT_ALIASES: dict[str, tuple[str, ...]] = {
    "person": (
        "person",
        "people",
        "human",
        "nguoi",
        "nguoi dan ong",
        "nguoi phu nu",
    ),
    "car": ("car", "cars", "o to", "oto", "xe hoi", "xe con"),
    "motorcycle": ("motorcycle", "motorbike", "mo to", "xe may"),
    "bicycle": ("bicycle", "bike", "xe dap"),
    "bus": ("bus", "xe buyt"),
    "truck": ("truck", "lorry", "xe tai"),
    "train": ("train", "tau hoa", "xe lua"),
    "boat": ("boat", "ship", "thuyen", "tau thuy"),
    "airplane": ("airplane", "aircraft", "plane", "may bay"),
    "mobile phone": ("mobile phone", "cell phone", "smartphone", "dien thoai"),
    "laptop": ("laptop", "notebook", "may tinh xach tay"),
    "television": ("television", "tv", "tivi"),
    "microphone": ("microphone", "mic", "mi cro"),
    "chair": ("chair", "ghe", "cai ghe"),
    # Accent folding makes `bàn` and the common pronoun `bạn` identical, so a
    # bare `ban` alias would create many false object filters.
    "table": ("table", "desk", "cai ban"),
    "bottle": ("bottle", "chai", "chai nuoc", "binh nuoc"),
    "cup": ("cup", "mug", "coc", "cai coc", "ly nuoc"),
    "traffic light": ("traffic light", "den giao thong", "den tin hieu"),
    "traffic sign": ("traffic sign", "bien bao", "bien giao thong"),
    "tree": ("tree", "cay"),
    # `cho` is also a very common Vietnamese function word; require the noun
    # phrase instead of silently treating every occurrence as a dog.
    "dog": ("dog", "con cho"),
    "cat": ("cat", "con meo"),
    "food": ("food", "do an", "thuc an", "mon an"),
}


def fold_text(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("text must be a string")
    value = value.replace("Đ", "D").replace("đ", "d")
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(
        character for character in normalized if not unicodedata.combining(character)
    ).casefold()


def tokenize_for_metadata(text: str, *, max_terms: int = 16) -> tuple[str, ...]:
    if isinstance(max_terms, bool) or not isinstance(max_terms, int) or max_terms < 1:
        raise ValueError("max_terms must be a positive integer")
    tokens = TOKEN_RE.findall(fold_text(text))
    selected: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in STOPWORDS or (len(token) < 2 and not token.isdigit()):
            continue
        if token in seen:
            continue
        seen.add(token)
        selected.append(token)
        if len(selected) == max_terms:
            break
    return tuple(selected)


@dataclass(frozen=True)
class ParsedObjectQuery:
    concepts: tuple[str, ...]
    labels_by_concept: Mapping[str, tuple[str, ...]]


class RuleBasedObjectParser:
    """Extract detector concepts with exact phrase matching and aliases.

    The class is deliberately replaceable by an LLM-backed parser later. It
    never claims to understand color or spatial relations; it only emits object
    concepts that the supplied Faster R-CNN vocabulary can actually search.
    """

    def __init__(
        self,
        detector_labels: Iterable[str],
        *,
        concept_labels: Mapping[str, tuple[str, ...]] = DEFAULT_CONCEPT_LABELS,
        aliases: Mapping[str, tuple[str, ...]] = DEFAULT_ALIASES,
        max_concepts: int = 6,
    ) -> None:
        if isinstance(max_concepts, bool) or not isinstance(max_concepts, int) or max_concepts < 1:
            raise ValueError("max_concepts must be a positive integer")
        available = {label.casefold(): label for label in detector_labels}
        groups: dict[str, tuple[str, ...]] = {}
        default_concept_by_label: dict[str, str] = {}
        for concept, labels in concept_labels.items():
            present = tuple(
                available[label.casefold()]
                for label in labels
                if label.casefold() in available
            )
            if present:
                groups[concept] = present
                for label in present:
                    default_concept_by_label.setdefault(label.casefold(), concept)

        phrase_to_concept: dict[str, str] = {}
        for concept, phrases in aliases.items():
            if concept not in groups:
                continue
            for phrase in phrases:
                normalized = " ".join(TOKEN_RE.findall(fold_text(phrase)))
                if normalized:
                    phrase_to_concept[normalized] = concept

        # Every exact detector label is also searchable in English, including
        # labels not covered by the hand-written Vietnamese alias map.
        for label in available.values():
            normalized = " ".join(TOKEN_RE.findall(fold_text(label)))
            if not normalized or normalized in AUTO_LABEL_BLOCKLIST:
                continue
            concept = default_concept_by_label.get(
                label.casefold(), phrase_to_concept.get(normalized, normalized)
            )
            groups.setdefault(concept, (label,))
            phrase_to_concept.setdefault(normalized, concept)

        self._labels_by_concept = groups
        self._max_concepts = max_concepts
        self._phrases = tuple(
            sorted(phrase_to_concept.items(), key=lambda item: (-len(item[0]), item[0]))
        )

    def parse(self, text: str) -> ParsedObjectQuery:
        normalized = " ".join(TOKEN_RE.findall(fold_text(text)))
        padded = f" {normalized} "
        concepts: list[str] = []
        occupied: list[tuple[int, int]] = []
        for phrase, concept in self._phrases:
            if concept in concepts:
                continue
            pattern = f" {phrase} "
            search_from = 0
            while True:
                start = padded.find(pattern, search_from)
                if start < 0:
                    break
                # Exclude the separator spaces from the occupied interval so
                # telegraphic adjacent nouns such as `person car` both survive.
                content_start = start + 1
                content_end = start + len(pattern) - 1
                if not any(
                    content_start < other_end and content_end > other_start
                    for other_start, other_end in occupied
                ):
                    concepts.append(concept)
                    occupied.append((content_start, content_end))
                    break
                search_from = start + 1
            if len(concepts) == self._max_concepts:
                break
        return ParsedObjectQuery(
            concepts=tuple(concepts),
            labels_by_concept={
                concept: self._labels_by_concept[concept] for concept in concepts
            },
        )


def analyze_query(text: str, object_parser: RuleBasedObjectParser) -> QueryAnalysis:
    parsed = object_parser.parse(text)
    return QueryAnalysis(
        text=text,
        metadata_terms=tokenize_for_metadata(text),
        object_concepts=parsed.concepts,
    )


__all__ = [
    "DEFAULT_ALIASES",
    "DEFAULT_CONCEPT_LABELS",
    "ParsedObjectQuery",
    "RuleBasedObjectParser",
    "analyze_query",
    "fold_text",
    "tokenize_for_metadata",
]
