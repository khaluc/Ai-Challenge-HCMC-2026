from __future__ import annotations

from typing import Literal

from retrieval.processing import TOKEN_RE, fold_text

QuestionType = Literal["closed_set", "open_ended"]

# Deterministic, rule-based classifier — the spec's distinction here (counting
# / color / yes-no vs. descriptive) is a shallow lexical pattern, unlike the
# scene/question split or event decomposition, which need real language
# understanding. No LLM call needed, so this stays free and instant.

COUNTING_PHRASES: tuple[str, ...] = (
    "bao nhieu",
    "may nguoi",
    "may vat",
    "so luong",
    "so nguoi",
    "how many",
    "what number",
    "count of",
    "number of",
)

COLOR_PHRASES: tuple[str, ...] = (
    "mau gi",
    "mau sac gi",
    "mau nao",
    "what color",
    "which color",
    "what colour",
)

# Vietnamese "co ... khong" wraps the whole clause; English yes/no questions
# front-load an auxiliary verb. Both are checked on the folded (no-diacritics,
# lowercased) text.
YES_NO_ENGLISH_STARTS: tuple[str, ...] = (
    "is ",
    "are ",
    "was ",
    "were ",
    "does ",
    "did ",
    "do ",
    "can ",
    "could ",
    "will ",
    "would ",
    "has ",
    "have ",
)


def _is_yes_no_question(folded_tokens: list[str]) -> bool:
    if not folded_tokens:
        return False
    folded_text = " ".join(folded_tokens)
    if folded_text.startswith(YES_NO_ENGLISH_STARTS):
        return True
    if folded_tokens[0] == "co" and folded_tokens[-1] == "khong":
        return True
    return False


def _contains_any(folded_text: str, phrases: tuple[str, ...]) -> bool:
    padded = f" {folded_text} "
    for phrase in phrases:
        normalized = " ".join(TOKEN_RE.findall(fold_text(phrase)))
        if normalized and f" {normalized} " in padded:
            return True
    return False


def classify_question_type(question: str) -> QuestionType:
    """Closed-set (counting/color/yes-no) vs. open-ended (descriptive) — the
    spec's basis for routing to a different answer strategy."""
    tokens = TOKEN_RE.findall(fold_text(question))
    folded_text = " ".join(tokens)
    if _contains_any(folded_text, COUNTING_PHRASES):
        return "closed_set"
    if _contains_any(folded_text, COLOR_PHRASES):
        return "closed_set"
    if _is_yes_no_question(tokens):
        return "closed_set"
    return "open_ended"


__all__ = [
    "COLOR_PHRASES",
    "COUNTING_PHRASES",
    "QuestionType",
    "YES_NO_ENGLISH_STARTS",
    "classify_question_type",
]
