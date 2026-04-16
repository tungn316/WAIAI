from __future__ import annotations

import re

_SENTENCE_BREAK_RE = re.compile(r"(?:[\r\n]+|(?<=[.!?;])\s+)")
_CLAUSE_BREAK_RE = re.compile(r"\s+(?:but|however|though|although|yet|except|while)\s+", re.IGNORECASE)
_COMMA_BREAK_RE = re.compile(
    r",\s+(?=(?:the|our|my|their|it|there|room|bathroom|front|staff|parking|location|wifi|ac|air|bed|shower|toilet|pool|breakfast)\b)",
    re.IGNORECASE,
)


def normalize_review_text(text: str) -> str:
    normalized = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", normalized).strip()


def split_review_into_snippets(text: str) -> list[str]:
    normalized = normalize_review_text(text)
    if not normalized:
        return []

    chunks: list[str] = []
    for sentence in _SENTENCE_BREAK_RE.split(normalized):
        sentence = sentence.strip(" -,\t")
        if not sentence:
            continue
        chunks.extend(_split_sentence(sentence))

    deduped: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        cleaned = re.sub(r"\s+", " ", chunk).strip(" ,.-")
        if len(cleaned.split()) < 4:
            continue
        fingerprint = cleaned.casefold()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped.append(cleaned)

    return deduped or [normalized]


def _split_sentence(sentence: str) -> list[str]:
    sentence_parts: list[str] = [sentence]
    split_once: list[str] = []
    for part in sentence_parts:
        split_once.extend(piece.strip() for piece in _CLAUSE_BREAK_RE.split(part) if piece.strip())

    results: list[str] = []
    for part in split_once:
        words = part.split()
        if len(words) > 24 and part.count(",") >= 2:
            comma_parts = [piece.strip() for piece in _COMMA_BREAK_RE.split(part) if piece.strip()]
            if len(comma_parts) > 1:
                results.extend(comma_parts)
                continue
        results.append(part)
    return results
