from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReviewRecord:
    review_id: str
    eg_property_id: str
    acquisition_date: str
    overall_rating: float | None
    review_title: str
    review_text: str


@dataclass(frozen=True)
class ComplaintSnippet:
    snippet_id: str
    review_id: str
    eg_property_id: str
    acquisition_date: str
    text: str
    overall_rating: float | None


@dataclass(frozen=True)
class ThemeDefinition:
    key: str
    label: str
    prototypes: tuple[str, ...]


@dataclass(frozen=True)
class ThemeMatch:
    theme_key: str
    theme_label: str
    confidence: float
