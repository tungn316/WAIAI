"""Choose final follow-up questions from an offline-curated candidate bank."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Iterable

STAR_DIMENSIONS = ("overall", "room", "vibe")


class Willingness(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class ReviewSignals:
    """Signals collected from the user review form before follow-up selection."""

    star_ratings: dict[str, float | int | None] = field(default_factory=dict)
    review_text: str = ""
    existing_answer_count: int = 0

    @property
    def answered_star_count(self) -> int:
        return sum(
            1
            for key in STAR_DIMENSIONS
            if self.star_ratings.get(key) not in (None, "", 0)
        )

    @property
    def has_review_text(self) -> bool:
        return bool(self.review_text.strip())


@dataclass(frozen=True, slots=True)
class QuestionCandidate:
    """Candidate produced by the offline ranking pipeline."""

    candidate_id: str
    question: str
    response_type: str
    property_priority: float = 0.0
    offline_rank: int | None = None
    rationale: str = ""
    facet_ids: tuple[str, ...] = ()
    opportunity_id: str | None = None
    source_model: str | None = None


@dataclass(frozen=True, slots=True)
class FinalQuestion:
    """Question that should actually be shown to the user at runtime."""

    candidate_id: str
    question: str
    response_type: str
    ui_hint: str
    why_selected: str
    facet_ids: tuple[str, ...] = ()
    opportunity_id: str | None = None


@dataclass(frozen=True, slots=True)
class QuestionSelectionResult:
    willingness: Willingness
    selected_question_count: int
    selected_questions: tuple[FinalQuestion, ...]
    suppressed_candidate_ids: tuple[str, ...]
    selection_strategy: str
    backend: str = "rule_based"


def classify_willingness(review: ReviewSignals) -> Willingness:
    """Infer how much friction the user is likely to tolerate."""

    if review.has_review_text:
        w = Willingness.HIGH
    elif review.answered_star_count >= 2:
        w = Willingness.MEDIUM
    else: w = Willingness.LOW

    print(f"DEBUG: classify_willingness -> {w.value.upper()} (stars: {review.answered_star_count}, text_present: {review.has_review_text})")

    return w


def select_followup_questions(
    review: ReviewSignals,
    candidates: Iterable[QuestionCandidate],
) -> QuestionSelectionResult:
    """Pick final questions using offline priority plus runtime willingness."""

    willingness = classify_willingness(review)
    ordered_candidates = sorted(
        candidates,
        key=lambda candidate: (
            -candidate.property_priority,
            candidate.offline_rank if candidate.offline_rank is not None else 10**9,
            candidate.question.lower(),
        ),
    )

    question_budget = _question_budget(willingness, review.answered_star_count)
    selected: list[FinalQuestion] = []
    used_facets: set[str] = set()

    for candidate in ordered_candidates:
        candidate_facets = set(candidate.facet_ids)
        if candidate_facets and candidate_facets & used_facets:
            continue

        selected.append(_adapt_candidate(candidate, willingness))
        used_facets.update(candidate_facets)
        if len(selected) >= question_budget:
            break

    selected_ids = {question.candidate_id for question in selected}
    suppressed = tuple(
        candidate.candidate_id
        for candidate in ordered_candidates
        if candidate.candidate_id not in selected_ids
    )

    return QuestionSelectionResult(
        willingness=willingness,
        selected_question_count=len(selected),
        selected_questions=tuple(selected),
        suppressed_candidate_ids=suppressed,
        selection_strategy=_selection_strategy(willingness),
        backend="rule_based",
    )


def build_openai_decision_payload(
    review: ReviewSignals,
    candidates: Iterable[QuestionCandidate],
) -> dict[str, object]:
    """Create a structured payload for the final OpenAI selection call."""

    willingness = classify_willingness(review)
    candidate_list = list(candidates)

    return {
        "runtime_review_signals": {
            "star_ratings": {
                key: review.star_ratings.get(key) for key in STAR_DIMENSIONS
            },
            "answered_star_count": review.answered_star_count,
            "review_text_present": review.has_review_text,
            "review_text": review.review_text,
            "existing_answer_count": review.existing_answer_count,
            "willingness": willingness.value,
        },
        "selection_policy": {
            "goal": (
                "Prefer the highest-value question set while matching the user's "
                "observed willingness to answer follow-up prompts."
            ),
            "willingness_rules": {
                "low": (
                    "One answered star rating and no review text. Ask one easy "
                    "multiple-choice question."
                ),
                "medium": (
                    "At least two answered star ratings and no review text. Ask "
                    "one or two medium-friction questions."
                ),
                "high": (
                    "Any review text present. Ask up to two richer questions that "
                    "can support optional text answers."
                ),
            },
            "current_budget": _question_budget(willingness, review.answered_star_count),
            "current_strategy": _selection_strategy(willingness),
        },
        "offline_curated_candidates": [asdict(candidate) for candidate in candidate_list],
    }


def _question_budget(willingness: Willingness, answered_star_count: int) -> int:
    if willingness is Willingness.LOW:
        return 1
    if willingness is Willingness.MEDIUM:
        return 2 if answered_star_count >= 3 else 1
    return 2


def _selection_strategy(willingness: Willingness) -> str:
    if willingness is Willingness.LOW:
        return "single_easy_multiple_choice"
    if willingness is Willingness.MEDIUM:
        return "balanced_structured_followup"
    return "rich_optional_text_followup"


def _adapt_candidate(candidate: QuestionCandidate, willingness: Willingness) -> FinalQuestion:
    if willingness is Willingness.LOW:
        response_type = "multiple_choice"
        ui_hint = "Keep this lightweight with a single-tap answer."
    elif willingness is Willingness.MEDIUM:
        response_type = _medium_response_type(candidate.response_type)
        ui_hint = "Use a structured answer with optional short detail."
    else:
        response_type = _high_response_type(candidate.response_type)
        ui_hint = "Invite richer feedback while keeping text optional."

    why_selected = candidate.rationale or (
        "High offline priority and a good fit for the user's current answer depth."
    )

    return FinalQuestion(
        candidate_id=candidate.candidate_id,
        question=candidate.question,
        response_type=response_type,
        ui_hint=ui_hint,
        why_selected=why_selected,
        facet_ids=candidate.facet_ids,
        opportunity_id=candidate.opportunity_id,
    )


def _medium_response_type(response_type: str) -> str:
    if response_type in {"free_text", "open_text"}:
        return "scale_plus_optional_text"
    if response_type in {"boolean", "multiple_choice"}:
        return "multiple_choice"
    if response_type == "boolean_plus_optional_text":
        return "boolean_plus_optional_text"
    return "scale_plus_optional_text"


def _high_response_type(response_type: str) -> str:
    if response_type in {"free_text", "open_text"}:
        return response_type
    if response_type == "multiple_choice":
        return "multiple_choice_plus_optional_text"
    if response_type == "boolean":
        return "boolean_plus_optional_text"
    if response_type == "scale":
        return "scale_plus_optional_text"
    return response_type
