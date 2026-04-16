"""OpenAI-backed runtime selection for final follow-up questions."""

from __future__ import annotations

import json
import os
import streamlit as st
from dataclasses import replace
from enum import StrEnum
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

from followup_selector import (
    QuestionCandidate,
    QuestionSelectionResult,
    ReviewSignals,
    Willingness,
    _adapt_candidate,
    _question_budget,
    _selection_strategy,
    build_openai_decision_payload,
    classify_willingness,
    select_followup_questions,
)

OPENAI_SELECTION_INSTRUCTIONS = """
You are the final selection layer for a hotel review follow-up system.

Your job is not to generate new questions.
Your job is to choose the best final question or questions from an existing
offline-curated candidate bank.

The candidate bank has already been heavily processed offline for relevance,
quality, business value, and ranking. Treat it as the source of truth.

You will receive:
- runtime review signals from the current user
- a willingness classification derived from the user's review behavior
- a selection policy with the current budget and UX strategy
- an offline-curated candidate bank

Core product goal:
Maximize the value of the follow-up answers while matching the user's observed
willingness to engage. Prefer quality when possible, but protect completion rate
by keeping friction aligned to the user.

Willingness policy:
- low: the user only provided a minimal star rating signal. They are showing low
  tolerance for friction. Favor one lightweight, easy, broadly answerable follow-up.
- medium: the user provided at least two structured ratings, which signals some
  willingness to engage. Favor one or two structured follow-ups with moderate effort.
- high: the user wrote review text, which signals willingness to provide richer
  input. Favor the highest-quality opportunities, including ones that support
  optional text, while still staying concise.

Selection rules:
1. Respect the current budget exactly. Never select more candidates than allowed.
2. Only return candidate IDs that already exist in the input.
3. Prefer the highest-value candidates from the curated bank.
4. Avoid redundant selections that ask about the same facet unless there is a
   very strong reason, and generally prefer facet diversity.
5. Prefer questions that are easy for this user to answer firsthand.
6. For low willingness, favor simple verification or multiple-choice style prompts.
7. For medium willingness, favor structured questions that add signal without
   feeling like work.
8. For high willingness, favor the best unresolved opportunities, especially when
   optional text could produce richer insight.
9. If several candidates are close, pick the ones with better clarity, better
   answerability, and stronger business value.
10. Do not explain the whole policy back. Just make the decision.

Output requirements:
- Return selected_candidate_ids in the final order they should be shown.
- Return questions in the same final order.
- Each question must include:
  - candidate_id
  - question
  - response_format
- response_format must be exactly one of: LOW, MEDIUM, HIGH
- LOW means the UI should render the lightest response interaction.
- MEDIUM means the UI should render a balanced structured interaction.
- HIGH means the UI should render the richest interaction pattern.
- Return a short rationale that explains why these candidates are the best fit
  for this user's willingness and the product goal.
""".strip()


class QuestionResponseFormat(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class OpenAISelectedQuestion(BaseModel):
    """Typed question payload returned by the final OpenAI selection layer."""

    candidate_id: str = Field(default="")
    question: str = Field(default="")
    response_format: QuestionResponseFormat = Field(default=QuestionResponseFormat.MEDIUM)


class OpenAISelectionDecision(BaseModel):
    """Structured output from the OpenAI runtime selection call."""

    selected_candidate_ids: list[str] = Field(default_factory=list)
    questions: list[OpenAISelectedQuestion] = Field(default_factory=list)
    rationale: str = Field(default="")


def select_followup_questions_with_openai(
    review: ReviewSignals,
    candidates: list[QuestionCandidate],
    *,
    model: str = "gpt-5.4",
    api_key: str | None = None,
    client: Any | None = None,
) -> QuestionSelectionResult:
    """Use OpenAI to choose which curated candidates survive final synthesis."""

    fallback_result = select_followup_questions(review, candidates)
    if not candidates:
        return fallback_result

    parsed_decision = _request_openai_decision(
        review=review,
        candidates=candidates,
        model=model,
        api_key=api_key,
        client=client,
    )

    if parsed_decision is None:
        return fallback_result

    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    parsed_question_by_id = {
        question.candidate_id: question
        for question in parsed_decision.questions
        if question.candidate_id
    }
    chosen_ids: list[str] = []
    for candidate_id in parsed_decision.selected_candidate_ids:
        if candidate_id in candidate_by_id and candidate_id not in chosen_ids:
            chosen_ids.append(candidate_id)
    if not chosen_ids:
        for candidate_id in parsed_question_by_id:
            if candidate_id in candidate_by_id and candidate_id not in chosen_ids:
                chosen_ids.append(candidate_id)

    if not chosen_ids:
        return fallback_result

    willingness = classify_willingness(review)
    question_budget = _question_budget(willingness, review.answered_star_count)
    selected_candidates: list[QuestionCandidate] = []
    used_facets: set[str] = set()
    for candidate_id in chosen_ids:
        candidate = candidate_by_id[candidate_id]
        candidate_facets = set(candidate.facet_ids)
        if candidate_facets and candidate_facets & used_facets:
            continue
        selected_candidates.append(candidate)
        used_facets.update(candidate_facets)
        if len(selected_candidates) >= question_budget:
            break

    selected_questions = tuple(
        _build_final_question(candidate, willingness, parsed_decision.rationale, parsed_question_by_id)
        for candidate in selected_candidates
    )
    suppressed = tuple(
        candidate.candidate_id
        for candidate in candidates
        if candidate.candidate_id not in {question.candidate_id for question in selected_questions}
    )

    return QuestionSelectionResult(
        willingness=willingness,
        selected_question_count=len(selected_questions),
        selected_questions=selected_questions,
        suppressed_candidate_ids=suppressed,
        selection_strategy=_selection_strategy(willingness),
        backend="openai_responses_parse",
    )


def _request_openai_decision(
    *,
    review: ReviewSignals,
    candidates: list[QuestionCandidate],
    model: str,
    api_key: str | None,
    client: Any | None,
) -> OpenAISelectionDecision | None:
    try:
        runtime_client = client or OpenAI(api_key=_resolve_api_key(api_key))
        payload = build_openai_decision_payload(review, candidates)
        response = runtime_client.responses.parse(
            model=model,
            instructions=OPENAI_SELECTION_INSTRUCTIONS,
            input=json.dumps(payload),
            text_format=OpenAISelectionDecision,
        )
        return response.output_parsed
    except Exception:
        return None


def _resolve_api_key(api_key: str | None) -> str:
    resolved = api_key or st.secrets.get("OPENAI_API_KEY")
    if resolved:
        return resolved
    raise RuntimeError("OPENAI_API_KEY is not set.")


def _build_final_question(
    candidate: QuestionCandidate,
    willingness: Willingness,
    rationale: str,
    parsed_question_by_id: dict[str, OpenAISelectedQuestion],
):
    parsed_question = parsed_question_by_id.get(candidate.candidate_id)
    effective_willingness = (
        _map_response_format_to_willingness(parsed_question.response_format)
        if parsed_question is not None
        else willingness
    )
    adapted = _adapt_candidate(candidate, effective_willingness)
    final_question_text = adapted.question
    if parsed_question is not None and parsed_question.question:
        final_question_text = parsed_question.question
    return replace(
        adapted,
        question=final_question_text,
        why_selected=rationale or adapted.why_selected,
    )


def _map_response_format_to_willingness(response_format: QuestionResponseFormat) -> Willingness:
    if response_format is QuestionResponseFormat.LOW:
        return Willingness.LOW
    if response_format is QuestionResponseFormat.HIGH:
        return Willingness.HIGH
    return Willingness.MEDIUM
