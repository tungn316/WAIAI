# app.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
import urllib.request

import pandas as pd
import streamlit as st

from belief_system import AspectBelief, update_belief_from_answer
from context_profile import ContextProfile
from followup_selector import (
        FinalQuestion,
        QuestionCandidate,
        QuestionSelectionResult,
        ReviewSignals,
        Willingness,
        select_followup_questions,
        )
from openai_runtime import select_followup_questions_with_openai
from question_generator import (
        enrich_candidates_for_cpsat,
        generate_question_candidates,
        )
from llm_selector import llm_select_candidates

PROFILES_URL = "https://raw.githubusercontent.com/tungn316/WAIAI/refs/heads/main/src/outputs/property_profiles.json"
DESCRIPTIONS_URL = "https://raw.githubusercontent.com/tungn316/WAIAI/refs/heads/main/src/data/Description_PROC.csv"

_WILLINGNESS_LABELS: dict[Willingness, tuple[str, str]] = {
        Willingness.LOW: ("🟡", "Quick mode — we'll keep it short"),
        Willingness.MEDIUM: (
            "🟠",
            "Balanced — a couple of structured questions",
            ),
        Willingness.HIGH: (
            "🟢",
            "Detailed mode — richer follow-ups enabled",
            ),
        }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_profiles() -> dict:
    try:
        with urllib.request.urlopen(PROFILES_URL) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        st.error(f"Failed to load property profiles: {e}")
        return {}


@st.cache_data
def load_descriptions() -> pd.DataFrame:
    return pd.read_csv(DESCRIPTIONS_URL)


profiles = load_profiles()
descriptions = load_descriptions()


def deserialize_beliefs(raw: dict) -> dict[str, AspectBelief]:
    return {k: AspectBelief(**v) for k, v in raw.items()}


def deserialize_context(raw: dict) -> ContextProfile:
    return ContextProfile(**raw)


def format_property_label(pid: str) -> str:
    row = descriptions[
            descriptions["eg_property_id"].astype(str) == str(pid)
            ]
    if row.empty:
        return pid
    r = row.iloc[0]
    city = r.get("city", "")
    stars = r.get("star_rating", "")
    star_str = (
            f" {'⭐' * int(float(stars))}" if pd.notna(stars) else ""
            )
    return f"{pid} — {city}{star_str}"


# ---------------------------------------------------------------------------
# Greedy selector (replaces CP-SAT)
# ---------------------------------------------------------------------------

# Weights for the greedy sort key
_W_COMPOSITE = 1.0
_W_STALE_BONUS = 0.25
_W_MISSING_BONUS = 0.35
_W_LOW_CONFIDENCE_BONUS = 0.15
_W_DIVERSITY_BONUS = 0.10


def _greedy_sort_key(
        candidate: dict[str, Any],
        beliefs: dict[str, AspectBelief] | None,
        ) -> float:
    """Higher is better."""
    score = candidate.get("composite_score", 0.0) * _W_COMPOSITE

    aspect = candidate.get("aspect", "")
    if beliefs and aspect in beliefs:
        b = beliefs[aspect]
        if b.is_missing:
            score += _W_MISSING_BONUS
        elif b.is_stale:
            score += _W_STALE_BONUS
        if b.confidence < 0.3:
            score += (0.3 - b.confidence) * _W_LOW_CONFIDENCE_BONUS
    elif beliefs:
        # Aspect not in beliefs at all — treat as missing
        score += _W_MISSING_BONUS

    return score


def greedy_select(
        candidates: list[dict[str, Any]],
        *,
        question_budget: int = 2,
        token_budget: int = 800,
        beliefs: dict[str, AspectBelief] | None = None,
        ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Pure-Python greedy selector.

    1. Sort candidates by composite score + belief bonuses (descending).
    2. Walk the sorted list and greedily pick candidates that satisfy:
       - token budget not exceeded
       - no duplicate aspect
       - pool diversity: prefer covering ≥ 2 pools when budget ≥ 2
    3. Return (selected, stats).
    """
    if not candidates:
        return [], {
                "status": "NO_CANDIDATES",
                "candidates_total": 0,
                "candidates_selected": 0,
                "token_budget": token_budget,
                "tokens_used": 0,
                }

    # Trivial path: everything fits
    if len(candidates) <= question_budget:
        total_cost = sum(
                c.get("estimated_token_cost", 0) for c in candidates
                )
        if total_cost <= token_budget:
            for c in candidates:
                c["selected"] = True
            return list(candidates), {
                    "status": "TRIVIAL",
                    "candidates_total": len(candidates),
                    "candidates_selected": len(candidates),
                    "token_budget": token_budget,
                    "tokens_used": total_cost,
                    }

    # Score and sort descending
    scored = sorted(
            candidates,
            key=lambda c: _greedy_sort_key(c, beliefs),
            reverse=True,
            )

    selected: list[dict[str, Any]] = []
    used_aspects: set[str] = set()
    used_pools: set[str] = set()
    tokens_used = 0

    # --- Pass 1: greedy pick with diversity preference ---
    deferred: list[dict[str, Any]] = []

    for cand in scored:
        if len(selected) >= question_budget:
            break

        aspect = cand.get("aspect", "")
        pool = cand.get("cluster_pool", "Pool A")
        cost = cand.get("estimated_token_cost", 150)

        # Skip duplicate aspect
        if aspect in used_aspects:
            continue

        # Skip if token budget would be exceeded
        if tokens_used + cost > token_budget:
            continue

        # If we already have 1 pick and this is the same pool,
        # defer it — we might find a different-pool candidate
        if (
                question_budget >= 2
                and len(selected) == 1
                and pool in used_pools
                and len(deferred) < 3
                ):
            deferred.append(cand)
            continue

        cand["selected"] = True
        selected.append(cand)
        used_aspects.add(aspect)
        used_pools.add(pool)
        tokens_used += cost

    # --- Pass 2: backfill from deferred if budget remains ---
    for cand in deferred:
        if len(selected) >= question_budget:
            break

        aspect = cand.get("aspect", "")
        cost = cand.get("estimated_token_cost", 150)

        if aspect in used_aspects:
            continue
        if tokens_used + cost > token_budget:
            continue

        cand["selected"] = True
        selected.append(cand)
        used_aspects.add(aspect)
        used_pools.add(cand.get("cluster_pool", "Pool A"))
        tokens_used += cost

    # Mark unselected
    selected_ids = {id(c) for c in selected}
    for cand in candidates:
        if id(cand) not in selected_ids:
            cand["selected"] = False

    pool_dist: dict[str, int] = {}
    for c in selected:
        p = c.get("cluster_pool", "Pool A")
        pool_dist[p] = pool_dist.get(p, 0) + 1

    stats = {
            "status": "GREEDY",
            "candidates_total": len(candidates),
            "candidates_selected": len(selected),
            "token_budget": token_budget,
            "tokens_used": tokens_used,
            "tokens_remaining": token_budget - tokens_used,
            "pool_distribution": pool_dist,
            "aspects_covered": sorted(used_aspects),
            }
    return selected, stats


# ---------------------------------------------------------------------------
# Candidate conversion
# ---------------------------------------------------------------------------


def raw_to_candidate(q: dict[str, Any]) -> QuestionCandidate:
    return QuestionCandidate(
            candidate_id=q["candidate_id"],
            question=q["question"],
            response_type=q.get("suggested_response_type", "free_text"),
            property_priority=q.get(
                "composite_score", q.get("property_priority", 0.0)
                ),
            offline_rank=q.get("offline_rank"),
            rationale=q.get("rationale", ""),
            facet_ids=(q.get("aspect", ""),),
            opportunity_id=None,
            source_model="gpt-4.1-nano",
            )


# ---------------------------------------------------------------------------
# Widget rendering
# ---------------------------------------------------------------------------


def render_question_widget(
        final_q: FinalQuestion,
        widget_key: str,
        ) -> str | None:
    rt = final_q.response_type
    label = final_q.question
    hint = final_q.ui_hint

    if rt == "multiple_choice":
        options = ["Yes", "No", "Not sure"]
        answer = st.radio(
                label, options, key=widget_key, horizontal=True
                )
        st.caption(hint)
        return answer

    if rt == "multiple_choice_plus_optional_text":
        options = ["Yes", "No", "Not sure"]
        choice = st.radio(
                label, options, key=f"{widget_key}_radio", horizontal=True
                )
        detail = st.text_input(
                "Any details? _(optional)_",
                key=f"{widget_key}_text",
                )
        st.caption(hint)
        return (
                f"{choice}. {detail}".strip(". ") if detail else choice
                )

    if rt == "boolean":
        choice = st.radio(
                label, ["Yes", "No"], key=widget_key, horizontal=True
                )
        st.caption(hint)
        return choice

    if rt == "boolean_plus_optional_text":
        choice = st.radio(
                label,
                ["Yes", "No"],
                key=f"{widget_key}_radio",
                horizontal=True,
                )
        detail = st.text_input(
                "Anything to add? _(optional)_",
                key=f"{widget_key}_text",
                )
        st.caption(hint)
        return (
                f"{choice}. {detail}".strip(". ") if detail else choice
                )

    if rt == "scale_plus_optional_text":
        scale = st.slider(
                label, 1, 5, 3, key=f"{widget_key}_scale"
                )
        detail = st.text_input(
                "Want to elaborate? _(optional)_",
                key=f"{widget_key}_text",
                )
        st.caption(hint)
        return (
                f"Rating: {scale}/5. {detail}".strip()
                if detail
                else f"Rating: {scale}/5"
                )

    # free_text / open_text / fallback
    answer = st.text_input(label, key=widget_key)
    st.caption(hint)
    return answer


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(
        page_title="Smart Review Assistant", page_icon="🏨"
        )
st.title("🏨 Smart Review Assistant")
st.caption(
        "Your feedback helps future travellers make better decisions."
        )

# --- Property selection ---
available_pids = [
        pid
        for pid in profiles
        if pid in descriptions["eg_property_id"].astype(str).values
        ]

if not available_pids:
    st.error(
            "No property profiles found. "
            "Run `python -m precompute` first."
            )
    st.stop()

property_id = st.selectbox(
        "Which property are you reviewing?",
        options=available_pids,
        format_func=format_property_label,
        )

profile = profiles[property_id]

if (
        "beliefs" not in st.session_state
        or st.session_state.get("loaded_pid") != property_id
        ):
    st.session_state.beliefs = deserialize_beliefs(profile["beliefs"])
    st.session_state.context = deserialize_context(profile["context"])
    st.session_state.raw_candidates = []
    st.session_state.enriched_candidates = []
    st.session_state.optimized_candidates = []
    st.session_state.selection_result = None
    st.session_state.answers = {}
    st.session_state.submitted = False
    st.session_state.loaded_pid = property_id

context: ContextProfile = st.session_state.context

# --- Property snapshot ---
with st.expander("📄 Property Info", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tier", context.tier.title())
    col2.metric("Stars", context.star_rating or "N/A")
    col3.metric("Avg Rating", context.avg_guest_rating or "N/A")
    col4.metric("Total Reviews", context.total_review_count)

    if context.top_negative_themes:
        st.markdown("**Known recurring concerns:**")
        for theme in context.top_negative_themes:
            label = theme.get(
                    "theme_label", theme.get("label", "")
                    )
            mentions = theme.get("review_mentions", "?")
            st.markdown(
                    f"- {label} _(mentioned in {mentions} reviews)_"
                    )

st.divider()

# --- Review form ---
st.subheader("Your Review")

col_left, col_right = st.columns([1, 2])
with col_left:
    overall_rating = st.slider("Overall rating", 1, 5, 4)
with col_right:
    room_rating = st.select_slider(
            "Room quality",
            options=[None, 1, 2, 3, 4, 5],
            value=None,
            format_func=lambda v: "Skip" if v is None else str(v),
            )
    vibe_rating = st.select_slider(
            "Atmosphere / vibe",
            options=[None, 1, 2, 3, 4, 5],
            value=None,
            format_func=lambda v: "Skip" if v is None else str(v),
            )

review_text = st.text_area(
        "Tell us about your stay",
        placeholder="What stood out — good or bad?",
        height=120,
        )

# --- Generate candidates ---
if (
        st.button("Continue →", type="primary")
        and not st.session_state.submitted
        ):
    with st.spinner(
            "Finding the most useful follow-up questions..."
            ):
        try:
            # Stage 5: Generate raw LLM candidates
                raw = generate_question_candidates(
                        reviewer_text=review_text,
                        context=context,
                        beliefs=st.session_state.beliefs,
                        )
                st.session_state.raw_candidates = raw

                # Enrich with belief, context, cluster, composite
                enriched = enrich_candidates_for_cpsat(
                        raw_candidates=raw,
                        reviewer_text=review_text,
                        context=context,
                        beliefs=st.session_state.beliefs,
                        )
                st.session_state.enriched_candidates = enriched

                # LLM deduplication + selection
                optimized, llm_stats = llm_select_candidates(
                        review_text=review_text,
                        candidates=enriched,
                        )
                st.session_state.optimized_candidates = optimized

                print(f"DEBUG llm_selector stats: {llm_stats}")

                # Build ReviewSignals
                review_signals = ReviewSignals(
                        star_ratings={
                            "overall": overall_rating,
                            "room": room_rating,
                            "vibe": vibe_rating,
                            },
                        review_text=review_text,
                        existing_answer_count=0,
                        )

                # Convert selected candidates to FinalQuestion via followup selector
                candidates = [raw_to_candidate(q) for q in optimized]
                result = select_followup_questions(review_signals, candidates)

                st.session_state.selection_result = result
                st.session_state.answers = {}

        except Exception as exc:
            st.error(
                    f"Could not generate questions: {exc}"
                    )

# --- Follow-up questions ---
result: QuestionSelectionResult | None = (
        st.session_state.selection_result
        )

if result and not st.session_state.submitted:
    st.divider()

    # Willingness badge
    icon, label = _WILLINGNESS_LABELS[result.willingness]
    st.caption(
            f"{icon} {label}  ·  backend: `{result.backend}`"
            )
    st.subheader("📋 A couple of quick follow-ups")

    for i, final_q in enumerate(result.selected_questions):
        aspect = (
                final_q.facet_ids[0]
                if final_q.facet_ids
                else f"q{i}"
                )
        badge = (
                "🤔"
                if "benefit_of_doubt"
                in final_q.why_selected.lower()
                else "🔍"
                )
        st.markdown(f"{badge} _{final_q.why_selected}_")

        answer = render_question_widget(
                final_q,
                widget_key=f"answer_{final_q.candidate_id}_{i}",
                )
        st.session_state.answers[aspect] = answer or ""

    st.divider()

    if st.button("Submit Review", type="primary"):
        for aspect, answer in st.session_state.answers.items():
            if answer.strip():
                st.session_state.beliefs = (
                        update_belief_from_answer(
                            st.session_state.beliefs,
                            aspect=aspect,
                            answer_text=answer,
                            days_ago=0,
                            )
                        )
        st.session_state.submitted = True

# --- Post-submit ---
if st.session_state.submitted:
    st.success("Thank you — your review has been submitted! 🎉")

    filled = {
            k: v
            for k, v in st.session_state.answers.items()
            if v.strip()
            }

    if filled:
        st.subheader("✅ How your answers help future guests")
        for aspect, answer in filled.items():
            updated_belief = st.session_state.beliefs.get(aspect)
            score = (
                    updated_belief.score if updated_belief else 0.5
                    )
            quality = (
                    "✅ Confirmed positive"
                    if score >= 0.65
                    else "⚠️ Flagged for attention"
                    if score < 0.4
                    else "📝 Noted"
                    )
            st.markdown(
                    f"- **{aspect.replace('_', ' ').title()}** — "
                    f"{quality}  \n"
                    f"  _{answer}_"
                    )
