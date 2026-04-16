# question_generator.py
from __future__ import annotations

import json
import os
import streamlit as st
import re
import urllib.error
import urllib.request
from typing import Any

from belief_system import summarize_belief_gaps, AspectBelief
from context_profile import ContextProfile
from hotel_theme_tool.embeddings import build_ssl_context
from composite_score import compute_composite_score
from stage2_embedding_clustering import classify_by_keywords

SYSTEM_PROMPT = """\
You are an expert travel review assistant helping to collect \
missing or outdated information about hotel properties. Your job is to generate \
up to 4 natural, conversational candidate follow-up questions for a traveller \
who just left a review. A downstream optimizer will pick the final 1–2 to show.

Rules:
- Walk through the known_issues list IN ORDER (rank 1 first).
- SKIP any issue the reviewer has already addressed — positively or negatively.
- Generate questions for the first 3–4 unaddressed issues.
- If all known issues are covered, fall back to the belief_gaps list.
- Be conversational and warm — like a friendly follow-up, not a survey.
- If a theme has very few mentions (low_confidence: true), give the property \
benefit of the doubt.
- Output ONLY valid JSON in the exact shape specified."""

QUESTION_SHAPE = """
{
  "questions": [
    {
      "aspect": "the aspect key (use theme slug if no aspect key matches)",
      "question": "the question to ask the reviewer",
      "rationale": "one sentence explaining why this gap matters",
      "benefit_of_doubt": true or false,
      "suggested_response_type": "one of: free_text | boolean | boolean_plus_optional_text | multiple_choice | scale_plus_optional_text"
    }
  ]
}
"""


def _slugify(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _annotate_themes_addressed(
    reviewer_text: str,
    themes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lower = reviewer_text.lower()
    annotated = []
    for theme in themes:
        label = (
            theme.get("theme_label") or theme.get("label") or ""
        ).lower()
        label_words = [w for w in re.split(r"\W+", label) if len(w) > 3]
        possibly_addressed = any(word in lower for word in label_words)
        annotated.append({**theme, "possibly_addressed": possibly_addressed})
    return annotated


def _debug_print_ranked_issues(
    ranked_issues: list[dict[str, Any]],
    fallback_gaps: list[dict[str, Any]],
    reviewer_text: str,
) -> None:
    sep = "-" * 60
    print(f"\n{sep}")
    print("DEBUG: question_generator — theme ranking")
    print(sep)
    print(
        f"Reviewer text: "
        f"{reviewer_text[:120]!r}"
        f"{'...' if len(reviewer_text) > 120 else ''}"
    )
    print()
    print("Known issues (primary, ordered by volume):")
    if not ranked_issues:
        print("  (none)")
    for issue in ranked_issues:
        addressed = (
            "✓ addressed"
            if issue["possibly_addressed_in_review"]
            else "✗ not addressed"
        )
        confidence = (
            " [low confidence]" if issue["low_confidence"] else ""
        )
        print(
            f"  #{issue['rank']:>2}  {issue['theme']:<35}"
            f"  mentions={issue['review_mentions']:<4}"
            f"  {addressed}{confidence}"
        )
    print()
    print("Fallback belief gaps:")
    if not fallback_gaps:
        print("  (none)")
    for gap in fallback_gaps:
        flags = []
        if gap.get("is_missing"):
            flags.append("missing")
        if gap.get("is_stale"):
            flags.append("stale")
        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        print(
            f"  aspect={gap['aspect']:<25}"
            f"  urgency={gap['urgency']:<6}"
            f"  score={gap['score']:<6}{flag_str}"
        )
    print(sep)
    print()


def build_question_prompt(
    reviewer_text: str,
    context: ContextProfile,
    belief_gaps: list[dict[str, Any]],
    beliefs: dict[str, AspectBelief],
) -> str:
    themes = _annotate_themes_addressed(
        reviewer_text, context.top_negative_themes
    )

    ranked_issues = []
    for rank, theme in enumerate(themes, start=1):
        label = theme.get("theme_label") or theme.get("label") or ""
        mentions = theme.get("review_mentions", 0)
        share = theme.get("share_of_negative_reviews", 1.0)
        low_confidence = mentions <= 2 and share < 0.15
        ranked_issues.append(
            {
                "rank": rank,
                "theme": label,
                "slug": _slugify(label),
                "review_mentions": mentions,
                "low_confidence": low_confidence,
                "possibly_addressed_in_review": theme[
                    "possibly_addressed"
                ],
            }
        )

    covered_slugs = {r["slug"] for r in ranked_issues}
    fallback_gaps = [
        {
            "aspect": g["aspect"],
            "urgency": g["urgency"],
            "score": g["score"],
            "is_missing": g["is_missing"],
            "is_stale": g["is_stale"],
        }
        for g in belief_gaps[:6]
        if g["aspect"] not in covered_slugs
    ]

    _debug_print_ranked_issues(ranked_issues, fallback_gaps, reviewer_text)

    prompt_payload = {
        "property_context": {
            "tier": context.tier,
            "star_rating": context.star_rating,
            "city": context.city,
            "country": context.country,
            "avg_guest_rating": context.avg_guest_rating,
            "total_reviews": context.total_review_count,
            "tier_implicit_expectations": context.implicit_expectations,
        },
        "reviewer_text": reviewer_text,
        "known_issues": ranked_issues,
        "fallback_belief_gaps": fallback_gaps,
        "instructions": (
            f"This is a {context.tier}-tier property. "
            "Walk through known_issues in rank order. "
            "Skip addressed issues. "
            "Generate questions for the first 3-4 unaddressed issues. "
            "If all known issues are covered, use fallback_belief_gaps. "
            "For low_confidence issues, frame questions to allow "
            "confirmation or dismissal. "
            "Include a suggested_response_type for each question."
        ),
        "output_shape": QUESTION_SHAPE,
    }

    return json.dumps(prompt_payload, ensure_ascii=False, indent=2)


def generate_question_candidates(
    reviewer_text: str,
    context: ContextProfile,
    beliefs: dict[str, AspectBelief],
    model: str = "gpt-4.1-nano",
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 30,
) -> list[dict[str, Any]]:
    """
    Generate 3–4 raw question candidates from the LLM.
    Enriches each with candidate_id, property_priority, offline_rank.
    """
    key = api_key or st.secrets.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required.")

    url = (
        base_url
        or st.secrets.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    ).rstrip("/")

    belief_gaps = summarize_belief_gaps(beliefs)
    urgency_by_aspect = {g["aspect"]: g["urgency"] for g in belief_gaps}

    prompt = build_question_prompt(
        reviewer_text, context, belief_gaps, beliefs
    )

    payload = json.dumps(
        {
            "model": model,
            "temperature": 0.3,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        url=f"{url}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    ssl_context = build_ssl_context()
    try:
        with urllib.request.urlopen(
            request, timeout=timeout, context=ssl_context
        ) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenAI call failed (HTTP {exc.code}): {detail}"
        ) from exc

    content = body["choices"][0]["message"]["content"]
    data = json.loads(content)
    raw_questions = data.get("questions", [])

    enriched = []
    for rank, q in enumerate(raw_questions):
        aspect = q.get("aspect", f"q{rank}")
        enriched.append(
            {
                **q,
                "candidate_id": f"{aspect}_{rank}",
                "property_priority": urgency_by_aspect.get(aspect, 0.0),
                "offline_rank": rank,
            }
        )

    print("DEBUG: LLM returned candidates:")
    for q in enriched:
        print(
            f"  [{q['offline_rank']}] aspect={q.get('aspect'):<25}"
            f"  priority={q['property_priority']:<6}"
            f"  type={q.get('suggested_response_type', 'free_text')}"
        )
        print(f"    Q: {q.get('question')}")
    print()

    return enriched


def enrich_candidates_for_cpsat(
    raw_candidates: list[dict[str, Any]],
    reviewer_text: str,
    context: ContextProfile,
    beliefs: dict[str, AspectBelief],
) -> list[dict[str, Any]]:
    """
    Take raw LLM candidates and produce dicts ready for
    cpsat_optimizer.optimize_single_property_candidates().

    Adds: belief_score, context_weight, cluster_pool, cluster_label,
          cluster_confidence, estimated_token_cost, composite_score.
    """
    from belief_system import recency_weight
    from stage3_context_weight import get_context_weight

    # Determine property type from tier
    tier_to_type = {
        "budget": "Hotel",
        "midscale": "Hotel",
        "upscale": "Hotel",
        "luxury": "Hotel",
    }
    property_type = tier_to_type.get(context.tier, "Hotel")

    enriched = []
    for cand in raw_candidates:
        aspect = cand.get("aspect", "")
        belief = beliefs.get(aspect)

        # Belief score: use existing belief's last_confirmed_days_ago
        if belief and belief.last_confirmed_days_ago is not None:
            b_score = recency_weight(belief.last_confirmed_days_ago)
        else:
            b_score = 0.5  # unknown = moderate

        # Context weight from property type matrix
        ctx_weight = get_context_weight(property_type, aspect)

        # Cluster assignment (keyword-based, fast)
        pool, label, confidence = classify_by_keywords(
            cand.get("question", "") + " " + aspect
        )

        # Estimated token cost per cluster pool
        from config import TOPIC_CLUSTERS

        est_tokens = TOPIC_CLUSTERS.get(pool, {}).get(
            "estimated_token_cost", 150
        )

        enriched_cand = {
            **cand,
            "review_text": reviewer_text,
            "gap": aspect,
            "property_type": property_type,
            "belief_score": round(b_score, 4),
            "context_weight": round(ctx_weight, 4),
            "cluster_pool": pool,
            "cluster_label": label,
            "cluster_confidence": confidence,
            "estimated_token_cost": est_tokens,
        }

        # Composite score
        score = compute_composite_score(enriched_cand)
        enriched_cand["composite_score"] = score

        enriched.append(enriched_cand)

    return enriched
