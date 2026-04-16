"""
LLM-based candidate deduplication and final question selection.

Flow:
1. Send full composite-scored waitlist + review text to LLM
2. LLM deduplicates topics already covered in the review
3. LLM selects top 2 remaining candidates
4. Falls back to general/delight questions if waitlist is exhausted
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Any

import streamlit as st

from hotel_theme_tool.embeddings import build_ssl_context

FALLBACK_QUESTIONS = {
    "general_feedback": {
        "candidate_id": "fallback_general",
        "aspect": "general_feedback",
        "question": "Is there anything else about your stay you'd like to share with future guests?",
        "suggested_response_type": "free_text",
        "rationale": "Fallback: review covered all known gaps.",
        "composite_score": 0.0,
        "cluster_pool": "Pool B",
    },
    "surprise_delight": {
        "candidate_id": "fallback_delight",
        "aspect": "surprise_delight",
        "question": "Was there anything about your stay that genuinely surprised or delighted you?",
        "suggested_response_type": "boolean_plus_optional_text",
        "rationale": "Fallback: opportunity to capture positive signal.",
        "composite_score": 0.0,
        "cluster_pool": "Pool B",
    },
}

SYSTEM_PROMPT = """\
You are a final selection layer for a hotel review follow-up system.

You will receive:
- The guest's review text
- A ranked waitlist of candidate follow-up questions, each with an aspect and composite score

Your job:
1. DEDUPLICATE — Remove any candidate whose topic is already clearly addressed \
in the review text, positively or negatively. Be strict: if the reviewer \
mentioned it at all, remove it.
2. SELECT — From the remaining candidates, pick the top 2 by composite score. \
If only 1 remains, pick 1. If 0 remain, return an empty selected list.
3. FALLBACK — If fewer than 2 candidates survive deduplication, indicate \
how many fallback questions are needed (0, 1, or 2) to reach the 2-question guarantee.

Output ONLY valid JSON in exactly this shape:
{
  "selected_candidate_ids": ["id1", "id2"],
  "deduplicated_ids": ["id_removed_1"],
  "fallbacks_needed": 0,
  "rationale": "one sentence explanation"
}
"""


def _build_payload(
    review_text: str,
    candidates: list[dict[str, Any]],
) -> str:
    waitlist = [
        {
            "candidate_id": c["candidate_id"],
            "aspect": c.get("aspect", ""),
            "question": c.get("question", ""),
            "composite_score": c.get("composite_score", 0.0),
            "rationale": c.get("rationale", ""),
        }
        for c in sorted(
            candidates,
            key=lambda x: x.get("composite_score", 0.0),
            reverse=True,
        )
    ]

    return json.dumps(
        {
            "review_text": review_text,
            "candidate_waitlist": waitlist,
        },
        ensure_ascii=False,
        indent=2,
    )


def _call_llm(
    payload: str,
    model: str,
    api_key: str,
    base_url: str,
    timeout: int,
) -> dict[str, Any]:
    body = json.dumps(
        {
            "model": model,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": payload},
            ],
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        url=f"{base_url}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    ssl_context = build_ssl_context()
    try:
        with urllib.request.urlopen(
            request, timeout=timeout, context=ssl_context
        ) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
            return json.loads(raw["choices"][0]["message"]["content"])
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"LLM selector call failed (HTTP {exc.code}): {detail}"
        ) from exc


def llm_select_candidates(
    review_text: str,
    candidates: list[dict[str, Any]],
    *,
    model: str = "gpt-4.1-nano",
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 30,
    question_guarantee: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Use an LLM to deduplicate and select final candidates.

    Returns (selected_candidates, stats).
    Selected candidates are ready to be passed into raw_to_candidate().
    Fallback slots are filled with FALLBACK_QUESTIONS entries.
    """
    key = api_key or st.secrets.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required.")

    url = (
        base_url
        or st.secrets.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    ).rstrip("/")

    if not candidates:
        print("DEBUG llm_selector: no candidates, returning fallbacks")
        fallbacks = list(FALLBACK_QUESTIONS.values())[:question_guarantee]
        return fallbacks, {
            "status": "NO_CANDIDATES",
            "selected": [],
            "deduplicated": [],
            "fallbacks_used": len(fallbacks),
        }

    payload = _build_payload(review_text, candidates)
    decision = _call_llm(payload, model, key, url, timeout)

    print(f"DEBUG llm_selector: decision -> {json.dumps(decision, indent=2)}")

    selected_ids: list[str] = decision.get("selected_candidate_ids", [])
    deduplicated_ids: list[str] = decision.get("deduplicated_ids", [])
    fallbacks_needed: int = max(0, decision.get("fallbacks_needed", 0))

    candidate_by_id = {c["candidate_id"]: c for c in candidates}

    selected: list[dict[str, Any]] = [
        candidate_by_id[cid]
        for cid in selected_ids
        if cid in candidate_by_id
    ]

    # Fill fallback slots up to question_guarantee
    fallback_pool = list(FALLBACK_QUESTIONS.values())
    fallback_index = 0
    while len(selected) < question_guarantee and fallback_index < len(fallback_pool):
        selected.append(fallback_pool[fallback_index])
        fallback_index += 1

    fallbacks_used = fallback_index

    stats = {
        "status": "LLM_SELECTED",
        "selected": selected_ids,
        "deduplicated": deduplicated_ids,
        "fallbacks_needed": fallbacks_needed,
        "fallbacks_used": fallbacks_used,
        "rationale": decision.get("rationale", ""),
        "total_candidates_in": len(candidates),
        "total_selected_out": len(selected),
    }

    return selected, stats
