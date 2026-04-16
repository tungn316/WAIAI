# composite_score.py
"""
Composite Score — CP-SAT Utility Input
CompositeScore = (BeliefScore * 0.35) + (ContextWeight * 0.35)
                 + (ReviewContentScore * 0.30)
"""

from config import (
    BELIEF_WEIGHT,
    CONTEXT_WEIGHT_WEIGHT,
    CONTENT_SCORE_WEIGHT,
    COMPOSITE_MIN_THRESHOLD,
)

EXPLICIT_KEYWORDS = [
    "broken", "not working", "didn't work", "no ", "missing",
    "terrible", "awful", "worst", "unacceptable", "non-functional",
    "out of service", "unavailable", "never", "nothing", "failed",
]
IMPLICIT_KEYWORDS = [
    "struggled", "difficult", "confusing", "limited", "small", "slow",
    "spotty", "noisy", "unclear", "expensive", "worn", "outdated",
    "insufficient", "lacking", "poor", "low", "saggy",
]


def compute_review_content_score(review_text: str, gap: str) -> float:
    """1.0 = explicit mention, 0.5 = implicit, 0.2 = inferred."""
    text_lower = review_text.lower()
    gap_lower = gap.lower()

    if gap_lower in text_lower:
        for kw in EXPLICIT_KEYWORDS:
            if kw in text_lower:
                return 1.0
        for kw in IMPLICIT_KEYWORDS:
            if kw in text_lower:
                return 0.5
        return 0.5

    return 0.2


def compute_composite_score(candidate: dict) -> float:
    belief = candidate.get("belief_score", 0.0)
    context = candidate.get("context_weight", 0.0)
    content = compute_review_content_score(
        candidate.get("review_text", ""),
        candidate.get("gap", ""),
    )
    candidate["review_content_score"] = round(content, 4)

    composite = (
        belief * BELIEF_WEIGHT
        + context * CONTEXT_WEIGHT_WEIGHT
        + content * CONTENT_SCORE_WEIGHT
    )
    return round(composite, 4)


def score_and_filter(
    candidates: list[dict],
) -> tuple[list[dict], dict]:
    filtered = []
    dropped = 0

    for candidate in candidates:
        score = compute_composite_score(candidate)
        candidate["composite_score"] = score

        if score >= COMPOSITE_MIN_THRESHOLD:
            filtered.append(candidate)
        else:
            dropped += 1

    filtered.sort(key=lambda c: c["composite_score"], reverse=True)

    stats = {
        "total_scored": len(candidates),
        "passed": len(filtered),
        "dropped": dropped,
        "threshold": COMPOSITE_MIN_THRESHOLD,
        "avg_score": round(
            sum(c["composite_score"] for c in filtered)
            / max(len(filtered), 1),
            4,
        ),
    }
    return filtered, stats
