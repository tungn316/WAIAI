# stage3_context_weight.py
"""
Stage 3 - Context Weight Scoring
Scores each candidate gap for relevance against property type × topic matrix.
"""

from config import PROPERTY_TYPE_MATRIX, CONTEXT_WEIGHT_MIN


def get_context_weight(property_type: str, gap_topic: str) -> float:
    type_weights = PROPERTY_TYPE_MATRIX.get(property_type, {})
    return type_weights.get(gap_topic, 0.3)


def expand_review_to_candidates(review: dict) -> list[dict]:
    candidates = []
    gaps = review.get("gaps_mentioned", [])

    if not gaps:
        return candidates

    for gap in gaps:
        candidate = {
            "review_id": review["review_id"],
            "reviewer_id": review["reviewer_id"],
            "reviewer_name": review.get("reviewer_name", ""),
            "property_id": review["property_id"],
            "property_name": review.get("property_name", ""),
            "property_type": review["property_type"],
            "review_date": review["review_date"],
            "review_text": review["review_text"],
            "gap": gap,
            "belief_score": review.get("belief_score", 0.0),
            "cluster_pool": review.get("cluster_pool", "Pool A"),
            "cluster_label": review.get("cluster_label", ""),
            "cluster_confidence": review.get("cluster_confidence", 0.0),
            "estimated_token_cost": review.get(
                "estimated_token_cost", 150
            ),
            "cluster_method": review.get("cluster_method", "keyword"),
        }
        candidates.append(candidate)

    return candidates


def score_and_filter_candidates(
    reviews: list[dict],
) -> tuple[list[dict], dict]:
    all_candidates = []
    for review in reviews:
        all_candidates.extend(expand_review_to_candidates(review))

    filtered = []
    dropped = 0

    for candidate in all_candidates:
        weight = get_context_weight(
            candidate["property_type"], candidate["gap"]
        )
        candidate["context_weight"] = round(weight, 4)

        if weight >= CONTEXT_WEIGHT_MIN:
            filtered.append(candidate)
        else:
            dropped += 1

    stats = {
        "total_candidates": len(all_candidates),
        "passed": len(filtered),
        "dropped": dropped,
        "threshold": CONTEXT_WEIGHT_MIN,
    }
    return filtered, stats
