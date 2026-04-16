# belief_system.py
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

# Aspects we track beliefs for, mapped to description fields and review keywords
ASPECTS: dict[str, dict[str, Any]] = {
    "cleanliness": {
        "description_fields": [],
        "keywords": ["clean", "dirty", "filthy", "dust", "hygiene", "spotless", "grime", "stain"],
    },
    "staff_quality": {
        "description_fields": [],
        "keywords": ["staff", "rude", "helpful", "friendly", "concierge", "front desk", "service", "attentive"],
    },
    "wifi": {
        "description_fields": ["popular_amenities_list"],
        "keywords": ["wifi", "wi-fi", "internet", "connection", "connectivity"],
    },
    "pool": {
        "description_fields": ["popular_amenities_list", "property_amenity_pool"],
        "keywords": ["pool", "swimming", "heated pool", "outdoor pool"],
    },
    "parking": {
        "description_fields": ["popular_amenities_list"],
        "keywords": ["parking", "valet", "garage", "lot", "car park"],
    },
    "breakfast": {
        "description_fields": ["popular_amenities_list"],
        "keywords": ["breakfast", "buffet", "continental", "morning meal", "brunch"],
    },
    "noise": {
        "description_fields": [],
        "keywords": ["noisy", "noise", "loud", "quiet", "thin walls", "soundproof", "disruptive"],
    },
    "room_condition": {
        "description_fields": ["property_description"],
        "keywords": ["room", "outdated", "renovated", "broken", "maintenance", "worn", "modern"],
    },
    "check_in": {
        "description_fields": ["check_in_instructions", "check_in_start_time"],
        "keywords": ["check-in", "checkin", "arrival", "key", "front desk", "waited", "smooth"],
    },
    "pet_policy": {
        "description_fields": ["pet_policy"],
        "keywords": ["pet", "dog", "cat", "animal", "fur"],
    },
    "value": {
        "description_fields": ["star_rating"],
        "keywords": ["price", "value", "worth", "overpriced", "expensive", "cheap", "fair"],
    },
    "location": {
        "description_fields": ["area_description"],
        "keywords": ["location", "area", "neighborhood", "nearby", "access", "transport", "walk"],
    },
    "temperature": {
        "description_fields": [],
        "keywords": ["hot", "cold", "ac", "air conditioning", "hvac", "stuffy", "ventilation", "temperature"],
    },
    "bathroom": {
        "description_fields": [],
        "keywords": ["bathroom", "shower", "toilet", "plumbing", "water", "leak", "drain"],
    },
    "safety": {
        "description_fields": ["know_before_you_go"],
        "keywords": ["safe", "unsafe", "security", "lock", "hazard", "dangerous"],
    },
}

NEGATIVE_SENTIMENT_WORDS = {
    "dirty", "filthy", "noisy", "loud", "rude", "broken", "slow", "bad", "terrible",
    "awful", "horrible", "disgusting", "disappointing", "worst", "poor", "never", "no",
    "not", "didn't", "couldn't", "wouldn't", "wasn't", "didn't", "leak", "mold", "pest",
    "bug", "unsafe", "outdated", "worn", "overpriced", "expensive", "unavailable",
    "closed", "missing", "wrong", "refused", "ignored", "unhelpful",
}

POSITIVE_SENTIMENT_WORDS = {
    "clean", "spotless", "quiet", "friendly", "helpful", "attentive", "modern", "renovated",
    "excellent", "great", "perfect", "wonderful", "amazing", "comfortable", "smooth",
    "fast", "efficient", "responsive", "good", "nice", "beautiful", "spacious", "worth",
}


def recency_weight(days_ago: int) -> float:
    """Exponential decay: ~100% today, ~37% at 1yr, ~5% at 3yrs."""
    return math.exp(-days_ago / 365.0)


def review_sentiment_for_aspect(text: str, aspect: str) -> tuple[float, bool]:
    """
    Returns (sentiment_score, mentioned) where sentiment_score is in [-1, 1].
    Only considers the review mentioned the aspect at all.
    """
    lower = text.lower()
    keywords = ASPECTS[aspect]["keywords"]

    mentioned = any(kw in lower for kw in keywords)
    if not mentioned:
        return 0.0, False

    words = set(lower.split())
    pos = sum(1 for w in POSITIVE_SENTIMENT_WORDS if w in words)
    neg = sum(1 for w in NEGATIVE_SENTIMENT_WORDS if w in words)
    total = pos + neg
    if total == 0:
        return 0.1, True  # mentioned neutrally, slight positive lean

    score = (pos - neg * 1.2) / total  # negatives weighted slightly more
    return max(-1.0, min(1.0, score)), True


@dataclass
class AspectBelief:
    aspect: str
    # 0.0 = certainly false/bad, 1.0 = certainly true/good, 0.5 = unknown
    score: float = 0.5
    confidence: float = 0.0  # 0-1, grows with evidence
    description_claimed: bool = False
    description_value: str = ""
    review_mention_count: int = 0
    weighted_sentiment_sum: float = 0.0
    last_confirmed_days_ago: int | None = None
    is_stale: bool = False  # True if last mention > 180 days ago
    is_missing: bool = True  # True if never mentioned anywhere

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_belief_system(
    description_row: dict[str, Any],
    reviews: list[dict[str, Any]],  # each: {text, rating, days_ago}
    stale_threshold_days: int = 180,
) -> dict[str, AspectBelief]:
    beliefs: dict[str, AspectBelief] = {}

    for aspect, config in ASPECTS.items():
        belief = AspectBelief(aspect=aspect)

        # Seed from description
        for field_name in config["description_fields"]:
            val = description_row.get(field_name)
            if val and str(val).strip() and str(val).strip().lower() not in ("nan", "none", ""):
                belief.description_claimed = True
                belief.description_value = str(val).strip()[:200]
                belief.score = 0.75  # description is near ground truth
                belief.confidence = 0.4
                belief.is_missing = False
                break

        # Update from reviews
        weighted_sum = 0.0
        weight_total = 0.0
        most_recent_mention = None

        for review in reviews:
            sentiment, mentioned = review_sentiment_for_aspect(review["text"], aspect)
            if not mentioned:
                continue

            days_ago = review.get("days_ago", 365)
            w = recency_weight(days_ago)
            weighted_sum += sentiment * w
            weight_total += w
            belief.review_mention_count += 1
            belief.is_missing = False

            if most_recent_mention is None or days_ago < most_recent_mention:
                most_recent_mention = days_ago

        if weight_total > 0:
            # Blend description prior with review evidence
            review_score = (weighted_sum / weight_total + 1.0) / 2.0  # map [-1,1] → [0,1]
            prior_weight = 0.3 if belief.description_claimed else 0.1
            review_weight = min(1.0, belief.review_mention_count / 10.0)
            blend = prior_weight + review_weight

            belief.score = (
                belief.score * prior_weight + review_score * review_weight
            ) / blend

            # Confidence grows with more weighted evidence, saturates around 20 reviews
            belief.confidence = min(0.95, 0.4 * math.log1p(weight_total) / math.log1p(20))
            belief.weighted_sentiment_sum = round(weighted_sum, 3)
            belief.last_confirmed_days_ago = most_recent_mention

        # Staleness
        if most_recent_mention is not None:
            belief.is_stale = most_recent_mention > stale_threshold_days
        elif belief.description_claimed:
            belief.is_stale = True  # described but never reviewed

        belief.score = round(belief.score, 4)
        belief.confidence = round(belief.confidence, 4)
        beliefs[aspect] = belief

    return beliefs


def update_belief_from_answer(
    beliefs: dict[str, AspectBelief],
    aspect: str,
    answer_text: str,
    days_ago: int = 0,
) -> dict[str, AspectBelief]:
    """
    Called after a reviewer answers a follow-up question.
    Treats the answer as a high-recency review snippet.
    """
    if aspect not in beliefs:
        return beliefs

    sentiment, mentioned = review_sentiment_for_aspect(answer_text, aspect)
    if not mentioned:
        # Still count it as a neutral, very recent touch
        sentiment = 0.05

    belief = beliefs[aspect]
    w = recency_weight(days_ago)  # essentially 1.0 for today
    belief.weighted_sentiment_sum += sentiment * w
    belief.review_mention_count += 1
    belief.last_confirmed_days_ago = days_ago
    belief.is_stale = False
    belief.is_missing = False

    # Nudge score
    direction = (sentiment * w) / (belief.review_mention_count + 1)
    belief.score = max(0.0, min(1.0, belief.score + direction))
    belief.confidence = min(0.95, belief.confidence + 0.05)
    belief.score = round(belief.score, 4)
    belief.confidence = round(belief.confidence, 4)

    beliefs[aspect] = belief
    return beliefs


def summarize_belief_gaps(
    beliefs: dict[str, AspectBelief],
) -> list[dict[str, Any]]:
    """
    Returns aspects ordered by urgency of needing more information.
    Urgency = low score + low confidence + missing/stale.
    """
    gaps = []
    for aspect, belief in beliefs.items():
        urgency = 0.0

        if belief.is_missing:
            urgency += 3.0
        elif belief.is_stale:
            urgency += 1.5

        if belief.score < 0.4:
            urgency += (0.4 - belief.score) * 4.0  # heavily penalise bad scores

        if belief.confidence < 0.3:
            urgency += (0.3 - belief.confidence) * 2.0

        if urgency > 0:
            gaps.append(
                {
                    "aspect": aspect,
                    "urgency": round(urgency, 3),
                    "score": belief.score,
                    "confidence": belief.confidence,
                    "is_missing": belief.is_missing,
                    "is_stale": belief.is_stale,
                    "description_claimed": belief.description_claimed,
                    "review_mentions": belief.review_mention_count,
                }
            )

    gaps.sort(key=lambda x: x["urgency"], reverse=True)
    return gaps
