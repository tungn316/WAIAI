# context_profile.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Aspects that matter most per property tier, ordered by importance
TIER_PRIORITIES: dict[str, list[str]] = {
    "budget": [
        "cleanliness",
        "value",
        "noise",
        "wifi",
        "check_in",
        "bathroom",
        "room_condition",
        "safety",
        "parking",
        "location",
    ],
    "midscale": [
        "cleanliness",
        "staff_quality",
        "wifi",
        "breakfast",
        "noise",
        "room_condition",
        "value",
        "check_in",
        "parking",
        "pool",
    ],
    "upscale": [
        "staff_quality",
        "cleanliness",
        "room_condition",
        "breakfast",
        "pool",
        "noise",
        "wifi",
        "temperature",
        "check_in",
        "value",
    ],
    "luxury": [
        "staff_quality",
        "room_condition",
        "cleanliness",
        "breakfast",
        "pool",
        "temperature",
        "noise",
        "check_in",
        "wifi",
        "safety",
    ],
}

# Aspects that are implicit / expected at each tier
# (i.e., we should ask about them if belief is low because it really matters here)
TIER_IMPLICIT_EXPECTATIONS: dict[str, dict[str, str]] = {
    "budget": {
        "cleanliness": "Even budget travellers expect basic cleanliness.",
        "safety": "Safety is a baseline expectation at any price point.",
    },
    "midscale": {
        "wifi": "Reliable Wi-Fi is expected at midscale properties.",
        "breakfast": "Breakfast is a common midscale differentiator.",
    },
    "upscale": {
        "staff_quality": "Attentive, proactive service defines the upscale experience.",
        "pool": "Guests expect a pool at upscale properties.",
    },
    "luxury": {
        "staff_quality": "Concierge-level service is a core luxury promise.",
        "room_condition": "Luxury guests expect immaculate, modern rooms.",
        "breakfast": "A high-quality breakfast is expected at luxury price points.",
    },
}


def star_to_tier(star_rating: Any) -> str:
    try:
        stars = float(star_rating)
    except (TypeError, ValueError):
        return "midscale"

    if stars <= 2:
        return "budget"
    elif stars <= 3:
        return "midscale"
    elif stars <= 4:
        return "upscale"
    else:
        return "luxury"


@dataclass
class ContextProfile:
    property_id: str
    tier: str
    star_rating: float | None
    city: str
    country: str
    avg_guest_rating: float | None
    negative_review_count: int
    total_review_count: int
    top_negative_themes: list[dict[str, Any]]
    priority_aspects: list[str]
    implicit_expectations: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "property_id": self.property_id,
            "tier": self.tier,
            "star_rating": self.star_rating,
            "city": self.city,
            "country": self.country,
            "avg_guest_rating": self.avg_guest_rating,
            "negative_review_count": self.negative_review_count,
            "total_review_count": self.total_review_count,
            "top_negative_themes": self.top_negative_themes,
            "priority_aspects": self.priority_aspects,
            "implicit_expectations": self.implicit_expectations,
        }


def build_context_profile(
    description_row: dict[str, Any],
    negative_themes: list[dict[str, Any]],
    total_review_count: int,
    negative_review_count: int,
) -> ContextProfile:
    star_rating = description_row.get("star_rating")
    tier = star_to_tier(star_rating)

    try:
        avg_rating = float(description_row.get("guestrating_avg_expedia") or 0) or None
    except (TypeError, ValueError):
        avg_rating = None

    try:
        stars = float(star_rating) if star_rating else None
    except (TypeError, ValueError):
        stars = None

    return ContextProfile(
        property_id=str(description_row.get("eg_property_id", "")),
        tier=tier,
        star_rating=stars,
        city=str(description_row.get("city") or ""),
        country=str(description_row.get("country") or ""),
        avg_guest_rating=avg_rating,
        negative_review_count=negative_review_count,
        total_review_count=total_review_count,
        top_negative_themes=negative_themes[:10],
        priority_aspects=TIER_PRIORITIES.get(tier, TIER_PRIORITIES["midscale"]),
        implicit_expectations=TIER_IMPLICIT_EXPECTATIONS.get(tier, {}),
    )
