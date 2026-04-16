# config.py
"""Central configuration for all pipeline stages."""

import os

# --- Stage 1: Belief Score (Temporal Decay) ---
BELIEF_DECAY_LAMBDA = 0.002  # exp(-lambda * days), ~50% at 1 year
BELIEF_MIN_THRESHOLD = 0.05  # drop reviews below this freshness

# --- Stage 2: Embedding + Clustering ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
EMBEDDING_MODEL = "text-embedding-3-small"

TOPIC_CLUSTERS = {
    "Pool A": {
        "label": "Facilities & Amenities",
        "keywords": [
            "pool", "gym", "parking", "wifi", "elevator", "ac",
            "air conditioning", "hot water", "breakfast", "room service",
            "laundry", "spa", "beach",
        ],
        "estimated_token_cost": 120,
    },
    "Pool B": {
        "label": "Service & Experience",
        "keywords": [
            "staff", "check-in", "checkout", "front desk", "concierge",
            "housekeeping", "noise", "clean", "dirty", "smell", "rude",
            "helpful", "friendly", "slow", "wait",
        ],
        "estimated_token_cost": 150,
    },
    "Pool C": {
        "label": "Accuracy & Expectations",
        "keywords": [
            "photos", "description", "misleading", "expected", "different",
            "mismatch", "listing", "advertised", "review", "star",
            "overpriced", "value", "worth", "location", "area",
        ],
        "estimated_token_cost": 180,
    },
}

# --- Stage 3: Context Weight ---
CONTEXT_WEIGHT_MIN = 0.2

PROPERTY_TYPE_MATRIX = {
    "Hotel": {
        "parking": 0.6, "noise": 0.8, "AC": 0.9, "wifi": 0.9,
        "cleanliness": 1.0, "elevators": 0.7, "gym": 0.5,
        "photos": 0.6, "mismatch": 0.7, "check-in": 0.8,
        "staff": 0.9, "breakfast": 0.7, "bathroom": 0.8,
        "pool": 0.6, "room service": 0.5, "hot water": 0.8,
        "bed": 0.8, "housekeeping": 0.7, "pet policy": 0.3,
        "cancellation": 0.5, "safety": 0.7, "accessibility": 0.5,
        "beach access": 0.3, "listing accuracy": 0.6,
    },
    "Vacation Rental": {
        "parking": 0.8, "noise": 0.6, "AC": 0.9, "wifi": 1.0,
        "cleanliness": 1.0, "elevators": 0.2, "gym": 0.1,
        "photos": 0.9, "mismatch": 0.9, "check-in": 0.9,
        "staff": 0.4, "breakfast": 0.2, "bathroom": 0.8,
        "pool": 0.7, "room service": 0.1, "hot water": 0.9,
        "bed": 0.8, "housekeeping": 0.4, "pet policy": 0.7,
        "cancellation": 0.7, "safety": 0.8, "accessibility": 0.4,
        "beach access": 0.6, "listing accuracy": 0.9,
    },
    "Resort": {
        "parking": 0.5, "noise": 0.7, "AC": 0.9, "wifi": 0.7,
        "cleanliness": 1.0, "elevators": 0.6, "gym": 0.7,
        "photos": 0.7, "mismatch": 0.7, "check-in": 0.7,
        "staff": 0.9, "breakfast": 0.8, "bathroom": 0.8,
        "pool": 0.9, "room service": 0.7, "hot water": 0.7,
        "bed": 0.7, "housekeeping": 0.8, "pet policy": 0.4,
        "cancellation": 0.5, "safety": 0.6, "accessibility": 0.5,
        "beach access": 0.8, "listing accuracy": 0.6,
    },
}

# --- Composite Score Weights ---
BELIEF_WEIGHT = 0.35
CONTEXT_WEIGHT_WEIGHT = 0.35
CONTENT_SCORE_WEIGHT = 0.30
COMPOSITE_MIN_THRESHOLD = 0.15

# --- Stage 4: CP-SAT Optimizer ---
TOKEN_BUDGET = 5000
MAX_QUESTIONS_PER_REVIEWER = 2
GAP_SATURATION_K = 3  # max times same gap×property can be asked
MIN_POOL_A = 1
MIN_POOL_B = 1
MIN_POOL_C = 0
SOLVER_MAX_TIME_SECONDS = 10.0

# --- Stage 5: Question Generation ---
GENERATION_MODEL = "gpt-4.1-nano"
QUESTION_PROMPT_TEMPLATE = (
    "Generate a brief, specific follow-up question for a guest who stayed "
    "at a {property_type}. Their review touched on {cluster} topics. "
    "Ask about this specific gap: {gap_description}. "
    "Keep it under 25 words, conversational, and answerable from memory."
)
