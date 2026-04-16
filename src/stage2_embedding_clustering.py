# stage2_embedding_clustering.py
"""
Stage 2 - Semantic Embedding + Topic Clustering
Embeds review text and assigns to Pool A/B/C via cosine similarity.
Uses keyword-based fallback when OpenAI API is unavailable.
"""

import numpy as np
from config import TOPIC_CLUSTERS, OPENAI_API_KEY, EMBEDDING_MODEL

_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None and OPENAI_API_KEY:
        from openai import OpenAI

        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def get_embedding(text: str) -> list[float] | None:
    client = _get_openai_client()
    if client is None:
        return None
    try:
        response = client.embeddings.create(
            input=text, model=EMBEDDING_MODEL
        )
        return response.data[0].embedding
    except Exception:
        return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    dot = np.dot(a_arr, b_arr)
    norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def classify_by_keywords(
    review_text: str,
) -> tuple[str, str, float]:
    text_lower = review_text.lower()
    best_pool = "Pool A"
    best_label = TOPIC_CLUSTERS["Pool A"]["label"]
    best_score = 0.0

    for pool_name, pool_config in TOPIC_CLUSTERS.items():
        matches = sum(
            1 for kw in pool_config["keywords"] if kw in text_lower
        )
        score = matches / max(len(pool_config["keywords"]), 1)
        if score > best_score:
            best_score = score
            best_pool = pool_name
            best_label = pool_config["label"]

    return best_pool, best_label, round(best_score, 4)


def classify_by_embedding(
    review_text: str, cluster_embeddings: dict
) -> tuple[str, str, float]:
    review_emb = get_embedding(review_text)
    if review_emb is None:
        return classify_by_keywords(review_text)

    best_pool = "Pool A"
    best_label = TOPIC_CLUSTERS["Pool A"]["label"]
    best_sim = -1.0

    for pool_name, centroid in cluster_embeddings.items():
        sim = cosine_similarity(review_emb, centroid)
        if sim > best_sim:
            best_sim = sim
            best_pool = pool_name
            best_label = TOPIC_CLUSTERS[pool_name]["label"]

    return best_pool, best_label, round(best_sim, 4)


def build_cluster_centroids() -> dict | None:
    client = _get_openai_client()
    if client is None:
        return None

    centroids = {}
    for pool_name, pool_config in TOPIC_CLUSTERS.items():
        desc = (
            f"{pool_config['label']}: "
            f"{', '.join(pool_config['keywords'])}"
        )
        emb = get_embedding(desc)
        if emb:
            centroids[pool_name] = emb

    return centroids if len(centroids) == len(TOPIC_CLUSTERS) else None


def assign_clusters(reviews: list[dict]) -> tuple[list[dict], dict]:
    cluster_centroids = build_cluster_centroids()

    for review in reviews:
        text = review["review_text"]

        if cluster_centroids:
            pool, label, confidence = classify_by_embedding(
                text, cluster_centroids
            )
            review["cluster_method"] = "embedding"
        else:
            pool, label, confidence = classify_by_keywords(text)
            review["cluster_method"] = "keyword"

        review["cluster_pool"] = pool
        review["cluster_label"] = label
        review["cluster_confidence"] = confidence
        review["estimated_token_cost"] = TOPIC_CLUSTERS[pool][
            "estimated_token_cost"
        ]

    stats = {
        "total_classified": len(reviews),
        "method": (
            reviews[0]["cluster_method"] if reviews else "none"
        ),
        "pool_distribution": {},
    }
    for pool_name in TOPIC_CLUSTERS:
        count = sum(
            1
            for r in reviews
            if r.get("cluster_pool") == pool_name
        )
        stats["pool_distribution"][pool_name] = count

    return reviews, stats
