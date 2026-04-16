from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .embeddings import EmbeddingClient, HashingEmbeddingClient, cosine_similarity, normalize_vector
from .models import ComplaintSnippet, ReviewRecord, ThemeDefinition, ThemeMatch
from .segmentation import split_review_into_snippets
from .theme_catalog import DEFAULT_THEME_CATALOG


DEFAULT_REVIEWS_PATH = Path("data/Reviews_PROC.csv")
DEFAULT_DESCRIPTIONS_PATH = Path("data/Description_PROC.csv")

# Official OpenAI pricing references used for the budget guard on 2026-04-14:
# - text-embedding-3-small: $0.02 / 1M input tokens
# - gpt-5.4-nano: $0.20 / 1M input tokens, $1.25 / 1M output tokens
PRICE_USD_PER_1M_INPUT: dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "gpt-5.4-nano": 0.20,
    "gpt-5.4-mini": 0.60,
}
PRICE_USD_PER_1M_OUTPUT: dict[str, float] = {
    "gpt-5.4-nano": 1.25,
    "gpt-5.4-mini": 2.40,
}


@dataclass
class ThemeBucket:
    theme_key: str
    theme_label: str
    review_ids: set[str] = field(default_factory=set)
    snippet_count: int = 0
    confidences: list[float] = field(default_factory=list)
    examples: list[tuple[float, str]] = field(default_factory=list)

    def add(self, review_id: str, confidence: float, snippet: str) -> None:
        self.review_ids.add(review_id)
        self.snippet_count += 1
        self.confidences.append(confidence)
        self.examples.append((confidence, snippet))

    def to_summary(self, total_reviews: int, max_examples: int = 3) -> dict[str, Any]:
        unique_examples: list[str] = []
        seen: set[str] = set()
        for _, snippet in sorted(self.examples, key=lambda item: item[0], reverse=True):
            fingerprint = snippet.casefold()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            unique_examples.append(snippet)
            if len(unique_examples) == max_examples:
                break

        review_mentions = len(self.review_ids)
        return {
            "theme_key": self.theme_key,
            "theme_label": self.theme_label,
            "review_mentions": review_mentions,
            "share_of_negative_reviews": round(review_mentions / total_reviews, 3) if total_reviews else 0.0,
            "snippet_mentions": self.snippet_count,
            "average_confidence": round(sum(self.confidences) / len(self.confidences), 3),
            "example_snippets": unique_examples,
        }


class ThemeAssigner:
    def __init__(
        self,
        embedder: EmbeddingClient,
        themes: tuple[ThemeDefinition, ...] = DEFAULT_THEME_CATALOG,
        min_similarity: float = 0.18,
        min_margin: float = 0.015,
        high_confidence: float = 0.28,
    ) -> None:
        self.embedder = embedder
        self.themes = themes
        self.min_similarity = min_similarity
        self.min_margin = min_margin
        self.high_confidence = high_confidence
        self._theme_lookup = {theme.key: theme for theme in self.themes}
        self._prototype_vectors = self._build_prototype_vectors()

    def assign(self, snippets: list[ComplaintSnippet]) -> list[tuple[ComplaintSnippet, ThemeMatch]]:
        if not snippets:
            return []

        snippet_vectors = self.embedder.embed_texts([snippet.text for snippet in snippets])
        assignments: list[tuple[ComplaintSnippet, ThemeMatch]] = []

        for snippet, vector in zip(snippets, snippet_vectors):
            ranked = sorted(
                (
                    (
                        theme.key,
                        max(cosine_similarity(vector, prototype) for prototype in self._prototype_vectors[theme.key]),
                    )
                    for theme in self.themes
                ),
                key=lambda item: item[1],
                reverse=True,
            )

            best_key, best_score = ranked[0]
            second_score = ranked[1][1] if len(ranked) > 1 else -1.0

            if best_score < self.min_similarity:
                continue
            if best_score < self.high_confidence and (best_score - second_score) < self.min_margin:
                continue

            theme = self._theme_lookup[best_key]
            assignments.append(
                (
                    snippet,
                    ThemeMatch(
                        theme_key=theme.key,
                        theme_label=theme.label,
                        confidence=best_score,
                    ),
                )
            )
        return assignments

    def _build_prototype_vectors(self) -> dict[str, list[list[float]]]:
        prototype_texts: list[str] = []
        owner_keys: list[str] = []
        for theme in self.themes:
            for prototype in theme.prototypes:
                prototype_texts.append(prototype)
                owner_keys.append(theme.key)

        vectors = self.embedder.embed_texts(prototype_texts)
        grouped: dict[str, list[list[float]]] = defaultdict(list)
        for owner_key, vector in zip(owner_keys, vectors):
            grouped[owner_key].append(vector)
        return dict(grouped)


@dataclass
class CandidateCluster:
    cluster_id: str
    eg_property_id: str
    member_indices: list[int]
    review_ids: set[str]
    centroid: list[float]
    average_similarity: float
    example_snippets: list[str]

    @property
    def review_mentions(self) -> int:
        return len(self.review_ids)

    @property
    def snippet_mentions(self) -> int:
        return len(self.member_indices)


@dataclass(frozen=True)
class ConsolidatedTheme:
    label: str
    summary: str
    cluster_ids: tuple[str, ...]


class ThemeConsolidator(Protocol):
    model_name: str
    input_tokens_used: int
    output_tokens_used: int

    def consolidate(
        self,
        *,
        property_id: str,
        property_metadata: dict[str, str],
        candidate_clusters: list[CandidateCluster],
        top_themes: int,
    ) -> list[ConsolidatedTheme]:
        """Return final recurring themes for a single property."""


def load_reviews(reviews_path: Path, max_overall_rating: float | None = 2.0) -> list[ReviewRecord]:
    reviews: list[ReviewRecord] = []
    with reviews_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            overall_rating = _parse_overall_rating(row.get("rating", ""))
            if max_overall_rating is not None and overall_rating is not None and overall_rating > max_overall_rating:
                continue

            review_title = (row.get("review_title") or "").strip()
            review_text = (row.get("review_text") or "").strip()
            merged_text = " ".join(part for part in (review_title, review_text) if part).strip()
            if not merged_text:
                continue

            reviews.append(
                ReviewRecord(
                    review_id=f"{row['eg_property_id']}:{index}",
                    eg_property_id=row["eg_property_id"],
                    acquisition_date=(row.get("acquisition_date") or "").strip(),
                    overall_rating=overall_rating,
                    review_title=review_title,
                    review_text=merged_text,
                )
            )
    return reviews


def load_property_metadata(descriptions_path: Path) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    with descriptions_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metadata[row["eg_property_id"]] = {
                "city": (row.get("city") or "").strip(),
                "province": (row.get("province") or "").strip(),
                "country": (row.get("country") or "").strip(),
                "star_rating": (row.get("star_rating") or "").strip(),
                "guestrating_avg_expedia": (row.get("guestrating_avg_expedia") or "").strip(),
            }
    return metadata


def build_snippets(reviews: list[ReviewRecord]) -> list[ComplaintSnippet]:
    snippets: list[ComplaintSnippet] = []
    for review in reviews:
        parts = split_review_into_snippets(review.review_text)
        for position, part in enumerate(parts):
            snippets.append(
                ComplaintSnippet(
                    snippet_id=f"{review.review_id}:{position}",
                    review_id=review.review_id,
                    eg_property_id=review.eg_property_id,
                    acquisition_date=review.acquisition_date,
                    text=part,
                    overall_rating=review.overall_rating,
                )
            )
    return snippets


def aggregate_theme_summaries(
    assignments: list[tuple[ComplaintSnippet, ThemeMatch]],
    reviews: list[ReviewRecord],
    property_metadata: dict[str, dict[str, str]],
    top_themes: int = 3,
    min_review_mentions: int = 2,
    min_review_share: float = 0.08,
    max_examples: int = 3,
) -> list[dict[str, Any]]:
    reviews_by_property: dict[str, list[ReviewRecord]] = defaultdict(list)
    for review in reviews:
        reviews_by_property[review.eg_property_id].append(review)

    buckets_by_property: dict[str, dict[str, ThemeBucket]] = defaultdict(dict)
    for snippet, match in assignments:
        property_buckets = buckets_by_property[snippet.eg_property_id]
        bucket = property_buckets.setdefault(match.theme_key, ThemeBucket(match.theme_key, match.theme_label))
        bucket.add(snippet.review_id, match.confidence, snippet.text)

    summaries: list[dict[str, Any]] = []
    for property_id, property_reviews in sorted(reviews_by_property.items(), key=lambda item: item[0]):
        total_reviews = len(property_reviews)
        property_buckets = list(buckets_by_property.get(property_id, {}).values())
        property_min_mentions = 1 if total_reviews < 4 else min_review_mentions

        qualified = [
            bucket.to_summary(total_reviews=total_reviews, max_examples=max_examples)
            for bucket in property_buckets
            if len(bucket.review_ids) >= property_min_mentions
            and (total_reviews < 10 or len(bucket.review_ids) / total_reviews >= min_review_share)
        ]
        qualified.sort(
            key=lambda item: (item["review_mentions"], item["share_of_negative_reviews"], item["average_confidence"]),
            reverse=True,
        )

        if not qualified and property_buckets:
            fallback = sorted(
                (bucket.to_summary(total_reviews=total_reviews, max_examples=max_examples) for bucket in property_buckets),
                key=lambda item: (item["review_mentions"], item["average_confidence"]),
                reverse=True,
            )
            qualified = fallback[:1]

        summaries.append(
            {
                "eg_property_id": property_id,
                "property": property_metadata.get(property_id, {}),
                "negative_review_count": total_reviews,
                "themes": qualified[:top_themes],
            }
        )

    return summaries


def run_catalog_analysis(
    reviews_path: Path = DEFAULT_REVIEWS_PATH,
    descriptions_path: Path = DEFAULT_DESCRIPTIONS_PATH,
    *,
    embedder: EmbeddingClient,
    themes: tuple[ThemeDefinition, ...] = DEFAULT_THEME_CATALOG,
    top_themes: int = 3,
    min_review_mentions: int = 2,
    min_review_share: float = 0.08,
    max_overall_rating: float | None = 2.0,
    min_similarity: float = 0.18,
    min_margin: float = 0.015,
    high_confidence: float = 0.28,
) -> dict[str, Any]:
    reviews = load_reviews(reviews_path=Path(reviews_path), max_overall_rating=max_overall_rating)
    property_metadata = load_property_metadata(Path(descriptions_path))
    snippets = build_snippets(reviews)
    assigner = ThemeAssigner(
        embedder=embedder,
        themes=themes,
        min_similarity=min_similarity,
        min_margin=min_margin,
        high_confidence=high_confidence,
    )
    assignments = assigner.assign(snippets)
    return {
        "run_metadata": {
            "analysis_strategy": "catalog_matching",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "hotels": aggregate_theme_summaries(
            assignments=assignments,
            reviews=reviews,
            property_metadata=property_metadata,
            top_themes=top_themes,
            min_review_mentions=min_review_mentions,
            min_review_share=min_review_share,
        ),
    }


def run_llm_discovery_analysis(
    reviews_path: Path = DEFAULT_REVIEWS_PATH,
    descriptions_path: Path = DEFAULT_DESCRIPTIONS_PATH,
    *,
    embedder: EmbeddingClient,
    consolidator: ThemeConsolidator,
    embedding_model: str,
    top_themes: int = 3,
    min_review_mentions: int = 2,
    min_review_share: float = 0.08,
    max_overall_rating: float | None = 2.0,
    max_budget_usd: float = 1.0,
    cluster_assignment_threshold: float = 0.55,
    cluster_merge_threshold: float = 0.65,
    refinement_passes: int = 2,
    max_candidate_clusters: int = 8,
) -> dict[str, Any]:
    reviews = load_reviews(reviews_path=Path(reviews_path), max_overall_rating=max_overall_rating)
    property_metadata = load_property_metadata(Path(descriptions_path))
    snippets = build_snippets(reviews)
    reviews_by_property: dict[str, list[ReviewRecord]] = defaultdict(list)
    snippets_by_property: dict[str, list[ComplaintSnippet]] = defaultdict(list)
    for review in reviews:
        reviews_by_property[review.eg_property_id].append(review)
    for snippet in snippets:
        snippets_by_property[snippet.eg_property_id].append(snippet)

    estimated_embedding_cost = estimate_embedding_cost_usd(snippets, embedding_model)
    snippet_vectors = embedder.embed_texts([snippet.text for snippet in snippets])
    vector_lookup = {snippet.snippet_id: vector for snippet, vector in zip(snippets, snippet_vectors)}

    candidate_clusters_by_property: dict[str, list[CandidateCluster]] = {}
    for property_id, property_snippets in snippets_by_property.items():
        total_reviews = len(reviews_by_property[property_id])
        property_min_mentions = 1 if total_reviews < 4 else min_review_mentions
        candidate_clusters_by_property[property_id] = discover_candidate_clusters(
            snippets=property_snippets,
            vector_lookup=vector_lookup,
            assignment_threshold=cluster_assignment_threshold,
            merge_threshold=cluster_merge_threshold,
            refinement_passes=refinement_passes,
            min_review_mentions=property_min_mentions,
        )

    estimated_label_input_tokens = 0
    for property_id, candidate_clusters in candidate_clusters_by_property.items():
        prompt_preview = build_cluster_prompt(
            property_id=property_id,
            property_metadata=property_metadata.get(property_id, {}),
            candidate_clusters=candidate_clusters[:max_candidate_clusters],
            top_themes=top_themes,
        )
        estimated_label_input_tokens += estimate_token_count(prompt_preview)

    estimated_label_output_tokens = max(400, len(candidate_clusters_by_property) * max(150, top_themes * 120))
    estimated_total_cost = estimated_embedding_cost + estimate_model_cost_usd(
        model_name=getattr(consolidator, "model_name", ""),
        input_tokens=estimated_label_input_tokens,
        output_tokens=estimated_label_output_tokens,
    )
    if estimated_total_cost > max_budget_usd:
        raise RuntimeError(
            f"Estimated OpenAI cost ${estimated_total_cost:.4f} exceeds the budget cap of ${max_budget_usd:.2f}."
        )

    hotels: list[dict[str, Any]] = []
    for property_id, property_reviews in sorted(reviews_by_property.items(), key=lambda item: item[0]):
        total_reviews = len(property_reviews)
        property_min_mentions = 1 if total_reviews < 4 else min_review_mentions
        candidate_clusters = candidate_clusters_by_property[property_id][:max_candidate_clusters]
        final_themes = consolidator.consolidate(
            property_id=property_id,
            property_metadata=property_metadata.get(property_id, {}),
            candidate_clusters=candidate_clusters,
            top_themes=top_themes,
        )
        hotels.append(
            build_property_theme_summary(
                property_id=property_id,
                property_metadata=property_metadata.get(property_id, {}),
                property_reviews=property_reviews,
                candidate_clusters=candidate_clusters_by_property[property_id],
                final_themes=final_themes,
                top_themes=top_themes,
                min_review_mentions=property_min_mentions,
                min_review_share=min_review_share,
            )
        )

    actual_embedding_cost = estimate_model_cost_usd(
        model_name=embedding_model,
        input_tokens=getattr(embedder, "input_tokens_used", 0),
        output_tokens=0,
    )
    actual_label_cost = estimate_model_cost_usd(
        model_name=getattr(consolidator, "model_name", ""),
        input_tokens=getattr(consolidator, "input_tokens_used", 0),
        output_tokens=getattr(consolidator, "output_tokens_used", 0),
    )

    return {
        "run_metadata": {
            "analysis_strategy": "semantic_discovery",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "embedding_model": embedding_model,
            "label_model": getattr(consolidator, "model_name", "heuristic"),
            "budget_limit_usd": round(max_budget_usd, 2),
            "estimated_cost_usd": round(estimated_total_cost, 4),
            "actual_cost_usd": round(actual_embedding_cost + actual_label_cost, 4),
            "usage": {
                "embedding_input_tokens": getattr(embedder, "input_tokens_used", 0),
                "label_input_tokens": getattr(consolidator, "input_tokens_used", 0),
                "label_output_tokens": getattr(consolidator, "output_tokens_used", 0),
            },
        },
        "hotels": hotels,
    }


def run_analysis(
    reviews_path: Path = DEFAULT_REVIEWS_PATH,
    descriptions_path: Path = DEFAULT_DESCRIPTIONS_PATH,
    *,
    embedder: EmbeddingClient,
    consolidator: ThemeConsolidator | None = None,
    embedding_model: str = "text-embedding-3-small",
    analysis_strategy: str = "semantic_discovery",
    top_themes: int = 3,
    min_review_mentions: int = 2,
    min_review_share: float = 0.08,
    max_overall_rating: float | None = 2.0,
    min_similarity: float = 0.18,
    min_margin: float = 0.015,
    high_confidence: float = 0.28,
    max_budget_usd: float = 1.0,
    cluster_assignment_threshold: float = 0.55,
    cluster_merge_threshold: float = 0.65,
    refinement_passes: int = 2,
    max_candidate_clusters: int = 8,
) -> dict[str, Any]:
    if analysis_strategy == "catalog_matching":
        return run_catalog_analysis(
            reviews_path=reviews_path,
            descriptions_path=descriptions_path,
            embedder=embedder,
            top_themes=top_themes,
            min_review_mentions=min_review_mentions,
            min_review_share=min_review_share,
            max_overall_rating=max_overall_rating,
            min_similarity=min_similarity,
            min_margin=min_margin,
            high_confidence=high_confidence,
        )

    if consolidator is None:
        raise RuntimeError("A theme consolidator is required for semantic_discovery mode.")

    return run_llm_discovery_analysis(
        reviews_path=reviews_path,
        descriptions_path=descriptions_path,
        embedder=embedder,
        consolidator=consolidator,
        embedding_model=embedding_model,
        top_themes=top_themes,
        min_review_mentions=min_review_mentions,
        min_review_share=min_review_share,
        max_overall_rating=max_overall_rating,
        max_budget_usd=max_budget_usd,
        cluster_assignment_threshold=cluster_assignment_threshold,
        cluster_merge_threshold=cluster_merge_threshold,
        refinement_passes=refinement_passes,
        max_candidate_clusters=max_candidate_clusters,
    )


def discover_candidate_clusters(
    *,
    snippets: list[ComplaintSnippet],
    vector_lookup: dict[str, list[float]],
    assignment_threshold: float = 0.55,
    merge_threshold: float = 0.65,
    refinement_passes: int = 2,
    min_review_mentions: int = 2,
) -> list[CandidateCluster]:
    if not snippets:
        return []

    order = sorted(
        range(len(snippets)),
        key=lambda index: (-len(snippets[index].text.split()), snippets[index].review_id, snippets[index].snippet_id),
    )

    clusters = _online_cluster(order, snippets, vector_lookup, assignment_threshold)
    for _ in range(refinement_passes):
        clusters = _reassign_clusters(order, snippets, vector_lookup, clusters, assignment_threshold)
    clusters = _merge_clusters(snippets, vector_lookup, clusters, merge_threshold)

    draft_clusters: list[CandidateCluster] = []
    for member_indices in clusters:
        centroid = _cluster_centroid(member_indices, snippets, vector_lookup)
        scored_members = sorted(
            (
                (
                    cosine_similarity(vector_lookup[snippets[index].snippet_id], centroid),
                    index,
                )
                for index in member_indices
            ),
            reverse=True,
        )
        review_ids = {snippets[index].review_id for index in member_indices}
        if len(review_ids) < min_review_mentions:
            continue

        examples: list[str] = []
        seen_reviews: set[str] = set()
        for _, member_index in scored_members:
            snippet = snippets[member_index]
            if snippet.review_id in seen_reviews and len(examples) >= 3:
                continue
            seen_reviews.add(snippet.review_id)
            examples.append(snippet.text)
            if len(examples) == 5:
                break

        draft_clusters.append(
            CandidateCluster(
                cluster_id="",
                eg_property_id=snippets[member_indices[0]].eg_property_id,
                member_indices=member_indices,
                review_ids=review_ids,
                centroid=centroid,
                average_similarity=round(sum(score for score, _ in scored_members) / len(scored_members), 3),
                example_snippets=examples,
            )
        )

    draft_clusters.sort(
        key=lambda cluster: (cluster.review_mentions, cluster.snippet_mentions, cluster.average_similarity),
        reverse=True,
    )
    materialized: list[CandidateCluster] = []
    for cluster_index, cluster in enumerate(draft_clusters, start=1):
        materialized.append(
            CandidateCluster(
                cluster_id=f"c{cluster_index}",
                eg_property_id=cluster.eg_property_id,
                member_indices=cluster.member_indices,
                review_ids=cluster.review_ids,
                centroid=cluster.centroid,
                average_similarity=cluster.average_similarity,
                example_snippets=cluster.example_snippets,
            )
        )
    return materialized


def build_cluster_prompt(
    *,
    property_id: str,
    property_metadata: dict[str, str],
    candidate_clusters: list[CandidateCluster],
    top_themes: int,
) -> str:
    location_bits = [property_metadata.get("city", ""), property_metadata.get("province", ""), property_metadata.get("country", "")]
    location = ", ".join(bit for bit in location_bits if bit)
    lines = [
        "Hotel complaint cluster consolidation task.",
        f"Property ID: {property_id}",
        f"Location: {location or 'unknown'}",
        f"Keep at most {top_themes} final recurring themes.",
        "",
        "Candidate clusters:",
    ]
    for cluster in candidate_clusters:
        lines.append(
            f"{cluster.cluster_id} | reviews={cluster.review_mentions} | snippets={cluster.snippet_mentions} | similarity={cluster.average_similarity}"
        )
        for snippet in cluster.example_snippets[:4]:
            lines.append(f'- "{snippet}"')
        lines.append("")
    lines.append("Merge clusters that describe the same underlying issue even if phrased differently.")
    return "\n".join(lines)


def build_property_theme_summary(
    *,
    property_id: str,
    property_metadata: dict[str, str],
    property_reviews: list[ReviewRecord],
    candidate_clusters: list[CandidateCluster],
    final_themes: list[ConsolidatedTheme],
    top_themes: int,
    min_review_mentions: int,
    min_review_share: float,
) -> dict[str, Any]:
    total_reviews = len(property_reviews)
    cluster_lookup = {cluster.cluster_id: cluster for cluster in candidate_clusters}
    themes: list[dict[str, Any]] = []
    used_clusters: set[str] = set()

    for theme in final_themes:
        cluster_ids = [cluster_id for cluster_id in theme.cluster_ids if cluster_id in cluster_lookup]
        if not cluster_ids:
            continue
        unique_cluster_ids: list[str] = []
        for cluster_id in cluster_ids:
            if cluster_id in unique_cluster_ids:
                continue
            unique_cluster_ids.append(cluster_id)
        used_clusters.update(unique_cluster_ids)

        review_ids: set[str] = set()
        example_snippets: list[str] = []
        seen_examples: set[str] = set()
        snippet_mentions = 0
        similarities: list[float] = []
        for cluster_id in unique_cluster_ids:
            cluster = cluster_lookup[cluster_id]
            review_ids.update(cluster.review_ids)
            snippet_mentions += cluster.snippet_mentions
            similarities.append(cluster.average_similarity)
            for snippet in cluster.example_snippets:
                fingerprint = snippet.casefold()
                if fingerprint in seen_examples:
                    continue
                seen_examples.add(fingerprint)
                example_snippets.append(snippet)
                if len(example_snippets) == 3:
                    break

        review_mentions = len(review_ids)
        if review_mentions < min_review_mentions:
            continue
        share = review_mentions / total_reviews if total_reviews else 0.0
        if total_reviews >= 10 and share < min_review_share:
            continue

        label = clean_label(theme.label)
        themes.append(
            {
                "theme_key": slugify(label),
                "theme_label": label,
                "theme_summary": clean_sentence(theme.summary),
                "review_mentions": review_mentions,
                "share_of_negative_reviews": round(share, 3),
                "snippet_mentions": snippet_mentions,
                "average_cluster_similarity": round(sum(similarities) / len(similarities), 3) if similarities else 0.0,
                "source_cluster_ids": unique_cluster_ids,
                "example_snippets": example_snippets[:3],
            }
        )

    if not themes:
        for cluster in candidate_clusters[:top_themes]:
            share = cluster.review_mentions / total_reviews if total_reviews else 0.0
            themes.append(
                {
                    "theme_key": slugify(cluster.example_snippets[0][:32] if cluster.example_snippets else cluster.cluster_id),
                    "theme_label": heuristic_cluster_label(cluster),
                    "theme_summary": heuristic_cluster_summary(cluster),
                    "review_mentions": cluster.review_mentions,
                    "share_of_negative_reviews": round(share, 3),
                    "snippet_mentions": cluster.snippet_mentions,
                    "average_cluster_similarity": cluster.average_similarity,
                    "source_cluster_ids": [cluster.cluster_id],
                    "example_snippets": cluster.example_snippets[:3],
                }
            )

    deduped: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for theme in sorted(themes, key=lambda item: (item["review_mentions"], item["snippet_mentions"]), reverse=True):
        fingerprint = theme["theme_label"].casefold()
        if fingerprint in seen_labels:
            continue
        seen_labels.add(fingerprint)
        deduped.append(theme)
        if len(deduped) == top_themes:
            break

    return {
        "eg_property_id": property_id,
        "property": property_metadata,
        "negative_review_count": total_reviews,
        "themes": deduped,
    }


def estimate_embedding_cost_usd(snippets: list[ComplaintSnippet], model_name: str) -> float:
    input_tokens = sum(estimate_token_count(snippet.text) for snippet in snippets)
    return estimate_model_cost_usd(model_name=model_name, input_tokens=input_tokens, output_tokens=0)


def estimate_model_cost_usd(model_name: str, input_tokens: int, output_tokens: int) -> float:
    input_price = PRICE_USD_PER_1M_INPUT.get(model_name)
    output_price = PRICE_USD_PER_1M_OUTPUT.get(model_name, 0.0)
    if input_price is None:
        return 0.0
    return ((input_tokens * input_price) + (output_tokens * output_price)) / 1_000_000


def estimate_token_count(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def heuristic_cluster_label(cluster: CandidateCluster) -> str:
    if not cluster.example_snippets:
        return "Recurring complaint"
    first = clean_sentence(cluster.example_snippets[0])
    words = first.split()
    return clean_label(" ".join(words[:6])) or "Recurring complaint"


def heuristic_cluster_summary(cluster: CandidateCluster) -> str:
    if not cluster.example_snippets:
        return "Guests repeatedly raised a similar negative issue."
    return f"Guests repeatedly raised complaints like: {clean_sentence(cluster.example_snippets[0])}"


def write_analysis(output_path: Path, payload: Any) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def clean_label(label: str) -> str:
    cleaned = clean_sentence(label)
    return cleaned[:80] if cleaned else "Recurring complaint"


def clean_sentence(text: str) -> str:
    return " ".join(str(text).strip().split())


def slugify(text: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "_" for character in text)
    parts = [part for part in cleaned.split("_") if part]
    return "_".join(parts[:8]) or "recurring_complaint"


def _online_cluster(
    order: list[int],
    snippets: list[ComplaintSnippet],
    vector_lookup: dict[str, list[float]],
    assignment_threshold: float,
) -> list[list[int]]:
    clusters: list[list[int]] = []
    centroids: list[list[float]] = []
    for index in order:
        vector = vector_lookup[snippets[index].snippet_id]
        if not centroids:
            clusters.append([index])
            centroids.append(vector)
            continue

        best_cluster = -1
        best_score = -1.0
        for cluster_index, centroid in enumerate(centroids):
            score = cosine_similarity(vector, centroid)
            if score > best_score:
                best_score = score
                best_cluster = cluster_index

        if best_score >= assignment_threshold:
            clusters[best_cluster].append(index)
            centroids[best_cluster] = _cluster_centroid(clusters[best_cluster], snippets, vector_lookup)
        else:
            clusters.append([index])
            centroids.append(vector)
    return clusters


def _reassign_clusters(
    order: list[int],
    snippets: list[ComplaintSnippet],
    vector_lookup: dict[str, list[float]],
    previous_clusters: list[list[int]],
    assignment_threshold: float,
) -> list[list[int]]:
    if not previous_clusters:
        return []

    seed_centroids = [_cluster_centroid(cluster, snippets, vector_lookup) for cluster in previous_clusters]
    seeded_assignments: list[list[int]] = [[] for _ in seed_centroids]
    leftovers: list[int] = []

    for index in order:
        vector = vector_lookup[snippets[index].snippet_id]
        best_cluster = -1
        best_score = -1.0
        for cluster_index, centroid in enumerate(seed_centroids):
            score = cosine_similarity(vector, centroid)
            if score > best_score:
                best_score = score
                best_cluster = cluster_index
        if best_score >= assignment_threshold:
            seeded_assignments[best_cluster].append(index)
        else:
            leftovers.append(index)

    clusters = [cluster for cluster in seeded_assignments if cluster]
    for index in leftovers:
        clusters.append([index])
    return clusters


def _merge_clusters(
    snippets: list[ComplaintSnippet],
    vector_lookup: dict[str, list[float]],
    clusters: list[list[int]],
    merge_threshold: float,
) -> list[list[int]]:
    merged = [sorted(cluster) for cluster in clusters if cluster]
    while True:
        best_pair: tuple[int, int] | None = None
        best_score = merge_threshold
        centroids = [_cluster_centroid(cluster, snippets, vector_lookup) for cluster in merged]
        for left in range(len(merged)):
            for right in range(left + 1, len(merged)):
                score = cosine_similarity(centroids[left], centroids[right])
                if score > best_score:
                    best_pair = (left, right)
                    best_score = score
        if best_pair is None:
            return merged

        left, right = best_pair
        merged[left] = sorted(set(merged[left] + merged[right]))
        del merged[right]


def _cluster_centroid(
    member_indices: list[int],
    snippets: list[ComplaintSnippet],
    vector_lookup: dict[str, list[float]],
) -> list[float]:
    if not member_indices:
        return []
    dimensions = len(vector_lookup[snippets[member_indices[0]].snippet_id])
    vector_sum = [0.0] * dimensions
    for index in member_indices:
        vector = vector_lookup[snippets[index].snippet_id]
        for position, value in enumerate(vector):
            vector_sum[position] += value
    return normalize_vector(vector_sum)


def _parse_overall_rating(raw_rating: str) -> float | None:
    if not raw_rating:
        return None
    try:
        payload = json.loads(raw_rating)
    except json.JSONDecodeError:
        return None
    overall = payload.get("overall")
    try:
        return float(overall) if overall is not None else None
    except (TypeError, ValueError):
        return None
