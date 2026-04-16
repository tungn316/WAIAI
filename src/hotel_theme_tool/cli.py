from __future__ import annotations

import argparse
import os
from pathlib import Path

from .embeddings import HashingEmbeddingClient, OpenAIEmbeddingClient
from .labeling import HeuristicThemeConsolidator, OpenAIThemeConsolidator
from .pipeline import (
    DEFAULT_DESCRIPTIONS_PATH,
    DEFAULT_REVIEWS_PATH,
    run_analysis,
    write_analysis,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find the top recurring negative themes for each hotel based on review text."
    )
    parser.add_argument(
        "--strategy",
        choices=("semantic_discovery", "catalog_matching"),
        default="semantic_discovery",
        help="Use semantic discovery by default, or fall back to the older fixed-theme catalog matcher.",
    )
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS_PATH, help="Path to the reviews CSV.")
    parser.add_argument(
        "--descriptions",
        type=Path,
        default=DEFAULT_DESCRIPTIONS_PATH,
        help="Path to the property descriptions CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/hotel_negative_themes.json"),
        help="Where to write the JSON report.",
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "hashing"),
        default="openai",
        help="Embedding provider. OpenAI is recommended for semantic discovery; hashing is an offline fallback.",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="OpenAI embedding model when --provider openai is used.",
    )
    parser.add_argument(
        "--label-model",
        default="gpt-5.4-nano",
        help="OpenAI model used to merge and label recurring complaint themes in semantic discovery mode.",
    )
    parser.add_argument(
        "--max-budget-usd",
        type=float,
        default=1.0,
        help="Hard estimated budget cap for semantic discovery mode.",
    )
    parser.add_argument("--top-themes", type=int, default=3, help="Maximum number of themes to keep per hotel.")
    parser.add_argument(
        "--min-review-mentions",
        type=int,
        default=2,
        help="Minimum number of distinct negative reviews required for a theme.",
    )
    parser.add_argument(
        "--min-review-share",
        type=float,
        default=0.08,
        help="Minimum share of a hotel's negative reviews that must mention a theme for larger properties.",
    )
    parser.add_argument(
        "--max-overall-rating",
        type=float,
        default=2.0,
        help="Ignore reviews above this overall rating. Use a large number or omit filtering with --max-overall-rating -1.",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.18,
        help="Minimum similarity score required before a snippet is assigned to a theme.",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.015,
        help="Required gap between the best theme and the runner-up when the match is not already high confidence.",
    )
    parser.add_argument(
        "--high-confidence",
        type=float,
        default=0.28,
        help="Similarity score that lets a snippet pass even if the runner-up theme is close.",
    )
    parser.add_argument(
        "--cluster-assignment-threshold",
        type=float,
        default=0.55,
        help="Semantic similarity threshold used when building candidate complaint clusters.",
    )
    parser.add_argument(
        "--cluster-merge-threshold",
        type=float,
        default=0.65,
        help="Semantic similarity threshold used to merge near-duplicate complaint clusters.",
    )
    parser.add_argument(
        "--max-candidate-clusters",
        type=int,
        default=8,
        help="Maximum number of candidate clusters to send to the consolidator for each hotel.",
    )
    return parser


def load_local_env_files(base_dir: Path | None = None) -> None:
    base = (base_dir or Path.cwd()).resolve()
    for name in (".env", ".env.local"):
        path = base / name
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            os.environ.setdefault(key, value)


def main(argv: list[str] | None = None) -> int:
    load_local_env_files()
    parser = build_parser()
    args = parser.parse_args(argv)

    max_overall_rating = None if args.max_overall_rating < 0 else args.max_overall_rating

    if args.provider == "openai":
        embedder = OpenAIEmbeddingClient(model=args.embedding_model)
        consolidator = OpenAIThemeConsolidator(model=args.label_model)
    else:
        embedder = HashingEmbeddingClient()
        consolidator = HeuristicThemeConsolidator()

    payload = run_analysis(
        reviews_path=args.reviews,
        descriptions_path=args.descriptions,
        embedder=embedder,
        consolidator=consolidator,
        embedding_model=args.embedding_model,
        analysis_strategy=args.strategy,
        top_themes=args.top_themes,
        min_review_mentions=args.min_review_mentions,
        min_review_share=args.min_review_share,
        max_overall_rating=max_overall_rating,
        min_similarity=args.min_similarity,
        min_margin=args.min_margin,
        high_confidence=args.high_confidence,
        max_budget_usd=args.max_budget_usd,
        cluster_assignment_threshold=args.cluster_assignment_threshold,
        cluster_merge_threshold=args.cluster_merge_threshold,
        max_candidate_clusters=args.max_candidate_clusters,
    )
    write_analysis(args.output, payload)

    hotel_count = len(payload["hotels"]) if isinstance(payload, dict) and "hotels" in payload else len(payload)

    print(
        f"Wrote {hotel_count} hotel summaries to {args.output} using strategy={args.strategy} and provider={args.provider}.",
    )
    return 0
