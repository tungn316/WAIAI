# precompute.py
"""
Run offline to build property_profiles.json.
Usage: python -m your_package.precompute
"""
from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from belief_system import build_belief_system
from context_profile import build_context_profile

REVIEWS_PATH = Path("data/Reviews_PROC.csv")
DESCRIPTIONS_PATH = Path("data/Description_PROC.csv")
THEMES_PATH = Path("outputs/hotel_negative_themes.json")
OUTPUT_PATH = Path("outputs/property_profiles.json")


def parse_days_ago(date_str: str) -> int:
    if not date_str:
        return 730  # assume 2 years old if unknown
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(date_str.strip(), fmt).replace(tzinfo=timezone.utc)
            return max(0, (datetime.now(timezone.utc) - dt).days)
        except ValueError:
            continue
    return 730


def parse_rating(raw: str) -> float | None:
    if not raw:
        return None
    try:
        payload = json.loads(raw)
        overall = payload.get("overall")
        return float(overall) if overall is not None else None
    except (json.JSONDecodeError, TypeError, ValueError):
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None


def load_descriptions(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("eg_property_id", "").strip()
            if pid:
                rows[pid] = dict(row)
    return rows


def load_reviews_by_property(path: Path) -> dict[str, list[dict[str, Any]]]:
    by_property: dict[str, list[dict[str, Any]]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("eg_property_id", "").strip()
            if not pid:
                continue
            title = (row.get("review_title") or "").strip()
            text = (row.get("review_text") or "").strip()
            merged = " ".join(p for p in (title, text) if p)
            if not merged:
                continue
            by_property.setdefault(pid, []).append(
                {
                    "text": merged,
                    "rating": parse_rating(row.get("rating", "")),
                    "days_ago": parse_days_ago(row.get("acquisition_date", "")),
                }
            )
    return by_property


def load_negative_themes(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    hotels = data.get("hotels", data) if isinstance(data, dict) else data
    return {h["eg_property_id"]: h.get("themes", []) for h in hotels}


def main() -> None:
    print("Loading data...")
    descriptions = load_descriptions(DESCRIPTIONS_PATH)
    reviews_by_property = load_reviews_by_property(REVIEWS_PATH)
    negative_themes_by_property = load_negative_themes(THEMES_PATH)

    profiles: dict[str, Any] = {}
    total = len(descriptions)

    for idx, (pid, desc_row) in enumerate(descriptions.items(), 1):
        print(f"  [{idx}/{total}] {pid}", end="\r")
        reviews = reviews_by_property.get(pid, [])
        negative_themes = negative_themes_by_property.get(pid, [])

        negative_count = sum(
            1 for r in reviews
            if r["rating"] is not None and r["rating"] <= 2.0
        )

        beliefs = build_belief_system(
            description_row=desc_row,
            reviews=reviews,
        )

        context = build_context_profile(
            description_row=desc_row,
            negative_themes=negative_themes,
            total_review_count=len(reviews),
            negative_review_count=negative_count,
        )

        profiles[pid] = {
            "context": context.to_dict(),
            "beliefs": {k: v.to_dict() for k, v in beliefs.items()},
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(profiles, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {len(profiles)} property profiles to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
