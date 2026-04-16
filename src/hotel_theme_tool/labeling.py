from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from .embeddings import HashingEmbeddingClient, build_ssl_context
from .models import ComplaintSnippet
from .pipeline import CandidateCluster, ConsolidatedTheme, ThemeAssigner, clean_label, clean_sentence
from .theme_catalog import DEFAULT_THEME_CATALOG


class OpenAIThemeConsolidator:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-5.4-nano",
        base_url: str | None = None,
        timeout_seconds: int = 60,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model_name = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.ssl_context = build_ssl_context()
        self.input_tokens_used = 0
        self.output_tokens_used = 0
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI theme consolidation.")

    def consolidate(
        self,
        *,
        property_id: str,
        property_metadata: dict[str, str],
        candidate_clusters: list[CandidateCluster],
        top_themes: int,
    ) -> list[ConsolidatedTheme]:
        if not candidate_clusters:
            return []

        prompt = self._build_prompt(
            property_id=property_id,
            property_metadata=property_metadata,
            candidate_clusters=candidate_clusters,
            top_themes=top_themes,
        )
        payload = {
            "model": self.model_name,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You analyze negative hotel review themes. "
                        "Merge different phrasings that point to the same underlying issue. "
                        "Prefer concrete recurring operational issues over vague dissatisfaction."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        body = self._post_json("/chat/completions", payload)
        content = body["choices"][0]["message"]["content"]
        data = json.loads(strip_json_fences(content))

        known_cluster_ids = {cluster.cluster_id for cluster in candidate_clusters}
        results: list[ConsolidatedTheme] = []
        for item in data.get("themes", [])[:top_themes]:
            cluster_ids: list[str] = []
            for cluster_id in item.get("cluster_ids", []):
                if cluster_id in known_cluster_ids and cluster_id not in cluster_ids:
                    cluster_ids.append(cluster_id)
            if not cluster_ids:
                continue
            label = clean_label(item.get("label", "Recurring complaint"))
            summary = clean_sentence(item.get("summary", ""))
            results.append(
                ConsolidatedTheme(
                    label=label,
                    summary=summary or f"Guests repeatedly mention problems related to {label.lower()}.",
                    cluster_ids=tuple(cluster_ids),
                )
            )
        return results

    def _build_prompt(
        self,
        *,
        property_id: str,
        property_metadata: dict[str, str],
        candidate_clusters: list[CandidateCluster],
        top_themes: int,
    ) -> str:
        lines = [
            "Consolidate these candidate complaint clusters into the final recurring problems for one hotel.",
            f"Property ID: {property_id}",
            "Hotel metadata:",
            json.dumps(property_metadata, ensure_ascii=False),
            "",
            "Rules:",
            f"- Return between 1 and {top_themes} themes.",
            "- Return 2-3 themes when the evidence supports multiple recurring issues; return only 1 if one problem clearly dominates and the rest are sparse.",
            "- Merge clusters that reflect the same latent issue even when wording differs.",
            "- Avoid labels like 'bad experience' or 'bad hotel' when a more concrete issue exists.",
            "- Prefer labels such as noise, rude front desk, dirty rooms, smell, heat, sunlight, location/access, parking, maintenance, misleading listing, missing amenities, or similar concrete issues when supported.",
            "- Use only the provided cluster_ids.",
            "",
            "Candidate clusters:",
        ]
        for cluster in candidate_clusters:
            lines.append(
                f"{cluster.cluster_id}: reviews={cluster.review_mentions}, snippets={cluster.snippet_mentions}, similarity={cluster.average_similarity}"
            )
            for snippet in cluster.example_snippets[:4]:
                lines.append(f'- "{snippet}"')
            lines.append("")
        lines.append("Return JSON only in this shape:")
        lines.append(
            '{"themes":[{"label":"short concrete label","summary":"one-sentence explanation","cluster_ids":["c1","c2"]}]}'
        )
        return "\n".join(lines)

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            url=f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds, context=self.ssl_context) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI theme consolidation failed with HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI theme consolidation failed: {exc.reason}") from exc

        usage = body.get("usage", {})
        self.input_tokens_used += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        self.output_tokens_used += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        return body


class HeuristicThemeConsolidator:
    def __init__(self, assigner: ThemeAssigner | None = None) -> None:
        self.assigner = assigner or ThemeAssigner(embedder=HashingEmbeddingClient(), themes=DEFAULT_THEME_CATALOG)
        self.model_name = "heuristic"
        self.input_tokens_used = 0
        self.output_tokens_used = 0

    def consolidate(
        self,
        *,
        property_id: str,
        property_metadata: dict[str, str],
        candidate_clusters: list[CandidateCluster],
        top_themes: int,
    ) -> list[ConsolidatedTheme]:
        if not candidate_clusters:
            return []

        synthetic_snippets: list[ComplaintSnippet] = []
        for cluster in candidate_clusters:
            synthetic_snippets.append(
                ComplaintSnippet(
                    snippet_id=cluster.cluster_id,
                    review_id=cluster.cluster_id,
                    eg_property_id=property_id,
                    acquisition_date="",
                    text=" ".join(cluster.example_snippets[:3]),
                    overall_rating=None,
                )
            )
        match_lookup = {
            snippet.snippet_id: match
            for snippet, match in self.assigner.assign(synthetic_snippets)
        }

        results: list[ConsolidatedTheme] = []
        for cluster in candidate_clusters[:top_themes]:
            match = match_lookup.get(cluster.cluster_id)
            label = match.theme_label if match else clean_label(cluster.example_snippets[0] if cluster.example_snippets else cluster.cluster_id)
            summary = (
                f"Guests repeatedly mention issues related to {label.lower()}."
                if match
                else f"Guests repeatedly mention complaints like: {clean_sentence(cluster.example_snippets[0])}"
            )
            results.append(
                ConsolidatedTheme(
                    label=label,
                    summary=summary,
                    cluster_ids=(cluster.cluster_id,),
                )
            )
        return results


def strip_json_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped
