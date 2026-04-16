from __future__ import annotations

import hashlib
import json
import math
import os
import re
import ssl
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol


class EmbeddingClient(Protocol):
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector per text."""


class OpenAIEmbeddingClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        base_url: str | None = None,
        batch_size: int = 128,
        timeout_seconds: int = 60,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        self.ssl_context = build_ssl_context()
        self.input_tokens_used = 0
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required when --provider openai is selected.")

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            if not batch:
                continue
            embeddings.extend(self._embed_batch(batch))
        return embeddings

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        payload = json.dumps({"input": batch, "model": self.model}).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.base_url}/embeddings",
            data=payload,
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
            raise RuntimeError(f"OpenAI embedding request failed with HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI embedding request failed: {exc.reason}") from exc

        data = sorted(body["data"], key=lambda item: item["index"])
        usage = body.get("usage", {})
        self.input_tokens_used += int(usage.get("prompt_tokens") or usage.get("total_tokens") or 0)
        return [normalize_vector(item["embedding"]) for item in data]


class HashingEmbeddingClient:
    """Offline fallback for local testing.

    This is not truly semantic, but it gives the rest of the pipeline a deterministic
    vector interface so the tool can be tested without network access.
    """

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions
        self.input_tokens_used = 0

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_single(text) for text in texts]

    def _embed_single(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        normalized = re.sub(r"\s+", " ", text.casefold()).strip()
        tokens = re.findall(r"[a-z0-9']+", normalized)

        for token in tokens:
            self._accumulate(vector, f"tok:{token}", 1.0)
        for left, right in zip(tokens, tokens[1:]):
            self._accumulate(vector, f"bigram:{left}_{right}", 1.4)
        for index in range(max(0, len(normalized) - 3)):
            gram = normalized[index : index + 4]
            if gram.strip():
                self._accumulate(vector, f"char:{gram}", 0.15)

        return normalize_vector(vector)

    def _accumulate(self, vector: list[float], feature: str, weight: float) -> None:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % self.dimensions
        sign = 1.0 if digest[-1] % 2 == 0 else -1.0
        vector[bucket] += sign * weight


def normalize_vector(values: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return [0.0 for _ in values]
    return [value / norm for value in values]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def build_ssl_context() -> ssl.SSLContext:
    bundle_path = os.environ.get("SSL_CERT_FILE")
    if bundle_path and Path(bundle_path).exists():
        return ssl.create_default_context(cafile=bundle_path)

    try:
        import certifi  # type: ignore
    except ModuleNotFoundError:
        certifi = None

    if certifi is not None:
        return ssl.create_default_context(cafile=certifi.where())

    for candidate in ("/etc/ssl/cert.pem", "/private/etc/ssl/cert.pem"):
        if Path(candidate).exists():
            return ssl.create_default_context(cafile=candidate)

    return ssl.create_default_context()
