"""Thin HTTP client for the MINIPC embedding service (multilingual-e5-base).

The service accepts POST /embed/batch with {"texts": [...]} and returns
{"embeddings": [[float, ...], ...]}.

Usage:
    client = EmbedClient("http://192.168.68.63:9092", api_key="...")
    vecs = client.embed(["query text", "sentence one", "sentence two"])
"""

from __future__ import annotations

import json
import math
from urllib.parse import urljoin
from urllib.request import Request, urlopen


class EmbedClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        collection: str = "search-engine",
        timeout: int = 60,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.collection = collection
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for each text.

        Calls POST /embed/batch and extracts the 'vectors' field
        (the MINIPC embedding-svc response schema).
        """
        if not texts:
            return []
        body = json.dumps({"texts": texts, "collection": self.collection}).encode()
        req = Request(
            urljoin(self.base_url + "/", "embed/batch"),
            data=body,
            headers={"Content-Type": "application/json", "X-API-Key": self.api_key},
        )
        resp = json.loads(urlopen(req, timeout=self.timeout).read())
        return resp["vectors"]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def embed_scores(query_vec: list[float], sentence_vecs: list[list[float]]) -> list[float]:
    """Return cosine similarity of each sentence vector against the query vector."""
    return [cosine_similarity(query_vec, sv) for sv in sentence_vecs]
