"""
EmbeddingProvider: the provider abstraction Phase 7 asks for ("create a
provider abstraction so embedding providers can be changed later"), plus a
deterministic, zero-cost, zero-dependency FakeEmbeddingProvider.

Why FakeEmbeddingProvider lives in THIS module (not under tests/): it's a
legitimate, reusable implementation of the same Protocol a real provider
satisfies -- useful for demos and for any evaluator's own unit tests, not
test-only scaffolding (the same judgment call Day 3 made for MockSystem).
"""

import hashlib
import math
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    model: str

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, same order."""
        ...


class FakeEmbeddingProvider:
    """Deterministic, offline, free embeddings for testing the HARNESS'S
    OWN MECHANICS -- score computation, threshold logic, pass/fail -- never
    for judging real semantic quality. Builds a fixed-dimension vector via
    character-trigram feature hashing (the same family of technique as
    scikit-learn's HashingVectorizer): identical text always hashes
    identically (similarity 1.0); near-duplicate text shares most trigrams
    (high similarity); unrelated text is very unlikely to share many
    (low similarity).

    It has NO notion that "30" and "thirty" mean the same thing -- that
    requires a real embedding model's learned semantics. Confirming THAT
    works is exactly what tests/live/test_openai_embedding.py checks
    instead, against the real OpenAIEmbeddingProvider, with a real key."""

    model = "fake-trigram-hash-v1"

    def __init__(self, dimensions: int = 128):
        self._dims = dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dims
        padded = f"  {text.lower().strip()}  "
        for i in range(len(padded) - 2):
            trigram = padded[i : i + 3]
            bucket = int(hashlib.sha256(trigram.encode("utf-8")).hexdigest(), 16) % self._dims
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec
