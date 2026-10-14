"""Pure vector-similarity math. No I/O, no Pydantic -- trivially testable
and reusable anywhere a vector comparison is needed (semantic evaluation
today; RAG context-relevance scoring, Phase 8, will reuse this too)."""

import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """The cosine of the angle between two vectors: 1.0 = pointing the same
    direction, 0.0 = orthogonal (unrelated), -1.0 = pointing opposite
    directions. Chosen over raw dot product or Euclidean distance because it
    ignores vector MAGNITUDE -- for text embeddings, magnitude is mostly an
    artifact of text length, not of meaning, so a metric that ignores it is
    the right one for "how similar in meaning are these two texts"."""
    if len(a) != len(b):
        raise ValueError(f"Vectors must be the same length ({len(a)} vs {len(b)}).")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0  # a zero vector has no direction to compare -- defined as maximally dissimilar, not an error
    return dot / (norm_a * norm_b)
