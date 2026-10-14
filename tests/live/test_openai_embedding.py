"""
Live test: real OpenAI embeddings, real cost (a fraction of a cent).
Skips automatically with no OPENAI_API_KEY set -- the exact same
skip-don't-fail pattern production-rag's own OCR tests use (Day 21 there)
when their optional dependency isn't present. Run explicitly with:

    OPENAI_API_KEY=sk-... pytest tests/live -v
"""

import os

import pytest

pytest.importorskip("openai")

from eval_harness.providers.openai_embedding import OpenAIEmbeddingProvider
from eval_harness.similarity import cosine_similarity

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="live test needs a real OPENAI_API_KEY"
)


async def test_real_embeddings_rate_a_paraphrase_as_similar():
    """The exact example from the brief: different wording, same meaning,
    should score high -- something FakeEmbeddingProvider (unit tests)
    cannot prove, since it has no notion of real-world meaning."""
    provider = OpenAIEmbeddingProvider(api_key=os.environ["OPENAI_API_KEY"])
    vectors = await provider.embed([
        "The refund period is 30 days.",
        "Customers have thirty days to request their refund.",
    ])
    score = cosine_similarity(vectors[0], vectors[1])
    assert score > 0.7


async def test_real_embeddings_rate_unrelated_text_as_dissimilar():
    provider = OpenAIEmbeddingProvider(api_key=os.environ["OPENAI_API_KEY"])
    vectors = await provider.embed([
        "The refund period is 30 days.",
        "Mount Everest is the tallest mountain on Earth.",
    ])
    score = cosine_similarity(vectors[0], vectors[1])
    assert score < 0.5
