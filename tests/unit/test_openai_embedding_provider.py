import pytest

pytest.importorskip("openai")  # this file only tests construction, not real calls -- no OPENAI_API_KEY needed

from eval_harness.providers.openai_embedding import OpenAIEmbeddingProvider


def test_constructs_with_default_model():
    provider = OpenAIEmbeddingProvider(api_key="sk-fake-not-a-real-key")
    assert provider.model == "text-embedding-3-small"


def test_constructs_with_custom_model():
    provider = OpenAIEmbeddingProvider(api_key="sk-fake-not-a-real-key", model="text-embedding-3-large")
    assert provider.model == "text-embedding-3-large"
