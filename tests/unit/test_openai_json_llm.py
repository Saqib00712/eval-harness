import pytest

pytest.importorskip("openai")  # construction-only -- no real API calls, no OPENAI_API_KEY needed

from eval_harness.providers.openai_llm import OpenAIJsonLLM


def test_constructs_with_api_key():
    llm = OpenAIJsonLLM(api_key="sk-fake-not-a-real-key")
    assert llm is not None
