"""
Real, OpenAI-backed EmbeddingProvider. Optional dependency
(requirements-openai.txt) -- importing/instantiating this without the
`openai` package installed raises a clear ImportError, not a cryptic one.

This is a deliberate, proactive application of a lesson Project 1 learned
the hard way (Day 20 there): hnswlib had been an unconditional hard
dependency even though most users never selected that vector-index
backend, forcing everyone to have C++ build tools just to install the
project. Here, the base `eval_harness` package works fully (deterministic
evaluators, MockSystem, FakeEmbeddingProvider-backed tests) with zero
OpenAI SDK installed; only code that actually wants REAL embeddings pulls
in `openai`.
"""


class OpenAIEmbeddingProvider:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAIEmbeddingProvider needs the `openai` package: "
                "pip install -r requirements-openai.txt"
            ) from exc
        self.model = model
        self._client = AsyncOpenAI(api_key=api_key)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        resp = await self._client.embeddings.create(model=self.model, input=texts)
        ordered = sorted(resp.data, key=lambda d: d.index)
        return [d.embedding for d in ordered]
