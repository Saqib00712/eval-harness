"""
OpenAIJsonLLM: the real, billed JsonLLM implementation. Optional dependency
on purpose -- same discipline as providers/openai_embedding.py (Day 6):
`openai` is only imported lazily, inside __init__, and only installed via
`requirements-openai.txt`, so the base harness and its deterministic/
semantic/RAG evaluators keep working with zero AI SDK installed. This is
the exact bug Project 1 hit the hard way with `hnswlib` (Day 20 there) --
Day 6 already fixed it once for embeddings; this applies the same fix to
the LLM-judge provider.

The call shape is a near-direct reuse of Project 1's own `OpenAIJsonLLM`
(app/services/llm.py) -- JSON-mode structured output with a strict,
named schema, refusal detection, and a clear LLMError on anything
unusable, rather than silently returning a default score for a failed call.
"""

from eval_harness.providers.llm import LLMError
from eval_harness.system_output import TokenUsage


class OpenAIJsonLLM:
    def __init__(self, api_key: str):
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAIJsonLLM needs the `openai` package. Install it with: "
                "pip install -r requirements-openai.txt"
            ) from exc
        self._client = AsyncOpenAI(api_key=api_key)

    async def complete_json(self, *, model, system, user, schema_name, schema, max_tokens):
        from openai import OpenAIError

        try:
            resp = await self._client.chat.completions.create(
                model=model,
                temperature=0,  # deterministic-ish judging; remove if you switch to a reasoning model
                max_completion_tokens=max_tokens,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "strict": True, "schema": schema},
                },
            )
        except OpenAIError as exc:
            raise LLMError(f"LLM request failed: {type(exc).__name__}") from exc

        msg = resp.choices[0].message
        if getattr(msg, "refusal", None):
            raise LLMError("The model refused the request.")

        import json

        try:
            data = json.loads(msg.content or "")
        except json.JSONDecodeError as exc:
            raise LLMError("The model returned invalid JSON.") from exc
        if not isinstance(data, dict):
            raise LLMError("The model returned an unexpected JSON shape.")

        usage = (
            TokenUsage(input_tokens=resp.usage.prompt_tokens, output_tokens=resp.usage.completion_tokens)
            if resp.usage
            else TokenUsage()
        )
        return data, usage
