"""
JsonLLM: the provider abstraction Day 8's LLM-as-judge evaluators depend
on, plus FakeJsonLLM -- a deterministic, zero-cost, zero-dependency stand-in
for testing the JUDGE EVALUATORS' OWN LOGIC (prompt wiring, score
normalization, threshold handling) offline, the same role
FakeEmbeddingProvider plays for Day 6's SemanticSimilarityEvaluator.

The shape -- `complete_json(*, model, system, user, schema_name, schema,
max_tokens) -> (dict, TokenUsage)` -- is a direct reuse of Project 1's own
`JsonLLM` Protocol (app/services/llm.py): structured-output JSON mode with
a named/strict schema, so a judge's response is always a typed dict, never
free text to regex apart. Reusing the exact same shape here means the real
OpenAI-backed implementation (providers/openai_llm.py) is nearly a copy of
Project 1's `OpenAIJsonLLM` -- the pattern was already battle-tested there.
"""

from typing import Protocol, runtime_checkable

from eval_harness.system_output import TokenUsage


class LLMError(Exception):
    """An LLM call failed after retries, or returned unusable output
    (a refusal, invalid JSON, the wrong JSON shape)."""


@runtime_checkable
class JsonLLM(Protocol):
    async def complete_json(
        self, *, model: str, system: str, user: str, schema_name: str, schema: dict, max_tokens: int
    ) -> tuple[dict, TokenUsage]:
        """Return the parsed JSON response (matching `schema`) and the
        token usage the call consumed. Raise LLMError for anything that
        makes the response unusable -- the judge evaluator decides what an
        unusable judge call means for ITS result; this layer just reports
        failure honestly rather than silently returning a default score."""
        ...


class FakeJsonLLM:
    """Scripted, offline responses for testing judge-evaluator LOGIC, never
    for judging real answer quality -- it has no understanding of anything,
    it just plays back what it's told to.

    Two ways to script it:
      * `responses=<dict>` (a single canned response) -- returned for every
        call. Good for "does the evaluator wire a judge's score through
        correctly" tests.
      * `responses=<list of dicts>` -- returned one at a time, in call
        order; calling past the end raises IndexError (a test asked for
        more judge calls than it scripted, which is a test bug, not a
        valid outcome to paper over).

    Every call is recorded in `.calls` (model/system/user/schema_name) so a
    test can assert on exactly what prompt a judge evaluator sent, without
    needing a real API to inspect it."""

    def __init__(
        self,
        responses: dict | list[dict] | None = None,
        *,
        default: dict | None = None,
        usage: TokenUsage | None = None,
    ):
        if isinstance(responses, list):
            self._queue: list[dict] = list(responses)
            self._repeat_single = False
        elif isinstance(responses, dict):
            self._queue = [responses]
            self._repeat_single = True
        else:
            self._queue = []
            self._repeat_single = False
        self._default = default
        self._usage = usage or TokenUsage(input_tokens=50, output_tokens=20)
        self.calls: list[dict] = []

    async def complete_json(self, *, model, system, user, schema_name, schema, max_tokens):
        self.calls.append({"model": model, "system": system, "user": user, "schema_name": schema_name})
        if self._queue:
            data = self._queue[0] if self._repeat_single else self._queue.pop(0)
        elif self._default is not None:
            data = self._default
        else:
            raise IndexError(
                "FakeJsonLLM: no more scripted responses and no default set -- "
                "pass more `responses` or a `default`."
            )
        return data, self._usage
