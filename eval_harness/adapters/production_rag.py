"""
Phase 38 (originally Day 19/Phase 30; corrected here against the real
source, not a reconstruction): the Project 1 integration adapter.

`production-rag` (Project 1 of this curriculum) is a separate repo this
package has never imported from and never will -- the independence rule
stated since this project's own Day 1 ("Make separate this Project"). The
only way to evaluate an already-deployed, running system with this harness
is the same way any real caller would use it: over HTTP, against whatever
`/ask` endpoint it exposes -- exactly as production-rag's own Streamlit
app, browser console, or a teammate's script already do.
`ProductionRagAdapter` satisfies `SystemUnderTest` (Phase 3) by wrapping
that one HTTP call. Nothing about `eval_harness.runner` or any evaluator
needed to change for a production-grade, multi-tenant, auth-protected RAG
system to become evaluable -- that's the Protocol-based design (Phase 3)
paying off exactly as intended: an adapter is a three-method class, not a
new runner mode.

CORRECTION FROM THE ORIGINAL (DAY 19) VERSION: that version's response
mapping was RECONSTRUCTED from this curriculum's build notes because the
actual production-rag source wasn't inspected directly. Phase 38 requires
inspecting the real implementation, and it turned out to exist in this
environment -- `production-rag/app/models/ask.py` and
`production-rag/app/routers/ask.py` were read directly. The real
`AskResponse` differs from the Day 19 guess in several real ways:

  * Refusal is `insufficient_evidence: bool` (+ `refusal_reason: str | None`),
    not a field called `refused`.
  * `citations` are real `Citation` objects --
    `{source_id, chunk_id, document_id, page_number, snippet, rerank_score}`
    -- not the guessed `{"page": N, "text": ...}` shape. Retrieved-context
    text lives in `snippet`, not `text`.
  * Cost is a NESTED `CostBreakdown` object (`cost.total_cost_usd`), not a
    flat `cost` float, and tracks ~10 separate token counts across
    embedding/rerank/generation/contextualizer/memory stages rather than
    one `prompt_tokens`/`completion_tokens` pair.
  * Latency is a per-stage `timings_ms: dict[str, float]`, not one number.
  * The model used is a per-stage `models: dict[str, str]`, not one
    `model` string.
  * There are also `degraded: bool` and `warnings: list[str]` fields the
    Day 19 guess didn't have at all.

This version reads the real shape directly (no more
`response_field_map`/`DEFAULT_FIELD_MAP` indirection -- that existed only
to hedge against a guessed contract being wrong, and it does not need to
exist once the contract is actually known; see `_to_system_output` below,
and `docs/day22.md` for the full before/after).

What this adapter deliberately does NOT do: it doesn't call
`/ask/stream` (production-rag's own SSE endpoint) -- evaluation wants one
complete answer per test case, and streaming buys nothing for that. It
doesn't manage conversation state across test cases -- each `TestCase` is
evaluated independently unless its `metadata` carries a `conversation_id`
(a valid UUID string, created via production-rag's own `POST
/conversations` -- `AskRequest.conversation_id` is typed as a UUID), in
which case that id is forwarded as-is. Useful for a dataset deliberately
built to exercise production-rag's multi-turn memory, but the harness
itself has no concept of test-case ordering or turns -- that's the dataset
author's responsibility, not this adapter's.
"""

import time
from typing import Any

from eval_harness.models import TestCase
from eval_harness.providers.http import HttpClient
from eval_harness.system_output import SystemOutput, TokenUsage

# production-rag/app/models/ask.py's CostBreakdown field names, split into
# which side of a call they represent. embedding_tokens and
# memory_embedding_tokens have no "completion" side (an embedding call has
# no generated text), so they're counted as input tokens -- the same
# convention OpenAI's own usage objects use for embedding calls.
_INPUT_TOKEN_FIELDS = (
    "embedding_tokens",
    "rerank_prompt_tokens",
    "generation_prompt_tokens",
    "contextualizer_prompt_tokens",
    "memory_extraction_prompt_tokens",
    "memory_embedding_tokens",
)
_OUTPUT_TOKEN_FIELDS = (
    "rerank_completion_tokens",
    "generation_completion_tokens",
    "contextualizer_completion_tokens",
    "memory_extraction_completion_tokens",
)


class ProductionRagAdapterError(Exception):
    """Raised only when production-rag itself reports an outright failure
    (a non-2xx HTTP status -- its routers raise `HTTPException(503/502/...)`
    for a missing API key, an embedding-provider failure, or a generation
    failure; see `app/routers/ask.py`). NOT raised for an ordinary refusal:
    production-rag represents a refusal as a normal 200 response with
    `insufficient_evidence=True`, and `run()` carries that through as
    `SystemOutput.metadata`, not an exception, per `SystemUnderTest.run`'s
    own contract -- only a crash should raise."""


class ProductionRagAdapter:
    """Wraps a running production-rag deployment's `POST /ask` endpoint as
    a `SystemUnderTest`. Construction never makes a network call -- only
    `run()` does, once per test case, exactly like any other
    `SystemUnderTest` implementation in this harness."""

    def __init__(
        self,
        http_client: HttpClient,
        base_url: str,
        api_key: str,
        *,
        name: str = "production-rag",
        version: str = "unknown",
        ask_path: str = "/ask",
        api_key_header: str = "X-API-Key",
        timeout_s: float = 30.0,
        top_k: int = 5,
        rerank: bool = True,
    ):
        self._http = http_client
        self._url = base_url.rstrip("/") + ask_path
        # production-rag's own app/auth.py accepts EITHER
        # `Authorization: Bearer <key>` or `X-API-Key: <key>` -- confirmed
        # by reading get_current_tenant there directly, not guessed.
        # X-API-Key is the default here because it needs no "Bearer "
        # prefix string-building at the call site.
        self._headers = {api_key_header: api_key, "Content-Type": "application/json"}
        self._name = name
        self._version = version
        self._timeout_s = timeout_s
        self._top_k = top_k
        self._rerank = rerank

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        """Defaults to `"unknown"` because production-rag's own version --
        a git commit, a deployed tag -- isn't something this adapter can
        observe over HTTP. Pass it explicitly at construction time (e.g.
        the commit you deployed) so comparison/regression reports (Days 10
        and 12) can tell one deployment from another."""
        return self._version

    async def run(self, test_case: TestCase) -> SystemOutput:
        # Matches production-rag's real AskRequest (app/models/ask.py):
        # question (required), document_id/conversation_id (optional
        # UUIDs), top_k, rerank. document_id/top_k/rerank come from this
        # adapter's own construction, not per-test-case, since a golden
        # dataset's TestCase has no first-class field for them; a dataset
        # that needs to vary them per case can still do so via metadata
        # (read below), the same escape hatch every other evaluator in
        # this harness already uses for per-case overrides.
        body: dict[str, Any] = {
            "question": test_case.input,
            "top_k": test_case.metadata.get("top_k", self._top_k),
            "rerank": test_case.metadata.get("rerank", self._rerank),
        }
        document_id = test_case.metadata.get("document_id")
        if document_id:
            body["document_id"] = document_id
        conversation_id = test_case.metadata.get("conversation_id")
        if conversation_id:
            body["conversation_id"] = conversation_id

        start = time.monotonic()
        status, data = await self._http.post(
            self._url, json=body, headers=self._headers, timeout_s=self._timeout_s
        )
        latency_ms = (time.monotonic() - start) * 1000

        if status >= 400:
            raise ProductionRagAdapterError(
                f"production-rag returned HTTP {status} for test case {test_case.test_id!r}: {data}"
            )

        return self._to_system_output(data, latency_ms)

    def _to_system_output(self, data: dict, latency_ms: float) -> SystemOutput:
        raw_citations = data.get("citations") or []
        # Each citation is kept as production-rag's own real shape
        # (source_id/chunk_id/document_id/page_number/snippet/
        # rerank_score) rather than remapped to some other key set -- a
        # malformed or partial citation dict (missing a key, an
        # unexpected type) is still kept as-is here; evaluators that
        # match on specific keys (CitationPrecisionEvaluator et al.)
        # simply won't match it, which is the correct degrade -- never a
        # crash in THIS adapter over a shape it doesn't own.
        citations = [c for c in raw_citations if isinstance(c, dict)]
        # Retrieved-context text for RAG evaluators (hit@k/recall@k/MRR,
        # context relevance) -- production-rag's Citation.snippet is the
        # matched chunk's text, confirmed from the real model (NOT a
        # guessed "text" key as Day 19's version assumed).
        context = [c["snippet"] for c in citations if isinstance(c.get("snippet"), str)]

        insufficient_evidence = bool(data.get("insufficient_evidence") or False)
        refusal_reason = data.get("refusal_reason")
        metadata: dict[str, Any] = {
            # Kept as "refused" (this harness's own vocabulary, used
            # consistently across every evaluator/report that reads
            # SystemOutput.metadata) even though production-rag's own
            # field is named insufficient_evidence -- the adapter's job is
            # translating into the harness's standardized format, not
            # importing Project 1's field names verbatim.
            "refused": insufficient_evidence,
        }
        if refusal_reason:
            metadata["refusal_reason"] = refusal_reason
        if data.get("degraded"):
            metadata["degraded"] = True
        if data.get("warnings"):
            metadata["warnings"] = data["warnings"]
        if data.get("search_query") and data["search_query"] != data.get("question"):
            metadata["search_query"] = data["search_query"]  # conversation follow-up was rewritten
        timings_ms = data.get("timings_ms")
        if timings_ms:
            # production-rag's own per-stage breakdown, kept for anyone
            # who wants it -- SystemOutput.latency_ms below is this
            # adapter's OWN measured round-trip time instead (see below
            # for why that's the more honest choice for the first-class
            # field).
            metadata["production_rag_timings_ms"] = timings_ms
        models_used = data.get("models")
        if models_used:
            metadata["production_rag_models"] = models_used

        cost_breakdown = data.get("cost") or {}
        total_cost = cost_breakdown.get("total_cost_usd")
        input_tokens = sum(int(cost_breakdown.get(f, 0) or 0) for f in _INPUT_TOKEN_FIELDS)
        output_tokens = sum(int(cost_breakdown.get(f, 0) or 0) for f in _OUTPUT_TOKEN_FIELDS)
        # None, not a TokenUsage(0, 0), when production-rag reported no
        # cost breakdown at all (an empty/missing `cost` object) -- the
        # same "undefined, not zero" discipline Day 13's efficiency.py
        # already applies to cost-per-passing-test: a system that didn't
        # REPORT token usage is a different, visible case from one that
        # genuinely used zero tokens.
        token_usage = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens) if cost_breakdown else None

        return SystemOutput(
            output=data.get("answer"),
            context=context,
            citations=citations,
            metadata=metadata,
            # This adapter's OWN wall-clock measurement around the HTTP
            # call, not a sum of production-rag's internal timings_ms --
            # the internal breakdown doesn't include network time, queueing,
            # or anything outside production-rag's own process, so it would
            # systematically understate what a caller actually experienced.
            # The internal breakdown is still preserved above
            # (metadata["production_rag_timings_ms"]) for anyone who wants
            # the stage-by-stage detail.
            latency_ms=latency_ms,
            token_usage=token_usage,
            estimated_cost=float(total_cost) if total_cost is not None else None,
            # The generation model is "the" model that produced this
            # answer; the full per-stage dict (embedding/rerank/generation/
            # contextualizer models) is preserved in metadata above.
            model=(models_used or {}).get("generation") if models_used else None,
        )
