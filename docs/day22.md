# Day 22 — Phase 38: Project 1 integration, corrected against the real source

The original Day 19 delivered `ProductionRagAdapter` with an honestly
disclosed limitation: production-rag's real source wasn't available to
inspect from within this harness's own environment, so its `AskResponse`
shape was *reconstructed* from this curriculum's build notes across
Project 1's 21 days. That reconstruction turned out to be wrong in
several concrete ways. This day fixes it against the actual
implementation.

## What changed, and why

Both `production-rag/app/models/ask.py` and
`production-rag/app/routers/ask.py` were read directly (not guessed, not
inferred from memory) before touching the adapter. The real `AskResponse`:

| Field | Day 19 guess | Real shape |
|---|---|---|
| Refusal | `refused: bool` | `insufficient_evidence: bool` |
| Citations | `{"page": N, "text": ...}` | `Citation{source_id, chunk_id, document_id, page_number, snippet, rerank_score}` |
| Cost | flat `cost: float` | nested `cost: CostBreakdown` (`cost.total_cost_usd`, ~10 separate token fields across embedding/rerank/generation/contextualizer/memory stages) |
| Latency | flat `latency` (implied) | `timings_ms: dict[str, float]` (per pipeline stage) |
| Model | `model: str` | `models: dict[str, str]` (per pipeline stage) |
| — | (not modeled) | `degraded: bool`, `warnings: list[str]` — both real, both now mapped |

`eval_harness/adapters/production_rag.py` was rewritten to read the real
shape directly. The `response_field_map`/`DEFAULT_FIELD_MAP` indirection
from Day 19 is gone — it existed only to hedge against a guessed contract
being wrong, and there's no reason to keep that hedge once the contract
is actually known from the source. Specific decisions:

- **Citations are preserved verbatim**, not remapped to a different key
  set — `CitationPrecisionEvaluator`/`CitationRecallEvaluator` already
  support `match_fields=[...]` for exactly this (compare on `page_number`
  only, ignore `snippet`/`rerank_score`/etc.), so there's no reason for
  the adapter to reshape the real object.
- **Token usage is a documented aggregation**, not an invented field:
  `TokenUsage.input_tokens` sums `embedding_tokens` +
  `*_prompt_tokens` + `memory_embedding_tokens` (an embedding call has no
  "completion" side — the same convention OpenAI's own usage objects
  use); `output_tokens` sums the `*_completion_tokens` fields. `None`
  (not a zero-filled `TokenUsage`) when `cost` is absent entirely — the
  same "undefined, not zero" discipline Day 13's `efficiency.py` already
  uses.
- **`SystemOutput.latency_ms` stays the adapter's own wall-clock
  measurement**, not a sum of `timings_ms` — the internal breakdown
  excludes network time and anything outside production-rag's own
  process, so summing it would systematically understate what a caller
  actually experienced. The real per-stage breakdown is preserved
  verbatim in `metadata["production_rag_timings_ms"]` for anyone who
  wants it.
- **`SystemOutput.model`** is `models["generation"]` specifically — the
  model that actually produced the answer — with the full per-stage dict
  in `metadata["production_rag_models"]`.
- `degraded`/`warnings`, when present and non-empty, land in `metadata`
  — the same "escape hatch for anything system-specific" pattern Day 4's
  `SystemOutput.metadata` was designed for.
- Request shape also corrected: production-rag's real `AskRequest` needs
  `top_k`/`rerank` (not just `question`), both now sent with constructor
  defaults overridable via `test_case.metadata` (`top_k`, `rerank`,
  `document_id`), the same per-case-override convention every evaluator
  in this harness already uses.
- Auth: both `Authorization: Bearer <key>` and `X-API-Key: <key>` are
  genuinely accepted (confirmed by reading `app/auth.py`'s
  `get_current_tenant` directly) — the adapter's `X-API-Key` default was
  accidentally already correct.

## Malformed/partial citations: verified, not assumed

A citation missing expected keys, or a non-dict entry in the `citations`
list, is kept as-is or dropped (never raises) — an evaluator matching on
a missing key simply won't match it. Explicitly tested
(`test_malformed_citation_entries_are_dropped_not_crashed_on`,
`test_citation_missing_expected_keys_is_kept_as_is`).

## Tests

`tests/unit/test_production_rag_adapter.py` was rewritten end to end
against the real contract (32 tests total between it and the e2e file,
up from 19) — request shape (`top_k`/`rerank`/`document_id` forwarding),
response mapping (citations, `insufficient_evidence`, `degraded`,
`warnings`, `search_query`, summed token usage, the adapter's own
latency vs. the preserved `timings_ms`), malformed-citation handling, and
error propagation (non-2xx → `ProductionRagAdapterError`, transport
failure → `HttpError` untouched).

`tests/unit/test_production_rag_adapter_e2e.py` gained two new tests
proving the corrected mapping actually feeds the evaluators Phase 38
requires: `test_adapter_output_feeds_retrieval_and_citation_evaluators`
(hit@k, citation precision, citation recall, all through the real
pipeline) and `test_adapter_output_feeds_mocked_faithfulness_judge`
(faithfulness via a `FakeJsonLLM`-mocked judge — zero real/paid calls).

Day 20's demo script (`scripts/demo_end_to_end.py`) and its dataset
builder were also updated — its scripted responses now use the real
`AskResponse` shape, and `expected_citations` is remapped from the
shared dataset's generic `page` key to production-rag's real
`page_number` for this demo's purposes (documented inline; the dataset
file on disk is untouched). The demo's narrative (candidate wins overall
but fails the quality gate on one regression) is unchanged — only the
wire format underneath it is now accurate.

**319 passed, 4 skipped** (up from 306 — 13 net new/rewritten tests
across the adapter and e2e files).

## Scope note (per this phase's own requirement #4)

The harness evaluates production-rag's `POST /ask` only — the complete
question-in, grounded-answer-out pipeline, which is what `SystemUnderTest`
is built to evaluate. `POST /search` (retrieval-only) was inspected but
deliberately not wrapped as a second adapter: retrieval quality is
already evaluated from `/ask`'s own citations (which carry the same
`snippet`/`page_number` a `/search` hit would), and a second adapter type
for a raw-retrieval system isn't something this harness's abstraction
needs duplicated — `SystemUnderTest` models an end-to-end system, and
`/ask` already is one.

## Next

Phase 39 — failure injection and regression verification.
