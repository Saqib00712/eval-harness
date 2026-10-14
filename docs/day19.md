# Day 19 — Project 1 integration adapter

Phase 30. The first day this harness evaluates a real, external system
rather than `MockSystem` or a `Fake*` provider standing in for one:
`production-rag` (Project 1 of this curriculum), wired in as a
`SystemUnderTest` without this package ever importing a single line from
that repo.

## Why HTTP, not an import

The independence rule has held since Project 2's own Day 1: this harness
and `production-rag` are separate repos, and neither imports the other.
Project 1 is a running FastAPI service with an `/ask` endpoint, API-key
auth, and multi-tenant isolation (its own Day 10) — the same interface its
Streamlit app, browser console, and any real caller already use. Evaluating
it the same way is both the only option this independence rule leaves open
and the most honest one: it tests what's actually deployed, auth and all,
not an in-process shortcut that could drift from what a real request
experiences.

## What was built

**`eval_harness/providers/http.py`** — a small `HttpClient` Protocol
(`post(url, json, headers) -> (status, body)`), following the same
optional-dependency shape as `providers/llm.py`/`providers/embedding.py`:

- `FakeHttpClient` — scripted responses (`default` or `queue`, same two
  modes `FakeJsonLLM` already established), every call recorded in
  `.calls` for assertions. This is what every adapter test below runs
  against — no network, no live server.
- `HttpxHttpClient` — the real transport, `httpx` as a lazy, optional
  import (`requirements-httpx.txt`). Unlike the OpenAI providers' live
  call paths, `httpx` ships `httpx.MockTransport`, so this class's actual
  request/response handling — status codes, JSON parsing, error wrapping —
  is unit-tested for real (a fake transport, but real `httpx` code
  underneath) instead of being excluded from coverage the way a judge
  LLM's real call necessarily is.

**`eval_harness/adapters/production_rag.py`** — `ProductionRagAdapter`,
a `SystemUnderTest` wrapping one HTTP call to `/ask`:

- Sends `{"question": test_case.input}`, plus `conversation_id` when a
  test case's `metadata` carries one (for a dataset deliberately built to
  exercise Project 1's Day 13 multi-turn memory — the harness itself has
  no concept of test-case ordering or turns, so this is opt-in per case,
  never inferred).
- Maps the response's `answer` → `SystemOutput.output`, `citations` →
  `SystemOutput.citations` (and doubles as `context` when a citation
  carries its own `text`), `cost` → `estimated_cost`, `refused`/
  `refusal_reason` → `metadata` (a refusal is data, not an exception — see
  below), `prompt_tokens`/`completion_tokens` → `TokenUsage`.
- A non-2xx HTTP status raises `ProductionRagAdapterError` — that's
  `production-rag` itself failing, which `SystemUnderTest.run`'s own
  contract says should raise so the runner's failure isolation (Day 5)
  can classify it as a `DEPENDENCY_FAILURE` (Day 9). An ordinary refusal
  (a normal 2xx response with `refused: true`) is NOT an error — it's a
  legitimate thing production-rag can represent in-band, so it comes back
  as a populated `SystemOutput` with `metadata["refused"] = True`.

**The honesty about the response shape**: `production-rag`'s exact
`AskResponse` fields were reconstructed from this curriculum's own build
notes across Project 1's 21 days, not read from its source — this package
structurally can't do that. If a real deployment's field names differ,
nothing about the adapter's logic needs to change: `response_field_map`
remaps any of the 8 fields `DEFAULT_FIELD_MAP` names to whatever a real
`AskResponse` actually calls them, from outside the adapter.

## Tests

26 new tests across three files:

- `test_http_provider.py` — `FakeHttpClient`'s default/queue/exhausted/
  no-response/raises behavior, plus `HttpxHttpClient.post` exercised for
  real via `httpx.MockTransport`: success, a passed-through error status,
  the request body/headers actually sent, a non-JSON response wrapped as
  `HttpError`, and a transport-level failure wrapped as `HttpError`.
- `test_production_rag_adapter.py` — request shape (URL, header, body,
  trailing-slash normalization, conversation_id forwarding), response
  mapping (answer, citations→context, refusal, token usage, latency,
  `response_field_map` overrides), and error handling (4xx/5xx raising
  `ProductionRagAdapterError`, a transport failure propagating as
  `HttpError` untouched).
- `test_production_rag_adapter_e2e.py` — proves the adapter isn't just
  structurally shaped like a `SystemUnderTest` (Day 18's audit found
  exactly that gap once already, for an unrelated class — an `isinstance`
  check against a `runtime_checkable` Protocol only inspects attribute
  names) but actually runs end-to-end through the real `run_evaluation`
  pipeline with a real evaluator, against `FakeHttpClient` standing in
  for a live deployment.

**301 passed, 4 skipped.** `eval_harness/*` coverage: 98% — the only
uncovered lines are `HttpxHttpClient`'s `except ImportError` guard (same
structural, by-design exclusion as every other optional-dependency guard
in this codebase) and the two OpenAI providers' live call paths.

## Using it against a real deployment

```python
from eval_harness.adapters.production_rag import ProductionRagAdapter
from eval_harness.providers.http import HttpxHttpClient

adapter = ProductionRagAdapter(
    HttpxHttpClient(),
    base_url="http://localhost:8000",
    api_key="<a key from production-rag's scripts/manage_keys.py>",
    version="<git commit or deployed tag>",
)
run = await run_evaluation(dataset, adapter, evaluators)
```

`pip install -r requirements-httpx.txt` first — the base package still
needs nothing beyond `requirements-dev.txt` for its own test suite.

## Next

Day 20 — a worked demonstration: a real eval dataset against a real
(or realistically faked) production-rag deployment, run through comparison
(Day 12), regression (Day 10), and a quality gate (Day 11) together, as
the first end-to-end story of why every prior day exists.
