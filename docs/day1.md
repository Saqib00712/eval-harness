# Project 2, Day 1 — Repo inspection + evaluation data model

This project lives in its own repo/folder, separate from `production-rag`
(Project 1) — no shared code, no shared git history, no shared Python
environment required. It only needs to be read side-by-side with Project 1
once (this inspection), to decide what to reuse as a *pattern*.

## Inspection (the working rule from the brief)

Before writing anything, Project 1's repo was inspected end-to-end.
Baseline there: `pytest -q` → **363 passed**, before any of this project's
code existed.

### Reusable as a PATTERN (not a direct import — eval_harness must never import `app.*`)

| Project 1 piece | Pattern worth reusing |
|---|---|
| `app/eval/retrieval_metrics.py` | Pure, I/O-free scoring functions (hit@k, MRR, recall@k) — the model for Phase 8's RAG evaluators |
| `app/services/llm.py` (`JsonLLM` Protocol, `OpenAIJsonLLM`) | Structured-output LLM wrapper with `response_format=json_schema` — exactly what Phase 9's LLM judge needs |
| `app/services/openai_client.py` | One shared `AsyncOpenAI` client per process, built lazily, closed in lifespan |
| `app/services/cost.py` | `estimate_cost_usd(tokens, price_per_million)` — trivial, reusable as-is conceptually |
| `app/eval/cost_gate.py` | Hard ceiling + typed exception — the template for Phase 17's quality gate |
| `app/observability/tracing.py` | contextvar-based trace/span recording, additive OTel export — the template for Phase 27 |
| `app/config.py` | `Pydantic BaseSettings` + `lru_cache get_settings()` — the template for eval_harness's own settings |
| `app/repositories/tenant_repository.py` | SQLite schema-in-code, context-managed connections, repository boundary — the template for Phase 25 storage |
| `app/services/embedder.py` (`Embedder` Protocol) | The model for Phase 26's `EmbeddingProvider` abstraction |

### NOT reusable as-is — tightly coupled to Project 1's RAG shape, must be generalized

- `app/eval/models.py` (`GoldenQuestion`, `EvalReport`) — hardcodes `expected_pages: list[int]`. Replaced by this package's generic `TestCase`.
- `app/eval/runner.py` — hardcodes a `SearchFn` + `AskFn` pair. Project 2's runner (coming Day 5) takes a generic `SystemUnderTest` + a list of `Evaluator`s instead.
- `app/eval/answer_metrics.py` — RAG-citation-specific pass/fail logic. Becomes one pluggable `CitationEvaluator` among several (Phase 8), not the only path.
- `app/services/judge.py` (`OpenAIJudge`) — one rubric baked into the system prompt. Phase 9/10 needs a rubric-*driven* judge; the Pydantic-validated-JSON-score pattern is kept, the hardcoded rubric is not.

### Decision

`eval_harness` is its own repo/project, with its own test tree (`tests/{unit,integration,live}/`) so Phase 30's "no paid API in normal tests" split is structural, not a convention to remember. Project 1 becomes a consumer of this package later (Phase 38: an adapter wrapping its `app.services.rag_pipeline`, installed as a dependency or imported via path — decided when that day comes), never the other way around.

## Day 1 deliverable — Phase 1: the evaluation data model

`eval_harness/models.py` — `TestCase`: `test_id`, `input`, `expected_output` (optional — not every system has one correct string), `reference_context`, `expected_citations` (plain `dict`s, not fixed to Project 1's `{"page": N}` shape), `metadata`, `tags`. Frozen (golden data shouldn't mutate mid-run). Validated with Pydantic.

`eval_harness/loader.py` — `load_test_cases(path)`: loads `.json` (one array) or `.jsonl` (one record per line), wraps not-found / malformed-JSON / wrong-shape / duplicate-`test_id`/empty-dataset into one `DatasetError` type.

19 tests, all offline (no OpenAI key needed) — `tests/unit/test_models.py`, `tests/unit/test_loader.py`.

**Total: 19 passed.** Project 1 is untouched (this is a separate repo).

## Next

Day 2 — Phase 2 (dataset versioning: `dataset_id`/version/hash, reproducibility) + Phase 31 (a realistic 20-example synthetic sample dataset), since you need a dataset to version.
