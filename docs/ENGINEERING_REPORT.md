# eval-harness — Final Engineering Report (Phase 41)

Covers the full project (Days 1–24 / Phases 1–40), with Phases 38–41
(Days 22–24, this continuation) as the most recent work. All numbers in
this report were produced by running the real test suite and the real
demo immediately before writing them down — none are recalled from
memory or estimated.

---

## 1. Project summary

**Problem:** teams building AI systems (RAG apps, LLM apps, agents,
classifiers) have no automatic, repeatable way to answer "did this change
make the system better or worse?" Manual spot-checking doesn't scale, and
a single pass-rate number hides *which* failure mode (wrong answer vs.
hallucinated claim vs. broken retrieval vs. a crashed dependency) is
responsible for a regression.

**Who this is for:** an engineer or team shipping an AI system who wants
the same discipline as unit testing applied to model/prompt/retrieval
changes — "does this PR regress anything, and can CI block it
automatically" — without hand-building evaluation plumbing from scratch
for every project.

**Industry value:** this is the same category of tool as DeepEval,
Ragas, or a custom internal eval harness at an AI product company — a
generic evaluation engine with pluggable evaluators (deterministic,
embedding-based, LLM-judge, RAG-specific) sitting between a dataset of
golden test cases and any system that can be wrapped in a simple
`run(question) -> answer` interface.

## 2. Architecture

```
Dataset (versioned, hashed)                 SystemUnderTest (Protocol)
        |                                            |
        |                 run_evaluation()           |
        +------------------>  runs each  <------------+
                  test case through the system,
                  then through every Evaluator
                           |
                           v
              PerTestResult (score, passed,
              failure_category, latency, tokens)
                           |
                           v
                  EvaluationRun (aggregated:
                  pass_rate, average_score,
                  per-run latency/cost)
                 /          |            \
                v           v             v
          storage.py   regression.py   efficiency.py /
         (persist,     (compare vs.    observability.py
          reload)       a baseline)    (diagnose, watch)
                           |
                           v
                   quality_gate.py
              (hard pass/fail verdict,
               exit code for CI)
```

- **Dataset ingestion** (`dataset.py`/`dataset_io.py`/`loader.py`): a
  `Dataset` is a named, versioned collection of `TestCase`s with a
  computed content hash (detects silent tampering/edits) and enforced
  resource ceilings on untrusted file input (`security_limits.py`).
- **SUT abstraction** (`system.py::SystemUnderTest`): a 3-method
  `Protocol` (`name`, `version`, async `run`). Anything satisfying it —
  `MockSystem`, `ProductionRagAdapter`, or a future adapter for a
  different system — plugs into the exact same `run_evaluation()` with
  zero changes to the runner or any evaluator.
- **Evaluator interfaces** (`evaluator.py`/`evaluators/*`): a common
  `Evaluator` Protocol with three families — deterministic (exact match,
  contains, regex, JSON schema, length, required fields — free, instant,
  no LLM), embedding-based (`SemanticSimilarityEvaluator`), and
  LLM-judge (`CorrectnessJudgeEvaluator`/`FaithfulnessJudgeEvaluator`/
  `RubricJudgeEvaluator`), plus RAG-specific retrieval and citation
  evaluators (Hit@K/Recall@K/MRR, citation precision/recall,
  context relevance, faithfulness-by-overlap).
- **Run orchestration** (`runner.py`): `run_evaluation()` wires a
  `Dataset` + `SystemUnderTest` + a list of `Evaluator`s together,
  isolating failures at both the system level (a crash becomes
  `DEPENDENCY_FAILURE`, not an unhandled exception) and the evaluator
  level (`EVALUATION_FAILURE`) — a normal scored-but-failing result is
  `THRESHOLD_MISS`. An optional `RunObserver` hook watches progress live.
- **Persistence/metrics** (`storage.py`/`run_models.py`): `EvaluationRun`s
  and `HumanLabel`s persist to disk as JSON, organized by
  `(dataset_id, system_name)` with path-traversal hardening on every
  caller-controlled path segment.
- **Failure classification** (`failure_taxonomy.py`): the closed
  three-category taxonomy above, applied to every failed result by the
  runner itself, not left to callers to infer.
- **Regression comparison** (`regression.py`): `compare_runs()` compares
  two runs of the *same* dataset (rejects cross-dataset comparisons with
  a clear error), reporting newly-passing/regressed test IDs and
  per-evaluator deltas.
- **Quality gates** (`quality_gate.py`): `evaluate_quality_gate()` turns
  a run (optionally plus a regression report) into one pass/fail verdict
  and a CI-usable exit code — static thresholds default to *unconfigured
  (always pass)*, since there's no universal "good enough" number across
  every possible system.
- **API/CLI** (`cli.py`): **CLI only**, by explicit scope decision (see
  §10) — `validate-dataset`, `quality-gate`, `compare`, `demo`.
- **Project 1 adapter** (`adapters/production_rag.py`): wraps
  `production-rag`'s real `POST /ask` endpoint as a `SystemUnderTest`,
  built against the actual source (not a guess) after Phase 38's
  inspection — see §3 and §4.
- **End-to-end data flow**: dataset file → `load_dataset()` → `Dataset` →
  `run_evaluation(dataset, system, evaluators, observer)` → one
  `EvaluationRun` → `save_run()` to disk → later, `load_run()` both a
  baseline and candidate → `compare_runs()` → `RegressionReport` →
  `evaluate_quality_gate()` → `QualityGateResult` → `exit_code_for()` →
  CI's pass/fail. The CLI's `quality-gate --baseline` command and
  `compare` command are both thin wrappers over this exact same chain of
  function calls, not a reimplementation of it.

## 3. Files changed (Phases 38–41 / Days 22–24)

**Rewritten:**
- `eval_harness/adapters/production_rag.py` — full rewrite against the
  real `production-rag` contract (was a reconstructed guess). No public
  `SystemUnderTest`-facing behavior changed (still `name`/`version`/
  `run`); internal field-mapping is now correct.
- `tests/unit/test_production_rag_adapter.py` — rewritten end to end (28
  tests now, from 19) against the real request/response shapes.
- `tests/unit/test_production_rag_adapter_e2e.py` — existing test's
  fixture data corrected to the real nested shape; 2 new tests added
  (retrieval/citation evaluators, faithfulness judge, both fed through
  the real adapter mapping).
- `scripts/demo_end_to_end.py` — scripted response bodies rewritten to
  the real `AskResponse` shape; `expected_citations` remapped from a
  generic `page` key to the real `page_number` key for this demo only
  (the dataset file on disk is untouched); `CitationPrecisionEvaluator`
  now constructed with `match_fields=["page_number"]`.
- `README.md` — status line, module list, and CLI usage updated to match
  the actual current repository.

**Created:**
- `eval_harness/cli.py` — the Phase 40 CLI.
- `tests/unit/test_cli.py` — 12 tests covering all four subcommands and
  the parser shape.
- `tests/integration/test_failure_injection.py` — 17 tests, the Phase 39
  failure-injection suite.
- `docs/day22.md`, `docs/day23.md`, `docs/day24.md` — day-by-day
  implementation notes for Phases 38, 39, 40.
- `docs/DEFINITION_OF_DONE.md` — the Phase 40 checklist.
- `docs/ENGINEERING_REPORT.md` — this file.

**Deleted:** nothing. No existing public interface, test, or evaluator
was removed; the adapter rewrite changed *internal* field mapping only.

**Project 1 components reused:** none of `production-rag`'s code is
imported or copied — the integration is purely an HTTP adapter, exactly
as Phase 38 required ("never duplicate the RAG pipeline"). What *was*
reused is Project 1's real **interface** (its documented request/response
JSON shape, read directly from its source) and its **auth convention**
(`Authorization: Bearer` or `X-API-Key`, confirmed by reading
`app/auth.py`) — the adapter conforms to that interface rather than
inventing its own.

**Interface changes:** `ProductionRagAdapter.__init__` changed from
`(base_url, api_key, response_field_map=...)` to
`(base_url, api_key, top_k=5, rerank=True, ...)` — the old
`response_field_map` indirection existed only to hedge against a guessed
contract being wrong, and was removed once the real contract was known.
Any code outside this adapter module constructing it with the old
keyword argument would break; nothing else in this project did.

## 4. Evaluator inventory

| Evaluator | Type | Input | Output | Scoring | Limitations | Tests |
|---|---|---|---|---|---|---|
| `ExactMatchEvaluator` | deterministic | output, expected | 0/1 | string equality (case-sensitivity configurable) | brittle to phrasing | `test_deterministic_evaluators.py` |
| `ContainsEvaluator` | deterministic | output, expected substring(s) | 0/1 | substring presence | no semantic match | same |
| `RegexEvaluator` | deterministic | output, pattern | 0/1 | regex match, with a ReDoS-safety cap on pattern complexity (Day 17) | pattern must be hand-written | same |
| `JSONSchemaEvaluator` | deterministic | output, JSON schema | 0/1 | schema validation | only checks shape, not content correctness | same |
| `LengthEvaluator` | deterministic | output | 0/1 | length within bounds | no semantic signal | same |
| `RequiredFieldsEvaluator` | deterministic | structured output | 0/1 | required keys present | no type/value checking | same |
| `SemanticSimilarityEvaluator` | embedding | output, expected | continuous 0–1 | cosine similarity vs. threshold | only as good as the embedding model; `FakeEmbeddingProvider`'s trigram-hash fake is **not** a real similarity signal (see the Phase 39 flaky-test note below) | `test_semantic_similarity_evaluator.py` |
| `CorrectnessJudgeEvaluator` | LLM-judge | output, expected | 0/1 + rationale | LLM asked to verdict correctness | nondeterministic in production (mocked in all non-live tests); prompt-injection limits documented in `llm_judge.py` | `test_llm_judge_evaluators.py` |
| `FaithfulnessJudgeEvaluator` | LLM-judge | output, retrieved context | 0/1 + rationale | LLM asked whether claims are supported by context | same as above; this is the evaluator that catches hallucination even when correctness/citation checks pass (Phase 39 scenarios #4/#8) | same |
| `RubricJudgeEvaluator` | LLM-judge | output, custom rubric | continuous or categorical | LLM scores against an arbitrary rubric | most flexible, least deterministic | same |
| `HitAtKEvaluator` | RAG/deterministic | retrieved docs, expected relevant doc(s) | 0/1 | expected doc present in top-K | needs a correct "expected relevant" label | `test_retrieval_evaluators.py` |
| `RecallAtKEvaluator` | RAG/deterministic | same | continuous 0–1 | fraction of relevant docs retrieved | same | same |
| `MRREvaluator` | RAG/deterministic | same | continuous 0–1 | reciprocal rank of first relevant hit | same | same |
| `ContextRelevanceEvaluator` | RAG/embedding | retrieved context, question | continuous | embedding similarity of context to question | embedding-quality-dependent | `test_retrieval_evaluators.py` |
| `FaithfulnessEvaluator` | RAG/deterministic | output, context | continuous | lexical overlap between answer and context | cheap proxy for faithfulness, not a substitute for the judge version — this is why both exist | `test_groundedness_evaluators.py` |
| `CitationPrecisionEvaluator` | RAG/deterministic | citations, expected citations | 0–1 (vacuously 1.0 if none claimed) | fraction of cited sources that are valid/expected, configurable `match_fields` | vacuous-pass case must be read correctly (Phase 39 scenario #6 explicitly tests this) | `test_citation_evaluators.py` |
| `CitationRecallEvaluator` | RAG/deterministic | same | 0–1 | fraction of expected citations actually cited | same | same |

**Deterministic vs. embedding vs. LLM-judge**, explicitly: deterministic
evaluators are free, instant, and 100% reproducible but blind to
semantics; embedding evaluators add semantic matching at the cost of
needing a real embedding provider (or an explicitly-labeled fake in
tests) and being only as good as that embedding space; LLM-judge
evaluators catch what the first two structurally cannot (hallucination,
nuanced correctness, open-ended rubrics) at the cost of being
nondeterministic and requiring a real LLM call in production (always
mocked via `FakeJsonLLM` in this project's own test suite).

## 5. Execution instructions

Only commands verified to work against the actual repository as it
stands right now. No Docker commands are listed — Docker was explicitly
descoped (§10).

```bash
# Install -- verified via a fresh venv + extraction smoke test (this phase):
pip install -r requirements-dev.txt -r requirements-openai.txt -r requirements-httpx.txt
# `requirements-dev.txt` alone is enough to run the suite, but yields
# `345 passed, 4 skipped`, not `348 passed, 4 skipped` -- 3 UNIT tests
# (test_openai_embedding_provider.py, test_openai_json_llm.py) test
# construction/wrapping only, no network call, but still
# `pytest.importorskip("openai")` and silently skip without the package.
# `requirements-httpx.txt` is only needed for the real HTTP transport
# (FakeHttpClient covers every adapter test without it).

# Validate a dataset file
python -m eval_harness.cli validate-dataset datasets/sample_eval_v1.json

# Run the full worked demonstration (comparison, regression, quality gate,
# efficiency, storage, observability -- all together)
python -m eval_harness.cli demo
# equivalently: python scripts/demo_end_to_end.py

# Compare two persisted runs for regressions
python -m eval_harness.cli compare path/to/baseline_run.json path/to/candidate_run.json

# Evaluate a quality gate against a persisted run (optionally vs. a baseline)
python -m eval_harness.cli quality-gate path/to/candidate_run.json \
    --baseline path/to/baseline_run.json \
    --min-pass-rate 0.9 --min-average-score 0.85

# Full unit + integration test suite (no network, no API key needed)
python3 -m pytest -q

# Live-provider tests (optional; real OpenAI calls; needs OPENAI_API_KEY)
OPENAI_API_KEY=sk-... python3 -m pytest -q tests/live

# Local pre-push check (same steps CI runs for free)
bash scripts/pre_push_check.sh
```

There is no `eval-harness run <system>` command and no HTTP server to
start — see §10 for why.

## 6. Test results (real counts, this run)

```
python3 -m pytest -q
348 passed, 4 skipped
```

- **Executed: 348**, across 33 unit-test files + 2 integration-test
  files (`test_demo_end_to_end.py`, `test_failure_injection.py`).
- **Skipped: 4** — the full contents of `tests/live/` (two tests each in
  `test_openai_embedding.py` and `test_openai_llm_judge.py`), which
  self-skip via `pytest.mark.skipif` with no `OPENAI_API_KEY` set. These
  were **not run** in this session — no real/paid API call was made
  anywhere in producing this report.
- **Blocked/not-run:** Docker-based verification was not performed (no
  Docker image exists — §10); no live-provider run was performed (no API
  key configured in this environment).
- No coverage percentage or build-time figure is claimed in this report
  beyond what Day 18's audit already recorded (`docs/day18.md`:
  `eval_harness/*` at 100% coverage excluding the two OpenAI live-call
  provider modules) — coverage was not re-run as part of Phases 38–41,
  so that number is not re-asserted as current without re-measuring it.

## 7. Demonstration results

From `python -m eval_harness.cli demo`, run immediately before writing
this report (real values, not illustrative):

```
Dataset: TechNova Corp policy handbook -- sample synthetic eval set (17 test cases)

Comparison:
  production-rag-baseline   pass_rate=0.8824  avg_score=0.9088
  production-rag-candidate  pass_rate=0.9412  avg_score=0.9902
  Overall winner: production-rag-candidate (wins on both pass rate and
  average score, and on both per-evaluator comparisons: semantic_similarity
  and citation_precision)

Regression check, baseline -> candidate:
  pass_rate_delta=+0.0588  average_score_delta=+0.0814
  newly passing: ['qa-009', 'qa-013']
  regressed:     ['qa-019']
  Regression check passed: False
    - 1 test case(s) regressed (allowed: 0): qa-019

Quality gate on the candidate: GATE FAILED
    - regression check failed: 1 test case(s) regressed (allowed: 0): qa-019

Cost efficiency:
  production-rag-candidate: cost_per_passing_test=0.00035612
  production-rag-baseline:  cost_per_passing_test=0.0003698
```

This is the required "at least one gate PASS and one gate FAIL"
demonstration in miniature: the *static* thresholds (pass rate, average
score) would PASS for the candidate on their own — it's strictly better
by every aggregate number — but the *regression* check correctly FAILS
the gate anyway, because one previously-passing test case (`qa-019`)
broke. This is the demo's entire point (`docs/day20.md`): a candidate
that's better on average can still ship a real regression, and a quality
gate that only looks at aggregates would miss it.

## 8. Cost and performance

- **Cost calculation:** `cost.py` computes USD cost from token counts
  using per-model price tables; `TokenUsage.input_tokens`/
  `output_tokens` are the inputs. For the Project 1 adapter specifically,
  input tokens are the sum of `embedding_tokens` + every `*_prompt_tokens`
  field + `memory_embedding_tokens` from production-rag's real
  `CostBreakdown`; output tokens are the sum of every
  `*_completion_tokens` field (embedding calls have no completion side —
  the same convention OpenAI's own usage objects use).
- **Avoiding unnecessary calls:** every unit and integration test in this
  project uses `FakeEmbeddingProvider`/`FakeJsonLLM`/`FakeHttpClient`/
  `MockSystem` — zero real embedding or LLM calls are made by the default
  test suite or by the demo script. Real calls are opt-in and isolated
  to `tests/live/`, which self-skips without a configured API key.
- **Latency measurement:** `SystemOutput.latency_ms` is the adapter's own
  wall-clock measurement of the full round trip to the system under
  test — deliberately **not** a sum of production-rag's internal
  `timings_ms` breakdown, because that breakdown excludes network time
  and anything outside production-rag's own process; summing it would
  systematically understate what a caller actually experiences. The full
  internal per-stage breakdown is preserved verbatim in
  `metadata["production_rag_timings_ms"]` for anyone who wants it.
- **Two documented gaps, carried forward from Phase 39 (not fixed in
  this phase — see `docs/DEFINITION_OF_DONE.md` items D3b/E7 for the
  full reasoning on why fixing them was judged out of scope here):**
  1. **No latency-based quality gate exists.** Latency is fully
     observable at the run level (`average_system_latency_ms`/
     `max_system_latency_ms`) but nothing in `QualityGatePolicy` can
     currently fail a run over it — a system running 40x slower still
     passes the gate on pass rate/average score alone.
  2. **Run-level cost does not distinguish "not reported" from
     "genuinely free."** `compute_efficiency()`'s *per-passing-test* cost
     already makes this distinction correctly (`None`, not `0.0`, when
     nothing passed), but `EvaluationRun.estimated_cost`/`total_tokens`
     themselves are summed directly with no `reported: bool` escape
     hatch — a system that never reports cost ends up indistinguishable
     from one that costs exactly `$0.0`.

## 9. Security and reliability

**Safeguards in place:**
- Untrusted dataset files are bounded on file size (50MB), record count
  (100k), and per-record size (1MB) before being trusted —
  `security_limits.py`, enforced in both `loader.py` and `dataset_io.py`.
- `RunStore`'s on-disk paths are built from caller-supplied fields
  (`dataset_id`/`system_name`/`run_id`); every one of the three is
  checked by `_safe_path_component` against path separators and `..`
  segments before being joined into a filesystem path — a classic
  path-traversal class of bug, closed.
- `RegexEvaluator` caps pattern complexity against ReDoS (Day 17).
- API keys are passed as HTTP headers and never logged — verified by
  reading every `logging.*` call site in `observability.py`; structured
  log records carry `test_id`/evaluator name/score, never request
  headers or bodies.
- Outbound HTTP calls to a system under test carry an explicit timeout
  (`timeout_s`, default 30s, always forwarded to `httpx.AsyncClient`).
- Data models used across evaluation boundaries (`TestCase`,
  `EvaluatorResult`, `SystemOutput`, `QualityGatePolicy`/
  `QualityGateResult`) are Pydantic-frozen, preventing accidental
  in-place mutation of a result after it's been produced.
- Prompt-injection limits for LLM-judge evaluators are documented
  in-module (`llm_judge.py`) rather than claimed solved — a judge
  evaluator's own prompt is, by nature, exposed to whatever text the
  system under test produced.

**Known limitations (do not claim beyond what's true):**
- **No automatic retry on transient HTTP failure.** A single transport
  failure (`HttpError`) propagates immediately as a `DEPENDENCY_FAILURE`;
  there is no backoff/retry logic anywhere in `providers/http.py` or the
  adapter. A flaky network blip currently looks identical to a real
  outage.
- Live-provider code paths (`providers/openai_embedding.py`/
  `openai_llm.py`'s actual API-call branches) are excluded from the
  100%-coverage figure recorded in `docs/day18.md` — they're exercised
  only by `tests/live/`, which was not run in this session.
- **This application is not claimed to be completely secure or
  production-proven.** The security review (Day 17, this phase) covers
  the specific, concrete risks found by direct inspection — untrusted
  file input, path traversal, ReDoS, secret logging, prompt injection
  awareness — not a formal security audit, penetration test, or
  dependency vulnerability scan. No claim is made about risks outside
  that list.

## 10. Production-readiness assessment

| Area | Status | Evidence / next step |
|---|---|---|
| Dataset management | **READY** | Versioned, hashed, resource-bounded, tested (A1–A7 in `docs/DEFINITION_OF_DONE.md`) |
| SUT abstraction + Project 1 adapter | **READY** | Generic Protocol, real adapter built against the actual source, fully offline-tested, e2e-verified against real evaluators (B1–B8) |
| Evaluation engine | **READY** | All three evaluator families implemented, each independently tested, failure modes distinguished (C1–C9) |
| Runs/results/metrics | **NEEDS IMPROVEMENT** | Core aggregation solid, but run-level cost/token "not reported vs. zero" is unresolved (D3b) — low risk unless a system silently never reports cost |
| Regression & quality gates | **NEEDS IMPROVEMENT** | Regression detection and static/forbidden-category gates are solid and demonstrated; no latency-based gate exists yet (E7) |
| API and CLI | **CLI: READY. HTTP API: MISSING. Docker: MISSING.** | CLI covers validate/compare/gate/demo with correct exit codes, tested (F1–F3). HTTP API and Docker were explicitly descoped by direct user decision before building (F4/F6) — not attempted, not faked. A generic `run <arbitrary system>` command is also MISSING by design (F5) — it would need a config language this project doesn't have. |
| Persistence/security/reliability | **NEEDS IMPROVEMENT** | Path-traversal hardening, secret handling, and timeouts are in place and tested; no HTTP retry logic exists (G6) — a real deployment behind this harness should expect to handle transient-failure retries itself, or this should be built before relying on the harness unattended against a flaky network |
| Testing/operations | **READY** | 348 passing, 0 failing, fully offline by default, live tests cleanly separated and self-skipping, CI wired via `.github/workflows/ci.yml` (Day 21) |

**Overall:** the core evaluation engine (dataset → SUT → evaluators →
run → regression → gate) is production-ready for its stated scope — an
offline-verifiable, pluggable eval pipeline with a real, source-verified
adapter for one concrete system. The CLI is a thin, tested, honest
wrapper over that engine, not a separate product. What is **not** ready
is anything requiring network-level resilience (no retry) or a
service/deployment surface (no HTTP API, no Docker) — both are
documented gaps, not silent ones, and both were the result of explicit,
asked-and-answered scope decisions or defects found and recorded rather
than hidden.

## 11. Final learning checklist + interview questions

**Learning checklist (what this project actually demonstrates, grounded
in its own code):**
- [x] Designing a generic abstraction (`SystemUnderTest` as a `Protocol`)
      that a real, independently-developed system (Project 1) can satisfy
      without either side depending on the other's internals.
- [x] Correcting a previously-guessed external API contract against the
      real source, and verifying the correction is behavior-preserving
      (not just representation-preserving) by re-running an existing
      demonstration before and after and confirming identical results.
- [x] Distinguishing three genuinely different "why did this fail"
      categories (dependency / evaluation / threshold-miss) and proving,
      with a deterministic test, that they never get confused with each
      other even within a single run.
- [x] Building a regression-detection system that looks at per-test-case
      deltas, not just aggregate averages — and proving with a real
      example that a system can win on every aggregate metric and still
      fail a well-designed quality gate.
- [x] Recognizing and documenting real gaps found while writing
      verification tests (no latency gate; cost-not-reported ≈ free)
      instead of quietly fixing them out of scope or ignoring them.
- [x] Making an explicit, asked scope decision (CLI vs. API vs. Docker)
      rather than silently guessing or silently over-building.

**Practical interview questions, based on the actual code:**
1. `ProductionRagAdapter.__init__` used to take a `response_field_map`
   parameter and no longer does. Why was that removed, and what does
   its removal tell you about the cost of designing around an uncertain
   external contract?
2. `QualityGatePolicy`'s static thresholds default to `None` (always
   pass), but `RegressionPolicy`'s `max_new_regressions` defaults to `0`
   (zero tolerance). Why the asymmetry — what's the actual argument for
   treating these two defaults differently?
3. `failure_taxonomy.py` has exactly three categories, no more. Why not
   a fourth category per evaluator family (e.g. "retrieval_miss" vs.
   "faithfulness_miss")? Where does that information actually live
   instead?
4. `compute_efficiency()`'s cost-per-passing-test returns `None` when
   zero tests passed, not `0.0`. `EvaluationRun.estimated_cost` doesn't
   make this same distinction. Walk through a concrete scenario where
   that inconsistency would produce a misleading report.
5. `storage.py`'s `_safe_path_component` rejects `/`, `\`, and `..`
   wholesale rather than attempting to sanitize them. Why reject instead
   of sanitize, given that `dataset_id` and `system_name` come from a
   caller this harness doesn't fully control?
6. The adapter's `SystemOutput.latency_ms` is the adapter's own
   wall-clock measurement, not a sum of production-rag's internal
   `timings_ms`. What specific thing would that sum miss, and why does
   it matter for anyone using this harness to set an SLA?
7. `CitationPrecisionEvaluator` is documented as "vacuously passes" when
   no citations are claimed at all. Why is that the right default
   rather than a failure, and what's the one test in
   `test_failure_injection.py` that exists specifically because this is
   easy to get backwards?
8. If you had to add retry logic to `providers/http.py`'s
   `HttpxHttpClient`, what decision would you have to make about how
   `FakeHttpClient`'s test double represents "fails twice then
   succeeds" — and would the existing `queue` mechanism already support
   it, or would it need to change?
