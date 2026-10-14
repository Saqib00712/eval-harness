# Phase 40 — Final Definition of Done

Every applicable item below is marked from direct evidence (code read,
tests run, or both) — never from "the file exists so it must work."
Legend: **PASS** / **FAIL** / **BLOCKED** / **N/A**.

Baseline for all test-count evidence in this document:
`python3 -m pytest -q` → **348 passed, 4 skipped** (the 4 skips are the
two live OpenAI tests, each parametrized/split across 2 cases, skipped
automatically with no `OPENAI_API_KEY` set — never run against a paid
API in this project).

---

## A. Dataset management

| # | Item | Verdict | Evidence |
|---|---|---|---|
| A1 | A versioned, named dataset format exists (not a bare list) | PASS | `dataset.py::Dataset` — `dataset_id`, `version`, `name`, `test_cases`, computed `test_count`/`content_hash` |
| A2 | A lighter "bare array of test cases" entry point exists for ad-hoc use | PASS | `loader.py::load_test_cases` |
| A3 | Dataset content is content-addressed (detects silent edits) | PASS | `hashing.py` + `Dataset.content_hash` (computed field, recomputed on load, never trusted from the file — `dataset_io.py` docstring and `test_dataset_io.py`) |
| A4 | Loading untrusted dataset files enforces resource ceilings (size, record count, per-record size) | PASS | `security_limits.py` (50MB file / 100k records / 1MB per record, all overridable); enforced in both `loader.py` and `dataset_io.py`; `tests/unit/test_security_limits.py` (7 tests) |
| A5 | Malformed/invalid dataset files fail with a clear error, not a crash or silent partial load | PASS | `DatasetError`/`load_dataset` — `tests/unit/test_dataset_io.py` (8 tests), `test_cli.py::test_validate_dataset_malformed_json_exits_one` |
| A6 | A CLI or script exists to validate a dataset file without running an evaluation | PASS | `eval_harness/cli.py validate-dataset` |
| A7 | A real sample dataset ships with the project | PASS | `datasets/sample_eval_v1.json` (20 test cases) — `tests/unit/test_sample_dataset.py` |

## B. System Under Test (SUT) abstraction

| # | Item | Verdict | Evidence |
|---|---|---|---|
| B1 | A generic, system-agnostic interface exists (not tied to any one app) | PASS | `system.py::SystemUnderTest` — a 3-method `Protocol` (`name`, `version`, async `run`) |
| B2 | At least one reference/mock implementation exists for testing the harness itself | PASS | `mock_system.py::MockSystem` |
| B3 | A real external system (Project 1) is integrated via an adapter, not by modifying the harness's core | PASS | `adapters/production_rag.py::ProductionRagAdapter` — implements the same `Protocol`; zero changes to `runner.py` or any evaluator were needed |
| B4 | The adapter is built against the real external system's actual interface, not a guess | PASS | `app/models/ask.py`/`app/routers/ask.py`/`app/auth.py` read directly from `/home/claude/production-rag`; corrections from the original guess documented in `docs/day22.md` |
| B5 | The adapter is independently unit-testable without the real external system running | PASS | `FakeHttpClient` (`providers/http.py`); `tests/unit/test_production_rag_adapter.py` (28 tests), zero network calls |
| B6 | The adapter's output correctly feeds the harness's real evaluators (not just unit-tested in isolation) | PASS | `tests/unit/test_production_rag_adapter_e2e.py` (4 tests) — hit@k, citation precision/recall, faithfulness judge, all through the real adapter mapping |
| B7 | Malformed/partial data from the external system (e.g. malformed citations) is handled without crashing | PASS | `test_malformed_citation_entries_are_dropped_not_crashed_on`, `test_citation_missing_expected_keys_is_kept_as_is` |
| B8 | External-system failures (non-2xx, transport errors) are surfaced distinguishably, not swallowed | PASS | `ProductionRagAdapterError` (non-2xx) vs `HttpError` propagated untouched (transport failure) — both map to `DEPENDENCY_FAILURE` in the runner, verified in `test_failure_injection.py::test_11`/`test_production_rag_http_error_status_is_also_a_dependency_failure` |

## C. Evaluation engine

| # | Item | Verdict | Evidence |
|---|---|---|---|
| C1 | A common `Evaluator` interface/result model exists | PASS | `evaluator.py::Evaluator` (Protocol), `EvaluatorResult` |
| C2 | Deterministic evaluators exist (exact match, contains, regex, schema, length, required fields) | PASS | `evaluators/deterministic.py` — 6 evaluator classes, 31 tests |
| C3 | Embedding-based evaluators exist, distinct from deterministic and LLM-judge | PASS | `evaluators/semantic.py::SemanticSimilarityEvaluator`; 7 tests |
| C4 | LLM-judge evaluators exist for correctness, faithfulness, and open-ended rubrics | PASS | `evaluators/llm_judge.py` — `CorrectnessJudgeEvaluator`, `FaithfulnessJudgeEvaluator`, `RubricJudgeEvaluator`; 13 tests, all against `FakeJsonLLM` |
| C5 | RAG-specific retrieval evaluators exist (Hit@K, Recall@K, MRR) | PASS | `evaluators/rag.py::HitAtKEvaluator`/`RecallAtKEvaluator`/`MRREvaluator`; `retrieval_metrics.py` underlying math, 10 tests |
| C6 | RAG-specific citation evaluators exist (precision, recall — "valid + supports the claim") | PASS | `evaluators/rag.py::CitationPrecisionEvaluator`/`CitationRecallEvaluator`; 7 tests in `test_citation_evaluators.py` |
| C7 | Groundedness/faithfulness is evaluated both deterministically (context overlap) and via judge | PASS | `evaluators/rag.py::FaithfulnessEvaluator` (deterministic) + `llm_judge.py::FaithfulnessJudgeEvaluator`; `test_groundedness_evaluators.py` (7 tests); Phase 39 scenario #4/#8 prove correctness-passing isn't enough on its own |
| C8 | Evaluator failures (the check itself crashing) are distinguishable from the system under test failing | PASS | `FailureCategory.EVALUATION_FAILURE` vs `.DEPENDENCY_FAILURE`; `test_failure_injection.py::test_12`/`test_13` explicitly assert a failed evaluator is never reported `passed=True` |
| C9 | No duplicate metric implementations without justification | PASS | `docs/day22.md` — citations preserved verbatim and matched via existing `match_fields=[...]`, no new citation-matching logic added for the adapter |

## D. Runs, results, and metrics

| # | Item | Verdict | Evidence |
|---|---|---|---|
| D1 | A run aggregates per-test results into pass rate, average score | PASS | `run_models.py::EvaluationRun` (`pass_rate`, `average_score`), `runner.py::run_evaluation` |
| D2 | Latency is measured per test case and aggregated at the run level | PASS | `SystemOutput.latency_ms`; `EvaluationRun.average_system_latency_ms`/`max_system_latency_ms` (Day 13) |
| D3 | Token usage is measured and aggregated, with "not reported" distinguished from "zero" at the per-test level | PASS | `system_output.py::TokenUsage`; adapter returns `None` (not a zero `TokenUsage`) when `cost` is absent entirely — `test_no_token_usage_when_cost_breakdown_entirely_absent` |
| D3b | The SAME "not reported vs. zero" distinction holds at the **run-aggregate** level for cost/tokens | **FAIL — documented, not fixed** | `EvaluationRun.estimated_cost`/`total_tokens` sum directly in `runner.py` with no `reported: bool` escape hatch; a system that never reports cost ends up `== 0.0`, identical to genuinely free. Pinned by `test_failure_injection.py::test_15b`. Recorded as a known gap in `docs/day23.md`, not silently left undiscovered. Fixing it is a schema change (`Optional[float]` or a `cost_reported` flag) judged out of this phase's "verify, then fix defects found" scope — it's new design work on `EvaluationRun`, not a defect introduced by Phases 38–40. |
| D4 | Cost-per-passing-test correctly distinguishes undefined from zero | PASS | `efficiency.py::compute_efficiency` — `None` when zero tests passed, not `0.0`; `test_efficiency.py` (8 tests) |
| D5 | Failure categorization is a closed, documented taxonomy applied to every failed result | PASS | `failure_taxonomy.py::FailureCategory` (`DEPENDENCY_FAILURE`/`EVALUATION_FAILURE`/`THRESHOLD_MISS`); `test_failure_injection.py::test_13` proves all three appear distinctly in one run |
| D6 | Runs persist to disk and can be reloaded in a later process | PASS | `storage.py::save_run`/`load_run`/`RunStore`; `test_storage.py` (16 tests) |
| D7 | Persisted run files resist path-traversal via attacker-controlled fields (`dataset_id`, `system_name`, `run_id`) | PASS | `storage.py::_safe_path_component` rejects `/`, `\`, `..` before path construction; covered in `test_storage.py` |

## E. Regression analysis & quality gates

| # | Item | Verdict | Evidence |
|---|---|---|---|
| E1 | Two runs of the same dataset can be compared for regressions/improvements | PASS | `regression.py::compare_runs`/`RegressionReport`; `test_regression.py` (9 tests) |
| E2 | Comparing runs of two DIFFERENT datasets is rejected, not silently computed | PASS | `compare_runs` raises `ValueError`; now also handled cleanly by the CLI (`test_compare_mismatched_datasets_reports_error`) rather than a raw traceback |
| E3 | A regression policy defaults to strict (zero tolerance) | PASS | `RegressionPolicy` docstring + default `max_new_regressions=0` |
| E4 | A configurable quality gate turns a run into a pass/fail verdict for CI | PASS | `quality_gate.py::QualityGatePolicy`/`evaluate_quality_gate`/`exit_code_for`; `test_quality_gate.py` (12 tests) |
| E5 | A quality gate can incorporate a regression check, not just static thresholds | PASS | `evaluate_quality_gate(..., regression_report=...)`; `test_quality_gate_with_regressing_baseline_fails_even_if_static_thresholds_pass` |
| E6 | A quality gate can fail on a forbidden failure category appearing at all (e.g. any dependency failure blocks CI) | PASS | `QualityGatePolicy.forbid_failure_categories` |
| E7 | A latency-based quality gate exists | **FAIL — documented, not built** | No `min_*_latency` field on `QualityGatePolicy`. Latency is fully observable (D2) but nothing can currently fail a run over it. Pinned by `test_failure_injection.py::test_14`. Recorded in `docs/day23.md` as a known gap, same reasoning as D3b — a policy schema addition, out of this phase's fix-what-was-found scope. |
| E8 | A real baseline-vs-candidate demonstration exists with at least one regression AND one improvement, using mocked (not live) judges | PASS | `scripts/demo_end_to_end.py` / `docs/day20.md` — candidate wins on pass rate, average score, and both per-evaluator rates, but still fails the gate on one regressed test case; `FakeJsonLLM` throughout |
| E9 | Live-provider tests are kept separate from the deterministic regression/gate test suite | PASS | `tests/live/` — `pytest.mark.skipif` with no `OPENAI_API_KEY`; never imported by or mixed into `tests/unit`/`tests/integration` |

## F. API and CLI

**Scope decision (explicit, asked and answered before building — see
`docs/day24.md`): CLI only. No HTTP API, no Docker image.** This project
has been a pure Python library + scripts for 21 prior days; building a
full HTTP service was judged substantial new scope, not a gap in what was
already planned, and the user chose "CLI only (recommended)" when asked
directly. The items below are marked against that explicit decision, not
glossed over as accidentally incomplete.

| # | Item | Verdict | Evidence |
|---|---|---|---|
| F1 | A CLI exists for the core operations (validate, run comparisons, gate) | PASS | `eval_harness/cli.py` — `validate-dataset`, `quality-gate`, `compare`, `demo`; `test_cli.py` (12 tests) |
| F2 | CLI commands produce correct process exit codes for CI use (0 pass / 1 fail) | PASS | `exit_code_for`; every CLI test asserts the real exit code |
| F3 | CLI errors (missing file, malformed input, incompatible comparison) are reported cleanly, not as raw tracebacks | PASS | `ERROR`/`INVALID` on stderr with exit 1, for every failure path tested |
| F4 | An HTTP API exists for triggering evaluations / fetching results | **MISSING (not built)** | Explicit scope decision — no HTTP framework, no routes, no server anywhere in this project. Not a defect; a deliberate, user-confirmed scope boundary. |
| F5 | A generic `run <arbitrary system>` CLI/API command exists | **MISSING (not built)** | Would require a config language for instantiating an arbitrary `SystemUnderTest` (adapter type, evaluators, providers, keys) that doesn't exist in this project — judged real new design work, not a thin wrapper; see `docs/day24.md`'s "What's deliberately NOT here." |
| F6 | A Docker image exists for deployment | **MISSING (not built)** | Same scope decision as F4 — no `Dockerfile`/`docker-compose` anywhere in the repo. |

## G. Persistence, security, and reliability

| # | Item | Verdict | Evidence |
|---|---|---|---|
| G1 | Runs and human labels persist to disk in a documented, versioned format | PASS | `storage.py` (D6); `calibration.py::HumanLabel` + `save_labels`/`load_labels` |
| G2 | Path-traversal via user/system-controlled identifiers is prevented | PASS | G/D7 above (`_safe_path_component`) |
| G3 | Untrusted file input (datasets) has enforced resource ceilings | PASS | A4 above |
| G4 | API keys/secrets are never logged | PASS | Grepped `observability.py`/every `logging.*` call site — no call logs `api_key`, headers, or request bodies; `StructuredLogObserver` logs only `test_id`/evaluator name/score, never adapter request contents |
| G5 | Outbound HTTP calls (to the SUT) have an explicit timeout | PASS | `HttpClient.post(..., timeout_s: float = 30.0)`; `HttpxHttpClient` passes it straight to `httpx.AsyncClient(timeout=...)` |
| G6 | Outbound HTTP calls have automatic retry on transient failure | **FAIL — not implemented, documented here** | No retry logic anywhere in `providers/http.py` or `adapters/production_rag.py` — a transport failure (`HttpError`) propagates once, immediately, as a `DEPENDENCY_FAILURE`. This is a real, current limitation, not previously documented in this form; recording it now rather than leaving it implicit. A caller re-running a flaky single test case is the current workaround; there is no built-in backoff/retry. |
| G7 | No live/paid API call is ever made by the default (non-`live`) test suite | PASS | `348 passed, 4 skipped` — the 4 skips ARE the only tests that would call a real API, and they skip automatically without a key; every other test uses `FakeHttpClient`/`FakeJsonLLM`/`FakeEmbeddingProvider`/`MockSystem` |
| G8 | Frozen/immutable data models prevent accidental in-place mutation of results | PASS | `TestCase`, `EvaluatorResult`, `QualityGatePolicy`/`QualityGateResult`, `SystemOutput` all `ConfigDict(frozen=True)` or Pydantic frozen models — `test_system_output.py::test_system_output_is_frozen` |

## H. Testing and operations

| # | Item | Verdict | Evidence |
|---|---|---|---|
| H1 | Unit tests cover every evaluator, adapter, and core module in isolation | PASS | 33 unit test files, see per-file counts in `docs/day24.md`/this document's header |
| H2 | Integration tests prove components work together end to end | PASS | `tests/integration/test_demo_end_to_end.py` (5), `tests/integration/test_failure_injection.py` (17) |
| H3 | All 15 required synthetic failure/regression scenarios are implemented and pass | PASS | `docs/day23.md` scenario table; all 17 tests pass |
| H4 | Evaluator failures are never misreported as successful evaluations | PASS | `test_12_evaluator_failure_is_never_reported_as_a_successful_evaluation` |
| H5 | Infrastructure failures are distinguishable from quality failures within the same run | PASS | `test_13_dependency_failure_is_distinguishable_from_poor_quality_in_the_same_run` |
| H6 | Test suite runs fully offline (no network) except explicitly-separated live tests | PASS | G7 above |
| H7 | A reproducible way to run the full suite is documented | PASS | `README.md` (updated Day 24) + `docs/ENGINEERING_REPORT.md` execution instructions |
| H8 | Test counts in project documentation match actual, current `pytest` output (no fabricated numbers) | PASS | This document's header count (348/4) was produced by running the suite immediately before writing it down, not copied from memory |
| H9 | A fresh-clone/fresh-venv install, following only the README's own instructions, reproduces the documented test count | PASS (after a real fix) | A full smoke test (fresh venv, fresh extraction, `pip install -r requirements-dev.txt` only) initially reproduced `345 passed, 4 skipped`, not `348/4` — 3 unit tests (`test_openai_embedding_provider.py`, `test_openai_json_llm.py`) `pytest.importorskip("openai")` and silently skip without `requirements-openai.txt` installed, even though they make no real network call. This was a real, previously-undocumented setup gap caught by actually running the smoke test, not assumed clean. Fixed by updating `README.md`'s Setup section and `docs/ENGINEERING_REPORT.md`'s execution instructions to install all three requirements files; re-verified `348 passed, 4 skipped` after the fix. |

---

## Summary

- **PASS: 38**
- **FAIL (documented, not fixed in this phase): 3** — D3b (run-level cost "not reported" vs "zero"), E7 (no latency quality gate), G6 (no HTTP retry)
- **MISSING (explicit scope decision, not a defect): 3** — F4 (HTTP API), F5 (generic run command), F6 (Docker)
- **BLOCKED: 0**

No item above is marked PASS on the basis of code existing alone — each
cites either a specific test (by name, with its current pass/fail
status) or a specific code path read directly. The three FAIL items and
three MISSING items are carried forward verbatim into the production-
readiness section of `docs/ENGINEERING_REPORT.md`, not dropped here and
re-raised there as if newly discovered.
