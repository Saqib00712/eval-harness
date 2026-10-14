# Day 5 — The evaluation runner

## Phase 11 + 12: EvaluationRun + PerTestResult

`eval_harness/run_models.py`:

- **`PerTestResult`** — one row per `(test_case, evaluator)` pair, not one per test case. A test case scored by three evaluators produces three rows, each independently pass/fail. This is what makes a later per-evaluator breakdown ("faithfulness: 91% pass rate") possible without re-deriving it from something coarser.
- **`EvaluationRun`** — records `dataset_id` + `dataset_version` **and** `dataset_content_hash` (Day 2's computed hash). Version alone is only as trustworthy as whoever remembered to bump it; the hash is proof, not a claim, of exactly which dataset a run used.

## The runner

`eval_harness/runner.py` — `run_evaluation(dataset, system, evaluators, ...)`: runs every test case through the system, scores each `SystemOutput` with every evaluator, and aggregates into one `EvaluationRun`.

**A test case passes the run iff every evaluator that scored it passed** — one failing check fails the whole case, the usual CI-like meaning of "this test passed."

**Failure isolation** (this is most of what Day 5 actually had to get right):

- The **system** raising for one test case doesn't abort the run — every evaluator that would have scored it instead gets a result with `failure_category="dependency_failure"` (the system produced nothing, so it can't be said to have passed anything), and the **next test case still runs normally**.
- An **evaluator** raising doesn't abort the run, and doesn't stop *other* evaluators from scoring that same test case — only that one `(test_case, evaluator)` pair gets `failure_category="evaluation_failure"`.
- An **empty dataset** or **empty evaluator list** raises immediately, loudly — the same judgment call Day 4 made for a misconfigured evaluator.

**Cost/token aggregation bug avoided on purpose:** a system's generation cost (`SystemOutput.estimated_cost`/`token_usage`) is counted **once per test case**, not once per evaluator that scored it — attaching it to every `PerTestResult` row would have silently multiplied a single generation's real cost by however many evaluators ran. `PerTestResult.estimated_cost`/`token_usage_total` are reserved for the *evaluator's own* cost instead (relevant once Day 8's LLM judge exists; always `None` for today's free deterministic evaluators). Proven by `test_system_cost_is_counted_once_per_test_case_not_per_evaluator`.

## Tests

11 new tests, all offline, using `MockSystem` + Day 4's deterministic evaluators — including both failure-isolation paths and the cost-double-counting trap. **Total: 101 passed.**

## Next

Day 6 — Phase 7: semantic similarity evaluation (embedding-based), with a provider abstraction so unit tests never need a real OpenAI call.
