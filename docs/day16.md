# Day 16 — Observability

Phase 23. Every `PerTestResult` already carries `latency_ms`/
`token_usage_total`/`estimated_cost` (Day 5/8/13), but two things were
still missing: seeing any of that **while** a run is happening rather than
only after it finishes, and a ready-made way to ask "which test cases were
actually slow or expensive" instead of eyeballing a results list by hand.

## `eval_harness/observability.py`

- **`RunObserver`** — a base class with six no-op hooks
  (`on_test_case_start`, `on_system_error`, `on_system_output`,
  `on_evaluator_error`, `on_evaluator_result`, `on_run_complete`).
  Subclass it and override only what you need — the same design
  `logging.Handler` and `unittest.TestResult` use, for the same reason:
  most callers want one or two events, not all six.

- **A generalized reuse of Project 1's `tracing.py` pattern** — not the
  contextvar/span machinery itself (that solves "one trace id per
  concurrent HTTP request," a problem this single-process, sequential-by-
  design harness doesn't have — Day 12 already explained why runs go one
  at a time), but the *shape* of the idea: a lightweight hook interface
  the pipeline calls into at fixed points, so watching a run never requires
  modifying the pipeline itself.

- **`StructuredLogObserver`** — a ready-made `RunObserver` that emits one
  stdlib `logging` record per event, fields under `extra` rather than
  interpolated into the message — meant to pair with a JSON log formatter
  downstream, not to be read raw in a terminal.

- **`slowest_results(run, n=5)` / `most_expensive_results(run, n=5)`** —
  post-hoc diagnostics over an already-completed run: the `n`
  `PerTestResult` rows with the highest latency / cost, descending. Rows
  with no recorded measurement are excluded, not treated as `0.0` — a
  missing measurement isn't a fast or free one.

## `runner.py` change

`run_evaluation` takes an optional `observer` (default: a plain, silent
`RunObserver()`) and calls it at six points through the loop — test case
start, system success/failure, each evaluator's success/failure, and run
completion. **The observer is a pure side channel**: verified directly
with a test asserting that an observed run and an unobserved run of the
same dataset/system/evaluators produce identical `pass_rate`/`total_tests`
— watching a run never changes what it returns.

## Tests

11 new tests (`test_observability.py`: base hooks never raise, structured
log fields, slowest/most-expensive ranking and their "exclude missing
measurements" rule) plus 2 new runner tests (hooks fire at the right
points with the right arguments including the error-path hooks, and the
observer never affects the result). **249 passed, 4 skipped.**

## Next

Day 17 — security review: a pass over every place this harness accepts
external input (loaded datasets, evaluator configuration, LLM judge
prompts) for injection and resource-exhaustion risks, before the Project 1
integration work (phases 33+) starts feeding it real, less-trusted data.
