# Day 13 — Cost & efficiency evaluation

Phase 19/20. Turns the token/cost/latency numbers a run already reports
into real efficiency metrics — chief among them cost **per passing test**,
which `EvaluationRun` never surfaced directly before today.

## `eval_harness/efficiency.py`

- **`compute_efficiency(run) -> EfficiencyReport`** — `cost_per_test`,
  `tokens_per_test` (divide by `total_tests`), plus
  `cost_per_passing_test` / `tokens_per_passing_test` (divide by
  `passed_tests` instead). **This is the module's whole point**: a run's
  raw `estimated_cost` can't say whether a system is a good deal — a cheap
  system that fails half its tests can easily cost MORE per correct answer
  than an expensive system that rarely fails. Same "a raw total hides the
  number that matters" lesson Day 12 already applied to picking a
  comparison winner by pass rate instead of eyeballing totals.

- **Zero passing tests → `None`, never `0.0` or `inf`.** A run that passed
  nothing has an *undefined* cost-per-success, not a free one and not an
  infinite one. `None` keeps it distinguishable from "this system is free"
  at a glance and sorts predictably.

- **`rank_by_cost_per_passing_test(runs)`** — cheapest-per-success first;
  runs with `None` (zero passes) sort **last** regardless of how low their
  total cost was. A system that passed nothing achieved nothing — ranking
  it "cheapest" because the denominator is undefined would reward failure.
  Verified directly with a test: a near-free system that passes nothing
  ranks behind an expensive system that actually works.

## A real gap closed: system latency

`SystemOutput.latency_ms` has existed since Day 3 but was never
aggregated anywhere. `PerTestResult.latency_ms` is deliberately the
*evaluator's* own latency (Day 5/8) — copying the system's latency onto
every evaluator row scoring the same test case would be exactly the kind
of double-count Day 5's cost aggregation was already careful to avoid.
The fix follows that same once-per-test-case precedent: `runner.py` now
collects one latency sample per test case and `EvaluationRun` gains
`average_system_latency_ms` / `max_system_latency_ms` (both `None` when
nothing in the run ever reported a latency — e.g. `MockSystem` by
default).

## Tests

8 new tests (`test_efficiency.py`) plus 2 new runner tests confirming
latency is aggregated once per test case — not once per `(test_case,
evaluator)` row, which would have silently divided the real average by
the number of evaluators. **220 passed, 4 skipped.**

## Next

Day 14 — human calibration foundation: a way to record a human's own
judgment on a sample of results and compare it against what the automated
evaluators scored, the first real signal on whether a threshold (any
threshold set since Day 4) is actually calibrated to what a person would
say.
