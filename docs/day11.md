# Day 11 — Quality gates

Phase 17. Turns an `EvaluationRun` (and, optionally, a Day 10
`RegressionReport`) into one hard pass/fail verdict a CI pipeline can act
on directly — `sys.exit(exit_code_for(result))` — instead of a human
reading a report and deciding by eye whether to block a deploy.

## `eval_harness/quality_gate.py`

- **`QualityGatePolicy`** — `min_pass_rate`, `min_average_score`,
  `min_evaluator_pass_rate` (per-evaluator minimums, e.g.
  `{"faithfulness": 0.9}`), `forbid_failure_categories` (e.g.
  `["dependency_failure"]` — a single crash fails the build no matter what
  the pass rate is).
- **`evaluate_quality_gate(run, policy=None, regression_report=None) ->
  QualityGateResult`** — checks every configured threshold, plus folds in
  a Day 10 `RegressionReport` if one is passed: a failing regression check
  always fails the gate.
- **`per_evaluator_pass_rates(run)`** — pass rate *per evaluator* across a
  run, distinct from `EvaluationRun.pass_rate` (which is per test case,
  requiring every evaluator to pass). A system can have a perfectly healthy
  overall pass rate while one specific evaluator — `faithfulness`, say —
  is quietly failing most of the time; this is how the gate catches that.
- **`exit_code_for(result) -> int`** — `0`/`1`, the standard CI convention,
  as a tiny named function rather than an inline `int(not result.passed)`
  scattered at every call site.

## Deliberate contrast with Day 10

`RegressionPolicy` (Day 10) defaults **strict** — zero tolerance, because
"don't let this get worse" is a sane universal default for any system.
`QualityGatePolicy` here defaults to the **opposite** — every threshold
starts unset, and an unconfigured gate always passes. There's no
universally correct minimum pass rate across every possible system this
harness evaluates (70% might be strong for one task and broken for
another) — inventing a default number would be a silent, wrong opinion.
The honest default is "nothing is gated until you say what matters for
*your* system," the same philosophy every evaluator's own
constructor-default-or-override pattern already follows at the test-case
level, applied here at the run level. `forbid_failure_categories` is
opt-in for the identical reason: a one-off transient crash during a dev
run shouldn't silently block every future CI run just because this module
assumed it should matter.

## Tests

13 new tests (`test_quality_gate.py`): unconfigured policy always passes,
each threshold violating independently, per-evaluator grouping and
violations (including a required evaluator that never ran), forbidden vs.
unlisted failure categories, a failing/passing `RegressionReport` folded
in, no regression report leaving `regression_passed=None`, and
`exit_code_for`'s 0/1 mapping. **204 passed, 4 skipped.**

## Next

Day 12 — model/prompt comparison: running the same dataset through two or
more systems (or the same system with different prompt versions) side by
side and reporting which one actually wins, per evaluator, not just
overall.
