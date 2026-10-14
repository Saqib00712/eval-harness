# Day 12 — Model/prompt comparison

Phase 18. Runs the same dataset through two or more systems (different
models, different prompt versions, a retrieval config change — anything
satisfying `SystemUnderTest`) and reports which one actually wins, **per
evaluator, not just overall**.

## `eval_harness/comparison.py`

- **`compare_systems(dataset, systems, evaluators, **run_kwargs) ->
  ComparisonReport`** — reuses Day 5's `run_evaluation` for each system
  (no scoring logic duplicated here) and Day 11's `per_evaluator_pass_rates`
  for the per-evaluator breakdown. Requires at least two systems and unique
  `system.name` values (raises `ValueError` otherwise — a comparison with
  one system, or two systems indistinguishable by name, isn't one).

- **Per-evaluator, not just overall, is the point.** Two systems can tie
  (or nearly tie) on overall pass rate while one is clearly better at
  retrieval and the other clearly better at faithfulness — a single
  "winner" number hides exactly the trade-off a real decision needs to
  see. `ComparisonReport.per_evaluator_winner` surfaces that directly
  (verified with a test where the overall-correctness winner and the
  length-constraint winner are different systems).

- **Ties are reported as ties, never broken arbitrarily.** If two or more
  systems share the best score (after rounding to 4 decimals, to avoid
  floating-point noise manufacturing a fake winner), the winner is `None`
  — not whichever system happened to come first in the input list, which
  would silently fabricate a "winner" that isn't actually one.

- **Sequential, not parallel, on purpose.** Each system's full evaluation
  runs one at a time rather than via `asyncio.gather`. Running N systems'
  worth of possibly-paid, possibly-rate-limited LLM calls all at once
  trades a predictable cost/rate profile for a faster wall clock most
  comparison workflows don't need — and a failing system's run is
  immediately attributable rather than requiring `gather`'s exception
  bookkeeping. A caller who genuinely wants concurrency can call
  `run_evaluation` directly per system themselves.

## Tests

6 new tests (`test_comparison.py`): a clear overall winner, a genuine tie
reported as `None`, per-evaluator winners diverging from the overall
winner, `runs` keyed correctly by system name, and the two guardrails
(fewer than two systems, duplicate system names). **210 passed, 4
skipped.**

## Next

Day 13 — cost & efficiency evaluation: turning the token/cost numbers
every `SystemOutput` and paid evaluator already reports into a real
cost-per-passing-test comparison across systems, not just a raw total.
