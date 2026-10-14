# Day 9 — Failure taxonomy

Phase 13. Closes a real gap that's existed since Day 5, not a cosmetic
addition.

## The gap

`PerTestResult.failure_category` has existed since Day 5, but it was only
ever populated on the two EXCEPTION paths the runner already caught:
`"dependency_failure"` (the system under test raised) and
`"evaluation_failure"` (the evaluator raised). The far more common
failure — an evaluator runs to completion and simply scores below its
threshold — left the field `None` forever. A `None` category on a
`passed=False` row isn't a taxonomy; it's indistinguishable from "nothing
went wrong," which made "of our failures, how many are crashes vs. genuine
quality misses" an unanswerable question from the data alone.

## `eval_harness/failure_taxonomy.py`

- **`FailureCategory`** — the complete, closed set of three categories:
  `DEPENDENCY_FAILURE`, `EVALUATION_FAILURE`, `THRESHOLD_MISS`. `runner.py`
  now assigns one of these to *every* `passed=False` row, no exceptions
  (literally).
- **`summarize_failures(run)`** — groups every failed result by category,
  and within each category, by which evaluator produced it (`by_evaluator`)
  plus the affected `test_ids`.

**Deliberately not in this taxonomy**: a per-evaluator-family breakdown
baked into the category itself (e.g. `"retrieval_miss"` vs.
`"faithfulness_miss"`). That information already lives on
`PerTestResult.evaluator` — hardcoding an evaluator-name-to-category
mapping into the runner would silently mis-tag (or just ignore) any custom
evaluator someone plugs into the harness that this module has never heard
of, which cuts directly against Day 1's founding decision: this harness is
generic, not tied to any fixed evaluator set. `summarize_failures`'s
`by_evaluator` breakdown gives you that cross-cut without the runner
needing to know what any given evaluator measures.

## Runner change

One line: the success path now sets
`failure_category=None if result.passed else FailureCategory.THRESHOLD_MISS`
instead of never touching the field at all. The two exception paths are
unchanged in behavior — just sourced from `FailureCategory` constants
instead of bare string literals, so a typo in one of the three category
names is now a `NameError` at import time, not a silent mismatch discovered
later when a report's numbers don't add up.

## Tests

9 new tests (`test_failure_taxonomy.py`) plus one new runner test
confirming an ordinary threshold miss is now categorized instead of left
`None`. **183 passed, 4 skipped.**

## Next

Day 10 — regression testing / policy: comparing one `EvaluationRun` against
a previous baseline run of the *same* dataset+system to flag regressions,
not just report a standalone pass rate.
