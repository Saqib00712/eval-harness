# Day 10 — Regression testing & policy

Phase 15/16. A standalone pass rate answers "is this system good"; it can't
answer "did today's change make things worse than yesterday." That needs a
second run to compare against.

## `eval_harness/regression.py`

- **`compare_runs(baseline, candidate, policy=None) -> RegressionReport`**
  — takes two `EvaluationRun`s however the caller got them (kept in memory,
  loaded from hand-saved JSON — storage itself is a later phase). Raises
  `ValueError` if the two runs are of different datasets (`dataset_id`
  mismatch) — comparing unrelated datasets isn't a regression check.
  **Different systems/models/prompt versions are NOT an error** — comparing
  model v1 against model v2 on the same dataset is the normal use case this
  exists for, so only `dataset_id` is checked, never `system_name`.

- **Regression definition**: a test case that passed in the baseline (every
  evaluator that scored it passed — the same definition `runner.py`'s own
  aggregation already uses for `passed_tests`) and failed in the candidate.
  Foregrounded over the blunter "pass rate went down by X%" signal, which
  can hide one real regression behind several unrelated improvements, or
  raise a false alarm over one flaky test. Both signals are in the report;
  `regressed_test_ids` is the actionable one.

- **`newly_passing_test_ids`** — the mirror image (failed in baseline,
  passed in candidate). Purely informational — never counts against the
  policy, however large.

- **`only_in_baseline_test_ids` / `only_in_candidate_test_ids`** — test
  cases present in only one run (the dataset grew, shrank, or changed
  between the two runs). Tracked separately from regressions on purpose: a
  test case that no longer exists didn't "fail," it's just gone.

- **`dataset_changed`** — `True` when the two runs' `dataset_content_hash`
  values differ (Day 2's reproducibility hash, finally paying off here): a
  caller can tell at a glance whether the dataset itself moved between
  baseline and candidate, not just the system under test.

## `RegressionPolicy` — strict by default, on purpose

`max_pass_rate_drop`, `max_average_score_drop`, `max_new_regressions` all
default to **`0`** — any drop at all, or any newly regressed test case,
fails the policy. Same "fail loud, don't silently tolerate" judgment call
Day 4's deterministic evaluators already make for missing configuration: a
policy that quietly accepts a little erosion every run is how a system
drifts from 95% to 60% one accepted percent at a time. Loosen a threshold
explicitly when some tolerance is genuinely wanted — don't rely on a
lenient default.

## Tests

9 new tests (`test_regression.py`) covering: no regression, a real
regression (and that it fails the strict default policy with a specific
violation message), newly-passing test cases never counting against the
policy, a lenient policy explicitly tolerating a drop, an average-score-only
regression (pass rate unchanged), mismatched `dataset_id` raising, the
`dataset_changed` flag, dataset test-id churn tracked separately from
regressions, and comparing different systems/models as the intended,
error-free use case. **192 passed, 4 skipped.**

## Next

Day 11 — quality gates: turning a `RegressionReport` (or a standalone
`EvaluationRun`) into a hard CI pass/fail exit code, the piece that actually
blocks a bad deploy instead of just reporting on one after the fact.
