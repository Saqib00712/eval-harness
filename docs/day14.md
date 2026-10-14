# Day 14 — Human calibration foundation

Phase 22. Every threshold set anywhere in this harness since Day 4 (0.80
for semantic similarity, 0.70 for an LLM judge, whatever a
`RubricJudgeEvaluator` was told) is a number someone picked, not one
derived from ground truth. This is the first place the harness can check
one of those numbers against what a human would actually say.

**"Foundation" is deliberate, not a hedge**: this is recording + comparison,
not a review UI, not an active-learning sampling strategy, not automatic
threshold retuning. Those are real future work this module makes *possible*
(you need calibrated agreement data before you can tune a threshold against
it) but doesn't itself attempt.

## `eval_harness/calibration.py`

- **`select_sample_for_review(run, n, seed=0)`** — a deterministic, seeded
  random sample of `n` `PerTestResult` rows for a human to label. Same seed
  → same sample, always — a reviewer re-running this gets a reproducible
  review set, the same reproducibility principle Day 2's dataset content
  hash was built on, not a fresh shuffle every time.

- **`HumanLabel`** — `run_id`, `test_id`, `evaluator`, `human_passed`,
  optional `human_score`, `rater`, `notes`. One label calibrates against
  exactly one `(test_id, evaluator)` result row.

- **`compute_calibration(run, labels) -> CalibrationReport`** — matches
  each label to its result row and raises `ValueError` loudly on a label
  for the wrong run, or for a `(test_id, evaluator)` pair that doesn't
  exist in the run — calibrating against the wrong data silently would be
  worse than refusing.

## Two kinds of disagreement, never netted against each other

- **False positive** (evaluator passed, human says fail) — the evaluator
  is too *lenient*. The more dangerous direction in most real pipelines: a
  bad answer that looks like it passed is a silent quality escape, exactly
  what an eval harness exists to catch.
- **False negative** (evaluator failed, human says pass) — the evaluator
  is too *strict*. Less dangerous, but still costly: wasted review time,
  blocked good work.

A single "disagreement rate" would collapse these into one number and hide
which direction a threshold needs to move. Both lists are reported
separately, overall and per evaluator (`CalibrationReport.per_evaluator`).

## Tests

12 new tests (`test_calibration.py`): full agreement, a false positive, a
false negative, the per-evaluator breakdown, a label for a missing result
raising, a label for the wrong run raising, zero labels raising, and the
sampler's determinism (same seed → same sample, different seeds can
differ) plus its two guardrails. **231 passed, 4 skipped.**

## Next

Day 15 — provider abstraction review & storage: persisting `EvaluationRun`s
(and now `HumanLabel`s) to disk so a baseline for Day 10's regression
checks, and a calibration dataset for this day, actually survive between
runs instead of living only in memory.
