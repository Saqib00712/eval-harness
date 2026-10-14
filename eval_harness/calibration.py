"""
Phase 22: human calibration foundation. Every threshold set anywhere in
this harness since Day 4 (0.80 for semantic similarity, 0.70 for an LLM
judge, whatever a `RubricJudgeEvaluator` was told) is a NUMBER SOMEONE
PICKED, not a number derived from ground truth. This module is the first
place the harness can actually check one of those numbers against what a
human would say, by recording a human verdict on a SAMPLE of results and
comparing it to what the automated evaluator scored for the exact same
(test_case, evaluator) pair.

"Foundation" in the phase name is deliberate, not a hedge: this is
recording + comparison, not a review UI, not a full active-learning
sampling strategy, not automatic threshold tuning from disagreement data.
Those are real future work this module makes POSSIBLE (you need calibrated
agreement data before you can tune a threshold against it) but does not
itself attempt.

TWO KINDS OF DISAGREEMENT, KEPT SEPARATE, NOT NETTED AGAINST EACH OTHER:
  FALSE POSITIVE (evaluator passed, human says fail) -- the evaluator is
    too LENIENT here. This is the more dangerous direction in most real
    pipelines: a bad answer that looks like it passed is a silent quality
    escape, exactly what an eval harness exists to catch.
  FALSE NEGATIVE (evaluator failed, human says pass) -- the evaluator is
    too STRICT here. Less dangerous, but still costly: it wastes review
    time and can block genuinely good work.
A single "disagreement rate" would collapse these into one undifferentiated
number and hide which direction a threshold needs to move. Both lists are
reported; neither is netted against the other.
"""

import random

from pydantic import BaseModel, ConfigDict, Field

from eval_harness.run_models import EvaluationRun, PerTestResult


class HumanLabel(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    test_id: str
    evaluator: str  # which evaluator's result this label is calibrating against
    human_passed: bool
    human_score: float | None = None  # optional finer-grained score, if the rater gives one
    rater: str
    notes: str = ""


class EvaluatorCalibration(BaseModel):
    model_config = ConfigDict(frozen=True)

    label_count: int
    agreement_rate: float
    false_positive_test_ids: list[str] = Field(default_factory=list)
    false_negative_test_ids: list[str] = Field(default_factory=list)


class CalibrationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    label_count: int
    agreement_rate: float
    false_positive_test_ids: list[str] = Field(default_factory=list)  # evaluator too lenient
    false_negative_test_ids: list[str] = Field(default_factory=list)  # evaluator too strict
    per_evaluator: dict[str, EvaluatorCalibration] = Field(default_factory=dict)


def select_sample_for_review(run: EvaluationRun, n: int, *, seed: int = 0) -> list[PerTestResult]:
    """A deterministic, seeded random sample of `n` PerTestResult rows for
    a human to label. Seeded (not plain `random.sample`) so the SAME seed
    always produces the SAME sample -- a reviewer re-running this gets a
    reproducible review set, not a different one every time, the same
    reproducibility principle Day 2's dataset content hash was built on."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}.")
    if n > len(run.results):
        raise ValueError(f"Cannot sample {n} results from a run with only {len(run.results)}.")
    return random.Random(seed).sample(run.results, n)


def compute_calibration(run: EvaluationRun, labels: list[HumanLabel]) -> CalibrationReport:
    if not labels:
        raise ValueError("Cannot compute calibration with zero human labels.")

    by_key: dict[tuple[str, str], PerTestResult] = {}
    for r in run.results:
        by_key[(r.test_id, r.evaluator)] = r  # one row per (test_case, evaluator) pair, same as the runner emits

    matched: list[tuple[HumanLabel, PerTestResult]] = []
    for label in labels:
        if label.run_id != run.run_id:
            raise ValueError(
                f"Label for test_id {label.test_id!r} is for run {label.run_id!r}, not {run.run_id!r} -- "
                f"calibrating against the wrong run would silently compare apples to oranges."
            )
        key = (label.test_id, label.evaluator)
        if key not in by_key:
            raise ValueError(
                f"No result for test_id={label.test_id!r} evaluator={label.evaluator!r} in run {run.run_id!r}."
            )
        matched.append((label, by_key[key]))

    false_positives = [l.test_id for l, r in matched if r.passed and not l.human_passed]
    false_negatives = [l.test_id for l, r in matched if not r.passed and l.human_passed]
    agreement_count = sum(1 for l, r in matched if l.human_passed == r.passed)

    per_evaluator: dict[str, EvaluatorCalibration] = {}
    by_evaluator: dict[str, list[tuple[HumanLabel, PerTestResult]]] = {}
    for l, r in matched:
        by_evaluator.setdefault(l.evaluator, []).append((l, r))
    for evaluator_name, pairs in by_evaluator.items():
        ev_agree = sum(1 for l, r in pairs if l.human_passed == r.passed)
        per_evaluator[evaluator_name] = EvaluatorCalibration(
            label_count=len(pairs), agreement_rate=round(ev_agree / len(pairs), 4),
            false_positive_test_ids=[l.test_id for l, r in pairs if r.passed and not l.human_passed],
            false_negative_test_ids=[l.test_id for l, r in pairs if not r.passed and l.human_passed],
        )

    return CalibrationReport(
        run_id=run.run_id, label_count=len(matched),
        agreement_rate=round(agreement_count / len(matched), 4),
        false_positive_test_ids=false_positives, false_negative_test_ids=false_negatives,
        per_evaluator=per_evaluator,
    )
