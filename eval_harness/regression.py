"""
Phase 15/16: regression testing and policy. A standalone pass rate
("91% passed") answers "is this system good"; it can't answer the question
that actually matters day-to-day -- "did TODAY'S change make things worse
than YESTERDAY". That needs a second run to compare against, which is what
this module does: compare a new `EvaluationRun` (the candidate -- a new
model, a new prompt version, a new retrieval config) against a previous one
of the SAME dataset (the baseline) and decide, by an explicit, configurable
POLICY, whether the candidate regressed.

Comparing runs is deliberately NOT tied to any storage mechanism (that's a
later phase) -- `compare_runs` takes two `EvaluationRun` objects, however
the caller got them (kept in memory, loaded from wherever they saved JSON,
round-tripped through dataset_io.py's save/load pattern by hand). This
module's only job is the comparison and the pass/fail policy decision.

REGRESSION DEFINITION: a test case that PASSED in the baseline (every
evaluator that scored it passed, same definition `runner.py`'s aggregation
already uses for `EvaluationRun.passed_tests`) and FAILED in the candidate.
This is the most actionable signal a regression report can give -- "these
specific test cases used to work and now don't" -- deliberately foregrounded
over the blunter instrument of "the overall pass rate went down by X%",
which can hide a regression (many small improvements masking one real
regression) just as easily as it can flag a false alarm (dataset
reshuffled, one flaky test).

POLICY: `RegressionPolicy`'s defaults are strict -- ANY pass-rate drop,
ANY average-score drop, or ANY newly-regressed test case fails the policy
by default. This is the same "fail loud, don't silently tolerate" judgment
call every deterministic evaluator's missing-configuration check already
makes (Day 4) -- a regression policy that quietly accepts a little erosion
every run is how a system drifts from 95% to 60% one accepted percent at a
time. Loosen the thresholds explicitly if some tolerance is actually
wanted; don't rely on a lenient default.
"""

from pydantic import BaseModel, ConfigDict, Field

from eval_harness.run_models import EvaluationRun


class RegressionPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_pass_rate_drop: float = 0.0
    max_average_score_drop: float = 0.0
    max_new_regressions: int = 0


class RegressionReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    baseline_run_id: str
    candidate_run_id: str
    dataset_id: str
    dataset_changed: bool  # baseline/candidate ran against different dataset content_hash

    baseline_pass_rate: float
    candidate_pass_rate: float
    pass_rate_delta: float  # candidate - baseline; negative = worse

    baseline_average_score: float
    candidate_average_score: float
    average_score_delta: float  # candidate - baseline; negative = worse

    regressed_test_ids: list[str] = Field(default_factory=list)  # passed in baseline, failed in candidate
    newly_passing_test_ids: list[str] = Field(default_factory=list)  # failed in baseline, passed in candidate -- informational, never a policy violation
    only_in_baseline_test_ids: list[str] = Field(default_factory=list)  # dataset shrank or test_ids changed
    only_in_candidate_test_ids: list[str] = Field(default_factory=list)  # dataset grew or test_ids changed

    passed: bool  # does the candidate satisfy the policy
    violations: list[str] = Field(default_factory=list)  # human-readable reasons `passed` is False


def _test_case_pass_status(run: EvaluationRun) -> dict[str, bool]:
    """A test_id passed iff every PerTestResult scored for it passed --
    the identical definition runner.py's own aggregation uses for
    EvaluationRun.passed_tests, applied per test_id instead of summed."""
    by_test: dict[str, list[bool]] = {}
    for r in run.results:
        by_test.setdefault(r.test_id, []).append(r.passed)
    return {test_id: all(flags) for test_id, flags in by_test.items()}


def compare_runs(
    baseline: EvaluationRun, candidate: EvaluationRun, policy: RegressionPolicy | None = None
) -> RegressionReport:
    if baseline.dataset_id != candidate.dataset_id:
        raise ValueError(
            f"Cannot compare runs of different datasets ({baseline.dataset_id!r} vs "
            f"{candidate.dataset_id!r}) -- a regression report only makes sense for two runs "
            f"of the SAME dataset (system/model/prompt_version may differ -- that's the usual "
            f"point of comparing them)."
        )
    policy = policy or RegressionPolicy()

    baseline_status = _test_case_pass_status(baseline)
    candidate_status = _test_case_pass_status(candidate)
    baseline_ids, candidate_ids = set(baseline_status), set(candidate_status)
    shared_ids = baseline_ids & candidate_ids

    regressed = sorted(tid for tid in shared_ids if baseline_status[tid] and not candidate_status[tid])
    newly_passing = sorted(tid for tid in shared_ids if not baseline_status[tid] and candidate_status[tid])

    pass_rate_delta = round(candidate.pass_rate - baseline.pass_rate, 6)
    average_score_delta = round(candidate.average_score - baseline.average_score, 6)

    violations: list[str] = []
    if pass_rate_delta < -policy.max_pass_rate_drop:
        violations.append(
            f"pass_rate dropped by {-pass_rate_delta:.4f} "
            f"(allowed: {policy.max_pass_rate_drop:.4f})"
        )
    if average_score_delta < -policy.max_average_score_drop:
        violations.append(
            f"average_score dropped by {-average_score_delta:.4f} "
            f"(allowed: {policy.max_average_score_drop:.4f})"
        )
    if len(regressed) > policy.max_new_regressions:
        violations.append(
            f"{len(regressed)} test case(s) regressed (allowed: {policy.max_new_regressions}): "
            f"{', '.join(regressed)}"
        )

    return RegressionReport(
        baseline_run_id=baseline.run_id, candidate_run_id=candidate.run_id, dataset_id=baseline.dataset_id,
        dataset_changed=baseline.dataset_content_hash != candidate.dataset_content_hash,
        baseline_pass_rate=baseline.pass_rate, candidate_pass_rate=candidate.pass_rate,
        pass_rate_delta=pass_rate_delta,
        baseline_average_score=baseline.average_score, candidate_average_score=candidate.average_score,
        average_score_delta=average_score_delta,
        regressed_test_ids=regressed, newly_passing_test_ids=newly_passing,
        only_in_baseline_test_ids=sorted(baseline_ids - candidate_ids),
        only_in_candidate_test_ids=sorted(candidate_ids - baseline_ids),
        passed=not violations, violations=violations,
    )
