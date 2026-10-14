"""
Phase 17: quality gates -- turning an `EvaluationRun` (and, optionally, a
Day 10 `RegressionReport`) into a single hard pass/fail verdict a CI
pipeline can actually act on, instead of a human reading a report and
deciding by eye whether to block a deploy.

DELIBERATE CONTRAST WITH DAY 10: `RegressionPolicy` (regression.py)
defaults to STRICT -- zero tolerance for any regression, because "don't
let this get any worse than it already is" is a sane universal default
for every system. `QualityGatePolicy` here defaults to the OPPOSITE --
every threshold starts unset, and an unconfigured gate always passes its
static checks. There is no universally correct minimum pass rate or
average score across every possible system this harness might evaluate
(70% might be a strong RAG system or a broken classifier, depending on the
task) -- inventing one would be a silent, wrong opinion. The honest
default is "nothing is gated until you say what matters for YOUR system",
the same philosophy every evaluator's constructor-default-or-metadata-
override pattern already follows at the test-case level, applied here at
the run level.

The one exception: `forbid_failure_categories` also defaults to empty
(opt-in, not opt-out) for the same reason -- a one-off `dependency_failure`
from a transient network blip during a dev run shouldn't silently block
every CI run forever just because this module decided it should matter.
Set it explicitly (e.g. `forbid_failure_categories=["dependency_failure"]`)
for "a single crash fails the build, no matter what the pass rate is."
"""

from pydantic import BaseModel, ConfigDict, Field

from eval_harness.regression import RegressionReport
from eval_harness.run_models import EvaluationRun


class QualityGatePolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    min_pass_rate: float | None = None
    min_average_score: float | None = None
    min_evaluator_pass_rate: dict[str, float] = Field(default_factory=dict)
    forbid_failure_categories: list[str] = Field(default_factory=list)


class QualityGateResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    passed: bool
    violations: list[str] = Field(default_factory=list)
    pass_rate: float
    average_score: float
    per_evaluator_pass_rate: dict[str, float] = Field(default_factory=dict)
    regression_passed: bool | None = None  # None iff no RegressionReport was given


def per_evaluator_pass_rates(run: EvaluationRun) -> dict[str, float]:
    """Pass rate PER EVALUATOR across the whole run -- distinct from
    `EvaluationRun.pass_rate`, which is per TEST CASE (every evaluator
    scoring it has to pass). A system can have a healthy overall pass rate
    while one specific evaluator (say, `faithfulness`) is quietly failing
    most of the time; this is how a gate catches that."""
    tally: dict[str, list[bool]] = {}
    for r in run.results:
        tally.setdefault(r.evaluator, []).append(r.passed)
    return {name: round(sum(flags) / len(flags), 4) for name, flags in tally.items()}


def evaluate_quality_gate(
    run: EvaluationRun,
    policy: QualityGatePolicy | None = None,
    regression_report: RegressionReport | None = None,
) -> QualityGateResult:
    policy = policy or QualityGatePolicy()
    violations: list[str] = []

    if policy.min_pass_rate is not None and run.pass_rate < policy.min_pass_rate:
        violations.append(f"pass_rate {run.pass_rate:.4f} is below minimum {policy.min_pass_rate:.4f}")

    if policy.min_average_score is not None and run.average_score < policy.min_average_score:
        violations.append(
            f"average_score {run.average_score:.4f} is below minimum {policy.min_average_score:.4f}"
        )

    evaluator_rates = per_evaluator_pass_rates(run)
    for evaluator_name, minimum in policy.min_evaluator_pass_rate.items():
        actual = evaluator_rates.get(evaluator_name)
        if actual is None:
            violations.append(f"evaluator {evaluator_name!r} required by policy never ran in this run")
        elif actual < minimum:
            violations.append(
                f"evaluator {evaluator_name!r} pass_rate {actual:.4f} is below minimum {minimum:.4f}"
            )

    if policy.forbid_failure_categories:
        forbidden_present = {
            r.failure_category for r in run.results if r.failure_category in policy.forbid_failure_categories
        }
        if forbidden_present:
            violations.append(f"forbidden failure categories present: {sorted(forbidden_present)}")

    regression_passed = None
    if regression_report is not None:
        regression_passed = regression_report.passed
        if not regression_report.passed:
            violations.append(f"regression check failed: {'; '.join(regression_report.violations)}")

    return QualityGateResult(
        passed=not violations, violations=violations,
        pass_rate=run.pass_rate, average_score=run.average_score,
        per_evaluator_pass_rate=evaluator_rates, regression_passed=regression_passed,
    )


def exit_code_for(result: QualityGateResult) -> int:
    """0 for a CI script to treat as success, 1 for failure -- the standard
    shell/CI convention. A thin, deliberately trivial wrapper (not just
    `int(not result.passed)` inline at every call site) so a CI script
    reads as `sys.exit(exit_code_for(result))` and the convention being
    followed is explicit, not implicit in a bare boolean-to-int cast."""
    return 0 if result.passed else 1
