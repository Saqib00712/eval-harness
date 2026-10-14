"""
Phase 19/20: cost & efficiency evaluation -- turning the token/cost/latency
numbers a run already reports into real efficiency metrics, chief among
them cost PER PASSING TEST, which `EvaluationRun` never surfaced directly
before today.

COST PER PASSING TEST, NOT JUST TOTAL COST: a run's `estimated_cost` alone
can't say whether a system is actually a good deal -- a cheap system that
fails half its tests can easily cost MORE per correct answer than an
expensive system that rarely fails. Dividing by `passed_tests` instead of
`total_tests` is this module's whole point, the same "a raw total hides
the number that actually matters" lesson Day 12 already applied when
picking a comparison winner by pass rate rather than by eyeballing totals.

UNDEFINED, NOT INFINITE, WHEN NOTHING PASSED: a run with zero passing
tests has an undefined cost-per-success -- `cost_per_passing_test` (and
`tokens_per_passing_test`) is `None` in that case, never `0.0` (which
would misread as "free") and never `float('inf')` (which doesn't
JSON-serialize and complicates every comparison). `rank_by_cost_per_
passing_test` below sorts these runs LAST, not first: a system that
passed nothing achieved nothing, and ranking it as "cheapest" because the
denominator is undefined would reward failure.

SYSTEM LATENCY -- A GAP CLOSED, NOT A NEW FEATURE BOLTED ON:
`SystemOutput.latency_ms` has existed since Day 3 but was never aggregated
anywhere. `PerTestResult.latency_ms` is deliberately the EVALUATOR's own
latency (Day 5/8's docstrings are explicit about this); copying the
system's latency onto every evaluator row scoring the same test case would
be exactly the kind of double-count Day 5's cost aggregation was already
careful to avoid. The fix (this day, in runner.py) follows the same
once-per-test-case precedent system cost already used:
`EvaluationRun.average_system_latency_ms` / `max_system_latency_ms`.
"""

from pydantic import BaseModel, ConfigDict

from eval_harness.run_models import EvaluationRun


class EfficiencyReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    system_name: str
    total_tests: int
    passed_tests: int

    total_cost: float
    total_tokens: int
    cost_per_test: float
    tokens_per_test: float
    cost_per_passing_test: float | None  # None iff passed_tests == 0 -- undefined, not free
    tokens_per_passing_test: float | None

    average_system_latency_ms: float | None
    max_system_latency_ms: float | None


def compute_efficiency(run: EvaluationRun) -> EfficiencyReport:
    if run.total_tests == 0:
        raise ValueError(f"Cannot compute efficiency for a run with zero test cases ({run.run_id!r}).")

    cost_per_test = round(run.estimated_cost / run.total_tests, 8)
    tokens_per_test = round(run.total_tokens / run.total_tests, 4)
    cost_per_passing_test = round(run.estimated_cost / run.passed_tests, 8) if run.passed_tests else None
    tokens_per_passing_test = round(run.total_tokens / run.passed_tests, 4) if run.passed_tests else None

    return EfficiencyReport(
        run_id=run.run_id, system_name=run.system_name,
        total_tests=run.total_tests, passed_tests=run.passed_tests,
        total_cost=run.estimated_cost, total_tokens=run.total_tokens,
        cost_per_test=cost_per_test, tokens_per_test=tokens_per_test,
        cost_per_passing_test=cost_per_passing_test, tokens_per_passing_test=tokens_per_passing_test,
        average_system_latency_ms=run.average_system_latency_ms, max_system_latency_ms=run.max_system_latency_ms,
    )


def rank_by_cost_per_passing_test(runs: list[EvaluationRun]) -> list[EfficiencyReport]:
    """Cheapest-per-success first. Runs with an undefined cost-per-success
    (zero passing tests) sort LAST regardless of how low their total cost
    was -- see the module docstring for why that's not an edge case to
    special-case away but the whole reason this isn't a plain `0.0`."""
    reports = [compute_efficiency(r) for r in runs]
    return sorted(reports, key=lambda r: (r.cost_per_passing_test is None, r.cost_per_passing_test or 0.0))
