"""
Phase 23: observability. Everything in `EvaluationRun.results` already
has the numbers needed to answer "where did the time/cost go" -- Day 5
put `latency_ms`/`token_usage_total`/`estimated_cost` on every
`PerTestResult`, Day 13 added run-level system latency. What's still
missing is: (1) seeing any of that WHILE a run is happening, not only
after the whole thing finishes, and (2) a ready-made way to ask "which
test cases were actually slow/expensive" instead of eyeballing a results
list by hand. This module adds both.

RunObserver IS A DIRECT, GENERALIZED REUSE of Project 1's own
`app/observability/tracing.py` pattern -- not the contextvar/span
machinery itself (that solves "one trace id per concurrent HTTP request",
a problem this single-process, sequential-by-design harness doesn't have,
Day 12 already explained why), but the SHAPE of the idea: a lightweight
hook interface the pipeline calls into at well-defined points, so
"how do I see what's happening" doesn't require modifying the pipeline
itself. Every method on `RunObserver` is a no-op by default -- subclass it
and override only the hooks you care about, the same design `logging.
Handler` and `unittest.TestResult` use for the identical reason: most
callers want one or two events, not all of them, and a Protocol with every
method mandatory would force implementing hooks nobody asked for.

`runner.py`'s `run_evaluation` takes an optional `observer` (defaulting to
a plain `RunObserver()`, i.e. silent) and calls it at six points: a test
case starting, the system failing or succeeding, each evaluator failing or
scoring, and the run completing. None of this changes what `run_evaluation`
RETURNS -- an observer is a side channel for watching, never a dependency
of the result.
"""

import logging

from eval_harness.evaluator import EvaluatorResult
from eval_harness.models import TestCase
from eval_harness.run_models import EvaluationRun, PerTestResult
from eval_harness.system_output import SystemOutput


class RunObserver:
    """Base class: override only the hooks you need. Every method here
    does nothing, so a subclass that only cares about, say, evaluator
    failures can implement `on_evaluator_error` alone and ignore the
    other five."""

    def on_test_case_start(self, test_case: TestCase) -> None:
        pass

    def on_system_error(self, test_case: TestCase, exc: Exception) -> None:
        pass

    def on_system_output(self, test_case: TestCase, output: SystemOutput) -> None:
        pass

    def on_evaluator_error(self, test_case: TestCase, evaluator_name: str, exc: Exception) -> None:
        pass

    def on_evaluator_result(self, test_case: TestCase, evaluator_name: str, result: EvaluatorResult) -> None:
        pass

    def on_run_complete(self, run: EvaluationRun) -> None:
        pass


class StructuredLogObserver(RunObserver):
    """Emits one structured log record per event via the stdlib `logging`
    module, fields attached under `extra` rather than interpolated into
    the message string -- the point is to pair this with a JSON (or any
    structured) log formatter downstream, not to be readable raw in a
    terminal. A caller who wants readable terminal output instead can
    subclass `RunObserver` directly and `print(...)` -- this class is one
    ready-made option, not the only way to use the hooks."""

    def __init__(self, logger: logging.Logger | None = None):
        self._logger = logger or logging.getLogger("eval_harness")

    def on_test_case_start(self, test_case: TestCase) -> None:
        self._logger.info("test_case_start", extra={"test_id": test_case.test_id})

    def on_system_error(self, test_case: TestCase, exc: Exception) -> None:
        self._logger.warning(
            "system_error", extra={"test_id": test_case.test_id, "error_type": type(exc).__name__}
        )

    def on_system_output(self, test_case: TestCase, output: SystemOutput) -> None:
        self._logger.info(
            "system_output",
            extra={"test_id": test_case.test_id, "latency_ms": output.latency_ms, "estimated_cost": output.estimated_cost},
        )

    def on_evaluator_error(self, test_case: TestCase, evaluator_name: str, exc: Exception) -> None:
        self._logger.warning(
            "evaluator_error",
            extra={"test_id": test_case.test_id, "evaluator": evaluator_name, "error_type": type(exc).__name__},
        )

    def on_evaluator_result(self, test_case: TestCase, evaluator_name: str, result: EvaluatorResult) -> None:
        self._logger.info(
            "evaluator_result",
            extra={
                "test_id": test_case.test_id, "evaluator": evaluator_name,
                "score": result.score, "passed": result.passed,
            },
        )

    def on_run_complete(self, run: EvaluationRun) -> None:
        self._logger.info(
            "run_complete",
            extra={
                "run_id": run.run_id, "pass_rate": run.pass_rate,
                "total_tests": run.total_tests, "estimated_cost": run.estimated_cost,
            },
        )


def slowest_results(run: EvaluationRun, n: int = 5) -> list[PerTestResult]:
    """The `n` PerTestResult rows with the highest `latency_ms`, descending
    -- post-hoc "where did the time go" for a run that already finished.
    Rows with no recorded latency (deterministic evaluators never set it
    meaningfully fast, but a loaded/failed row might have `None`) are
    excluded, not treated as 0.0 -- a missing measurement isn't a fast one."""
    timed = [r for r in run.results if r.latency_ms is not None]
    return sorted(timed, key=lambda r: r.latency_ms, reverse=True)[:n]


def most_expensive_results(run: EvaluationRun, n: int = 5) -> list[PerTestResult]:
    """The `n` PerTestResult rows with the highest `estimated_cost`,
    descending -- surfaces which specific (test_case, evaluator) pairs are
    actually driving a run's bill, typically LLM-judge calls (Day 8).
    Rows with no recorded cost are excluded, same reasoning as
    `slowest_results`."""
    priced = [r for r in run.results if r.estimated_cost is not None]
    return sorted(priced, key=lambda r: r.estimated_cost, reverse=True)[:n]
