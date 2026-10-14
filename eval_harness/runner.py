"""
The evaluation runner: the orchestrator that actually runs a Dataset
through a SystemUnderTest and scores each result with a list of
Evaluators, producing one EvaluationRun (Phase 11) made of many
PerTestResults (Phase 12).

Failure handling (Phase 29's "partial failure" / "evaluator failure" /
"system-under-test failure" test cases, decided here):

  * The SYSTEM raising for one test case does not abort the run. Every
    evaluator that WOULD have scored that test case instead gets one
    PerTestResult with passed=False, score=0.0,
    failure_category="dependency_failure" -- the system never produced
    output, so by definition it cannot be said to have passed anything.
  * An EVALUATOR raising for one (test_case, evaluator) pair does not abort
    the run either, and does not stop the OTHER evaluators from scoring
    that same test case. Only that one pair gets
    failure_category="evaluation_failure".
  * An EMPTY dataset or an EMPTY evaluator list raises ValueError
    immediately -- a run that silently "succeeds" with zero tests measured
    is worse than one that fails loudly, the same judgment call made for
    every deterministic evaluator's missing configuration (Day 4).
  * An evaluator that runs to completion and simply scores below its
    threshold gets failure_category="threshold_miss" (Day 9's addition --
    see failure_taxonomy.py for why this was a real gap, not a nice-to-
    have: before Day 9 this case left failure_category as None forever,
    indistinguishable from "nothing went wrong").

A test case counts as PASSED at the run level iff EVERY evaluator that
scored it passed -- one failing check fails the whole test case, the usual
meaning of "this test passed" in a CI-like setting.

`observer` (Day 16, observability.py) is an optional side channel for
watching a run AS it happens, never a dependency of what gets returned --
a plain, silent `RunObserver()` is used when none is given, so passing
nothing behaves exactly as it always has.
"""

import time
import uuid
from datetime import datetime, timezone

from eval_harness.dataset import Dataset
from eval_harness.evaluator import Evaluator
from eval_harness.failure_taxonomy import FailureCategory
from eval_harness.observability import RunObserver
from eval_harness.run_models import EvaluationRun, PerTestResult
from eval_harness.system import SystemUnderTest


async def run_evaluation(
    dataset: Dataset,
    system: SystemUnderTest,
    evaluators: list[Evaluator],
    *,
    model: str | None = None,
    prompt_version: str | None = None,
    configuration: dict | None = None,
    observer: RunObserver | None = None,
) -> EvaluationRun:
    if not dataset.test_cases:
        raise ValueError(f"Cannot run an evaluation on empty dataset {dataset.dataset_id!r}.")
    if not evaluators:
        raise ValueError("Cannot run an evaluation with no evaluators configured.")
    observer = observer or RunObserver()

    run_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)
    results: list[PerTestResult] = []
    system_tokens_total = 0
    system_cost_total = 0.0
    system_latencies: list[float] = []

    for test_case in dataset.test_cases:
        observer.on_test_case_start(test_case)
        try:
            output = await system.run(test_case)
        except Exception as exc:  # the system crashed outright for this test case
            observer.on_system_error(test_case, exc)
            for evaluator in evaluators:
                results.append(PerTestResult(
                    run_id=run_id, test_id=test_case.test_id,
                    evaluator=getattr(evaluator, "name", evaluator.__class__.__name__),
                    score=0.0, passed=False,
                    reason=f"System under test raised {type(exc).__name__}: {exc}",
                    failure_category=FailureCategory.DEPENDENCY_FAILURE,
                ))
            continue

        observer.on_system_output(test_case, output)

        if output.token_usage is not None:
            system_tokens_total += output.token_usage.total_tokens
        if output.estimated_cost is not None:
            system_cost_total += output.estimated_cost
        if output.latency_ms is not None:
            system_latencies.append(output.latency_ms)  # once per test case -- see run_models.py's note

        for evaluator in evaluators:
            evaluator_name = getattr(evaluator, "name", evaluator.__class__.__name__)
            t0 = time.perf_counter()
            try:
                result = await evaluator.evaluate(test_case, output)
            except Exception as exc:  # the evaluator itself is misconfigured or crashed
                observer.on_evaluator_error(test_case, evaluator_name, exc)
                results.append(PerTestResult(
                    run_id=run_id, test_id=test_case.test_id, evaluator=evaluator_name,
                    score=0.0, passed=False,
                    reason=f"Evaluator raised {type(exc).__name__}: {exc}",
                    latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                    failure_category=FailureCategory.EVALUATION_FAILURE,
                ))
                continue
            observer.on_evaluator_result(test_case, evaluator_name, result)
            results.append(PerTestResult(
                run_id=run_id, test_id=test_case.test_id, evaluator=result.evaluator,
                score=result.score, threshold=result.threshold, passed=result.passed, reason=result.reason,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                token_usage_total=result.token_usage_total, estimated_cost=result.estimated_cost,
                failure_category=None if result.passed else FailureCategory.THRESHOLD_MISS,
                metadata=result.metadata,
            ))

    completed_at = datetime.now(timezone.utc)
    run = _aggregate(
        run_id=run_id, dataset=dataset, system=system, results=results,
        started_at=started_at, completed_at=completed_at,
        system_tokens_total=system_tokens_total, system_cost_total=system_cost_total,
        system_latencies=system_latencies,
        model=model, prompt_version=prompt_version, configuration=configuration or {},
    )
    observer.on_run_complete(run)
    return run


def _aggregate(
    *, run_id: str, dataset: Dataset, system: SystemUnderTest, results: list[PerTestResult],
    started_at: datetime, completed_at: datetime,
    system_tokens_total: int, system_cost_total: float, system_latencies: list[float],
    model: str | None, prompt_version: str | None, configuration: dict,
) -> EvaluationRun:
    by_test: dict[str, list[PerTestResult]] = {}
    for r in results:
        by_test.setdefault(r.test_id, []).append(r)

    total_tests = len(by_test)
    passed_tests = sum(1 for rows in by_test.values() if all(r.passed for r in rows))
    failed_tests = total_tests - passed_tests
    pass_rate = round(passed_tests / total_tests, 4) if total_tests else 0.0
    average_score = round(sum(r.score for r in results) / len(results), 4) if results else 0.0

    evaluator_tokens = sum(r.token_usage_total or 0 for r in results)
    evaluator_cost = sum(r.estimated_cost or 0.0 for r in results)

    average_latency = round(sum(system_latencies) / len(system_latencies), 2) if system_latencies else None
    max_latency = round(max(system_latencies), 2) if system_latencies else None

    return EvaluationRun(
        run_id=run_id, dataset_id=dataset.dataset_id, dataset_version=dataset.version,
        dataset_content_hash=dataset.content_hash, system_name=system.name, system_version=system.version,
        model=model, prompt_version=prompt_version, configuration=configuration,
        started_at=started_at, completed_at=completed_at, status="completed",
        total_tests=total_tests, passed_tests=passed_tests, failed_tests=failed_tests,
        pass_rate=pass_rate, average_score=average_score,
        total_tokens=system_tokens_total + evaluator_tokens,
        estimated_cost=round(system_cost_total + evaluator_cost, 8),
        average_system_latency_ms=average_latency, max_system_latency_ms=max_latency,
        results=results,
    )
