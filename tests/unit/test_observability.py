import logging

from eval_harness.observability import RunObserver, StructuredLogObserver, most_expensive_results, slowest_results
from eval_harness.run_models import EvaluationRun, PerTestResult

NOW = "2026-01-01T00:00:00Z"


def _run(results) -> EvaluationRun:
    return EvaluationRun(
        run_id="r1", dataset_id="d1", dataset_version="1.0.0", dataset_content_hash="h1",
        system_name="sys", system_version="0.0.0",
        started_at=NOW, completed_at=NOW, status="completed",
        total_tests=len({r.test_id for r in results}), passed_tests=0, failed_tests=0,
        pass_rate=0.0, average_score=0.0, total_tokens=0, estimated_cost=0.0,
        results=results,
    )


def _result(test_id, evaluator="x", latency_ms=None, estimated_cost=None):
    return PerTestResult(
        run_id="r1", test_id=test_id, evaluator=evaluator, score=1.0, passed=True,
        latency_ms=latency_ms, estimated_cost=estimated_cost,
    )


# ---- base RunObserver: every hook is a no-op ----

def test_base_observer_hooks_do_nothing_and_never_raise():
    observer = RunObserver()
    observer.on_test_case_start(None)
    observer.on_system_error(None, RuntimeError("x"))
    observer.on_system_output(None, None)
    observer.on_evaluator_error(None, "ev", RuntimeError("x"))
    observer.on_evaluator_result(None, "ev", None)
    observer.on_run_complete(None)  # none of this should raise


# ---- StructuredLogObserver: emits log records with structured fields ----

def test_structured_log_observer_emits_records(caplog):
    from eval_harness.models import TestCase
    from eval_harness.system_output import SystemOutput

    logger = logging.getLogger("eval_harness.test")
    observer = StructuredLogObserver(logger)
    tc = TestCase(test_id="t1", input="x")

    with caplog.at_level(logging.INFO, logger="eval_harness.test"):
        observer.on_test_case_start(tc)
        observer.on_system_output(tc, SystemOutput(output="y", latency_ms=12.0))

    messages = [r.message for r in caplog.records]
    assert "test_case_start" in messages
    assert "system_output" in messages
    start_record = next(r for r in caplog.records if r.message == "test_case_start")
    assert start_record.test_id == "t1"


def test_structured_log_observer_emits_remaining_hooks(caplog):
    # Day 18 test-suite audit: the first test above only exercised 2 of the
    # 6 hooks (on_test_case_start, on_system_output). These four were never
    # called by any test at all.
    from eval_harness.evaluator import EvaluatorResult
    from eval_harness.models import TestCase
    from eval_harness.run_models import EvaluationRun

    logger = logging.getLogger("eval_harness.test2")
    observer = StructuredLogObserver(logger)
    tc = TestCase(test_id="t1", input="x")
    run = EvaluationRun(
        run_id="r1", dataset_id="d1", dataset_version="1.0.0", dataset_content_hash="h1",
        system_name="sys", system_version="0.0.0",
        started_at="2026-01-01T00:00:00Z", completed_at="2026-01-01T00:00:00Z", status="completed",
        total_tests=1, passed_tests=1, failed_tests=0, pass_rate=1.0, average_score=1.0,
        total_tokens=0, estimated_cost=0.0,
    )

    with caplog.at_level(logging.INFO, logger="eval_harness.test2"):
        observer.on_system_error(tc, RuntimeError("boom"))
        observer.on_evaluator_error(tc, "exact_match", RuntimeError("boom"))
        observer.on_evaluator_result(tc, "exact_match", EvaluatorResult(evaluator="exact_match", score=1.0, passed=True))
        observer.on_run_complete(run)

    messages = {r.message for r in caplog.records}
    assert messages == {"system_error", "evaluator_error", "evaluator_result", "run_complete"}
    error_record = next(r for r in caplog.records if r.message == "system_error")
    assert error_record.error_type == "RuntimeError"
    complete_record = next(r for r in caplog.records if r.message == "run_complete")
    assert complete_record.run_id == "r1"


# ---- slowest_results / most_expensive_results ----

def test_slowest_results_sorted_descending():
    run = _run([_result("t1", latency_ms=50.0), _result("t2", latency_ms=200.0), _result("t3", latency_ms=100.0)])
    top = slowest_results(run, n=2)
    assert [r.test_id for r in top] == ["t2", "t3"]


def test_slowest_results_excludes_rows_with_no_latency():
    run = _run([_result("t1", latency_ms=None), _result("t2", latency_ms=5.0)])
    top = slowest_results(run, n=5)
    assert [r.test_id for r in top] == ["t2"]


def test_most_expensive_results_sorted_descending():
    run = _run([
        _result("t1", estimated_cost=0.01), _result("t2", estimated_cost=0.05), _result("t3", estimated_cost=0.02),
    ])
    top = most_expensive_results(run, n=2)
    assert [r.test_id for r in top] == ["t2", "t3"]


def test_most_expensive_results_excludes_rows_with_no_cost():
    run = _run([_result("t1", estimated_cost=None), _result("t2", estimated_cost=0.1)])
    top = most_expensive_results(run, n=5)
    assert [r.test_id for r in top] == ["t2"]
