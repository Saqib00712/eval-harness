from eval_harness.quality_gate import (
    QualityGatePolicy,
    evaluate_quality_gate,
    exit_code_for,
    per_evaluator_pass_rates,
)
from eval_harness.regression import RegressionReport
from eval_harness.run_models import EvaluationRun, PerTestResult

NOW = "2026-01-01T00:00:00Z"


def _run(pass_rate=1.0, average_score=1.0, results=None) -> EvaluationRun:
    results = results or []
    return EvaluationRun(
        run_id="r1", dataset_id="d1", dataset_version="1.0.0", dataset_content_hash="h1",
        system_name="sys", system_version="0.0.0",
        started_at=NOW, completed_at=NOW, status="completed",
        total_tests=len({r.test_id for r in results}), passed_tests=0, failed_tests=0,
        pass_rate=pass_rate, average_score=average_score, total_tokens=0, estimated_cost=0.0,
        results=results,
    )


def _result(test_id, evaluator, passed, failure_category=None):
    return PerTestResult(
        run_id="r1", test_id=test_id, evaluator=evaluator, score=1.0 if passed else 0.0,
        passed=passed, failure_category=failure_category,
    )


def test_unconfigured_policy_always_passes():
    run = _run(pass_rate=0.1, average_score=0.1)
    result = evaluate_quality_gate(run)
    assert result.passed
    assert result.violations == []


def test_min_pass_rate_violation():
    run = _run(pass_rate=0.5)
    result = evaluate_quality_gate(run, QualityGatePolicy(min_pass_rate=0.9))
    assert not result.passed
    assert any("pass_rate" in v for v in result.violations)


def test_min_average_score_violation():
    run = _run(average_score=0.4)
    result = evaluate_quality_gate(run, QualityGatePolicy(min_average_score=0.8))
    assert not result.passed
    assert any("average_score" in v for v in result.violations)


def test_per_evaluator_pass_rates_grouped_correctly():
    results = [
        _result("t1", "faithfulness", True), _result("t2", "faithfulness", False),
        _result("t1", "exact_match", True), _result("t2", "exact_match", True),
    ]
    rates = per_evaluator_pass_rates(_run(results=results))
    assert rates == {"faithfulness": 0.5, "exact_match": 1.0}


def test_min_evaluator_pass_rate_violation():
    results = [_result("t1", "faithfulness", True), _result("t2", "faithfulness", False)]
    run = _run(results=results)
    result = evaluate_quality_gate(run, QualityGatePolicy(min_evaluator_pass_rate={"faithfulness": 0.9}))
    assert not result.passed
    assert any("faithfulness" in v for v in result.violations)


def test_required_evaluator_missing_from_run_is_a_violation():
    run = _run(results=[_result("t1", "exact_match", True)])
    result = evaluate_quality_gate(run, QualityGatePolicy(min_evaluator_pass_rate={"faithfulness": 0.5}))
    assert not result.passed
    assert any("never ran" in v for v in result.violations)


def test_forbidden_failure_category_blocks_even_with_good_pass_rate():
    results = [_result("t1", "exact_match", False, failure_category="dependency_failure")]
    run = _run(pass_rate=0.99, results=results)
    policy = QualityGatePolicy(forbid_failure_categories=["dependency_failure"])
    result = evaluate_quality_gate(run, policy)
    assert not result.passed
    assert any("forbidden failure categories" in v for v in result.violations)


def test_unlisted_failure_category_is_not_forbidden():
    results = [_result("t1", "exact_match", False, failure_category="threshold_miss")]
    run = _run(results=results)
    policy = QualityGatePolicy(forbid_failure_categories=["dependency_failure"])
    result = evaluate_quality_gate(run, policy)
    assert result.passed


def test_failing_regression_report_fails_the_gate():
    run = _run(pass_rate=1.0, average_score=1.0)
    bad_regression = RegressionReport(
        baseline_run_id="b", candidate_run_id="c", dataset_id="d1", dataset_changed=False,
        baseline_pass_rate=1.0, candidate_pass_rate=0.5, pass_rate_delta=-0.5,
        baseline_average_score=1.0, candidate_average_score=0.5, average_score_delta=-0.5,
        regressed_test_ids=["t1"], passed=False, violations=["1 test case(s) regressed: t1"],
    )
    result = evaluate_quality_gate(run, regression_report=bad_regression)
    assert not result.passed
    assert result.regression_passed is False
    assert any("regression check failed" in v for v in result.violations)


def test_passing_regression_report_does_not_block_gate():
    run = _run()
    good_regression = RegressionReport(
        baseline_run_id="b", candidate_run_id="c", dataset_id="d1", dataset_changed=False,
        baseline_pass_rate=1.0, candidate_pass_rate=1.0, pass_rate_delta=0.0,
        baseline_average_score=1.0, candidate_average_score=1.0, average_score_delta=0.0,
        passed=True,
    )
    result = evaluate_quality_gate(run, regression_report=good_regression)
    assert result.passed
    assert result.regression_passed is True


def test_no_regression_report_leaves_regression_passed_none():
    result = evaluate_quality_gate(_run())
    assert result.regression_passed is None


def test_exit_code_for_pass_and_fail():
    run = _run(pass_rate=1.0)
    passing = evaluate_quality_gate(run, QualityGatePolicy(min_pass_rate=0.5))
    failing = evaluate_quality_gate(run, QualityGatePolicy(min_pass_rate=2.0))
    assert exit_code_for(passing) == 0
    assert exit_code_for(failing) == 1
