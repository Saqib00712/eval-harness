import pytest

from eval_harness.regression import RegressionPolicy, compare_runs
from eval_harness.run_models import EvaluationRun, PerTestResult

NOW = "2026-01-01T00:00:00Z"


def _run(run_id, dataset_id="d1", content_hash="h1", pass_rate=1.0, average_score=1.0, results=None) -> EvaluationRun:
    results = results or []
    return EvaluationRun(
        run_id=run_id, dataset_id=dataset_id, dataset_version="1.0.0", dataset_content_hash=content_hash,
        system_name="sys", system_version="0.0.0",
        started_at=NOW, completed_at=NOW, status="completed",
        total_tests=len({r.test_id for r in results}), passed_tests=0, failed_tests=0,
        pass_rate=pass_rate, average_score=average_score, total_tokens=0, estimated_cost=0.0,
        results=results,
    )


def _result(test_id, passed, evaluator="exact_match"):
    return PerTestResult(run_id="x", test_id=test_id, evaluator=evaluator, score=1.0 if passed else 0.0, passed=passed)


def test_identical_runs_have_no_regression_and_pass_policy():
    results = [_result("t1", True), _result("t2", True)]
    baseline = _run("b1", pass_rate=1.0, average_score=1.0, results=results)
    candidate = _run("c1", pass_rate=1.0, average_score=1.0, results=results)
    report = compare_runs(baseline, candidate)
    assert report.passed
    assert report.regressed_test_ids == []
    assert report.pass_rate_delta == 0.0


def test_regressed_test_case_detected_and_fails_strict_default_policy():
    baseline = _run("b1", pass_rate=1.0, results=[_result("t1", True), _result("t2", True)])
    candidate = _run("c1", pass_rate=0.5, results=[_result("t1", True), _result("t2", False)])
    report = compare_runs(baseline, candidate)
    assert report.regressed_test_ids == ["t2"]
    assert not report.passed
    assert any("regressed" in v for v in report.violations)
    assert any("pass_rate dropped" in v for v in report.violations)


def test_newly_passing_test_case_is_informational_not_a_violation():
    baseline = _run("b1", pass_rate=0.5, results=[_result("t1", False), _result("t2", True)])
    candidate = _run("c1", pass_rate=1.0, results=[_result("t1", True), _result("t2", True)])
    report = compare_runs(baseline, candidate)
    assert report.newly_passing_test_ids == ["t1"]
    assert report.regressed_test_ids == []
    assert report.passed  # improvement, not a regression


def test_policy_tolerance_allows_small_drop():
    baseline = _run("b1", pass_rate=1.0, results=[_result("t1", True), _result("t2", True)])
    candidate = _run("c1", pass_rate=0.5, results=[_result("t1", True), _result("t2", False)])
    lenient = RegressionPolicy(max_pass_rate_drop=1.0, max_new_regressions=5)
    report = compare_runs(baseline, candidate, lenient)
    assert report.passed


def test_average_score_drop_flagged_even_if_pass_rate_unchanged():
    baseline = _run("b1", pass_rate=1.0, average_score=0.95, results=[_result("t1", True)])
    candidate = _run("c1", pass_rate=1.0, average_score=0.70, results=[_result("t1", True)])
    report = compare_runs(baseline, candidate)
    assert not report.passed
    assert any("average_score dropped" in v for v in report.violations)


def test_different_dataset_ids_raise():
    baseline = _run("b1", dataset_id="d1")
    candidate = _run("c1", dataset_id="d2")
    with pytest.raises(ValueError, match="different datasets"):
        compare_runs(baseline, candidate)


def test_dataset_changed_flag_reflects_content_hash_mismatch():
    baseline = _run("b1", content_hash="h1", results=[_result("t1", True)])
    candidate = _run("c1", content_hash="h2", results=[_result("t1", True)])
    report = compare_runs(baseline, candidate)
    assert report.dataset_changed


def test_test_ids_only_in_one_run_are_tracked_separately_from_regressions():
    baseline = _run("b1", results=[_result("t1", True), _result("old", True)])
    candidate = _run("c1", results=[_result("t1", True), _result("new", True)])
    report = compare_runs(baseline, candidate)
    assert report.only_in_baseline_test_ids == ["old"]
    assert report.only_in_candidate_test_ids == ["new"]
    assert report.regressed_test_ids == []  # dropped/added test ids are not "regressions"


def test_different_system_or_model_is_the_normal_use_case_not_an_error():
    baseline = _run("b1", results=[_result("t1", True)])
    candidate = _run("c1", results=[_result("t1", True)])
    # system_name/version intentionally not checked -- comparing two different
    # models/prompt versions against the same dataset is the whole point.
    report = compare_runs(baseline, candidate)
    assert report.passed
