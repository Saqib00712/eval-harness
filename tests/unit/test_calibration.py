import pytest

from eval_harness.calibration import HumanLabel, compute_calibration, select_sample_for_review
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


def _result(test_id, evaluator, passed):
    return PerTestResult(run_id="r1", test_id=test_id, evaluator=evaluator, score=1.0 if passed else 0.0, passed=passed)


def _label(test_id, evaluator, human_passed, run_id="r1"):
    return HumanLabel(run_id=run_id, test_id=test_id, evaluator=evaluator, human_passed=human_passed, rater="alice")


def test_full_agreement():
    run = _run([_result("t1", "faithfulness", True), _result("t2", "faithfulness", False)])
    labels = [_label("t1", "faithfulness", True), _label("t2", "faithfulness", False)]
    report = compute_calibration(run, labels)
    assert report.agreement_rate == 1.0
    assert report.false_positive_test_ids == []
    assert report.false_negative_test_ids == []


def test_false_positive_evaluator_too_lenient():
    run = _run([_result("t1", "faithfulness", True)])  # evaluator says pass
    labels = [_label("t1", "faithfulness", False)]  # human says fail
    report = compute_calibration(run, labels)
    assert report.false_positive_test_ids == ["t1"]
    assert report.false_negative_test_ids == []
    assert report.agreement_rate == 0.0


def test_false_negative_evaluator_too_strict():
    run = _run([_result("t1", "faithfulness", False)])  # evaluator says fail
    labels = [_label("t1", "faithfulness", True)]  # human says pass
    report = compute_calibration(run, labels)
    assert report.false_negative_test_ids == ["t1"]
    assert report.false_positive_test_ids == []


def test_per_evaluator_breakdown():
    run = _run([
        _result("t1", "faithfulness", True), _result("t2", "faithfulness", True),
        _result("t1", "exact_match", False),
    ])
    labels = [
        _label("t1", "faithfulness", False),  # disagree -- false positive
        _label("t2", "faithfulness", True),   # agree
        _label("t1", "exact_match", False),   # agree
    ]
    report = compute_calibration(run, labels)
    assert report.per_evaluator["faithfulness"].agreement_rate == 0.5
    assert report.per_evaluator["faithfulness"].false_positive_test_ids == ["t1"]
    assert report.per_evaluator["exact_match"].agreement_rate == 1.0


def test_label_for_missing_result_raises():
    run = _run([_result("t1", "faithfulness", True)])
    with pytest.raises(ValueError, match="No result for"):
        compute_calibration(run, [_label("t99", "faithfulness", True)])


def test_label_for_wrong_run_raises():
    run = _run([_result("t1", "faithfulness", True)])
    with pytest.raises(ValueError, match="not 'r1'"):
        compute_calibration(run, [_label("t1", "faithfulness", True, run_id="other-run")])


def test_no_labels_raises():
    run = _run([_result("t1", "faithfulness", True)])
    with pytest.raises(ValueError, match="zero human labels"):
        compute_calibration(run, [])


# ---- select_sample_for_review ----

def test_sample_is_deterministic_for_same_seed():
    run = _run([_result(f"t{i}", "x", True) for i in range(20)])
    sample1 = select_sample_for_review(run, 5, seed=42)
    sample2 = select_sample_for_review(run, 5, seed=42)
    assert [r.test_id for r in sample1] == [r.test_id for r in sample2]


def test_different_seeds_can_give_different_samples():
    run = _run([_result(f"t{i}", "x", True) for i in range(20)])
    sample1 = select_sample_for_review(run, 5, seed=1)
    sample2 = select_sample_for_review(run, 5, seed=2)
    assert [r.test_id for r in sample1] != [r.test_id for r in sample2]


def test_sample_size_too_large_raises():
    run = _run([_result("t1", "x", True)])
    with pytest.raises(ValueError, match="only 1"):
        select_sample_for_review(run, 5)


def test_sample_size_zero_or_negative_raises():
    run = _run([_result("t1", "x", True)])
    with pytest.raises(ValueError, match="positive"):
        select_sample_for_review(run, 0)
