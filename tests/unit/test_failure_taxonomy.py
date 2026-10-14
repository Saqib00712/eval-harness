from eval_harness.failure_taxonomy import FailureCategory, summarize_failures
from eval_harness.run_models import EvaluationRun, PerTestResult


def _run(results: list[PerTestResult]) -> EvaluationRun:
    return EvaluationRun(
        run_id="r1", dataset_id="d", dataset_version="1.0.0", dataset_content_hash="hash",
        system_name="sys", system_version="0.0.0",
        started_at="2026-01-01T00:00:00Z", completed_at="2026-01-01T00:01:00Z", status="completed",
        total_tests=len({r.test_id for r in results}), passed_tests=0, failed_tests=0,
        pass_rate=0.0, average_score=0.0, total_tokens=0, estimated_cost=0.0,
        results=results,
    )


def test_summarize_excludes_passing_results():
    results = [
        PerTestResult(run_id="r1", test_id="t1", evaluator="exact_match", score=1.0, passed=True),
    ]
    assert summarize_failures(_run(results)) == {}


def test_summarize_groups_by_category_and_evaluator():
    results = [
        PerTestResult(run_id="r1", test_id="t1", evaluator="faithfulness", score=0.2, passed=False,
                      failure_category=FailureCategory.THRESHOLD_MISS),
        PerTestResult(run_id="r1", test_id="t2", evaluator="faithfulness", score=0.3, passed=False,
                      failure_category=FailureCategory.THRESHOLD_MISS),
        PerTestResult(run_id="r1", test_id="t3", evaluator="exact_match", score=0.0, passed=False,
                      failure_category=FailureCategory.THRESHOLD_MISS),
        PerTestResult(run_id="r1", test_id="t4", evaluator="exact_match", score=0.0, passed=False,
                      failure_category=FailureCategory.DEPENDENCY_FAILURE),
    ]
    summary = summarize_failures(_run(results))

    assert summary[FailureCategory.THRESHOLD_MISS]["count"] == 3
    assert summary[FailureCategory.THRESHOLD_MISS]["by_evaluator"] == {"faithfulness": 2, "exact_match": 1}
    assert summary[FailureCategory.THRESHOLD_MISS]["test_ids"] == ["t1", "t2", "t3"]

    assert summary[FailureCategory.DEPENDENCY_FAILURE]["count"] == 1
    assert summary[FailureCategory.DEPENDENCY_FAILURE]["test_ids"] == ["t4"]


def test_summarize_labels_missing_category_as_uncategorized():
    results = [PerTestResult(run_id="r1", test_id="t1", evaluator="exact_match", score=0.0, passed=False)]
    summary = summarize_failures(_run(results))
    assert summary["uncategorized"]["count"] == 1


def test_all_lists_every_category():
    assert FailureCategory.ALL == ("dependency_failure", "evaluation_failure", "threshold_miss")
