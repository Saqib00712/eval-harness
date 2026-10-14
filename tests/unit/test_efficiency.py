import pytest

from eval_harness.efficiency import compute_efficiency, rank_by_cost_per_passing_test
from eval_harness.run_models import EvaluationRun

NOW = "2026-01-01T00:00:00Z"


def _run(run_id, system_name="sys", total_tests=4, passed_tests=2, total_cost=1.0, total_tokens=1000,
         avg_latency=None, max_latency=None) -> EvaluationRun:
    return EvaluationRun(
        run_id=run_id, dataset_id="d1", dataset_version="1.0.0", dataset_content_hash="h1",
        system_name=system_name, system_version="0.0.0",
        started_at=NOW, completed_at=NOW, status="completed",
        total_tests=total_tests, passed_tests=passed_tests, failed_tests=total_tests - passed_tests,
        pass_rate=passed_tests / total_tests if total_tests else 0.0, average_score=0.5,
        total_tokens=total_tokens, estimated_cost=total_cost,
        average_system_latency_ms=avg_latency, max_system_latency_ms=max_latency,
    )


def test_cost_and_tokens_per_test():
    report = compute_efficiency(_run("r1", total_tests=4, total_cost=1.0, total_tokens=1000))
    assert report.cost_per_test == 0.25
    assert report.tokens_per_test == 250.0


def test_cost_and_tokens_per_passing_test_divides_by_passed_not_total():
    report = compute_efficiency(_run("r1", total_tests=4, passed_tests=2, total_cost=1.0, total_tokens=1000))
    assert report.cost_per_passing_test == 0.5  # 1.0 / 2, not 1.0 / 4
    assert report.tokens_per_passing_test == 500.0


def test_zero_passing_tests_is_none_not_zero_or_infinite():
    report = compute_efficiency(_run("r1", total_tests=4, passed_tests=0, total_cost=1.0))
    assert report.cost_per_passing_test is None
    assert report.tokens_per_passing_test is None
    # total/per-test metrics are still well-defined even with zero passes
    assert report.cost_per_test == 0.25


def test_zero_total_tests_raises():
    run = _run("r1", total_tests=0, passed_tests=0)
    with pytest.raises(ValueError, match="zero test cases"):
        compute_efficiency(run)


def test_latency_fields_pass_through():
    report = compute_efficiency(_run("r1", avg_latency=120.5, max_latency=400.0))
    assert report.average_system_latency_ms == 120.5
    assert report.max_system_latency_ms == 400.0


def test_latency_none_when_run_never_reported_it():
    report = compute_efficiency(_run("r1"))
    assert report.average_system_latency_ms is None
    assert report.max_system_latency_ms is None


def test_rank_orders_cheapest_per_success_first():
    cheap = _run("r1", system_name="cheap", total_tests=10, passed_tests=10, total_cost=1.0)   # 0.10/pass
    pricey = _run("r2", system_name="pricey", total_tests=10, passed_tests=10, total_cost=5.0)  # 0.50/pass
    ranking = rank_by_cost_per_passing_test([pricey, cheap])
    assert [r.system_name for r in ranking] == ["cheap", "pricey"]


def test_rank_puts_zero_passing_runs_last_even_if_cheapest_overall():
    free_but_useless = _run("r1", system_name="useless", total_tests=10, passed_tests=0, total_cost=0.01)
    expensive_but_works = _run("r2", system_name="works", total_tests=10, passed_tests=10, total_cost=5.0)
    ranking = rank_by_cost_per_passing_test([free_but_useless, expensive_but_works])
    assert [r.system_name for r in ranking] == ["works", "useless"]
