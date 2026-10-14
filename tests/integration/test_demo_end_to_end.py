"""
Day 20: exercises loader + adapter + runner + comparison + regression +
quality_gate + efficiency + storage + observability together, via
`scripts/demo_end_to_end.py`'s own `build_demo_report()` -- the same
cross-module integration `tests/integration/README.md` has described
since the runner first existed, finally populated now that there's a
worked story to test. Zero paid API calls: everything runs against
`FakeHttpClient` and `FakeEmbeddingProvider`, same as every other test in
this suite.

Deliberately asserts on the SPECIFIC narrative `demo_end_to_end.py`
scripts (qa-009/qa-013 newly passing, qa-019 regressed, gate failing
despite the candidate's higher pass rate) rather than just "it runs
without raising" -- a demo whose own numbers silently changed out from
under it would be a worse regression than any of the ones it's
demonstrating how to catch.
"""

import shutil

import pytest

from eval_harness.comparison import ComparisonReport
from eval_harness.quality_gate import QualityGateResult
from eval_harness.regression import RegressionReport

from scripts.demo_end_to_end import DEMO_STORE_PATH, build_demo_dataset, build_demo_report


@pytest.fixture(autouse=True)
def _clean_demo_store():
    # The demo writes real files under demo_runs/ (RunStore, Day 15) --
    # clean up before and after so repeated test runs don't accumulate
    # runs or depend on a previous run's leftovers.
    shutil.rmtree(DEMO_STORE_PATH, ignore_errors=True)
    yield
    shutil.rmtree(DEMO_STORE_PATH, ignore_errors=True)


def test_demo_dataset_excludes_cases_with_no_expected_output():
    dataset = build_demo_dataset()
    assert all(tc.expected_output is not None for tc in dataset.test_cases)
    assert {"qa-002", "qa-007", "qa-017"}.isdisjoint({tc.test_id for tc in dataset.test_cases})


async def test_demo_report_tells_the_scripted_regression_story():
    report = await build_demo_report()

    comparison: ComparisonReport = report["comparison"]
    assert comparison.overall_winner == "production-rag-candidate"
    assert comparison.overall_pass_rate["production-rag-candidate"] > comparison.overall_pass_rate["production-rag-baseline"]

    regression: RegressionReport = report["regression"]
    assert regression.newly_passing_test_ids == ["qa-009", "qa-013"]
    assert regression.regressed_test_ids == ["qa-019"]
    assert regression.passed is False  # strict default policy: any regression fails

    gate: QualityGateResult = report["gate"]
    # The whole point of the demo: the candidate has a HIGHER pass rate
    # and average score than the baseline, yet the gate still fails --
    # because it regressed a previously-passing test case, and a quality
    # gate wired to a regression report (Day 11) catches that even when
    # every static threshold on the run itself would have passed.
    assert gate.passed is False
    assert any("regression" in v for v in gate.violations)


async def test_demo_report_persists_both_runs_and_latest_lookup_works():
    report = await build_demo_report()
    assert report["stored_baseline_run_id"] == report["baseline_run"].run_id

    from eval_harness.storage import RunStore

    store = RunStore(DEMO_STORE_PATH)
    baseline_runs = store.list_runs(report["dataset"].dataset_id, "production-rag-baseline")
    candidate_runs = store.list_runs(report["dataset"].dataset_id, "production-rag-candidate")
    assert len(baseline_runs) == 1
    assert len(candidate_runs) == 1


async def test_demo_report_cost_ranking_includes_both_systems():
    report = await build_demo_report()
    names = {eff.system_name for eff in report["cost_ranking"]}
    assert names == {"production-rag-baseline", "production-rag-candidate"}
    # Every scripted response passed at least one test case, so neither
    # run has zero passing tests -- cost_per_passing_test is a real number
    # for both, not the "undefined" None case efficiency.py documents.
    assert all(eff.cost_per_passing_test is not None for eff in report["cost_ranking"])


async def test_demo_report_observed_run_matches_candidate_scripted_story():
    report = await build_demo_report()
    observed = report["observed_run"]
    candidate = report["candidate_run"]
    # Same scripted responses as the candidate run above -- the observer
    # (Day 16) is a side channel and must not have changed the outcome.
    assert observed.pass_rate == candidate.pass_rate
    assert observed.average_score == candidate.average_score
    assert observed.system_name == candidate.system_name

    # This demo's evaluators (semantic_similarity, citation_precision)
    # never set their own estimated_cost -- only an LLM judge (Day 8)
    # does -- so slowest_results has entries but costliest_results is
    # correctly empty, not a bug. The console script explains this; the
    # test pins it so a future evaluator change can't silently flip it.
    assert all(hasattr(r, "test_id") for r in report["slowest_results"])
    assert report["slowest_results"] != []
    assert report["costliest_results"] == []
