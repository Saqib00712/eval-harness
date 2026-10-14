"""
Day 20 (Phase 32): worked end-to-end demonstration -- the payoff.

Every prior day built one piece in isolation, each proven correct by its
own unit tests. This script is the first time they're wired together the
way a real CI pipeline would use them, against Project 1
(`production-rag`) via Day 19's `ProductionRagAdapter` -- the story this
whole harness exists to tell:

    1. Load the shared golden dataset (Day 31's sample_eval_v1.json).
    2. Run it through TWO versions of production-rag (Day 12's
       `compare_systems`) -- "baseline" (what's deployed today) and
       "candidate" (a proposed change).
    3. Check the candidate for regressions against the baseline (Day 10's
       `compare_runs`), not just a raw pass-rate comparison.
    4. Turn the candidate's run + regression report into one hard
       pass/fail verdict (Day 11's `evaluate_quality_gate`) -- the kind a
       CI job would act on with `sys.exit(exit_code_for(result))`.
    5. Rank both versions by cost efficiency (Day 13).
    6. Persist both runs (Day 15's `RunStore`) so tomorrow's run has a
       baseline to compare against without this script needing to be run
       twice in the same process.
    7. Re-run once more with an observer attached (Day 16) to show a run
       being watched without changing its result, then surface the
       slowest and most expensive individual results from it.

No live production-rag deployment is required to run this: both "systems"
are `ProductionRagAdapter`s backed by `FakeHttpClient` with SCRIPTED
responses standing in for two real deployments -- see `_response_for`
below for exactly what each one is scripted to say and why. Point
`ProductionRagAdapter` at `HttpxHttpClient()` and a real `base_url`/
`api_key` instead (see docs/day19.md) to run this exact same script
against a real deployment; nothing else in this file would need to change,
which is the point of Day 19's adapter existing in the first place.

Run it:

    python scripts/demo_end_to_end.py

`build_demo_report()` is the part worth importing elsewhere (and what
tests/integration/test_demo_end_to_end.py asserts against) -- `main()` is
just console formatting on top of it.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from eval_harness.adapters.production_rag import ProductionRagAdapter  # noqa: E402
from eval_harness.comparison import ComparisonReport, compare_systems  # noqa: E402
from eval_harness.dataset import Dataset  # noqa: E402
from eval_harness.dataset_io import load_dataset  # noqa: E402
from eval_harness.efficiency import EfficiencyReport, compute_efficiency, rank_by_cost_per_passing_test  # noqa: E402
from eval_harness.evaluators.rag import CitationPrecisionEvaluator  # noqa: E402
from eval_harness.evaluators.semantic import SemanticSimilarityEvaluator  # noqa: E402
from eval_harness.observability import StructuredLogObserver, most_expensive_results, slowest_results  # noqa: E402
from eval_harness.providers.embedding import FakeEmbeddingProvider  # noqa: E402
from eval_harness.providers.http import FakeHttpClient  # noqa: E402
from eval_harness.quality_gate import QualityGatePolicy, QualityGateResult, evaluate_quality_gate  # noqa: E402
from eval_harness.regression import RegressionPolicy, RegressionReport, compare_runs  # noqa: E402
from eval_harness.run_models import EvaluationRun  # noqa: E402
from eval_harness.runner import run_evaluation  # noqa: E402
from eval_harness.storage import RunStore  # noqa: E402

DATASET_PATH = ROOT / "datasets" / "sample_eval_v1.json"
DEMO_STORE_PATH = ROOT / "demo_runs"

# Three deliberate scripted differences between "baseline" and "candidate"
# -- everything else, both versions answer correctly. This is what turns
# the demo from a flat "both passed" into the actual point of Days 10/11:
# an aggregate number can hide a regression just as easily as it can flag
# a false alarm.
#   qa-009, qa-013: baseline gets these WRONG, candidate FIXES them.
#   qa-019:         candidate introduces a NEW wrong answer here -- a real
#                    regression a pure "pass rate went up" read would miss.
SCRIPTED_OVERRIDES = {
    "qa-009": {
        "baseline": {"answer": "Our pricing is competitive and affordable.", "page_number": 1},
    },
    "qa-013": {
        "baseline": {
            "answer": "Yes, Enterprise customers get an extended refund window.",
            "page_number": 1,
        },
    },
    "qa-019": {
        "candidate": {"answer": "99.5%", "page_number": 6},
    },
}


def build_demo_dataset() -> Dataset:
    """The shared golden set (Day 31), narrowed to the test cases that have
    a reference `expected_output` -- `SemanticSimilarityEvaluator` needs
    one to compare against and raises `ValueError` for a test case without
    one (by design -- see its own module docstring). The three excluded
    cases (qa-002, qa-007, qa-017) are refusal/out-of-scope questions with
    no single correct answer string; evaluating those properly needs a
    refusal-aware evaluator this demo doesn't build, to keep the story
    focused on correctness + citation evaluation.

    Phase 38's adapter correction revealed production-rag's REAL citation
    field is `page_number`, not the generic `page` key this shared Day 31
    dataset was authored with. `expected_citations` is remapped to
    `page_number` here so `CitationPrecisionEvaluator`/
    `CitationRecallEvaluator` (both configured with
    `match_fields=["page_number"]` below) can actually match
    `ProductionRagAdapter`'s real citation shape -- the dataset itself
    stays untouched on disk; this is a per-demo view of it."""
    full = load_dataset(DATASET_PATH)
    cases = []
    for tc in full.test_cases:
        if tc.expected_output is None:
            continue
        remapped = [{"page_number": c["page"]} for c in tc.expected_citations if "page" in c]
        cases.append(tc.model_copy(update={"expected_citations": remapped}))
    return Dataset(
        dataset_id=full.dataset_id,
        name=full.name,
        version=full.version,
        description=(
            f"{full.description} (Day 20 demo subset: {len(cases)} of "
            f"{full.test_count} cases with a reference answer; "
            f"expected_citations remapped page -> page_number, Phase 38.)"
        ),
        test_cases=cases,
    )


def _response_for(test_case, mode: str) -> dict:
    """Builds a response in production-rag's REAL `AskResponse` shape
    (Phase 38's correction -- nested `cost.total_cost_usd`, real `Citation`
    objects keyed by `page_number`/`snippet`, `models` as a per-stage dict)
    rather than the flatter, reconstructed shape this demo originally used
    under Day 19's guessed contract."""
    override = SCRIPTED_OVERRIDES.get(test_case.test_id, {}).get(mode)
    answer = override["answer"] if override else test_case.expected_output
    page_number = override["page_number"] if override else (
        test_case.expected_citations[0]["page_number"] if test_case.expected_citations else None
    )
    citations = []
    if page_number is not None:
        citations = [{
            "source_id": "S1", "chunk_id": f"{test_case.test_id}-c1", "document_id": "technova-handbook",
            "page_number": page_number, "snippet": test_case.reference_context[0] if test_case.reference_context else answer,
            "rerank_score": 0.91,
        }]
    generation_prompt_tokens = 180 + len(test_case.input)
    generation_completion_tokens = max(5, len(answer or "") // 4)
    return {
        "answer": answer,
        "citations": citations,
        "insufficient_evidence": False,
        "cost": {
            "embedding_tokens": 12,
            "generation_prompt_tokens": generation_prompt_tokens,
            "generation_completion_tokens": generation_completion_tokens,
            "total_cost_usd": round(0.00025 + 0.0000015 * len(answer or ""), 6),
        },
        "models": {"generation": "gpt-4o-mini" if mode == "baseline" else "gpt-4o-mini-ft-v2"},
        "timings_ms": {"retrieval": 40.0, "generation": 650.0},
    }


def _fake_http_for(dataset: Dataset, mode: str) -> FakeHttpClient:
    """A FakeHttpClient scripted to answer each test case in dataset order,
    standing in for one production-rag deployment ('baseline' or
    'candidate'). Swap this for `HttpxHttpClient()` and a real
    `base_url`/`api_key` to run this same demo against a live
    deployment -- see this module's own docstring."""
    return FakeHttpClient(queue=[(200, _response_for(tc, mode)) for tc in dataset.test_cases])


def build_evaluators() -> list:
    # match_fields=["page_number"] -- production-rag's real Citation object
    # (Phase 38) carries source_id/chunk_id/document_id/snippet/
    # rerank_score alongside page_number, none of which the (deliberately
    # generic) golden dataset's expected_citations tries to predict; only
    # page_number is compared.
    return [
        SemanticSimilarityEvaluator(FakeEmbeddingProvider(), threshold=0.90),
        CitationPrecisionEvaluator(match_fields=["page_number"]),
    ]


async def build_demo_report() -> dict:
    dataset = build_demo_dataset()
    evaluators = build_evaluators()

    baseline = ProductionRagAdapter(
        _fake_http_for(dataset, "baseline"), "http://localhost:8000", "demo-key",
        name="production-rag-baseline", version="v1.0-gpt-4o-mini",
    )
    candidate = ProductionRagAdapter(
        _fake_http_for(dataset, "candidate"), "http://localhost:8000", "demo-key",
        name="production-rag-candidate", version="v1.1-gpt-4o-mini-ft",
    )

    comparison: ComparisonReport = await compare_systems(dataset, [baseline, candidate], evaluators)
    baseline_run: EvaluationRun = comparison.runs[baseline.name]
    candidate_run: EvaluationRun = comparison.runs[candidate.name]

    regression: RegressionReport = compare_runs(baseline_run, candidate_run, RegressionPolicy())

    gate_policy = QualityGatePolicy(min_pass_rate=0.85, min_average_score=0.85)
    gate: QualityGateResult = evaluate_quality_gate(candidate_run, gate_policy, regression)

    baseline_efficiency: EfficiencyReport = compute_efficiency(baseline_run)
    candidate_efficiency: EfficiencyReport = compute_efficiency(candidate_run)
    cost_ranking = rank_by_cost_per_passing_test([baseline_run, candidate_run])

    store = RunStore(DEMO_STORE_PATH)
    store.save(baseline_run)
    store.save(candidate_run)
    # The pattern a real CI job follows the NEXT time it runs: look up
    # yesterday's run instead of needing it passed in by hand.
    stored_baseline = store.latest(dataset.dataset_id, baseline.name)

    # A separate run (Day 16's observer is a pure side channel -- see
    # test_observer_hooks_fire_and_never_affect_the_result, Day 16): shows
    # a run being watched without needing to touch the comparison above.
    observer = StructuredLogObserver(logging.getLogger("eval_harness.demo"))
    observed_run = await run_evaluation(
        dataset,
        ProductionRagAdapter(
            _fake_http_for(dataset, "candidate"), "http://localhost:8000", "demo-key",
            name="production-rag-candidate", version="v1.1-gpt-4o-mini-ft",
        ),
        evaluators,
        observer=observer,
    )

    return {
        "dataset": dataset,
        "comparison": comparison,
        "baseline_run": baseline_run,
        "candidate_run": candidate_run,
        "regression": regression,
        "gate": gate,
        "baseline_efficiency": baseline_efficiency,
        "candidate_efficiency": candidate_efficiency,
        "cost_ranking": cost_ranking,
        "stored_baseline_run_id": stored_baseline.run_id if stored_baseline else None,
        "observed_run": observed_run,
        "slowest_results": slowest_results(observed_run, n=3),
        "costliest_results": most_expensive_results(observed_run, n=3),
    }


def _print_report(report: dict) -> None:
    comparison: ComparisonReport = report["comparison"]
    regression: RegressionReport = report["regression"]
    gate: QualityGateResult = report["gate"]

    print(f"Dataset: {report['dataset'].name} ({report['dataset'].test_count} test cases)\n")

    print("== Comparison (Day 12) ==")
    for name in comparison.system_names:
        print(
            f"  {name:<24} pass_rate={comparison.overall_pass_rate[name]:.4f}  "
            f"avg_score={comparison.overall_average_score[name]:.4f}"
        )
    print(f"  Overall winner: {comparison.overall_winner or 'tie'}")
    for evaluator_name, winner in comparison.per_evaluator_winner.items():
        print(f"  {evaluator_name} winner: {winner or 'tie'}")

    print("\n== Regression check, baseline -> candidate (Day 10) ==")
    print(f"  pass_rate_delta={regression.pass_rate_delta:+.4f}  average_score_delta={regression.average_score_delta:+.4f}")
    print(f"  newly passing: {regression.newly_passing_test_ids}")
    print(f"  regressed:     {regression.regressed_test_ids}")
    print(f"  Regression check passed: {regression.passed}")
    for v in regression.violations:
        print(f"    - {v}")

    print("\n== Quality gate on the candidate (Day 11) ==")
    print(f"  GATE {'PASSED' if gate.passed else 'FAILED'}")
    for v in gate.violations:
        print(f"    - {v}")

    print("\n== Cost efficiency (Day 13) ==")
    for eff in report["cost_ranking"]:
        print(f"  {eff.system_name}: cost_per_passing_test={eff.cost_per_passing_test}")

    print("\n== Storage (Day 15) ==")
    print(f"  Saved both runs under {DEMO_STORE_PATH}")
    print(f"  RunStore.latest() baseline lookup -> run_id {report['stored_baseline_run_id']}")

    print("\n== Observability (Day 16) ==")
    print(f"  Slowest results (by per-evaluator latency): {[r.test_id for r in report['slowest_results']]}")
    costliest = [r.test_id for r in report["costliest_results"]]
    note = " (none -- this demo's evaluators don't report their own cost; only an LLM judge, Day 8, does)" if not costliest else ""
    print(f"  Costliest results (by per-evaluator cost): {costliest}{note}")

    print(f"\nEXIT CODE this would give a CI job: {0 if gate.passed else 1}")


async def main() -> None:
    report = await build_demo_report()
    _print_report(report)


if __name__ == "__main__":
    asyncio.run(main())
