"""
Phase 39: failure injection and regression verification.

Fifteen deterministic, synthetic scenarios, each proving the harness
detects a specific kind of degraded AI output -- or a specific kind of
harness-level failure -- and classifies it correctly. Every evaluator
call here is either a deterministic/embedding evaluator or a
`FakeJsonLLM`-mocked judge; nothing makes a real, paid LLM call, and
nothing here is nondeterministic.

These are SYNTHETIC demonstrations of the harness's own detection
machinery, not claims about any real production model's behavior.

The baseline-vs-candidate regression demonstration this phase also asks
for already exists and is asserted on in
`tests/integration/test_demo_end_to_end.py` (Day 20) -- it is not
duplicated here; see that file for the full comparison/regression/
quality-gate story (a candidate that wins on every aggregate metric still
fails the quality gate because it regressed one previously-passing case).
"""

from eval_harness.adapters.production_rag import ProductionRagAdapter
from eval_harness.dataset import Dataset
from eval_harness.evaluators.deterministic import JSONSchemaEvaluator
from eval_harness.evaluators.llm_judge import FaithfulnessJudgeEvaluator
from eval_harness.evaluators.rag import (
    CitationPrecisionEvaluator,
    CitationRecallEvaluator,
    HitAtKEvaluator,
)
from eval_harness.evaluators.semantic import SemanticSimilarityEvaluator
from eval_harness.failure_taxonomy import FailureCategory
from eval_harness.models import TestCase
from eval_harness.providers.embedding import FakeEmbeddingProvider
from eval_harness.providers.http import FakeHttpClient, HttpError
from eval_harness.providers.llm import FakeJsonLLM
from eval_harness.quality_gate import QualityGatePolicy, evaluate_quality_gate, exit_code_for
from eval_harness.runner import run_evaluation


def _tc(**kwargs) -> TestCase:
    defaults = dict(test_id="t1", input="What is the refund policy?")
    defaults.update(kwargs)
    return TestCase(**defaults)


def _dataset(*cases) -> Dataset:
    return Dataset(dataset_id="failure-injection", name="Failure injection", version="1.0.0", test_cases=list(cases))


def _adapter(response: dict | None = None, *, raises: Exception | None = None, status: int = 200) -> ProductionRagAdapter:
    http = FakeHttpClient(raises=raises) if raises else FakeHttpClient(default=(status, response or {}))
    return ProductionRagAdapter(http, "http://localhost:8000", "test-key")


# ---- 1. Correct answer with valid citations ----

async def test_01_correct_answer_with_valid_citations_passes_cleanly():
    tc = _tc(
        expected_output="Refunds within 30 days.",
        reference_context=["Refunds are available within 30 days of purchase."],
        expected_citations=[{"page_number": 4}],
    )
    adapter = _adapter({
        "answer": "Refunds are available within 30 days.",
        "citations": [{"source_id": "S1", "chunk_id": "c1", "document_id": "d1", "page_number": 4,
                        "snippet": "Refunds are available within 30 days of purchase."}],
    })
    run = await run_evaluation(
        _dataset(tc), adapter,
        [SemanticSimilarityEvaluator(FakeEmbeddingProvider(), threshold=0.5),
         CitationPrecisionEvaluator(match_fields=["page_number"])],
    )
    assert run.pass_rate == 1.0
    assert all(r.failure_category is None for r in run.results)
    gate = evaluate_quality_gate(run, QualityGatePolicy(min_pass_rate=0.9))
    assert gate.passed is True
    assert exit_code_for(gate) == 0


# ---- 2. Incorrect answer ----

async def test_02_incorrect_answer_fails_correctness_as_threshold_miss():
    tc = _tc(expected_output="Refunds are available within 30 days of purchase.")
    adapter = _adapter({"answer": "We do not offer refunds under any circumstances."})
    run = await run_evaluation(_dataset(tc), adapter, [SemanticSimilarityEvaluator(FakeEmbeddingProvider(), threshold=0.5)])
    result = run.results[0]
    assert result.passed is False
    assert result.failure_category == FailureCategory.THRESHOLD_MISS
    assert run.pass_rate == 0.0


# ---- 3. Partially correct answer with missing information ----

async def test_03_partially_correct_answer_scores_below_full_credit():
    tc = _tc(expected_output="Refunds are available within 30 days, and shipping costs are non-refundable.")
    adapter = _adapter({"answer": "Refunds are available within 30 days."})  # misses the shipping-cost clause
    run = await run_evaluation(
        _dataset(tc), adapter, [SemanticSimilarityEvaluator(FakeEmbeddingProvider(), threshold=0.95)]
    )
    result = run.results[0]
    # Partial overlap scores somewhere above 0 but below a strict
    # threshold -- neither "fully correct" nor "fully wrong".
    assert 0.0 < result.score < 1.0
    assert result.passed is False
    assert result.failure_category == FailureCategory.THRESHOLD_MISS


# ---- 4. Hallucinated / unsupported factual claims ----

async def test_04_hallucinated_claim_fails_faithfulness_judge_while_correctness_passes():
    """A claim that happens to read as plausible can still pass a
    similarity-based correctness check while failing a faithfulness check
    against the actual retrieved context -- this is exactly why Phase 38
    requires BOTH, not one standing in for the other."""
    tc = _tc(expected_output="Refunds are available within 30 days.")
    adapter = _adapter({
        "answer": "Refunds are available within 30 days, and we also offer free lifetime warranty on all products.",
        "citations": [{"source_id": "S1", "chunk_id": "c1", "document_id": "d1", "page_number": 4,
                        "snippet": "Refunds are available within 30 days of purchase."}],
    })
    judge = FakeJsonLLM(default={"score": 2, "reasoning": "the lifetime warranty claim is not supported by the cited context"})
    run = await run_evaluation(
        _dataset(tc), adapter,
        [SemanticSimilarityEvaluator(FakeEmbeddingProvider(), threshold=0.5), FaithfulnessJudgeEvaluator(judge, "gpt-test")],
    )
    by_evaluator = {r.evaluator: r for r in run.results}
    assert by_evaluator["semantic_similarity"].passed is True
    assert by_evaluator["faithfulness_judge"].passed is False
    assert by_evaluator["faithfulness_judge"].failure_category == FailureCategory.THRESHOLD_MISS


# ---- 5. Irrelevant answer ----

async def test_05_irrelevant_answer_fails_correctness():
    tc = _tc(expected_output="Refunds are available within thirty days of purchase.")
    adapter = _adapter({"answer": "Mount Everest is the tallest mountain on planet Earth."})
    run = await run_evaluation(_dataset(tc), adapter, [SemanticSimilarityEvaluator(FakeEmbeddingProvider(), threshold=0.5)])
    assert run.results[0].passed is False
    assert run.pass_rate == 0.0


# ---- 6. Missing citations ----

async def test_06_missing_citations_fails_recall_but_vacuously_passes_precision():
    """CitationRecallEvaluator (citations that SHOULD have been given) and
    CitationPrecisionEvaluator (citations that WERE given) diverge
    exactly as designed when none are given at all: nothing given means
    nothing can be wrong (precision=1.0, vacuous) but something expected
    is missing (recall=0.0)."""
    tc = _tc(expected_citations=[{"page_number": 4}])
    adapter = _adapter({"answer": "Refunds are available within 30 days."})  # no citations field at all
    run = await run_evaluation(
        _dataset(tc), adapter,
        [CitationPrecisionEvaluator(match_fields=["page_number"]), CitationRecallEvaluator(match_fields=["page_number"])],
    )
    by_evaluator = {r.evaluator: r for r in run.results}
    assert by_evaluator["citation_precision"].score == 1.0
    assert by_evaluator["citation_precision"].passed is True
    assert by_evaluator["citation_recall"].score == 0.0
    assert by_evaluator["citation_recall"].passed is False


# ---- 7. Invalid or fabricated citation references ----

async def test_07_fabricated_citation_fails_precision():
    tc = _tc(expected_citations=[{"page_number": 4}])
    adapter = _adapter({
        "answer": "Refunds are available within 30 days.",
        # page 99 was never a real expected source -- a fabricated/wrong reference.
        "citations": [{"source_id": "S9", "chunk_id": "c9", "document_id": "d1", "page_number": 99, "snippet": "unrelated text"}],
    })
    run = await run_evaluation(_dataset(tc), adapter, [CitationPrecisionEvaluator(match_fields=["page_number"])])
    result = run.results[0]
    assert result.score == 0.0
    assert result.passed is False


# ---- 8. Citations that exist but do not support the claims ----

async def test_08_citation_matches_but_faithfulness_judge_catches_unsupported_claim():
    """The citation IS one of the expected ones (precision passes) -- but
    the answer makes a claim the cited snippet doesn't actually support.
    Dict-equality/key-matching citation checks structurally cannot catch
    this (they only check WHICH source was cited, never whether the
    claim is supported by it) -- this is exactly the faithfulness judge's
    job, and exactly why Phase 38 lists citation correctness and
    faithfulness as separate requirements."""
    tc = _tc(expected_citations=[{"page_number": 4}])
    adapter = _adapter({
        "answer": "Refunds are available within 90 days, per the cited policy.",
        "citations": [{"source_id": "S1", "chunk_id": "c1", "document_id": "d1", "page_number": 4,
                        "snippet": "Refunds are available within 30 days of purchase."}],  # says 30, not 90
    })
    judge = FakeJsonLLM(default={"score": 1, "reasoning": "the cited snippet says 30 days, not 90 -- the claim is unsupported"})
    run = await run_evaluation(
        _dataset(tc), adapter,
        [CitationPrecisionEvaluator(match_fields=["page_number"]), FaithfulnessJudgeEvaluator(judge, "gpt-test")],
    )
    by_evaluator = {r.evaluator: r for r in run.results}
    assert by_evaluator["citation_precision"].passed is True  # the citation itself is a real, expected source
    assert by_evaluator["faithfulness_judge"].passed is False  # but the claim it's used for isn't supported


# ---- 9. Poor retrieval with relevant documents missing ----

async def test_09_poor_retrieval_fails_hit_at_k():
    tc = _tc(reference_context=["Refunds are available within 30 days of purchase."])
    adapter = _adapter({
        "answer": "I don't have information on that.",
        "citations": [{"source_id": "S9", "chunk_id": "c9", "document_id": "d1", "page_number": 1,
                        "snippet": "This document covers shipping logistics, not refunds."}],
    })
    run = await run_evaluation(_dataset(tc), adapter, [HitAtKEvaluator(k=5)])
    result = run.results[0]
    assert result.passed is False
    assert result.score == 0.0


# ---- 10. Malformed structured output ----

async def test_10_malformed_structured_output_fails_schema_validation_without_crashing():
    """A classifier-shaped system returning structured_data that doesn't
    match its schema -- JSONSchemaEvaluator degrades to a clean failure,
    never an unhandled exception."""
    tc = TestCase(test_id="t1", input="classify this", metadata={"json_schema": {
        "type": "object", "required": ["label"], "properties": {"label": {"type": "string"}},
    }})

    class _MalformedStructuredSystem:
        name = "malformed-classifier"
        version = "1.0"

        async def run(self, test_case):
            from eval_harness.system_output import SystemOutput
            return SystemOutput(structured_data={"wrong_key": 123})  # missing required "label"

    run = await run_evaluation(_dataset(tc), _MalformedStructuredSystem(), [JSONSchemaEvaluator()])
    result = run.results[0]
    assert result.passed is False
    assert result.failure_category == FailureCategory.THRESHOLD_MISS  # a clean failure, not a crash
    assert "Schema validation failed" in result.reason


# ---- 11. System timeout ----

async def test_11_system_timeout_is_a_dependency_failure_not_a_quality_failure():
    tc = _tc()
    adapter = _adapter(raises=HttpError("request timed out after 30s"))
    run = await run_evaluation(_dataset(tc), adapter, [SemanticSimilarityEvaluator(FakeEmbeddingProvider())])
    result = run.results[0]
    assert result.passed is False
    assert result.failure_category == FailureCategory.DEPENDENCY_FAILURE
    assert "timed out" in result.reason


# ---- 12. Evaluator failure ----

async def test_12_evaluator_failure_is_never_reported_as_a_successful_evaluation():
    """A broken/misconfigured evaluator (here: SemanticSimilarityEvaluator
    against a test case with no expected_output, which it explicitly
    raises ValueError for) must score as a clean EVALUATION_FAILURE --
    passed=False, never silently treated as passed, and never confused
    with the application itself producing a bad answer."""
    tc = TestCase(test_id="t1", input="x")  # no expected_output
    adapter = _adapter({"answer": "anything"})
    run = await run_evaluation(_dataset(tc), adapter, [SemanticSimilarityEvaluator(FakeEmbeddingProvider())])
    result = run.results[0]
    assert result.passed is False  # never mistaken for success
    assert result.failure_category == FailureCategory.EVALUATION_FAILURE
    assert result.failure_category != FailureCategory.DEPENDENCY_FAILURE  # distinguishable from an infra failure
    assert result.failure_category != FailureCategory.THRESHOLD_MISS  # distinguishable from poor quality


# ---- 13. Infrastructure / dependency failure, distinguished from poor model quality ----

async def test_13_dependency_failure_is_distinguishable_from_poor_quality_in_the_same_run():
    """Three test cases in ONE run: one where the system crashes outright
    (infra), one where the system answers but answers badly (quality),
    and one that succeeds cleanly -- their failure_category must all
    differ, proving the harness doesn't collapse 'the system is down'
    and 'the system gave a bad answer' into the same signal."""
    good_tc = _tc(test_id="good", expected_output="Refunds are available within 30 days.")
    bad_tc = _tc(test_id="bad", expected_output="Refunds are available within 30 days.")
    down_tc = _tc(test_id="down", expected_output="Refunds are available within 30 days.")

    class _MixedSystem:
        name = "mixed"
        version = "1.0"

        async def run(self, test_case):
            from eval_harness.system_output import SystemOutput
            if test_case.test_id == "down":
                raise ConnectionError("upstream unreachable")
            if test_case.test_id == "bad":
                return SystemOutput(output="We do not offer refunds.")
            return SystemOutput(output="Refunds are available within 30 days.")

    run = await run_evaluation(
        _dataset(good_tc, bad_tc, down_tc), _MixedSystem(),
        [SemanticSimilarityEvaluator(FakeEmbeddingProvider(), threshold=0.5)],
    )
    by_test = {r.test_id: r for r in run.results}
    assert by_test["good"].failure_category is None
    assert by_test["bad"].failure_category == FailureCategory.THRESHOLD_MISS
    assert by_test["down"].failure_category == FailureCategory.DEPENDENCY_FAILURE
    assert len({by_test["good"].failure_category, by_test["bad"].failure_category, by_test["down"].failure_category}) == 3


# ---- 14. Increased latency ----

async def test_14_increased_latency_is_observable_in_run_aggregates():
    """Latency itself is observable (average_system_latency_ms) even
    though no quality-gate policy field currently gates on it -- see
    docs/ENGINEERING_REPORT.md's production-readiness section for this
    documented gap (NEEDS IMPROVEMENT: no latency-based gate exists)."""
    import asyncio

    tc = _tc(expected_output="Refunds are available within 30 days.")

    class _SlowSystem:
        name = "slow"
        version = "1.0"

        async def run(self, test_case):
            from eval_harness.system_output import SystemOutput
            await asyncio.sleep(0.02)
            return SystemOutput(output="Refunds are available within 30 days.", latency_ms=2000.0)

    class _FastSystem:
        name = "fast"
        version = "1.0"

        async def run(self, test_case):
            from eval_harness.system_output import SystemOutput
            return SystemOutput(output="Refunds are available within 30 days.", latency_ms=50.0)

    evaluators = [SemanticSimilarityEvaluator(FakeEmbeddingProvider(), threshold=0.5)]
    fast_run = await run_evaluation(_dataset(tc), _FastSystem(), evaluators)
    slow_run = await run_evaluation(_dataset(tc), _SlowSystem(), evaluators)

    assert fast_run.average_system_latency_ms == 50.0
    assert slow_run.average_system_latency_ms == 2000.0
    assert slow_run.average_system_latency_ms > fast_run.average_system_latency_ms
    # Both still pass on quality -- latency regressed, but nothing in
    # QualityGatePolicy currently gates on it (documented limitation).
    assert fast_run.pass_rate == slow_run.pass_rate == 1.0


# ---- 15. Increased token usage / cost, including when NOT available ----

async def test_15_increased_cost_is_observable_via_efficiency_report():
    from eval_harness.efficiency import compute_efficiency

    tc = _tc(expected_output="Refunds are available within 30 days.")
    cheap = _adapter({"answer": "Refunds are available within 30 days.", "cost": {"generation_prompt_tokens": 50, "generation_completion_tokens": 10, "total_cost_usd": 0.0001}})
    expensive = _adapter({"answer": "Refunds are available within 30 days.", "cost": {"generation_prompt_tokens": 500, "generation_completion_tokens": 100, "total_cost_usd": 0.01}})
    evaluators = [SemanticSimilarityEvaluator(FakeEmbeddingProvider(), threshold=0.5)]

    cheap_run = await run_evaluation(_dataset(tc), cheap, evaluators)
    expensive_run = await run_evaluation(_dataset(tc), expensive, evaluators)

    cheap_eff = compute_efficiency(cheap_run)
    expensive_eff = compute_efficiency(expensive_run)
    assert expensive_eff.total_cost > cheap_eff.total_cost
    assert expensive_eff.total_tokens > cheap_eff.total_tokens


async def test_15b_cost_not_reported_is_indistinguishable_from_free_at_the_run_level():
    """DOCUMENTED LIMITATION (Phase 40/41): unlike Day 13's
    cost_per_passing_test, which distinguishes 'undefined' (zero passing
    tests) from 'zero' (None vs 0.0), EvaluationRun.estimated_cost itself
    does NOT distinguish 'no system in this run ever reported a cost' from
    'every system in this run genuinely cost $0'. Both produce
    estimated_cost == 0.0. This test pins that behavior so it's a known,
    documented gap rather than a silent surprise."""
    tc = _tc(expected_output="Refunds are available within 30 days.")
    adapter = _adapter({"answer": "Refunds are available within 30 days."})  # no "cost" key at all
    run = await run_evaluation(_dataset(tc), adapter, [SemanticSimilarityEvaluator(FakeEmbeddingProvider(), threshold=0.5)])
    assert run.estimated_cost == 0.0  # indistinguishable from "genuinely free" -- see docs note above
    assert run.total_tokens == 0


# ---- Infrastructure failure also surfaces through ProductionRagAdapterError (non-2xx) ----

async def test_production_rag_http_error_status_is_also_a_dependency_failure():
    tc = _tc()
    adapter = _adapter({"detail": "Embedding provider failed; try again later."}, status=502)
    run = await run_evaluation(_dataset(tc), adapter, [SemanticSimilarityEvaluator(FakeEmbeddingProvider())])
    result = run.results[0]
    assert result.failure_category == FailureCategory.DEPENDENCY_FAILURE
    assert "ProductionRagAdapterError" in result.reason or "HTTP 502" in result.reason
