"""
Phase 38 (originally Day 19): proves `ProductionRagAdapter` isn't just structurally shaped like a
`SystemUnderTest` (the Protocol's `isinstance` check only inspects
attribute names -- Day 18's audit found exactly this gap once already,
for an unrelated class) but actually works end-to-end through the real
`run_evaluation` pipeline (Day 5), with real evaluators, against a fake
HTTP transport standing in for a live production-rag deployment.
"""

from eval_harness.adapters.production_rag import ProductionRagAdapter
from eval_harness.dataset import Dataset
from eval_harness.evaluators.deterministic import ContainsEvaluator
from eval_harness.evaluators.llm_judge import FaithfulnessJudgeEvaluator
from eval_harness.evaluators.rag import CitationPrecisionEvaluator, CitationRecallEvaluator, HitAtKEvaluator
from eval_harness.models import TestCase
from eval_harness.providers.http import FakeHttpClient
from eval_harness.providers.llm import FakeJsonLLM
from eval_harness.runner import run_evaluation
from eval_harness.system import SystemUnderTest


def test_adapter_satisfies_the_system_under_test_protocol():
    adapter = ProductionRagAdapter(FakeHttpClient(default=(200, {})), "http://x", "key")
    assert isinstance(adapter, SystemUnderTest)


async def test_adapter_runs_through_the_real_evaluation_pipeline():
    dataset = Dataset(
        dataset_id="d1",
        name="production-rag smoke set",
        version="1.0.0",
        test_cases=[
            TestCase(
                test_id="t1", input="What is the refund policy?",
                expected_output="30 days", metadata={"required_terms": ["30 days"]},
            ),
            TestCase(
                test_id="t2", input="What is the warranty period?",
                expected_output="1 year", metadata={"required_terms": ["1 year"]},
            ),
        ],
    )
    http = FakeHttpClient(
        queue=[
            (200, {
                "answer": "Refunds are available within 30 days.",
                "citations": [{"source_id": "S1", "chunk_id": "c1", "document_id": "d1", "page_number": 4, "snippet": "..."}],
                "cost": {"total_cost_usd": 0.0004},
            }),
            (200, {
                "answer": "The warranty period is 1 year.",
                "citations": [{"source_id": "S1", "chunk_id": "c2", "document_id": "d1", "page_number": 9, "snippet": "..."}],
                "cost": {"total_cost_usd": 0.0003},
            }),
        ]
    )
    adapter = ProductionRagAdapter(http, "http://localhost:8000", "test-key", version="commit-abc123")

    run = await run_evaluation(dataset, adapter, [ContainsEvaluator()])

    assert run.system_name == "production-rag"
    assert run.system_version == "commit-abc123"
    assert run.total_tests == 2
    assert run.passed_tests == 2
    assert run.pass_rate == 1.0
    assert len(http.calls) == 2
    assert all(c["headers"]["X-API-Key"] == "test-key" for c in http.calls)


async def test_adapter_output_feeds_retrieval_and_citation_evaluators():
    """Phase 38's required evaluation surface: retrieval quality (hit@k),
    citation correctness (precision) and completeness (recall), all fed
    directly from the adapter's mapped SystemOutput -- no evaluator-side
    special-casing for production-rag's shape, because the adapter already
    translated it into the harness's standardized fields (context,
    citations) at the boundary, exactly as `SystemUnderTest` is meant to
    let it."""
    tc = TestCase(
        test_id="t1", input="What is the refund policy?",
        reference_context=["Refunds within 30 days of purchase."],
        expected_citations=[{"page_number": 4}],
    )
    dataset = Dataset(dataset_id="d1", name="retrieval smoke set", version="1.0.0", test_cases=[tc])
    http = FakeHttpClient(default=(200, {
        "answer": "Refunds are available within 30 days.",
        "citations": [{
            "source_id": "S1", "chunk_id": "c1", "document_id": "d1",
            "page_number": 4, "snippet": "Refunds within 30 days of purchase.",
        }],
    }))
    adapter = ProductionRagAdapter(http, "http://localhost:8000", "test-key")

    run = await run_evaluation(
        dataset, adapter,
        [HitAtKEvaluator(k=5), CitationPrecisionEvaluator(match_fields=["page_number"]), CitationRecallEvaluator(match_fields=["page_number"])],
    )

    by_evaluator = {r.evaluator: r for r in run.results}
    assert by_evaluator["hit_at_k"].passed is True  # the real snippet matched reference_context verbatim
    assert by_evaluator["citation_precision"].score == 1.0
    assert by_evaluator["citation_recall"].score == 1.0


async def test_adapter_output_feeds_mocked_faithfulness_judge():
    """Faithfulness/groundedness evaluation (Phase 38's requirement) via a
    MOCKED LLM judge -- FakeJsonLLM, never a real/paid call -- proving the
    adapter's `context`/`output` mapping is exactly what
    FaithfulnessJudgeEvaluator (Day 8) expects, with no adapter-specific
    wiring needed in the evaluator itself."""
    tc = TestCase(test_id="t1", input="What is the refund policy?")
    dataset = Dataset(dataset_id="d1", name="faithfulness smoke set", version="1.0.0", test_cases=[tc])
    http = FakeHttpClient(default=(200, {
        "answer": "Refunds are available within 30 days.",
        "citations": [{
            "source_id": "S1", "chunk_id": "c1", "document_id": "d1",
            "page_number": 4, "snippet": "Refunds within 30 days of purchase.",
        }],
    }))
    adapter = ProductionRagAdapter(http, "http://localhost:8000", "test-key")
    judge = FakeJsonLLM(default={"score": 9, "reasoning": "fully supported by the cited snippet"})

    run = await run_evaluation(dataset, adapter, [FaithfulnessJudgeEvaluator(judge, "gpt-test")])

    result = run.results[0]
    assert result.passed is True
    assert result.score == 0.9
