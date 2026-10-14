import pytest

from eval_harness.evaluators.rag import ContextRelevanceEvaluator, FaithfulnessEvaluator
from eval_harness.models import TestCase
from eval_harness.providers.embedding import FakeEmbeddingProvider
from eval_harness.system_output import SystemOutput

TC = TestCase(test_id="g-1", input="What is the refund policy?")


async def test_context_relevance_scores_similar_chunks_high():
    ev = ContextRelevanceEvaluator(FakeEmbeddingProvider(), threshold=0.1)
    out = SystemOutput(context=["What is the refund policy?", "What is the refund policy?"])
    r = await ev.evaluate(TC, out)
    assert r.score == pytest.approx(1.0)
    assert r.passed


async def test_context_relevance_no_context_scores_zero():
    ev = ContextRelevanceEvaluator(FakeEmbeddingProvider())
    r = await ev.evaluate(TC, SystemOutput(context=[]))
    assert r.score == 0.0
    assert not r.passed
    assert r.metadata["chunk_count"] == 0


async def test_context_relevance_metadata_threshold_override():
    tc = TestCase(test_id="g-2", input="refund policy", metadata={"context_relevance_threshold": 0.99})
    ev = ContextRelevanceEvaluator(FakeEmbeddingProvider(), threshold=0.1)
    out = SystemOutput(context=["totally unrelated text about mountains"])
    r = await ev.evaluate(tc, out)
    assert r.threshold == 0.99


async def test_faithfulness_identical_answer_and_context_scores_high():
    ev = FaithfulnessEvaluator(FakeEmbeddingProvider(), threshold=0.5)
    out = SystemOutput(output="The refund window is 30 days.", context=["The refund window is 30 days."])
    r = await ev.evaluate(TC, out)
    assert r.score == pytest.approx(1.0)
    assert r.passed


async def test_faithfulness_no_context_scores_zero():
    ev = FaithfulnessEvaluator(FakeEmbeddingProvider())
    out = SystemOutput(output="Some answer.", context=[])
    r = await ev.evaluate(TC, out)
    assert r.score == 0.0
    assert not r.passed


async def test_faithfulness_requires_output_text():
    ev = FaithfulnessEvaluator(FakeEmbeddingProvider())
    out = SystemOutput(output=None, context=["something"])
    with pytest.raises(ValueError, match="no `output` text"):
        await ev.evaluate(TC, out)


async def test_faithfulness_metadata_documents_proxy_nature():
    ev = FaithfulnessEvaluator(FakeEmbeddingProvider(), threshold=0.1)
    out = SystemOutput(output="The refund window is 30 days.", context=["The refund window is 30 days."])
    r = await ev.evaluate(TC, out)
    assert "proxy" in r.reason.lower()
