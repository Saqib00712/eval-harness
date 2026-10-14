import pytest

from eval_harness.evaluators.rag import HitAtKEvaluator, MRREvaluator, RecallAtKEvaluator
from eval_harness.models import TestCase
from eval_harness.system_output import SystemOutput

TC = TestCase(test_id="r-1", input="x", reference_context=["chunk-a", "chunk-b"])


async def test_hit_at_k_passes_when_expected_chunk_retrieved():
    ev = HitAtKEvaluator(k=3)
    out = SystemOutput(context=["chunk-z", "chunk-a"])
    r = await ev.evaluate(TC, out)
    assert r.passed and r.score == 1.0


async def test_hit_at_k_fails_when_nothing_matches():
    ev = HitAtKEvaluator(k=3)
    out = SystemOutput(context=["chunk-z", "chunk-y"])
    r = await ev.evaluate(TC, out)
    assert not r.passed and r.score == 0.0


async def test_hit_at_k_respects_metadata_override():
    tc = TestCase(test_id="r-2", input="x", reference_context=["chunk-a"], metadata={"retrieval_k": 1})
    ev = HitAtKEvaluator(k=5)
    out = SystemOutput(context=["chunk-z", "chunk-a"])  # chunk-a at rank 2
    r = await ev.evaluate(tc, out)
    assert not r.passed  # rank 2 > k=1 override
    assert r.metadata["k"] == 1


async def test_hit_at_k_requires_reference_context():
    tc = TestCase(test_id="r-3", input="x")
    ev = HitAtKEvaluator()
    with pytest.raises(ValueError, match="no reference_context"):
        await ev.evaluate(tc, SystemOutput())


async def test_recall_at_k_partial_credit():
    tc = TestCase(test_id="r-4", input="x", reference_context=["a", "b"])
    ev = RecallAtKEvaluator(k=5, threshold=0.5)
    out = SystemOutput(context=["a", "z"])  # only 1 of 2 found
    r = await ev.evaluate(tc, out)
    assert r.score == 0.5
    assert r.passed  # threshold 0.5


async def test_recall_at_k_fails_below_threshold():
    tc = TestCase(test_id="r-5", input="x", reference_context=["a", "b"])
    ev = RecallAtKEvaluator(k=5, threshold=1.0)
    out = SystemOutput(context=["a"])
    r = await ev.evaluate(tc, out)
    assert r.score == 0.5
    assert not r.passed


async def test_recall_at_k_requires_reference_context():
    # Day 18 test-suite audit: HitAtKEvaluator and MRREvaluator's identical
    # guard were tested; RecallAtKEvaluator's own copy of it was not.
    tc = TestCase(test_id="r-5b", input="x")
    ev = RecallAtKEvaluator()
    with pytest.raises(ValueError, match="no reference_context"):
        await ev.evaluate(tc, SystemOutput())


async def test_mrr_scores_first_expected_rank():
    tc = TestCase(test_id="r-6", input="x", reference_context=["a"])
    ev = MRREvaluator(threshold=0.4)
    out = SystemOutput(context=["z", "y", "a"])  # rank 3
    r = await ev.evaluate(tc, out)
    assert r.score == pytest.approx(1 / 3, abs=1e-4)
    assert not r.passed  # 0.333 < 0.4


async def test_mrr_requires_reference_context():
    tc = TestCase(test_id="r-7", input="x")
    ev = MRREvaluator()
    with pytest.raises(ValueError, match="no reference_context"):
        await ev.evaluate(tc, SystemOutput())
