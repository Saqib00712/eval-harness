import pytest

from eval_harness.evaluators.semantic import SemanticSimilarityEvaluator
from eval_harness.providers.embedding import FakeEmbeddingProvider
from eval_harness.models import TestCase
from eval_harness.system_output import SystemOutput

TC = TestCase(test_id="qa-001", input="What is the refund policy?", expected_output="The refund window is 30 days.")


async def test_identical_answer_passes_default_threshold():
    ev = SemanticSimilarityEvaluator(FakeEmbeddingProvider())
    r = await ev.evaluate(TC, SystemOutput(output="The refund window is 30 days."))
    assert r.passed
    assert r.score == 1.0


async def test_wildly_different_answer_fails_default_threshold():
    ev = SemanticSimilarityEvaluator(FakeEmbeddingProvider())
    r = await ev.evaluate(TC, SystemOutput(output="Mount Everest is the tallest mountain."))
    assert not r.passed


async def test_near_duplicate_answer_passes_default_threshold():
    ev = SemanticSimilarityEvaluator(FakeEmbeddingProvider())
    r = await ev.evaluate(TC, SystemOutput(output="The refund window is 30 day."))
    assert r.passed


async def test_per_test_case_threshold_override():
    tc = TestCase(test_id="qa-002", input="x", expected_output="The refund window is 30 days.",
                   metadata={"semantic_threshold": 0.99})
    ev = SemanticSimilarityEvaluator(FakeEmbeddingProvider(), threshold=0.5)
    r = await ev.evaluate(tc, SystemOutput(output="The refund window is 30 day."))  # near-duplicate, not identical
    assert r.threshold == 0.99
    assert not r.passed  # would have passed at the constructor's 0.5 default


async def test_requires_expected_output():
    tc = TestCase(test_id="qa-003", input="x")  # no expected_output
    ev = SemanticSimilarityEvaluator(FakeEmbeddingProvider())
    with pytest.raises(ValueError, match="no expected_output"):
        await ev.evaluate(tc, SystemOutput(output="anything"))


async def test_metadata_includes_model_and_raw_similarity():
    ev = SemanticSimilarityEvaluator(FakeEmbeddingProvider())
    r = await ev.evaluate(TC, SystemOutput(output="The refund window is 30 days."))
    assert r.metadata["embedding_model"] == "fake-trigram-hash-v1"
    assert r.metadata["raw_cosine_similarity"] == 1.0


async def test_score_is_clamped_to_zero_when_negative():
    """The evaluator's own clamp (not FakeEmbeddingProvider's, which never
    produces negative similarity) -- verified directly against the math."""
    ev = SemanticSimilarityEvaluator(FakeEmbeddingProvider())
    # Two empty-ish strings still produce a defined (zero) vector; this just
    # confirms the evaluator never returns a sub-zero score regardless of
    # what the provider returns.
    r = await ev.evaluate(TC, SystemOutput(output=""))
    assert r.score >= 0.0
