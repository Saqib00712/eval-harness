import pytest

from eval_harness.evaluators.llm_judge import (
    CorrectnessJudgeEvaluator,
    FaithfulnessJudgeEvaluator,
    RubricJudgeEvaluator,
)
from eval_harness.models import TestCase
from eval_harness.providers.llm import FakeJsonLLM
from eval_harness.system_output import SystemOutput

TC = TestCase(test_id="j-1", input="What is the refund policy?", expected_output="30 days.")


# ---- CorrectnessJudgeEvaluator ----

async def test_correctness_passes_on_high_score():
    llm = FakeJsonLLM(default={"score": 9, "reasoning": "matches expected"})
    ev = CorrectnessJudgeEvaluator(llm, "gpt-test")
    r = await ev.evaluate(TC, SystemOutput(output="You have 30 days."))
    assert r.passed
    assert r.score == 0.9
    assert r.reason == "matches expected"


async def test_correctness_fails_on_low_score():
    llm = FakeJsonLLM(default={"score": 2, "reasoning": "wrong"})
    ev = CorrectnessJudgeEvaluator(llm, "gpt-test")
    r = await ev.evaluate(TC, SystemOutput(output="42 days."))
    assert not r.passed


async def test_correctness_requires_expected_output():
    tc = TestCase(test_id="j-2", input="x")
    llm = FakeJsonLLM(default={"score": 9, "reasoning": "x"})
    ev = CorrectnessJudgeEvaluator(llm, "gpt-test")
    with pytest.raises(ValueError, match="no expected_output"):
        await ev.evaluate(tc, SystemOutput(output="anything"))


async def test_correctness_metadata_threshold_override():
    tc = TestCase(test_id="j-3", input="x", expected_output="30 days.", metadata={"correctness_threshold": 0.95})
    llm = FakeJsonLLM(default={"score": 9, "reasoning": "close"})
    ev = CorrectnessJudgeEvaluator(llm, "gpt-test")
    r = await ev.evaluate(tc, SystemOutput(output="30 days."))
    assert not r.passed  # 0.9 < 0.95 override


async def test_correctness_cost_tracked_when_pricing_given():
    llm = FakeJsonLLM(default={"score": 9, "reasoning": "x"})
    ev = CorrectnessJudgeEvaluator(llm, "gpt-test", price_per_million_tokens=5.0)
    r = await ev.evaluate(TC, SystemOutput(output="30 days."))
    assert r.token_usage_total == 70  # FakeJsonLLM default usage: 50 + 20
    assert r.estimated_cost is not None and r.estimated_cost > 0


async def test_correctness_no_cost_fields_without_pricing():
    llm = FakeJsonLLM(default={"score": 9, "reasoning": "x"})
    ev = CorrectnessJudgeEvaluator(llm, "gpt-test")
    r = await ev.evaluate(TC, SystemOutput(output="30 days."))
    assert r.token_usage_total is None
    assert r.estimated_cost is None


async def test_correctness_bad_judge_response_raises():
    llm = FakeJsonLLM(default={"nonsense": True})
    ev = CorrectnessJudgeEvaluator(llm, "gpt-test")
    with pytest.raises(ValueError, match="unusable response"):
        await ev.evaluate(TC, SystemOutput(output="30 days."))


async def test_correctness_llm_call_failure_wrapped_as_value_error():
    # Day 18 test-suite audit: this covered a BAD RESPONSE (above) but never
    # a call that fails outright (LLMError) -- a different code path in
    # _JudgeEvaluatorBase._judge.
    from eval_harness.providers.llm import LLMError

    class _AlwaysFailsLLM:
        async def complete_json(self, **kwargs):
            raise LLMError("upstream request failed")

    ev = CorrectnessJudgeEvaluator(_AlwaysFailsLLM(), "gpt-test")
    with pytest.raises(ValueError, match="judge call failed"):
        await ev.evaluate(TC, SystemOutput(output="30 days."))


# ---- FaithfulnessJudgeEvaluator ----

async def test_faithfulness_judge_no_context_scores_zero_without_calling_llm():
    llm = FakeJsonLLM()  # would raise IndexError if ever called
    ev = FaithfulnessJudgeEvaluator(llm, "gpt-test")
    r = await ev.evaluate(TC, SystemOutput(output="x", context=[]))
    assert r.score == 0.0 and not r.passed
    assert llm.calls == []


async def test_faithfulness_judge_requires_output():
    llm = FakeJsonLLM(default={"score": 9, "reasoning": "x"})
    ev = FaithfulnessJudgeEvaluator(llm, "gpt-test")
    with pytest.raises(ValueError, match="no `output` text"):
        await ev.evaluate(TC, SystemOutput(output=None, context=["ctx"]))


async def test_faithfulness_judge_passes_on_high_score():
    llm = FakeJsonLLM(default={"score": 10, "reasoning": "fully supported"})
    ev = FaithfulnessJudgeEvaluator(llm, "gpt-test")
    r = await ev.evaluate(TC, SystemOutput(output="30 days.", context=["Refunds are available within 30 days."]))
    assert r.passed and r.score == 1.0


# ---- RubricJudgeEvaluator ----

async def test_rubric_judge_uses_constructor_rubric():
    llm = FakeJsonLLM(default={"score": 8, "reasoning": "mostly on-topic"})
    ev = RubricJudgeEvaluator(llm, "gpt-test", rubric="Answer must be polite and under 50 words.")
    r = await ev.evaluate(TC, SystemOutput(output="You have 30 days, thanks for asking!"))
    assert r.passed
    assert "polite" in llm.calls[0]["system"]
    assert r.metadata["rubric"] == "Answer must be polite and under 50 words."


async def test_rubric_judge_metadata_override():
    tc = TestCase(test_id="j-4", input="x", metadata={"rubric": "Must mention a dollar amount."})
    llm = FakeJsonLLM(default={"score": 3, "reasoning": "no dollar amount mentioned"})
    ev = RubricJudgeEvaluator(llm, "gpt-test", rubric="default rubric")
    r = await ev.evaluate(tc, SystemOutput(output="some answer"))
    assert "Must mention a dollar amount." in llm.calls[0]["system"]
    assert r.metadata["rubric"] == "Must mention a dollar amount."
