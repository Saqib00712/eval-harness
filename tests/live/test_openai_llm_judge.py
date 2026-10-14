"""
Live test: real OpenAI LLM-judge call, real cost (a fraction of a cent).
Skips automatically with no OPENAI_API_KEY set -- the same skip-don't-fail
pattern as tests/live/test_openai_embedding.py (Day 6) and, before that,
production-rag's own OCR tests (Day 21 there). Run explicitly with:

    OPENAI_API_KEY=sk-... pytest tests/live -v
"""

import os

import pytest

pytest.importorskip("openai")

from eval_harness.evaluators.llm_judge import CorrectnessJudgeEvaluator
from eval_harness.models import TestCase
from eval_harness.providers.openai_llm import OpenAIJsonLLM
from eval_harness.system_output import SystemOutput

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="live test needs a real OPENAI_API_KEY"
)


async def test_real_judge_recognizes_a_correct_paraphrased_answer():
    llm = OpenAIJsonLLM(api_key=os.environ["OPENAI_API_KEY"])
    ev = CorrectnessJudgeEvaluator(llm, "gpt-4o-mini")
    tc = TestCase(test_id="live-correct", input="What is the refund window?", expected_output="30 days.")
    output = SystemOutput(output="Customers have thirty days to request a refund.")
    r = await ev.evaluate(tc, output)
    assert r.passed


async def test_real_judge_recognizes_a_wrong_answer():
    llm = OpenAIJsonLLM(api_key=os.environ["OPENAI_API_KEY"])
    ev = CorrectnessJudgeEvaluator(llm, "gpt-4o-mini")
    tc = TestCase(test_id="live-wrong", input="What is the refund window?", expected_output="30 days.")
    output = SystemOutput(output="Mount Everest is the tallest mountain on Earth.")
    r = await ev.evaluate(tc, output)
    assert not r.passed
