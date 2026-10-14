import pytest
from pydantic import ValidationError

from eval_harness.evaluator import Evaluator, EvaluatorResult


def test_evaluator_result_defaults():
    r = EvaluatorResult(evaluator="exact_match", score=1.0, passed=True)
    assert r.threshold is None
    assert r.reason == ""
    assert r.metadata == {}


def test_evaluator_result_is_frozen():
    r = EvaluatorResult(evaluator="exact_match", score=1.0, passed=True)
    with pytest.raises(ValidationError):
        r.score = 0.0


class _FakeEvaluator:
    @property
    def name(self) -> str:
        return "fake"

    async def evaluate(self, test_case, output) -> EvaluatorResult:
        return EvaluatorResult(evaluator=self.name, score=1.0, passed=True)


def test_a_conforming_object_satisfies_the_protocol():
    assert isinstance(_FakeEvaluator(), Evaluator)


async def test_a_conforming_object_actually_behaves_like_an_evaluator():
    # Day 18 test-suite audit: the isinstance check above only proves
    # _FakeEvaluator has the right SHAPE (runtime_checkable only inspects
    # attribute names) -- it never actually called .name or .evaluate(),
    # so neither line of _FakeEvaluator's own body was ever exercised.
    fake = _FakeEvaluator()
    assert fake.name == "fake"
    result = await fake.evaluate(test_case=None, output=None)
    assert result.evaluator == "fake"
    assert result.passed is True


def test_a_non_conforming_object_does_not_satisfy_the_protocol():
    class NotAnEvaluator:
        pass

    assert not isinstance(NotAnEvaluator(), Evaluator)
