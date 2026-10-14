import pytest

from eval_harness.comparison import compare_systems
from eval_harness.dataset import Dataset
from eval_harness.evaluators import ExactMatchEvaluator, LengthEvaluator
from eval_harness.mock_system import MockSystem
from eval_harness.models import TestCase
from eval_harness.system_output import SystemOutput

CASES = [
    TestCase(test_id="qa-001", input="Capital of France?", expected_output="Paris"),
    TestCase(test_id="qa-002", input="Capital of Japan?", expected_output="Tokyo"),
]


def _dataset() -> Dataset:
    return Dataset(dataset_id="demo", name="Demo", version="1.0.0", test_cases=CASES)


def _system(name, responses):
    return MockSystem(responses, name=name)


async def test_clear_overall_winner():
    good = _system("good-model", {"qa-001": SystemOutput(output="Paris"), "qa-002": SystemOutput(output="Tokyo")})
    bad = _system("bad-model", {"qa-001": SystemOutput(output="Paris"), "qa-002": SystemOutput(output="Berlin")})
    report = await compare_systems(_dataset(), [good, bad], [ExactMatchEvaluator()])
    assert report.overall_winner == "good-model"
    assert report.overall_pass_rate["good-model"] == 1.0
    assert report.overall_pass_rate["bad-model"] == 0.5


async def test_tied_systems_report_none_not_an_arbitrary_winner():
    a = _system("model-a", {"qa-001": SystemOutput(output="Paris"), "qa-002": SystemOutput(output="Tokyo")})
    b = _system("model-b", {"qa-001": SystemOutput(output="Paris"), "qa-002": SystemOutput(output="Tokyo")})
    report = await compare_systems(_dataset(), [a, b], [ExactMatchEvaluator()])
    assert report.overall_winner is None


async def test_per_evaluator_winner_can_differ_from_overall_winner():
    # model-a: exactly correct, but answers run one character too long for the length cap.
    # model-b: always within the length cap, but never exactly correct.
    a = _system("model-a", {"qa-001": SystemOutput(output="Paris"), "qa-002": SystemOutput(output="Tokyo")})
    b = _system("model-b", {"qa-001": SystemOutput(output="Par"), "qa-002": SystemOutput(output="Toky")})
    evaluators = [ExactMatchEvaluator(), LengthEvaluator(max_length=4)]
    report = await compare_systems(_dataset(), [a, b], evaluators)
    assert report.per_evaluator_winner["exact_match"] == "model-a"
    assert report.per_evaluator_winner["length"] == "model-b"


async def test_runs_dict_keyed_by_system_name():
    a = _system("model-a", {"qa-001": SystemOutput(output="Paris"), "qa-002": SystemOutput(output="Tokyo")})
    b = _system("model-b", {"qa-001": SystemOutput(output="x"), "qa-002": SystemOutput(output="y")})
    report = await compare_systems(_dataset(), [a, b], [ExactMatchEvaluator()])
    assert set(report.runs.keys()) == {"model-a", "model-b"}
    assert report.runs["model-a"].system_name == "model-a"


async def test_requires_at_least_two_systems():
    a = _system("model-a", {})
    with pytest.raises(ValueError, match="at least two"):
        await compare_systems(_dataset(), [a], [ExactMatchEvaluator()])


async def test_duplicate_system_names_rejected():
    a = _system("same-name", {"qa-001": SystemOutput(output="Paris"), "qa-002": SystemOutput(output="Tokyo")})
    b = _system("same-name", {"qa-001": SystemOutput(output="x"), "qa-002": SystemOutput(output="y")})
    with pytest.raises(ValueError, match="unique"):
        await compare_systems(_dataset(), [a, b], [ExactMatchEvaluator()])
