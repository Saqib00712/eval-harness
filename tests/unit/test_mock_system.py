import pytest

from eval_harness.mock_system import MockSystem
from eval_harness.models import TestCase
from eval_harness.system import SystemUnderTest
from eval_harness.system_output import SystemOutput

TC = TestCase(test_id="qa-001", input="What is the refund policy?")


def test_mock_system_satisfies_the_protocol():
    """runtime_checkable means isinstance() actually works here -- this is
    the structural-typing guarantee the whole abstraction depends on."""
    mock = MockSystem({})
    assert isinstance(mock, SystemUnderTest)


async def test_returns_canned_response_for_known_test_id():
    canned = SystemOutput(output="Within 30 days.")
    mock = MockSystem({"qa-001": canned})
    result = await mock.run(TC)
    assert result is canned


async def test_falls_back_to_default_for_unknown_test_id():
    default = SystemOutput(output="(no canned response)")
    mock = MockSystem({}, default=default)
    unknown = TestCase(test_id="qa-999", input="anything")
    result = await mock.run(unknown)
    assert result is default


async def test_raises_when_no_default_and_unknown_test_id():
    mock = MockSystem({})
    unknown = TestCase(test_id="qa-999", input="anything")
    with pytest.raises(KeyError, match="qa-999"):
        await mock.run(unknown)


def test_name_and_version_are_configurable():
    mock = MockSystem({}, name="baseline-v1", version="2026.01")
    assert mock.name == "baseline-v1"
    assert mock.version == "2026.01"


def test_name_and_version_have_defaults():
    mock = MockSystem({})
    assert mock.name == "mock-system"
    assert mock.version == "0.0.0"
