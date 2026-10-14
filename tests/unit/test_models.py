import pytest
from pydantic import ValidationError

from eval_harness.models import TestCase


def test_minimal_test_case_uses_defaults():
    tc = TestCase(test_id="qa-001", input="What is the refund policy?")
    assert tc.test_id == "qa-001"
    assert tc.expected_output is None
    assert tc.reference_context == []
    assert tc.expected_citations == []
    assert tc.metadata == {}
    assert tc.tags == []


def test_full_test_case():
    tc = TestCase(
        test_id="qa-001",
        input="What is the refund policy?",
        expected_output="Customers can request a refund within 30 days.",
        reference_context=["Customers may request a refund within 30 days..."],
        expected_citations=[{"page": 4}],
        metadata={"category": "policy"},
        tags=["refund", "policy"],
    )
    assert tc.expected_citations == [{"page": 4}]
    assert tc.tags == ["refund", "policy"]


def test_citation_shape_is_not_fixed():
    """Different systems cite differently -- the model must not assume
    `page` is the only valid key."""
    tc = TestCase(test_id="qa-002", input="x", expected_citations=[{"url": "https://x", "section": "3.2"}])
    assert tc.expected_citations[0]["url"] == "https://x"


@pytest.mark.parametrize("field", ["test_id", "input"])
def test_blank_required_field_rejected(field):
    kwargs = {"test_id": "qa-001", "input": "question"}
    kwargs[field] = "   "
    with pytest.raises(ValidationError):
        TestCase(**kwargs)


def test_missing_test_id_rejected():
    with pytest.raises(ValidationError):
        TestCase(input="question")


def test_missing_input_rejected():
    with pytest.raises(ValidationError):
        TestCase(test_id="qa-001")


def test_test_case_is_frozen():
    """Golden data describes what SHOULD happen; it must not be mutated
    while a run is scoring against it."""
    tc = TestCase(test_id="qa-001", input="question")
    with pytest.raises(ValidationError):
        tc.test_id = "qa-002"
