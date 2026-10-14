import pytest
from pydantic import ValidationError

from eval_harness.system_output import SystemOutput, TokenUsage


def test_defaults_for_a_bare_output():
    out = SystemOutput()
    assert out.output is None
    assert out.context == []
    assert out.citations == []
    assert out.structured_data is None
    assert out.metadata == {}
    assert out.latency_ms is None
    assert out.token_usage is None
    assert out.estimated_cost is None
    assert out.model is None


def test_rag_shaped_output():
    out = SystemOutput(
        output="Customers can request a refund within 30 days.",
        context=["Section 4: Refunds -- ..."],
        citations=[{"page": 4}],
        latency_ms=842.5,
        token_usage=TokenUsage(input_tokens=120, output_tokens=40),
        estimated_cost=0.00012,
        model="gpt-4o-mini",
    )
    assert out.citations == [{"page": 4}]
    assert out.token_usage.total_tokens == 160


def test_classifier_shaped_output_has_no_context_or_citations():
    out = SystemOutput(structured_data={"label": "spam", "confidence": 0.94})
    assert out.output is None
    assert out.context == []
    assert out.citations == []
    assert out.structured_data == {"label": "spam", "confidence": 0.94}


def test_token_usage_total_is_computed():
    tu = TokenUsage(input_tokens=100, output_tokens=25)
    assert tu.total_tokens == 125


def test_token_usage_defaults_to_zero():
    tu = TokenUsage()
    assert tu.total_tokens == 0


def test_system_output_is_frozen():
    out = SystemOutput(output="x")
    with pytest.raises(ValidationError):
        out.output = "y"
