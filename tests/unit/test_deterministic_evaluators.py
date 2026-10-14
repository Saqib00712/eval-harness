import pytest

from eval_harness.evaluators import (
    ContainsEvaluator,
    ExactMatchEvaluator,
    JSONSchemaEvaluator,
    LengthEvaluator,
    RegexEvaluator,
    RequiredFieldsEvaluator,
)
from eval_harness.models import TestCase
from eval_harness.system_output import SystemOutput

TC = TestCase(test_id="qa-001", input="question", expected_output="Thirty days.")


# ---------- ExactMatchEvaluator ----------

async def test_exact_match_passes_on_identical_text():
    r = await ExactMatchEvaluator().evaluate(TC, SystemOutput(output="Thirty days."))
    assert r.passed and r.score == 1.0


async def test_exact_match_is_case_insensitive_by_default():
    r = await ExactMatchEvaluator().evaluate(TC, SystemOutput(output="thirty days."))
    assert r.passed


async def test_exact_match_case_sensitive_mode_rejects_case_difference():
    r = await ExactMatchEvaluator(case_sensitive=True).evaluate(TC, SystemOutput(output="thirty days."))
    assert not r.passed


async def test_exact_match_ignores_surrounding_whitespace():
    r = await ExactMatchEvaluator().evaluate(TC, SystemOutput(output="  Thirty days.  "))
    assert r.passed


async def test_exact_match_fails_on_different_text():
    r = await ExactMatchEvaluator().evaluate(TC, SystemOutput(output="Forty-five days."))
    assert not r.passed and r.score == 0.0


async def test_exact_match_requires_expected_output():
    tc = TestCase(test_id="qa-002", input="x")  # no expected_output
    with pytest.raises(ValueError, match="no expected_output"):
        await ExactMatchEvaluator().evaluate(tc, SystemOutput(output="anything"))


# ---------- ContainsEvaluator ----------

async def test_contains_passes_when_all_terms_present():
    r = await ContainsEvaluator(terms=["30", "days"]).evaluate(TC, SystemOutput(output="You get 30 days."))
    assert r.passed and r.score == 1.0


async def test_contains_fails_when_require_all_and_one_missing():
    r = await ContainsEvaluator(terms=["30", "refund"]).evaluate(TC, SystemOutput(output="You get 30 days."))
    assert not r.passed
    assert r.score == 0.5
    assert "refund" in r.metadata["missing"]


async def test_contains_require_any_passes_with_one_hit():
    ev = ContainsEvaluator(terms=["30", "refund"], require_all=False)
    r = await ev.evaluate(TC, SystemOutput(output="You get 30 days."))
    assert r.passed


async def test_contains_per_test_case_override_wins_over_constructor():
    tc = TestCase(test_id="qa-003", input="x", metadata={"required_terms": ["banana"]})
    r = await ContainsEvaluator(terms=["30"]).evaluate(tc, SystemOutput(output="banana split"))
    assert r.passed


async def test_contains_raises_when_unconfigured():
    with pytest.raises(ValueError, match="no required terms"):
        await ContainsEvaluator().evaluate(TC, SystemOutput(output="anything"))


# ---------- RegexEvaluator ----------

async def test_regex_passes_on_match():
    r = await RegexEvaluator(pattern=r"\d+ days").evaluate(TC, SystemOutput(output="You get 30 days."))
    assert r.passed
    assert r.metadata["matched_text"] == "30 days"


async def test_regex_fails_without_match():
    r = await RegexEvaluator(pattern=r"\d+ weeks").evaluate(TC, SystemOutput(output="You get 30 days."))
    assert not r.passed


async def test_regex_raises_when_unconfigured():
    with pytest.raises(ValueError, match="no pattern"):
        await RegexEvaluator().evaluate(TC, SystemOutput(output="anything"))


# ---------- RegexEvaluator resource ceilings (Day 17 security review) ----------

async def test_regex_rejects_oversized_pattern():
    ev = RegexEvaluator(pattern="a" * 500, max_pattern_length=200)
    with pytest.raises(ValueError, match="char.*limit"):
        await ev.evaluate(TC, SystemOutput(output="anything"))


async def test_regex_rejects_oversized_text():
    ev = RegexEvaluator(pattern=r"\d+", max_text_length=10)
    with pytest.raises(ValueError, match="char.*limit"):
        await ev.evaluate(TC, SystemOutput(output="x" * 100))


async def test_regex_defaults_accept_normal_pattern_and_text():
    r = await RegexEvaluator(pattern=r"\d+ days").evaluate(TC, SystemOutput(output="You get 30 days."))
    assert r.passed  # unaffected by the new caps at realistic sizes


# ---------- JSONSchemaEvaluator ----------

SCHEMA = {"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"]}


async def test_json_schema_passes_on_valid_data():
    r = await JSONSchemaEvaluator(schema=SCHEMA).evaluate(TC, SystemOutput(structured_data={"label": "spam"}))
    assert r.passed


async def test_json_schema_fails_on_missing_required_key():
    r = await JSONSchemaEvaluator(schema=SCHEMA).evaluate(TC, SystemOutput(structured_data={}))
    assert not r.passed
    assert "label" in r.reason


async def test_json_schema_fails_when_structured_data_is_none():
    r = await JSONSchemaEvaluator(schema=SCHEMA).evaluate(TC, SystemOutput())
    assert not r.passed


async def test_json_schema_raises_when_unconfigured():
    with pytest.raises(ValueError, match="no JSON schema"):
        await JSONSchemaEvaluator().evaluate(TC, SystemOutput(structured_data={}))


# ---------- LengthEvaluator ----------

async def test_length_passes_within_bounds():
    r = await LengthEvaluator(min_length=5, max_length=20).evaluate(TC, SystemOutput(output="Thirty days."))
    assert r.passed


async def test_length_fails_when_too_short():
    r = await LengthEvaluator(min_length=100).evaluate(TC, SystemOutput(output="Short."))
    assert not r.passed
    assert "below minimum" in r.reason


async def test_length_fails_when_too_long():
    r = await LengthEvaluator(max_length=3).evaluate(TC, SystemOutput(output="Thirty days."))
    assert not r.passed
    assert "exceeds maximum" in r.reason


async def test_length_counts_words_when_configured():
    ev = LengthEvaluator(min_length=1, max_length=2, unit="words")
    r = await ev.evaluate(TC, SystemOutput(output="Thirty days."))
    assert r.passed
    assert r.metadata["length"] == 2


async def test_length_raises_when_unconfigured():
    with pytest.raises(ValueError, match="no min_length/max_length"):
        await LengthEvaluator().evaluate(TC, SystemOutput(output="anything"))


def test_length_rejects_invalid_unit():
    # Day 18 test-suite audit: this constructor-time guard had no test at all.
    with pytest.raises(ValueError, match="unit must be"):
        LengthEvaluator(max_length=10, unit="sentences")


# ---------- RequiredFieldsEvaluator ----------

async def test_required_fields_passes_when_all_present():
    ev = RequiredFieldsEvaluator(fields=["label", "confidence"])
    r = await ev.evaluate(TC, SystemOutput(structured_data={"label": "spam", "confidence": 0.9}))
    assert r.passed


async def test_required_fields_fails_when_a_field_is_missing():
    ev = RequiredFieldsEvaluator(fields=["label", "confidence"])
    r = await ev.evaluate(TC, SystemOutput(structured_data={"label": "spam"}))
    assert not r.passed
    assert r.metadata["missing"] == ["confidence"]


async def test_required_fields_fails_when_a_field_is_none():
    ev = RequiredFieldsEvaluator(fields=["label"])
    r = await ev.evaluate(TC, SystemOutput(structured_data={"label": None}))
    assert not r.passed


async def test_required_fields_raises_when_unconfigured():
    with pytest.raises(ValueError, match="no required_fields"):
        await RequiredFieldsEvaluator().evaluate(TC, SystemOutput(structured_data={}))
