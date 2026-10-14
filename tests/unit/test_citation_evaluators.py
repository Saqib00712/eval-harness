from eval_harness.evaluators.rag import CitationPrecisionEvaluator, CitationRecallEvaluator
from eval_harness.models import TestCase
from eval_harness.system_output import SystemOutput


async def test_precision_all_citations_correct():
    tc = TestCase(test_id="c-1", input="x", expected_citations=[{"page": 1}, {"page": 2}])
    ev = CitationPrecisionEvaluator()
    out = SystemOutput(citations=[{"page": 1}])
    r = await ev.evaluate(tc, out)
    assert r.score == 1.0 and r.passed


async def test_precision_partial_credit_for_wrong_citation():
    tc = TestCase(test_id="c-2", input="x", expected_citations=[{"page": 1}])
    ev = CitationPrecisionEvaluator(threshold=0.4)
    out = SystemOutput(citations=[{"page": 1}, {"page": 99}])
    r = await ev.evaluate(tc, out)
    assert r.score == 0.5
    assert r.passed  # threshold 0.4


async def test_precision_no_citations_given_is_vacuously_perfect():
    tc = TestCase(test_id="c-3", input="x", expected_citations=[{"page": 1}])
    ev = CitationPrecisionEvaluator()
    r = await ev.evaluate(tc, SystemOutput(citations=[]))
    assert r.score == 1.0 and r.passed


async def test_precision_with_match_fields_ignores_other_keys():
    tc = TestCase(test_id="c-4", input="x", expected_citations=[{"page": 1, "snippet": "foo"}])
    ev = CitationPrecisionEvaluator(match_fields=["page"])
    out = SystemOutput(citations=[{"page": 1, "snippet": "a totally different snippet"}])
    r = await ev.evaluate(tc, out)
    assert r.score == 1.0


async def test_recall_all_expected_citations_given():
    tc = TestCase(test_id="c-5", input="x", expected_citations=[{"page": 1}, {"page": 2}])
    ev = CitationRecallEvaluator()
    out = SystemOutput(citations=[{"page": 1}, {"page": 2}, {"page": 3}])
    r = await ev.evaluate(tc, out)
    assert r.score == 1.0 and r.passed


async def test_recall_partial_when_some_citations_missing():
    tc = TestCase(test_id="c-6", input="x", expected_citations=[{"page": 1}, {"page": 2}])
    ev = CitationRecallEvaluator(threshold=0.4)
    out = SystemOutput(citations=[{"page": 1}])
    r = await ev.evaluate(tc, out)
    assert r.score == 0.5
    assert r.passed  # threshold 0.4


async def test_recall_no_expected_citations_is_vacuously_complete():
    tc = TestCase(test_id="c-7", input="x", expected_citations=[])
    ev = CitationRecallEvaluator()
    r = await ev.evaluate(tc, SystemOutput(citations=[]))
    assert r.score == 1.0 and r.passed
