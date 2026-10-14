import pytest

from eval_harness.dataset import Dataset
from eval_harness.evaluator import EvaluatorResult
from eval_harness.evaluators import ExactMatchEvaluator, LengthEvaluator
from eval_harness.mock_system import MockSystem
from eval_harness.models import TestCase
from eval_harness.runner import run_evaluation
from eval_harness.system_output import SystemOutput, TokenUsage

CASES = [
    TestCase(test_id="qa-001", input="What is the refund policy?", expected_output="30 days"),
    TestCase(test_id="qa-002", input="What is the capital of France?", expected_output="Paris"),
]


def _dataset(cases=None) -> Dataset:
    return Dataset(dataset_id="demo", name="Demo", version="1.0.0", test_cases=cases or CASES)


class _AlwaysRaisesEvaluator:
    @property
    def name(self) -> str:
        return "broken"

    async def evaluate(self, test_case, output) -> EvaluatorResult:
        raise RuntimeError("boom")


# ---------- happy path ----------

async def test_run_produces_one_result_per_test_case_per_evaluator():
    system = MockSystem({
        "qa-001": SystemOutput(output="30 days"),
        "qa-002": SystemOutput(output="Paris"),
    })
    run = await run_evaluation(_dataset(), system, [ExactMatchEvaluator()])
    assert len(run.results) == 2
    assert run.total_tests == 2
    assert run.passed_tests == 2
    assert run.pass_rate == 1.0


async def test_run_records_dataset_and_system_identity():
    dataset = _dataset()
    system = MockSystem({"qa-001": SystemOutput(output="30 days"), "qa-002": SystemOutput(output="Paris")},
                         name="baseline", version="2026.01")
    run = await run_evaluation(dataset, system, [ExactMatchEvaluator()])
    assert run.dataset_id == "demo"
    assert run.dataset_version == "1.0.0"
    assert run.dataset_content_hash == dataset.content_hash
    assert run.system_name == "baseline"
    assert run.system_version == "2026.01"
    assert run.status == "completed"


async def test_a_test_case_fails_the_run_if_any_evaluator_fails_it():
    system = MockSystem({
        "qa-001": SystemOutput(output="30 days"),   # exact match passes
        "qa-002": SystemOutput(output="Lyon"),       # exact match fails (expected Paris)
    })
    run = await run_evaluation(_dataset(), system, [ExactMatchEvaluator()])
    assert run.passed_tests == 1
    assert run.failed_tests == 1
    assert run.pass_rate == 0.5


async def test_multiple_evaluators_all_must_pass_for_the_test_case_to_pass():
    system = MockSystem({
        "qa-001": SystemOutput(output="30 days"),   # exact match passes, but...
        "qa-002": SystemOutput(output="Paris"),
    }, default=None)
    evaluators = [ExactMatchEvaluator(), LengthEvaluator(min_length=100)]  # length always fails here
    run = await run_evaluation(_dataset(), system, evaluators)
    assert len(run.results) == 4  # 2 test cases x 2 evaluators
    assert run.passed_tests == 0  # every test case fails the length check


# ---------- cost/token aggregation (must not double count across evaluators) ----------

async def test_system_cost_is_counted_once_per_test_case_not_per_evaluator():
    output = SystemOutput(output="30 days", token_usage=TokenUsage(input_tokens=100, output_tokens=20), estimated_cost=0.001)
    system = MockSystem({"qa-001": output}, default=output)
    run = await run_evaluation(_dataset([CASES[0]]), system, [ExactMatchEvaluator(), LengthEvaluator(min_length=1)])
    assert len(run.results) == 2  # 1 test case x 2 evaluators
    assert run.total_tokens == 120  # NOT 240 -- counted once, not once per evaluator
    assert run.estimated_cost == pytest.approx(0.001)


# ---------- failure handling ----------

class _AlwaysRaisesSystem:
    name = "broken-system"
    version = "0.0.0"

    async def run(self, test_case):
        raise RuntimeError("system is down")


async def test_system_failure_is_captured_not_raised():
    run = await run_evaluation(_dataset([CASES[0]]), _AlwaysRaisesSystem(), [ExactMatchEvaluator()])
    assert run.total_tests == 1
    assert run.passed_tests == 0
    assert run.results[0].failure_category == "dependency_failure"
    assert "system is down" in run.results[0].reason


async def test_evaluator_failure_is_captured_not_raised():
    system = MockSystem({"qa-001": SystemOutput(output="30 days")})
    run = await run_evaluation(_dataset([CASES[0]]), system, [_AlwaysRaisesEvaluator()])
    assert run.results[0].failure_category == "evaluation_failure"
    assert "boom" in run.results[0].reason


async def test_one_evaluator_failing_does_not_block_other_evaluators_on_same_test_case():
    system = MockSystem({"qa-001": SystemOutput(output="30 days")})
    run = await run_evaluation(_dataset([CASES[0]]), system, [_AlwaysRaisesEvaluator(), ExactMatchEvaluator()])
    assert len(run.results) == 2
    categories = {r.evaluator: r.failure_category for r in run.results}
    assert categories["broken"] == "evaluation_failure"
    assert categories["exact_match"] is None  # exact_match succeeded normally


async def test_ordinary_threshold_miss_is_categorized_not_left_none():
    """Day 9: an evaluator that runs fine and simply scores below its
    threshold is no longer indistinguishable from 'nothing went wrong'."""
    tc = TestCase(test_id="qa-003", input="x", expected_output="Paris")
    run = await run_evaluation(_dataset([tc]), MockSystem({"qa-003": SystemOutput(output="London")}), [ExactMatchEvaluator()])
    result = run.results[0]
    assert not result.passed
    assert result.failure_category == "threshold_miss"


async def test_system_failure_on_one_case_does_not_block_other_cases():
    class _FailsOnlyFirst:
        name = "flaky"
        version = "0.0.0"

        async def run(self, test_case):
            if test_case.test_id == "qa-001":
                raise RuntimeError("nope")
            return SystemOutput(output="Paris")

    run = await run_evaluation(_dataset(), _FailsOnlyFirst(), [ExactMatchEvaluator()])
    assert run.total_tests == 2
    assert run.passed_tests == 1  # qa-002 still got scored normally
    assert run.failed_tests == 1


# ---------- guardrails ----------

async def test_empty_dataset_raises():
    empty = Dataset(dataset_id="x", name="x", version="1.0.0", test_cases=[])
    with pytest.raises(ValueError, match="empty dataset"):
        await run_evaluation(empty, MockSystem({}), [ExactMatchEvaluator()])


async def test_no_evaluators_raises():
    with pytest.raises(ValueError, match="no evaluators"):
        await run_evaluation(_dataset(), MockSystem({}), [])


# ---------- evaluator-reported cost (Day 8: LLM judges cost money too) ----------

async def test_evaluator_reported_cost_is_added_to_run_totals_separately_from_system_cost():
    class _PaidEvaluator:
        name = "paid_judge"

        async def evaluate(self, test_case, output):
            return EvaluatorResult(
                evaluator=self.name, score=1.0, passed=True,
                token_usage_total=100, estimated_cost=0.001,
            )

    system = MockSystem({
        c.test_id: SystemOutput(output="x", token_usage=TokenUsage(input_tokens=10, output_tokens=5), estimated_cost=0.0002)
        for c in CASES
    })
    run = await run_evaluation(_dataset(), system, [_PaidEvaluator()])

    # system cost counted once per test case (2 cases) + evaluator cost counted once per result (2 results)
    assert run.total_tokens == (15 * 2) + (100 * 2)
    assert run.estimated_cost == pytest.approx((0.0002 * 2) + (0.001 * 2))
    assert all(r.token_usage_total == 100 and r.estimated_cost == 0.001 for r in run.results)


# ---------- system latency aggregation (Day 13) ----------

async def test_system_latency_aggregated_once_per_test_case_not_per_evaluator():
    system = MockSystem({
        "qa-001": SystemOutput(output="30 days", latency_ms=100.0),
        "qa-002": SystemOutput(output="Paris", latency_ms=300.0),
    })
    run = await run_evaluation(_dataset(), system, [ExactMatchEvaluator(), LengthEvaluator(max_length=50)])
    assert run.average_system_latency_ms == 200.0  # mean of 100, 300 -- NOT divided by 4 (2 cases x 2 evaluators)
    assert run.max_system_latency_ms == 300.0


async def test_system_latency_none_when_never_reported():
    run = await run_evaluation(_dataset(), MockSystem({"qa-001": SystemOutput(output="x"), "qa-002": SystemOutput(output="y")}), [ExactMatchEvaluator()])
    assert run.average_system_latency_ms is None
    assert run.max_system_latency_ms is None


# ---------- observer hooks (Day 16) ----------

async def test_observer_hooks_fire_and_never_affect_the_result():
    from eval_harness.observability import RunObserver

    class _RecordingObserver(RunObserver):
        def __init__(self):
            self.events = []

        def on_test_case_start(self, test_case):
            self.events.append(("start", test_case.test_id))

        def on_system_output(self, test_case, output):
            self.events.append(("system_output", test_case.test_id))

        def on_evaluator_result(self, test_case, evaluator_name, result):
            self.events.append(("evaluator_result", test_case.test_id, evaluator_name))

        def on_run_complete(self, run):
            self.events.append(("complete", run.run_id))

    observer = _RecordingObserver()
    plain_run = await run_evaluation(_dataset(), MockSystem({c.test_id: SystemOutput(output=c.expected_output) for c in CASES}), [ExactMatchEvaluator()])
    observed_run = await run_evaluation(
        _dataset(), MockSystem({c.test_id: SystemOutput(output=c.expected_output) for c in CASES}),
        [ExactMatchEvaluator()], observer=observer,
    )

    assert ("start", "qa-001") in observer.events
    assert ("system_output", "qa-001") in observer.events
    assert ("evaluator_result", "qa-001", "exact_match") in observer.events
    assert any(e[0] == "complete" for e in observer.events)
    # the observer is a pure side channel -- it never changes the run's own numbers
    assert observed_run.pass_rate == plain_run.pass_rate
    assert observed_run.total_tests == plain_run.total_tests


async def test_observer_on_system_error_and_on_evaluator_error_fire():
    from eval_harness.observability import RunObserver

    class _FailingSystem:
        name = "flaky"
        version = "0.0.0"

        async def run(self, test_case):
            raise RuntimeError("system boom")

    class _FailingEvaluator:
        name = "broken"

        async def evaluate(self, test_case, output):
            raise RuntimeError("evaluator boom")

    class _RecordingObserver(RunObserver):
        def __init__(self):
            self.system_errors = []
            self.evaluator_errors = []

        def on_system_error(self, test_case, exc):
            self.system_errors.append(test_case.test_id)

        def on_evaluator_error(self, test_case, evaluator_name, exc):
            self.evaluator_errors.append((test_case.test_id, evaluator_name))

    observer = _RecordingObserver()
    await run_evaluation(_dataset(), _FailingSystem(), [ExactMatchEvaluator()], observer=observer)
    assert "qa-001" in observer.system_errors

    working_system = MockSystem({c.test_id: SystemOutput(output=c.expected_output) for c in CASES})
    observer2 = _RecordingObserver()
    await run_evaluation(_dataset(), working_system, [_FailingEvaluator()], observer=observer2)
    assert ("qa-001", "broken") in observer2.evaluator_errors
