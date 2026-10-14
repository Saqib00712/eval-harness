"""
Phase 11 + 12: EvaluationRun and PerTestResult -- what the runner (Day 5's
other half, runner.py) produces.

One PerTestResult per (test_case, evaluator) pair -- NOT one per test case.
A test case scored by three evaluators produces three PerTestResult rows,
each independently showing which specific check passed or failed. This is
what makes Phase 14's per-evaluator metrics ("faithfulness: 91% pass_rate")
possible without re-deriving them from something coarser later.

`latency_ms` / `token_usage_total` / `estimated_cost` on PerTestResult
describe the EVALUATOR's own cost (relevant once Day 8's LLM judge exists
-- a judge call costs money independently of the system being judged).
They are deliberately NOT a copy of the system's own generation cost
(SystemOutput.estimated_cost) -- copying it onto every evaluator row would
multiply a single generation's cost by the number of evaluators that
scored it. The system's cost is counted exactly once per test case, in
EvaluationRun.total_tokens/estimated_cost (see runner.py's aggregation).

`EvaluationRun.dataset_content_hash` goes beyond what Phase 11 literally
asked for (dataset_id + dataset_version) because version alone is only as
trustworthy as whoever remembered to bump it (Day 2's whole reason for
making content_hash a computed, never-stale field). Recording the hash
too means "exactly which dataset was used" is provable, not just claimed.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PerTestResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    test_id: str
    evaluator: str
    score: float
    threshold: float | None = None
    passed: bool
    reason: str = ""
    latency_ms: float | None = None
    token_usage_total: int | None = None
    estimated_cost: float | None = None
    failure_category: str | None = None  # one of failure_taxonomy.FailureCategory.ALL when passed=False (Day 9); None when passed=True
    metadata: dict = Field(default_factory=dict)


class EvaluationRun(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    dataset_id: str
    dataset_version: str
    dataset_content_hash: str
    system_name: str
    system_version: str
    model: str | None = None
    prompt_version: str | None = None
    configuration: dict = Field(default_factory=dict)
    started_at: datetime
    completed_at: datetime
    status: str  # "completed" -- see runner.py for what can fail a run outright

    total_tests: int
    passed_tests: int
    failed_tests: int
    pass_rate: float
    average_score: float
    total_tokens: int
    estimated_cost: float

    # The SYSTEM's own generation latency, aggregated once per test case
    # (Day 13) -- deliberately NOT on PerTestResult, whose own latency_ms
    # is the EVALUATOR's latency (see above). Copying the system's latency
    # onto every evaluator row scoring the same test case would be exactly
    # the kind of double-count Day 5's cost aggregation was already careful
    # to avoid; this follows that same once-per-test-case precedent.
    # None when no system call in this run reported a latency at all.
    average_system_latency_ms: float | None = None
    max_system_latency_ms: float | None = None

    results: list[PerTestResult] = Field(default_factory=list)
