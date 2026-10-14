"""
Phase 5: the Evaluator abstraction + its standardized result.

Every evaluator -- deterministic (Day 4), semantic (Day 6), RAG-specific
(Day 7), or an LLM judge (Day 8) -- implements the same tiny interface and
returns the same result shape, so the runner (Day 5... the orchestrator,
not this day) can run a list of mixed evaluators over one SystemOutput
without caring which kind each one is.

`evaluate` is async even though today's deterministic evaluators never
await anything -- same reasoning as SystemUnderTest.run (Day 3): a uniform
interface lets the runner treat a free, instant ExactMatchEvaluator and a
paid, slow LLMJudgeEvaluator identically, which is exactly what Phase 21
("deterministic evaluators before LLM judges") needs to be able to express
as an ordering, not a special case.

`token_usage_total` / `estimated_cost` (added Day 8) let an evaluator that
itself calls a paid API -- an LLM judge -- report its OWN cost, separately
from the system-under-test's generation cost. Every evaluator before Day 8
(deterministic, semantic, RAG) leaves both `None`: they're free to run, so
there's nothing to report. The runner (runner.py) copies these straight
onto the matching PerTestResult; `run_models.py`'s docstring already
anticipated this exact field the day it was written ("relevant once Day
8's LLM judge exists").
"""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from eval_harness.models import TestCase
from eval_harness.system_output import SystemOutput


class EvaluatorResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    evaluator: str
    score: float
    threshold: float | None = None
    passed: bool
    reason: str = ""
    token_usage_total: int | None = None
    estimated_cost: float | None = None
    metadata: dict = Field(default_factory=dict)


@runtime_checkable
class Evaluator(Protocol):
    @property
    def name(self) -> str:
        """A short, stable identifier (e.g. "exact_match", "faithfulness"),
        used as EvaluatorResult.evaluator and as the key later days group
        per-evaluator metrics by (Phase 14: "faithfulness: pass_rate=91%")."""
        ...

    async def evaluate(self, test_case: TestCase, output: SystemOutput) -> EvaluatorResult:
        """Score one SystemOutput against the TestCase it was produced for.
        Must not raise for an ordinary scoring outcome (even a 0.0/failed
        score is a normal result, not an error) -- only for genuine
        misconfiguration (e.g. this evaluator needs a field the test case
        never set). The runner (Day 5) is responsible for deciding what an
        actual exception here means for the overall run."""
        ...
