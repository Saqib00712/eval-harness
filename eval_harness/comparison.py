"""
Phase 18: model/prompt comparison -- running the SAME dataset through two
or more systems (different models, different prompt versions, a retrieval
config change, anything satisfying `SystemUnderTest`) and reporting which
one actually wins, PER EVALUATOR, not just on one overall number.

"Per evaluator, not just overall" is the whole point of this module, not
an incidental feature: two systems can tie (or nearly tie) on overall pass
rate while one is clearly better at retrieval and the other clearly better
at faithfulness -- a single "winner" number would hide exactly the
trade-off a real decision needs to see. `compare_systems` reuses Day 11's
`per_evaluator_pass_rates` for this rather than re-deriving it, and reuses
Day 5's `run_evaluation` to actually produce each system's run rather than
reimplementing any part of scoring here -- this module's only job is
running several systems through the SAME existing pipeline and comparing
their results.

SEQUENTIAL, NOT PARALLEL, ON PURPOSE: `compare_systems` runs each system's
full evaluation one at a time rather than concurrently (e.g. via
`asyncio.gather`). Running N systems' worth of (possibly paid, possibly
rate-limited) LLM calls all at once trades a real, predictable cost/rate
profile for a faster wall-clock time most comparison workflows don't
need -- and if a system's run throws, attributing the failure to exactly
which system is immediate rather than requiring `gather`'s exception
bookkeeping. A caller who genuinely wants concurrent runs can call
`run_evaluation` directly for each system themselves; this module
optimizes for predictability and honest cost by default, not raw speed.

TIES ARE REPORTED AS TIES, NEVER BROKEN ARBITRARILY: if two or more
systems share the best score (after rounding to the harness's usual 4
decimal places, to avoid floating-point noise manufacturing a fake
winner), the winner is `None`, not whichever system happened to come
first in the input list or sort alphabetically first -- either of those
would silently fabricate a "winner" that isn't actually one.
"""

from pydantic import BaseModel, ConfigDict, Field

from eval_harness.dataset import Dataset
from eval_harness.evaluator import Evaluator
from eval_harness.quality_gate import per_evaluator_pass_rates
from eval_harness.run_models import EvaluationRun
from eval_harness.runner import run_evaluation
from eval_harness.system import SystemUnderTest


class ComparisonReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset_id: str
    system_names: list[str]
    runs: dict[str, EvaluationRun]

    overall_pass_rate: dict[str, float]
    overall_average_score: dict[str, float]
    overall_winner: str | None  # None means a tie, not "no data"

    per_evaluator_pass_rate: dict[str, dict[str, float]] = Field(default_factory=dict)
    per_evaluator_winner: dict[str, str | None] = Field(default_factory=dict)


def _winner(scores: dict[str, float]) -> str | None:
    rounded = {name: round(score, 4) for name, score in scores.items()}
    best = max(rounded.values())
    leaders = [name for name, score in rounded.items() if score == best]
    return leaders[0] if len(leaders) == 1 else None


async def compare_systems(
    dataset: Dataset,
    systems: list[SystemUnderTest],
    evaluators: list[Evaluator],
    *,
    model: str | None = None,
    prompt_version: str | None = None,
    configuration: dict | None = None,
) -> ComparisonReport:
    if len(systems) < 2:
        raise ValueError(f"compare_systems needs at least two systems to compare, got {len(systems)}.")
    names = [s.name for s in systems]
    if len(set(names)) != len(names):
        raise ValueError(f"system names must be unique to compare them -- got {names}.")

    runs: dict[str, EvaluationRun] = {}
    for system in systems:
        runs[system.name] = await run_evaluation(
            dataset, system, evaluators, model=model, prompt_version=prompt_version, configuration=configuration
        )

    overall_pass_rate = {name: run.pass_rate for name, run in runs.items()}
    overall_average_score = {name: run.average_score for name, run in runs.items()}

    per_evaluator_pass_rate: dict[str, dict[str, float]] = {}
    for name, run in runs.items():
        for evaluator_name, rate in per_evaluator_pass_rates(run).items():
            per_evaluator_pass_rate.setdefault(evaluator_name, {})[name] = rate

    return ComparisonReport(
        dataset_id=dataset.dataset_id, system_names=names, runs=runs,
        overall_pass_rate=overall_pass_rate, overall_average_score=overall_average_score,
        overall_winner=_winner(overall_pass_rate),
        per_evaluator_pass_rate=per_evaluator_pass_rate,
        per_evaluator_winner={name: _winner(rates) for name, rates in per_evaluator_pass_rate.items()},
    )
