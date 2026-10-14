"""
Phase 13: failure taxonomy -- classifying WHY a test failed, not just
recording THAT it did.

`PerTestResult.failure_category` has existed since Day 5, but until today
it was only ever set in the two EXCEPTION paths the runner already caught
(`dependency_failure` -- the system under test crashed; `evaluation_failure`
-- the evaluator itself crashed). The far more common case -- an evaluator
runs perfectly fine and simply scores below its threshold -- left the field
`None` forever. A `None` category on a `passed=False` row isn't a taxonomy,
it's a gap: anyone trying to answer "of our failures, how many were crashes
vs. genuine quality misses" couldn't, because the two cases were
indistinguishable from `None`.

Today closes that gap. `FailureCategory` below is the complete, closed set
of three categories the runner (runner.py) now assigns to EVERY row where
`passed=False`:

  DEPENDENCY_FAILURE  -- the system under test raised. No output was ever
                         produced, so nothing about the harness's checks
                         applies -- this is an infrastructure/system
                         problem, not a quality problem.
  EVALUATION_FAILURE  -- the evaluator itself raised (misconfiguration, a
                         bad judge response, etc). The system DID produce
                         output; the CHECK is what broke.
  THRESHOLD_MISS       -- the normal case: the evaluator ran to completion
                         and scored the output below its threshold. This is
                         what "my system needs improvement" actually looks
                         like in the data, as opposed to "my pipeline is
                         broken."

Deliberately NOT in this taxonomy: a per-evaluator-family breakdown (e.g.
"retrieval_miss" vs "faithfulness_miss" vs "correctness_miss"). That
information already exists -- on `PerTestResult.evaluator`, the evaluator's
own `name` -- and hardcoding a name -> category mapping here would silently
mis-tag (or simply ignore) any evaluator someone plugs into the harness that
this module has never heard of, which cuts against this project's whole
reason for existing (Day 1: generic, not tied to any one app's evaluator
set). `summarize_failures` below gives you that cross-cut -- category BY
evaluator -- without the runner needing to know what any given evaluator
measures.
"""

from eval_harness.run_models import EvaluationRun


class FailureCategory:
    DEPENDENCY_FAILURE = "dependency_failure"
    EVALUATION_FAILURE = "evaluation_failure"
    THRESHOLD_MISS = "threshold_miss"

    ALL = (DEPENDENCY_FAILURE, EVALUATION_FAILURE, THRESHOLD_MISS)


def summarize_failures(run: EvaluationRun) -> dict:
    """Break down every failed PerTestResult in `run` by failure_category,
    and within each category, by which evaluator produced it. A result
    with `passed=True` is never a failure, whatever its (always-None)
    failure_category is, so it's excluded entirely -- this function
    answers "what went wrong", not "what happened".

    Returns e.g.:
        {
          "threshold_miss": {
              "count": 3,
              "by_evaluator": {"faithfulness": 2, "exact_match": 1},
              "test_ids": ["qa-002", "qa-004", "qa-007"],
          },
          "dependency_failure": {"count": 1, "by_evaluator": {...}, "test_ids": [...]},
        }
    A category with zero failures is simply absent from the result (not
    present with count 0) -- an empty dict means "nothing failed"."""
    summary: dict[str, dict] = {}
    for result in run.results:
        if result.passed:
            continue
        category = result.failure_category or "uncategorized"  # defensive: a run built
        # outside this harness's own runner (e.g. loaded from older storage,
        # or hand-constructed in a test) might have a failed row with no
        # category at all -- "uncategorized" says so honestly instead of
        # silently dropping it from every total.
        bucket = summary.setdefault(category, {"count": 0, "by_evaluator": {}, "test_ids": []})
        bucket["count"] += 1
        bucket["by_evaluator"][result.evaluator] = bucket["by_evaluator"].get(result.evaluator, 0) + 1
        bucket["test_ids"].append(result.test_id)
    return summary
