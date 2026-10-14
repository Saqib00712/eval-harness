# Day 18 — Full test suite audit

Phase 29. A pass over the whole test suite (263 tests at the start of
today) looking for real coverage gaps rather than guessing at them.
Written as an audit report, same format as Day 17: what was checked, what
was found, what was fixed — including one genuine bug the audit itself
surfaced.

## Method

Speculating about what "feels" undertested is how you miss things and
waste time padding tests that don't add coverage. Instead:

```
pip install coverage
python3 -m coverage run -m pytest -q
python3 -m coverage report -m --include="eval_harness/*"
```

`providers/openai_embedding.py` and `providers/openai_llm.py` were
excluded from the "gap" list below on purpose — their uncovered lines are
the live OpenAI call paths, which are deliberately exercised only by
`tests/live/` (real API key required, not run in CI). Every other module
in `eval_harness/` was at 100% or had a small number of genuinely missed
lines worth a look.

## Gaps found and fixed

**1. `evaluators/deterministic.py` — `LengthEvaluator`'s invalid-`unit`
guard was never tested.** The constructor raises `ValueError` for any
`unit` other than `"chars"`/`"words"`, but no test ever passed a bad one.
Added `test_length_rejects_invalid_unit`.

**2. `evaluators/llm_judge.py` — `_JudgeEvaluatorBase._judge`'s
call-failure path was never tested.** Existing tests covered a judge
returning a *badly shaped* response (`FakeJsonLLM` returning nonsense
JSON); none covered the call itself raising `LLMError` — a separate
`except` branch that wraps it as a `ValueError`. Added
`test_correctness_llm_call_failure_wrapped_as_value_error`, using a
minimal fake LLM whose `complete_json` always raises.

**3. `evaluators/rag.py` — `RecallAtKEvaluator`'s missing-`reference_context`
guard was never tested.** Its siblings `HitAtKEvaluator` and
`MRREvaluator` both had this exact guard tested; `RecallAtKEvaluator`'s
copy of it didn't. Added `test_recall_at_k_requires_reference_context`.

**4. `observability.py` — `StructuredLogObserver` only had 2 of its 6
hooks exercised.** `on_test_case_start` and `on_system_output` had a
test; `on_system_error`, `on_evaluator_error`, `on_evaluator_result`, and
`on_run_complete` had none. Added
`test_structured_log_observer_emits_remaining_hooks`, calling all four
and asserting on both the log messages and specific structured fields
(`error_type`, `run_id`).

**5. `storage.py` — `load_labels()`'s exception-handling branches were
never tested, and testing them found a real bug.** Writing a test that
fed `load_labels` malformed JSON and expected `StorageError` with a
"not valid JSON" message (mirroring `load_run`'s existing test) failed:

```
AssertionError: Regex pattern did not match.
 Regex: 'not valid JSON'
 Input: "...bad.json failed validation as a list of HumanLabels: 1
 validation error for list[HumanLabel]\n  Invalid JSON: key must be a
 string at line 1 column 2 [type=json_invalid, ...]"
```

`load_run` parses JSON itself (`json.loads()`) in a separate step before
validating, so it can raise `json.JSONDecodeError` on its own. `load_labels`
uses `TypeAdapter.validate_json()`, which parses *and* validates in one
call — malformed JSON there surfaces as a Pydantic `ValidationError`
(`json_invalid`), never a `json.JSONDecodeError`. The `except
json.JSONDecodeError` branch in `load_labels()` was consequently **dead
code**: unreachable, and nothing in the suite had ever proven that until
a test was written to actually hit it. Fixed by removing the dead branch
and correcting the test's expectation to match what the function
actually does (a `StorageError` with "failed validation", not "not valid
JSON") — with a comment on both sides explaining the difference from
`load_run` so a future reader doesn't "fix" it back. Also added
`test_load_labels_wrong_shape_raises` for the structurally-valid-JSON
but wrong-shape case, which was likewise untested.

**6. `tests/unit/test_evaluator.py` — the `Evaluator` Protocol conformance
test never actually called the conforming object.** `test_a_conforming_
object_satisfies_the_protocol` only checked `isinstance(_FakeEvaluator(),
Evaluator)` — because `Evaluator` is `@runtime_checkable`, `isinstance`
only inspects attribute *names*, not behavior. Neither line of
`_FakeEvaluator`'s own body (`.name`, `.evaluate()`) had ever actually
run. Added `test_a_conforming_object_actually_behaves_like_an_evaluator`,
which calls both and asserts on the result.

## Not changed

No redundant or stale tests were removed. Everything else the audit
turned up (the live-only OpenAI provider branches) is correctly excluded
from the normal suite by design, not a gap — re-confirmed rather than
assumed.

## Result

7 new tests. **270 passed, 4 skipped.** Coverage of `eval_harness/*`
(excluding the two OpenAI live-call modules) is 100%.

## Next

Day 19 — Project 1 integration adapter: a `SystemUnderTest` implementation
that wraps the `production-rag` repo (Project 1) so it can be evaluated
by this harness end-to-end, without this package ever importing from
that repo.
