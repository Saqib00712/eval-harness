# Day 4 — Evaluator interface + deterministic evaluators

## Phase 5: the Evaluator abstraction

`eval_harness/evaluator.py` — `Evaluator` (a `@runtime_checkable` `Protocol`, same reasoning as `SystemUnderTest`) and `EvaluatorResult`, the one result shape every evaluator family returns:

```json
{"evaluator": "exact_match", "score": 1.0, "threshold": 1.0, "passed": true, "reason": "...", "metadata": {}}
```

`evaluate()` is `async` even though today's evaluators never await anything — so the Day 5 runner can mix free, instant deterministic evaluators with paid, slow LLM judges (Day 8) in one list, without a special case for either.

## Phase 6: deterministic evaluators

`eval_harness/evaluators/deterministic.py` — six evaluators, all free, instant, and 100% reproducible:

| Evaluator | Checks |
|---|---|
| `ExactMatchEvaluator` | `output.output` == `test_case.expected_output` (normalized) |
| `ContainsEvaluator` | required terms present (all-or-any mode) |
| `RegexEvaluator` | `output.output` matches a pattern |
| `JSONSchemaEvaluator` | `output.structured_data` validates against a real JSON Schema (`jsonschema` package) |
| `LengthEvaluator` | character/word count within `[min, max]` |
| `RequiredFieldsEvaluator` | required keys present and non-`None` in `output.structured_data` |

**Configuration pattern (shared by all six):** a value can be fixed once at construction (one rule applied to every test case) *or* overridden per test case via `test_case.metadata[...]` (different questions needing different checks). If neither is set, the evaluator raises `ValueError` rather than silently scoring 0 — a misconfigured evaluator should fail loudly during a run, not produce a number indistinguishable from a real failure.

New dependency: `jsonschema` (real Draft 2020-12 validation, not a hand-rolled approximation).

## Tests

31 new tests, all offline. **Total: 90 passed.**

## Next

Day 5 — the evaluation runner: wires a `Dataset` + a `SystemUnderTest` + a list of `Evaluator`s together into one end-to-end run, using `MockSystem` and these deterministic evaluators to prove it works with zero API calls before any real system or LLM judge gets involved.
