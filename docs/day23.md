# Day 23 — Phase 39: failure injection and regression verification

`tests/integration/test_failure_injection.py` — 17 deterministic tests
proving the harness correctly detects and classifies every one of the 15
required synthetic failure scenarios, plus two supplementary tests for a
real gap found while writing them (see "What this surfaced" below).
Nothing here makes a real, paid LLM call: every judge is `FakeJsonLLM`-
mocked, every embedding is `FakeEmbeddingProvider`. These are synthetic
demonstrations of the harness's own detection machinery, never a claim
about any real production model's behavior.

## Scenario -> verification mapping

| # | Scenario | Verified via | Result |
|---|---|---|---|
| 1 | Correct answer, valid citations | `SemanticSimilarityEvaluator` + `CitationPrecisionEvaluator` | passes cleanly; quality gate passes, exit code 0 |
| 2 | Incorrect answer | `SemanticSimilarityEvaluator` | `THRESHOLD_MISS` |
| 3 | Partially correct, missing info | `SemanticSimilarityEvaluator` | score strictly between 0 and 1; `THRESHOLD_MISS` |
| 4 | Hallucinated/unsupported claim | correctness passes, `FaithfulnessJudgeEvaluator` (mocked) fails | proves correctness alone is insufficient |
| 5 | Irrelevant answer | `SemanticSimilarityEvaluator` | `THRESHOLD_MISS` |
| 6 | Missing citations | `CitationPrecisionEvaluator` (vacuous 1.0) vs `CitationRecallEvaluator` (0.0) | the two evaluators correctly diverge |
| 7 | Invalid/fabricated citation | `CitationPrecisionEvaluator` | score 0.0 |
| 8 | Citation exists but doesn't support the claim | precision passes, faithfulness judge (mocked) fails | proves citation-matching alone is insufficient |
| 9 | Poor retrieval, relevant doc missing | `HitAtKEvaluator` | score 0.0 |
| 10 | Malformed structured output | `JSONSchemaEvaluator` | clean `THRESHOLD_MISS`, never an unhandled exception |
| 11 | System timeout | `HttpError` from the adapter's transport | `DEPENDENCY_FAILURE` |
| 12 | Evaluator failure | `SemanticSimilarityEvaluator` against a case with no `expected_output` | `EVALUATION_FAILURE` -- explicitly asserted `!= DEPENDENCY_FAILURE` and `!= THRESHOLD_MISS`, and `passed is False` (never mistaken for success) |
| 13 | Infrastructure failure vs. poor quality | one run, three test cases (crash / bad answer / clean pass) | three distinct `failure_category` values in the same run |
| 14 | Increased latency | `average_system_latency_ms`, fast vs. slow system | observable, correctly ordered -- see limitation below |
| 15 | Increased token usage/cost | `compute_efficiency`, cheap vs. expensive system | `total_cost`/`total_tokens` correctly ordered -- see limitation below |

Also verified directly: a `ProductionRagAdapter` non-2xx response (the
real `HTTPException(502)` production-rag's own router raises for an
embedding-provider failure) surfaces as `DEPENDENCY_FAILURE`, not a
quality failure -- the adapter correction from Phase 38 and the failure
taxonomy from Day 9 working together correctly.

## Baseline-vs-candidate regression demonstration

Already built and tested: Day 20's `scripts/demo_end_to_end.py` +
`tests/integration/test_demo_end_to_end.py`. Not duplicated here, per
this project's own "reuse existing components" rule -- a candidate that
wins on pass rate and average score (and both per-evaluator comparisons)
still fails the quality gate, because it regressed one previously-passing
test case. See `docs/day20.md` for the full walkthrough.

## What this surfaced: two real, documented gaps

Writing scenarios 14 and 15 honestly (not just "does the number move")
surfaced two things worth stating plainly rather than glossing over:

1. **No latency-based quality gate exists.** `QualityGatePolicy`
   (Day 11) has `min_pass_rate`, `min_average_score`,
   `min_evaluator_pass_rate`, and `forbid_failure_categories` -- no
   latency field. Increased latency is fully *observable*
   (`EvaluationRun.average_system_latency_ms`/`max_system_latency_ms`,
   Day 13) but nothing in this harness can currently FAIL a run over it.
   `test_14_increased_latency_is_observable_in_run_aggregates` pins this
   exact behavior (both the fast and slow run "pass" on quality despite
   a 40x latency difference) so it's a documented limitation, not a
   silent gap. See the engineering report's production-readiness
   section.

2. **A run's total cost does not distinguish "not reported" from
   "genuinely free."** Day 13's `compute_efficiency` already makes this
   distinction correctly at the *cost-per-passing-test* level (`None`,
   not `0.0`, when nothing passed) -- but `EvaluationRun.estimated_cost`
   itself, summed directly in `runner.py`, has no such distinction: a
   system that never reports `estimated_cost` at all ends with
   `run.estimated_cost == 0.0`, identical to a system that genuinely
   cost nothing. `test_15b_cost_not_reported_is_indistinguishable_from_
   free_at_the_run_level` pins this. Not fixed here -- fixing it changes
   `EvaluationRun`'s schema (a `cost_reported: bool` field, or an
   `Optional[float]`), which is a real, separate decision outside this
   phase's "verify, then fix defects found" scope; it's recorded as a
   known gap for the next day that touches cost accounting.

Neither of these was a crash or a wrong result -- both are cases where
the harness currently CAN'T tell two different real situations apart.
Per this phase's own rule ("do not hide, suppress, or misrepresent
failures"), both are written up here and in the final report rather than
quietly left for someone to discover later.

## Result

**336 passed, 4 skipped** (up from 319 -- 17 new integration tests).

## Next

Phase 40 — a CLI (per the user's explicit scope decision: CLI only, no
HTTP API, no Docker), then the full Definition-of-Done checklist with
real evidence per item.
