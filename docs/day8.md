# Day 8 — LLM-as-judge

Phase 9. Fixes the two blind spots Days 6 and 7 each documented on
themselves up front, plus adds a third capability no other evaluator here
has at all.

## Provider abstraction — `eval_harness/providers/llm.py` + `openai_llm.py`

`JsonLLM` (`Protocol`): `complete_json(*, model, system, user, schema_name,
schema, max_tokens) -> (dict, TokenUsage)` — a direct reuse of Project 1's
own `JsonLLM` shape (`app/services/llm.py`): structured-output JSON mode
with a strict, named schema, so a judge's response is always a typed dict,
never free text to regex apart.

- **`FakeJsonLLM`** (same file) — scripted, offline responses (a single
  dict repeated for every call, or a list consumed in call order). Records
  every call it receives (`.calls`) so a test can assert on the exact
  prompt a judge sent, with zero API cost.
- **`OpenAIJsonLLM`** (`providers/openai_llm.py`) — the real, billed
  implementation. Kept behind the same optional `requirements-openai.txt`
  as Day 6's embedding provider — same `hnswlib`-avoidance discipline,
  applied a second time.

## Three judge evaluators — `eval_harness/evaluators/llm_judge.py`

- **`CorrectnessJudgeEvaluator`** — fixes Day 6's documented blind spot:
  `SemanticSimilarityEvaluator` can't tell "right answer, different words"
  from "wrong answer, same topic." An LLM reading the actual answer against
  the actual expected answer can. Requires `expected_output`.
- **`FaithfulnessJudgeEvaluator`** — fixes Day 7's documented blind spot:
  `rag.FaithfulnessEvaluator`'s cosine-similarity proxy can't tell
  "grounded in context" from "topically similar but contradicts it." An LLM
  reading the answer against the retrieved context can, approximately — a
  single holistic 0..10 judgment, not per-claim entailment checking (that's
  a further upgrade, not attempted here). No retrieved context scores 0.0
  **without calling the LLM** — there's nothing to check faithfulness
  against, so no reason to spend money on the call.
- **`RubricJudgeEvaluator`** — open-ended quality scoring against a
  free-text rubric (tone, completeness, structure, "did it actually answer
  what was asked"). The one evaluator in this harness that needs no fixed
  reference answer at all. Rubric is a constructor default, overridable per
  test case via `metadata["rubric"]`.

All three normalize the judge's raw `0..max_score` integer to this
harness's universal `[0, 1]` scale, same reasoning Day 6 used for cosine
similarity. All three raise `ValueError` on a missing required field or an
unusable judge response — never silently return a default score.

## Cost tracking — a small but real extension to `EvaluatorResult`

An LLM judge call costs real money, independently of whatever the system
under test cost. `EvaluatorResult` gained `token_usage_total` /
`estimated_cost` (both `None` for every pre-Day-8 evaluator — they're
free to run) and `runner.py` now copies them onto the matching
`PerTestResult`. `run_models.py`'s own docstring had already set this field
aside on Day 5 ("relevant once Day 8's LLM judge exists") — this is that
day. `eval_harness/cost.py` (`estimate_cost_usd`) is a direct, one-line
reuse of Project 1's `app/services/cost.py`. Pricing is opt-in per judge
evaluator (`price_per_million_tokens=...`); without it, cost fields stay
`None`, same as every other evaluator.

## Ordering note

These are the slowest, most expensive evaluators in the harness. The
runner doesn't enforce an order — nothing stops a caller from listing
cheap evaluators (deterministic, semantic, RAG) first and reserving judges
for what only an LLM can score, which is the sane default for any real
pipeline (the ordering `evaluator.py` has referenced as "Phase 21" since
Day 5).

## Tests

24 new unit tests (`test_cost.py`, `test_fake_json_llm.py`,
`test_llm_judge_evaluators.py`, `test_openai_json_llm.py`) plus one runner
integration test confirming evaluator-reported cost is added to
`EvaluationRun` totals separately from system cost (same double-counting
discipline Day 5's runner already enforced for system cost). Plus 2 new
live tests (`tests/live/test_openai_llm_judge.py`), skipped without a real
key. **178 passed, 4 skipped.**

## Next

Day 9 — failure taxonomy: turning `PerTestResult.failure_category` (a
placeholder field since Day 5) into a real, structured classification of
*why* a test failed, not just that it did.
