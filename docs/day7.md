# Day 7 — RAG-specific evaluators

Phase 8. No new data-model fields needed — everything reads
`TestCase.reference_context` / `expected_citations` and
`SystemOutput.context` / `citations`, set aside for exactly this back on
Day 1/Day 3.

## Retrieval quality — `eval_harness/retrieval_metrics.py`

Pure functions: `compute_ranks`, `hit_at_k`, `recall_at_k`, `reciprocal_rank`.
A deliberate **generalization** of Project 1's own
`app/eval/retrieval_metrics.py` — identical formulas, but keyed on generic
string identifiers (context chunk text) instead of `int` page numbers, since
this harness can't assume every system retrieves pages.

Matching is **exact string equality**. A retrieved chunk that paraphrases an
expected one without matching it verbatim scores as a miss — a documented
limitation (a smarter fuzzy/overlap matcher is future work), not a bug.

Three evaluators wrap the metrics, one per metric (same one-class-per-check
convention as Day 4's deterministic evaluators):

- `HitAtKEvaluator(k=5)` — binary, did any expected context item appear in
  the top k?
- `RecallAtKEvaluator(k=5, threshold=1.0)` — fraction of expected items
  found in the top k.
- `MRREvaluator(threshold=0.5)` — 1 / rank of the first expected item found.

All three raise `ValueError` if the test case has no `reference_context` —
there is nothing to score retrieval against. `k` and thresholds are
overridable per test case via `metadata` (`retrieval_k`, `recall_threshold`,
`mrr_threshold`), the same constructor-default-or-metadata-override pattern
every evaluator in this harness uses.

## Context relevance & faithfulness — reusing Day 6's embedding machinery

- **`ContextRelevanceEvaluator`** — does retrieved context relate to the
  *question*? Average cosine similarity between `test_case.input` and each
  chunk in `output.context`. No context retrieved scores `0.0` (not an
  error — nothing relevant was retrieved).
- **`FaithfulnessEvaluator`** — does the *answer* actually come from the
  retrieved context? Cosine similarity between the answer and the
  concatenated context.

**Faithfulness is explicitly a proxy, not true groundedness checking** — the
evaluator's `reason` string says so on every result. Cosine similarity
measures topical closeness; it cannot tell "this answer is supported by the
context" apart from "this answer is topically similar to the context but
contradicts it outright." Real faithfulness checking needs an LLM reading
each claim against the context — that's Day 8 (LLM-as-judge). This evaluator
is a cheap, fast, embedding-only first pass, useful for catching wildly
ungrounded answers before paying for a judge call on every test case.

## Citation precision & completeness

- **`CitationPrecisionEvaluator`** — of the citations the system *gave*, how
  many were right? No citations given scores `1.0` (vacuously precise —
  nothing wrong was cited; *missing* citations is recall's job).
- **`CitationRecallEvaluator`** — of the citations the system *should* have
  given, how many did it give? No citations expected scores `1.0`
  (vacuously complete).

Both take `match_fields: list[str] | None` — `expected_citations` /
`citations` are deliberately not fixed-shape (Day 1), so matching defaults to
exact dict equality but can be narrowed to specific keys (e.g.
`match_fields=["page"]` to ignore a snippet field that will never match
verbatim).

Note the recall evaluator's "no expectation → 1.0" convention **deliberately
differs** from `retrieval_metrics.recall_at_k`'s "no expectation → 0.0."
`recall_at_k` is one number among several for a single retrieval question,
where 0.0 keeps the aggregate meaningful; `CitationRecallEvaluator` is a
standalone pass/fail check, where 0.0 would wrongly fail every legitimately
citation-less test case (e.g. a correctly refused/unanswerable question).
Documented in the evaluator's own docstring, not just here.

## Tests

32 new tests across `test_retrieval_metrics.py` (pure math),
`test_retrieval_evaluators.py`, `test_groundedness_evaluators.py`,
`test_citation_evaluators.py`. **154 passed, 2 skipped** (live tests,
correctly skipped with no `OPENAI_API_KEY`).

## Next

Day 8 — LLM-as-judge: rubric-driven scoring for the things embeddings can't
reliably judge (true faithfulness/entailment, correctness vs. "same topic,
wrong answer," open-ended quality rubrics).
