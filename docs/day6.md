# Day 6 — Semantic similarity evaluation

## Provider abstraction

`eval_harness/providers/embedding.py` — `EmbeddingProvider` (`Protocol`): `model` + async `embed(texts) -> list[vectors]`. Two implementations:

- **`FakeEmbeddingProvider`** (same file, no extra dependency) — deterministic, offline, free. Builds a fixed-dimension vector via character-trigram feature hashing (the same family of technique as scikit-learn's `HashingVectorizer`). It has **no notion that "30" and "thirty" mean the same thing** — only that identical/near-identical text shares trigrams. It exists to test the evaluator's own *mechanics* (score math, threshold logic) offline, per the brief's explicit "do not make unit tests dependent on paid API calls."
- **`OpenAIEmbeddingProvider`** (`providers/openai_embedding.py`) — real embeddings. Made an **optional dependency** (`requirements-openai.txt`) on purpose: this is a proactive application of a bug Project 1 hit the hard way (Day 20 there — `hnswlib` had been an unconditional hard dependency even though most users never needed it). The base harness now works fully — deterministic evaluators, the runner, `MockSystem`, `FakeEmbeddingProvider` — with zero AI SDK installed.

## Phase 7: SemanticSimilarityEvaluator

`eval_harness/evaluators/semantic.py`. The five things the brief asked to be explained are documented directly in the module docstring, not buried in a commit message:

- **Embedding generation**: via the injected `EmbeddingProvider` — real or fake, the evaluator doesn't know or care which.
- **Similarity metric**: cosine similarity (`eval_harness/similarity.py`) — measures the *angle* between vectors, ignoring magnitude (which is mostly an artifact of text length, not meaning).
- **Normalization**: raw cosine similarity is mathematically in `[-1, 1]`; the score is clamped to `[0, 1]` so every evaluator in the harness reports on the same scale.
- **Threshold**: constructor default (0.80) or per-test-case override via `test_case.metadata['semantic_threshold']` — same pattern as Day 4.
- **Threshold limitations**: a cosine-similarity threshold is **not comparable across embedding models or model versions** — "0.80" means something different per model, and a provider upgrade can silently shift what a score means. It's a starting point to calibrate, not a universal constant — and it measures *topical closeness*, not *correctness*: it can't reliably tell "right answer, different words" apart from "wrong answer, same topic" the way a rubric-driven LLM judge (Day 8) can.

## Live test

`tests/live/test_openai_embedding.py` — the brief's own paraphrase example ("The refund period is 30 days" vs. "Customers have thirty days…"), run against the real provider. Skips automatically with no `OPENAI_API_KEY` — the identical skip-don't-fail convention Project 1's OCR tests established (Day 21 there). Run it explicitly with:

```
OPENAI_API_KEY=sk-... pytest tests/live -v
```

## Tests

21 new tests. **122 passed, 2 skipped** (the live tests — correctly skipped with no key set).

## Next

Day 7 — RAG-specific evaluators (retrieval Hit@K/Recall@K/MRR, context relevance, faithfulness, citation correctness/completeness) — reusing Day 6's similarity machinery and Project 1's own pure retrieval-metric functions as the pattern.
