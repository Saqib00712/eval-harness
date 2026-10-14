"""
Phase 8: RAG-specific evaluators. Five families, all reading the fields
Day 1/Day 3 already set aside for exactly this (`TestCase.reference_context`
/ `expected_citations`, `SystemOutput.context` / `citations`) -- nothing new
added to the data model for this day.

  RETRIEVAL QUALITY (hit@k / recall@k / MRR): pure-math, reused straight
  from `eval_harness/retrieval_metrics.py` -- a generalization of Project
  1's own app/eval/retrieval_metrics.py (same formulas, generic string
  identifiers instead of int page numbers). One evaluator class per metric,
  matching Day 4's deterministic-evaluator convention (one class per check,
  not one do-everything class).

  CONTEXT RELEVANCE: does the RETRIEVED context actually relate to the
  QUESTION? Reuses Day 6's EmbeddingProvider + cosine_similarity --
  average similarity between the query and each retrieved chunk. A proxy,
  not ground truth: embeddings measure topical closeness, the same caveat
  Day 6 documented for SemanticSimilarityEvaluator.

  FAITHFULNESS / GROUNDEDNESS: does the ANSWER actually come from the
  retrieved context, or did the system make something up? Approximated
  here as embedding similarity between the answer and the concatenated
  context -- an explicitly DOCUMENTED LIMITATION (not a design flaw): true
  faithfulness checking needs semantic entailment ("does the context
  SUPPORT this specific claim"), which plain cosine similarity cannot do.
  An answer can be topically similar to its context while contradicting it
  outright, and this evaluator would still score it high. Real faithfulness
  checking needs an LLM judge reading the claim against the context
  sentence-by-sentence -- that is Day 8's job (LLM-as-judge), not this
  evaluator's. This one is a cheap, fast, embedding-only first pass.

  CITATION PRECISION / COMPLETENESS: of the citations the system actually
  gave, how many were right (precision)? Of the citations it SHOULD have
  given, how many did it give (completeness/recall)? Matching is
  configurable because `expected_citations`/`citations` are deliberately
  not fixed-shape (Day 1) -- one app's citation is {"page": 4}, another's
  is {"url": ..., "section": ...}. Default: exact dict equality. Pass
  `match_fields=[...]` to match on a subset of keys instead (e.g. just
  "page", ignoring a "snippet" field that will never match verbatim).
"""

from eval_harness.evaluator import EvaluatorResult
from eval_harness.models import TestCase
from eval_harness.providers.embedding import EmbeddingProvider
from eval_harness.retrieval_metrics import compute_ranks, hit_at_k, recall_at_k, reciprocal_rank
from eval_harness.similarity import cosine_similarity
from eval_harness.system_output import SystemOutput


def _citations_match(actual: dict, expected: dict, match_fields: list[str] | None) -> bool:
    if match_fields is None:
        return actual == expected
    return all(actual.get(f) == expected.get(f) for f in match_fields)


class HitAtKEvaluator:
    """Did at least one expected context item appear in the top k retrieved
    items? Binary -- score is 1.0 or 0.0. `k` is a constructor default,
    overridable per test case via `test_case.metadata['retrieval_k']`, the
    same pattern every other configurable evaluator here uses."""

    def __init__(self, k: int = 5):
        self._k = k

    @property
    def name(self) -> str:
        return "hit_at_k"

    async def evaluate(self, test_case: TestCase, output: SystemOutput) -> EvaluatorResult:
        if not test_case.reference_context:
            raise ValueError(f"{self.name}: test case {test_case.test_id!r} has no reference_context.")
        k = test_case.metadata.get("retrieval_k", self._k)
        ranks = compute_ranks(output.context, test_case.reference_context)
        hit = hit_at_k(ranks, k)
        score = 1.0 if hit else 0.0
        return EvaluatorResult(
            evaluator=self.name, score=score, threshold=1.0, passed=hit,
            reason=f"{'Found' if hit else 'Did not find'} an expected context item in the top {k} retrieved.",
            metadata={"k": k, "ranks": ranks},
        )


class RecallAtKEvaluator:
    """Of ALL expected context items, what fraction appeared in the top k
    retrieved? Threshold default 1.0 (require full recall to "pass") --
    overridable per instance or per test case, same as every other scored
    evaluator here."""

    def __init__(self, k: int = 5, *, threshold: float = 1.0):
        self._k = k
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "recall_at_k"

    async def evaluate(self, test_case: TestCase, output: SystemOutput) -> EvaluatorResult:
        if not test_case.reference_context:
            raise ValueError(f"{self.name}: test case {test_case.test_id!r} has no reference_context.")
        k = test_case.metadata.get("retrieval_k", self._k)
        threshold = test_case.metadata.get("recall_threshold", self._threshold)
        ranks = compute_ranks(output.context, test_case.reference_context)
        score = recall_at_k(ranks, test_case.reference_context, k)
        passed = score >= threshold
        return EvaluatorResult(
            evaluator=self.name, score=round(score, 4), threshold=threshold, passed=passed,
            reason=f"Recall@{k} = {score:.4f} ({'meets' if passed else 'is below'} threshold {threshold}).",
            metadata={"k": k, "ranks": ranks},
        )


class MRREvaluator:
    """Mean Reciprocal Rank for a single test case: 1 / rank of the FIRST
    expected context item found, 0.0 if never found. Rewards finding the
    right context EARLY, not just eventually."""

    def __init__(self, *, threshold: float = 0.5):
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "mrr"

    async def evaluate(self, test_case: TestCase, output: SystemOutput) -> EvaluatorResult:
        if not test_case.reference_context:
            raise ValueError(f"{self.name}: test case {test_case.test_id!r} has no reference_context.")
        threshold = test_case.metadata.get("mrr_threshold", self._threshold)
        ranks = compute_ranks(output.context, test_case.reference_context)
        score = reciprocal_rank(ranks)
        passed = score >= threshold
        return EvaluatorResult(
            evaluator=self.name, score=round(score, 4), threshold=threshold, passed=passed,
            reason=f"Reciprocal rank {score:.4f} ({'meets' if passed else 'is below'} threshold {threshold}).",
            metadata={"ranks": ranks},
        )


class ContextRelevanceEvaluator:
    """Does the retrieved context actually relate to the QUESTION? Average
    cosine similarity between `test_case.input` and each retrieved chunk in
    `output.context`. No retrieved context at all scores 0.0 (nothing
    relevant was retrieved, which is the correct verdict, not an error)."""

    def __init__(self, embedder: EmbeddingProvider, *, threshold: float = 0.5):
        self._embedder = embedder
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "context_relevance"

    async def evaluate(self, test_case: TestCase, output: SystemOutput) -> EvaluatorResult:
        threshold = test_case.metadata.get("context_relevance_threshold", self._threshold)
        if not output.context:
            return EvaluatorResult(
                evaluator=self.name, score=0.0, threshold=threshold, passed=False,
                reason="No context was retrieved.", metadata={"chunk_count": 0},
            )
        vectors = await self._embedder.embed([test_case.input, *output.context])
        query_vec, chunk_vecs = vectors[0], vectors[1:]
        similarities = [max(0.0, min(1.0, cosine_similarity(query_vec, v))) for v in chunk_vecs]
        score = sum(similarities) / len(similarities)
        passed = score >= threshold
        return EvaluatorResult(
            evaluator=self.name, score=round(score, 4), threshold=threshold, passed=passed,
            reason=f"Average query-context similarity {score:.4f} "
                   f"({'meets' if passed else 'is below'} threshold {threshold}).",
            metadata={"embedding_model": self._embedder.model, "per_chunk_similarity": [round(s, 4) for s in similarities]},
        )


class FaithfulnessEvaluator:
    """Approximate groundedness: cosine similarity between the ANSWER and
    its concatenated retrieved CONTEXT. Read the module docstring above --
    this is a fast, cheap, embedding-only proxy, not true entailment
    checking; it cannot catch an answer that is topically on-point but
    factually contradicts its own context. Use Day 8's LLM judge for that.
    No retrieved context scores 0.0 -- an answer cannot be "grounded" in
    context that doesn't exist."""

    def __init__(self, embedder: EmbeddingProvider, *, threshold: float = 0.5):
        self._embedder = embedder
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "faithfulness"

    async def evaluate(self, test_case: TestCase, output: SystemOutput) -> EvaluatorResult:
        threshold = test_case.metadata.get("faithfulness_threshold", self._threshold)
        if output.output is None:
            raise ValueError(f"{self.name}: output has no `output` text to check for faithfulness.")
        if not output.context:
            return EvaluatorResult(
                evaluator=self.name, score=0.0, threshold=threshold, passed=False,
                reason="No retrieved context to ground the answer in.", metadata={"chunk_count": 0},
            )
        joined_context = "\n\n".join(output.context)
        vectors = await self._embedder.embed([output.output, joined_context])
        raw_similarity = cosine_similarity(vectors[0], vectors[1])
        score = max(0.0, min(1.0, raw_similarity))
        passed = score >= threshold
        return EvaluatorResult(
            evaluator=self.name, score=round(score, 4), threshold=threshold, passed=passed,
            reason=f"Answer-context similarity {score:.4f} "
                   f"({'meets' if passed else 'is below'} threshold {threshold}) -- "
                   f"an embedding-similarity PROXY for faithfulness, not true entailment checking.",
            metadata={"embedding_model": self._embedder.model, "raw_cosine_similarity": round(raw_similarity, 4)},
        )


class CitationPrecisionEvaluator:
    """Of the citations the system actually GAVE, how many were correct?
    No citations given scores 1.0 (vacuously correct -- a citation that was
    never made cannot be a wrong one; whether citations were MISSING
    is CitationRecallEvaluator's job, not this one's)."""

    def __init__(self, *, match_fields: list[str] | None = None, threshold: float = 1.0):
        self._match_fields = match_fields
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "citation_precision"

    async def evaluate(self, test_case: TestCase, output: SystemOutput) -> EvaluatorResult:
        threshold = test_case.metadata.get("citation_precision_threshold", self._threshold)
        if not output.citations:
            return EvaluatorResult(
                evaluator=self.name, score=1.0, threshold=threshold, passed=True,
                reason="No citations were given, so none were wrong.", metadata={"citation_count": 0},
            )
        expected = test_case.expected_citations
        matches = sum(
            1 for c in output.citations if any(_citations_match(c, e, self._match_fields) for e in expected)
        )
        score = matches / len(output.citations)
        passed = score >= threshold
        return EvaluatorResult(
            evaluator=self.name, score=round(score, 4), threshold=threshold, passed=passed,
            reason=f"{matches}/{len(output.citations)} given citations matched an expected one.",
            metadata={"match_fields": self._match_fields},
        )


class CitationRecallEvaluator:
    """Of the citations the system SHOULD have given, how many did it
    actually give? No expected citations scores 1.0 (vacuously complete --
    nothing was supposed to be cited, so nothing can be missing). This
    deliberately differs from `retrieval_metrics.recall_at_k`'s convention
    of scoring 0.0 when nothing is expected: that function is one input
    among several metrics for a single retrieval question, where 0.0 keeps
    the number meaningful in aggregate; this evaluator stands alone as a
    pass/fail check, where 0.0 would wrongly fail every test case that
    legitimately has no citations to give (e.g. a refused/unanswerable
    question)."""

    def __init__(self, *, match_fields: list[str] | None = None, threshold: float = 1.0):
        self._match_fields = match_fields
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "citation_recall"

    async def evaluate(self, test_case: TestCase, output: SystemOutput) -> EvaluatorResult:
        threshold = test_case.metadata.get("citation_recall_threshold", self._threshold)
        expected = test_case.expected_citations
        if not expected:
            return EvaluatorResult(
                evaluator=self.name, score=1.0, threshold=threshold, passed=True,
                reason="No citations were expected, so none are missing.", metadata={"expected_count": 0},
            )
        matches = sum(
            1 for e in expected if any(_citations_match(c, e, self._match_fields) for c in output.citations)
        )
        score = matches / len(expected)
        passed = score >= threshold
        return EvaluatorResult(
            evaluator=self.name, score=round(score, 4), threshold=threshold, passed=passed,
            reason=f"{matches}/{len(expected)} expected citations were given.",
            metadata={"match_fields": self._match_fields},
        )
