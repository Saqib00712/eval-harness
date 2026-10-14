"""
Phase 7: semantic similarity evaluation.

  * EMBEDDING GENERATION: text -> vector via an injected EmbeddingProvider
    (Protocol, providers/embedding.py). Real semantic understanding needs a
    real embedding model (providers/openai_embedding.py's optional,
    OpenAI-backed provider); FakeEmbeddingProvider is a deterministic
    stand-in for testing THIS EVALUATOR'S OWN LOGIC offline, per Phase 7's
    explicit requirement ("do not make unit tests dependent on paid API
    calls") -- it has no notion of real-world meaning, only of shared
    character trigrams.

  * SIMILARITY METRIC: cosine similarity (similarity.py), the standard
    choice for embeddings -- it measures the ANGLE between two vectors and
    ignores their magnitude, and embedding magnitude is mostly an artifact
    of text length, not of meaning.

  * NORMALIZATION: raw cosine similarity is mathematically in [-1, 1]; the
    score here is clamped to [0, 1] (negative values floored at 0) so every
    evaluator in this harness reports scores on the same scale. Real
    embedding models rarely produce meaningfully negative similarities for
    ordinary text, so this clamp loses little real signal.

  * THRESHOLD: configurable per evaluator instance (applies to every test
    case), overridden per-case via
    `test_case.metadata['semantic_threshold']` -- same pattern Day 4's
    deterministic evaluators use. Default 0.80.

  * THRESHOLD LIMITATIONS (read before trusting a default blindly):
    cosine-similarity thresholds are NOT comparable across embedding
    models, or even across versions of the SAME model -- "0.80" means
    something different for text-embedding-3-small than for a different
    model entirely, and a provider upgrade can silently shift what a given
    score means. A fixed threshold is a starting point to CALIBRATE against
    real examples (Phase 22's human-calibration foundation, not yet built
    in this harness), never a universal constant to trust blindly.
    Semantic similarity also measures topical/lexical closeness, not
    correctness -- it cannot reliably distinguish "says the same thing in
    different words" from "talks about the right topic but gets the actual
    answer wrong" as well as a rubric-driven LLM judge can (Day 8).
"""

from eval_harness.evaluator import EvaluatorResult
from eval_harness.models import TestCase
from eval_harness.providers.embedding import EmbeddingProvider
from eval_harness.similarity import cosine_similarity
from eval_harness.system_output import SystemOutput


class SemanticSimilarityEvaluator:
    def __init__(self, embedder: EmbeddingProvider, *, threshold: float = 0.80):
        self._embedder = embedder
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "semantic_similarity"

    async def evaluate(self, test_case: TestCase, output: SystemOutput) -> EvaluatorResult:
        expected = test_case.expected_output
        if expected is None:
            raise ValueError(
                f"{self.name}: test case {test_case.test_id!r} has no expected_output to compare against."
            )
        actual = output.output or ""
        threshold = test_case.metadata.get("semantic_threshold", self._threshold)

        vectors = await self._embedder.embed([expected, actual])
        raw_similarity = cosine_similarity(vectors[0], vectors[1])
        score = max(0.0, min(1.0, raw_similarity))  # normalization: clamp [-1, 1] -> [0, 1]
        passed = score >= threshold

        return EvaluatorResult(
            evaluator=self.name, score=round(score, 4), threshold=threshold, passed=passed,
            reason=f"Cosine similarity {score:.4f} {'meets' if passed else 'is below'} threshold {threshold}.",
            metadata={"embedding_model": self._embedder.model, "raw_cosine_similarity": round(raw_similarity, 4)},
        )
