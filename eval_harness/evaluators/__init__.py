"""
Evaluator implementations, grouped by family:

  evaluators/deterministic.py  -- Day 4: no LLM call, no embeddings, free.
  evaluators/semantic.py       -- Day 6: embedding similarity.
  evaluators/rag.py            -- Day 7: retrieval/relevance/faithfulness/citations.
  evaluators/llm_judge.py      -- Day 8: rubric-driven LLM-as-judge (correctness, faithfulness, open rubrics).

The base Evaluator Protocol and EvaluatorResult live one level up, in
eval_harness/evaluator.py -- importable without pulling in every family's
dependencies (e.g. you can use the deterministic evaluators with no
OpenAI SDK installed at all).
"""

from eval_harness.evaluators.deterministic import (
    ContainsEvaluator,
    ExactMatchEvaluator,
    JSONSchemaEvaluator,
    LengthEvaluator,
    RegexEvaluator,
    RequiredFieldsEvaluator,
)
from eval_harness.evaluators.llm_judge import (
    CorrectnessJudgeEvaluator,
    FaithfulnessJudgeEvaluator,
    RubricJudgeEvaluator,
)
from eval_harness.evaluators.rag import (
    CitationPrecisionEvaluator,
    CitationRecallEvaluator,
    ContextRelevanceEvaluator,
    FaithfulnessEvaluator,
    HitAtKEvaluator,
    MRREvaluator,
    RecallAtKEvaluator,
)
from eval_harness.evaluators.semantic import SemanticSimilarityEvaluator

__all__ = [
    "ContainsEvaluator",
    "ExactMatchEvaluator",
    "JSONSchemaEvaluator",
    "LengthEvaluator",
    "RegexEvaluator",
    "RequiredFieldsEvaluator",
    "SemanticSimilarityEvaluator",
    "CitationPrecisionEvaluator",
    "CitationRecallEvaluator",
    "ContextRelevanceEvaluator",
    "FaithfulnessEvaluator",
    "HitAtKEvaluator",
    "MRREvaluator",
    "RecallAtKEvaluator",
    "CorrectnessJudgeEvaluator",
    "FaithfulnessJudgeEvaluator",
    "RubricJudgeEvaluator",
]
