# eval-harness

A reusable, production-grade evaluation platform for AI systems — RAG applications, plain LLM apps, agents, classifiers, and structured-output systems. Answers one question: **how do we automatically know whether an AI system is getting better or worse?**

Built day-by-day as Project 2 of a 43-project AI Engineering curriculum. Project 1 (`production-rag`) is one example system this harness evaluates — this package never depends on it.

---

## What This Project Does

Most AI projects ship without any automated quality gate. When you change a prompt or swap a model, you have no systematic way to know if the system got better or worse. This harness fixes that.

```
Your AI System (RAG app / LLM app / agent / classifier)
    ↓
Dataset of test cases with expected outputs
    ↓
Evaluators — deterministic, semantic, LLM-judge, RAG-specific
    ↓
EvaluationRun — per-test scores, failure taxonomy, cost breakdown
    ↓
Regression check vs baseline + Quality gate (CI pass/fail)
    ↓
Dashboard — browse runs, compare versions, spot regressions
```

---

## Built Day by Day — 25 Days

| Day | What Was Built |
|-----|---------------|
| 1–2 | `TestCase` data model, JSON/JSONL loader, `Dataset` with content hash |
| 3 | `SystemUnderTest` interface, `SystemOutput`, `MockSystem` |
| 4 | `Evaluator`/`EvaluatorResult` interface, 6 deterministic evaluators (exact match, contains, regex, JSON schema, length, required fields) |
| 5 | `EvaluationRun`, `PerTestResult`, `run_evaluation()` — full end-to-end run with failure isolation |
| 6–7 | `EmbeddingProvider`, `OpenAIEmbeddingProvider`, `SemanticSimilarityEvaluator` |
| 8 | `retrieval_metrics.py` — pure hit@k / recall@k / MRR math |
| 9 | RAG evaluators — `HitAtK`, `RecallAtK`, `MRR`, `ContextRelevance`, `Faithfulness`, `CitationPrecision`, `CitationRecall` |
| 10 | `cost.py` — token→USD cost math |
| 11 | `JsonLLM`/`FakeJsonLLM`, `OpenAIJsonLLM` |
| 12 | LLM-judge evaluators — `CorrectnessJudge`, `FaithfulnessJudge`, `RubricJudge` |
| 13 | `FailureCategory`, `summarize_failures()` — classifies every failure as dependency / evaluation / threshold miss |
| 14 | `RegressionPolicy`, `compare_runs()` — flags regressions between baseline and candidate |
| 15 | `QualityGatePolicy`, `evaluate_quality_gate()`, `exit_code_for()` — hard CI pass/fail verdict |
| 16 | `compare_systems()` — run same dataset through N systems, report winner per evaluator |
| 17 | `efficiency.py` — cost/tokens per passing test, latency aggregation; security review (ReDoS caps, path-traversal hardening, dataset size ceilings) |
| 18 | Full test suite audit via `coverage.py` — 100% coverage on `eval_harness/*`; surfaced and fixed dead-code bug in `storage.py` |
| 19 | `HttpClient`/`FakeHttpClient`/`HttpxHttpClient`; `ProductionRagAdapter` — wraps Project 1's `/ask` endpoint as a `SystemUnderTest` over HTTP |
| 20 | `scripts/demo_end_to_end.py` — full demonstration: comparison + regression + quality gate + efficiency + storage + observability |
| 21 | Two-tier GitHub Actions CI — free `test` job (full suite + coverage gate) + `live-eval` job (real OpenAI, main only) |
| 22 | `ProductionRagAdapter` corrected against the real production-rag source — real nested `Citation`/`CostBreakdown` shapes, proper `latency_ms` measurement |
| 23 | 17-test failure injection suite — proves harness detects all 15 synthetic failure scenarios (wrong answer, hallucination, missing citations, timeout, evaluator crash, latency/cost regressions) |
| 24 | CLI — `validate-dataset`, `quality-gate`, `compare`, `demo` — CI-usable exit codes |
| 25 | Static browser dashboard — single-run summary, baseline-vs-candidate comparison, live quality-gate panel |

**Final: 348 tests — 100% coverage on all harness modules.**

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      EVAL HARNESS                            │
├──────────────────────────────────────────────────────────────┤
│  Input                                                       │
│  ├── Dataset (.json / .jsonl) — named, versioned, hashed     │
│  └── SystemUnderTest — your AI system via HTTP or in-process │
│                                                              │
│  Evaluators                                                  │
│  ├── Deterministic — exact match, contains, regex,          │
│  │                   JSON schema, length, required fields    │
│  ├── Semantic — embedding cosine similarity                  │
│  ├── RAG-specific — hit@k, recall@k, MRR,                   │
│  │                  context relevance, faithfulness,         │
│  │                  citation precision + recall              │
│  └── LLM-judge — correctness, faithfulness, rubric          │
│                                                              │
│  Run Engine                                                  │
│  ├── run_evaluation() — failure-isolated per-test execution  │
│  ├── FailureTaxonomy — dependency / eval / threshold miss    │
│  ├── CostAccounting — token→USD per evaluator + total        │
│  └── Observability — StructuredLogObserver, slowest/costliest│
│                                                              │
│  Analysis                                                    │
│  ├── RegressionPolicy — baseline vs candidate diff          │
│  ├── QualityGate — CI pass/fail with exit code              │
│  ├── compare_systems() — N-system comparison                 │
│  ├── efficiency() — cost per passing test                    │
│  └── calibration() — evaluator vs human label agreement     │
│                                                              │
│  Persistence                                                 │
│  └── RunStore — runs + labels to disk, keyed by dataset+system│
│                                                              │
│  Interfaces                                                  │
│  ├── CLI — validate-dataset, quality-gate, compare, demo    │
│  ├── GitHub Actions CI — two-tier (free + live-eval)        │
│  └── Static Dashboard — browse + compare runs in browser    │
└──────────────────────────────────────────────────────────────┘
```

---

## Evaluator Types

| Type | Evaluators | Cost |
|------|-----------|------|
| Deterministic | Exact match, Contains, Regex, JSON Schema, Length, Required fields | Free |
| Semantic | Cosine similarity via embeddings | Embedding API |
| RAG-specific | Hit@k, Recall@k, MRR, Context relevance, Faithfulness, Citation precision, Citation recall | Embedding API |
| LLM-judge | Correctness, Faithfulness, Rubric | LLM API |

All evaluators are **pluggable** — implement the `Evaluator` interface and pass it to `run_evaluation()`.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![OpenAI](https://img.shields.io/badge/OpenAI-Embeddings%20%2B%20LLM%20Judge-black?style=flat-square)
![pytest](https://img.shields.io/badge/pytest-348%20tests-green?style=flat-square)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-blue?style=flat-square)

- **Python 3.10+** — core harness, pure stdlib where possible
- **OpenAI** — optional: embeddings for semantic eval, GPT-4o for LLM-judge
- **httpx** — optional: HTTP transport for `ProductionRagAdapter`
- **pytest + coverage.py** — 100% coverage gate on all harness modules
- **GitHub Actions** — two-tier CI (free always-on + live-eval on main)
- **Vanilla HTML/JS** — zero-dependency static dashboard

---

## Project Structure

```
eval-harness/
│
├── eval_harness/
│   ├── models.py                     # TestCase — generic eval data model
│   ├── loader.py                     # Load test cases from .json / .jsonl
│   ├── dataset.py / dataset_io.py    # Dataset — named, versioned, hashed
│   ├── system.py                     # SystemUnderTest interface
│   ├── system_output.py              # StandardizedOutput
│   ├── mock_system.py                # MockSystem for harness testing
│   ├── evaluator.py                  # Evaluator / EvaluatorResult interface
│   ├── run_models.py / runner.py     # EvaluationRun + run_evaluation()
│   ├── retrieval_metrics.py          # Pure hit@k / recall@k / MRR math
│   ├── failure_taxonomy.py           # Failure classification
│   ├── regression.py                 # Baseline vs candidate comparison
│   ├── quality_gate.py               # CI pass/fail verdict
│   ├── comparison.py                 # N-system comparison
│   ├── efficiency.py                 # Cost per passing test
│   ├── calibration.py                # Evaluator vs human label agreement
│   ├── cost.py                       # Token → USD math
│   ├── storage.py                    # RunStore — persist runs + labels
│   ├── observability.py              # StructuredLogObserver, diagnostics
│   ├── security_limits.py            # Resource ceilings for untrusted input
│   ├── cli.py                        # CLI entry point
│   │
│   ├── evaluators/
│   │   ├── deterministic.py          # 6 free evaluators
│   │   ├── semantic.py               # SemanticSimilarityEvaluator
│   │   ├── rag.py                    # 7 RAG-specific evaluators
│   │   └── llm_judge.py              # 3 LLM-judge evaluators
│   │
│   ├── providers/
│   │   ├── embedding.py              # EmbeddingProvider / FakeEmbeddingProvider
│   │   ├── openai_embedding.py       # OpenAIEmbeddingProvider (optional)
│   │   ├── llm.py                    # JsonLLM / FakeJsonLLM
│   │   ├── openai_llm.py             # OpenAIJsonLLM (optional)
│   │   └── http.py                   # HttpClient / FakeHttpClient / HttpxHttpClient
│   │
│   └── adapters/
│       └── production_rag.py         # Wraps production-rag /ask as SystemUnderTest
│
├── datasets/
│   └── sample_eval_v1.json           # 20-example synthetic dataset
│
├── scripts/
│   ├── demo_end_to_end.py            # Full pipeline demonstration
│   ├── build_sample_dataset.py       # Generate sample datasets
│   └── pre_push_check.sh             # Local pre-push CI checks
│
├── dashboard/
│   └── index.html                    # Static run browser + comparison UI
│
├── tests/                            # 348 tests — 100% harness coverage
│   ├── integration/
│   │   └── test_failure_injection.py # 17 synthetic failure scenarios
│   └── live/                         # Real OpenAI calls (self-skip without key)
│
├── docs/
│   ├── day1.md … day25.md            # Day-by-day engineering log
│   ├── DEFINITION_OF_DONE.md         # Full checklist — PASS/FAIL with evidence
│   └── ENGINEERING_REPORT.md         # Final engineering report
│
├── .github/workflows/ci.yml          # Two-tier CI
├── requirements-dev.txt
├── requirements-openai.txt           # Optional: OpenAI evaluators
└── requirements-httpx.txt            # Optional: HTTP adapter
```

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/Saqib00712/eval-harness.git
cd eval-harness
```

### 2. Install dependencies
```bash
# Full suite — no API key needed
pip install -r requirements-dev.txt

# Optional: OpenAI evaluators (semantic + LLM-judge)
pip install -r requirements-openai.txt

# Optional: HTTP adapter for live system evaluation
pip install -r requirements-httpx.txt
```

### 3. Run the test suite
```bash
pytest -v
# 345 passed, 4 skipped  (without openai package)
# 348 passed, 4 skipped  (with openai package)
```

### 4. Try the CLI
```bash
# Validate a dataset file
python -m eval_harness.cli validate-dataset datasets/sample_eval_v1.json

# Run full demo (comparison + regression + quality gate)
python -m eval_harness.cli demo

# Compare two runs
python -m eval_harness.cli compare baseline_run.json candidate_run.json

# CI quality gate
python -m eval_harness.cli quality-gate candidate_run.json \
    --baseline baseline_run.json \
    --min-pass-rate 0.9
```

All CLI commands return CI-usable exit codes: `0` = pass, `1` = fail.

### 5. Open the dashboard
```bash
python -m eval_harness.cli demo      # writes two run files to demo_runs/
open dashboard/index.html            # drag in the files it wrote
```

---

## Evaluating Your Own System

Implement one interface:

```python
from eval_harness.system import SystemUnderTest
from eval_harness.system_output import SystemOutput

class MySystem(SystemUnderTest):
    def run(self, test_case) -> SystemOutput:
        answer = my_pipeline(test_case.input)
        return SystemOutput(answer=answer)
```

Then run:

```python
from eval_harness.runner import run_evaluation
from eval_harness.evaluators.deterministic import ExactMatchEvaluator

results = run_evaluation(
    dataset=my_dataset,
    system=MySystem(),
    evaluators=[ExactMatchEvaluator()]
)
print(results.pass_rate)
```

For a deployed system, use the HTTP adapter:

```python
from eval_harness.adapters.production_rag import ProductionRagAdapter

system = ProductionRagAdapter(base_url="http://localhost:8000", api_key="...")
```

---

## CI Integration

```yaml
# .github/workflows/ci.yml (minimal example)
- name: Run eval quality gate
  run: |
    python -m eval_harness.cli quality-gate candidate_run.json \
        --baseline baseline_run.json \
        --min-pass-rate 0.90
  # exit 1 blocks the merge
```

---

## Key Concepts Covered

- **Pluggable evaluator interface** — any scoring logic behind one `evaluate()` method
- **Failure isolation** — a system crash on one test doesn't kill the whole run
- **Failure taxonomy** — distinguishing infra failures from bad answers from evaluator bugs
- **Regression detection** — comparing runs by dataset+system pair, not by filename
- **Quality gate** — a single hard threshold that becomes a CI exit code
- **LLM-judge calibration** — measuring how well an LLM judge agrees with humans before trusting it in CI
- **Cost accounting** — tracking every API call per evaluator so you know what your eval pipeline costs
- **Security limits** — ReDoS caps on regex evaluators, path-traversal hardening in RunStore, dataset size ceilings
- **Adapter pattern** — wrapping any deployed system as a `SystemUnderTest` without importing it

---

## Related Certifications

Built applying skills from the IBM **RAG for Generative AI Applications Specialization**, **Advanced RAG with Vector Databases and Retrievers**, and **Building AI Agents and Agentic Workflows Specialization** on Coursera.

[![IBM Badge](https://img.shields.io/badge/IBM-RAG%20Specialization-blue?style=flat-square)](https://www.credly.com/users/muhammad-saqib.361f9b8c)
[![IBM Badge](https://img.shields.io/badge/IBM-AI%20Agents%20Specialization-blue?style=flat-square)](https://www.credly.com/users/muhammad-saqib.361f9b8c)

---

## Author

**Muhammad Saqib**
- GitHub: [@Saqib00712](https://github.com/Saqib00712)
- LinkedIn: [muhammad-saqib](https://www.linkedin.com/in/muhammad-saqib-68b9b3374/)
- Email: saqibkhosa649@gmail.com
- Credly: [15x IBM Certified](https://www.credly.com/users/muhammad-saqib.361f9b8c)`.
