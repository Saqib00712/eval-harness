# Live tests

Tests that make real OpenAI calls (real embeddings, real LLM judge). NEVER
run as part of the normal `pytest` suite or CI's free `test` job -- per
Phase 30, normal tests must not require OPENAI_API_KEY. These are opt-in
only, the same pattern as this repo's own `scripts/ci_eval.py` /
`live-eval` CI job. Populated once the harness has real providers to call.
