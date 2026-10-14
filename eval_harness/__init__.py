"""
Automated Eval Harness -- Project 2.

A generic, reusable evaluation platform for AI systems (RAG, plain LLM apps,
agents, classifiers, structured-output systems). Deliberately independent of
Project 1 (production-rag's app/ package): this package must never import
from `app.*`. Project 1 is consumed the other way around, as one example
SystemUnderTest plugged into this harness (see eval_harness/integrations/
once that lands).

Where Project 1's own ideas were worth reusing, they were reused as a
PATTERN (Protocol-based provider abstractions, Pydantic-validated LLM JSON
output, a shared AsyncOpenAI client, contextvar-based tracing, cost-ceiling
gates) rather than as a direct import -- see docs/day1.md for the full
inspection this package started from.
"""
