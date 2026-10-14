"""
Day 19: adapters -- thin `SystemUnderTest` wrappers around real,
already-deployed systems. A separate package from `eval_harness/providers/`
on purpose: providers are things THIS harness calls out to on its own
behalf (an embedding model, a judge LLM); an adapter is how an entirely
separate SYSTEM BEING EVALUATED plugs into the harness. One import here
today (`production_rag`), but the shape is meant to hold more over time --
one module per system, each a `SystemUnderTest` and nothing else exported.
"""

from eval_harness.adapters.production_rag import ProductionRagAdapter, ProductionRagAdapterError

__all__ = ["ProductionRagAdapter", "ProductionRagAdapterError"]
