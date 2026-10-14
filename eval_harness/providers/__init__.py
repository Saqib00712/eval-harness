"""
Provider abstractions: Protocols plus their implementations, split by
family so a dependency-free import stays dependency-free.

  providers/embedding.py        -- EmbeddingProvider Protocol + FakeEmbeddingProvider (no extra deps)
  providers/openai_embedding.py -- OpenAIEmbeddingProvider (needs requirements-openai.txt)
"""
