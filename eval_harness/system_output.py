"""
Phase 4: SystemOutput -- the standardized result a SystemUnderTest (Phase 3)
hands back to the harness, whatever kind of system it wrapped.

Every field is optional except nothing (not even `output`) -- a
retrieval-only system has no single answer string, a classifier has
`structured_data` but no `context`, a pure-text LLM app has neither
`citations` nor `structured_data`. The harness and its evaluators (Phase
5+) must read whatever fields are actually populated and simply skip what
isn't -- never assume a field exists.
"""

from pydantic import BaseModel, ConfigDict, Field, computed_field


class TokenUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    output_tokens: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class SystemOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    # The free-text answer, if this system produces one (a RAG app, a plain
    # LLM app). None for a system whose result is purely `structured_data`
    # (a classifier) or purely `context` (a retrieval-only system).
    output: str | None = None

    # Retrieved/supporting passages, if this system does retrieval (RAG,
    # some agents). Plain strings -- the harness doesn't assume a page
    # number or source id is attached; that lives in `citations` instead.
    context: list[str] = Field(default_factory=list)

    # Plain dicts, matching TestCase.expected_citations's shape choice
    # (Day 1): one app's citation is {"page": 4}, another's is
    # {"url": ..., "section": ...}. Never a fixed schema here either.
    citations: list[dict] = Field(default_factory=list)

    # A classifier's label, an agent's tool-call trace, any system whose
    # "answer" isn't prose. Kept as a plain dict rather than `Any` so it's
    # always JSON-serializable for storage (Phase 25) without extra work.
    structured_data: dict | None = None

    # The escape hatch for anything system-specific that doesn't deserve a
    # first-class field (e.g. a RAG system's refusal_reason, an agent's
    # number of tool calls).
    metadata: dict = Field(default_factory=dict)

    latency_ms: float | None = None
    token_usage: TokenUsage | None = None
    estimated_cost: float | None = None
    model: str | None = None
