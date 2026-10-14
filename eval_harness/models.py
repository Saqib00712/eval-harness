"""
Phase 1: the generic evaluation data model.

A TestCase is deliberately NOT shaped around Project 1's RAG-with-citations
world (compare app/eval/models.py's GoldenQuestion, which hardcodes
`expected_pages: list[int]`). Here:

  * `expected_output`     -- optional: some systems (retrieval-only, some
                              agents) have no single "correct string".
  * `reference_context`   -- optional supporting text (RAG ground-truth
                              passages), a plain list of strings so any
                              system's context shape fits without a new field
                              per app.
  * `expected_citations`  -- a list of plain dicts, not a fixed schema. One
                              app's citation is {"page": 4}; another's might
                              be {"url": ..., "section": ...}. The harness
                              never assumes the shape; evaluators that care
                              about citations (Phase 8) read whatever keys
                              their own config tells them to look for.
  * `metadata` / `tags`   -- the escape hatch for anything app-specific
                              (category, difficulty, source) that doesn't
                              deserve a first-class field here.

Note what's absent: there is no `actual_output` field. Per the brief, actual
output is produced by the SYSTEM UNDER TEST at run time (Phase 3/4), never
stored in the golden dataset -- a golden dataset describes what SHOULD
happen, not what some past run DID produce.
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TestCase(BaseModel):
    __test__ = False  # not a pytest test class -- it just happens to share the name "TestCase"

    model_config = ConfigDict(frozen=True)

    test_id: str
    input: str
    expected_output: str | None = None
    reference_context: list[str] = Field(default_factory=list)
    expected_citations: list[dict] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    @field_validator("test_id", "input")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v
