"""
Phase 3: the SystemUnderTest abstraction -- the one interface every kind of
AI system (RAG app, plain LLM app, agent, classifier, structured-output
system, or a mock for testing the harness itself) must satisfy to be
evaluated, without the evaluation engine (Day 5's runner) ever needing to
know which kind it's talking to.

Why a `Protocol`, not an abstract base class: this is the same choice
production-rag made for its Embedder/JsonLLM/Reranker interfaces (see
docs/day1.md's inspection) -- a Protocol is structural typing. A RAG
pipeline, an agent framework's own "agent" object, or a three-line adapter
function can all satisfy SystemUnderTest just by having a matching `name`,
`version`, and async `run` -- no shared base class to inherit from, no
import of this package required in the system's own code. That matters
specifically because Project 1 (production-rag) must stay fully
independent of this harness (Day 1's working rule): an ADAPTER (Day 19,
once it lands) can wrap `app.services.rag_pipeline` and satisfy this
Protocol without production-rag ever importing eval_harness.

`name`/`version` are not an afterthought: Day 11's EvaluationRun needs to
record which system (and which version of it -- a model swap, a prompt
change) produced a given run's results, so every run is attributable
without guessing from context.
"""

from typing import Protocol, runtime_checkable

from eval_harness.models import TestCase
from eval_harness.system_output import SystemOutput


@runtime_checkable
class SystemUnderTest(Protocol):
    @property
    def name(self) -> str:
        """A short, stable identifier for this system (e.g. "production-rag",
        "gpt-4o-mini-classifier"). Recorded on every EvaluationRun."""
        ...

    @property
    def version(self) -> str:
        """Whatever distinguishes one evaluated version of this system from
        another -- a semantic version, a git commit hash, a prompt version
        string, a model name. The harness never interprets this value, only
        records and displays it (Phase 19's prompt comparison is built on
        two runs whose `version` differs)."""
        ...

    async def run(self, test_case: TestCase) -> SystemOutput:
        """Run this system against one test case's `input` and return a
        SystemOutput. Must never raise for an ordinary failure the system
        itself can represent (a refusal, an empty answer) -- those belong in
        a populated SystemOutput (e.g. metadata={"refused": True}), not an
        exception. An exception here means the system crashed outright; the
        runner (Day 5) is responsible for deciding what that means for the
        test case's result, not this method."""
        ...
