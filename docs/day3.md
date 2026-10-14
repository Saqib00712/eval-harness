# Day 3 — SystemUnderTest abstraction + SystemOutput + MockSystem

## Phase 4: SystemOutput

`eval_harness/system_output.py` — the standardized result any evaluated system hands back. Every field is optional except nothing — a classifier populates `structured_data` and nothing else; a retrieval-only system populates `context` and nothing else; a RAG app populates most of them. Evaluators (Phase 5+) must read whatever's there and skip what isn't, never assume a field exists.

`TokenUsage` is a small nested model (`input_tokens`, `output_tokens`, computed `total_tokens`) — not folded directly into `SystemOutput`'s flat fields, so a system with no token concept (a classifier) can simply leave it `None` instead of three stray zero fields.

## Phase 3: SystemUnderTest

`eval_harness/system.py` — a `Protocol` (`@runtime_checkable`), not an abstract base class: structural typing, the same choice Project 1 made for its own `Embedder`/`JsonLLM` interfaces. Anything with a `name`, `version`, and an async `run(test_case) -> SystemOutput` satisfies it — no inheritance, no import of this package required in the system's own code. That's what keeps Day 19's future Project-1 adapter from forcing production-rag to depend on eval_harness.

`name`/`version` aren't decoration — Day 11's `EvaluationRun` needs to record which system, and which version of it, produced a given run's results.

`run()` must never raise for an *ordinary* system failure (a refusal, an empty answer) — that belongs in a populated `SystemOutput` (e.g. `metadata={"refused": True}`). An exception means the system crashed outright; deciding what that means for a test case's result is the runner's job (Day 5), not this method's.

## MockSystem

`eval_harness/mock_system.py` — a real, first-class `SystemUnderTest` backed by canned `SystemOutput`s keyed by `test_id`, with an optional `default` for anything not explicitly canned. This is what lets the Day 5 runner (and this project's own test suite from here on) be tested deterministically, for free, with zero network calls — matching Phase 3's explicit requirement that "Mock application" be one of the supported system kinds, not an afterthought.

## Tests

12 new tests, all offline, all async-capable now (`pytest-asyncio` added to `requirements-dev.txt`, `asyncio_mode = auto` added to `pytest.ini`). **Total: 59 passed.**

## Next

Day 4 — Phase 5 + 6: the `Evaluator` abstraction and the deterministic evaluators (exact match, contains, regex, JSON schema, length, required fields) — the first things that actually score a `SystemOutput` against a `TestCase`.
