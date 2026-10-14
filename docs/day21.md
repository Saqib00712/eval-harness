# Day 21 — CI/CD for eval-harness itself

Phase 33. Nineteen days of unit tests and one day of integration tests
(Day 20) only actually gate anything once they run automatically on every
push — otherwise they're a suite someone has to remember to run by hand.
This day wires this package into GitHub Actions, following the same
two-tier split Project 1's own Day 11 landed on, for the same reason: a
free check should run on everything, and a paid one should run sparingly
and never block the free one.

## `.github/workflows/ci.yml`

**`test` job — always runs, on every push and pull request, free.**
Matrix across Python 3.11/3.12. Three steps:

1. `pytest -q` — the full unit + integration suite (306 tests as of
   today), with `requirements-openai.txt` deliberately **not** installed
   in this job. The base suite has to prove it works with zero AI SDKs
   present, not just with them conveniently available — the exact
   discipline Day 6 and Day 8 built `FakeEmbeddingProvider`/`FakeJsonLLM`
   for in the first place. `requirements-httpx.txt` IS installed here,
   because `HttpxHttpClient` (Day 19) is tested via `httpx.MockTransport`
   — no network, no cost — unlike the paid OpenAI calls.
2. A coverage threshold (`--fail-under=98`, `eval_harness/*`, omitting the
   two OpenAI live-call modules) — Day 18's audit made this a measurable
   number in the first place; this is what turns it from "someone should
   check this sometime" into something that actually blocks a regression
   in coverage itself.
3. `python scripts/demo_end_to_end.py` — the Day 20 demo, run as a smoke
   test on every push. Entirely `FakeHttpClient`-backed, zero network, zero
   cost. This specifically catches the risk Day 20's own docs flagged: the
   demo script isn't imported by anything else, so nothing else would
   notice if its scripted story silently broke.

**`live-eval` job — only on a push to `main`, needs `test` to pass first.**
Installs `requirements-openai.txt` and runs `tests/live/` with
`OPENAI_API_KEY` from a repo secret. If that secret isn't configured,
every test in `tests/live/` self-skips (the `pytest.mark.skipif` pattern
those tests already use — see Day 6/Day 8's docs), so this job succeeds
either way. It's additive: proof the *real* providers work, not required
for the free gate to pass.

## `scripts/pre_push_check.sh`

Optional, not installed automatically — the same judgment call Project
1's own Day 11 made for its pre-push hook: a hook that silently blocks
`git push` without being asked for is a worse surprise than a red CI run
a few seconds later. Runs the exact same three steps as the `test` job
locally. Install it if you want it:

```
cp scripts/pre_push_check.sh .git/hooks/pre-push && chmod +x .git/hooks/pre-push
```

Or just run it by hand: `bash scripts/pre_push_check.sh`.

## What this doesn't do (yet)

No packaging (`pyproject.toml`, a real `pip install eval-harness`) —
the test suite still runs via `pytest.ini`'s `pythonpath = .`, same as
every prior day, which CI needs nothing extra for. Packaging is a real
gap for anyone who'd want to `pip install` this into another project
rather than copy the folder, but it's a separate concern from CI and
didn't need to block this day landing.

## Tests

No new unit tests today — there's no new `eval_harness` code, only CI
configuration and a shell script. Verified instead by actually running
all three `test`-job steps locally (`pytest -q`, the coverage gate, the
demo script) and confirming each one's real exit code, the same
verification a push to GitHub would get. **306 passed, 4 skipped**
(unchanged from Day 20).

## Next

Day 22 — not yet previewed. Packaging (`pyproject.toml`) is the most
obvious remaining gap; also open: a `CHANGELOG.md`, or extending the
Project 1 integration story (Day 19) with a second adapter for a
different system, to prove the `SystemUnderTest` Protocol generalizes
past just production-rag.
