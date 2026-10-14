# Day 20 — Worked end-to-end demonstration

Phase 32. Every day since Day 1 built one piece, proven correct by its own
unit tests in isolation. Today wires them together the way a real CI job
actually would, against Project 1 (`production-rag`) through Day 19's
`ProductionRagAdapter` — the first time this harness tells its whole
story in one run instead of nineteen separate ones.

## The script

`scripts/demo_end_to_end.py` — run it directly (`python
scripts/demo_end_to_end.py`) or import `build_demo_report()` for the
underlying data. No live production-rag deployment is required: both
"systems" are `ProductionRagAdapter`s backed by `FakeHttpClient` with
scripted responses standing in for two real deployments. Point
`ProductionRagAdapter` at `HttpxHttpClient()` and a real `base_url`/
`api_key` instead (`docs/day19.md`) to run this *exact same script*
against a real deployment — nothing else would need to change, which is
the point of Day 19's adapter existing in the first place.

## The story it tells

Uses the shared golden dataset (`datasets/sample_eval_v1.json`, Day 31),
narrowed to the 17 of 20 cases with a reference answer. Two scripted
`production-rag` versions:

- **baseline** (`v1.0-gpt-4o-mini`) — gets 2 questions wrong (an ambiguous
  pricing question, a multi-hop refund-window question).
- **candidate** (`v1.1-gpt-4o-mini-ft`) — **fixes** both of those, but
  **introduces one new wrong answer** on a previously-correct question
  (a hallucinated uptime percentage, 99.5% instead of 99.9% — note it
  still cites the *right* page, which is exactly why a citation-only check
  wouldn't catch this).

Run through the full pipeline:

1. **`compare_systems` (Day 12)** — candidate wins overall (pass rate
   0.8824 → 0.9412) and wins both evaluators individually.
2. **`compare_runs` (Day 10)** — `newly_passing_test_ids: [qa-009,
   qa-013]`, `regressed_test_ids: [qa-019]`. The strict default
   `RegressionPolicy` (zero tolerance) fails the check: **one regression
   is one regression**, regardless of the two fixes alongside it.
3. **`evaluate_quality_gate` (Day 11)**, given the regression report —
   **fails**, specifically on the regression violation, even though every
   static threshold on the candidate's own run (`min_pass_rate=0.85`,
   `min_average_score=0.85`) would have passed on its own.
4. **`compute_efficiency` / `rank_by_cost_per_passing_test` (Day 13)** —
   both versions ranked by cost per passing test.
5. **`RunStore` (Day 15)** — both runs saved to `demo_runs/`;
   `store.latest()` demonstrates the exact lookup a real CI job's *next*
   run would use to find its baseline, rather than needing one passed in
   by hand.
6. **`StructuredLogObserver` (Day 16)** — one more run, observed, proving
   the observer is a pure side channel (same pass rate, same average
   score as the un-observed candidate run); `slowest_results`/
   `most_expensive_results` surface the specific rows driving time/cost.

**This is the actual point of Days 9–11 existing**: a system that got
measurably better by every aggregate number can still fail a quality
gate, correctly, because a gate wired to a regression check catches the
one thing an aggregate number structurally can't show — that a specific,
previously-working case broke. A pass-rate-only gate would have shipped
this candidate.

## Why `tests/integration/` only gets populated today

`tests/integration/README.md` has said "populated starting the day the
runner exists" since Day 5 — and stayed empty through Day 19, because
every day until now had a clean single-module story that belonged in
`tests/unit/`. Today's whole point is multiple modules working together
(loader → adapter → runner → comparison → regression → quality_gate →
efficiency → storage → observability), which is exactly what that
directory was reserved for. `tests/integration/test_demo_end_to_end.py`
asserts on the demo's *specific* scripted numbers (which test cases
newly pass, which one regresses, that the gate fails despite the higher
pass rate) rather than just "it runs without raising" — a demo whose own
numbers silently drifted would be a worse regression than any of the
ones it exists to demonstrate catching.

## Tests

5 new integration tests. **306 passed, 4 skipped.**

## Next

Day 21 — not yet previewed; the remaining later phases of the original
spec (further demonstration/documentation polish, CI config for this
package itself, packaging) are open to revisit once this day's lesson has
landed.
