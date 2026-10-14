# Day 25 — a dashboard, scoped the same way the CLI was

## Scope decision

Asked before building, same as Phase 40's API-vs-CLI question: a static
dashboard reading already-saved run files (no backend), a full web app
with a new HTTP API, or a one-off view of just the Day 20 demo. Answer:
**static dashboard.**

Consequence: `dashboard/index.html` is a single, self-contained,
client-side-only page. There is no server, no `fetch()`, no network call
at all — it reads `EvaluationRun` JSON files the person picks (or
drags in) with the browser's own `FileReader`, entirely in-memory. This
keeps it consistent with the CLI-only scope decision from Phase 40: it
doesn't reintroduce the HTTP API that was explicitly descoped, it just
gives the *files that already exist* (anything `storage.save_run()`
wrote, including what `demo`/`quality-gate`/`compare` already produce) a
better way to be looked at than raw JSON or terminal text.

## What it does

- **One run loaded** → summary cards (pass rate, average score, system
  latency, cost), a per-evaluator pass-rate table, a failure-category
  breakdown (dependency / evaluation / threshold-miss, the same
  taxonomy `failure_taxonomy.py` defines), and a table of every failed
  `(test_id, evaluator)` row with its score, threshold, and reason.
- **Two runs loaded** (same `dataset_id`) → a comparison view: pass-rate
  and average-score deltas, which test cases newly passed, which
  regressed. Loading two runs of *different* datasets shows the same
  error `regression.py::compare_runs` raises, not a nonsense comparison.
- **A live quality gate panel**, on either view: type in `min_pass_rate`/
  `min_average_score` and get an instant PASS/FAIL verdict with the
  specific violation(s) listed — including a regression-check violation
  when two runs are loaded, the same "static thresholds alone would
  pass but the regression check fails it" case the Day 20 demo
  demonstrates.

## Correctness, not just appearance

The comparison and quality-gate logic in the page's JavaScript is a
direct re-implementation of `regression.py::compare_runs` and
`quality_gate.py::evaluate_quality_gate` — same per-`(test_id,
evaluator)` regression detection, same violation messages, same
dataset-mismatch error text. This was not left to "looks about right":
the page was smoke-tested headlessly (Playwright, Chromium) against the
*real* run files `python -m eval_harness.cli demo` produces, verifying:

- the single-run view surfaces the real pass rate / per-evaluator
  breakdown for the baseline run,
- loading the real candidate run on top switches to the comparison view
  and correctly shows `qa-019` as regressed and `qa-009`/`qa-013` as
  newly passing — the exact same IDs `docs/day20.md`'s CLI output shows,
- setting a quality-gate threshold produces a FAILED verdict citing the
  regression, matching the CLI's own `quality-gate --baseline` output,
- removing a loaded run correctly falls back to the single-run view,
- a malformed JSON file is reported as a clear error, not a blank page
  or a silent failure,
- zero JavaScript console or page errors across all of the above.

## Using it

```
open dashboard/index.html          # macOS
xdg-open dashboard/index.html      # Linux
```

No install, no build step, no dependency — just a browser. Drop in any
run file written by `storage.save_run()`, or run
`python -m eval_harness.cli demo` first and load the two files it writes
to `demo_runs/sample_eval/*/*.json`.

## What this is not

Not a replacement for the CLI's exit codes (CI still needs
`quality-gate`'s real process exit code, not a browser tab); not a
server, so it can't watch for new runs appearing or trigger an
evaluation itself; not a claim that Phase 40's CLI-only scope decision
changed — this reads files the CLI already produces, it doesn't add a
way to run an evaluation that didn't already exist.
