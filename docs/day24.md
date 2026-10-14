# Day 24 — Phase 40 (part 1): a CLI, scoped explicitly to what's real

## Scope decision (asked, not assumed)

Phase 40's checklist assumes an "API and CLI" section exists to verify.
This project has been a pure Python library + scripts for 21 days — no
HTTP service, no Docker image. Before building anything, this was flagged
back with three options: CLI only, CLI + a minimal HTTP API, or neither
(mark the whole area MISSING). The answer: **CLI only.**

Consequence for the rest of Phase 40/41: there is no HTTP API and no
Docker image in this project. The Definition-of-Done checklist (Day 25)
marks those specific items `MISSING (not built)` with this decision as
the evidence — not glossed over, not faked with a stub.

## What was built

`eval_harness/cli.py` — four subcommands, each a thin wrapper around a
function that already existed and was already tested. Nothing new was
implemented underneath the CLI; it is an entry point, not new logic.

| Command | Wraps | Exit code |
|---|---|---|
| `validate-dataset <path>` | `dataset_io.load_dataset` | 0 valid / 1 invalid |
| `quality-gate <run.json> [--baseline] [--min-pass-rate] [--min-average-score]` | `regression.compare_runs` + `quality_gate.evaluate_quality_gate` + `exit_code_for` | 0 pass / 1 fail |
| `compare <baseline.json> <candidate.json>` | `regression.compare_runs` | 0 pass / 1 regressed |
| `demo` | `scripts/demo_end_to_end.py` | always 0 (demo's own illustrative CI exit code is printed text, not the process's real exit code — documented in the module) |

Deliberately **not** built: a generic `eval-harness run <system>` command.
Running an arbitrary `SystemUnderTest` from argv would require a
config language for instantiating one (which HTTP adapter, which
evaluators, which embedding/judge provider, API keys, …) that doesn't
exist anywhere in this project and would be new design work, not a thin
CLI wrapper. `demo` already exercises the full pipeline end to end
(dataset → two systems → comparison → regression → quality gate →
storage → observability) using the adapters and evaluators that do
exist; `quality-gate` and `compare` operate on already-persisted
`EvaluationRun` JSON (`storage.load_run`), which is something this CLI
can do honestly without inventing a config format.

## Error handling

Both `compare` and `quality-gate --baseline` call `regression.compare_runs`,
which raises `ValueError` when the two runs are against different
datasets (comparing apples to oranges). The CLI catches this alongside
`StorageError` and reports it as `ERROR: ...` on stderr with exit code 1,
rather than letting a raw traceback reach the terminal — this was caught
by `test_compare_mismatched_datasets_reports_error` and fixed before this
day's work was considered done (a Phase 39 pattern repeated here: write
the test, let it catch the real gap, fix it, don't just make the test
match the current behavior).

## Manually verified end to end (then pinned as tests)

```
$ python3 -m eval_harness.cli validate-dataset datasets/sample_eval_v1.json
VALID: 'sample_eval_v1' (dataset_id=..., version=...)
  20 test cases, content_hash=...
exit 0

$ python3 -m eval_harness.cli quality-gate <candidate>.json --min-pass-rate 0.5
PASSED -- pass_rate=0.9412 average_score=0.9902
exit 0

$ python3 -m eval_harness.cli quality-gate <candidate>.json --baseline <baseline>.json --min-pass-rate 0.5
FAILED -- ...
  - regression check failed: 1 test case(s) regressed (allowed: 0): qa-019
exit 1

$ python3 -m eval_harness.cli compare <baseline>.json <candidate>.json
... Regression check: FAILED
exit 1

$ python3 -m eval_harness.cli demo
[full Day 20 demonstration report]
exit 0
```

## Tests

`tests/unit/test_cli.py` — 12 tests: both exit codes for
`validate-dataset` (valid file, missing file, malformed JSON);
`quality-gate` passing, failing a static threshold, failing via a
regressing baseline even when static thresholds alone would pass,
and a missing run file; `compare` for identical runs, a real
regression, and mismatched dataset IDs; parser-shape checks
(`--help`-level: a subcommand is required; `demo` is registered).
All synthetic — `EvaluationRun` objects built directly and round-tripped
through `storage.save_run`/`load_run`, no real adapter or LLM call
anywhere in this file.

**348 passed, 4 skipped** (up from 336 — 12 new CLI tests).

## Next

Phase 40 (part 2) — the full Definition-of-Done checklist, section by
section, with real evidence per item.
