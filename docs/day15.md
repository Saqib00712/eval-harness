# Day 15 — Storage

Phase 25. Persisting `EvaluationRun`s (Day 5) and `HumanLabel`s (Day 14) to
disk, the same one-file-per-object JSON pattern Day 2's `dataset_io.py`
established for `Dataset`. Everything built since Day 10 has assumed a run
can outlive the process that produced it — a regression check needs a
baseline run from a previous day; a calibration report needs the exact run
its labels were written against. Without this module, both only ever
worked within a single Python session holding both runs in memory at once.

## `eval_harness/storage.py`

- **`save_run` / `load_run`** — one `EvaluationRun` per JSON file. Loud,
  specific `StorageError`s: missing file, invalid JSON, or a file that
  parses but doesn't validate as an `EvaluationRun` are three different
  messages, not one generic failure.
- **`save_labels` / `load_labels`** — a list of `HumanLabel`s as one JSON
  array, via Pydantic's `TypeAdapter` (the harness's first use of it —
  everywhere else a single `BaseModel` was enough).
- **`RunStore`** — the part that goes beyond a bare save/load pair. Organizes
  runs on disk as `<root>/<dataset_id>/<system_name>/<run_id>.json` and can
  answer **"what's the most recent run of this dataset+system"** directly —
  exactly the baseline lookup Day 10's `compare_runs` needs, now answerable
  without a caller tracking run files by hand. `latest()` returns `None`
  (not an exception) when nothing's stored yet — "no baseline exists yet"
  is the normal state for a system's very first run, not an error.
- No database, no index file: the directory listing *is* the index. A run
  file can be deleted, copied, or inspected directly with zero other
  tooling — the same "a human can read the file" bias `dataset_io.py`
  already established on Day 2.

## Tests

12 new tests (`test_storage.py`): round-tripping a run and a label list,
three distinct `StorageError` cases for a bad run file, `RunStore` saving
and finding the latest run, an empty store returning `None`, runs listed
oldest-first by `completed_at` (not filename), and dataset/system
separation (saving under one dataset+system never leaks into another's
listing). **241 passed, 4 skipped.**

## Next

Day 16 — observability: structured logging through a run, so a slow or
expensive evaluation can actually be diagnosed (which test case, which
evaluator, how long, how much) instead of only seeing the final aggregate.
