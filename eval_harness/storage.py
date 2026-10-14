"""
Phase 25: storage -- persisting `EvaluationRun`s (Day 5) and `HumanLabel`s
(Day 14) to disk, the same one-file-per-object JSON pattern Day 2's
`dataset_io.py` already established for `Dataset`. Everything built since
Day 10 has assumed an `EvaluationRun` can outlive the process that produced
it -- a regression check (Day 10) needs a BASELINE run from a previous day;
a calibration report (Day 14) needs the exact run its labels were written
against. Without this module, both only ever worked within a single Python
session holding both runs in memory at once. This closes that gap.

`RunStore` is the part that goes beyond a bare save/load pair: it organizes
runs on disk by `(dataset_id, system_name)` and can answer "what's the most
recent run of this dataset+system" directly -- exactly the baseline lookup
Day 10's `compare_runs` needs, now answerable without a caller having to
track run files by hand. `save_run`/`load_run` and `save_labels`/
`load_labels` are the lower-level primitives `RunStore` is built on, and are
exposed directly for anyone who wants a single file somewhere specific
instead of the organized store.

Security (Day 17): `RunStore` builds filesystem paths directly from
`EvaluationRun.dataset_id` / `system_name` / `run_id` -- fields a caller
sets (often from whatever a `SystemUnderTest` or `Dataset` was named), not
fields this harness fully controls itself. Without a check, a
`dataset_id` like `"../../etc"` would let a saved run escape the store's
root directory entirely -- a classic path-traversal bug. `_safe_path_
component` rejects any component containing a path separator or a `..`
segment before it's ever joined into a path, for all three fields.
"""

import json
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from eval_harness.calibration import HumanLabel
from eval_harness.run_models import EvaluationRun

_LabelListAdapter = TypeAdapter(list[HumanLabel])


class StorageError(Exception):
    pass


def _safe_path_component(field_name: str, value: str) -> str:
    """Reject a value that could escape the intended directory when used
    as ONE path segment -- a path separator (forward or backward slash, so
    this catches both POSIX and Windows-style traversal attempts
    regardless of which OS is running) or a `..` segment. Raises
    StorageError, the same exception every other storage failure here
    raises, rather than silently sanitizing (stripping characters) the
    value into something that might collide with an unrelated run."""
    if not value or not value.strip():
        raise StorageError(f"{field_name} must not be blank.")
    if "/" in value or "\\" in value or value in (".", ".."):
        raise StorageError(f"{field_name} {value!r} is not safe to use as a storage path segment.")
    return value


def save_run(run: EvaluationRun, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(run.model_dump_json(indent=2), encoding="utf-8")


def load_run(path: str | Path) -> EvaluationRun:
    p = Path(path)
    if not p.exists():
        raise StorageError(f"Run file not found: {p}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StorageError(f"{p} is not valid JSON: {exc}") from exc
    try:
        return EvaluationRun.model_validate(raw)
    except ValidationError as exc:
        raise StorageError(f"{p} failed validation as an EvaluationRun: {exc}") from exc


def save_labels(labels: list[HumanLabel], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_LabelListAdapter.dump_json(labels, indent=2).decode("utf-8"), encoding="utf-8")


def load_labels(path: str | Path) -> list[HumanLabel]:
    p = Path(path)
    if not p.exists():
        raise StorageError(f"Labels file not found: {p}")
    # TypeAdapter.validate_json parses JSON internally -- unlike load_run's
    # manual json.loads() + model_validate() two-step, a malformed-JSON
    # input here surfaces as a pydantic ValidationError (json_invalid),
    # never a json.JSONDecodeError. (Day 18's test-suite audit caught this
    # the honest way: a test written to hit "not valid JSON" failed because
    # that branch was actually unreachable dead code.)
    try:
        raw = p.read_text(encoding="utf-8")
        return _LabelListAdapter.validate_json(raw)
    except ValidationError as exc:
        raise StorageError(f"{p} failed validation as a list of HumanLabels: {exc}") from exc


class RunStore:
    """Runs organized on disk as `<root>/<dataset_id>/<system_name>/<run_id>.json`
    -- one subtree per (dataset, system) pair being tracked over time, which
    is exactly the granularity Day 10's regression checks and Day 12's
    comparisons already operate at. No database, no index file: the
    directory listing IS the index, which means a run file can be deleted,
    copied, or inspected directly with zero other tooling, the same
    "a human can read the file" bias Day 2's dataset_io.py already made."""

    def __init__(self, root: str | Path):
        self._root = Path(root)

    def _dir_for(self, dataset_id: str, system_name: str) -> Path:
        return self._root / _safe_path_component("dataset_id", dataset_id) / _safe_path_component(
            "system_name", system_name
        )

    def save(self, run: EvaluationRun) -> Path:
        d = self._dir_for(run.dataset_id, run.system_name)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{_safe_path_component('run_id', run.run_id)}.json"
        save_run(run, path)
        return path

    def list_runs(self, dataset_id: str, system_name: str) -> list[EvaluationRun]:
        """All stored runs for this (dataset, system) pair, oldest first by
        `completed_at` -- not by filename, since a run_id (a uuid) carries
        no chronological meaning of its own."""
        d = self._dir_for(dataset_id, system_name)
        if not d.exists():
            return []
        runs = [load_run(p) for p in d.glob("*.json")]
        return sorted(runs, key=lambda r: r.completed_at)

    def latest(self, dataset_id: str, system_name: str) -> EvaluationRun | None:
        """The most recently completed run for this (dataset, system) pair
        -- the natural baseline for a Day 10 regression check against
        whatever runs next. `None`, not an exception, when nothing has
        been stored yet -- "no baseline exists yet" is the normal state for
        the very first run of a new system, not an error."""
        runs = self.list_runs(dataset_id, system_name)
        return runs[-1] if runs else None
