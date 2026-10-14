"""
Load a dataset of TestCase records from JSON (a single array) or JSONL (one
record per line). Mirrors app/eval/loader.py's error-handling shape
(not-found / malformed / duplicate-id / empty), generalized to TestCase and
to both file formats, since the brief requires both.

Security (Day 17 -- the Phase 28 ceilings this module promised back on Day
1): this loader never executes dataset content, and now enforces three
resource ceilings before trusting a file's size (DEFAULT_MAX_FILE_SIZE_BYTES,
checked before the file is even read), its record count
(DEFAULT_MAX_RECORDS, checked right after parsing), and any single record's
serialized size (DEFAULT_MAX_RECORD_SIZE_BYTES). All three raise the same
DatasetError every other failure here does -- a caller still only ever needs
to catch one exception type, whether a dataset is malformed, duplicated, or
simply too large to load safely. Defaults live in security_limits.py and are
overridable per call.
"""

import json
from pathlib import Path

from pydantic import ValidationError

from eval_harness.models import TestCase
from eval_harness.security_limits import (
    DEFAULT_MAX_FILE_SIZE_BYTES,
    DEFAULT_MAX_RECORD_SIZE_BYTES,
    DEFAULT_MAX_RECORDS,
)


class DatasetError(Exception):
    pass


def load_test_cases(
    path: str | Path,
    *,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
    max_records: int = DEFAULT_MAX_RECORDS,
    max_record_size_bytes: int = DEFAULT_MAX_RECORD_SIZE_BYTES,
) -> list[TestCase]:
    p = Path(path)
    if not p.exists():
        raise DatasetError(f"Dataset not found: {p}")

    file_size = p.stat().st_size
    if file_size > max_file_size_bytes:
        raise DatasetError(f"{p} is {file_size} bytes, over the {max_file_size_bytes}-byte limit.")

    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        records = _read_jsonl(p)
    elif suffix == ".json":
        records = _read_json(p)
    else:
        raise DatasetError(f"Unsupported dataset file type: {p.suffix!r} (use .json or .jsonl)")

    if len(records) > max_records:
        raise DatasetError(f"{p} contains {len(records)} records, over the {max_records}-record limit.")
    for i, record in enumerate(records):
        size = len(json.dumps(record))
        if size > max_record_size_bytes:
            raise DatasetError(
                f"{p}: record {i} is {size} bytes, over the {max_record_size_bytes}-byte per-record limit."
            )

    try:
        cases = [TestCase.model_validate(r) for r in records]
    except ValidationError as exc:
        raise DatasetError(f"{p} failed validation: {exc}") from exc

    ids = [c.test_id for c in cases]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise DatasetError(f"Duplicate test_id(s) in {p}: {dupes}")
    if not cases:
        raise DatasetError(f"{p} contains no test cases")
    return cases


def _read_json(p: Path) -> list[dict]:
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{p} is not valid JSON: {exc}") from exc
    if not isinstance(raw, list):
        raise DatasetError(f"{p} must contain a JSON array of test cases")
    return raw


def _read_jsonl(p: Path) -> list[dict]:
    records: list[dict] = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{p}:{lineno} is not valid JSON: {exc}") from exc
    return records
