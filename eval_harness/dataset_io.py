"""
Save/load a full Dataset manifest (metadata + test cases) as one JSON file.

Separate from loader.py's `load_test_cases`, which reads a BARE array of
test cases (no dataset_id/version/etc.) -- that stays the lighter-weight
entry point for "I just have a pile of golden questions". This module is
for the versioned, named, reproducible unit the rest of the harness will
reference by `dataset_id` + `version` (Day 11's EvaluationRun).

Note what's written back on save: `test_count` and `content_hash` ARE
included (Pydantic computed fields serialize by default) purely so a human
skimming the file sees them. On load they are IGNORED, not trusted -- a
freshly-built Dataset always recomputes both from `test_cases`. There is no
code path where a hand-edited (or tampered) hash field is believed over the
actual content.

Security (Day 17): the same file-size and record-count ceilings loader.py
enforces for a bare test-case array apply here too -- a manifest is loaded
from disk exactly as untrusted as a bare array is. No separate per-record
check is needed on top of those two: unlike loader.py's bare array (parsed
record-by-record before validation), every record here is validated as a
full TestCase by Pydantic as part of `Dataset.model_validate`, so a single
oversized or malformed record is already caught -- the ceilings here exist
to stop a whole FILE from being too large or containing too MANY records to
load safely, not to re-implement loader.py's record-by-record size check.
"""

import json
from pathlib import Path

from pydantic import ValidationError

from eval_harness.dataset import Dataset
from eval_harness.security_limits import DEFAULT_MAX_FILE_SIZE_BYTES, DEFAULT_MAX_RECORDS


class DatasetError(Exception):
    pass


def save_dataset(dataset: Dataset, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")


def load_dataset(
    path: str | Path,
    *,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
    max_records: int = DEFAULT_MAX_RECORDS,
) -> Dataset:
    p = Path(path)
    if not p.exists():
        raise DatasetError(f"Dataset not found: {p}")

    file_size = p.stat().st_size
    if file_size > max_file_size_bytes:
        raise DatasetError(f"{p} is {file_size} bytes, over the {max_file_size_bytes}-byte limit.")

    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{p} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise DatasetError(f"{p} must contain a JSON object (a dataset manifest, not a bare array)")

    raw_test_cases = raw.get("test_cases")
    if isinstance(raw_test_cases, list) and len(raw_test_cases) > max_records:
        raise DatasetError(f"{p} contains {len(raw_test_cases)} records, over the {max_records}-record limit.")

    try:
        return Dataset.model_validate(raw)
    except ValidationError as exc:
        raise DatasetError(f"{p} failed validation: {exc}") from exc
