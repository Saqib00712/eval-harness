"""
Pure content-hashing for reproducibility. No I/O, no Pydantic model
dependency beyond TestCase itself -- kept separate from dataset.py so it's
trivially unit-testable and reusable anywhere else a content fingerprint is
needed (e.g. Day 15's regression analysis: "did the candidate run use the
exact same golden set as the baseline run?").

Canonicalization matters more than the specific algorithm: two Python lists
holding the SAME test cases in a DIFFERENT order must hash identically (a
dataset's meaning doesn't depend on row order), so rows are sorted by
test_id before serializing, and `json.dumps(..., sort_keys=True)` removes
any dependency on dict key order within a single test case.
"""

import hashlib
import json

from eval_harness.models import TestCase


def compute_content_hash(test_cases: list[TestCase]) -> str:
    rows = sorted((tc.model_dump(mode="json") for tc in test_cases), key=lambda r: r["test_id"])
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
