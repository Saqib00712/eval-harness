"""
Phase 2: dataset versioning.

Design decision: `content_hash` and `test_count` are COMPUTED properties
(Pydantic `computed_field`s), never fields you set by hand and never the
source of truth you edit directly. This means a dataset's hash can never go
STALE -- there is no "I changed the test cases but forgot to update the
hash" bug class, because the hash is always derived fresh from whatever
`test_cases` currently holds. That is what makes "the same dataset must be
reproducible" true by construction rather than by discipline: hash two
Datasets built from the same content (regardless of list order -- see
`compute_content_hash`'s canonicalization) and you always get the same
digest; change one test case's text and the digest always changes.

`version` stays a separate, human-assigned field (e.g. "1.0.0", "2026-01-15")
-- intentionally NOT auto-derived from the hash. A content hash answers "is
this exactly the same data"; a version answers "what did a human call this
at the time" (useful in reports: "candidate was evaluated against golden
v1.0.0"). An `EvaluationRun` (Day 11) records BOTH, because they answer
different questions: version is what a person reads, content_hash is what a
program trusts.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from eval_harness.models import TestCase
from eval_harness.hashing import compute_content_hash


class Dataset(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset_id: str
    name: str
    version: str
    description: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    test_cases: list[TestCase]

    @field_validator("dataset_id", "name", "version")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v

    @model_validator(mode="after")
    def _unique_test_ids(self) -> "Dataset":
        ids = [tc.test_id for tc in self.test_cases]
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"Duplicate test_id(s) in dataset {self.dataset_id!r}: {dupes}")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def test_count(self) -> int:
        return len(self.test_cases)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def content_hash(self) -> str:
        """Reproducible fingerprint of `test_cases` ONLY -- not of
        dataset_id/name/version/description/created_at. Renaming a dataset
        or re-tagging its version does not change its content_hash;
        editing even one test case's text does. This is deliberate: the
        hash answers "is the DATA the same", not "is the record the same"."""
        return compute_content_hash(self.test_cases)
