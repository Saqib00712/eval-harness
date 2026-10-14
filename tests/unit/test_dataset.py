import pytest
from pydantic import ValidationError

from eval_harness.dataset import Dataset
from eval_harness.models import TestCase

CASES = [
    TestCase(test_id="qa-001", input="What is the refund policy?", expected_output="30 days"),
    TestCase(test_id="qa-002", input="What is the capital of France?", expected_output="Paris"),
]


def _dataset(test_cases=None, **overrides) -> Dataset:
    kwargs = dict(dataset_id="demo", name="Demo set", version="1.0.0", test_cases=test_cases or CASES)
    kwargs.update(overrides)
    return Dataset(**kwargs)


def test_test_count_matches_number_of_cases():
    assert _dataset().test_count == 2


def test_same_content_same_hash_regardless_of_list_order():
    a = _dataset(test_cases=[CASES[0], CASES[1]])
    b = _dataset(test_cases=[CASES[1], CASES[0]])
    assert a.content_hash == b.content_hash


def test_same_content_same_hash_across_independent_builds():
    """Reproducibility: two Datasets built separately from identical
    content must hash identically -- this is the whole point of Phase 2."""
    a = _dataset()
    b = _dataset()
    assert a.content_hash == b.content_hash


def test_different_content_different_hash():
    changed = TestCase(test_id="qa-002", input="What is the capital of France?", expected_output="Lyon")
    a = _dataset()
    b = _dataset(test_cases=[CASES[0], changed])
    assert a.content_hash != b.content_hash


def test_hash_ignores_dataset_metadata_not_content():
    """Renaming/re-versioning a dataset must NOT change its content_hash --
    the hash fingerprints test_cases only."""
    a = _dataset(name="Demo set", version="1.0.0")
    b = _dataset(name="Totally Different Name", version="2.0.0")
    assert a.content_hash == b.content_hash


def test_duplicate_test_ids_rejected():
    with pytest.raises(ValidationError, match="Duplicate"):
        _dataset(test_cases=[CASES[0], CASES[0]])


@pytest.mark.parametrize("field", ["dataset_id", "name", "version"])
def test_blank_required_field_rejected(field):
    kwargs = dict(dataset_id="demo", name="Demo set", version="1.0.0", test_cases=CASES)
    kwargs[field] = "   "
    with pytest.raises(ValidationError):
        Dataset(**kwargs)


def test_dataset_is_frozen():
    ds = _dataset()
    with pytest.raises(ValidationError):
        ds.version = "2.0.0"
