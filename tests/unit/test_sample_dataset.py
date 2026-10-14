"""Smoke-tests the checked-in sample dataset (Phase 31) actually loads and
is shaped the way Phase 32's later demonstration will expect."""

from eval_harness.dataset_io import load_dataset

PATH = "datasets/sample_eval_v1.json"


def test_sample_dataset_loads():
    ds = load_dataset(PATH)
    assert ds.dataset_id == "sample_eval"
    assert ds.test_count == 20


def test_sample_dataset_has_refusal_cases():
    """At least some questions must be designed to be UNANSWERABLE --
    otherwise the dataset never tests whether a system hallucinates."""
    ds = load_dataset(PATH)
    refusal_cases = [tc for tc in ds.test_cases if tc.expected_output is None]
    assert len(refusal_cases) >= 2


def test_sample_dataset_has_citation_bearing_cases():
    ds = load_dataset(PATH)
    cited = [tc for tc in ds.test_cases if tc.expected_citations]
    assert len(cited) >= 10


def test_sample_dataset_categories_are_varied():
    """Phase 31 asks for a MIX of challenge types, not 20 near-duplicates."""
    ds = load_dataset(PATH)
    categories = {tc.metadata.get("category") for tc in ds.test_cases}
    assert len(categories) >= 10


def test_sample_dataset_is_reproducible():
    a = load_dataset(PATH)
    b = load_dataset(PATH)
    assert a.content_hash == b.content_hash
