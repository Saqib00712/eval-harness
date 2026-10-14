import json

import pytest

from eval_harness.dataset import Dataset
from eval_harness.dataset_io import DatasetError, load_dataset, save_dataset
from eval_harness.models import TestCase

CASES = [
    TestCase(test_id="qa-001", input="What is the refund policy?", expected_output="30 days"),
    TestCase(test_id="qa-002", input="What is the capital of France?", expected_output="Paris"),
]


def test_round_trip_preserves_content_and_hash(tmp_path):
    original = Dataset(dataset_id="demo", name="Demo set", version="1.0.0", test_cases=CASES)
    p = tmp_path / "demo.json"
    save_dataset(original, p)

    loaded = load_dataset(p)
    assert loaded.dataset_id == original.dataset_id
    assert loaded.version == original.version
    assert loaded.test_count == 2
    assert loaded.content_hash == original.content_hash  # reproducibility across a save/load round trip


def test_save_writes_human_readable_json(tmp_path):
    ds = Dataset(dataset_id="demo", name="Demo set", version="1.0.0", test_cases=CASES)
    p = tmp_path / "demo.json"
    save_dataset(ds, p)
    raw = json.loads(p.read_text(encoding="utf-8"))
    assert raw["dataset_id"] == "demo"
    assert raw["content_hash"] == ds.content_hash  # present for humans, never trusted on load
    assert len(raw["test_cases"]) == 2


def test_hand_edited_test_cases_change_hash_on_reload(tmp_path):
    """A stale `content_hash` left in the file after a hand edit is never
    trusted -- load_dataset always recomputes it from test_cases."""
    ds = Dataset(dataset_id="demo", name="Demo set", version="1.0.0", test_cases=CASES)
    p = tmp_path / "demo.json"
    save_dataset(ds, p)

    raw = json.loads(p.read_text(encoding="utf-8"))
    raw["test_cases"][0]["expected_output"] = "45 days"  # edit content, leave the stored hash untouched
    p.write_text(json.dumps(raw), encoding="utf-8")

    reloaded = load_dataset(p)
    assert reloaded.content_hash != ds.content_hash


def test_missing_file(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        load_dataset(tmp_path / "nope.json")


def test_malformed_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid", encoding="utf-8")
    with pytest.raises(DatasetError, match="not valid JSON"):
        load_dataset(p)


def test_bare_array_rejected():
    """load_dataset expects a manifest OBJECT, not a bare test-case array
    (that's loader.load_test_cases's job)."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "bare.json"
        p.write_text(json.dumps([{"test_id": "qa-001", "input": "x"}]), encoding="utf-8")
        with pytest.raises(DatasetError, match="JSON object"):
            load_dataset(p)


def test_schema_validation_error_wrapped(tmp_path):
    p = tmp_path / "invalid.json"
    p.write_text(json.dumps({"dataset_id": "demo"}), encoding="utf-8")  # missing name/version/test_cases
    with pytest.raises(DatasetError, match="failed validation"):
        load_dataset(p)


def test_duplicate_ids_rejected_on_load(tmp_path):
    p = tmp_path / "dupes.json"
    p.write_text(json.dumps({
        "dataset_id": "demo", "name": "Demo", "version": "1.0.0",
        "test_cases": [
            {"test_id": "qa-001", "input": "a"},
            {"test_id": "qa-001", "input": "b"},
        ],
    }), encoding="utf-8")
    with pytest.raises(DatasetError, match="Duplicate"):
        load_dataset(p)
