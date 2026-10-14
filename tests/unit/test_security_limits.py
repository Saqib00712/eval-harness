import json

import pytest

from eval_harness.dataset_io import DatasetError as DatasetIOError
from eval_harness.dataset_io import load_dataset, save_dataset
from eval_harness.dataset import Dataset
from eval_harness.loader import DatasetError, load_test_cases
from eval_harness.models import TestCase


# ---- loader.py ceilings ----

def test_file_size_ceiling_rejects_oversized_file(tmp_path):
    p = tmp_path / "big.json"
    p.write_text(json.dumps([{"test_id": "t1", "input": "x"}]), encoding="utf-8")
    with pytest.raises(DatasetError, match="byte limit"):
        load_test_cases(p, max_file_size_bytes=1)  # the file is well over 1 byte


def test_record_count_ceiling_rejects_too_many_records(tmp_path):
    p = tmp_path / "many.json"
    records = [{"test_id": f"t{i}", "input": "x"} for i in range(10)]
    p.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(DatasetError, match="record limit"):
        load_test_cases(p, max_records=5)


def test_record_size_ceiling_rejects_oversized_record(tmp_path):
    p = tmp_path / "fat.json"
    records = [{"test_id": "t1", "input": "x" * 1000}]
    p.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(DatasetError, match="per-record limit"):
        load_test_cases(p, max_record_size_bytes=100)


def test_defaults_accept_a_normal_small_dataset(tmp_path):
    p = tmp_path / "ok.json"
    records = [{"test_id": "t1", "input": "x"}, {"test_id": "t2", "input": "y"}]
    p.write_text(json.dumps(records), encoding="utf-8")
    cases = load_test_cases(p)  # defaults -- should just work
    assert len(cases) == 2


# ---- dataset_io.py ceilings ----

def test_dataset_io_file_size_ceiling(tmp_path):
    dataset = Dataset(dataset_id="d1", name="D", version="1.0.0", test_cases=[TestCase(test_id="t1", input="x")])
    p = tmp_path / "d.json"
    save_dataset(dataset, p)
    with pytest.raises(DatasetIOError, match="byte limit"):
        load_dataset(p, max_file_size_bytes=1)


def test_dataset_io_record_count_ceiling(tmp_path):
    cases = [TestCase(test_id=f"t{i}", input="x") for i in range(10)]
    dataset = Dataset(dataset_id="d1", name="D", version="1.0.0", test_cases=cases)
    p = tmp_path / "d.json"
    save_dataset(dataset, p)
    with pytest.raises(DatasetIOError, match="record limit"):
        load_dataset(p, max_records=5)


def test_dataset_io_defaults_accept_normal_dataset(tmp_path):
    dataset = Dataset(dataset_id="d1", name="D", version="1.0.0", test_cases=[TestCase(test_id="t1", input="x")])
    p = tmp_path / "d.json"
    save_dataset(dataset, p)
    loaded = load_dataset(p)
    assert loaded.dataset_id == "d1"
