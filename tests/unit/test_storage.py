import pytest

from eval_harness.calibration import HumanLabel
from eval_harness.run_models import EvaluationRun
from eval_harness.storage import RunStore, StorageError, load_labels, load_run, save_labels, save_run

NOW = "2026-01-01T00:00:00Z"
LATER = "2026-01-02T00:00:00Z"


def _run(run_id, dataset_id="d1", system_name="sys", completed_at=NOW) -> EvaluationRun:
    return EvaluationRun(
        run_id=run_id, dataset_id=dataset_id, dataset_version="1.0.0", dataset_content_hash="h1",
        system_name=system_name, system_version="0.0.0",
        started_at=NOW, completed_at=completed_at, status="completed",
        total_tests=1, passed_tests=1, failed_tests=0, pass_rate=1.0, average_score=1.0,
        total_tokens=0, estimated_cost=0.0,
    )


# ---- save_run / load_run ----

def test_round_trips_a_run(tmp_path):
    run = _run("r1")
    path = tmp_path / "runs" / "r1.json"  # nested dir that doesn't exist yet
    save_run(run, path)
    loaded = load_run(path)
    assert loaded == run


def test_load_run_missing_file_raises(tmp_path):
    with pytest.raises(StorageError, match="not found"):
        load_run(tmp_path / "nope.json")


def test_load_run_invalid_json_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(StorageError, match="not valid JSON"):
        load_run(p)


def test_load_run_wrong_shape_raises(tmp_path):
    p = tmp_path / "wrong.json"
    p.write_text('{"not": "a run"}', encoding="utf-8")
    with pytest.raises(StorageError, match="failed validation"):
        load_run(p)


# ---- save_labels / load_labels ----

def test_round_trips_labels(tmp_path):
    labels = [
        HumanLabel(run_id="r1", test_id="t1", evaluator="faithfulness", human_passed=True, rater="alice"),
        HumanLabel(run_id="r1", test_id="t2", evaluator="faithfulness", human_passed=False, rater="alice"),
    ]
    path = tmp_path / "labels.json"
    save_labels(labels, path)
    loaded = load_labels(path)
    assert loaded == labels


def test_load_labels_missing_file_raises(tmp_path):
    with pytest.raises(StorageError, match="not found"):
        load_labels(tmp_path / "nope.json")


def test_load_labels_invalid_json_raises(tmp_path):
    # Day 18 test-suite audit: load_run has a "not valid JSON" test for its
    # own json.loads() call; load_labels uses TypeAdapter.validate_json
    # instead, which parses JSON internally and reports malformed JSON as a
    # ValidationError, not a json.JSONDecodeError -- so this is "failed
    # validation", not "not valid JSON". (This test originally asserted
    # the wrong message and failed, which is how this difference was
    # actually caught.)
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(StorageError, match="failed validation"):
        load_labels(p)


def test_load_labels_wrong_shape_raises(tmp_path):
    p = tmp_path / "wrong.json"
    p.write_text('[{"not": "a label"}]', encoding="utf-8")
    with pytest.raises(StorageError, match="failed validation"):
        load_labels(p)


# ---- RunStore ----

def test_run_store_saves_and_retrieves_latest(tmp_path):
    store = RunStore(tmp_path)
    older = _run("r-old", completed_at=NOW)
    newer = _run("r-new", completed_at=LATER)
    store.save(older)
    store.save(newer)
    latest = store.latest("d1", "sys")
    assert latest.run_id == "r-new"


def test_run_store_latest_is_none_when_nothing_stored(tmp_path):
    store = RunStore(tmp_path)
    assert store.latest("d1", "sys") is None


def test_run_store_list_runs_sorted_oldest_first(tmp_path):
    store = RunStore(tmp_path)
    store.save(_run("r-b", completed_at=LATER))
    store.save(_run("r-a", completed_at=NOW))
    runs = store.list_runs("d1", "sys")
    assert [r.run_id for r in runs] == ["r-a", "r-b"]


def test_run_store_separates_by_dataset_and_system(tmp_path):
    store = RunStore(tmp_path)
    store.save(_run("r1", dataset_id="d1", system_name="sys-a"))
    store.save(_run("r2", dataset_id="d1", system_name="sys-b"))
    assert len(store.list_runs("d1", "sys-a")) == 1
    assert len(store.list_runs("d1", "sys-b")) == 1
    assert store.list_runs("d2", "sys-a") == []


# ---- path traversal hardening (Day 17) ----

def test_run_store_rejects_path_traversal_in_dataset_id(tmp_path):
    store = RunStore(tmp_path)
    run = _run("r1", dataset_id="../../etc", system_name="sys")
    with pytest.raises(StorageError, match="not safe"):
        store.save(run)


def test_run_store_rejects_path_traversal_in_system_name(tmp_path):
    store = RunStore(tmp_path)
    run = _run("r1", dataset_id="d1", system_name="..\\..\\windows")
    with pytest.raises(StorageError, match="not safe"):
        store.save(run)


def test_run_store_rejects_blank_components(tmp_path):
    store = RunStore(tmp_path)
    run = _run("r1", dataset_id="   ", system_name="sys")
    with pytest.raises(StorageError, match="must not be blank"):
        store.save(run)


def test_run_store_save_never_escapes_root(tmp_path):
    store = RunStore(tmp_path)
    try:
        store.save(_run("r1", dataset_id="../escape", system_name="sys"))
    except StorageError:
        pass
    # nothing should have been written outside the configured root
    assert not (tmp_path.parent / "escape").exists()
