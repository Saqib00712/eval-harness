import json

import pytest

from eval_harness.loader import DatasetError, load_test_cases

GOLDEN = [
    {
        "test_id": "qa-001",
        "input": "What is the refund policy?",
        "expected_output": "Customers can request a refund within 30 days.",
        "reference_context": ["Customers may request a refund within 30 days..."],
        "expected_citations": [{"page": 4}],
        "metadata": {"category": "policy"},
        "tags": ["refund", "policy"],
    },
    {"test_id": "qa-002", "input": "What is the capital of France?"},
]


def test_load_json(tmp_path):
    p = tmp_path / "golden.json"
    p.write_text(json.dumps(GOLDEN), encoding="utf-8")
    cases = load_test_cases(p)
    assert len(cases) == 2
    assert cases[0].test_id == "qa-001"
    assert cases[1].expected_output is None


def test_load_jsonl(tmp_path):
    p = tmp_path / "golden.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in GOLDEN) + "\n", encoding="utf-8")
    cases = load_test_cases(p)
    assert len(cases) == 2


def test_jsonl_ignores_blank_lines(tmp_path):
    p = tmp_path / "golden.jsonl"
    p.write_text(f'\n{json.dumps(GOLDEN[0])}\n\n{json.dumps(GOLDEN[1])}\n\n', encoding="utf-8")
    cases = load_test_cases(p)
    assert len(cases) == 2


def test_missing_file(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        load_test_cases(tmp_path / "nope.json")


def test_malformed_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid", encoding="utf-8")
    with pytest.raises(DatasetError, match="not valid JSON"):
        load_test_cases(p)


def test_malformed_jsonl_line_reports_line_number(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"test_id": "qa-001", "input": "ok"}\nnot json\n', encoding="utf-8")
    with pytest.raises(DatasetError, match=r"bad\.jsonl:2 is not valid JSON"):
        load_test_cases(p)


def test_json_must_be_a_list(tmp_path):
    p = tmp_path / "bad2.json"
    p.write_text(json.dumps({"test_id": "qa-001", "input": "x"}), encoding="utf-8")
    with pytest.raises(DatasetError, match="must contain a JSON array"):
        load_test_cases(p)


def test_duplicate_ids_rejected(tmp_path):
    dupes = GOLDEN + [GOLDEN[0]]
    p = tmp_path / "dupes.json"
    p.write_text(json.dumps(dupes), encoding="utf-8")
    with pytest.raises(DatasetError, match="Duplicate"):
        load_test_cases(p)


def test_empty_dataset_rejected(tmp_path):
    p = tmp_path / "empty.json"
    p.write_text("[]", encoding="utf-8")
    with pytest.raises(DatasetError, match="no test cases"):
        load_test_cases(p)


def test_unsupported_extension_rejected(tmp_path):
    p = tmp_path / "golden.yaml"
    p.write_text("test_id: qa-001", encoding="utf-8")
    with pytest.raises(DatasetError, match="Unsupported"):
        load_test_cases(p)


def test_schema_validation_error_wrapped(tmp_path):
    p = tmp_path / "invalid.json"
    p.write_text(json.dumps([{"input": "no id given"}]), encoding="utf-8")
    with pytest.raises(DatasetError, match="failed validation"):
        load_test_cases(p)
