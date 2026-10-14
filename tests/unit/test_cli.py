"""
Phase 40: tests for `eval_harness/cli.py`. `demo` is covered separately in
`tests/integration/test_demo_end_to_end.py` (it IS `scripts.demo_end_to_
end.build_demo_report`'s own logic) -- not re-tested here beyond
confirming the subcommand exists and wires through, to avoid duplicating
that suite's real assertions.
"""

import json

import pytest

from eval_harness.cli import build_parser, main
from eval_harness.run_models import EvaluationRun
from eval_harness.storage import save_run

NOW = "2026-01-01T00:00:00Z"
LATER = "2026-01-02T00:00:00Z"


def _run(run_id, *, dataset_id="d1", pass_rate=1.0, average_score=1.0, completed_at=NOW) -> EvaluationRun:
    return EvaluationRun(
        run_id=run_id, dataset_id=dataset_id, dataset_version="1.0.0", dataset_content_hash="h1",
        system_name="sys", system_version="1.0",
        started_at=NOW, completed_at=completed_at, status="completed",
        total_tests=10, passed_tests=round(pass_rate * 10), failed_tests=round((1 - pass_rate) * 10),
        pass_rate=pass_rate, average_score=average_score,
        total_tokens=0, estimated_cost=0.0,
        results=[],
    )


# ---- validate-dataset ----

def test_validate_dataset_valid_file_exits_zero(capsys):
    code = main(["validate-dataset", "datasets/sample_eval_v1.json"])
    assert code == 0
    assert "VALID" in capsys.readouterr().out


def test_validate_dataset_missing_file_exits_one(tmp_path, capsys):
    code = main(["validate-dataset", str(tmp_path / "nope.json")])
    assert code == 1
    assert "INVALID" in capsys.readouterr().err


def test_validate_dataset_malformed_json_exits_one(tmp_path, capsys):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    code = main(["validate-dataset", str(p)])
    assert code == 1
    assert "INVALID" in capsys.readouterr().err


# ---- quality-gate ----

def test_quality_gate_passes_exits_zero(tmp_path, capsys):
    p = tmp_path / "run.json"
    save_run(_run("r1", pass_rate=0.95), p)
    code = main(["quality-gate", str(p), "--min-pass-rate", "0.9"])
    assert code == 0
    assert "PASSED" in capsys.readouterr().out


def test_quality_gate_fails_below_threshold_exits_one(tmp_path, capsys):
    p = tmp_path / "run.json"
    save_run(_run("r1", pass_rate=0.5), p)
    code = main(["quality-gate", str(p), "--min-pass-rate", "0.9"])
    assert code == 1
    out = capsys.readouterr().out
    assert "FAILED" in out
    assert "pass_rate" in out


def test_quality_gate_with_regressing_baseline_fails_even_if_static_thresholds_pass(tmp_path, capsys):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    save_run(_run("r1", pass_rate=1.0), baseline)
    save_run(_run("r2", pass_rate=0.5), candidate)  # regressed vs baseline
    code = main(["quality-gate", str(candidate), "--baseline", str(baseline), "--min-pass-rate", "0.1"])
    assert code == 1  # static threshold (0.1) would pass alone, but regression check fails it
    assert "regression check failed" in capsys.readouterr().out


def test_quality_gate_missing_run_file_exits_one(tmp_path, capsys):
    code = main(["quality-gate", str(tmp_path / "nope.json")])
    assert code == 1
    assert "ERROR" in capsys.readouterr().err


# ---- compare ----

def test_compare_identical_runs_exits_zero(tmp_path, capsys):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    save_run(_run("r1", pass_rate=1.0, completed_at=NOW), a)
    save_run(_run("r2", pass_rate=1.0, completed_at=LATER), b)
    code = main(["compare", str(a), str(b)])
    assert code == 0
    assert "PASSED" in capsys.readouterr().out


def test_compare_regression_exits_one(tmp_path, capsys):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    save_run(_run("r1", pass_rate=1.0, completed_at=NOW), a)
    save_run(_run("r2", pass_rate=0.0, completed_at=LATER), b)
    code = main(["compare", str(a), str(b)])
    assert code == 1
    out = capsys.readouterr().out
    assert "Regression check: FAILED" in out


def test_compare_mismatched_datasets_reports_error(tmp_path, capsys):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    save_run(_run("r1", dataset_id="d1"), a)
    save_run(_run("r2", dataset_id="d2"), b)
    code = main(["compare", str(a), str(b)])
    assert code == 1
    assert "ERROR" in capsys.readouterr().err


# ---- parser shape ----

def test_parser_requires_a_subcommand():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_demo_subcommand_is_registered():
    parser = build_parser()
    args = parser.parse_args(["demo"])
    assert args.command == "demo"
