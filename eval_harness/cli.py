"""
Phase 40: a CLI for eval-harness.

Scope decision (explicit, made with the user before building this): this
harness has been a Python library + scripts for all 21 prior days, with
no HTTP API and no Docker image. Building a full FastAPI service was
judged out of scope for "finishing remaining phases" -- it's substantial
new scope, not a gap in what was already planned. A CLI, by contrast, is
a thin wrapper around functions that already exist and are already
tested (`load_dataset`, `evaluate_quality_gate`, `compare_runs`,
`exit_code_for`, `scripts/demo_end_to_end.py`), so it was in scope.

What's deliberately NOT here: a generic `eval-harness run <system>`
command. Running an arbitrary `SystemUnderTest` from the command line
would need a config language for instantiating one (which HTTP adapter,
which evaluators, which embedding/judge provider, API keys...) that
doesn't exist and would be real new design work, not a thin wrapper.
`demo` exercises the full pipeline instead (dataset -> two systems ->
comparison -> regression -> quality gate -> storage -> observability);
`quality-gate` and `compare` operate on already-persisted
`EvaluationRun` JSON files (via `storage.load_run`), which IS something
this CLI can do honestly without inventing a system-config format.

Commands:
    validate-dataset <path>                 exit 0 if valid, 1 if not
    quality-gate <run.json> [options]       exit 0 if the gate passes, 1 if not
    compare <baseline.json> <candidate.json> [options]
    demo                                    runs scripts/demo_end_to_end.py
"""

import argparse
import asyncio
import sys
from pathlib import Path

from eval_harness.dataset_io import DatasetError, load_dataset
from eval_harness.quality_gate import QualityGatePolicy, evaluate_quality_gate, exit_code_for
from eval_harness.regression import RegressionPolicy, compare_runs
from eval_harness.storage import StorageError, load_run


def _cmd_validate_dataset(args: argparse.Namespace) -> int:
    try:
        dataset = load_dataset(args.path)
    except DatasetError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print(f"VALID: {dataset.name!r} (dataset_id={dataset.dataset_id}, version={dataset.version})")
    print(f"  {dataset.test_count} test cases, content_hash={dataset.content_hash}")
    return 0


def _cmd_quality_gate(args: argparse.Namespace) -> int:
    try:
        candidate = load_run(args.run)
    except StorageError as exc:
        print(f"ERROR loading {args.run}: {exc}", file=sys.stderr)
        return 1

    regression_report = None
    if args.baseline:
        try:
            baseline = load_run(args.baseline)
        except StorageError as exc:
            print(f"ERROR loading baseline {args.baseline}: {exc}", file=sys.stderr)
            return 1
        try:
            regression_report = compare_runs(baseline, candidate, RegressionPolicy())
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    policy = QualityGatePolicy(min_pass_rate=args.min_pass_rate, min_average_score=args.min_average_score)
    result = evaluate_quality_gate(candidate, policy, regression_report)

    print(f"{'PASSED' if result.passed else 'FAILED'} -- pass_rate={result.pass_rate:.4f} average_score={result.average_score:.4f}")
    for v in result.violations:
        print(f"  - {v}")
    return exit_code_for(result)


def _cmd_compare(args: argparse.Namespace) -> int:
    try:
        baseline = load_run(args.baseline)
        candidate = load_run(args.candidate)
    except StorageError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        report = compare_runs(baseline, candidate, RegressionPolicy())
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"baseline  pass_rate={report.baseline_pass_rate:.4f}  average_score={report.baseline_average_score:.4f}")
    print(f"candidate pass_rate={report.candidate_pass_rate:.4f}  average_score={report.candidate_average_score:.4f}")
    print(f"delta     pass_rate={report.pass_rate_delta:+.4f}  average_score={report.average_score_delta:+.4f}")
    print(f"newly passing: {report.newly_passing_test_ids}")
    print(f"regressed:     {report.regressed_test_ids}")
    print(f"Regression check: {'PASSED' if report.passed else 'FAILED'}")
    for v in report.violations:
        print(f"  - {v}")
    return 0 if report.passed else 1


def _cmd_demo(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    from scripts.demo_end_to_end import main as demo_main

    asyncio.run(demo_main())
    return 0  # the demo's own illustrative exit code is printed, not returned here -- see scripts/demo_end_to_end.py


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eval-harness", description="eval-harness CLI (Phase 40)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate-dataset", help="Validate a dataset JSON/JSONL file")
    p_validate.add_argument("path")
    p_validate.set_defaults(func=_cmd_validate_dataset)

    p_gate = sub.add_parser("quality-gate", help="Evaluate a quality gate against a persisted run")
    p_gate.add_argument("run", help="Path to a run JSON file (eval_harness.storage.save_run format)")
    p_gate.add_argument("--baseline", help="Optional baseline run JSON, for a regression check too")
    p_gate.add_argument("--min-pass-rate", type=float, default=None)
    p_gate.add_argument("--min-average-score", type=float, default=None)
    p_gate.set_defaults(func=_cmd_quality_gate)

    p_compare = sub.add_parser("compare", help="Compare two persisted runs for regressions")
    p_compare.add_argument("baseline")
    p_compare.add_argument("candidate")
    p_compare.set_defaults(func=_cmd_compare)

    p_demo = sub.add_parser("demo", help="Run the Day 20 end-to-end demonstration")
    p_demo.set_defaults(func=_cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
