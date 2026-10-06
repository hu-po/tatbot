"""CLI for exact-design simulation evaluation reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tatbot_sim.evaluation import (
    SimEvalError,
    build_evaluation_report,
    evaluate_dataset,
    write_evaluation_report,
)
from tatbot_sim.repo import repo_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tatbot_sim.eval_cli")
    parser.add_argument("dataset", nargs="+", type=Path, help="sim datasets generated with --judge")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-id", default="expert", help="policy/checkpoint label")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--base-model-sha256")
    parser.add_argument(
        "--training-seed-range", action="append", nargs=2, type=int, default=[],
        metavar=("MIN", "MAX"), help="repeatable inclusive range used by training",
    )
    parser.add_argument("--created-at", help="fixed report timestamp for reproducibility")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.is_relative_to(repo_root().resolve()):
        print("sim eval: --output-dir must be outside the repository", file=sys.stderr)
        return 2
    try:
        datasets = [evaluate_dataset(path) for path in args.dataset]
        report = build_evaluation_report(
            datasets,
            checkpoint_id=args.checkpoint_id,
            checkpoint_sha256=args.checkpoint_sha256,
            base_model_sha256=args.base_model_sha256,
            training_seed_ranges=tuple(tuple(value) for value in args.training_seed_range),
            created_at=args.created_at,
        )
        write_evaluation_report(output, report)
    except SimEvalError as exc:
        print(json.dumps({"ok": False, "code": exc.code, "error": str(exc)}), file=sys.stderr)
        return 2
    print(
        f"wrote {output / 'report.json'} f1={report['metrics']['f1']['mean']:.4f} "
        f"interpretation={report['interpretation']}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
