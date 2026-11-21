"""CPU-only commands for data preparation and training-only analysis."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from ai4i_pm.audit import run_audit
from ai4i_pm.baselines import run_baselines
from ai4i_pm.data import acquire
from ai4i_pm.eda import generate_eda
from ai4i_pm.splits import freeze_split, read_settings


def main(argv: list[str] | None = None) -> int:
    """Run one explicit reproducible workflow stage from a repository root."""
    parser = argparse.ArgumentParser(prog="ai4i-pm", description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path.cwd(), help="Repository root (default: cwd)"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    acquisition = commands.add_parser(
        "acquire", help="Official UCI download or immutable verification"
    )
    acquisition.add_argument(
        "--manual", type=Path, help="User-downloaded official UCI .zip or .csv"
    )
    commands.add_parser("audit", help="Full-source integrity verification only")
    split = commands.add_parser("split", help="Freeze primary partitions and training-only export")
    split.add_argument(
        "--seed", type=int, help="Explicit alternative to configs/01_data_exploration.toml"
    )
    split.add_argument(
        "--override", action="store_true", help="Archive and replace a changed split"
    )
    split.add_argument("--reason", help="Required audit reason for an override")
    commands.add_parser("eda", help="Generate training-only figures and observations")
    commands.add_parser("baselines", help="Evaluate fixed baselines on frozen training folds")
    commands.add_parser("compare-models", help="Compare classifiers on the saved Phase 02 folds")
    commands.add_parser("ablations", help="Evaluate fixed feature sets on the saved training folds")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        if args.command == "acquire":
            result = acquire(root, args.manual)
            print(f"Verified source bytes: CSV SHA-256 {result['csv_sha256']}")
        elif args.command == "audit":
            result = run_audit(root)
            print(
                f"Audit: {result['rows']} x {result['column_count']}; "
                f"issues: {result['unexpected_issues']}"
            )
            if result["unexpected_issues"]:
                return 1
        elif args.command == "split":
            settings = read_settings(root)
            if args.seed is not None:
                settings = replace(settings, seed=args.seed)
            result = freeze_split(root, settings, override=args.override, reason=args.reason)
            print(json.dumps(result["validation"], indent=2))
        elif args.command == "baselines":
            print(json.dumps(run_baselines(root), indent=2))
        elif args.command == "compare-models":
            from ai4i_pm.comparison import run_comparison

            print(json.dumps(run_comparison(root), indent=2))
        elif args.command == "ablations":
            from ai4i_pm.ablations import run_ablations

            record = run_ablations(root)
            print(f"Completed {record['fold_evaluations']} training-fold evaluations.")
        else:
            result = generate_eda(root)
            print(f"EDA: {result['rows']} training rows; {len(result['figures'])} PNG figures")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"{args.command} failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
