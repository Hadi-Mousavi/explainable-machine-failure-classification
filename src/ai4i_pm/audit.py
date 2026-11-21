"""Full-source integrity checks, kept separate from training-only scientific EDA."""

from pathlib import Path
from typing import Any

import pandas as pd

from ai4i_pm.data import load_raw
from ai4i_pm.schema import (
    IDENTIFIER,
    NUMERIC_FEATURES,
    PREDICTORS,
    PROHIBITED,
    TARGET,
    validate_source_schema,
    value_issues,
)
from ai4i_pm.utils import environment_versions, write_json


def counts(series: pd.Series) -> dict[str, int]:
    """Serialize counts with stable string keys, including missing labels."""
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def audit_frame(frame: pd.DataFrame) -> dict[str, Any]:
    """Describe integrity only; failure-mode/target relationships are never computed."""
    validate_source_schema(frame)
    issues = value_issues(frame)
    if len(frame) != 10_000:
        issues.append(f"Expected 10000 source observations; found {len(frame)}")
    numeric_ranges = {}
    for name in NUMERIC_FEATURES:
        values = pd.to_numeric(frame[name], errors="coerce").dropna()
        values = values[values.abs() != float("inf")]
        numeric_ranges[name] = {
            "min": float(values.min()) if len(values) else None,
            "max": float(values.max()) if len(values) else None,
        }
    positives = int(frame[TARGET].eq(1).sum())
    negatives = int(frame[TARGET].eq(0).sum())
    return {
        "scope": "full original dataset: integrity only, not scientific EDA",
        "rows": len(frame),
        "column_count": len(frame.columns),
        "columns": list(frame.columns),
        "dtypes": {name: str(dtype) for name, dtype in frame.dtypes.items()},
        "missing_by_column": {name: int(n) for name, n in frame.isna().sum().items()},
        "missing_total": int(frame.isna().sum().sum()),
        "duplicates": {
            "complete_rows": int(frame.duplicated().sum()),
            "predictor_combinations": int(frame.duplicated(subset=list(PREDICTORS)).sum()),
            "predictor_plus_target": int(frame.duplicated(subset=[*PREDICTORS, TARGET]).sum()),
            "definition": "number of repeated rows after the first occurrence",
        },
        "identifier_unique": {
            name: bool(frame[name].is_unique and frame[name].notna().all())
            for name in (IDENTIFIER, "Product ID")
        },
        "invalid_target_count": int((~frame[TARGET].isin([0, 1])).sum()),
        "class_counts": counts(frame[TARGET]),
        "positive_fraction": positives / len(frame) if len(frame) else None,
        "negative_to_positive_ratio": negatives / positives if positives else None,
        "type_counts": counts(frame["Type"]),
        "numeric_ranges": numeric_ranges,
        "unexpected_issues": issues,
        "conservative_checks": (
            "Positive Kelvin temperatures, speed, torque; nonnegative tool wear; finite numeric "
            "values; L/M/H Type; unique positive integer UDI; Product ID pattern [LMH]\\d{5}; "
            "binary labels. No data-driven full-source outlier thresholds."
        ),
        "leakage_risks": {
            "prohibited_predictors": list(PROHIBITED),
            "allowed_predictors": list(PREDICTORS),
            "failure_mode_relationships_examined": False,
            "no_cleaning_performed": True,
        },
    }


def run_audit(root: Path) -> dict[str, Any]:
    """Save machine-readable and human-readable integrity evidence."""
    frame, checksum = load_raw(root)
    report = audit_frame(frame)
    report.update(dataset_sha256=checksum, environment=environment_versions())
    ratio = report["negative_to_positive_ratio"]
    ratio_text = f"{ratio:.3f}" if ratio is not None else "undefined (no positive labels)"
    destination = root / "results/day1"
    write_json(destination / "audit.json", report)
    lines = [
        "# Original dataset integrity audit",
        "",
        "Scope: full-source integrity verification only. Analytical EDA uses training only.",
        "Failure-mode relationships were not examined. No observations were altered.",
        "",
        f"- CSV SHA-256: `{checksum}`",
        f"- Dimensions: {report['rows']} rows x {report['column_count']} columns",
        f"- Class counts: {report['class_counts']}",
        f"- Missing cells: {report['missing_total']}",
        f"- Invalid targets: {report['invalid_target_count']}",
        f"- Identifier uniqueness: {report['identifier_unique']}",
        f"- Type counts: {report['type_counts']}",
        f"- Negative / positive ratio: {ratio_text}",
        "",
        "## Columns and missingness",
        "",
        "| Column | dtype | Missing |",
        "| --- | --- | ---: |",
    ]
    for name in frame.columns:
        lines.append(f"| {name} | {report['dtypes'][name]} | {report['missing_by_column'][name]} |")
    lines += ["", "## Duplicate checks", "", report["duplicates"]["definition"], ""]
    for key, value in report["duplicates"].items():
        if key != "definition":
            lines.append(f"- {key}: {value}")
    lines += [
        "",
        "## Numeric ranges (integrity)",
        "",
        "| Feature | Min | Max |",
        "| --- | ---: | ---: |",
    ]
    for name, bounds in report["numeric_ranges"].items():
        lines.append(f"| {name} | {bounds['min']} | {bounds['max']} |")
    lines += [
        "",
        "## Unexpected values",
        "",
        report["conservative_checks"],
        "",
        *(report["unexpected_issues"] or ["No violations of the checks above were detected."]),
        "",
        "## Leakage controls",
        "",
        "Explicit predictors: " + ", ".join(PREDICTORS) + ".",
        "Excluded: " + ", ".join(PROHIBITED) + ".",
        "Source-level counts/ranges are integrity checks, not inputs to feature selection.",
        "No duplicate removal, outlier removal, imputation, or other cleaning was performed.",
        "See docs/leakage_risks.md and audit.json for complete evidence.",
    ]
    (destination / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return report
