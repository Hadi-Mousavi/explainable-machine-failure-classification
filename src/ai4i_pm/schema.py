"""Central source schema and strict predictor allowlist; no inferred features."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

IDENTIFIER = "UDI"
TARGET = "Machine failure"
NUMERIC_FEATURES = (
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
)
PREDICTORS = ("Type", *NUMERIC_FEATURES)
FAILURE_MODES = ("TWF", "HDF", "PWF", "OSF", "RNF")
PROHIBITED = (IDENTIFIER, "Product ID", *FAILURE_MODES)
SOURCE_COLUMNS = (IDENTIFIER, "Product ID", *PREDICTORS, TARGET, *FAILURE_MODES)
TRAINING_COLUMNS = (IDENTIFIER, *PREDICTORS, TARGET)


def validate_source_schema(frame: pd.DataFrame) -> None:
    """Require the original CSV names and order, including UDI rather than UID."""
    if tuple(frame.columns) != SOURCE_COLUMNS:
        raise ValueError(f"Expected source columns {SOURCE_COLUMNS}; got {tuple(frame.columns)}")


def predictor_matrix(frame: pd.DataFrame, columns: Sequence[str] = PREDICTORS) -> pd.DataFrame:
    """Select only explicitly permitted features; reject unknown/duplicate requests."""
    requested = tuple(columns)
    unexpected = set(requested) - set(PREDICTORS)
    if not requested or unexpected or len(set(requested)) != len(requested):
        raise ValueError(
            f"Invalid predictor request: {requested}; prohibited/unknown: {unexpected}"
        )
    missing = set(requested) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing predictors: {missing}")
    return frame.loc[:, list(requested)].copy()


def validate_target(series: pd.Series) -> None:
    """Reject missing, nonnumeric, and nonbinary labels without coercion."""
    if not pd.api.types.is_numeric_dtype(series) or series.isna().any():
        raise ValueError("Target must be numeric, nonmissing, and binary")
    if not series.isin([0, 1]).all():
        raise ValueError("Target must contain only 0 and 1")


def value_issues(frame: pd.DataFrame) -> list[str]:
    """Check conservative domain constraints, without cleaning or label comparisons."""
    issues: list[str] = []
    if frame.isna().any().any():
        issues.append("Missing values detected")
    try:
        validate_target(frame[TARGET])
    except ValueError as exc:
        issues.append(str(exc))
    if not frame["Type"].isin(["L", "M", "H"]).all():
        issues.append("Type contains values outside L/M/H")
    ids = frame[IDENTIFIER]
    if (
        not pd.api.types.is_integer_dtype(ids)
        or ids.isna().any()
        or ids.duplicated().any()
        or (ids <= 0).any()
    ):
        issues.append("UDI must be unique, positive, nonmissing integers")
    if "Product ID" in frame:
        products = frame["Product ID"].astype("string")
        if not products.str.fullmatch(r"[LMH]\d{5}", na=False).all():
            issues.append("Product ID has unexpected format")
        if products.duplicated().any():
            issues.append("Product ID is not unique")
    for name in NUMERIC_FEATURES:
        values = frame[name]
        if not pd.api.types.is_numeric_dtype(values):
            issues.append(f"{name}: nonnumeric dtype")
            continue
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            issues.append(f"{name}: missing or nonfinite values")
        invalid = values < 0 if name == "Tool wear [min]" else values <= 0
        if invalid.any():
            issues.append(f"{name}: outside conservative physical domain")
    # Univariate validity only: never compare failure modes with the target.
    for name in FAILURE_MODES:
        if name in frame and not frame[name].isin([0, 1]).all():
            issues.append(f"{name}: nonbinary or missing indicator")
    return issues
