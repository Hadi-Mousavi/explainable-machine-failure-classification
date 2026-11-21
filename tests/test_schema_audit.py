"""Offline tests of source integrity and explicit feature restrictions."""

import numpy as np
import pytest

from ai4i_pm.audit import audit_frame, run_audit
from ai4i_pm.schema import (
    FAILURE_MODES,
    PREDICTORS,
    PROHIBITED,
    TARGET,
    predictor_matrix,
    validate_source_schema,
    validate_target,
    value_issues,
)


def test_original_source_names_and_order(source_frame):
    validate_source_schema(source_frame)
    for changed in (
        source_frame.rename(columns={"UDI": "UID"}),
        source_frame.drop(columns="RNF"),
        source_frame.assign(extra=0),
        source_frame.iloc[:, ::-1],
    ):
        with pytest.raises(ValueError, match="Expected source columns"):
            validate_source_schema(changed)


def test_predictors_are_exactly_allowlisted(source_frame):
    selected = predictor_matrix(source_frame)
    assert tuple(selected.columns) == PREDICTORS
    assert set(selected.columns).isdisjoint(PROHIBITED)
    assert TARGET not in selected


@pytest.mark.parametrize("name", [*PROHIBITED, TARGET, "unknown"])
def test_reject_prohibited_or_unknown_predictors(source_frame, name):
    with pytest.raises(ValueError, match="Invalid predictor request"):
        predictor_matrix(source_frame, [*PREDICTORS, name])


def test_reject_duplicate_empty_or_missing_features(source_frame):
    for request in ([], ["Type", "Type"]):
        with pytest.raises(ValueError):
            predictor_matrix(source_frame, request)
    with pytest.raises(ValueError, match="Missing predictors"):
        predictor_matrix(source_frame.drop(columns="Type"))


@pytest.mark.parametrize("invalid", [2, -1, np.nan, "failure"])
def test_target_validity(source_frame, invalid):
    validate_target(source_frame[TARGET])
    labels = source_frame[TARGET].astype(object)
    labels.iloc[0] = invalid
    if invalid != "failure":
        labels = labels.astype(float)
    with pytest.raises(ValueError):
        validate_target(labels)


def test_missingness_detected_without_mutation(source_frame):
    source_frame.loc[0, "Torque [Nm]"] = np.nan
    original = source_frame.copy(deep=True)
    audit = audit_frame(source_frame)
    assert audit["missing_total"] == 1
    assert audit["missing_by_column"]["Torque [Nm]"] == 1
    assert "Missing values detected" in audit["unexpected_issues"]
    assert source_frame.equals(original)


def test_three_duplicate_definitions(source_frame):
    base = source_frame.iloc[[0, 0, 0]].copy()
    # Three identical predictors: two same-target repeats, one opposite target.
    base.iloc[2, base.columns.get_loc(TARGET)] = 1
    audit = audit_frame(base)
    assert audit["duplicates"]["complete_rows"] == 1
    assert audit["duplicates"]["predictor_combinations"] == 2
    assert audit["duplicates"]["predictor_plus_target"] == 1
    assert not audit["identifier_unique"]["UDI"]


def test_domain_anomalies_are_reported(source_frame):
    source_frame.loc[0, "Type"] = "X"
    source_frame.loc[1, "Torque [Nm]"] = -1
    source_frame.loc[2, "Air temperature [K]"] = np.inf
    issues = value_issues(source_frame)
    assert any("Type" in issue for issue in issues)
    assert any("physical domain" in issue for issue in issues)
    assert any("nonfinite" in issue for issue in issues)


def test_audit_does_not_compare_failure_mode_relationships(source_frame):
    before = audit_frame(source_frame)
    for name in FAILURE_MODES:
        source_frame[name] = 1 - source_frame[name]
    after = audit_frame(source_frame)
    assert before == after
    assert after["leakage_risks"]["failure_mode_relationships_examined"] is False


def test_readable_audit_handles_no_positive_labels(source_frame, tmp_path, monkeypatch):
    source_frame[TARGET] = 0
    monkeypatch.setattr("ai4i_pm.audit.load_raw", lambda root: (source_frame, "a" * 64))
    report = run_audit(tmp_path)
    assert report["negative_to_positive_ratio"] is None
    assert (
        "undefined (no positive labels)"
        in (tmp_path / "results/01_data_exploration/audit.md").read_text()
    )
