"""Artificial fixtures for the Phase 07A fit boundary and threshold decision."""

from copy import deepcopy

import joblib
import numpy as np
import pytest
from sklearn.metrics import fbeta_score

from ai4i_pm import final_validation as study
from ai4i_pm.feature_engineering import POWER, TEMPERATURE
from ai4i_pm.schema import PREDICTORS, SOURCE_COLUMNS, TARGET, TRAINING_COLUMNS
from ai4i_pm.validation_threshold import (
    GRID,
    decision_from_grid,
    f2_ratio,
    threshold_grid,
    verify_decision,
)


@pytest.mark.parametrize("candidate", list(study.ESTIMATORS))
def test_restore_complete_map_and_fit_identity(source_frame, tmp_path, candidate):
    params = study.ESTIMATORS[candidate](random_state=42).get_params(deep=False)
    features = list(PREDICTORS)
    if candidate.startswith("random_forest"):
        params.update(n_estimators=5, max_depth=3)
        features.extend([TEMPERATURE, POWER])
    else:
        params.update(max_iter=2)
    spec = dict(candidate=candidate, parameters=params, features=features)
    pipe = study.restore_pipeline(spec, params, features)
    with pytest.raises(ValueError, match="specification"):
        study.restore_pipeline(dict(spec, parameters={}), params, features)
    frame = source_frame.loc[:, list(TRAINING_COLUMNS)].copy()
    ids = frame.UDI.tolist()
    positives = frame[TARGET].sum()
    with pytest.raises(ValueError, match="Training-only"):
        study.fit_training(pipe, frame, ids, positives)
    frame.attrs["partition"] = "train"
    with pytest.raises(ValueError, match="Training-only"):
        study.fit_training(pipe, frame, ids[:-1], positives)
    with study.threadpool_limits(1):
        study.fit_training(pipe, frame, ids, positives)
    path = tmp_path / "local.joblib"
    joblib.dump(pipe, path)
    digest = study.sha256_file(path)
    restored = study.load_frozen(path, digest)
    assert restored.classes_.tolist() == [0, 1]
    assert restored.named_steps["classifier"].get_params(deep=False) == params
    assert study.sha256_file(path) == digest
    with path.open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        study.load_frozen(path, digest)


def test_membership_and_field_projection(tmp_path):
    parts = {"train": [1], "validation": [2, 3], "test": [4]}
    sizes = {k: len(v) for k, v in parts.items()}
    assert study.memberships(parts, sizes) == parts
    overlap = dict(parts, validation=[2, 4])
    with pytest.raises(ValueError, match="overlapping"):
        study.memberships(overlap, sizes)
    path = tmp_path / "fixture.csv"
    # Invalid UTF-8 in every forbidden payload would fail a full-record decoder.
    path.write_bytes(
        (",".join(SOURCE_COLUMNS) + "\r\n").encode("utf-8-sig")
        + b"1,\xff\n4,\xff\n3,\xff,L,300,310,1500,45,10,1,\xff\n"
        + b"2,\xff,M,299,308,1450,40,5,0,\xff\n"
    )
    with path.open("rb") as f:
        with pytest.raises(ValueError, match="Only validation"):
            study.extract_validation(f, parts, "test")
        with pytest.raises(ValueError, match="overlap"):
            study.extract_validation(f, overlap)
        frame = study.extract_validation(f, parts)
        with pytest.raises(ValueError, match="membership"):
            study.extract_validation(f, dict(parts, validation=[2, 3, 5]))
    assert frame.UDI.tolist() == [2, 3] and frame[TARGET].tolist() == [0, 1]
    assert tuple(frame.columns) == TRAINING_COLUMNS


def test_stage_reservations_survive_interruptions_and_decisions(tmp_path):
    study.create(tmp_path / "approved_protocol_record.json", b"{}")
    study.reserve_stage(tmp_path, "fit")
    with pytest.raises(ValueError, match="prohibit a fit"):
        study.reserve_stage(tmp_path, "fit")
    study.reserve_stage(tmp_path, "validate")
    with pytest.raises(FileExistsError):
        study.reserve_stage(tmp_path, "validate")
    study.create(tmp_path / "validation_decision.json", b"{}")
    with pytest.raises(ValueError, match="frozen"):
        study.reserve_stage(tmp_path, "validate")
    with pytest.raises(ValueError, match="prohibit a fit"):
        study.reserve_stage(tmp_path, "fit")


class ReversedClasses:
    classes_ = np.array([1, 0])

    def __init__(self, score):
        self.score = score
        self.calls = 0

    def predict_proba(self, features):
        self.calls += 1
        return np.array([[self.score, 1 - self.score]])


def test_probability_uses_actual_class_order():
    model = ReversedClasses(0.2)
    assert study.positive_scores(model, None).tolist() == [0.2]
    assert model.calls == 1


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -0.01, 1.01])
def test_reject_invalid_scores(invalid):
    with pytest.raises(ValueError, match="probabilities"):
        study.positive_scores(ReversedClasses(invalid), None)
    with pytest.raises(ValueError, match="scores"):
        threshold_grid([1, 0], [invalid, 0.3], GRID)


def test_grid_counts_inclusive_cutoff_and_decision():
    labels = [0, 0, 0, 1, 1, 1, 1]
    scores = np.array([0.1, 0.2, 0.5, 0.4, 0.5, 0.6, 0.8])
    rows = threshold_grid(labels, scores, GRID)
    assert len(rows) == 19
    for row in rows:
        assert row["f2"] == pytest.approx(fbeta_score(labels, scores >= row["threshold"], beta=2))
    d = decision_from_grid(rows)
    assert [d["diagnostic_050"][k] for k in ("tn", "fp", "fn", "tp")] == [2, 1, 1, 3]
    assert d["selected"]["threshold"] == 0.4
    assert d["maximizing_thresholds"] == [0.25, 0.3, 0.35, 0.4]
    assert not d["fallback"]
    verify_decision(rows, d)
    with pytest.raises(ValueError, match="disagrees"):
        verify_decision(rows, dict(d, selected=rows[-1]))
    wrong = deepcopy(rows)
    wrong[0]["recall"] = -1
    with pytest.raises(ValueError, match="counts"):
        verify_decision(wrong, d)
    with pytest.raises(ValueError, match="grid"):
        threshold_grid(labels, scores, GRID[:-1])


def test_ties_use_integer_ratios_even_when_binary64_collides():
    rows = [dict(threshold=cutoff, tp=10**17, fn=0, tn=0, fp=2) for cutoff in GRID]
    rows[0]["fp"] = 1
    assert float(f2_ratio(rows[0])) == float(f2_ratio(rows[1]))
    assert decision_from_grid(rows)["selected"] == rows[0]


def test_all_zero_overrides_highest_tie():
    rows = threshold_grid([1, 0], [0, 1], GRID)
    d = decision_from_grid(rows)
    assert d["selected"]["threshold"] == 0.50 and d["fallback"]
    assert d["selected"]["precision"] == d["selected"]["recall"] == d["selected"]["f2"] == 0
    verify_decision(rows, d)
