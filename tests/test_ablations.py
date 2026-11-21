"""Phase 04 unit and replay checks. All observations and scores here are artificial."""

import json
import runpy
import tomllib
from math import pi
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from test_baselines import exported_root as exported_root
from threadpoolctl import threadpool_limits

from ai4i_pm.ablation_report import paired_summary
from ai4i_pm.ablations import (
    CANDIDATES,
    CONFIG,
    ESTIMATORS,
    FOLDS,
    INPUTS,
    REFERENCE,
    SELECTED,
    check_reference,
    evaluate_condition,
    file_hashes,
    load_design,
)
from ai4i_pm.cli import main
from ai4i_pm.feature_engineering import (
    POWER,
    TEMPERATURE,
    ablation_pipeline,
    engineer_features,
    mechanical_power,
    temperature_difference,
)
from ai4i_pm.model_search import frozen_splits, pipelines, score_ap
from ai4i_pm.schema import IDENTIFIER, PREDICTORS, PROHIBITED, TARGET
from ai4i_pm.splits import load_training
from ai4i_pm.utils import read_json, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = tomllib.loads((ROOT / CONFIG).read_text())


def test_units_and_frozen_matrix(source_frame):
    assert temperature_difference(296.25, 300) == -3.75
    assert mechanical_power(10, 60) == pytest.approx(20 * pi)
    original = list(PREDICTORS)
    both = [TEMPERATURE, POWER]
    expected = [
        original,
        original[:-1],
        original + [TEMPERATURE],
        original + [POWER],
        original + both,
        original[:-1] + both,
    ]
    assert list(PROTOCOL["features"].items()) == list(
        zip([f"A{i}" for i in range(6)], expected, strict=True)
    )
    frame = source_frame.set_index(IDENTIFIER).iloc[::-1].loc[:, PREDICTORS]
    for columns in expected:
        result = engineer_features(frame, columns)
        assert result.columns.tolist() == columns
        pd.testing.assert_index_equal(result.index, frame.index)
        pd.testing.assert_frame_equal(result, engineer_features(frame.copy(), columns))
        retained = [col for col in columns if col in original]
        pd.testing.assert_frame_equal(frame[retained], result[retained])
        if POWER in columns:
            np.testing.assert_allclose(
                result[POWER],
                frame["Torque [Nm]"] * frame["Rotational speed [rpm]"] * 2 * pi / 60,
                rtol=1e-15,
            )
        if TEMPERATURE in columns:
            np.testing.assert_array_equal(
                result[TEMPERATURE], frame["Process temperature [K]"] - frame["Air temperature [K]"]
            )


@pytest.mark.parametrize("extra", [*PROHIBITED, TARGET, "other"])
def test_reject_forbidden_schema_and_feature_requests(source_frame, extra):
    frame = source_frame.loc[:, PREDICTORS].copy()
    with pytest.raises(ValueError, match="authorized"):
        engineer_features(frame, [*PREDICTORS, extra])
    with pytest.raises(ValueError, match="exactly"):
        engineer_features(frame.assign(**{extra: 999}), PREDICTORS)
    with pytest.raises(ValueError, match="exactly"):
        engineer_features(frame.iloc[:, :-1], PREDICTORS)


def test_fold_local_type_and_numeric_order(source_frame):
    frame = source_frame.loc[:, PREDICTORS].copy()
    frame.loc[:9, "Type"] = "unseen"
    columns = PROTOCOL["features"]["A5"]
    model = ablation_pipeline(RandomForestClassifier(n_estimators=2, random_state=42), columns)
    model.fit(frame.iloc[10:], source_frame[TARGET].iloc[10:])
    assert "unseen" not in model["features"].named_transformers_["type"].categories_[0]
    matrix = model[:-1].transform(frame.iloc[:10])
    assert np.all(matrix[:, :3] == 0)
    np.testing.assert_array_equal(
        matrix[:, 3:], engineer_features(frame.iloc[:10], columns).iloc[:, 1:]
    )


@pytest.fixture
def artificial_design(exported_root):
    root = exported_root
    prior = tomllib.loads((root / "configs/02_baseline_evaluation.toml").read_text())
    data = load_training(root).set_index(IDENTIFIER).sort_index()
    x, y = data.loc[:, PREDICTORS], data[TARGET]
    # Saved UDI order deliberately differs from the usual Phase 02 splitter.
    groups = [[] for _ in range(5)]
    offset = 0
    for _, part in data.groupby(TARGET):
        for j, identity in enumerate(part.index[::-1]):
            groups[(offset + j) % 5].append(identity)
        offset += len(part)
    manifest = {
        "primary_partition": "train",
        "inputs": prior["data"],
        "settings": prior["cv"],
        "evaluation_udis": {str(i): list(reversed(g)) for i, g in enumerate(groups, 1)},
    }
    write_json(root / FOLDS, manifest)
    splits, records = frozen_splits(y, manifest)
    models = pipelines(prior["models"], 42)
    trials, reference = [], []
    with threadpool_limits(limits=1):
        for candidate in ESTIMATORS:
            family = candidate.rsplit("-", 1)[0]
            model = models[family]
            if family == "random_forest":
                model.set_params(classifier__n_estimators=3)
            else:
                model.set_params(classifier__max_iter=2, classifier__class_weight="balanced")
            params = model["classifier"].get_params()
            trials.append(
                {
                    "candidate_id": candidate,
                    "classifier_parameters": json.dumps(params),
                    "failed_folds": 0,
                }
            )
            for (fit, held), record in zip(splits, records, strict=True):
                fitted = clone(model).fit(x.iloc[fit], y.iloc[fit])
                reference.append(
                    {
                        "candidate_id": candidate,
                        **record,
                        "ap": score_ap(fitted, x.iloc[held], y.iloc[held]),
                    }
                )
    write_json(
        root / SELECTED,
        {
            "candidate_id": trials[0]["candidate_id"],
            "status": "provisional",
            "classifier_parameters": json.loads(trials[0]["classifier_parameters"]),
        },
    )
    pd.DataFrame(trials).to_csv(root / CANDIDATES, index=False)
    pd.DataFrame(reference).to_csv(root / REFERENCE, index=False)
    cfg = (ROOT / CONFIG).read_text()
    for name in INPUTS:
        cfg = cfg.replace(PROTOCOL["inputs_sha256"][name], sha256_file(root / name))
    (root / CONFIG).write_text(cfg, encoding="utf-8")
    return root


def test_training_only_cli_reference_pairs_and_replay(artificial_design, monkeypatch, capsys):
    root = artificial_design
    original = Path.open

    def guard(path, *args, **kwargs):
        if path.is_relative_to(root / "data"):
            assert path == root / "data/processed/train.csv"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard)
    assert main(["--root", str(root), "ablations"]) == 0
    output = root / "results/04_feature_ablation"
    summary = pd.read_csv(output / "ablation_summary.csv")
    metrics = pd.read_csv(output / "ablation_fold_metrics.csv")
    assert len(metrics) == 60 and len(summary) == 12
    expected = pd.read_csv(root / REFERENCE)
    for row in summary.itertuples():
        part = metrics[
            (metrics.condition == row.condition) & (metrics.candidate_id == row.candidate_id)
        ]
        ref = expected[expected.candidate_id == row.candidate_id]
        assert part.eval_udis_sha256.tolist() == ref.eval_udis_sha256.tolist()
        delta = part.ap.to_numpy() - ref.ap.to_numpy()
        np.testing.assert_allclose(
            [getattr(row, f"delta_fold_{i}") for i in range(1, 6)], delta, atol=1e-15
        )
        assert row.mean_delta_ap == pytest.approx(delta.mean())
        assert row.sample_sd == pytest.approx(np.std(part.ap, ddof=1))
        if row.condition == "A0":
            np.testing.assert_array_equal(part.ap, ref.ap)
    before = file_hashes(output)
    runpy.run_path(str(ROOT / "scripts/verify_04_feature_ablation.py"))["verify"](root)
    assert "10/10 artifacts byte-identical" in capsys.readouterr().out
    assert before == file_hashes(output)
    assert read_json(output / "execution.json")["artifacts_sha256"] == {
        k: v for k, v in before.items() if k != "execution.json"
    }
    ref = metrics[metrics.condition == "A0"].iloc[:5].to_dict("records")
    ref[0]["ap"] += 0.01
    with pytest.raises(ValueError, match="A0 AP differs"):
        check_reference(ref, expected, 1e-12)
    with pytest.raises(ValueError, match="matched A0"):
        paired_summary(metrics.iloc[1:], "A0", 1)
    (root / SELECTED).write_text("{}")
    with pytest.raises(ValueError, match="frozen"):
        load_design(root)


def test_fit_error_omits_observations(source_frame):
    class Broken(RandomForestClassifier):
        def fit(self, X, y):
            raise ValueError("sensitive fixture value")

    x = source_frame.loc[:, PREDICTORS]
    y = source_frame[TARGET]
    manifest = {
        "primary_partition": "train",
        "evaluation_udis": {str(i + 1): list(range(i * 20, (i + 1) * 20)) for i in range(5)},
    }
    splits, records = frozen_splits(y, manifest)
    with pytest.raises(ValueError, match=r"^A0 random_forest-11 fold 1 failed \(ValueError\)$"):
        evaluate_condition("A0", "random_forest-11", Broken(), x, y, splits, records)


def test_observed_fit_and_score_ids_and_continuous_ap(source_frame):
    """Audit actual estimator calls and compute AP from precision at each positive rank."""
    seen_fit, seen_score = [], []

    class FixtureRanker(BaseEstimator):
        def fit(self, X, y):
            seen_fit.append(list(X.index))
            assert X.index.equals(y.index)
            self.classes_ = np.array([0, 1])
            return self

        def predict_proba(self, X):
            seen_score.append(list(X.index))
            probability = X["Torque [Nm]"].to_numpy() / 100
            return np.column_stack((1 - probability, probability))

    data = source_frame.set_index(IDENTIFIER).iloc[::-1]
    x, y = data.loc[:, PREDICTORS], data[TARGET]
    members = {str(i + 1): list(range(1 + i * 20, 21 + i * 20)) for i in range(5)}
    splits, records = frozen_splits(y, {"primary_partition": "train", "evaluation_udis": members})
    rows = evaluate_condition("A0", "fixture", FixtureRanker(), x, y, splits, records)
    for i, ids in enumerate(members.values()):
        assert seen_score[i] == ids
        assert seen_fit[i] == list(x.index[~x.index.isin(ids)])
        order = x.loc[ids, "Torque [Nm]"].sort_values(ascending=False).index
        positive_ranks = np.flatnonzero(y.loc[order].to_numpy()) + 1
        expected_ap = np.mean(np.arange(1, len(positive_ranks) + 1) / positive_ranks)
        assert rows[i]["ap"] == pytest.approx(expected_ap)
