"""Day 2 regression tests use artificial observations exclusively."""

import tomllib
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from ai4i_pm.baselines import run_baselines, verified_training
from ai4i_pm.cli import main
from ai4i_pm.cv import assign_folds, cross_validate
from ai4i_pm.evaluation import fold_metrics
from ai4i_pm.models import baseline_pipelines, positive_scores
from ai4i_pm.schema import IDENTIFIER, NUMERIC_FEATURES, PREDICTORS, PROHIBITED, TARGET
from ai4i_pm.splits import build_manifest, training_csv
from ai4i_pm.utils import read_json, sha256_bytes, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (ROOT / "configs/day2.toml").read_text(encoding="utf-8")
SETTINGS = tomllib.loads(CONFIG)


@pytest.mark.parametrize("extra", [*PROHIBITED, TARGET, "unapproved", "missing", "duplicate"])
def test_model_rejects_invalid_input_matrix(source_frame, extra):
    features = source_frame.loc[:, PREDICTORS].copy()
    if extra == "missing":
        features = features.iloc[:, :-1]
    elif extra == "duplicate":
        features = pd.concat([features, features.iloc[:, [0]]], axis=1)
    else:
        features[extra] = 0
    model = baseline_pipelines(SETTINGS["models"])["logistic_regression"]
    with pytest.raises(ValueError, match="exactly|unique column names"):
        model.fit(features, source_frame[TARGET])


def test_binary_score_column_uses_fitted_class_order(source_frame):
    class Reversed:
        classes_ = np.array([1, 0])

        def predict_proba(self, features):
            return np.tile([0.2, 0.8], (len(features), 1))

    np.testing.assert_array_equal(positive_scores(Reversed(), source_frame), 0.2)


def test_hand_computed_ap_and_threshold_ties():
    actual = np.array([1, 0, 1, 1, 0])
    scores = np.array([0.5, 0.8, 0.3, 0.2, 0.1])
    result = fold_metrics(actual, scores, SETTINGS["metrics"])
    assert result == pytest.approx(
        {
            "average_precision": (1 / 2 + 2 / 3 + 3 / 4) / 3,
            "diagnostic_threshold": 0.5,
            "precision": 1 / 2,
            "recall": 1 / 3,
            "f1": 2 / 5,
            "f2": 5 / 14,
            "tn": 1,
            "fp": 1,
            "fn": 2,
            "tp": 1,
        }
    )
    zero = fold_metrics(actual, np.zeros(5), SETTINGS["metrics"])
    assert all(zero[k] == 0 for k in ("precision", "recall", "f1", "f2"))


def test_folds_and_preprocessing_are_paired_and_isolated(source_frame, monkeypatch):
    target = source_frame.set_index(IDENTIFIER)[TARGET]
    folds = assign_folds(target, SETTINGS["cv"])
    assert folds.equals(assign_folds(target.sample(frac=1, random_state=7), SETTINGS["cv"]))
    assert set(folds) == {1, 2, 3, 4, 5}
    # Give one evaluation fold a category absent from its fit partition.
    source_frame.loc[source_frame[IDENTIFIER].isin(folds[folds == 1].index), "Type"] = "X"
    actual_fit = ColumnTransformer.fit_transform
    fit_sets = []

    def inspect_fit(self, X, y=None, **params):
        assert tuple(X.columns) == PREDICTORS
        result = actual_fit(self, X, y, **params)
        assert set(self.named_transformers_["type"].categories_[0]) == set(X["Type"])
        np.testing.assert_allclose(
            self.named_transformers_["numeric"].mean_, X.loc[:, NUMERIC_FEATURES].mean()
        )
        fit_sets.append(set(X.index))
        return result

    monkeypatch.setattr(ColumnTransformer, "fit_transform", inspect_fit)
    models = baseline_pipelines(SETTINGS["models"])
    assigned, metrics, oof = cross_validate(
        source_frame, models, SETTINGS["cv"], SETTINGS["metrics"]
    )
    assert assigned.equals(folds)
    assert len(fit_sets) == 10
    assert not oof.duplicated(["model", IDENTIFIER]).any()
    for name in models:
        scored = oof[oof.model == name].set_index(IDENTIFIER)
        pd.testing.assert_series_equal(scored.fold, folds)
        pd.testing.assert_series_equal(scored[TARGET], target)
    assert set(metrics.model) == set(models)
    for k in range(1, 6):
        evaluation = set(folds[folds == k].index)
        fit = set(folds.index) - evaluation
        assert fit_sets[2 * k - 2 : 2 * k] == [fit, fit]
        assert len(evaluation) == 20
        assert set(target.loc[sorted(fit)]) == set(target.loc[sorted(evaluation)]) == {0, 1}
        dummy = oof[(oof.fold == k) & (oof.model == "dummy_prior")]
        np.testing.assert_allclose(dummy.positive_score, target.loc[sorted(fit)].mean())
        ap = metrics.loc[
            (metrics.fold == k) & (metrics.model == "dummy_prior"), "average_precision"
        ]
        assert ap.item() == pytest.approx(dummy[TARGET].mean())


@pytest.fixture
def exported_root(tmp_path, source_frame):
    """An isolated partial checkout with only an artificial training export on disk."""
    day1 = tmp_path / "results/day1"
    source_hash = sha256_bytes(source_frame.to_csv(index=False).encode())
    manifest = build_manifest(source_frame, source_hash)
    write_json(day1 / "split_manifest.json", manifest)
    ids = manifest["partitions"]["train"]
    export = tmp_path / "data/processed/train.csv"
    export.parent.mkdir(parents=True)
    export.write_bytes(training_csv(source_frame, ids))
    refs = {
        "dataset_sha256": source_hash,
        "split_manifest_sha256": sha256_file(day1 / "split_manifest.json"),
        "training_sha256": sha256_file(export),
    }
    write_json(day1 / "provenance.json", {"csv_sha256": source_hash})
    write_json(
        day1 / "training_partition.json", {**refs, "partition": "train", "row_ids": sorted(ids)}
    )
    config = CONFIG.replace("rows = 6000", "rows = 60").replace("positives = 203", "positives = 12")
    for key, digest in refs.items():
        config = config.replace(SETTINGS["data"][key], digest)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/day2.toml").write_text(config, encoding="utf-8")
    copyfile(ROOT / "uv.lock", tmp_path / "uv.lock")
    return tmp_path


@pytest.mark.parametrize(
    "key", ["dataset_sha256", "split_manifest_sha256", "training_sha256", "rows", "positives"]
)
def test_reject_wrong_day2_inputs(exported_root, key):
    config = tomllib.loads((exported_root / "configs/day2.toml").read_text())
    config["data"][key] = 0
    with pytest.raises(ValueError, match="approved"):
        verified_training(exported_root, config["data"])


def test_cli_reads_only_training_and_repeats_every_artifact(exported_root, monkeypatch):
    root = exported_root
    actual_open = Path.open
    actual_csv = pd.read_csv
    seen = []

    def guard_open(path, *args, **kwargs):
        if path.is_relative_to(root / "data"):
            assert path == root / "data/processed/train.csv"
        return actual_open(path, *args, **kwargs)

    def guard_csv(path, *args, **kwargs):
        assert path == root / "data/processed/train.csv"
        seen.append(path)
        return actual_csv(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard_open)
    monkeypatch.setattr(pd, "read_csv", guard_csv)
    assert main(["--root", str(root), "baselines"]) == 0
    output = root / "results/day2"
    execution = read_json(output / "execution.json")
    first_hash = sha256_file(output / "execution.json")
    run_baselines(root)
    assert len(seen) == 2
    assert sha256_file(output / "execution.json") == first_hash
    for name, digest in execution["artifacts_sha256"].items():
        assert sha256_file(output / name) == digest
    for path, digest in execution["inputs_sha256"].items():
        assert sha256_file(root / path) == digest
