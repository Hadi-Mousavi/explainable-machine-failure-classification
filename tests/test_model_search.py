"""Offline Day 3 checks on the artificial training fixture and arbitrary fold assignments."""

import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score
from test_baselines import exported_root as exported_root
from threadpoolctl import threadpool_limits

from ai4i_pm.baselines import run_baselines
from ai4i_pm.cli import main
from ai4i_pm.model_search import (
    FAMILIES,
    candidate_plan,
    evaluate_family,
    frozen_splits,
    pipelines,
    ranked_candidates,
    score_ap,
)
from ai4i_pm.schema import IDENTIFIER, PREDICTORS, PROHIBITED, TARGET
from ai4i_pm.utils import read_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]
CONFIG = tomllib.loads((ROOT / "configs/day3.toml").read_text())
BASE = tomllib.loads((ROOT / "configs/day2.toml").read_text())


@pytest.fixture
def design(source_frame):
    data = source_frame.set_index(IDENTIFIER).sample(frac=1, random_state=9)
    # Deliberately use a saved assignment unrelated to sklearn's stratified splitter.
    ids = sorted(data.index)
    memberships = {str(i + 1): ids[i * 20 : i * 20 + 20][::-1] for i in range(5)}
    return (
        data.loc[:, PREDICTORS].copy(),
        data[TARGET],
        {"primary_partition": "train", "evaluation_udis": memberships},
    )


@pytest.mark.parametrize("family", FAMILIES[2:])
@pytest.mark.parametrize("extra", [*PROHIBITED, TARGET])
def test_tree_allowlist(source_frame, family, extra):
    template = pipelines(BASE["models"], 42)[family]
    features = source_frame.loc[:, PREDICTORS].copy()
    features[extra] = source_frame[extra]
    with pytest.raises(ValueError, match="exactly"):
        template.fit(features, source_frame[TARGET])


def test_reproducible_paired_design():
    a = candidate_plan(CONFIG)
    assert a == candidate_plan(CONFIG)
    assert a != candidate_plan({**CONFIG, "seed": 7})
    for family in FAMILIES[2:]:
        assert len(a[family]) == 12
        assert len({json.dumps(p, sort_keys=True) for p in a[family]}) == 12
        for raw, weighted in zip(a[family][::2], a[family][1::2], strict=True):
            assert raw["class_weight"] is None
            assert weighted == {**raw, "class_weight": "balanced"}
            # The locked estimator validates all parameters before fitting.
            for p in (raw, weighted):
                clf = pipelines(BASE["models"], 42)[family]["classifier"].set_params(**p)
                clf._validate_params()
    assert a["logistic_regression"] == [{"class_weight": None}, {"class_weight": "balanced"}]


@pytest.mark.parametrize("fault", ["duplicate", "foreign", "missing", "partition"])
def test_fold_manifest_rejects_invalid_membership(design, fault):
    _, y, manifest = design
    ids = manifest["evaluation_udis"]["1"]
    if fault == "duplicate":
        ids[0] = ids[1]
    elif fault == "foreign":
        ids[0] = -1
    elif fault == "missing":
        ids.pop()
    else:
        manifest["primary_partition"] = "validation"
    with pytest.raises(ValueError):
        frozen_splits(y, manifest)


@pytest.mark.parametrize("family", FAMILIES[2:])
def test_each_candidate_uses_exact_folds_and_local_encoding(design, family, monkeypatch):
    x, y, manifest = design
    splits, records = frozen_splits(y, manifest)
    held = manifest["evaluation_udis"]["1"]
    x.loc[held, "Type"] = "only-in-fold-one"
    original = ColumnTransformer.fit_transform
    seen_fit, seen_eval = [], []

    def inspect_fit(self, X, y=None, **params):
        seen_fit.append(tuple(X.index))
        assert tuple(X.columns) == PREDICTORS
        matrix = original(self, X, y, **params)
        assert isinstance(matrix, np.ndarray)
        np.testing.assert_array_equal(matrix[:, -5:], X.iloc[:, 1:])
        assert set(self.named_transformers_["type"].categories_[0]) == set(X.Type)
        return matrix

    def inspect_scores(model, features, actual):
        seen_eval.append(tuple(features.index))
        return score_ap(model, features, actual)

    monkeypatch.setattr(ColumnTransformer, "fit_transform", inspect_fit)
    monkeypatch.setattr("ai4i_pm.model_search.score_ap", inspect_scores)
    light = {"n_estimators": 2} if family == "random_forest" else {"max_iter": 2}
    plan = [{**light, "class_weight": w} for w in [None, "balanced"]]
    with threadpool_limits(limits=1):
        rows, values, _ = evaluate_family(
            family, pipelines(BASE["models"], 42)[family], plan, x, y, splits, records, 42
        )
    assert [r["failed_folds"] for r in rows] == [0, 0]
    assert len(values) == 10
    assert seen_fit == [tuple(x.index[fit]) for fit, _ in splits] * 2
    assert seen_eval == [tuple(manifest["evaluation_udis"][str(k)]) for k in range(1, 6)] * 2


def test_positive_label_and_average_precision():
    class Reversed:
        classes_ = np.array([1, 0])

        def predict_proba(self, features):
            return np.column_stack([features, 1 - features])

    scores = np.array([0.9, 0.7, 0.6, 0.1])
    actual = np.array([0, 1, 1, 0])
    expected = (1 / 2 + 2 / 3) / 2
    assert score_ap(Reversed(), scores, actual) == pytest.approx(expected)
    assert average_precision_score(actual, scores) == pytest.approx(expected)


@pytest.mark.parametrize("valid", [False, True])
def test_search_retains_failed_candidates(design, valid):
    x, y, manifest = design
    splits, records = frozen_splits(y, manifest)
    candidates = [{"n_estimators": 0}] + ([{"n_estimators": 2}] if valid else [])
    rows, folds, history = evaluate_family(
        "random_forest",
        pipelines(BASE["models"], 42)["random_forest"],
        candidates,
        x,
        y,
        splits,
        records,
        42,
    )
    assert rows[0]["failed_folds"] == 5 and np.isnan(rows[0]["mean_ap"])
    assert all("n_estimators" in r["error"] for r in folds[:5])
    assert history["messages"]
    assert ranked_candidates(rows) == rows[1:]


def test_exact_ties_and_small_differences():
    def row(family, ident, params, score=0.5):
        return {
            "family": family,
            "candidate_id": ident,
            "classifier_parameters": json.dumps(params),
            "mean_ap": score,
            "failed_folds": 0,
        }

    rf = {"max_depth": 8, "min_samples_leaf": 3, "n_estimators": 150}
    hgb = {"max_leaf_nodes": 15, "max_iter": 100, "l2_regularization": 1.0, "learning_rate": 0.05}
    rows = [
        row("random_forest", "a", rf),
        row("random_forest", "b", {**rf, "max_depth": None}),
        row("random_forest", "c", {**rf, "class_weight": "balanced"}),
        row("logistic_regression", "d", {"class_weight": None}),
        row("hist_gradient_boosting", "e", hgb, 0.5000000000001),
        row("hist_gradient_boosting", "f", {**hgb, "max_iter": 200}, 0.5000000000001),
    ]
    assert [r["candidate_id"] for r in ranked_candidates(rows[::-1])] == list("efdacb")


def test_cli_uses_only_training_export_and_serializes_complete_results(exported_root, monkeypatch):
    root = exported_root
    run_baselines(root)
    cfg = (ROOT / "configs/day3.toml").read_text()
    for key, path in [
        ("day2_config_sha256", "configs/day2.toml"),
        ("cv_manifest_sha256", "results/day2/cv_manifest.json"),
        ("day2_metrics_sha256", "results/day2/fold_metrics.csv"),
    ]:
        cfg = cfg.replace(CONFIG[key], sha256_file(root / path))
    cfg = cfg.replace("structures_per_family = 6", "structures_per_family = 1")
    cfg = cfg.replace("[150, 300]", "[2]").replace("[100, 200]", "[2]")
    (root / "configs/day3.toml").write_text(cfg, encoding="utf-8")
    csv_reads = []
    original_csv, original_open = pd.read_csv, Path.open

    def guarded_open(path, *args, **kwargs):
        if path.is_relative_to(root / "data"):
            assert path == root / "data/processed/train.csv"
        return original_open(path, *args, **kwargs)

    def guarded_csv(path, *args, **kwargs):
        csv_reads.append(path.relative_to(root).as_posix())
        assert csv_reads[-1] in {"data/processed/train.csv", "results/day2/fold_metrics.csv"}
        return original_csv(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(pd, "read_csv", guarded_csv)
    assert main(["--root", str(root), "compare-models"]) == 0
    assert csv_reads == ["data/processed/train.csv", "results/day2/fold_metrics.csv"]
    folder = root / "results/day3"
    record = read_json(folder / "execution.json")
    assert record["candidates"] == 7 and record["planned_fits"] == 35
    assert record["failed_folds"] == 0
    for name, digest in record["artifacts_sha256"].items():
        assert sha256_file(folder / name) == digest
    trials = original_csv(folder / "candidate_results.csv")
    assert len(original_csv(folder / "candidate_fold_metrics.csv")) == 35
    assert len(original_csv(folder / "class_weight_analysis.csv")) == 3
    assert all(isinstance(json.loads(s), dict) for s in trials.classifier_parameters)
    assert read_json(folder / "selected_configuration.json")["candidate_id"] in set(
        trials.candidate_id
    )
    assert read_json(folder / "model_comparison.json")["rows"] == 60
