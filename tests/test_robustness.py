"""Phase 06 split, scoring and isolation checks. Generated fixture rows are artificial."""

import builtins
import io
import runpy
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from test_ablations import artificial_design as artificial_design
from test_baselines import exported_root as exported_root

from ai4i_pm import robustness as study
from ai4i_pm.ablations import run_ablations
from ai4i_pm.cli import main
from ai4i_pm.robustness_figures import describe
from ai4i_pm.schema import PREDICTORS, TARGET
from ai4i_pm.splits import load_training
from ai4i_pm.utils import read_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def expected_members(target, seed):
    y = target.sort_index()
    return {
        str(i): y.index[held].tolist()
        for i, (_, held) in enumerate(
            StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(y, y), 1
        )
    }


@pytest.mark.parametrize("seed", [7, 123])
def test_stratified_membership_order_and_complements(seed):
    y = pd.Series([0, 0, 0, 0, 1] * 10, index=np.arange(1, 101, 2))
    members = expected_members(y, seed)
    cfg = {
        "folds": 5,
        "shuffle": True,
        "membership_sha256": {str(seed): study.digest(members)},
        "fit_rows": 40,
        "eval_rows": 10,
        "sorted_eval_positives": [2] * 5,
    }
    for target in (y, y.sample(frac=1, random_state=99)):
        actual, splits = study.alternative_folds(target, cfg, seed)
        assert actual == members
        for i, (fit, held) in enumerate(splits, 1):
            assert target.index[held].tolist() == members[str(i)]
            assert set(target.index[fit]) == set(y.index) - set(members[str(i)])
    cfg["membership_sha256"][str(seed)] = "changed"
    with pytest.raises(ValueError, match="preregistration"):
        study.alternative_folds(y, cfg, seed)


def test_expanding_windows_and_order():
    y = pd.Series([1, 0] * 10, index=np.arange(5, 45, 2))
    cuts = {"fit_ends": [8, 12, 16], "eval_ends": [12, 16, 20]}
    splits = study.ordered_splits(y, cuts)
    np.testing.assert_array_equal(np.concatenate([b for _, b in splits]), np.arange(8, 20))
    for (fit, held), end in zip(splits, cuts["fit_ends"], strict=True):
        np.testing.assert_array_equal(fit, np.arange(end))
        np.testing.assert_array_equal(held, np.arange(end, end + 4))
        assert y.index[fit].max() < y.index[held].min()
        counts = study.support(y, fit, held)
        assert counts["fit_positives"] == counts["fit_negatives"] == end // 2
        assert counts["prevalence_baseline"] == 0.5
    with pytest.raises(ValueError, match="sorted"):
        study.ordered_splits(y.iloc[::-1], cuts)
    with pytest.raises(ValueError, match="contiguous"):
        study.ordered_splits(y, {**cuts, "eval_ends": [13, 16, 20]})


@pytest.mark.parametrize("labels", [[0, 0, 0, 0], [1, 1, 1, 1], [1, 0, 0, 1]])
def test_ap_and_undefined_support(source_frame, monkeypatch, labels):
    x = source_frame.loc[:7, PREDICTORS]
    y = pd.Series([1, 0, 1, 0, *labels], index=x.index)
    scores = np.array([0.2, 0.7, 0.4, 0.9])

    class Ranking:
        classes_ = np.array([0, 1])

        def predict_proba(self, rows):
            return np.column_stack((1 - scores, scores))

    monkeypatch.setattr(study, "fit_fold", lambda *args: Ranking())
    fit, held = np.arange(4), np.arange(4, 8)
    ap, status, elapsed = study.evaluate("", {}, [], x, y, fit, held)
    assert elapsed >= 0
    assert study.support(y, fit, held)["prevalence_baseline"] == np.mean(labels)
    if any(labels):
        assert status == "defined"
        assert ap == pytest.approx(average_precision_score(labels, scores))
    else:
        assert ap is None and status == "no evaluation positives"
    y.iloc[fit] = 0
    assert study.evaluate("", {}, [], x, y, fit, held) == (None, "single-class fit", None)


def test_partial_block_aggregate_is_undefined():
    frame = pd.DataFrame(
        {"model": np.repeat(["M1", "M2", "M3", "M4"], 3), "ap": [0.1, 0.5, 0.9] * 4}
    )
    frame.loc[2, "ap"] = np.nan
    summary = describe(frame)
    assert summary["models"]["M1"] == dict(
        mean=None, sd=None, min=None, max=None, defined=2, planned=3
    )
    assert summary["rank"]["M1"] is None and summary["a4_minus_a0"]["HGB"] is None
    assert summary["models"]["M2"]["sd"] == pytest.approx(0.4)


def test_frozen_robustness_cli_and_isolated_replay(artificial_design, monkeypatch):
    root = artificial_design
    run_ablations(root)
    cfg = study.protocol(ROOT)
    target = load_training(root).set_index("UDI")[TARGET].sort_index()
    cfg["rows"], cfg["positives"] = len(target), int(target.sum())
    n = len(target) // 5
    cv = cfg["random_cv"]
    cv.update(fit_rows=4 * n, eval_rows=n)
    cfg["ordered"].update(fit_ends=[2 * n, 3 * n, 4 * n], eval_ends=[3 * n, 4 * n, 5 * n])
    expected = []
    for seed in cv["seeds"]:
        members = expected_members(target, seed)
        cv["membership_sha256"][str(seed)] = study.digest(members)
        cv["sorted_eval_positives"] = sorted(int(target.loc[v].sum()) for v in members.values())
        expected.extend(frozenset(set(target.index) - set(v)) for v in members.values())
    expected.extend(frozenset(target.index[:j]) for j in cfg["ordered"]["fit_ends"])
    cfg["inputs_sha256"] = {p: sha256_file(root / p) for p in cfg["inputs_sha256"]}
    (root / study.CONFIG).write_bytes((ROOT / study.CONFIG).read_bytes())
    monkeypatch.setattr(study, "protocol", lambda _: cfg)
    opened = set()

    def guard(original):
        def checked(name, *args, **kwargs):
            if isinstance(name, (str, Path)):
                path = Path(name).resolve()
                if path.is_relative_to(root / "data"):
                    assert path == root / "data/processed/train.csv"
                    opened.add(path)
            return original(name, *args, **kwargs)

        return checked

    monkeypatch.setattr(builtins, "open", guard(builtins.open))
    monkeypatch.setattr(io, "open", guard(io.open))
    fits = []
    original_fit = Pipeline.fit

    def tracked(model, x, y, **kwargs):
        assert list(x.columns) == list(PREDICTORS) and x.index.equals(y.index)
        fits.append(frozenset(x.index))
        result = original_fit(model, x, y, **kwargs)
        encoder = model["features"].named_transformers_["type"]
        assert set(encoder.categories_[0]) == set(x.Type)
        return result

    monkeypatch.setattr(Pipeline, "fit", tracked)
    assert main(["--root", str(root), "robustness"]) == 0
    assert Counter(fits) == Counter(expected * 4) and len(opened) == 1
    check = runpy.run_path(str(ROOT / "scripts/verify_06_robustness.py"))["verify"](root)
    assert check["max_absolute_ap_difference"] == 0 and check["new_fits_replayed"] == 52
    assert check["evaluation_rows_compared"] == 72
    folder = root / study.OUTPUT
    for filename, size in [
        ("random_split_fold_metrics.csv", 60),
        ("ordered_block_metrics.csv", 12),
    ]:
        table = pd.read_csv(folder / filename)
        assert len(table) == size
        for _, group in table.groupby(["seed", "fold_or_block"], dropna=False):
            assert set(group.model) == set(cfg["models"])
            assert group.eval_udis_sha256.nunique() == group.fit_udis_sha256.nunique() == 1
        assert (
            table.prevalence_baseline - table.eval_positives / table.eval_rows
        ).abs().max() < 1e-14
    assert read_json(folder / "execution.json")["new_fits"] == 52
    (folder / "ordered_block_metrics.csv").write_bytes(b"changed")
    with pytest.raises(ValueError, match="inventory"):
        study.run_robustness(root)


def test_protocol_tampering_rejected(tmp_path):
    path = tmp_path / study.CONFIG
    path.parent.mkdir()
    path.write_bytes((ROOT / study.CONFIG).read_bytes() + b"\n")
    with pytest.raises(ValueError, match="prespecified"):
        study.protocol(tmp_path)


def test_research_artifacts_match_preregistered_membership_and_reference():
    cfg = study.protocol(ROOT)
    folder = ROOT / study.OUTPUT
    record = study.check_inventory(folder)
    assert record["inputs_sha256"] == cfg["inputs_sha256"]
    original = read_json(ROOT / "results/02_baseline_evaluation/cv_manifest.json")[
        "evaluation_udis"
    ]
    ids = {i for fold in original.values() for i in fold}
    seeds = read_json(folder / "alternative_cv_manifest.json")["evaluation_udis_by_seed"]
    assert set(seeds) == {"7", "123"}
    for seed, groups in seeds.items():
        assert study.digest(groups) == cfg["random_cv"]["membership_sha256"][seed]
        assert sorted(i for fold in groups.values() for i in fold) == sorted(ids)
    table = pd.read_csv(folder / "random_split_fold_metrics.csv", float_precision="round_trip")
    ref = pd.read_csv(
        ROOT / "results/04_feature_ablation/ablation_fold_metrics.csv", float_precision="round_trip"
    ).set_index(["candidate_id", "condition", "fold"])
    for row in table.itertuples():
        memberships = original if row.seed == 42 else seeds[str(row.seed)]
        members = memberships[str(row.fold_or_block)]
        assert row.eval_udis_sha256 == study.digest(members)
        assert row.fit_udis_sha256 == study.digest(sorted(ids - set(members)))
        if row.seed == 42:
            assert row.ap == ref.loc[(row.candidate, row.condition, row.fold_or_block), "ap"]
    blocks = read_json(folder / "ordered_split_manifest.json")["blocks"]
    for entry, cut in zip(blocks, cfg["ordered"]["fit_ends"], strict=True):
        assert entry["fit_udis"] == sorted(ids)[:cut]
        assert entry["evaluation_udis"] == sorted(ids)[cut : cut + 1200]
