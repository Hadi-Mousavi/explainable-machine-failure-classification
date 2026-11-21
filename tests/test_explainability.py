"""Phase 05 regression tests use artificial observations exclusively."""

import builtins
import io
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from test_ablations import artificial_design as artificial_design
from test_baselines import exported_root as exported_root
from threadpoolctl import threadpool_limits

from ai4i_pm import explainability as study
from ai4i_pm.ablations import FOLDS, file_hashes, run_ablations
from ai4i_pm.feature_engineering import POWER, TEMPERATURE, engineer_features
from ai4i_pm.schema import PREDICTORS, PROHIBITED, TARGET
from ai4i_pm.shap_analysis import checked_attribution, diagnostic_cases, preflight, sample_ids
from ai4i_pm.utils import read_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]
CFG = study.load_protocol(ROOT)


@pytest.fixture(autouse=True)
def single_thread():
    with threadpool_limits(limits=1):
        yield


def test_sampling_order_cutoff_and_empty_categories():
    actual = pd.Series([1, 0, 1, 0, 1, 1], index=[8, 7, 5, 4, 3, 2])
    scores = np.array([0.8, 0.4, 0.5, 0.0, 0.2, 0.9])
    selected, counts = diagnostic_cases(actual, scores, 0.5, 2)
    assert selected == {"TP": [2, 5], "FP": [], "FN": [3], "TN": [4, 7]}
    assert counts == {"TP": 3, "FP": 0, "FN": 1, "TN": 2}
    ids = sample_ids(actual.index, 3, 5)
    np.testing.assert_array_equal(ids, sample_ids(actual.index[::-1], 3, 5))
    assert len(set(ids)) == 3 and set(ids) <= set(actual.index)
    np.testing.assert_array_equal(sample_ids(actual.index, 300, 5), sorted(actual.index))


def test_permutations_rescore_raw_columns_and_recompute_derived(source_frame):
    x, y = source_frame.loc[:, PREDICTORS], source_frame[TARGET]
    seen = []

    class Ranker:
        classes_ = np.array([0, 1])

        def predict_proba(self, raw):
            derived = engineer_features(raw, [*PREDICTORS, TEMPERATURE, POWER])
            seen.append(derived.copy())
            score = derived[POWER].to_numpy() / 50_000
            return np.column_stack((1 - score, score))

    baseline = study.score_ap(Ranker(), x, y)
    original = seen.pop()
    records = study.permutation_rows(Ranker(), "M4", 1, x, y, baseline, CFG["permutation"])
    for row, frame in zip(records, seen, strict=True):
        name = row["predictor"]
        np.testing.assert_array_equal(
            frame[name], np.random.default_rng(row["seed"]).permutation(x[name])
        )
        other = [col for col in PREDICTORS if col != name]
        pd.testing.assert_frame_equal(frame[other], x[other])
        np.testing.assert_allclose(
            frame[POWER], frame["Torque [Nm]"] * frame["Rotational speed [rpm]"] * 2 * np.pi / 60
        )
        np.testing.assert_allclose(
            frame[TEMPERATURE], frame["Process temperature [K]"] - frame["Air temperature [K]"]
        )
        ordered = y.iloc[np.argsort(-frame[POWER].to_numpy())].to_numpy()
        ap = np.mean((ordered.cumsum() / np.arange(1, len(y) + 1))[ordered == 1])
        assert row["permuted_ap"] == pytest.approx(ap)
        assert row["ap_drop"] == pytest.approx(baseline - ap)
    assert set(original.columns) == {*PREDICTORS, TEMPERATURE, POWER}
    assert len(records) == len({row["seed"] for row in records}) == 30
    assert records == study.permutation_rows(Ranker(), "M4", 1, x, y, baseline, CFG["permutation"])
    for name in (*PROHIBITED, TARGET):
        with pytest.raises(ValueError, match="exactly"):
            study.permutation_rows(
                Ranker(), "M4", 1, x.assign(**{name: 0}), y, baseline, CFG["permutation"]
            )


@pytest.mark.parametrize(
    ("candidate", "params", "scale", "width"),
    [
        ("hist_gradient_boosting-10", {"max_iter": 3, "random_state": 42}, "class-1 log-odds", 8),
        ("random_forest-11", {"n_estimators": 3, "random_state": 42}, "class-1 probability", 10),
    ],
)
def test_shap_alignment_scale_positive_axis_and_additivity(
    source_frame, candidate, params, scale, width
):
    pytest.importorskip("shap")
    x, y = source_frame.loc[:, PREDICTORS], source_frame[TARGET]
    columns = list(PREDICTORS) if width == 8 else [*PREDICTORS, TEMPERATURE, POWER]
    fit, held = np.arange(70), np.arange(70, 100)
    with pytest.raises(ValueError, match="Overlapping"):
        study.fit_fold(candidate, params, columns, x, y, fit, fit)
    model = study.fit_fold(candidate, params, columns, x, y, fit, held)
    rows = x.iloc[held]
    explainer, meta = preflight(model, rows, CFG["shap"])
    batch = checked_attribution(model, explainer, rows, 1e-6)
    assert meta["scale"] == scale and meta["positive_index"] == 1
    assert batch.values.shape == (30, width)
    np.testing.assert_allclose(batch.base + batch.values.sum(axis=1), batch.output, atol=1e-10)
    np.testing.assert_allclose(
        batch.output if width == 10 else 1 / (1 + np.exp(-batch.output)),
        model.predict_proba(rows)[:, 1],
        atol=1e-12,
    )

    def swapped(matrix):
        exp = explainer(matrix)
        if width == 10:
            exp.values = exp.values[:, :, ::-1]
        else:
            exp.feature_names = list(reversed(exp.feature_names))
        return exp

    with pytest.raises(ValueError, match="additivity|coordinates"):
        checked_attribution(model, swapped, rows, 1e-6)


def test_training_only_frozen_fold_replay(artificial_design, monkeypatch):
    """Observe real fit IDs and reject every observation-file open except the fixture train CSV."""
    root = artificial_design
    run_ablations(root)
    requirements = "configs/05_shap_requirements.txt"
    copyfile(ROOT / requirements, root / requirements)
    text = (ROOT / study.CONFIG).read_text()
    for path, digest in CFG["inputs_sha256"].items():
        text = text.replace(digest, sha256_file(root / path))
    protocol = root / study.CONFIG
    protocol.write_bytes(text.encode())
    monkeypatch.setattr(study, "PROTOCOL_SHA256", sha256_file(protocol))
    opened = set()

    def guard(original):
        def checked(file, *args, **kwargs):
            if isinstance(file, (str, Path)):
                path = Path(file).resolve()
                if path.is_relative_to(root / "data"):
                    assert path == root / "data/processed/train.csv"
                    opened.add(path)
            return original(file, *args, **kwargs)

        return checked

    monkeypatch.setattr(builtins, "open", guard(builtins.open))
    monkeypatch.setattr(io, "open", guard(io.open))
    expected = read_json(root / FOLDS)["evaluation_udis"]
    ids = set().union(*expected.values())
    fitted = []
    original_fit = Pipeline.fit

    def audit_fit(pipeline, x, y, **kwargs):
        assert tuple(x.columns) == PREDICTORS and x.index.equals(y.index)
        fitted.append(frozenset(x.index))
        return original_fit(pipeline, x, y, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", audit_fit)
    record = study.run_explanations(root)
    for members in expected.values():
        assert fitted.count(ids - set(members)) == 4
    assert len(fitted) == 20 and len(opened) == 1
    folder = root / study.OUTPUT
    rows = pd.read_csv(folder / "permutation_fold_results.csv")
    assert rows.columns.tolist() == [
        "model",
        "fold",
        "predictor",
        "repeat",
        "seed",
        "original_ap",
        "permuted_ap",
        "ap_drop",
    ]
    assert len(rows) == record["permutation_measurements"] == 600
    assert not rows.duplicated(["model", "fold", "predictor", "repeat"]).any()
    meta = read_json(folder / "shap_metadata.json")
    assert set(meta["global_udis"]) <= set(expected["1"])
    hashes = file_hashes(folder)
    assert record["artifacts_sha256"] == {k: v for k, v in hashes.items() if k != "execution.json"}
    study.run_explanations(root)
    assert hashes == file_hashes(folder)

    # Compatibility errors retain the exact diagnostic and do not suppress permutation evidence.
    def unsupported(*args):
        raise ValueError("unsupported fixture model")

    monkeypatch.setattr(study, "preflight", unsupported)
    stage = root / "blocked"
    study.run_study(root, stage, study.load_protocol(root))
    assert (
        read_json(stage / "shap_metadata.json")["models"]["M4"]["error"]
        == "unsupported fixture model"
    )
    assert (stage / "permutation_summary.csv").is_file() and not (
        stage / "shap_values.npz"
    ).exists()
