"""Reproduce the frozen Phase 05 study on evaluation folds within training."""

import platform
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from ai4i_pm import ablations
from ai4i_pm.baselines import verified_training
from ai4i_pm.explainability_figures import permutation_figure, shap_figures, write_summary
from ai4i_pm.feature_engineering import ablation_pipeline
from ai4i_pm.model_search import frozen_splits, score_ap
from ai4i_pm.models import check_model_inputs
from ai4i_pm.schema import IDENTIFIER, PREDICTORS, TARGET, predictor_matrix
from ai4i_pm.shap_analysis import explain_sample, preflight, sample_ids
from ai4i_pm.utils import environment_versions, read_json, sha256_file, write_json

CONFIG = "configs/05_explainability.toml"
PROTOCOL_SHA256 = "5f1b86d0bc986b8054f48d7909adbab90e6b9bec90671b54c2636346bbff0704"
OUTPUT = "results/05_explainability"
ABLATION = "results/04_feature_ablation"


def load_protocol(root: Path) -> dict:
    if sha256_file(root / CONFIG) != PROTOCOL_SHA256:
        raise ValueError("Phase 05 protocol differs from the frozen design")
    return tomllib.loads((root / CONFIG).read_text(encoding="utf-8"))


def load_inputs(root: Path, cfg: dict) -> tuple:
    for path, expected in cfg["inputs_sha256"].items():
        if sha256_file(root / path) != expected:
            raise ValueError(f"Phase 05 input checksum mismatch: {path}")
    prior = tomllib.loads((root / "configs/02_baseline_evaluation.toml").read_text())
    manifest = read_json(root / ablations.FOLDS)
    if manifest["inputs"] != prior["data"] or manifest["settings"] != prior["cv"]:
        raise ValueError("Frozen CV provenance mismatch")
    data = (
        verified_training(root, prior["data"])
        .set_index(IDENTIFIER, verify_integrity=True)
        .sort_index()
    )
    features, target = predictor_matrix(data), data[TARGET]
    splits, records = frozen_splits(target, manifest)
    design = read_json(root / ABLATION / "ablation_config.json")
    specs = ablations.fixed_specifications(root)
    if specs != design["classifier_parameters"] or records != design["folds"]:
        raise ValueError("Specifications or memberships differ from Phase 04")
    if design["configuration_sha256"] != cfg["inputs_sha256"][ablations.CONFIG]:
        raise ValueError("Phase 04 feature protocol digest mismatch")
    reference = pd.read_csv(
        root / ABLATION / "ablation_fold_metrics.csv", float_precision="round_trip"
    ).set_index(["candidate_id", "condition", "fold"], verify_integrity=True)
    return features, target, splits, records, specs, design["protocol"]["features"], reference


def fit_fold(
    candidate: str,
    params: dict,
    columns: list,
    features: pd.DataFrame,
    target: pd.Series,
    fit: np.ndarray,
    evaluate: np.ndarray,
) -> Pipeline:
    if np.intersect1d(fit, evaluate).size or not features.index.equals(target.index):
        raise ValueError("Overlapping fit/evaluation membership or misaligned labels")
    model = ablation_pipeline(ablations.ESTIMATORS[candidate](**params), columns)
    return model.fit(features.iloc[fit], target.iloc[fit])


def permutation_rows(
    model: Pipeline,
    key: str,
    fold: int,
    held: pd.DataFrame,
    target: pd.Series,
    baseline: float,
    settings: dict,
) -> list[dict]:
    check_model_inputs(held)
    rows = []
    for j, name in enumerate(PREDICTORS):
        shuffled = held.copy()
        for repeat in range(1, settings["repeats"] + 1):
            seed = settings["seed"] + 100 * fold + 10 * j + repeat
            shuffled[name] = np.random.default_rng(seed).permutation(held[name].to_numpy())
            ap = score_ap(model, shuffled, target)
            rows.append(
                {
                    "model": key,
                    "fold": fold,
                    "predictor": name,
                    "repeat": repeat,
                    "seed": seed,
                    "original_ap": baseline,
                    "permuted_ap": ap,
                    "ap_drop": baseline - ap,
                }
            )
    return rows


def summarize_permutations(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = rows.groupby(["model", "fold", "predictor"], as_index=False).agg(
        mean_drop=("ap_drop", "mean"), repeat_sd=("ap_drop", "std")
    )
    overall = folds.groupby(["model", "predictor"], as_index=False).agg(
        mean_drop=("mean_drop", "mean"),
        fold_sd=("mean_drop", "std"),
        fold_min=("mean_drop", "min"),
        fold_max=("mean_drop", "max"),
        mean_repeat_sd=("repeat_sd", "mean"),
    )
    return folds, overall.sort_values(["model", "mean_drop"], ascending=[True, False])


def blocked(error: Exception) -> dict:
    return {"status": "BLOCKED", "error_type": type(error).__name__, "error": str(error)}


def write_csv(folder: Path, name: str, table: pd.DataFrame) -> None:
    table.to_csv(folder / f"{name}.csv", index=False, lineterminator="\n")


def run_study(root: Path, stage: Path, cfg: dict) -> dict:
    features, target, splits, records, specs, columns, reference = load_inputs(root, cfg)
    write_json(
        stage / "explanation_config.json",
        {
            "protocol": cfg,
            "protocol_sha256": PROTOCOL_SHA256,
            "classifier_parameters": specs,
            "features": {k: columns[k] for k in ("A0", "A4")},
            "folds": records,
        },
    )
    settings = cfg["shap"]
    fit, evaluate = splits[settings["fold"] - 1]
    held, actual = features.iloc[evaluate], target.iloc[evaluate]
    ids = sample_ids(held.index, settings["sample_size"], settings["sample_seed"])
    models, explainers, metadata = {}, {}, {}
    for key in settings["models"]:
        candidate, condition = cfg["models"][key]
        model = fit_fold(
            candidate, specs[candidate], columns[condition], features, target, fit, evaluate
        )
        models[key] = model
        try:
            explainers[key], metadata[key] = preflight(model, held, settings)
        except Exception as exc:
            metadata[key] = blocked(exc)

    rows, ap_errors = [], []
    for key, (candidate, condition) in cfg["models"].items():
        for (fit, evaluate), record in zip(splits, records, strict=True):
            fold = record["fold"]
            if fold == settings["fold"] and key in models:
                model = models[key]
            else:
                model = fit_fold(
                    candidate, specs[candidate], columns[condition], features, target, fit, evaluate
                )
            x_eval, y_eval = features.iloc[evaluate], target.iloc[evaluate]
            ap = score_ap(model, x_eval, y_eval)
            error = abs(ap - reference.loc[(candidate, condition, fold), "ap"])
            if not np.isfinite(error) or error > cfg["ap_atol"]:
                raise ValueError(
                    f"{key} fold {fold} AP differs from the Phase 04 reference by {error}"
                )
            ap_errors.append(error)
            rows.extend(permutation_rows(model, key, fold, x_eval, y_eval, ap, cfg["permutation"]))
    measurements = pd.DataFrame(rows)
    folds, overall = summarize_permutations(measurements)
    write_csv(stage, "permutation_fold_results", measurements)
    write_csv(stage, "permutation_fold_summary", folds)
    write_csv(stage, "permutation_summary", overall)
    figures = stage / "figures"
    figures.mkdir()
    permutation_figure(figures, folds, overall)
    arrays, rankings, cases = {}, [], []
    for key, explainer in explainers.items():
        try:
            meta, batch, ranking, local = explain_sample(
                key, models[key], explainer, held, actual, ids, settings, cfg["models"][key]
            )
        except Exception as exc:
            metadata[key].update(blocked(exc))
            continue
        metadata[key].update(meta, status="PASS")
        arrays.update({f"{key}_{field}": value for field, value in batch.items()})
        rankings.append(ranking)
        cases.append(local)
        shap_figures(figures, key, batch, ranking, local)
    if arrays:
        np.savez_compressed(stage / "shap_values.npz", **arrays)
        write_csv(stage, "shap_summary", pd.concat(rankings, ignore_index=True))
        write_csv(stage, "local_explanations", pd.concat(cases, ignore_index=True))
    write_json(
        stage / "shap_metadata.json",
        {
            "fold": settings["fold"],
            "method": settings["method"],
            "background": settings["background"],
            "background_udis": [],
            "global_udis": ids.tolist(),
            "sample_seed": settings["sample_seed"],
            "models": metadata,
        },
    )
    write_summary(stage, overall, metadata, rankings, cases)
    try:
        shap_version = version("shap")
    except PackageNotFoundError:
        shap_version = None
    execution = {
        "command": "uv run --locked ai4i-pm explain",
        "partition": "train",
        "protocol_sha256": PROTOCOL_SHA256,
        "inputs_sha256": cfg["inputs_sha256"],
        "fold_fits": len(ap_errors),
        "permutation_measurements": len(rows),
        "max_phase04_ap_error": float(max(ap_errors)),
        "shap_status": {key: meta["status"] for key, meta in metadata.items()},
        "environment": {
            **environment_versions(),
            "shap": shap_version,
            **{name: version(name) for name in ("scipy", "matplotlib", "threadpoolctl", "joblib")},
        },
        "platform": platform.platform(),
        "artifacts_sha256": ablations.file_hashes(stage),
    }
    write_json(stage / "execution.json", execution)
    return execution


def run_explanations(root: Path) -> dict:
    cfg = load_protocol(root)
    output = root / OUTPUT
    with (
        TemporaryDirectory(dir=output.parent) as directory,
        threadpool_limits(limits=cfg["native_threads"]),
    ):
        stage = Path(directory) / "phase05"
        execution = run_study(root, stage, cfg)
        if output.exists():
            if ablations.file_hashes(stage) != ablations.file_hashes(output):
                raise ValueError(
                    "Existing Phase 05 output differs. Replay in an isolated directory."
                )
        else:
            stage.rename(output)
    return execution
