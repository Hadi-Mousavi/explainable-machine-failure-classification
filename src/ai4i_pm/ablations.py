"""Run the frozen Day 4 experiment using approved training data and metadata."""

import json
import platform
import tomllib
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from threadpoolctl import threadpool_limits

from ai4i_pm.ablation_report import paired_summary, save_figures, write_markdown
from ai4i_pm.baselines import verified_training
from ai4i_pm.feature_engineering import ablation_pipeline
from ai4i_pm.model_search import frozen_splits, score_ap
from ai4i_pm.schema import IDENTIFIER, TARGET, predictor_matrix
from ai4i_pm.utils import (
    environment_versions,
    json_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
    write_json,
)

CONFIG = "configs/day4.toml"
FOLDS = "results/day2/cv_manifest.json"
SELECTED = "results/day3/selected_configuration.json"
CANDIDATES = "results/day3/candidate_results.csv"
REFERENCE = "results/day3/candidate_fold_metrics.csv"
INPUTS = (
    "configs/day2.toml",
    "uv.lock",
    "data/processed/train.csv",
    "results/day1/provenance.json",
    "results/day1/split_manifest.json",
    "results/day1/training_partition.json",
    FOLDS,
    SELECTED,
    CANDIDATES,
    REFERENCE,
)
ESTIMATORS = {
    "hist_gradient_boosting-10": HistGradientBoostingClassifier,
    "random_forest-11": RandomForestClassifier,
}


def load_design(root):
    config_bytes = (root / CONFIG).read_bytes()
    cfg = tomllib.loads(config_bytes.decode())
    hashes = {name: sha256_file(root / name) for name in INPUTS}
    if cfg["inputs_sha256"] != hashes:
        raise ValueError("An input differs from the frozen Day 4 inventory")
    if (
        list(cfg["features"]) != [f"A{i}" for i in range(6)]
        or (cfg["primary_model"], cfg["secondary_model"]) != tuple(ESTIMATORS)
        or cfg["reference"] != "A0"
        or cfg["positive_class"] != 1
        or cfg["metric"] != "average_precision_score"
        or cfg["sd_ddof"] != 1
    ):
        raise ValueError("Unexpected Day 4 experimental design")
    hashes[CONFIG] = sha256_bytes(config_bytes)
    return cfg, hashes


def fixed_specifications(root):
    selected = read_json(root / SELECTED)
    table = pd.read_csv(root / CANDIDATES, float_precision="round_trip").set_index(
        "candidate_id", verify_integrity=True
    )
    specs = {}
    for candidate, constructor in ESTIMATORS.items():
        row = table.loc[candidate]
        params = json.loads(row.classifier_parameters)
        if row.failed_folds != 0 or set(constructor().get_params(deep=False)) != set(params):
            raise ValueError(f"Incomplete Day 3 specification for {candidate}")
        specs[candidate] = params
    primary = "hist_gradient_boosting-10"
    if (
        selected["candidate_id"] != primary
        or selected["status"] != "provisional"
        or selected["classifier_parameters"] != specs[primary]
    ):
        raise ValueError("Provisional selection differs from the saved candidate specification")
    return specs


def evaluate_condition(condition, candidate, pipeline, features, target, splits, records):
    rows = []
    for (fit, evaluate), record in zip(splits, records, strict=True):
        held = features.iloc[evaluate]
        digest = sha256_bytes(json_bytes(held.index.tolist()))
        if digest != record["eval_udis_sha256"]:
            raise ValueError("Scoring UDIs differ from the frozen evaluation membership")
        try:
            model = clone(pipeline).fit(features.iloc[fit], target.iloc[fit])
            ap = score_ap(model, held, target.iloc[evaluate])
        except Exception as exc:
            # Some estimator exceptions include row values. Keep diagnostics at fold scope.
            raise ValueError(
                f"{condition} {candidate} fold {record['fold']} failed ({type(exc).__name__})"
            ) from None
        rows.append({"condition": condition, "candidate_id": candidate, **record, "ap": ap})
    return rows


def check_reference(rows, saved, atol):
    candidate = rows[0]["candidate_id"]
    ref = saved[saved.candidate_id == candidate].sort_values("fold")
    current = pd.DataFrame(rows).sort_values("fold")
    for col in (
        "fold",
        "fit_rows",
        "eval_rows",
        "fit_positives",
        "eval_positives",
        "eval_udis_sha256",
    ):
        if not np.array_equal(current[col], ref[col]):
            raise ValueError(f"Day 3 reference membership differs for {candidate}")
    if not np.allclose(current.ap, ref.ap, rtol=0, atol=atol):
        raise ValueError(f"A0 AP differs from Day 3 for {candidate}")
    return float(np.max(np.abs(current.ap.to_numpy() - ref.ap.to_numpy())))


def file_hashes(folder):
    return {
        p.relative_to(folder).as_posix(): sha256_file(p)
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def run_ablations(root: Path):
    cfg, hashes = load_design(root)
    prior = tomllib.loads((root / "configs/day2.toml").read_text(encoding="utf-8"))
    manifest = read_json(root / FOLDS)
    if manifest["inputs"] != prior["data"] or manifest["settings"] != prior["cv"]:
        raise ValueError("Fold manifest differs from frozen training provenance")
    data = (
        verified_training(root, prior["data"])
        .set_index(IDENTIFIER, verify_integrity=True)
        .sort_index()
    )
    features, target = predictor_matrix(data), data[TARGET]
    splits, records = frozen_splits(target, manifest)
    if any(record["eval_rows"] * 5 != len(data) for record in records):
        raise ValueError("Expected five equal-sized evaluation folds")
    specs = fixed_specifications(root)
    reference = pd.read_csv(root / REFERENCE, float_precision="round_trip")
    design = {
        "protocol": cfg,
        "configuration_sha256": hashes[CONFIG],
        "classifier_parameters": specs,
        "training": prior["data"],
        "folds": records,
    }
    output = root / "results/day4"
    with TemporaryDirectory(dir=output.parent) as directory:
        stage = Path(directory) / "day4"
        write_json(stage / "ablation_config.json", design)
        measurements, deviations = [], {}
        with threadpool_limits(limits=cfg["native_threads"]):
            for condition, columns in cfg["features"].items():
                for candidate, params in specs.items():
                    pipe = ablation_pipeline(ESTIMATORS[candidate](**params), columns)
                    rows = evaluate_condition(
                        condition, candidate, pipe, features, target, splits, records
                    )
                    if condition == cfg["reference"]:
                        deviations[candidate] = check_reference(
                            rows, reference, cfg["a0_absolute_tolerance"]
                        )
                    measurements.extend(rows)
        metrics = pd.DataFrame(measurements)
        summary, pairs = paired_summary(metrics, cfg["reference"], cfg["sd_ddof"])
        table = pd.DataFrame(summary)
        for name, frame in [
            ("ablation_fold_metrics", metrics),
            ("ablation_summary", table),
            ("paired_differences", pairs),
        ]:
            frame.to_csv(stage / f"{name}.csv", index=False, lineterminator="\n")
        write_json(
            stage / "ablation_summary.json",
            {"partition": "train", "reference": "A0", "folds": 5, "models_and_conditions": summary},
        )
        write_markdown(stage / "ablation_summary.md", summary)
        save_figures(stage / "figures", pairs, table)
        execution = {
            "command": "uv run --locked ai4i-pm ablations",
            "partition": "train",
            "fold_evaluations": len(metrics),
            "a0_max_absolute_error": deviations,
            "platform": platform.platform(),
            "inputs_sha256": hashes,
            "environment": {
                **environment_versions(),
                **{
                    name: version(name)
                    for name in ("matplotlib", "scipy", "joblib", "threadpoolctl")
                },
            },
            "artifacts_sha256": file_hashes(stage),
        }
        write_json(stage / "execution.json", execution)
        if output.exists():
            if file_hashes(output) != file_hashes(stage):
                raise ValueError("Existing Day 4 outputs differ. Use an isolated root for replay.")
        else:
            stage.rename(output)
    return execution
