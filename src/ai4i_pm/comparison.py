"""Run the training-only comparison and publish a checksum-linked research record."""

import json
import platform
import tomllib
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ai4i_pm.baselines import verified_training
from ai4i_pm.comparison_figures import LABELS, save_figures
from ai4i_pm.model_search import (
    FAMILIES,
    candidate_plan,
    evaluate_family,
    frozen_splits,
    pipelines,
    ranked_candidates,
    weight_pairs,
)
from ai4i_pm.schema import IDENTIFIER, PREDICTORS, TARGET, predictor_matrix
from ai4i_pm.utils import environment_versions, read_json, sha256_file, write_json

INPUTS = (
    "configs/day2.toml",
    "configs/day3.toml",
    "uv.lock",
    "data/processed/train.csv",
    "results/day1/split_manifest.json",
    "results/day1/training_partition.json",
    "results/day1/provenance.json",
    "results/day2/cv_manifest.json",
    "results/day2/fold_metrics.csv",
)


def verify_references(root, fold_table):
    original = pd.read_csv(root / "results/day2/fold_metrics.csv", float_precision="round_trip")
    for family in FAMILIES[:2]:
        frozen = original[original.model == family].sort_values("fold")
        current = fold_table[fold_table.candidate_id == f"{family}-01"].sort_values("fold")
        if not np.array_equal(frozen.fold, current.fold) or not np.allclose(
            frozen.average_precision, current.ap, rtol=0, atol=1e-12
        ):
            raise ValueError(f"Day 3 reference replay disagrees with Day 2: {family}")


def summary_records(ranked):
    groups = []
    for family in FAMILIES:
        eligible = [r for r in ranked if r["family"] == family]
        for r in eligible if family == "logistic_regression" else eligible[:1]:
            label = LABELS[family]
            if family == "logistic_regression":
                label += " (balanced)" if r["candidate_id"].endswith("02") else " (unweighted)"
            label += " · " + r["evaluation"]
            groups.append(
                {
                    k: r[k]
                    for k in (
                        "candidate_id",
                        "family",
                        "mean_ap",
                        "sample_sd",
                        "evaluation",
                        "settings",
                    )
                }
                | {"label": label}
            )
    return groups


def write_report(path, rows, metrics, selected, pairs):
    lines = [
        "# Five-fold training model comparison",
        "",
        f"Highest mean fold AP provisionally selects **{selected['candidate_id']}**.",
        "",
        "Fixed configurations and searched configurations share the Day 2 fold memberships. "
        "Selection on these folds makes the winning score optimistic. Sample SD describes "
        "fold variation. Candidate IDs map to complete settings and timing in "
        "[candidate_results.csv](candidate_results.csv).",
        "",
        "| Candidate | Evaluation | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 "
        "| Mean AP | Sample SD |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        scores = metrics[metrics.candidate_id == r["candidate_id"]].ap.tolist()
        values = [*scores, r["mean_ap"], r["sample_sd"]]
        cells = [f"{v:.6f}" if np.isfinite(v) else "failed" for v in values]
        lines.append("| " + " | ".join([r["candidate_id"], r["evaluation"], *cells]) + " |")
    lines += [
        "",
        "## Matched weight contrasts",
        "",
        "Each difference is balanced minus unweighted mean fold AP.",
        "",
        "| Family | Unweighted ID | Balanced ID | Difference |",
        "| --- | --- | --- | ---: |",
    ]
    for r in pairs.itertuples():
        lines.append(
            f"| {LABELS[r.family]} | {r.unweighted_id} | {r.balanced_id} | {r.delta_ap:+.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def run_comparison(root: Path):
    start = perf_counter()
    inputs = {name: sha256_file(root / name) for name in INPUTS}
    cfg = tomllib.loads((root / "configs/day3.toml").read_text(encoding="utf-8"))
    for name, key in (
        ("configs/day2.toml", "day2_config_sha256"),
        ("results/day2/cv_manifest.json", "cv_manifest_sha256"),
        ("results/day2/fold_metrics.csv", "day2_metrics_sha256"),
    ):
        if inputs[name] != cfg[key]:
            raise ValueError(f"Approved Day 2 input digest mismatch: {name}")
    prior = tomllib.loads((root / "configs/day2.toml").read_text(encoding="utf-8"))
    data = (
        verified_training(root, prior["data"])
        .set_index(IDENTIFIER, verify_integrity=True)
        .sort_index()
    )
    manifest = read_json(root / "results/day2/cv_manifest.json")
    if manifest["inputs"] != prior["data"] or manifest["settings"] != prior["cv"]:
        raise ValueError("Fold manifest has inconsistent training provenance")
    target = data[TARGET]
    features = predictor_matrix(data)
    splits, folds = frozen_splits(target, manifest)
    plan = candidate_plan(cfg)
    models = pipelines(prior["models"], cfg["seed"])
    rows, measurements, history = [], [], {}
    with threadpool_limits(limits=1):
        for family in FAMILIES:
            trials, values, history[family] = evaluate_family(
                family, models[family], plan[family], features, target, splits, folds, cfg["seed"]
            )
            rows.extend(trials)
            measurements.extend(values)
    fold_table = pd.DataFrame(measurements)
    pairs = weight_pairs(rows, measurements)
    ranked = ranked_candidates(rows)
    comparison = summary_records(ranked)
    selected = (
        {k: ranked[0][k] for k in ("candidate_id", "family", "mean_ap", "sample_sd", "evaluation")}
        if ranked
        else {"candidate_id": None}
    )
    if ranked:
        selected.update(
            status="provisional",
            settings=json.loads(ranked[0]["settings"]),
            classifier_parameters=json.loads(ranked[0]["classifier_parameters"]),
            predictors=list(PREDICTORS),
            cv_manifest_sha256=cfg["cv_manifest_sha256"],
        )
    results = root / "results/day3"
    with TemporaryDirectory(dir=root / "results") as directory:
        stage = Path(directory)
        write_json(
            stage / "search_config.json",
            {
                "configuration": cfg,
                "candidates": plan,
                "folds": folds,
                "predictors": list(PREDICTORS),
                "target": TARGET,
                "positive_class": 1,
                "scoring": "average_precision_score",
                "strategy": "Seeded structural sampling without replacement, paired class weights, "
                "RandomizedSearchCV over complete prepared lists",
                "n_jobs": 1,
                "native_threads": 1,
                "refit": False,
                "sd_ddof": 1,
                "ranking_rule": "docs/day3_model_selection.md#prespecified-decision-rule",
            },
        )
        pd.DataFrame(rows).to_csv(stage / "candidate_results.csv", index=False, lineterminator="\n")
        fold_table.to_csv(stage / "candidate_fold_metrics.csv", index=False, lineterminator="\n")
        pairs.to_csv(stage / "class_weight_analysis.csv", index=False, lineterminator="\n")
        write_json(stage / "selected_configuration.json", selected)
        write_json(
            stage / "model_comparison.json",
            {
                "partition": "train",
                "rows": len(data),
                "positives": int(target.sum()),
                "prevalence": float(target.mean()),
                "comparisons": comparison,
                "ranking": [r["candidate_id"] for r in ranked],
            },
        )
        write_report(stage / "model_comparison.md", rows, fold_table, selected, pairs)
        save_figures(stage / "figures", comparison, fold_table, pairs)
        artifacts = {
            p.relative_to(stage).as_posix(): sha256_file(p)
            for p in sorted(stage.rglob("*"))
            if p.is_file()
        }
        write_json(
            stage / "execution.json",
            {
                "command": "uv run --locked ai4i-pm compare-models",
                "platform": platform.platform(),
                "environment": {
                    **environment_versions(),
                    **{p: version(p) for p in ("matplotlib", "scipy", "joblib", "threadpoolctl")},
                },
                "inputs_sha256": inputs,
                "artifacts_sha256": artifacts,
                "families": history,
                "candidates": len(rows),
                "planned_fits": len(measurements),
                "failed_folds": sum(r["failed_folds"] for r in rows),
                "diagnostic_retries": sum(r["failed_folds"] for r in rows),
                "elapsed_seconds": perf_counter() - start,
            },
        )
        # A reference mismatch must retain a diagnostic record without replacing valid outputs.
        try:
            verify_references(root, fold_table)
        except ValueError:
            diagnostic = root / "results" / f"day3-reference-failure-{uuid4().hex}"
            stage.rename(diagnostic)
            raise ValueError(f"Baseline replay failed; diagnostic record in {diagnostic}") from None
        for name in [*artifacts, "execution.json"]:
            destination = results / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            (stage / name).replace(destination)
    return selected
