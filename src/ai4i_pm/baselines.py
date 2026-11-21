"""Run the two prespecified baselines against the approved training export."""

import platform
import tomllib
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from ai4i_pm.baseline_figures import NAMES, save_figures
from ai4i_pm.cv import cross_validate
from ai4i_pm.models import baseline_pipelines
from ai4i_pm.schema import NUMERIC_FEATURES, PREDICTORS, TARGET
from ai4i_pm.splits import load_training
from ai4i_pm.utils import environment_versions, read_json, sha256_file, write_json


def verified_training(root: Path, data: dict):
    """Check the approved source, training membership and class support before fitting."""
    exploration = root / "results/01_data_exploration"
    provenance = read_json(exploration / "provenance.json")
    descriptor = read_json(exploration / "training_partition.json")
    if provenance["csv_sha256"] != data["dataset_sha256"] or any(
        descriptor[key] != data[key]
        for key in ("dataset_sha256", "split_manifest_sha256", "training_sha256")
    ):
        raise ValueError("Provenance or training descriptor differs from approved Phase 02 hashes")
    training = load_training(root)
    if len(training) != data["rows"] or training[TARGET].sum() != data["positives"]:
        raise ValueError("Training support differs from the approved Phase 02 counts")
    return training


def run_baselines(root: Path) -> dict:
    config_file = root / "configs/02_baseline_evaluation.toml"
    with config_file.open("rb") as stream:
        config = tomllib.load(stream)
    training = verified_training(root, config["data"])
    pipelines = baseline_pipelines(config["models"])
    output = root / "results/02_baseline_evaluation"
    manifest_path = output / "cv_manifest.json"
    memberships = None
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["settings"] != config["cv"] or manifest["inputs"] != config["data"]:
            raise ValueError("Saved CV manifest differs from the baseline configuration")
        memberships = manifest["evaluation_udis"]
    folds, metrics, oof = cross_validate(
        training, pipelines, config["cv"], config["metrics"], evaluation_udis=memberships
    )
    sd_ddof = config["metrics"]["fold_ap_std_ddof"]
    summary = {
        "partition": "train",
        "rows": len(training),
        "positives": int(training[TARGET].sum()),
        "prevalence": float(training[TARGET].mean()),
        "fold_ap_std_ddof": sd_ddof,
        "models": {},
    }
    for name, group in metrics.groupby("model"):
        summary["models"][name] = {
            "mean_fold_ap": group.average_precision.mean(),
            "std_fold_ap": group.average_precision.std(ddof=sd_ddof),
            "oof_confusion_counts": group[["tn", "fp", "fn", "tp"]].sum().to_dict(),
        }
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as temp:
        stage = Path(temp)
        write_json(
            stage / "cv_manifest.json",
            {
                "primary_partition": "train",
                "identity": "UDI",
                "order": "UDI ascending",
                "splitter": "StratifiedKFold",
                "settings": config["cv"],
                "inputs": config["data"],
                "fit_membership": "All training UDIs outside the listed evaluation fold",
                "evaluation_udis": {
                    str(k): part.index.tolist() for k, part in folds.groupby(folds)
                },
            },
        )
        write_json(
            stage / "model_specs.json",
            {
                "input_columns": list(PREDICTORS),
                "remainder": "drop",
                "preprocessing_fit": "Independently within each model and fold-training partition",
                "categorical": {
                    "columns": ["Type"],
                    "encoder": "OneHotEncoder",
                    "handle_unknown": "ignore",
                },
                "numeric": {
                    "columns": list(NUMERIC_FEATURES),
                    "scaler": "StandardScaler",
                    "with_mean": True,
                    "with_std": True,
                },
                "classifiers": {
                    name: {
                        "class": type(pipe["classifier"]).__name__,
                        "parameters": pipe["classifier"].get_params(deep=False),
                    }
                    for name, pipe in pipelines.items()
                },
                "metrics": config["metrics"],
            },
        )
        metrics.to_csv(stage / "fold_metrics.csv", index=False, lineterminator="\n")
        oof.to_csv(stage / "oof_predictions.csv", index=False, lineterminator="\n")
        write_json(stage / "baseline_summary.json", summary)
        lines = [
            "# Phase 02 training cross-validation",
            "",
            f"{len(training):,} observations, {summary['positives']} failures "
            f"({summary['prevalence']:.2%}). Each model scored every training UDI once.",
            "",
            "| Model | Mean fold AP | Sample SD across folds |",
            "| --- | ---: | ---: |",
        ]
        for name, stats in summary["models"].items():
            lines.append(
                f"| {NAMES[name]} | {stats['mean_fold_ap']:.6f} | {stats['std_fold_ap']:.6f} |"
            )
        lines.extend(
            [
                "",
                "AP weights precision by increments in recall. The SD describes the "
                "five fold scores. Diagnostics use score ≥ 0.5 and assign zero to undefined "
                "ratios. The threshold was fixed before fitting.",
                "",
                "These results describe synthetic training observations. See "
                "[fold metrics](fold_metrics.csv) and the "
                "[Phase 02 protocol](../../docs/02_baseline_evaluation.md) for interpretation.",
            ]
        )
        (stage / "baseline_summary.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
        )
        save_figures(stage, metrics, oof)
        files = sorted(path.relative_to(stage) for path in stage.rglob("*") if path.is_file())
        inputs = [
            config_file,
            root / "uv.lock",
            root / "results/01_data_exploration/provenance.json",
            root / "results/01_data_exploration/training_partition.json",
        ]
        write_json(
            stage / "execution.json",
            {
                "command": "uv run --locked ai4i-pm baselines",
                "platform": platform.platform(),
                "environment": {
                    **environment_versions(),
                    **{p: version(p) for p in ["scipy", "matplotlib", "joblib", "threadpoolctl"]},
                },
                "inputs_sha256": {p.relative_to(root).as_posix(): sha256_file(p) for p in inputs},
                "artifacts_sha256": {p.as_posix(): sha256_file(stage / p) for p in files},
            },
        )
        for path in [*files, Path("execution.json")]:
            destination = output / path
            if destination.exists() and sha256_file(destination) != sha256_file(stage / path):
                raise ValueError(f"Existing baseline evidence differs on replay: {path}")
        for path in [*files, Path("execution.json")]:
            destination = output / path
            if not destination.exists():
                destination.parent.mkdir(exist_ok=True)
                (stage / path).rename(destination)
    return summary
