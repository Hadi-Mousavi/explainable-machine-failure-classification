"""Prespecified sensitivity of four fixed classifiers on the original training partition."""

import platform
import tomllib
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ai4i_pm.ablations import file_hashes
from ai4i_pm.cv import assign_folds
from ai4i_pm.explainability import fit_fold, load_inputs, write_csv
from ai4i_pm.model_search import frozen_splits, score_ap
from ai4i_pm.robustness_figures import figures, summarize
from ai4i_pm.utils import (
    environment_versions,
    json_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
    write_json,
)

CONFIG = "configs/06_robustness_analysis.toml"
PROTOCOL_SHA256 = "a50f7fbda76b3eb32d87dc6cae9824c13be74ade02c19aadd18093e672e23d6b"
OUTPUT = "results/06_robustness_analysis"


def protocol(root):
    if sha256_file(root / CONFIG) != PROTOCOL_SHA256:
        raise ValueError("Phase 06 protocol differs from the prespecified design")
    return tomllib.loads((root / CONFIG).read_text(encoding="utf-8"))


def digest(value):
    return sha256_bytes(json_bytes(value))


def alternative_folds(target, settings, seed):
    folds = assign_folds(
        target, dict(n_splits=settings["folds"], shuffle=settings["shuffle"], random_state=seed)
    )
    members = {str(k): v.index.tolist() for k, v in folds.groupby(folds)}
    if digest(members) != settings["membership_sha256"][str(seed)]:
        raise ValueError("Alternative evaluation UDI lists differ from preregistration")
    splits, records = frozen_splits(
        target, {"primary_partition": "train", "evaluation_udis": members}
    )
    if sorted(r["eval_positives"] for r in records) != settings["sorted_eval_positives"] or any(
        (r["fit_rows"], r["eval_rows"]) != (settings["fit_rows"], settings["eval_rows"])
        for r in records
    ):
        raise ValueError("Unexpected stratified fold support")
    return members, splits


def ordered_splits(target, settings):
    ids = target.index
    if (
        not pd.api.types.is_integer_dtype(ids)
        or not ids.is_unique
        or not ids.is_monotonic_increasing
    ):
        raise ValueError("Expected training rows sorted by unique numeric UDI")
    starts, ends = settings["fit_ends"], settings["eval_ends"]
    if not starts or ends[:-1] != starts[1:] or ends[-1] != len(ids):
        raise ValueError("Ordered evaluation blocks must be contiguous and reach the last row")
    splits = []
    for start, end in zip(starts, ends, strict=True):
        if not 0 < start < end:
            raise ValueError("Invalid expanding window")
        splits.append((np.arange(start), np.arange(start, end)))
    return splits


def support(target, fit, evaluate):
    record = {}
    for label, positions in (("fit", fit), ("eval", evaluate)):
        actual = target.iloc[positions]
        positives = int(actual.sum())
        record.update(
            {
                f"{label}_rows": len(actual),
                f"{label}_positives": positives,
                f"{label}_negatives": len(actual) - positives,
                f"{label}_udis_sha256": digest(actual.index.tolist()),
            }
        )
    record["prevalence_baseline"] = record["eval_positives"] / record["eval_rows"]
    return record


def evaluate(candidate, params, columns, features, target, fit, evaluate_ids):
    start = perf_counter()
    if target.iloc[fit].nunique() < 2:
        return None, "single-class fit", None
    model = fit_fold(candidate, params, columns, features, target, fit, evaluate_ids)
    actual = target.iloc[evaluate_ids]
    if actual.sum() == 0:
        return None, "no evaluation positives", perf_counter() - start
    ap = score_ap(model, features.iloc[evaluate_ids], actual)
    if not np.isfinite(ap):
        raise ValueError("Unexpected nonfinite AP")
    return ap, "defined", perf_counter() - start


def run_study(root, folder, cfg):
    features, target, original, _, params, columns, reference = load_inputs(root, cfg)
    if len(target) != cfg["rows"] or int(target.sum()) != cfg["positives"]:
        raise ValueError("Training support mismatch")
    model_specs = {
        key: {
            "candidate": candidate,
            "parameters": params[candidate],
            "condition": condition,
            "features": columns[condition],
        }
        for key, (candidate, condition) in cfg["models"].items()
    }
    write_json(
        folder / "robustness_config.json",
        {"protocol": cfg, "protocol_sha256": PROTOCOL_SHA256, "models": model_specs},
    )
    cv = cfg["random_cv"]
    sets = [("random_cv", cv["reference_seed"], original)]
    alt = {}
    for seed in cv["seeds"]:
        members, splits = alternative_folds(target, cv, seed)
        alt[str(seed)] = members
        sets.append(("random_cv", seed, splits))
    write_json(
        folder / "alternative_cv_manifest.json",
        {
            "training_sha256": cfg["inputs_sha256"]["data/processed/train.csv"],
            "fit_membership": "Complement of evaluation UDIs within the approved training export.",
            "evaluation_udis_by_seed": alt,
        },
    )
    ordered = ordered_splits(target, cfg["ordered"])
    write_json(
        folder / "ordered_split_manifest.json",
        {
            "training_sha256": cfg["inputs_sha256"]["data/processed/train.csv"],
            "blocks": [
                {
                    "block": i,
                    "fit_udis": target.index[a].tolist(),
                    "evaluation_udis": target.index[b].tolist(),
                    **support(target, a, b),
                }
                for i, (a, b) in enumerate(ordered, 1)
            ],
        },
    )
    sets.append(("udi_ordered", None, ordered))

    rows = []
    for strategy, seed, splits in sets:
        for block, (fit, evaluate_ids) in enumerate(splits, 1):
            counts = support(target, fit, evaluate_ids)
            for key, spec in model_specs.items():
                candidate, condition = cfg["models"][key]
                historical = seed == cv["reference_seed"]
                if historical:
                    ref = reference.loc[(candidate, condition, block)]
                    if any(ref[k] != counts[k] for k in ref.index if k in counts):
                        raise ValueError("Seed-42 reference membership or support mismatch")
                    ap, status, elapsed = float(ref.ap), "defined", None
                else:
                    ap, status, elapsed = evaluate(
                        candidate,
                        params[candidate],
                        columns[condition],
                        features,
                        target,
                        fit,
                        evaluate_ids,
                    )
                rows.append(
                    {
                        "experiment": strategy,
                        "seed": seed,
                        "fold_or_block": block,
                        "model": key,
                        "candidate": candidate,
                        "condition": condition,
                        **counts,
                        "ap": ap,
                        "metric_status": status,
                        "runtime_seconds": elapsed,
                        "source": "phase04_reference" if historical else "phase06_fit",
                        "protocol_sha256": PROTOCOL_SHA256,
                        "model_sha256": digest(spec),
                    }
                )
    metrics = pd.DataFrame(rows).astype({"seed": "Int64"})
    random = metrics[metrics.experiment == "random_cv"]
    blocks = metrics[metrics.experiment == "udi_ordered"]
    write_csv(folder, "random_split_fold_metrics", random)
    write_csv(folder, "ordered_block_metrics", blocks)
    summary = summarize(random, blocks, folder)
    figures(random, blocks, summary, folder)
    execution = {
        "protocol_sha256": PROTOCOL_SHA256,
        "inputs_sha256": cfg["inputs_sha256"],
        "partition": "train",
        "source_sha256_via_provenance": cfg["source_sha256"],
        "new_fits": int(metrics.runtime_seconds.notna().sum()),
        "reference_evaluations": int((metrics.source == "phase04_reference").sum()),
        "environment": environment_versions(),
        "platform": platform.platform(),
        "artifacts_sha256": file_hashes(folder),
    }
    write_json(folder / "execution.json", execution)
    return execution


def check_inventory(folder):
    record = read_json(folder / "execution.json")
    inventory = file_hashes(folder)
    del inventory["execution.json"]
    if record["protocol_sha256"] != PROTOCOL_SHA256 or inventory != record["artifacts_sha256"]:
        raise ValueError("Phase 06 result inventory or protocol mismatch")
    return record


def run_robustness(root: Path):
    cfg = protocol(root)
    output = root / OUTPUT
    if output.exists():
        record = check_inventory(output)
        if cfg["inputs_sha256"] != record["inputs_sha256"] or any(
            sha256_file(root / name) != expected for name, expected in cfg["inputs_sha256"].items()
        ):
            raise ValueError("Phase 06 inputs differ from the verified run")
        return record
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        TemporaryDirectory(dir=output.parent) as directory,
        threadpool_limits(limits=cfg["native_threads"]),
    ):
        stage = Path(directory) / "study"
        record = run_study(root, stage, cfg)
        stage.rename(output)
    return record
