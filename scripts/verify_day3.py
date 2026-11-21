"""Check the ledger, then repeat the comparison with only its approved inputs."""

import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from ai4i_pm.comparison import INPUTS, run_comparison
from ai4i_pm.utils import json_bytes, read_json, sha256_bytes, sha256_file


def read_table(folder, name):
    return pd.read_csv(folder / name, float_precision="round_trip")


def validate_ledger(root, output):
    execution = read_json(output / "execution.json")
    if set(execution["inputs_sha256"]) != set(INPUTS):
        raise ValueError("Unexpected input inventory")
    for path, digest in execution["inputs_sha256"].items():
        if sha256_file(root / path) != digest:
            raise ValueError(f"Input checksum mismatch: {path}")
    for path, digest in execution["artifacts_sha256"].items():
        if sha256_file(output / path) != digest:
            raise ValueError(f"Output checksum mismatch: {path}")
    config = read_json(output / "search_config.json")
    metrics = read_table(output, "candidate_fold_metrics.csv")
    trials = read_table(output, "candidate_results.csv").set_index("candidate_id")
    memberships = read_json(root / "results/day2/cv_manifest.json")["evaluation_udis"]
    grouped = metrics.groupby("candidate_id")
    if not trials.index.is_unique or set(grouped.groups) != set(trials.index):
        raise ValueError("Incomplete candidate or fold ledger")
    expected = {
        f"{family}-{i:02d}"
        for family, cases in config["candidates"].items()
        for i in range(1, len(cases) + 1)
    }
    if set(trials.index) != expected or len(metrics) != len(trials) * 5:
        raise ValueError("Candidate ledger differs from the search plan")
    if execution["candidates"] != len(trials) or execution["planned_fits"] != len(metrics):
        raise ValueError("Execution counts disagree with the ledgers")
    for family, cases in config["candidates"].items():
        for i, settings in enumerate(cases, 1):
            row = trials.loc[f"{family}-{i:02d}"]
            params = json.loads(row.classifier_parameters)
            if json.loads(row.settings) != settings or any(
                params[k] != v for k, v in settings.items()
            ):
                raise ValueError(f"Candidate settings differ from plan for {row.name}")
    for fold, ids in memberships.items():
        part = metrics[metrics.fold == int(fold)].set_index("candidate_id")
        if (
            not part.index.is_unique
            or set(part.index) != expected
            or not part.eval_udis_sha256.eq(sha256_bytes(json_bytes(ids))).all()
            or not part.eval_rows.eq(len(ids)).all()
        ):
            raise ValueError(f"Evaluation membership mismatch in fold {fold}")
    failures = grouped.ap.apply(lambda x: x.isna().sum()).reindex(trials.index)
    np.testing.assert_array_equal(trials.failed_folds, failures)
    for column, statistic in [("mean_ap", grouped.ap.mean()), ("sample_sd", grouped.ap.std())]:
        np.testing.assert_allclose(
            trials[column],
            statistic.reindex(trials.index).mask(failures > 0),
            atol=1e-14,
            rtol=0,
            equal_nan=True,
        )
    selected = read_json(output / "selected_configuration.json")
    eligible = trials[trials.failed_folds == 0]
    if (
        selected["candidate_id"] not in eligible.index
        or selected["mean_ap"] != eligible.mean_ap.max()
        or selected["mean_ap"] != trials.loc[selected["candidate_id"], "mean_ap"]
    ):
        raise ValueError("Selected configuration disagrees with eligible candidate scores")
    pairs = read_table(output, "class_weight_analysis.csv")
    paired_ids = [*pairs.unweighted_id, *pairs.balanced_id]
    if len(set(paired_ids)) != len(paired_ids) or set(paired_ids) != expected - {"dummy_prior-01"}:
        raise ValueError("Weight pairs must cover each eligible family configuration once")
    scores = metrics.pivot(index="candidate_id", columns="fold", values="ap")
    for pair in pairs.itertuples():
        unweighted = json.loads(trials.loc[pair.unweighted_id, "classifier_parameters"])
        weighted = json.loads(trials.loc[pair.balanced_id, "classifier_parameters"])
        if unweighted["class_weight"] is not None or weighted != {
            **unweighted,
            "class_weight": "balanced",
        }:
            raise ValueError(f"Unmatched classifier settings for {pair.balanced_id}")
        delta = scores.loc[pair.balanced_id] - scores.loc[pair.unweighted_id]
        np.testing.assert_allclose(
            [getattr(pair, f"delta_fold_{k}") for k in delta.index],
            delta,
            atol=1e-14,
            rtol=0,
            equal_nan=True,
        )
        np.testing.assert_allclose(pair.delta_ap, delta.mean(skipna=False), atol=1e-14, rtol=0)
    return execution


def verify(root):
    output = root / "results/day3"
    recorded = validate_ledger(root, output)
    with TemporaryDirectory() as directory:
        isolated = Path(directory)
        for name in INPUTS:
            (isolated / name).parent.mkdir(parents=True, exist_ok=True)
            copyfile(root / name, isolated / name)
        run_comparison(isolated)
        replica = isolated / "results/day3"
        replay = validate_ledger(isolated, replica)
        for name in (
            "candidate_results.csv",
            "candidate_fold_metrics.csv",
            "class_weight_analysis.csv",
        ):
            saved = read_table(output, name).drop(columns="fit_score_seconds", errors="ignore")
            repeated = read_table(replica, name).drop(columns="fit_score_seconds", errors="ignore")
            pd.testing.assert_frame_equal(saved, repeated, rtol=0, atol=1e-12)
        for name in ("search_config.json", "selected_configuration.json", "model_comparison.json"):
            if read_json(output / name) != read_json(replica / name):
                raise ValueError(f"Replay differs: {name}")
        matching = sorted(
            name
            for name, digest in recorded["artifacts_sha256"].items()
            if replay["artifacts_sha256"].get(name) == digest
        )
        print(
            f"Verified {recorded['candidates']} candidates and {recorded['planned_fits']} fold "
            f"evaluations in training-only replay. Byte-identical outputs: {', '.join(matching)}. "
            "Wall times are measured separately in each run."
        )


if __name__ == "__main__":
    verify(Path(__file__).resolve().parents[1])
