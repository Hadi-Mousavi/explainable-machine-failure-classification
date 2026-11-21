"""Repeat all Phase 06 fits in a training-only directory and compare scientific results."""

import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import pandas as pd

from ai4i_pm.ablations import file_hashes
from ai4i_pm.robustness import CONFIG, OUTPUT, check_inventory, protocol, run_robustness
from ai4i_pm.utils import read_json


def compare_summary(original, replay, tol):
    if isinstance(original, dict):
        if original.keys() != replay.keys():
            raise ValueError("Summary schema differs")
        for key in original:
            compare_summary(original[key], replay[key], tol)
    elif isinstance(original, float):
        if not isinstance(replay, (float, int)) or not abs(replay - original) <= tol:
            raise ValueError("Numerical summary differs")
    elif original != replay:
        raise ValueError("Summary metadata differs")


def verify(root):
    cfg = protocol(root)
    folder = root / OUTPUT
    original = check_inventory(folder)
    with TemporaryDirectory() as tmp:
        isolated = Path(tmp)
        for name in (CONFIG, *cfg["inputs_sha256"]):
            path = isolated / name
            path.parent.mkdir(parents=True, exist_ok=True)
            copyfile(root / name, path)
        replay = run_robustness(isolated)
        replica = isolated / OUTPUT
        check_inventory(replica)
        for key in (
            "protocol_sha256",
            "partition",
            "inputs_sha256",
            "source_sha256_via_provenance",
            "new_fits",
            "reference_evaluations",
        ):
            if original[key] != replay[key]:
                raise ValueError(f"Execution mismatch for {key}")
        left, right = file_hashes(folder), file_hashes(replica)
        if left.keys() != right.keys():
            raise ValueError("Result paths differ")
        different = [name for name in left if left[name] != right[name]]
        for name in (
            "robustness_config.json",
            "alternative_cv_manifest.json",
            "ordered_split_manifest.json",
        ):
            if name in different:
                raise ValueError(f"Deterministic artifact differs: {name}")
        tol = cfg["replay_atol"]
        compare_summary(
            read_json(folder / "robustness_summary.json"),
            read_json(replica / "robustness_summary.json"),
            tol,
        )
        max_error = 0.0
        n_rows = 0
        for name in ("random_split_fold_metrics.csv", "ordered_block_metrics.csv"):
            a, b = [pd.read_csv(f / name, float_precision="round_trip") for f in (folder, replica)]
            for table in (a, b):
                elapsed = table.pop("runtime_seconds")
                expected = (table.source == "phase06_fit") & (
                    table.metric_status != "single-class fit"
                )
                if not elapsed.notna().equals(expected) or not (elapsed[expected] >= 0).all():
                    raise ValueError("Missing or invalid runtime")
            pd.testing.assert_frame_equal(a, b, rtol=0, atol=tol)
            error = (a.ap - b.ap).abs().max()
            max_error = max(max_error, error) if pd.notna(error) else max_error
            n_rows += len(b)
        expected_rows = len(cfg["models"]) * (
            (1 + len(cfg["random_cv"]["seeds"])) * cfg["random_cv"]["folds"]
            + len(cfg["ordered"]["fit_ends"])
        )
        if n_rows != expected_rows:
            raise ValueError("Incomplete evaluation ledger")
        if max_error == 0 and "robustness_summary.md" in different:
            raise ValueError("Narrative summary changed despite identical scores")
        return {
            "new_fits_replayed": replay["new_fits"],
            "reference_rows": replay["reference_evaluations"],
            "evaluation_rows_compared": n_rows,
            "max_absolute_ap_difference": max_error,
            "absolute_tolerance": tol,
            "total_files": len(left),
            "identical_files": len(left) - len(different),
            "different_bytes": different,
            "observation_inputs": ["data/processed/train.csv"],
        }


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parents[1]), indent=2))
