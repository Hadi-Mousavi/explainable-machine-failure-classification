"""Replay Phase 05 with an allowlist of files and no raw or held-out observations."""

import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from ai4i_pm.ablations import file_hashes
from ai4i_pm.explainability import CONFIG, OUTPUT, load_protocol, run_explanations
from ai4i_pm.utils import read_json

ATOL = 1e-12


def verify_inventory(folder: Path) -> dict:
    record = read_json(folder / "execution.json")
    hashes = file_hashes(folder)
    del hashes["execution.json"]
    if hashes != record["artifacts_sha256"]:
        raise ValueError("Phase 05 artifact inventory or digest mismatch")
    if record["fold_fits"] != 20 or record["permutation_measurements"] != 600:
        raise ValueError("Incomplete permutation experiment")
    return record


def compare_arrays(original: Path, replay: Path) -> float:
    maximum = 0.0
    with (
        np.load(original, allow_pickle=False) as first,
        np.load(replay, allow_pickle=False) as second,
    ):
        if first.files != second.files:
            raise ValueError("NPZ keys differ")
        for key in first.files:
            a, b = first[key], second[key]
            if a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError(f"Changed array schema: {key}")
            if a.dtype.kind == "f":
                np.testing.assert_allclose(a, b, rtol=0, atol=ATOL, equal_nan=False)
                maximum = max(maximum, float(np.abs(a - b).max()))
            else:
                np.testing.assert_array_equal(a, b)
    return maximum


def compare_metadata(a: object, b: object) -> None:
    if isinstance(a, float):
        np.testing.assert_allclose(a, b, rtol=0, atol=ATOL, equal_nan=False)
    elif isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise ValueError("Replay metadata keys differ")
        for k in a:
            compare_metadata(a[k], b[k])
    elif isinstance(a, list):
        for left, right in zip(a, b, strict=True):
            compare_metadata(left, right)
    elif a != b:
        raise ValueError(f"Replay metadata differs: {a!r} vs {b!r}")


def verify(root: Path) -> dict:
    cfg = load_protocol(root)
    folder = root / OUTPUT
    original = verify_inventory(folder)
    with TemporaryDirectory() as directory:
        isolated = Path(directory)
        for name in [CONFIG, *cfg["inputs_sha256"]]:
            path = isolated / name
            path.parent.mkdir(parents=True, exist_ok=True)
            copyfile(root / name, path)
        run_explanations(isolated)
        replica = isolated / OUTPUT
        replay = verify_inventory(replica)
        left, right = file_hashes(folder), file_hashes(replica)
        if left.keys() != right.keys():
            raise ValueError("Replay inventory differs")
        compare_metadata(
            {
                k: v
                for k, v in original.items()
                if k not in ("artifacts_sha256", "platform", "environment")
            },
            {
                k: v
                for k, v in replay.items()
                if k not in ("artifacts_sha256", "platform", "environment")
            },
        )
        max_array_error = 0.0
        for name in left:
            if name.endswith(".csv"):
                a, b = [
                    pd.read_csv(f / name, float_precision="round_trip") for f in (folder, replica)
                ]
                pd.testing.assert_frame_equal(a, b, rtol=0, atol=ATOL)
            elif name == "shap_values.npz":
                max_array_error = compare_arrays(folder / name, replica / name)
            elif name == "shap_metadata.json":
                compare_metadata(read_json(folder / name), read_json(replica / name))
            elif name == "explanation_config.json" and left[name] != right[name]:
                raise ValueError(f"Deterministic artifact differs: {name}")
        differences = [name for name in left if left[name] != right[name]]
        # The report formats residuals that can change by round-off on another CPU.
        tables_match = all(left[n] == right[n] for n in left if n.endswith(".csv"))
        metadata_matches = left["shap_metadata.json"] == right["shap_metadata.json"]
        if tables_match and metadata_matches and "explanation_summary.md" in differences:
            raise ValueError("Summary text differs despite identical underlying results")
        return {
            "replayed_fits": replay["fold_fits"],
            "measurements": replay["permutation_measurements"],
            "shap_status": replay["shap_status"],
            "numerical_atol": ATOL,
            "maximum_npz_error": max_array_error,
            "identical_files": len(left) - len(differences),
            "total_files": len(left),
            "different_bytes": differences,
            "same_environment": original["environment"] == replay["environment"],
            "same_platform": original["platform"] == replay["platform"],
        }


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parents[1]), indent=2))
