"""Repeat all 60 fits in a root containing only the approved inputs."""

from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import pandas as pd

from ai4i_pm.ablations import CONFIG, INPUTS, file_hashes, run_ablations
from ai4i_pm.utils import read_json


def validate_inventory(folder):
    record = read_json(folder / "execution.json")
    hashes = file_hashes(folder)
    if record["fold_evaluations"] != 60 or record["artifacts_sha256"] != {
        k: v for k, v in hashes.items() if k != "execution.json"
    }:
        raise ValueError("Day 4 output inventory or evaluation count differs")
    return record, hashes


def verify(root):
    output = root / "results/day4"
    original, hashes = validate_inventory(output)
    with TemporaryDirectory() as directory:
        isolated = Path(directory)
        for name in (CONFIG, *INPUTS):
            (isolated / name).parent.mkdir(parents=True, exist_ok=True)
            copyfile(root / name, isolated / name)
        run_ablations(isolated)
        replica = isolated / "results/day4"
        replay, new_hashes = validate_inventory(replica)
        if (
            original["inputs_sha256"] != replay["inputs_sha256"]
            or set(hashes) != set(new_hashes)
            or hashes["ablation_config.json"] != new_hashes["ablation_config.json"]
        ):
            raise ValueError("Isolated replay used a different design or artifact inventory")
        for name in ("ablation_fold_metrics.csv", "ablation_summary.csv", "paired_differences.csv"):
            actual, repeated = [
                pd.read_csv(folder / name, float_precision="round_trip")
                for folder in (output, replica)
            ]
            pd.testing.assert_frame_equal(actual, repeated, rtol=0, atol=1e-12)
        changed = [name for name in hashes if hashes[name] != new_hashes[name]]
        print(
            f"Reproduced 60 fold scores and their paired comparisons within 1e-12. "
            f"{len(hashes) - len(changed)}/{len(hashes)} artifacts byte-identical. "
            f"Differing files: {', '.join(changed) or 'none'}."
        )


if __name__ == "__main__":
    verify(Path(__file__).resolve().parents[1])
