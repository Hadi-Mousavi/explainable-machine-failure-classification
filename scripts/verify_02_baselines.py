"""Repeat the baseline run in a temporary root containing only approved inputs."""

from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import pandas as pd

from ai4i_pm.baselines import run_baselines
from ai4i_pm.schema import IDENTIFIER, TARGET
from ai4i_pm.splits import load_training
from ai4i_pm.utils import read_json, sha256_file


def verify(root: Path) -> None:
    output = root / "results/02_baseline_evaluation"
    execution = read_json(output / "execution.json")
    manifest = read_json(output / "cv_manifest.json")
    target = load_training(root).set_index(IDENTIFIER)[TARGET].sort_index()
    fold_ids = manifest["evaluation_udis"]
    memberships = [(udi, int(fold)) for fold, ids in fold_ids.items() for udi in ids]
    folds = pd.DataFrame(memberships, columns=[IDENTIFIER, "fold"]).set_index(IDENTIFIER)
    folds = folds.sort_index()
    if sorted(fold_ids) != ["1", "2", "3", "4", "5"] or not folds.index.equals(target.index):
        raise ValueError("Expected five disjoint evaluation folds covering the frozen training IDs")
    oof = pd.read_csv(output / "oof_predictions.csv")
    if set(oof.model) != {"dummy_prior", "logistic_regression"}:
        raise ValueError("Unexpected models in OOF predictions")
    for _, rows in oof.groupby("model"):
        indexed = rows.set_index(IDENTIFIER, verify_integrity=True).sort_index()
        if not indexed.fold.equals(folds.fold) or not indexed[TARGET].equals(target):
            raise ValueError("OOF rows differ from the frozen training IDs, labels or shared folds")

    with TemporaryDirectory() as directory:
        isolated = Path(directory)
        for name in (
            "data/processed/train.csv",
            "results/01_data_exploration/provenance.json",
            "results/01_data_exploration/split_manifest.json",
            "results/01_data_exploration/training_partition.json",
            "configs/02_baseline_evaluation.toml",
            "uv.lock",
            "results/02_baseline_evaluation/cv_manifest.json",
        ):
            (isolated / name).parent.mkdir(parents=True, exist_ok=True)
            copyfile(root / name, isolated / name)
        run_baselines(isolated)
        replica = isolated / "results/02_baseline_evaluation"
        if sha256_file(output / "execution.json") != sha256_file(replica / "execution.json"):
            raise ValueError("Execution metadata or input/output digests changed on replay")
        for name, digest in execution["artifacts_sha256"].items():
            if sha256_file(output / name) != digest or sha256_file(replica / name) != digest:
                raise ValueError(f"Artifact replay checksum mismatch: {name}")
    print(
        f"Verified {len(folds)} training UDIs, {len(oof)} paired OOF rows and "
        f"{len(execution['artifacts_sha256']) + 1} byte-identical artifacts in isolated replay."
    )


if __name__ == "__main__":
    verify(Path(__file__).resolve().parents[1])
