"""Repeat deterministic Day 1 operations and save checksum-level verification evidence."""

from pathlib import Path

from PIL import Image

from ai4i_pm.audit import run_audit
from ai4i_pm.data import acquire, load_raw
from ai4i_pm.eda import generate_eda
from ai4i_pm.splits import build_manifest, freeze_split, read_settings
from ai4i_pm.utils import environment_versions, read_json, sha256_file, write_json


def main() -> None:
    """Check saved artifacts, repeat outputs, and verify real-data reorder invariance."""
    root = Path(__file__).resolve().parents[1]
    archive = read_json(root / "results/day1/provenance.json").get("archive_file")
    paths = [
        *([archive] if archive else []),
        "data/raw/ai4i2020.csv",
        "data/processed/train.csv",
        "results/day1/provenance.json",
        "results/day1/audit.json",
        "results/day1/audit.md",
        "results/day1/split_manifest.json",
        "results/day1/training_partition.json",
        "results/day1/eda_summary.json",
        "results/day1/eda.md",
        *[
            path.relative_to(root).as_posix()
            for path in sorted((root / "results/day1/figures").glob("*.png"))
        ],
    ]
    before = {name: sha256_file(root / name) for name in paths}
    acquire(root)
    audit = run_audit(root)
    frozen = freeze_split(root)
    eda = generate_eda(root)
    after = {name: sha256_file(root / name) for name in paths}
    if before != after:
        raise RuntimeError(
            f"Non-deterministic outputs: {[name for name in paths if before[name] != after[name]]}"
        )
    if audit["unexpected_issues"]:
        raise RuntimeError(f"Integrity issues: {audit['unexpected_issues']}")
    frame, checksum = load_raw(root)
    reordered = build_manifest(
        frame.sample(frac=1, random_state=123), checksum, read_settings(root)
    )
    if reordered["partitions"] != frozen["partitions"]:
        raise RuntimeError("Partition membership changed under dataframe row permutation")
    image_checks = {}
    for name in eda["figures"]:
        path = root / "results/day1/figures" / name
        with Image.open(path) as picture:
            dpi = picture.info.get("dpi", (0, 0))
            if picture.format != "PNG" or min(dpi) < 299:
                raise RuntimeError(f"Invalid publication figure format/resolution: {path}")
            picture.verify()
            image_checks[name] = {"width": picture.width, "height": picture.height, "dpi": dpi}
    executed_path = root / "notebooks/01_eda_executed.ipynb"
    notebook_check: dict = {"status": "NOT_RUN", "reason": "Execute notebook separately if desired"}
    if executed_path.exists():
        notebook = read_json(executed_path)
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        errors = [
            out for cell in code_cells for out in cell["outputs"] if out["output_type"] == "error"
        ]
        if errors or not all(cell["execution_count"] is not None for cell in code_cells):
            raise RuntimeError("Executed notebook contains errors or unexecuted code cells")
        notebook_check = {
            "status": "PASS",
            "code_cells": len(code_cells),
            "error_outputs": len(errors),
        }
    evidence = {
        "status": "PASS",
        "environment": environment_versions(),
        "uv_lock_sha256": sha256_file(root / "uv.lock"),
        "identical_repeat_artifacts": len(paths),
        "sha256_before": before,
        "sha256_after": after,
        "real_dataset_row_permutation_same_membership": True,
        "partition_invariants": frozen["validation"],
        "figures": image_checks,
        "notebook": notebook_check,
        "models_trained": False,
        "scope": "Local runtime only; CI runner and cross-platform pixel identity not tested",
    }
    write_json(root / "results/day1/reproducibility.json", evidence)
    print(
        f"PASS: {len(paths)} artifacts byte-identical on repeat; membership invariant to row order"
    )
    print(f"PASS: {len(image_checks)} PNGs at 300 dpi; notebook: {notebook_check['status']}")


if __name__ == "__main__":
    main()
