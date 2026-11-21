"""Canonical UDI-based frozen splitting and a training-only EDA boundary."""

import json
import math
import re
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from ai4i_pm.audit import counts
from ai4i_pm.data import load_raw
from ai4i_pm.schema import (
    IDENTIFIER,
    TARGET,
    TRAINING_COLUMNS,
    validate_source_schema,
    value_issues,
)
from ai4i_pm.utils import (
    environment_versions,
    json_bytes,
    preserve_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
    utc_now,
)


@dataclass(frozen=True)
class SplitSettings:
    """Two-step fractions; the second is relative to the non-test remainder."""

    seed: int = 42
    test_fraction: float = 0.20
    validation_fraction_of_remainder: float = 0.25

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("Seed must be an integer in [0, 2**32)")
        for fraction in (self.test_fraction, self.validation_fraction_of_remainder):
            if not 0 < fraction < 1:
                raise ValueError("Split fractions must be strictly between 0 and 1")


def read_settings(root: Path) -> SplitSettings:
    """Read the version-controlled primary protocol configuration."""
    with (root / "configs/01_data_exploration.toml").open("rb") as stream:
        return SplitSettings(**tomllib.load(stream)["split"])


def validate_partitions(
    frame: pd.DataFrame, partitions: dict[str, list[int]], settings: SplitSettings
) -> dict[str, Any]:
    """Verify sizes, disjointness, completeness and per-class rounding tolerance."""
    if set(partitions) != {"train", "validation", "test"}:
        raise ValueError("Expected train, validation, and test partitions")
    n_rows = len(frame)
    test_size = math.ceil(n_rows * settings.test_fraction)
    validation_size = math.ceil((n_rows - test_size) * settings.validation_fraction_of_remainder)
    expected = {
        "train": n_rows - test_size - validation_size,
        "validation": validation_size,
        "test": test_size,
    }
    all_ids = [row_id for ids in partitions.values() for row_id in ids]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Overlapping or repeated partition identifiers")
    if set(all_ids) != set(frame[IDENTIFIER]):
        raise ValueError("Partition identifiers are incomplete or unknown")
    indexed = frame.set_index(IDENTIFIER)
    global_counts = frame[TARGET].value_counts()
    summaries = {}
    for name, ids in partitions.items():
        if len(ids) != expected[name]:
            raise ValueError(f"Wrong partition size for {name}")
        labels = indexed.loc[ids, TARGET]
        observed = labels.value_counts()
        deviations = {
            str(label): abs(int(observed.get(label, 0)) - len(ids) * count / n_rows)
            for label, count in global_counts.items()
        }
        # Two allocation stages can each contribute rounding error (<2 rows).
        if any(deviation > 2 for deviation in deviations.values()):
            raise ValueError(f"Stratification check failed for {name}")
        if set(labels.unique()) != {0, 1}:
            raise ValueError(f"Both classes must occur in {name}")
        summaries[name] = {
            "size": len(ids),
            "class_counts": counts(labels),
            "positive_fraction": float(labels.mean()),
            "class_count_deviation_from_proportional": deviations,
        }
    return {
        "zero_overlap": True,
        "complete": True,
        "all_assigned_once": True,
        "stratification_checked": True,
        "stratification_rounding_tolerance_rows": 2,
        "partitions": summaries,
    }


def build_manifest(
    frame: pd.DataFrame, dataset_sha256: str, settings: SplitSettings = SplitSettings()
) -> dict[str, Any]:
    """Canonicalize by UDI before both stratified splits; never fit an estimator."""
    if not re.fullmatch(r"[0-9a-f]{64}", dataset_sha256):
        raise ValueError("Expected a lowercase SHA-256 dataset digest")
    validate_source_schema(frame)
    issues = value_issues(frame)
    if issues:
        raise ValueError(f"Integrity issues must be resolved before splitting: {issues}")
    ordered = frame.sort_values(IDENTIFIER)
    remainder_ids, test_ids = train_test_split(
        ordered[IDENTIFIER].to_numpy(),
        test_size=settings.test_fraction,
        random_state=settings.seed,
        stratify=ordered[TARGET].to_numpy(),
    )
    # Canonicalize the remainder as well; no implicit input ordering remains.
    remainder = ordered.set_index(IDENTIFIER).loc[sorted(remainder_ids)]
    train_ids, validation_ids = train_test_split(
        remainder.index.to_numpy(),
        test_size=settings.validation_fraction_of_remainder,
        random_state=settings.seed,
        stratify=remainder[TARGET].to_numpy(),
    )
    partitions = {
        "train": sorted(int(value) for value in train_ids),
        "validation": sorted(int(value) for value in validation_ids),
        "test": sorted(int(value) for value in test_ids),
    }
    return {
        "schema_version": 1,
        "dataset_sha256": dataset_sha256,
        "identity": "UDI (unique row identifier; never a predictor)",
        "algorithm": (
            "sort UDI; sklearn train_test_split test; sort remainder UDI; split validation"
        ),
        "stratify": TARGET,
        "settings": asdict(settings),
        "environment": environment_versions(),
        "partitions": partitions,
        "validation": validate_partitions(frame, partitions, settings),
    }


def training_csv(frame: pd.DataFrame, train_ids: list[int]) -> bytes:
    """Export only training rows and the permitted analysis columns."""
    selected = frame.set_index(IDENTIFIER).loc[train_ids].reset_index()
    return selected.loc[:, list(TRAINING_COLUMNS)].to_csv(index=False, lineterminator="\n").encode()


def freeze_split(
    root: Path,
    settings: SplitSettings | None = None,
    *,
    override: bool = False,
    reason: str | None = None,
) -> dict[str, Any]:
    """Preserve a frozen manifest; changed settings need an explicit audited override."""
    if override and (reason is None or not reason.strip()):
        raise ValueError("--override requires a nonempty --reason for the audit trail")
    frame, checksum = load_raw(root)
    if len(frame) != 10_000:
        raise ValueError("Primary research split requires the verified 10000-row source")
    settings = settings or read_settings(root)
    manifest = build_manifest(frame, checksum, settings)
    csv = training_csv(frame, manifest["partitions"]["train"])
    manifest["training_artifact"] = {
        "path": "data/processed/train.csv",
        "sha256": sha256_bytes(csv),
        "columns": list(TRAINING_COLUMNS),
    }
    destination = root / "results/01_data_exploration/split_manifest.json"
    manifest_content = json_bytes(manifest)
    manifest_hash = sha256_bytes(manifest_content)
    descriptor = {
        "partition": "train",
        "dataset_sha256": checksum,
        "split_manifest_sha256": manifest_hash,
        "training_sha256": sha256_bytes(csv),
        "row_ids": manifest["partitions"]["train"],
        "columns": list(TRAINING_COLUMNS),
    }
    descriptor_path = root / "results/01_data_exploration/training_partition.json"
    csv_path = root / "data/processed/train.csv"
    changed = destination.exists() and destination.read_bytes() != manifest_content
    if changed:
        if not override:
            raise ValueError(
                "Frozen split differs (data/settings/runtime). Refusing regeneration; "
                "use --override --reason TEXT only after protocol review."
            )
        old = destination.read_bytes()
        old_hash = sha256_bytes(old)
        preserve_bytes(root / f"results/01_data_exploration/split_history/{old_hash}.json", old)
        if csv_path.exists():
            preserve_bytes(
                root / f"data/processed/history/{sha256_file(csv_path)}.csv", csv_path.read_bytes()
            )
        event = {
            "at_utc": utc_now(),
            "reason": reason.strip(),
            "previous_manifest_sha256": old_hash,
            "new_manifest_sha256": manifest_hash,
        }
        # Append-only events; previous assignments are preserved by content hash.
        history = root / "results/01_data_exploration/split_overrides.jsonl"
        with history.open("ab") as stream:
            stream.write((json.dumps(event, sort_keys=True) + "\n").encode())
        csv_path.write_bytes(csv)
        destination.write_bytes(manifest_content)
        descriptor_path.write_bytes(json_bytes(descriptor))
    else:
        # Recreate missing deterministic artifacts; never silently replace changed exports.
        for path, content in (
            (destination, manifest_content),
            (csv_path, csv),
            (descriptor_path, json_bytes(descriptor)),
        ):
            if path.exists() and path.read_bytes() != content:
                raise ValueError(f"Existing split artifact was modified: {path}")
        for path, content in (
            (destination, manifest_content),
            (csv_path, csv),
            (descriptor_path, json_bytes(descriptor)),
        ):
            preserve_bytes(path, content)
    return manifest


def load_training(root: Path) -> pd.DataFrame:
    """Read only the frozen training export; raw/validation/test values are never read."""
    descriptor = read_json(root / "results/01_data_exploration/training_partition.json")
    if descriptor["partition"] != "train":
        raise ValueError("EDA accepts the training partition only")
    if descriptor["split_manifest_sha256"] != sha256_file(
        root / "results/01_data_exploration/split_manifest.json"
    ):
        raise ValueError("Training descriptor is stale relative to the frozen split")
    csv_path = root / "data/processed/train.csv"
    if sha256_file(csv_path) != descriptor["training_sha256"]:
        raise ValueError("Training export checksum mismatch")
    frame = pd.read_csv(csv_path)
    if tuple(frame.columns) != TRAINING_COLUMNS:
        raise ValueError("Unexpected columns in the training export")
    if frame[IDENTIFIER].tolist() != descriptor["row_ids"]:
        raise ValueError("Training row identities differ from the frozen descriptor")
    issues = value_issues(frame)
    if issues:
        raise ValueError(f"Invalid training export: {issues}")
    frame.attrs.update(
        partition="train",
        dataset_sha256=descriptor["dataset_sha256"],
        split_manifest_sha256=descriptor["split_manifest_sha256"],
        training_sha256=descriptor["training_sha256"],
    )
    return frame
