"""Partition invariants and isolation tests on artificial offline data only."""

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from ai4i_pm.eda import generate_eda
from ai4i_pm.schema import IDENTIFIER, PROHIBITED, TRAINING_COLUMNS
from ai4i_pm.splits import (
    SplitSettings,
    build_manifest,
    freeze_split,
    load_training,
    validate_partitions,
)
from ai4i_pm.utils import read_json, sha256_file, write_json


def test_sizes_disjoint_complete_and_stratified(source_frame):
    manifest = build_manifest(source_frame, "a" * 64)
    partitions = manifest["partitions"]
    assert {key: len(ids) for key, ids in partitions.items()} == {
        "train": 60,
        "validation": 20,
        "test": 20,
    }
    sets = {key: set(ids) for key, ids in partitions.items()}
    assert not sets["train"] & sets["test"]
    assert not sets["train"] & sets["validation"]
    assert not sets["validation"] & sets["test"]
    assert set.union(*sets.values()) == set(source_frame[IDENTIFIER])
    for summary in manifest["validation"]["partitions"].values():
        assert summary["positive_fraction"] == 0.2


def test_deterministic_under_repetition_and_row_reordering(source_frame):
    expected = build_manifest(source_frame, "a" * 64)
    assert build_manifest(source_frame, "a" * 64) == expected
    assert build_manifest(source_frame.sample(frac=1, random_state=77), "a" * 64) == expected
    assert (
        build_manifest(source_frame, "a" * 64, SplitSettings(seed=43))["partitions"]
        != expected["partitions"]
    )


@pytest.mark.parametrize("corruption", ["overlap", "incomplete", "wrong_size"])
def test_bad_partitions_rejected(source_frame, corruption):
    parts = build_manifest(source_frame, "a" * 64)["partitions"]
    if corruption == "overlap":
        parts["test"][0] = parts["train"][0]
    elif corruption == "incomplete":
        parts["test"][0] = 99999
    else:
        parts["train"].append(parts["validation"].pop())
    with pytest.raises(ValueError):
        validate_partitions(source_frame, parts, SplitSettings())


def test_invalid_digest_and_settings(source_frame):
    with pytest.raises(ValueError, match="SHA-256"):
        build_manifest(source_frame, "invalid")
    with pytest.raises(ValueError):
        SplitSettings(test_fraction=0)
    with pytest.raises(ValueError):
        SplitSettings(seed=-1)


def test_frozen_manifest_repeat_and_override_history(research_root):
    first = freeze_split(research_root)
    path = research_root / "results/day1/split_manifest.json"
    old_hash = sha256_file(path)
    assert freeze_split(research_root) == first
    assert sha256_file(path) == old_hash
    new_settings = replace(SplitSettings(), seed=43)
    with pytest.raises(ValueError, match="Refusing regeneration"):
        freeze_split(research_root, new_settings)
    with pytest.raises(ValueError, match="nonempty"):
        freeze_split(research_root, new_settings, override=True)
    second = freeze_split(
        research_root, new_settings, override=True, reason="ARTIFICIAL TEST OVERRIDE"
    )
    history = research_root / "results/day1/split_history" / f"{old_hash}.json"
    assert read_json(history) == first
    assert second["partitions"] != first["partitions"]
    events = (research_root / "results/day1/split_overrides.jsonl").read_text()
    assert "ARTIFICIAL TEST OVERRIDE" in events and old_hash in events
    assert len(load_training(research_root)) == 6000


def test_split_linked_to_dataset_checksum(research_root):
    freeze_split(research_root)
    raw = research_root / "data/raw/ai4i2020.csv"
    with raw.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="checksum differs"):
        freeze_split(research_root)
    provenance = research_root / "results/day1/provenance.json"
    data = read_json(provenance)
    data["csv_sha256"] = sha256_file(raw)
    write_json(provenance, data)
    with pytest.raises(ValueError, match="Refusing regeneration"):
        freeze_split(research_root)


def test_training_export_excludes_heldout_rows_and_modes(research_root):
    manifest = freeze_split(research_root)
    train = load_training(research_root)
    assert tuple(train.columns) == TRAINING_COLUMNS
    assert set(train[IDENTIFIER]) == set(manifest["partitions"]["train"])
    assert not set(train[IDENTIFIER]) & set(manifest["partitions"]["test"])
    assert not set(train[IDENTIFIER]) & set(manifest["partitions"]["validation"])
    assert not set(train.columns) & (set(PROHIBITED) - {IDENTIFIER})


def test_eda_rejects_tampered_export_or_wrong_partition(research_root):
    freeze_split(research_root)
    csv = research_root / "data/processed/train.csv"
    original = csv.read_bytes()
    csv.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        generate_eda(research_root)
    csv.write_bytes(original)
    descriptor = research_root / "results/day1/training_partition.json"
    data = read_json(descriptor)
    data["partition"] = "test"
    write_json(descriptor, data)
    with pytest.raises(ValueError, match="training partition only"):
        generate_eda(research_root)


def test_training_descriptor_rejects_changed_manifest(research_root):
    freeze_split(research_root)
    manifest = research_root / "results/day1/split_manifest.json"
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="stale"):
        load_training(research_root)


def test_eda_reads_only_training_values(research_root, monkeypatch):
    manifest = freeze_split(research_root)
    actual_read_csv = pd.read_csv
    paths_read = []

    def training_only_csv(path, *args, **kwargs):
        paths_read.append(Path(path).relative_to(research_root).as_posix())
        assert Path(path) == research_root / "data/processed/train.csv"
        return actual_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", training_only_csv)
    summary = generate_eda(research_root)
    assert paths_read == ["data/processed/train.csv"]
    assert summary["rows"] == len(manifest["partitions"]["train"])
    assert summary["partition"] == "train"
    assert not summary["failure_modes_examined"]
    assert not summary["models_trained"]
    assert len(summary["figures"]) == 7
    for name in summary["figures"]:
        assert (research_root / "results/day1/figures" / name).is_file()
