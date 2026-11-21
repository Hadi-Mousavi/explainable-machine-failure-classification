"""Run Phase 07A with python -m ai4i_pm.final_validation fit|validate.

Validation takes the SHA-256 printed by the trusted local fit. A started stage
is reserved on disk before access and cannot be replayed after interruption.
"""

import argparse
import csv
import hashlib
import io
import json
import mmap
import platform
import subprocess
import tomllib
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ai4i_pm.ablations import ESTIMATORS, fixed_specifications
from ai4i_pm.feature_engineering import ablation_pipeline
from ai4i_pm.models import positive_scores
from ai4i_pm.schema import (
    PREDICTORS,
    SOURCE_COLUMNS,
    TARGET,
    TRAINING_COLUMNS,
    predictor_matrix,
    value_issues,
)
from ai4i_pm.splits import load_training
from ai4i_pm.utils import (
    environment_versions,
    json_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
    utc_now,
)
from ai4i_pm.validation_threshold import decision_from_grid, threshold_grid, verify_decision

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("results/07_final_evaluation")
CONFIG = "configs/07_final_evaluation.toml"
COMMIT = "7c2232abf1018e45cf9b4d7ff3d080fb069a0e38"
PROTOCOL = "8e08ffcd2545b625b8e6822a4cafddbcd8c564f2772e4702e780262a7c12df90"
SOURCE = Path("data/raw/ai4i2020.csv")


def create(path, data):
    with path.open("xb") as stream:
        stream.write(data)


def check_hash(path, expected):
    if sha256_file(path) != expected:
        raise ValueError(f"SHA-256 mismatch for {path.name}")


def membership_digest(ids):
    return sha256_bytes(json_bytes(sorted(ids)))


def memberships(parts, sizes):
    if set(parts) != set(sizes) or any(len(parts[k]) != n for k, n in sizes.items()):
        raise ValueError("Unexpected split partitions or sizes")
    all_ids = [i for ids in parts.values() for i in ids]
    if any(type(i) is not int for i in all_ids) or set(all_ids) != set(range(1, len(all_ids) + 1)):
        raise ValueError("Duplicate, overlapping or incomplete UDI membership")
    return {key: sorted(ids) for key, ids in parts.items()}


def approved_inputs(root):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != COMMIT:
        raise ValueError("Unexpected research checkout")
    check_hash(root / CONFIG, PROTOCOL)
    cfg = tomllib.loads((root / CONFIG).read_text(encoding="utf-8"))
    inp, ident = cfg["inputs"], cfg["identity"]
    record = read_json(root / OUT / "approved_protocol_record.json")
    check_hash(
        root / OUT / "approved_protocol_record.json",
        "f7c768a0c72410b98995cd5022dfb2e345ffa735aaae8c0f6aa275b1806b358c",
    )
    check_hash(root / record["review"]["path"], record["review"]["sha256"])
    for key in ("split", "training_descriptor", "training_export", "resolved_models"):
        check_hash(root / inp[key], ident[key + "_sha256"])
    with (root / SOURCE).open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    provenance = read_json(root / inp["provenance"])
    manifest = read_json(root / inp["split"])
    descriptor = read_json(root / inp["training_descriptor"])
    if not (
        source_hash
        == ident["source_csv_sha256_via_provenance"]
        == provenance["csv_sha256"]
        == manifest["dataset_sha256"]
        == descriptor["dataset_sha256"]
    ):
        raise ValueError("Source identity mismatch")
    parts = memberships(manifest["partitions"], ident["partition_rows"])
    if parts["train"] != descriptor["row_ids"]:
        raise ValueError("Training descriptor membership mismatch")
    return cfg, parts


def software(root):
    versions = environment_versions()
    versions.update({n: version(n) for n in ("scipy", "joblib", "threadpoolctl")})
    lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    locked = {p["name"]: p["version"] for p in lock["package"]}
    locked["python"] = (root / ".python-version").read_text().strip()
    if any(locked[k] != v for k, v in versions.items()):
        raise ValueError("Execution environment differs from the lock")
    return versions


def restore_pipeline(spec, parameters, features):
    estimator = ESTIMATORS[spec["candidate"]](**spec["parameters"])
    if (
        estimator.get_params(deep=False) != spec["parameters"]
        or spec["parameters"] != parameters
        or spec["features"] != features
    ):
        raise ValueError("Incomplete or inconsistent model specification")
    return ablation_pipeline(estimator, features)


def fit_training(pipeline, frame, ids, positives):
    if (
        frame.attrs.get("partition") != "train"
        or tuple(frame.columns) != TRAINING_COLUMNS
        or frame["UDI"].tolist() != ids
        or set(frame[TARGET]) != {0, 1}
        or frame[TARGET].sum() != positives
    ):
        raise ValueError("Training-only fit requires exact authorized rows and support")
    return pipeline.fit(predictor_matrix(frame), frame[TARGET])


def load_frozen(path, expected):
    # Deserialize the very bytes verified against the trusted fit record.
    content = path.read_bytes()
    if sha256_bytes(content) != expected:
        raise ValueError("Fitted artifact SHA-256 mismatch")
    model = joblib.load(io.BytesIO(content))
    if model.classes_.tolist() != [0, 1]:
        raise ValueError("Unexpected fitted class labels")
    check_hash(path, expected)
    return model


def reserve_stage(folder, stage):
    if stage == "fit" and set(p.name for p in folder.iterdir()) != {
        "approved_protocol_record.json"
    }:
        raise ValueError("Existing execution artifacts prohibit a fit")
    if stage == "validate" and (folder / "validation_decision.json").exists():
        raise ValueError("Validation decision already frozen")
    create(folder / f"{stage}_started.json", json_bytes({"started_at_utc": utc_now()}))


def fit(root):
    cfg, parts = approved_inputs(root)
    inp, ident = cfg["inputs"], cfg["identity"]
    versions = software(root)
    record = read_json(root / inp["resolved_models"])
    for name, digest in record["protocol"]["inputs_sha256"].items():
        check_hash(root / name, digest)
    prior = fixed_specifications(root)
    pipes, specs = {}, {}
    for role, system in cfg["systems"].items():
        spec = record["models"][system["spec_pointer"].split("/")[-1]]
        if (spec["candidate"], spec["condition"]) != (system["candidate"], system["feature_set"]):
            raise ValueError("Unexpected candidate or feature set")
        pipes[role] = restore_pipeline(
            spec, prior[spec["candidate"]], cfg["features"][spec["condition"]]
        )
        specs[role] = spec.copy()
    frame = load_training(root)
    folder = root / OUT
    reserve_stage(folder, "fit")
    private = folder / "private"
    private.mkdir()
    create(private / "train.csv", (root / inp["training_export"]).read_bytes())
    check_hash(private / "train.csv", ident["training_export_sha256"])
    with threadpool_limits(limits=1):
        for role, pipeline in pipes.items():
            fit_training(pipeline, frame, parts["train"], 203)
            path = folder / f"{role}_pipeline.joblib"
            with path.open("xb") as stream:
                joblib.dump(pipeline, stream)
            digest = sha256_file(path)
            restored = load_frozen(path, digest)
            if (
                restored.named_steps["classifier"].get_params(deep=False)
                != specs[role]["parameters"]
            ):
                raise ValueError("Serialized estimator differs from the approved map")
            encoder = restored.named_steps["features"].named_transformers_["type"]
            specs[role].update(
                sha256=digest,
                file=path.name,
                classes=restored.classes_.tolist(),
                positive_column=int(np.flatnonzero(restored.classes_ == 1).item()),
                input_features=list(PREDICTORS),
                type_categories=encoder.categories_[0].tolist(),
                preprocessing=cfg["fit"]["preprocessing"],
                estimator=cfg["systems"][role]["estimator"],
            )
    manifest = dict(
        protocol_sha256=PROTOCOL,
        repository_commit=COMMIT,
        inputs=ident,
        systems=specs,
        rows=len(frame),
        positives=int(frame[TARGET].sum()),
        training_udi_sha256=membership_digest(parts["train"]),
        versions=versions,
        native_threads=1,
        estimator_seed=42,
        platform=platform.platform(),
        workflow_sha256=implementation_hashes(),
        completed_at_utc=utc_now(),
        lock_sha256=sha256_file(root / "uv.lock"),
        started=read_json(folder / "fit_started.json"),
    )
    path = folder / "fit_manifest.json"
    create(path, json_bytes(manifest))
    return sha256_file(path)


def implementation_hashes():
    return {
        n: sha256_file(Path(__file__).with_name(n))
        for n in ("final_validation.py", "validation_threshold.py")
    }


def extract_validation(stream, parts, partition="validation"):
    """Index the frozen unquoted, one-record-per-line CSV by UDI before slicing values.

    Scan delimiters in the source mapping. Only copy allowed fields on released
    UDIs. Other rows, product identifiers and failure-mode values stay opaque.
    """
    if partition != "validation":
        raise ValueError("Only validation extraction is authorized")
    wanted = set(parts["validation"])
    if wanted.intersection(parts["test"]) or wanted.intersection(parts["train"]):
        raise ValueError("Validation membership overlap")
    projected = io.BytesIO()
    projected.write((",".join(TRAINING_COLUMNS) + "\n").encode())
    with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        header = data.readline().decode("utf-8-sig").rstrip("\r\n")
        if header != ",".join(SOURCE_COLUMNS):
            raise ValueError("Unexpected source schema")
        start = data.tell()
        while start < len(data):
            end = data.find(b"\n", start)
            if end < 0:
                end = len(data)
            sep = data.find(b",", start, end)
            if sep < 0:
                raise ValueError("Missing UDI delimiter")
            key = data[start:sep]
            if int(key) in wanted:
                first = data.find(b",", sep + 1, end)
                if first < 0:
                    raise ValueError("Missing product identifier delimiter")
                last = first
                for _ in range(len(PREDICTORS) + 1):
                    last = data.find(b",", last + 1, end)
                    if last < 0:
                        raise ValueError("Incomplete validation record")
                projected.write(key + b"," + data[first + 1 : last] + b"\n")
            start = end + 1
    projected.seek(0)
    frame = pd.read_csv(projected).sort_values("UDI").reset_index(drop=True)
    if frame["UDI"].tolist() != sorted(wanted) or value_issues(frame):
        raise ValueError("Invalid validation membership or values")
    return frame


def validate(root, fit_digest):
    cfg, parts = approved_inputs(root)
    folder = root / OUT
    fit_path = folder / "fit_manifest.json"
    check_hash(fit_path, fit_digest)
    manifest = read_json(fit_path)
    versions = software(root)
    if (
        manifest["protocol_sha256"] != PROTOCOL
        or manifest["versions"] != versions
        or manifest["workflow_sha256"] != implementation_hashes()
        or manifest["training_udi_sha256"] != membership_digest(parts["train"])
    ):
        raise ValueError("Frozen fit inputs differ")
    for role, spec in manifest["systems"].items():
        check_hash(folder / f"{role}_pipeline.joblib", spec["sha256"])
    digest = manifest["systems"]["primary"]["sha256"]
    primary = load_frozen(folder / "primary_pipeline.joblib", digest)
    private = folder / "private"
    reserve_stage(folder, "validate")
    with (root / SOURCE).open("rb") as source:
        if (
            hashlib.file_digest(source, "sha256").hexdigest()
            != cfg["identity"]["source_csv_sha256_via_provenance"]
        ):
            raise ValueError("Changed source at extraction boundary")
        frame = extract_validation(source, parts)
    labels = frame[TARGET].to_numpy()
    if len(frame) != 2000 or set(labels) != {0, 1} or labels.sum() != 68:
        raise ValueError("Unexpected validation support; stop for review")
    val_path = private / "validation.csv"
    create(val_path, frame.to_csv(index=False, lineterminator="\n").encode())
    with threadpool_limits(limits=1):
        scores = np.asarray(positive_scores(primary, predictor_matrix(frame)), dtype="<f8")
    score_path = private / "validation_scores.npy"
    with score_path.open("xb") as stream:
        np.save(stream, scores, allow_pickle=False)
    rows = threshold_grid(labels, scores, cfg["validation"]["grid"])
    grid_path = folder / "validation_threshold_grid.csv"
    with grid_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    decision = decision_from_grid(rows)
    decision.update(
        protocol_sha256=PROTOCOL,
        primary_model_sha256=digest,
        fit_manifest_sha256=fit_digest,
        split_sha256=cfg["identity"]["split_sha256"],
        validation_udi_sha256=membership_digest(parts["validation"]),
        score_vector_sha256=sha256_bytes(scores.tobytes()),
        score_encoding="Ascending numeric UDI; little-endian IEEE binary64, C order",
        scores_file_sha256=sha256_file(score_path),
        validation_export_sha256=sha256_file(val_path),
        grid_sha256=sha256_file(grid_path),
        versions=versions,
        completed_at_utc=utc_now(),
    )
    decision_path = folder / "validation_decision.json"
    create(decision_path, json_bytes(decision))
    with grid_path.open(encoding="utf-8") as stream:
        saved = [{k: json.loads(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    verify_decision(saved, read_json(decision_path))
    digest = sha256_file(decision_path)
    create(folder / "validation_decision.sha256", f"{digest}  {decision_path.name}\n".encode())
    return digest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    commands.add_parser("fit")
    commands.add_parser("validate").add_argument("trusted_fit_sha256")
    args = parser.parse_args()
    print(fit(ROOT) if args.stage == "fit" else validate(ROOT, args.trusted_fit_sha256))
