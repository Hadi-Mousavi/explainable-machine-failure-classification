"""One held-out release. Invoke with the full, published pre-test commit SHA."""

import argparse
import csv
import hashlib
import io
import json
import platform
import subprocess
import tomllib
import traceback
from importlib.metadata import version

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.metrics import precision_recall_curve
from threadpoolctl import threadpool_limits

from ai4i_pm.evaluation import fold_average_precision, fold_metrics
from ai4i_pm.final_report import report_text
from ai4i_pm.final_validation import (
    CONFIG,
    OUT,
    PROTOCOL,
    ROOT,
    SOURCE,
    check_hash,
    create,
    load_frozen,
    membership_digest,
    memberships,
    software,
)
from ai4i_pm.models import positive_scores
from ai4i_pm.schema import PREDICTORS, SOURCE_COLUMNS, TARGET, TRAINING_COLUMNS, predictor_matrix
from ai4i_pm.schema import value_issues as schema_issues
from ai4i_pm.utils import json_bytes, read_json, sha256_bytes, sha256_file, utc_now
from ai4i_pm.validation_threshold import verify_decision

PARENT = "5fd17fe9862d9f50f892dbe9efca3eb2f3573079"
DECISION = "b4c4e51eb0c19e5b26da3060d3a71311dcbd51025322001df56b8188f323fed7"
APPROVAL = "f7c768a0c72410b98995cd5022dfb2e345ffa735aaae8c0f6aa275b1806b358c"
MODELS = {
    "primary": "b75e43e3095650adff6659e9f50f3e5d7646ed84760baec1bb45c4a4ec7daefb",
    "reference": "d3f1ee61633ca85778bf592ec9868a1168b93d956980b0141ab429a6c462022e",
}
PLAN = "configs/07b_execution_freeze.json"
THRESHOLD = 0.15
REPLICATES = 2000
SEED = 42
START = "test_started.json"
FAILED = "test_interrupted.json"


def verify_revision(root, revision):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    if (
        git("rev-parse", "HEAD") != revision
        or git("rev-parse", "refs/remotes/origin/main") != revision
        or git("rev-parse", "HEAD^") != PARENT
        or git("log", "-1", "--format=%s") != "test: prepare frozen held-out evaluation"
        or git("branch", "--show-current") != "main"
        or git("status", "--porcelain", "--untracked-files=all")
    ):
        raise ValueError("Evaluation requires the clean, published pre-test revision")


def prepare(root):
    folder = root / OUT
    check_hash(root / CONFIG, PROTOCOL)
    cfg = tomllib.loads((root / CONFIG).read_text(encoding="utf-8"))
    check_hash(folder / "approved_protocol_record.json", APPROVAL)
    review = read_json(folder / "approved_protocol_record.json")["review"]
    check_hash(root / review["path"], review["sha256"])
    check_hash(folder / "validation_decision.json", DECISION)
    seal = (folder / "validation_decision.sha256").read_text().split()
    if seal != [DECISION, "validation_decision.json"]:
        raise ValueError("Invalid independent decision seal")
    decision = read_json(folder / "validation_decision.json")
    check_hash(folder / "fit_manifest.json", decision["fit_manifest_sha256"])
    check_hash(folder / "validation_threshold_grid.csv", decision["grid_sha256"])
    with (folder / "validation_threshold_grid.csv").open(encoding="utf-8") as stream:
        grid = [{key: json.loads(v) for key, v in row.items()} for row in csv.DictReader(stream)]
    verify_decision(grid, decision)
    fit = read_json(folder / "fit_manifest.json")
    inp, ident = cfg["inputs"], cfg["identity"]
    for name in ("split", "training_descriptor", "training_export", "resolved_models"):
        check_hash(root / inp[name], ident[name + "_sha256"])
    split = read_json(root / inp["split"])
    parts = memberships(split["partitions"], ident["partition_rows"])
    source_hash = ident["source_csv_sha256_via_provenance"]
    plan = read_json(root / PLAN)
    for name, digest in plan["code_sha256"].items():
        check_hash(root / name, digest)
    check_hash(root / inp["environment_lock"], fit["lock_sha256"])
    environment = dict(
        software(root), matplotlib=version("matplotlib"), platform=platform.platform()
    )
    if not (
        decision["protocol_sha256"] == fit["protocol_sha256"] == PROTOCOL
        and decision["primary_model_sha256"] == MODELS["primary"]
        and decision["selected"]["threshold"] == THRESHOLD
        and decision["split_sha256"] == ident["split_sha256"]
        and fit["inputs"] == ident
        and decision["validation_udi_sha256"] == membership_digest(parts["validation"])
        and fit["training_udi_sha256"] == membership_digest(parts["train"])
        and split["dataset_sha256"]
        == read_json(root / inp["provenance"])["csv_sha256"]
        == source_hash
        and plan["environment"] == environment
        and all(environment[k] == v for k, v in fit["versions"].items())
        and plan["protocol_sha256"] == PROTOCOL
        and plan["decision_sha256"] == DECISION
        and plan["pipelines_sha256"] == MODELS
        and plan["bootstrap"] == {"seed": SEED, "replicates": REPLICATES}
        and plan["outputs"] == [START, *cfg["planned_outputs"]["after_test"]]
    ):
        raise ValueError("Inconsistent pre-test freeze")
    specs = read_json(root / inp["resolved_models"])["models"]
    models = {}
    for role, digest in MODELS.items():
        system = cfg["systems"][role]
        spec = specs[system["spec_pointer"].split("/")[-1]]
        record = fit["systems"][role]
        if (
            record["sha256"] != digest
            or any(
                spec[k] != record[k] for k in ("candidate", "condition", "features", "parameters")
            )
            or spec["candidate"] != system["candidate"]
            or spec["condition"] != system["feature_set"]
            or spec["features"] != cfg["features"][system["feature_set"]]
            or record["input_features"] != list(PREDICTORS)
        ):
            raise ValueError("Fitted specification differs from protocol")
        model = load_frozen(folder / f"{role}_pipeline.joblib", digest)
        if (
            model.named_steps["classifier"].get_params(deep=False) != spec["parameters"]
            or list(model.named_steps["allowlist"].kw_args["columns"]) != spec["features"]
        ):
            raise ValueError("Persisted estimator or feature map differs")
        models[role] = model
    return cfg, parts, models, decision, environment


def release(source, ids, expected_hash):
    """Verify and project a single snapshot. Decode allowed fields only for released UDIs."""
    content = source.read_bytes()
    if sha256_bytes(content) != expected_hash:
        raise ValueError("Source SHA-256 mismatch at release")
    stream = io.BytesIO(content)
    if stream.readline().decode("utf-8-sig").rstrip("\r\n").split(",") != list(SOURCE_COLUMNS):
        raise ValueError("Unexpected source schema")
    wanted = set(ids)
    projected = io.BytesIO((",".join(TRAINING_COLUMNS) + "\n").encode())
    projected.seek(0, 2)
    for line in stream:
        key, _, rest = line.partition(b",")
        if int(key) in wanted:
            fields = rest.rstrip(b"\r\n").split(b",")
            if len(fields) != len(SOURCE_COLUMNS) - 1:
                raise ValueError("Unexpected field count")
            projected.write(b",".join([key, *fields[1 : 2 + len(PREDICTORS)]]) + b"\n")
    projected.seek(0)
    frame = pd.read_csv(projected).sort_values("UDI").reset_index(drop=True)
    if (
        frame.UDI.tolist() != ids
        or len(wanted) != len(ids)
        or tuple(frame.columns) != TRAINING_COLUMNS
        or schema_issues(frame)
        or set(frame[TARGET]) != {0, 1}
    ):
        raise ValueError("Invalid test membership, feature values or class support")
    return frame


def bootstrap(labels, scores, point_ap, point_recall):
    pos, neg = (np.flatnonzero(labels == k) for k in (1, 0))
    rng = np.random.Generator(np.random.PCG64(SEED))
    draws = np.empty((REPLICATES, 2))
    for i in range(REPLICATES):
        p = rng.choice(pos, len(pos), replace=True)
        n = rng.choice(neg, len(neg), replace=True)
        indices = np.concatenate((p, n))
        draws[i] = (
            fold_average_precision(labels[indices], scores[indices]),
            np.mean(scores[p] >= THRESHOLD),
        )
    low, high = np.quantile(draws, [0.025, 0.975], axis=0, method="linear")
    return dict(
        seed=SEED,
        replicates=REPLICATES,
        generator="NumPy PCG64",
        method="Within-class pairs with replacement, ascending UDI, positive then negative",
        confidence=0.95,
        quantiles=[0.025, 0.975],
        quantile_method="linear",
        positives=len(pos),
        negatives=len(neg),
        threshold=THRESHOLD,
        average_precision=dict(estimate=point_ap, lower=low[0], upper=high[0]),
        recall=dict(estimate=point_recall, lower=low[1], upper=high[1]),
    )


def operating_point(labels, scores, threshold):
    result = fold_metrics(
        labels, scores, {"positive_class": 1, "diagnostic_threshold": threshold, "zero_division": 0}
    )
    ap = result.pop("average_precision")
    result["threshold"] = result.pop("diagnostic_threshold")
    return ap, result


def save_curve(folder, labels, scores, prevalence):
    precision, recall, cutoffs = precision_recall_curve(
        labels, scores, pos_label=1, drop_intermediate=False
    )
    data = io.StringIO(newline="")
    writer = csv.writer(data, lineterminator="\n")
    writer.writerow(["precision", "recall", "threshold"])
    writer.writerows(zip(precision, recall, [*cutoffs, None], strict=True))
    create(folder / "primary_pr_curve.csv", data.getvalue().encode())
    figure = Figure(figsize=(6.4, 4.8), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    ax.step(recall, precision, where="post", label="RF-11 + A4")
    ax.axhline(prevalence, color="gray", linestyle="--", label="Test prevalence")
    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        title="Held-out precision–recall curve",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.legend()
    with (folder / "primary_pr_curve.png").open("xb") as stream:
        figure.savefig(stream, format="png", dpi=150)


def evaluate(root, revision):
    verify_revision(root, revision)
    cfg, parts, models, decision, environment = prepare(root)
    folder = root / OUT
    outputs = cfg["planned_outputs"]["after_test"]
    if any((folder / name).exists() for name in [START, FAILED, *outputs]):
        raise ValueError("Existing test access or results prohibit a new release")
    source_hash = cfg["identity"]["source_csv_sha256_via_provenance"]
    record = dict(
        revision=revision,
        source_sha256=source_hash,
        split_sha256=decision["split_sha256"],
        test_udi_sha256=membership_digest(parts["test"]),
        reserved_utc=utc_now(),
    )
    create(folder / START, json_bytes(record))
    events = []

    def event(name):
        events.append({"event": name, "utc": utc_now()})

    try:
        event("release started")
        frame = release(root / SOURCE, parts["test"], source_hash)
        event("test membership, schema and class support verified")
        features, labels = predictor_matrix(frame), frame[TARGET].to_numpy()
        scores = {}
        with threadpool_limits(limits=1):
            for role, model in models.items():
                event(f"{role} predict_proba started")
                scores[role] = np.asarray(positive_scores(model, features), dtype="<f8")
                if scores[role].shape != labels.shape:
                    raise ValueError("Score and label shapes differ")
                scores[role].setflags(write=False)
                event(f"{role} predict_proba completed")
        ap, selected = operating_point(labels, scores["primary"], THRESHOLD)
        _, diagnostic = operating_point(labels, scores["primary"], cfg["validation"]["diagnostic"])
        ref_ap, reference = operating_point(
            labels, scores["reference"], cfg["systems"]["reference"]["fixed_threshold"]
        )
        metrics = dict(
            rows=len(labels),
            positives=int(labels.sum()),
            negatives=int((labels == 0).sum()),
            prevalence=float(labels.mean()),
            definitions=cfg["metrics"],
            primary=dict(
                system="RF-11 + A4",
                average_precision=ap,
                frozen=selected,
                diagnostic_050=diagnostic,
            ),
            reference=dict(system="HGB-10 + A0", average_precision=ref_ap, fixed_050=reference),
        )
        create(folder / "test_metrics.json", json_bytes(metrics))
        save_curve(folder, labels, scores["primary"], metrics["prevalence"])
        intervals = bootstrap(labels, scores["primary"], ap, selected["recall"])
        create(folder / "bootstrap_intervals.json", json_bytes(intervals))
        create(
            folder / "final_report.md", report_text(metrics, intervals, decision, revision).encode()
        )
        event("metrics, PR curve, intervals and report written")
        manifest = dict(
            **record,
            events=events,
            environment=environment,
            native_threads=1,
            phase_07a_commit=PARENT,
            protocol_sha256=PROTOCOL,
            approval_sha256=APPROVAL,
            decision_sha256=DECISION,
            pipelines_sha256=MODELS,
            fit_manifest_sha256=decision["fit_manifest_sha256"],
            plan_sha256=sha256_file(root / PLAN),
            positives=metrics["positives"],
            rows=len(labels),
            negatives=metrics["negatives"],
            score_vector_encoding="ascending UDI, little-endian float64",
            scores_sha256={k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in scores.items()},
            output_sha256={
                name: sha256_file(folder / name)
                for name in [START, *outputs]
                if name != "execution_manifest.json"
            },
            deviations=[],
            completed_utc=utc_now(),
        )
        create(folder / "execution_manifest.json", json_bytes(manifest))
    except BaseException:
        create(
            folder / FAILED,
            json_bytes(
                dict(
                    **record, events=events, failed_utc=utc_now(), traceback=traceback.format_exc()
                )
            ),
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("published_pretest_revision")
    evaluate(ROOT, parser.parse_args().published_pretest_revision)
