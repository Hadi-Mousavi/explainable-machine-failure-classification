"""Exercise the sealed evaluator using artificial labels, scores, models and CSV rows."""

import csv
import hashlib
import io
import shutil
import tomllib
from importlib.metadata import version

import joblib
import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from ai4i_pm import final_evaluation as ev
from ai4i_pm.feature_engineering import ablation_pipeline
from ai4i_pm.final_validation import ESTIMATORS
from ai4i_pm.schema import PREDICTORS, TARGET, predictor_matrix
from ai4i_pm.utils import read_json, sha256_file, write_json


def test_hand_calculated_metrics_curve_and_endpoint(tmp_path):
    labels = np.array([1, 0, 1, 0, 0])
    scores = np.array([0.8, 0.8, 0.15, 0.4, np.nextafter(0.15, 0)])
    ap, row = ev.operating_point(labels, scores, ev.THRESHOLD)
    assert ap == 0.5
    assert row == dict(
        threshold=0.15, tp=2, tn=1, fp=2, fn=0, precision=0.5, recall=1, f1=2 / 3, f2=5 / 6
    )
    ev.save_curve(tmp_path, labels, scores, 0.4)
    with (tmp_path / "primary_pr_curve.csv").open(newline="") as stream:
        curve = list(csv.DictReader(stream))
    assert [float(r["recall"]) for r in curve] == [1, 1, 0.5, 0.5, 0]
    assert [float(r["precision"]) for r in curve] == [0.4, 0.5, 1 / 3, 0.5, 1]
    assert [float(r["threshold"]) for r in curve[:-1]] == sorted(set(scores))
    assert curve[-1]["threshold"] == ""
    assert (tmp_path / "primary_pr_curve.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_bootstrap_draw_order_pairing_and_linear_percentiles(monkeypatch):
    labels = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
    scores = np.array([0.8, 0.1, 0.3, 0.25, 0.65, 0.7, 0.9, 0.2, 0.14, 0.07])
    pos, neg = np.flatnonzero(labels), np.flatnonzero(1 - labels)
    rng = np.random.Generator(np.random.PCG64(42))
    observed = []
    expected = []
    # Use an independent rank formula and explicit adjacent-order-statistic interpolation.
    for _ in range(2000):
        p, n = rng.choice(pos, 5, replace=True), rng.choice(neg, 5, replace=True)
        paired = scores[np.r_[p, n]]
        ap = np.mean([sum(paired[:5] >= v) / sum(paired >= v) for v in paired[:5]])
        expected.append((ap, np.mean(scores[p] >= 0.15), paired))
    original = ev.fold_average_precision

    def capture(y, s):
        np.testing.assert_array_equal(y, [1] * 5 + [0] * 5)
        observed.append(s.copy())
        return original(y, s)

    monkeypatch.setattr(ev, "fold_average_precision", capture)
    ap, point = ev.operating_point(labels, scores, 0.15)
    out = ev.bootstrap(labels, scores, ap, point["recall"])
    np.testing.assert_array_equal(observed, [r[2] for r in expected])
    for j, key in enumerate(["average_precision", "recall"]):
        ordered = sorted(r[j] for r in expected)
        for q, end in [(0.025, "lower"), (0.975, "upper")]:
            ix = (len(ordered) - 1) * q
            lo = int(ix)
            assert out[key][end] == pytest.approx(
                ordered[lo] + (ix - lo) * (ordered[lo + 1] - ordered[lo])
            )
    assert (
        out["average_precision"]["estimate"] == ap and out["recall"]["estimate"] == point["recall"]
    )
    assert (out["seed"], out["replicates"], out["positives"], out["negatives"]) == (42, 2000, 5, 5)


@pytest.fixture
def prepared(tmp_path, source_frame, monkeypatch):
    """Build a complete temporary freeze. Only approved aggregate templates are read."""
    root, folder = tmp_path, tmp_path / ev.OUT
    folder.mkdir(parents=True)
    cfg_text = (ev.ROOT / ev.CONFIG).read_text(encoding="utf-8")
    cfg = tomllib.loads(cfg_text)
    inp = cfg["inputs"]
    source = root / ev.SOURCE
    source.parent.mkdir(parents=True)
    source_frame.iloc[::-1].to_csv(source, index=False)
    parts = {
        "train": list(range(1, 61)),
        "validation": list(range(61, 81)),
        "test": list(range(81, 101)),
    }
    write_json(root / inp["split"], {"partitions": parts, "dataset_sha256": sha256_file(source)})
    write_json(root / inp["training_descriptor"], {"row_ids": parts["train"]})
    export = root / inp["training_export"]
    export.parent.mkdir(parents=True)
    train = source_frame.iloc[:60]
    train.to_csv(export, index=False)
    write_json(root / inp["provenance"], {"csv_sha256": sha256_file(source)})
    fit = read_json(ev.ROOT / ev.OUT / "fit_manifest.json")
    specs, hashes = {}, {}
    for role, system in cfg["systems"].items():
        info = fit["systems"][role]
        spec = {k: info[k] for k in ("candidate", "condition", "parameters", "features")}
        specs[system["spec_pointer"].split("/")[-1]] = spec
        estimator = ESTIMATORS[spec["candidate"]](**spec["parameters"])
        pipe = ablation_pipeline(estimator, spec["features"])
        with ev.threadpool_limits(1):
            pipe.fit(predictor_matrix(train), train[TARGET])
        path = folder / info["file"]
        joblib.dump(pipe, path)
        hashes[role] = info["sha256"] = sha256_file(path)
    write_json(root / inp["resolved_models"], {"models": specs})
    for name in ("split", "training_descriptor", "training_export", "resolved_models"):
        cfg_text = cfg_text.replace(
            cfg["identity"][name + "_sha256"], sha256_file(root / inp[name])
        )
    cfg_text = cfg_text.replace(
        cfg["identity"]["source_csv_sha256_via_provenance"], sha256_file(source)
    )
    cfg_text = cfg_text.replace(
        "train = 6000, validation = 2000, test = 2000", "train = 60, validation = 20, test = 20"
    )
    (root / ev.CONFIG).parent.mkdir(parents=True)
    (root / ev.CONFIG).write_text(cfg_text, encoding="utf-8")
    protocol = sha256_file(root / ev.CONFIG)
    write_json(
        folder / "approved_protocol_record.json",
        {"review": {"path": ev.CONFIG, "sha256": protocol}},
    )
    for file in ("uv.lock", ".python-version"):
        shutil.copyfile(ev.ROOT / file, root / file)
    fit.update(
        inputs=tomllib.loads(cfg_text)["identity"],
        protocol_sha256=protocol,
        training_udi_sha256=ev.membership_digest(parts["train"]),
    )
    write_json(folder / "fit_manifest.json", fit)
    # The threshold grid and its aggregate counts are public, frozen template metadata.
    shutil.copyfile(
        ev.ROOT / ev.OUT / "validation_threshold_grid.csv", folder / "validation_threshold_grid.csv"
    )
    d = read_json(ev.ROOT / ev.OUT / "validation_decision.json")
    d.update(
        protocol_sha256=protocol,
        fit_manifest_sha256=sha256_file(folder / "fit_manifest.json"),
        primary_model_sha256=hashes["primary"],
        split_sha256=sha256_file(root / inp["split"]),
        validation_udi_sha256=ev.membership_digest(parts["validation"]),
    )
    write_json(folder / "validation_decision.json", d)
    digest = sha256_file(folder / "validation_decision.json")
    (folder / "validation_decision.sha256").write_text(f"{digest}  validation_decision.json\n")
    monkeypatch.setattr(ev, "DECISION", digest)
    monkeypatch.setattr(ev, "PROTOCOL", protocol)
    monkeypatch.setattr(ev, "MODELS", hashes)
    monkeypatch.setattr(ev, "APPROVAL", sha256_file(folder / "approved_protocol_record.json"))
    monkeypatch.setattr(ev, "verify_revision", lambda root, revision: None)
    write_json(
        root / ev.PLAN,
        dict(
            code_sha256={},
            protocol_sha256=protocol,
            decision_sha256=digest,
            pipelines_sha256=hashes,
            bootstrap=dict(seed=42, replicates=2000),
            outputs=[ev.START, *cfg["planned_outputs"]["after_test"]],
            environment=dict(
                ev.software(root), matplotlib=version("matplotlib"), platform=ev.platform.platform()
            ),
        ),
    )
    return root


def test_complete_frozen_run_and_reuse_guard(prepared, monkeypatch):
    folder = prepared / ev.OUT
    original = Pipeline.predict_proba
    calls = []

    def score(model, x):
        assert tuple(x.columns) == PREDICTORS
        calls.append(type(model.named_steps["classifier"]).__name__)
        return original(model, x)

    def forbid(*args, **kwargs):
        pytest.fail("Unexpected fit during evaluation")

    monkeypatch.setattr(Pipeline, "predict_proba", score)
    monkeypatch.setattr(Pipeline, "fit", forbid)
    monkeypatch.setattr(Pipeline, "fit_transform", forbid)
    ev.evaluate(prepared, "artificial-revision")
    assert calls == ["RandomForestClassifier", "HistGradientBoostingClassifier"]
    metrics = read_json(folder / "test_metrics.json")
    assert metrics["primary"]["frozen"]["threshold"] == 0.15
    assert (
        metrics["reference"]["fixed_050"]["threshold"]
        == metrics["primary"]["diagnostic_050"]["threshold"]
        == 0.5
    )
    assert (metrics["rows"], metrics["positives"], metrics["negatives"]) == (20, 4, 16)
    manifest = read_json(folder / "execution_manifest.json")
    outputs = read_json(prepared / ev.PLAN)["outputs"]
    assert set(manifest["output_sha256"]) == set(outputs) - {"execution_manifest.json"}
    for name, h in manifest["output_sha256"].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == h
    assert len(manifest["scores_sha256"]) == 2 and not manifest["deviations"]
    before = {p: sha256_file(p) for p in folder.iterdir()}
    monkeypatch.setattr(ev, "release", forbid)
    with pytest.raises(ValueError, match="prohibit"):
        ev.evaluate(prepared, "artificial-revision")
    assert {p: sha256_file(p) for p in folder.iterdir()} == before


def test_integrity_failure_and_interrupted_run(prepared):
    folder = prepared / ev.OUT
    d = read_json(folder / "validation_decision.json")
    original = (folder / "validation_decision.json").read_bytes()
    d["selected"]["threshold"] = 0.25
    write_json(folder / "validation_decision.json", d)
    with pytest.raises(ValueError, match="SHA-256"):
        ev.evaluate(prepared, "fixture")
    assert not (folder / ev.START).exists()
    (folder / "validation_decision.json").write_bytes(original)
    with (prepared / ev.SOURCE).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Source SHA"):
        ev.evaluate(prepared, "fixture")
    assert (folder / ev.START).exists() and (folder / ev.FAILED).exists()
    with pytest.raises(ValueError, match="prohibit"):
        ev.evaluate(prepared, "fixture")
    assert not (folder / "test_metrics.json").exists()


def test_release_projects_only_unique_authorized_rows(tmp_path, source_frame):
    source = tmp_path / "fixture.csv"
    stream = io.BytesIO()
    source_frame.iloc[::-1].to_csv(stream, index=False)
    lines = stream.getvalue().splitlines()
    lines[-1] = b"1,\xff"
    source.write_bytes(b"\n".join(lines))
    ids = [2, 3, 4, 5]
    frame = ev.release(source, ids, sha256_file(source))
    assert frame.UDI.tolist() == ids
    assert tuple(frame.columns) == ev.TRAINING_COLUMNS
    for bad in ([2, 3, 4], [2, 2, 5], [2, 5, 101]):
        with pytest.raises(ValueError, match="membership"):
            ev.release(source, bad, sha256_file(source))
    with source.open("ab") as f:
        f.write(b"\n" + lines[-2])
    with pytest.raises(ValueError, match="membership"):
        ev.release(source, ids, sha256_file(source))


def test_revision_gate_rejects_changed_checkout(tmp_path, monkeypatch):
    state = {
        ("rev-parse", "HEAD"): "frozen",
        ("rev-parse", "refs/remotes/origin/main"): "frozen",
        ("rev-parse", "HEAD^"): ev.PARENT,
        ("log", "-1", "--format=%s"): "test: prepare frozen held-out evaluation",
        ("branch", "--show-current"): "main",
        ("status", "--porcelain", "--untracked-files=all"): "",
    }
    monkeypatch.setattr(ev.subprocess, "check_output", lambda cmd, **kwargs: state[tuple(cmd[1:])])
    ev.verify_revision(tmp_path, "frozen")
    for key, saved in list(state.items()):
        state[key] = "changed"
        with pytest.raises(ValueError, match="pre-test revision"):
            ev.verify_revision(tmp_path, "frozen")
        state[key] = saved
