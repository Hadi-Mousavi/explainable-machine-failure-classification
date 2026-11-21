"""Check the published experiment chain without downloading or fitting the dataset."""

import csv
import re
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

from ai4i_pm.ablations import CONFIG, FOLDS, SELECTED
from ai4i_pm.schema import PREDICTORS
from ai4i_pm.utils import read_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]
PHASES = (
    "01_data_exploration",
    "02_baseline_evaluation",
    "03_model_selection",
    "04_feature_ablation",
)


def test_migrated_inventory_and_execution_chains():
    with (ROOT / "docs/migration/artifacts.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    hashes = {row["path"]: row["sha256"] for row in rows}
    assert len(hashes) == len(rows)
    for name, digest in hashes.items():
        path = ROOT / name
        if name.startswith("data/") and not path.exists():
            continue  # The published provenance identifies these untracked local inputs.
        assert sha256_file(path) == digest, name
    for phase in PHASES:
        assert f"configs/{phase}.toml" in hashes
        assert (ROOT / "research" / phase / "README.md").is_file()
    for phase in PHASES[1:]:
        folder = f"results/{phase}"
        record = read_json(ROOT / folder / "execution.json")
        for name, digest in record["inputs_sha256"].items():
            assert hashes[name] == digest
        for name, digest in record["artifacts_sha256"].items():
            assert hashes[f"{folder}/{name}"] == digest
    first = ROOT / "results" / PHASES[0]
    repeat = read_json(first / "reproducibility.json")
    assert repeat["sha256_before"] == repeat["sha256_after"]
    assert all(hashes[name] == digest for name, digest in repeat["sha256_before"].items())
    assert hashes[FOLDS] == "1ddf0178f0561ec5b03f61931ba0db84a7e49a625eb7e4e9d36d137e1def1636"
    assert sha256_file(first / "split_manifest.json") == (
        "4ac9dea7352d9da8a9e7a52d241325e3e9df593aab1423c587e880d79b2fd310"
    )


def test_frozen_feature_design_and_model_references():
    protocol = tomllib.loads((ROOT / CONFIG).read_text(encoding="utf-8"))
    design = read_json(ROOT / "results" / PHASES[3] / "ablation_config.json")
    selected = read_json(ROOT / SELECTED)
    assert design["protocol"] == protocol
    assert design["configuration_sha256"] == sha256_file(ROOT / CONFIG)
    assert selected["candidate_id"] == protocol["primary_model"] == "hist_gradient_boosting-10"
    assert protocol["secondary_model"] == "random_forest-11"
    assert (
        selected["classifier_parameters"]
        == design["classifier_parameters"][selected["candidate_id"]]
    )
    assert selected["predictors"] == protocol["features"]["A0"] == list(PREDICTORS)
    assert selected["cv_manifest_sha256"] == sha256_file(ROOT / FOLDS)


def test_project_identity_and_phase_links():
    title = "Explainable Machine Learning for Equipment Failure Classification"
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert f'title: "{title}"' in citation
    repository = "https://github.com/Hadi-Mousavi/explainable-machine-failure-classification"
    assert f'repository-code: "{repository}"' in citation
    assert (ROOT / "README.md").read_text(encoding="utf-8").splitlines()[0] == f"# {title}"
    documents = [ROOT / "README.md"]
    for folder in ("docs", "research", "results"):
        documents.extend((ROOT / folder).rglob("*.md"))
    for source in documents:
        text = source.read_text(encoding="utf-8")
        for link in re.findall(r"\[[^\]]*\]\(([^)\s]+)\)", text):
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            target = source.parent / unquote(url.path) if url.path else source
            assert target.exists(), (source, link)
            if url.fragment:
                headings = re.findall(r"^#+\s+(.+)$", target.read_text(encoding="utf-8"), re.M)
                anchors = {re.sub(r"[^\w -]", "", h.lower()).replace(" ", "-") for h in headings}
                assert unquote(url.fragment) in anchors, (source, link)
