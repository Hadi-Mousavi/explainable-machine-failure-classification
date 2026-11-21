# Explainable Machine Learning for Predictive Maintenance

Industrial failure labels are often rare. This study asks how model choice,
imbalance handling, and decision thresholds affect rare-failure detection and
how simulator rules shape the resulting explanations.

The task is binary observation-level classification on the synthetic
[UCI AI4I 2020 dataset](https://doi.org/10.24432/C5HS5C), comprising 10,000
observations with 339 labeled failures. The source lacks verified timestamps and
equipment identities, so its findings are limited to the simulator and
observation-level classification.

## Research questions

1. How do prespecified models compare at detecting rare failures?
2. How do imbalance handling and validation-selected thresholds affect precision and recall?
3. What do model explanations and physics-informed feature ablations reveal about behavior?
4. How sensitive are results to the split design, including a split ordered by UDI?

The [research protocol](docs/research_protocol.md) specifies the later training
cross-validation, validation-only threshold selection, and one final held-out
test evaluation. Day 2 establishes fixed baselines. Day 3 compares those
baselines with two tree families on the same frozen training folds.

## Day 1 results

Day 1 covers original-byte acquisition, a full-source integrity audit, a frozen
60/20/20 train/validation/test split, and training-only exploratory analysis.
Both split stages stratify by the target, use seed 42, and sort by UDI before
sampling. The 6,000 training observations contain 203 labeled failures (3.38%).

Analyses use Type, air and process temperature, rotational speed, torque, and
tool wear. UDI identifies rows. UDI, Product ID, and all five failure-mode
indicators are excluded from predictors. The exploratory analysis reads only the
checksum-verified training export. See the [leakage controls](docs/leakage_risks.md).

## Day 2 baseline results

On the frozen training partition, the prior-only dummy classifier had mean
fold Average Precision (AP) of 0.033833. Unweighted logistic regression with
fixed settings reached 0.458571. The sample SDs across five folds were
0.000456 and 0.042855, respectively. Each fold contributed equally.

AP weights precision by increases in recall as the failure-score cutoff varies.
For separate diagnostics we classified scores ≥ 0.5 as failures. Logistic
regression detected 41 of 203 failures and produced 15 false alarms. Precision
is the fraction of alerts that are failures. Recall is the fraction of failures
detected. F1 balances precision and recall, while F2 gives recall more weight.
Undefined ratios are recorded as zero.

All scores come from training rows excluded from the corresponding fit. Type is
one-hot encoded and numeric measurements are standardized within each fit.
The synthetic observations and unverified time and equipment structure limit
these findings to the simulator. Operational alert utility and industrial
generalization remain open questions. See the [Day 2 methods](docs/day2_baselines.md),
[fold measurements](results/day2/fold_metrics.csv), and [figures](results/day2/figures).

## Day 3 model selection

Balanced histogram gradient boosting ranked first among 27 configurations,
with mean five-fold Average Precision of **0.759824**. The best random forest
reached 0.747959. Fixed logistic regression reached 0.458571 with equal
weights and 0.439010 with balanced weights. The prior dummy scored 0.033833.
These results describe rankings of synthetic observations on the training folds
used for selection. Held-out performance and operational alert utility remain unknown.

Each tree search sampled six structures and evaluated both class weights for
each structure. Balancing reduced mean AP for all six forests and increased
it for three of six boosting structures. The leading boosting configurations
differed by 0.001266 AP. See the [selection rule and interpretation](docs/day3_model_selection.md),
[all 27 configurations and fold scores](results/day3/model_comparison.md),
and [three figures](results/day3/figures).

With the local frozen training export available, run the comparison and its
isolated replay from the repository root.

```sh
uv run --locked ai4i-pm compare-models
uv run --locked python scripts/verify_day3.py
```

## Reproduce Day 2

Use the approved Day 1 checkout with its local `data/processed/train.csv` and
frozen metadata. The baseline command verifies the input hashes before fitting.
The verifier repeats the run in a temporary directory containing only the approved
inputs and compares all artifact bytes within the same runtime.

```sh
uv run --locked ai4i-pm baselines
uv run --locked python scripts/verify_day2.py
```

## Reproduce Day 1

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run from
the repository root. The project pins Python 3.12.16 and uses a locked CPU-only
environment.

```sh
uv python install
uv sync --locked --group notebook
uv run --locked ai4i-pm acquire
uv run --locked ai4i-pm audit
uv run --locked ai4i-pm split
uv run --locked ai4i-pm eda
uv run --locked python scripts/execute_notebook.py
uv run --locked python scripts/verify_day1.py
```

The notebook group and execution command are optional. The verification script
repeats the generated outputs in the current runtime and checks row-order
invariance of the original partition assignments. Figure bytes may vary across
platforms. The archived [reproducibility evidence](results/day1/reproducibility.json)
records a local run.

Acquisition uses the [official UCI archive](https://archive.ics.uci.edu/static/public/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset.zip).
If automated retrieval is blocked, download that archive and run
```sh
uv run --locked ai4i-pm acquire --manual path/to/uci-download.zip
```
A fresh checkout restores the archive and CSV only when both match the recorded
SHA-256 digests. Subsequent runs verify the local bytes offline. The CLI also
accepts `--root PATH` before the subcommand when running from another directory.

## Repository layout

| Location | Contents |
| --- | --- |
| [src/ai4i_pm](src/ai4i_pm) | CLI and reusable acquisition, schema, audit, split, and analysis modules |
| [configs/day1.toml](configs/day1.toml) | Frozen split settings |
| [configs/day2.toml](configs/day2.toml) | Approved input digests, models, CV and diagnostic settings |
| [configs/day3.toml](configs/day3.toml) | Frozen Day 2 input digests, seeds, search spaces and paired sampling budget |
| [docs](docs) | Research protocol, dataset attribution, and leakage controls |
| [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) and [scripts](scripts) | Notebook, execution, and reproducibility verification |
| [results/day1](results/day1) | Provenance, integrity audit, frozen memberships, training descriptor, training summaries, and seven figures |
| [results/day2](results/day2) | Paired training-fold assignments, model specifications, metrics, OOF scores, curves, figures and execution hashes |
| [results/day3](results/day3) | Candidate and fold ledgers, weight contrasts, provisional specification, figures and execution hashes |
| [tests](tests) and [.github/workflows/ci.yml](.github/workflows/ci.yml) | Offline tests and automated quality checks |

Raw UCI files, the training CSV, and the executed notebook are local outputs
excluded from Git.

## Testing

The test suite uses artificial fixtures and does not require access to UCI or
the research dataset. Run the same lint, format, and test checks used in CI.

```sh
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
```

## License and citation

The software is available under the [MIT License](LICENSE). Cite it using
[CITATION.cff](CITATION.cff).

AI4I 2020 Predictive Maintenance Dataset [Dataset]. (2020). UCI Machine
Learning Repository. [DOI 10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C).
The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The audit, split manifests, and exploratory figures and summaries are derived
from the original dataset. See [dataset attribution and provenance](docs/dataset.md).
