# Explainable Machine Learning for Equipment Failure Classification

How well do measurements from a single operation rank the likelihood of a machine-failure label? This independent study compares classifiers and physics-informed features on the synthetic AI4I 2020 dataset. Histogram gradient boosting reached **0.759824 mean average precision (AP)** on the original predictors in training cross-validation. Both tree families improved when given two quantities used in the simulator’s failure rules. Held-out performance and transfer to real equipment remain unknown.

## Research questions

- With rare failures, how much does each model improve on a prevalence-only ranking? How does class weighting affect that ranking?
- How much does tool wear contribute, and how do rankings change when temperature difference and mechanical power are supplied directly?
- How do model explanations relate to the generator, and how sensitive are the model and feature comparisons to alternative training partitions and UDI order? The precision–recall trade-off of a frozen alert rule on held-out observations remains open.

## Dataset and methodology

The source contains 10,000 synthetic observations and 339 machine-failure labels. Predictors are product type, air and process temperatures, rotational speed, torque, and tool wear. UDI identifies rows. UDI, Product ID, and the five failure-mode indicators are excluded from predictors. The source lacks verified equipment identities, timestamps, and a forecast horizon. [Dataset and provenance](docs/dataset.md) · [Leakage controls](docs/leakage_risks.md)

The stratified 60/20/20 primary split assigns 6,000 rows to training, including 203 failures. Baselines, model selection, and ablations share five stratified folds within those rows. Preprocessing is fitted within each fold. Mean AP gives each evaluation fold equal weight and measures how well continuous scores rank failures. The protocol reserves validation for threshold selection and the test partition for one final evaluation. [Research protocol](docs/research_protocol.md)

## Experimental results

| Training-fold comparison | Mean AP | Interpretation |
| --- | ---: | --- |
| Prior-only dummy / fixed logistic regression | 0.033833 / 0.458571 | Reference rankings with six original predictors |
| Histogram gradient boosting (HGB-10) / random forest (RF-11) | 0.759824 / 0.747959 | Leading configurations in their respective families. HGB-10 is the provisional selection. |
| HGB-10 / RF-11 with tool wear removed | 0.575906 / 0.607668 | Both lost AP on every paired fold. |
| HGB-10 / RF-11 with temperature difference and mechanical power added | 0.848176 / 0.869823 | Highest ablation means for both fixed models. Both gained on every paired fold. |

Each engineered feature also improved AP individually. Temperature difference enters the simulator’s heat-dissipation rule, and mechanical power enters its power-failure rule. The gains describe how readily models recover structure in synthetic labels. The highest forest result is exploratory and shares the folds used for selection. [Paired results](results/04_feature_ablation/ablation_summary.md) · [UCI generator description](https://doi.org/10.24432/C5HS5C)

## Research phases

| Completed phase | Evidence |
| --- | --- |
| [01 · Data exploration](research/01_data_exploration/README.md) | Integrity audit, partition membership, class imbalance, and training distributions |
| [02 · Baseline evaluation](research/02_baseline_evaluation/README.md) | Cross-fitted predictions, precision–recall curves, and diagnostic thresholds |
| [03 · Model selection](research/03_model_selection/README.md) | Candidate rankings, matched class-weight comparisons, and provisional selection |
| [04 · Feature ablation](research/04_feature_ablation/README.md) | Paired fold differences for six prespecified feature sets |
| [05 · Explainability](research/05_explainability/README.md) | Out-of-fold permutation results for four configurations, scale-checked SHAP and 16 diagnostic case explanations |
| [06 · Robustness analysis](research/06_robustness_analysis/README.md) | Two alternative CV assignments, the frozen seed-42 reference, and three expanding UDI-ordered evaluations |

Torque produced the largest mean AP loss under permutation in all four configurations. Tool wear had the largest mean absolute SHAP contribution in both models examined on fold 1. Only four failures appeared in the unconditional 300-row SHAP sample. [Explanation results and limitations](docs/05_explainability.md)

Mean AP ranks agreed across three training CV seeds. Under UDI ordering, RF overtook HGB on the original features (.568382 versus .522726). With both engineered features, ordered means were .738921 for RF and .710596 for HGB. Ordered evaluation prevalence ranged from 1.83% to 8.17%. UDI has no verified timestamp or equipment identity. [Robustness results and limitations](docs/06_robustness_analysis.md)

**Phase 07A freeze.** The primary system is RF-11 with A4. Its frozen cutoff of 0.15 was the unique F2 maximum across 19 approved validation thresholds. It detected 61 of 68 failures among 2,000 validation observations, with 32 false positives and F2 of 61/73. HGB-10 with A0 is the descriptive reference at 0.50. The fit manifest identifies both local fitted pipelines by SHA-256. [Approved protocol](docs/07_final_evaluation_protocol.md) · [Fit manifest](results/07_final_evaluation/fit_manifest.json) · [Validation decision](results/07_final_evaluation/validation_decision.json) · [Threshold grid](results/07_final_evaluation/validation_threshold_grid.csv)

## Installation and reproducibility

From the project root, install the pinned Python and locked environment with uv. The offline suite uses artificial training fixtures and verifies the published artifact inventory. [uv installation](https://docs.astral.sh/uv/getting-started/installation/)

```sh
uv python install
uv sync --locked
uv run --locked ai4i-pm --help
uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked ruff format --check .
```

The source archive, raw CSV, and training export are local files excluded from Git. For a fresh reproduction checkout, restore the source and training export using the first phase’s instructions. [Data preparation and optional notebook execution](research/01_data_exploration/README.md)

### Replay the published training studies

With the matching training export available, each verifier copies approved inputs into a temporary directory. The baseline verifier checks out-of-fold identities and artifact bytes. The selection verifier compares scores and contrasts while allowing run times to differ. The ablation verifier checks paired results within an absolute tolerance of 1e-12. Figure bytes can depend on the runtime.

```sh
uv run --locked python scripts/verify_02_baselines.py
uv run --locked python scripts/verify_03_model_selection.py
uv run --locked python scripts/verify_04_feature_ablation.py
```

For Phase 05, install the hash-pinned SHAP dependencies after the locked environment sync. The replay compares all 600 permutation measurements, SHAP coordinates and signed values, local decisions, and artifact digests.

```sh
uv pip install --require-hashes -r configs/05_shap_requirements.txt
uv run --locked ai4i-pm explain
uv run --locked python scripts/verify_05_explainability.py
```

With the same approved training export, run the robustness analysis and its independent 52-fit replay.

```sh
uv run --locked ai4i-pm robustness
uv run --locked python scripts/verify_06_robustness.py
```

## Repository architecture

| Location | Role |
| --- | --- |
| [research](research) | Scientific phase overviews and links to original evidence |
| [src/ai4i_pm](src/ai4i_pm) · [tests](tests) | Shared Python package, CLI, and fixture tests |
| [configs](configs) · [results](results) | Frozen designs, memberships, metrics, figures, and execution records |
| [docs](docs) · [notebooks](notebooks) | Original methodology, study reports, and exploration notebook |
| [scripts](scripts) · [CI workflow](.github/workflows/ci.yml) | Reproduction entry points and automated quality checks |
| [data](data) · [uv.lock](uv.lock) | Local data locations and locked software environment |

The Python package and CLI retain their dataset-specific names so existing installations and experiment commands remain usable. The migration ledger connects the approved artifacts to their current paths and SHA-256 digests. [Migration provenance](docs/migration/README.md)

## Attribution

AI4I 2020 Predictive Maintenance Dataset (2020), UCI Machine Learning Repository. The audit, figures, and experimental results derive from this dataset. [Dataset DOI](https://doi.org/10.24432/C5HS5C) · [CC BY 4.0 dataset license](https://creativecommons.org/licenses/by/4.0/)

The research software is available under the MIT License. [Software license](LICENSE) · [Software citation](CITATION.cff)
