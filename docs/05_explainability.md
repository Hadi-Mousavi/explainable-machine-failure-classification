# Phase 05 · Explaining equipment failure classifications

Shuffling torque reduced mean fold average precision (AP) more than shuffling any other original predictor in each of the four configurations. Tool wear had the largest mean absolute SHAP contribution in both models examined on fold 1. The two results answer different questions about the same classifiers.

## Evidence and design

The experiment starts from migration commit `1e8d63214da6cbac603f27ca583db79a24787eee`. It verifies the approved training export (6,000 observations, 203 failures), its ordered UDIs and the five frozen evaluation memberships. Each model fit used 4,800 observations and explained only observations from the other 1,200. The unperturbed APs of all 20 model-fold combinations exactly reproduced Phase 04. The only observation file in the replay directory was the training CSV. The loader verified the original dataset digest through the provenance record and hashed the split manifest as opaque bytes. [Protocol and input checksums](../configs/05_explainability.toml) · [Complete specifications and folds](../results/05_explainability/explanation_config.json)

A0 contains Type, air temperature, process temperature, rotational speed, torque and tool wear. A4 adds process minus air temperature (K) and torque × rotational speed × 2π/60 (W). Each fold learned its own Type encoding. The two classifiers kept their Phase 03 parameters, including HGB's balanced class weights and the forest's unweighted fit. HGB-10 + A0 retains its provisional selection status. RF-11 + A4 retains its exploratory status.

## Predictive reliance across five folds

| Configuration | Original AP¹ | Mean torque AP drop | SD across folds² | Mean within-fold repeat SD | Next largest mean drop |
| --- | ---: | ---: | ---: | ---: | --- |
| M1 · HGB-10 + A0 | 0.759824 | 0.54592 | 0.06297 | 0.02529 | Air temperature, 0.41142 |
| M2 · RF-11 + A0 | 0.747959 | 0.48402 | 0.08429 | 0.02385 | Rotational speed, 0.32157 |
| M3 · HGB-10 + A4 | 0.848176 | 0.64778 | 0.07104 | 0.03450 | Rotational speed, 0.52295 |
| M4 · RF-11 + A4 | 0.869823 | 0.75515 | 0.04743 | 0.01276 | Rotational speed, 0.55226 |

¹ Equal-weight mean of unperturbed fold AP. ² Sample SD of the five repeat-averaged drops.

Shuffling Type produced the smallest mean drop in every configuration (0.00876–0.02954). For both A0 models, its fold means included a small negative value. The five repeat seeds measure sensitivity to a particular permutation. Shared observations and overlapping fold fit sets preclude treating repeats as independent replications. [All 600 perturbations](../results/05_explainability/permutation_fold_results.csv) · [Fold summaries](../results/05_explainability/permutation_fold_summary.csv) · [Rankings and variation](../results/05_explainability/permutation_summary.csv) · [Permutation figure](../results/05_explainability/figures/permutation.png)

## Global attribution and feature effects

The fold-1 fit contained 163 failures and the evaluation fold contained 40. An unconditional draw without replacement selected 300 evaluation UDIs, including four failures. The same rows were used for both models. Tree SHAP used training path counts in the fitted trees as its background. This choice needs no separately sampled background observations. Type H, L and M remain separate fitted columns. Numeric columns retain their physical units on the feature-effect plots. [Exact sampled IDs and compatibility checks](../results/05_explainability/shap_metadata.json) · [TreeExplainer API](https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html)

| Check | M1 · HGB-10 + A0 | M4 · RF-11 + A4 |
| --- | --- | --- |
| Native SHAP shape on 8 preflight rows | 8 × 8 | 8 × 10 × 2 |
| Class-1 output verified | Decision margin in log-odds, sigmoid checked against probability | Probability, class axis index 1 in fitted order [0, 1] |
| Largest mean absolute attributions | Tool wear 0.8274, torque 0.8136, air temperature 0.5866 log-odds | Tool wear 0.0190, torque 0.0149, speed 0.0141 probability units |
| Largest absolute reconstruction error on global and local cases | 2.22 × 10⁻¹⁴ | 4.55 × 10⁻¹⁵ |

Each global ranking averages absolute contributions. Each effect figure plots the signed contribution against the observed value for that model's two largest contributors. Vertical spread at similar measurements is consistent with other predictors changing the contribution along a tree's decision path. The two vertical scales have different units. [Global ranking table](../results/05_explainability/shap_summary.csv) · [HGB magnitudes](../results/05_explainability/figures/M1_global.png) · [Forest magnitudes](../results/05_explainability/figures/M4_global.png) · [HGB effects](../results/05_explainability/figures/M1_effects.png) · [Forest effects](../results/05_explainability/figures/M4_effects.png)

In both effect plots, tool wear made its largest upward contributions near the upper end of observed wear, above roughly 200 minutes. M1 also assigned large positive torque contributions at both extremes of observed torque. These are descriptions of fitted responses within the sampled rows.

## Local decisions and errors

The fixed 0.5 diagnostic cutoff gave M1 33 true positives, 28 false positives, 7 false negatives and 1,132 true negatives. M4 had 29, 3, 11 and 1,157 respectively. The two smallest eligible UDIs from each category were selected independently for each model.

| Category | M1 UDIs | M4 UDIs |
| --- | --- | --- |
| True positive | 1168, 1497 | 1497, 2859 |
| False positive | 345, 760 | 4119, 5153 |
| False negative | 1764, 3530 | 1168, 1584 |
| True negative | 11, 19 | 11, 19 |

At UDI 1168, tool wear raised M1's output by 5.6714 log-odds and torque added 1.6134. All terms together raised its margin from −6.6338 to 1.4026, yielding a score of 0.80259. The forest also attributed a positive contribution to wear (+0.1911 probability units), but its final score of 0.33352 fell below the cutoff. At UDI 1497 both models assigned high scores. Torque contributed +11.6289 log-odds in M1. Mechanical power (+0.5878) and torque (+0.2971) accounted for most of the rise from the forest's 0.0340 baseline to 0.9833.

At UDI 345, torque and speed raised the HGB score to a false positive. At UDI 1764, a positive wear contribution of 3.8024 log-odds still left the total margin below zero. The case table gives the two largest signed terms for all 16 selections, and the CSV includes all terms, original measurements, applicable engineered values and reconstruction residuals. Case frequencies reflect the selection rule. [All case interpretations](../results/05_explainability/explanation_summary.md) · [Local data and signed terms](../results/05_explainability/local_explanations.csv) · [HGB cases](../results/05_explainability/figures/M1_local.png) · [Forest cases](../results/05_explainability/figures/M4_local.png)

## Why the rankings differ

AP permutation measures how much the ordering of positive labels deteriorates after a distribution intervention. A torque shuffle changes both measured torque and recomputed power in A4. It also breaks observed torque–speed associations, so the resulting AP loss can include the effect of unusual input combinations. Summing the torque and speed drops would count some shared information twice. Correlated temperatures create the same issue for temperature difference.

SHAP instead allocates the departure of a single model output from its baseline across fitted coordinates. Tree structure, interactions, class weighting and redundant features affect that allocation. The forest can split on measured speed, torque or their product. In the global forest summary, temperature difference ranked fourth (0.0117 probability units) and power fifth (0.0083). Power nonetheless made the largest contribution at UDI 1497. Likewise, frequent wear contributions on the mostly negative global sample can rank wear highly even when scrambling torque is more damaging to AP on rare positives. Only four sampled positives limit failure-specific interpretation of these averages.

The simulator explicitly uses temperature difference in heat dissipation and mechanical power in its power-failure rule. Providing these relationships as inputs lets trees approximate parts of the label generator directly. Their Phase 04 AP gains and their local contributions are consistent with that explanation. Because both derived quantities are deterministic functions of supplied measurements, these analyses cannot isolate a unique physical effect or identify which failure mechanism generated an individual label. The study reused selection folds and concerns classification of synthetic individual observations. Generalization to other equipment, future failures and operational thresholds remains open. [Generator description and dataset attribution](https://doi.org/10.24432/C5HS5C) · [Paired ablations](04_feature_ablation.md)

## Reproduction

From the repository root with the verified training export available, run the following commands. The SHAP requirements add a hash-locked dependency set constrained to the original environment. Install them after an exact environment sync, which can remove additional packages.

```sh
uv sync --locked
uv pip install --require-hashes -r configs/05_shap_requirements.txt
uv run --locked ai4i-pm explain
uv run --locked python scripts/verify_05_explainability.py
uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked ruff format --check .
```

The isolated replay reconstructed all 600 measurements and reproduced all 17 files byte for byte on Python 3.12.16, scikit-learn 1.9.1, NumPy 2.5.5 and SHAP 0.54.1. The maximum numeric difference in the SHAP archive was zero. The verifier checks archive arrays and tables at absolute tolerance 10⁻¹², verifies deterministic metadata and reports each differing file. Platform metadata, rendering libraries and floating-point reductions can affect byte comparisons across environments. Elapsed time is machine-dependent and is excluded from scientific artifacts. [Execution record and SHA-256 inventory](../results/05_explainability/execution.json) · [Replay verifier](../scripts/verify_05_explainability.py)

The NPZ stores each model's union of sampled and local UDIs. For each model prefix, `global_mask` identifies the 300 rows for aggregation. Arrays include transformed values and ordered feature names, signed SHAP values, baselines, matching model outputs, continuous class-1 probabilities and observed labels. The local CSV uses one row per selected model, UDI and transformed feature. A0 has no values in the two A4-only measurement fields. [Attribution archive](../results/05_explainability/shap_values.npz)

## Supervisor review

| Acceptance check | Status | Evidence |
| --- | --- | --- |
| Migrated baseline and earlier results | PASS | Clean starting tree on `main` at the expected commit. Git comparison verified the four earlier configurations, result trees, migration ledger and root lock. |
| Training and fold isolation | PASS | Training-export and CV digests match the frozen design. An isolated replay with only approved inputs produced identical results. |
| Fixed configurations and perturbations | PASS | 20 fold fits, 600 recorded permutations, zero deviation from the reference APs. |
| SHAP compatibility and sampling | PASS | Both public TreeExplainer preflights passed. The global IDs, class axes and reconstruction errors are recorded. |
| Global and local evidence | PASS | Seven figures, full signed values and 16 deterministic case selections across two models. |
| Reproducibility | PASS | All 17 files matched by SHA-256. Maximum SHAP archive difference was zero. |
| Software quality | PASS | 117 tests passed with SHAP installed. The suite ran in a copy without observation files. Ruff lint and formatting passed. Four existing Seaborn/Matplotlib deprecation warnings arose in the Phase 01 EDA fixture. |
| Phase boundary | PASS | Local, uncommitted Phase 05 evidence awaits review. Model adoption and later research phases remain with the supervisor. |

### Working-tree inventory

The three modified tracked files are the root README, the CLI in `src/ai4i_pm/cli.py` and `.github/workflows/ci.yml` for the hash-pinned SHAP test environment.

| New files by location | Files |
| --- | --- |
| `configs/` | `05_explainability.toml`, `05_shap_requirements.txt` |
| `src/ai4i_pm/` | `explainability.py`, `shap_analysis.py`, `explainability_figures.py` |
| Verification | `scripts/verify_05_explainability.py`, `tests/test_explainability.py` |
| Documentation | `docs/05_explainability.md`, `research/05_explainability/README.md` |
| `results/05_explainability/` | `execution.json`, `explanation_config.json`, `explanation_summary.md`, `permutation_fold_results.csv`, `permutation_fold_summary.csv`, `permutation_summary.csv`, `shap_metadata.json`, `shap_summary.csv`, `shap_values.npz`, `local_explanations.csv` |
| `results/05_explainability/figures/` | `permutation.png`, `M1_global.png`, `M1_effects.png`, `M1_local.png`, `M4_global.png`, `M4_effects.png`, `M4_local.png` |

AI4I 2020 Predictive Maintenance Dataset (2020), UCI Machine Learning Repository, CC BY 4.0. Tables, case measurements and figures derive from the dataset. [Provenance and license](dataset.md)
