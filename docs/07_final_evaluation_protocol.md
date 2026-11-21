# Approved final evaluation protocol

**Approved for Phase 07A by the user.** Authorization covers training and one primary validation pass. Held-out test access is prohibited pending separate review. The primary system is Random Forest RF-11 with feature set A4. The reference is HGB-10 with A0. This is a design for classifying the *Machine failure* label of one synthetic observation. The operating threshold and held-out performance are still unknown. The machine-readable specification fixes the choices below against Phase 06 commit `75fd793a7b2b94127619921024ea495294afe67a`. [Protocol in TOML](../configs/07_final_evaluation.toml)

## Decision history and evidence

| Stage | Decision or observation relevant to the final evaluation |
| --- | --- |
| 01 · Data | Freeze the source identity, six permitted predictors and seed-42 stratified split. Use 6,000 training observations and reserve 2,000 each for validation and test. [Research protocol](research_protocol.md) · [Leakage controls](leakage_risks.md) |
| 02 · Baselines | Use average precision (AP) for ranking, retain 0.50 as a diagnostic and report positive support. The training set contained 203 failures. [Baseline methods](02_baseline_evaluation.md) |
| 03 · Search | HGB-10 was the provisional choice on the original predictors. RF-11 was the retained forest configuration. The candidate table stores all estimator arguments. [Model selection](03_model_selection.md) · [Candidate table](../results/03_model_selection/candidate_results.csv) |
| 04 · Ablation | Adding both features increased AP on all five paired folds for both families. The original-fold mean gain was 0.121864 for RF and 0.088351 for HGB. [Ablation results](04_feature_ablation.md) |
| 05 · Explanations | Torque gave the largest mean permutation loss in each configuration. Tool wear led the fold-1 mean absolute SHAP rankings for the two models examined. These training diagnostics describe fitted behavior and provide no independent selection evidence. [Explainability study](05_explainability.md) |
| 06 · Robustness | RF-11 + A4 had the highest mean AP at each random seed (42, 7, 123 respectively: 0.869823, 0.871172, 0.880047) and across the three ordered blocks (0.738921). HGB + A4 scored higher on blocks 2 and 3. RF's advantage on block 1 drove the ordered mean. [Robustness analysis](06_robustness_analysis.md) |
| 07 · Protocol freeze | Designate RF-11 + A4 primary and preserve the original HGB-10 + A0 selection as the sole descriptive reference. Neither validation AP nor any test outcome can reopen selection. |

Temperature difference and mechanical power partly reconstruct relationships used by the synthetic failure generator. The feature gains are consistent with that mechanism. All three random assignments reuse the same training observations, and the ordered analysis also draws from them. Model and feature selection are training-driven and potentially optimistic. The observed ordering supplies no statistical superiority claim. [Dataset provenance](dataset.md) · [Measured robustness summary](../results/06_robustness_analysis/robustness_summary.md)

## What will be fitted

| Role | Frozen specification |
| --- | --- |
| Primary, M4 | `RandomForestClassifier`, RF-11, 300 bootstrap trees, Gini splits, depth 16, minimum leaf size 1, square-root feature sampling, no class weighting, one worker, seed 42. A4 uses the six original predictors plus both engineered features. |
| Descriptive reference, M1 | `HistGradientBoostingClassifier`, HGB-10, log loss, balanced weights from the training labels, learning rate 0.1, 100 iterations, up to 31 leaves, minimum leaf size 20, L2 regularization 1.0, early stopping disabled, seed 42. A0 uses the original predictors. |

The complete `parameters` maps and ordered `features` arrays at `/models/M4` and `/models/M1` in the pinned Phase 06 JSON are normative, including nulls and library defaults. Cross-check against the Phase 03 candidate rows and Phase 04 design. Use the locked environment, one native thread and the existing tree pipeline. [Resolved specifications](../results/06_robustness_analysis/robustness_config.json) · [Feature design](../results/04_feature_ablation/ablation_config.json) · [Phase 03 provisional choice](../results/03_model_selection/selected_configuration.json)

Both pipelines accept exactly Type, air temperature (K), process temperature (K), rotational speed (rpm), torque (Nm) and tool wear (min), in that order. A4 appends process minus air temperature in kelvins and torque × (rpm × 2π/60) in watts. Fit the dense Type one-hot encoder only on training rows, ignore unknown categories and pass numeric values through. UDI identifies and orders rows. Product ID, UDI, the target and the five failure-mode flags are excluded from the predictor matrix. [Feature transformation](../src/ai4i_pm/feature_engineering.py) · [Tree preprocessing](../src/ai4i_pm/model_search.py)

Before fitting, verify the identities in the TOML against the provenance, frozen split and training descriptor. Verify actual source bytes at the data boundary, and confirm the three UDI sets are unique, disjoint, complete and of the prescribed sizes. Materialize only the currently authorized partition for the analyst. Use the verified training export in ascending UDI order. Fit each complete pipeline once to all 6,000 training rows and preserve its serialized bytes and SHA-256. The score scale used to select the threshold belongs to that particular fitted forest. A training-plus-validation refit could change it. Keep the same forest and encoder through validation and test, with no refitting, calibration, resampling or additional search. [Source provenance](../results/01_data_exploration/provenance.json) · [Frozen split](../results/01_data_exploration/split_manifest.json) · [Training descriptor](../results/01_data_exploration/training_partition.json)

## Single validation pass

Score the 2,000 validation rows once with the frozen primary pipeline. Select the probability column by its fitted class label 1. Reject nonfinite or out-of-range scores. For each of the 19 fixed cutoffs from 0.05 through 0.95 in steps of 0.05, predict failure when the unrounded score is greater than or equal to the listed cutoff. Use the explicit TOML grid to avoid cumulative step rounding.

For every cutoff, record TN, FP, FN, TP, precision, recall, F1 and F2. Matrix rows are true labels 0 and 1, and columns are predicted labels 0 and 1. Precision is TP/(TP+FP), recall is TP/(TP+FN), F1 is 2TP/(2TP+FP+FN), and **F2 is 5TP/(5TP+FP+4FN)**. Use zero for an undefined ratio (`zero_division=0`). F2 gives recall extra weight for an illustrative operating decision. The study supplies no industrial cost estimates or probability calibration evidence. [Established metric conventions](../src/ai4i_pm/evaluation.py)

1. If every F2 is zero, select 0.50 and explicitly report that validation failed to establish a useful operating point.
2. Otherwise maximize F2. Compare the count-based ratios exactly, using integer cross-products, and select the highest cutoff among ties. Rounding is for display only.
3. Freeze the full grid, chosen threshold and its metrics, the separately labeled 0.50 diagnostic, and the fallback flag and reason before test access. If the thresholds coincide, report both roles with the same counts. Bind the decision record to the approved protocol and the fitted forest by SHA-256. Validation only selects the primary threshold. The reference has no validation pass or tuned threshold.

## One held-out evaluation

Before opening test observations, verify the recorded source and split identities, exact model and feature specifications, both fitted-pipeline digests, approval record, and digest of the frozen validation decision. Release only the 2,000 test UDIs from the original split, align labels and scores by UDI, and verify size, uniqueness, binary labels and both-class support. Stop on integrity failure and document any interrupted access before considering recovery.

Score each frozen pipeline once in a single evaluation. Use these fixed vectors for all subsequent calculations. Report test N, positive and negative support and positive prevalence. Primary AP is the sum of precision weighted by increments in recall over the full continuous class-1 ranking. Export the full primary precision–recall curve, leaving the terminal plotting endpoint without a threshold. Report primary precision, recall, F1, F2 and confusion counts at the validation-selected cutoff and at the separate 0.50 diagnostic. For the reference, report AP and the same hard-label metrics at fixed 0.50. Report both systems for every outcome. Test results close the comparison and cannot trigger new fits, threshold changes or a different primary designation.

## Descriptive uncertainty

Use the original test AP and primary recall at the frozen operating point as point estimates. For each of 2,000 replicates, sample observed label–score pairs with replacement within each class, retaining the original positive and negative counts. On rows ordered by ascending UDI, initialize NumPy PCG64 once with seed 42, draw positives first and negatives second on each replicate, and use the same resample for both metrics. Fixed scores and the recorded threshold suffice throughout. Report the 2.5th and 97.5th percentiles using linear quantile interpolation, the method, replicate count, seed and class supports. No reference intervals or model-difference tests are planned.

These intervals condition on the observed prevalence, the fitted model and the selected threshold. Few positives limit recall precision. Dependence between synthetic observations could make row-level intervals too narrow. The bootstrap does not correct repeated-selection bias or establish transfer to real equipment. UDI supplies neither a timestamp nor an equipment identity. Conclusions concern contemporaneous classification of this synthetic sample and cannot establish field deployment, physical causation or future failure forecasting.

## Records and reporting

The protocol publication includes the TOML, this review and the approval record. Phase 07A implementation and results stay local.

| When | Planned artifact and contents |
| --- | --- |
| Approval | `approved_protocol_record.json` records the scope of user authorization, recording time, code revision and SHA-256 of the approved TOML and review document. |
| After training | Local `primary_pipeline.joblib` and `reference_pipeline.joblib` preserve both fitted pipelines. `fit_manifest.json` records inputs, membership identity, model and feature specifications, fitted-pipeline digests, environment and seeds. |
| Before test access | `validation_threshold_grid.csv` contains all 19 candidate rows. `validation_decision.json` records validation support, selected and diagnostic metrics, selection reason, fallback status and hashes of the grid, primary pipeline and protocol. Seal and independently record its digest before test release. |
| After test access | `test_metrics.json` contains both named systems, support, thresholds, AP, classification metrics and confusion counts. `primary_pr_curve.csv` and `primary_pr_curve.png` preserve curve coordinates and the figure. `bootstrap_intervals.json` records original estimates, endpoints and resampling settings. |
| Closeout | `execution_manifest.json` records input and output digests (excluding its own digest), pre-test decision digest, access sequence, software, times and any deviations. `final_report.md` presents the study. Phase 07A retains the validation export and its ordered primary scores locally in `private/validation.csv` and `private/validation_scores.npy`. |

The final report follows sections A–M in the TOML: research objective, dataset provenance and limitations, experimental methodology, baseline comparison, hyperparameter selection, feature ablation, explainability, robustness, prespecified final model and threshold, independent held-out evaluation, uncertainty, limitations and future work, and reproducibility and software availability. Reserve J and K for authorized results. “Independent held-out” describes isolation from model and threshold selection; independence of equipment or observations has not been demonstrated.

**Approval scope.** The user authorized the prespecified training fits and selection of the primary validation threshold. A separate supervisor review is required before test access. The test and bootstrap sections describe the later design only.
