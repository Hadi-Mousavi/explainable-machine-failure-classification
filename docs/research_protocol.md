# Research protocol

This document records the design through the feature ablation study. The later phase records specify how the study carried out the analyses proposed here. [Explainability](05_explainability.md) · [Robustness](06_robustness_analysis.md) · [Final evaluation protocol](07_final_evaluation_protocol.md)

## Objective and research questions

Classify whether a synthetic industrial observation has a machine-failure label.
There is no verified forecasting horizon, timestamp, or equipment identity.

1. **RQ1:** How do prespecified models compare at detecting rare failures under
   the primary observation-level split?
2. **RQ2:** How do class imbalance handling and validation-selected thresholds
   affect precision/recall trade-offs?
3. **RQ3:** What descriptive explanations emerge, and how do physics-informed
   feature ablations affect predictive behavior?
4. **RQ4:** How sensitive are findings to alternative splitting assumptions,
   particularly a separately specified split ordered by UDI?

## Data boundary and frozen partitions

The only source is [UCI dataset 601](https://doi.org/10.24432/C5HS5C), a synthetic
dataset. The source CSV uses **UDI**; the UCI table sometimes calls it UID.
The exact source schema is validated before any downstream operation.
See `src/ai4i_pm/schema.py` for the centralized allowlist. Type plus five numeric
measurements are permitted; UDI, Product ID, and all five failure-mode indicators
are prohibited predictors. The target is Machine failure.

Full-source integrity checks, target counts, category validity, and stratification
are allowed before splitting. They do not drive feature selection, outlier rules,
or analytic comparisons. No observations are altered.

The primary settings are `configs/01_data_exploration.toml`: seed 42, first reserve 20% test,
then reserve 25% of the remaining 80% as validation. This yields 60/20/20.
Both stages use target-stratified `train_test_split`, with integer seed 42.
UDI is sorted before stage one and the remainder is sorted before stage two;
membership is independent of the incoming dataframe ordering for fixed IDs.
The [scikit-learn splitting API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
documents stratification and integer random-state reproducibility.

Frozen UDI assignments are linked to original CSV SHA-256, exact settings, and
runtime versions. Checks enforce sizes, disjointness, completeness, both classes,
and class-count deviations of at most two observations from exact proportional
allocation (two sequential rounding stages). Expected primary sizes are
6000/2000/2000. Changing settings, data, or recorded runtime requires an explicit
reasoned override; old manifests and override events are retained.

EDA reads a checksum-verified **training-only export**, not the full raw CSV.
The export excludes Product ID and every failure-mode indicator. The EDA loader
hashes the frozen manifest without parsing held-out memberships or summaries.
This is a workflow boundary, not an operating-system access restriction: a
researcher can still read local raw files and must respect this protocol.

## Model selection and evaluation

Prespecified comparators: Dummy Classifier, Logistic Regression, Random Forest,
and Histogram-based Gradient Boosting.

Model selection used five shuffled, stratified folds with seed 42. Every
candidate used the same training and evaluation UDIs. Each fit learned its
category encoding and, for logistic regression, numeric scaling from its
training fold. The registered search space included paired class-weight
settings. All fits retained the observed training prevalence.

The primary score is **Average Precision (AP)** from continuous positive-class
scores. AP is a recall-increment-weighted sum of precision values; it is not
trapezoidal PR-AUC, which linearly interpolates the PR curve. See the
[official AP definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
Use the actual metric name and implementation in later reports.

Secondary outcomes: precision, recall, F1, F2, confusion matrix, and
precision-recall curve. Threshold adjustment changes hard-label metrics but does
not directly change AP calculated from the same scores. Later reports must state
class prevalence, support counts, positive-class coding, threshold, and the
zero-division convention. Accuracy alone is inadequate for the rare positive class.

Select the classification threshold on validation only, using an objective
approved before validation scores are inspected. F2 is a candidate, not a settled
industrial cost function; a minimum-precision/recall constraint needs a defensible
application requirement. Do not repeatedly adapt feature or model choices to
validation results. Record decisions and the complete candidate history.

After model, features, preprocessing, threshold rule, and reporting decisions
are frozen, use the test set once for final evaluation. No detailed test EDA or
feedback-driven adjustment is permitted. The final refitting strategy must be
approved before test access; combining training and validation can change score
calibration and invalidate a previously chosen threshold. Keep the baseline
strategy as training-only fitting with validation thresholding unless amended.
Uncertainty reporting for the small positive sample and any repeated-CV design
must also be agreed before evaluation.

## Prespecified feature ablations

- Remove Tool wear while leaving all other primary controls fixed.
- Add process minus air temperature, measured in K.
- Add mechanical power in W as torque [Nm] times RPM times `2*pi/60`.

The ablation study compared prespecified variants on identical training CV folds, with preprocessing
inside the folds. These features may overlap synthetic generation rules. A gain
can reflect recovery of a simulator's construction rather than physical insight
or real equipment performance. Describe this overlap before interpreting gains;
do not inspect failure-mode/target relationships to choose the ablations.

## Original plan for robustness analysis

A separately prespecified sensitivity analysis will sort by UDI and form ordered
partitions. UDI is neither a verified timestamp nor an independent equipment ID.
This analysis probes an ordering assumption and is **not prospective validation**.
Fix cut points, preprocessing, model-selection rules, and how changing class
counts will be reported before running it. Retain the primary random split;
do not select whichever split gives the best score.

## Limits and decision gates

Synthetic observations, few positive failures, simulator rules, possible ordered
dependence, and absent equipment identities constrain inference. Random splits
may share local generating conditions across partitions. Feature importance or
SHAP-style explanations cannot establish physical causal effects. Neither
equipment-level transfer nor a future failure horizon is demonstrated.

The final protocol records the F2 selection rule, training-only fit and
class-stratified bootstrap. [Final decision record](../research/07_final_evaluation/README.md) · [Leakage controls](leakage_risks.md)
