# Phase 02 baseline methods

The binary target is Machine failure (failure = 1). The frozen training export
contains 6,000 observations, including 203 failures. Phase 02 links the approved
source digest to the Phase 01 provenance and training descriptor. The Phase 01 loader
hashes the split manifest and export, checks the UDI sequence, and validates the
schema. Only the training CSV is parsed for observation values. Each pipeline
accepts exactly Type, the two temperatures, rotational speed, torque and tool
wear. UDI identifies rows in the outputs.

## Paired five-fold comparison

Ascending UDI order makes fold assignment independent of input order. Shuffled
stratified five-fold CV with seed 42 gives each evaluation fold 40 or 41 failures
and leaves about four fifths of the observations for fitting. The two models
use the same evaluation UDIs. Each observation receives one cross-fitted score
per model. A fresh pipeline encodes Type with unknown categories ignored and
standardizes the five numeric predictors on the fit rows of each fold. The
column transformer drops the remainder. The validated export has no missing
values. Both pipelines use the observed measurements directly.

The dummy classifier assigns the fit-fold failure prevalence to every evaluation
row. Its fold AP therefore equals the *evaluation* prevalence. Logistic
regression uses C = 1, lbfgs, a 2,000-iteration limit, seed 42 and equal
observation weights. The fixed specifications and locked library defaults are
recorded in [model_specs.json](../results/02_baseline_evaluation/model_specs.json).

## Reading the metrics

An all-negative classifier would achieve 96.62% accuracy while missing every
failure. Average Precision measures how well continuous scores rank the rare
positives. Scikit-learn sums precision weighted by the increment in recall.
Trapezoidal interpolation gives a different area. The primary summary is the
arithmetic mean of the five fold APs. Pooling scores would compare observations
scored by different fitted models. The sample SD (denominator 4) describes
variation across these folds. Their overlapping fit sets preclude interpreting
that SD as a confidence interval.

For diagnostics, a score ≥ 0.5 produces an alert. This fixed default allows
the models to be compared before an application-specific operating objective is
approved. Precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN),
and F2 = 5TP/(5TP+FP+4FN). Undefined ratios receive zero. Confusion matrix
rows follow actual labels 0, 1 and columns follow predicted labels 0, 1.
The figure sums fold counts, so each training row contributes exactly once.
The PR figure shows separate fold curves. The final precision/recall coordinate
(1, 0) is a plotting endpoint with an empty threshold in the exported CSV.

## Interpretation and reproducibility

Logistic regression achieved higher AP than the dummy in each paired fold.
At 0.5 it still missed 162 of 203 failures. The data establish neither the cost
of a missed failure nor the cost of investigating a false alarm. Synthetic
generation, limited positive support and absent equipment IDs and verified
timestamps restrict inference about field performance and prospective failures.
This phase used only the training partition. [Final validation and test results](../research/07_final_evaluation/README.md)

The [execution record](../results/02_baseline_evaluation/execution.json) lists the environment and
SHA-256 digests. The replay command in the [phase guide](../research/02_baseline_evaluation/README.md) verifies the
saved fold/OOF identities and compares two runs on the same environment. Numeric
libraries, fonts and rendering backends can affect bytes across platforms.
The derived results use the synthetic UCI AI4I 2020 dataset under CC BY 4.0.
See the [Phase 01 provenance and attribution](dataset.md).
