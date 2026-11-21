# Day 3 model selection

## Prespecified design

Use only the Day 1 training export and the Day 2 five-fold UDI manifest. Validate
the pinned hashes for the export, primary split, fold assignments, Day 2 settings
and reference fold scores. Evaluate Machine failure = 1 from Type and the five
original numeric measurements. Each fit learns its Type vocabulary on the fit
rows and ignores unseen categories. Numeric values pass through in the tree
pipelines. Both tree pipelines produce dense arrays. Logistic regression retains
the Day 2 fold-local scaler, C = 1, lbfgs, max_iter = 2000 and seed 42.

Sample six distinct structures per tree family, uniformly without replacement
using sklearn ParameterSampler and seed 42. Cross each draw with equal and
balanced class weights before fitting. RandomizedSearchCV evaluates all 12
prepared configurations per family with the saved folds, seed 42, one worker
and refit disabled. Pairing guarantees a weight-only contrast at every sampled
structure. Balanced weights use class frequencies in the fit fold. Use one
numeric-library thread, RF n_jobs = 1 and HGB early_stopping = false so every
boosting fit uses its full fold-training partition and specified iteration count.
All other classifier parameters use the locked defaults. The configuration and
machine-readable search record supply the complete spaces and settings.

| Family | Structural search space |
| --- | --- |
| Random forest | 150 or 300 trees, depth unlimited, 8 or 16, minimum leaf size 1, 3 or 8 |
| Histogram gradient boosting | Learning rate 0.05 or 0.1, 100 or 200 iterations, 15 or 31 leaf nodes, L2 penalty 0 or 1 |

Repeat the two fixed Day 2 configurations for timing and verify each fold AP
against the published scores. The third fixed specification changes only the
logistic class weight to balanced. Record continuous positive-class AP on each
evaluation fold. Give each fold equal weight and describe variation with sample
SD (denominator four). Retain partial scores and error details for incomplete
candidates. Only configurations with five successful scores are eligible.

## Prespecified decision rule

Select the highest unrounded mean fold AP across all eligible configurations,
including the fixed baselines. For exact numerical ties use this order of
families by convention: dummy, logistic, RF, HGB. Within RF prefer smaller
maximum depth (unlimited last), larger minimum leaf, then fewer trees. Within
HGB prefer fewer leaf nodes, fewer iterations, greater L2 regularization, then
smaller learning rate. After structural ties prefer unweighted fits, then
candidate ID in lexical order. The family order makes cross-family ties
deterministic. It does not estimate relative computation or effective degrees
of freedom.

## Training-fold results

The frozen training export contains 6,000 synthetic observations and 203
failures (3.3833%). Each evaluation fold contains 1,200 observations and 40
or 41 failures. An all-negative rule would correctly classify 96.62% of
training observations while missing all 203 failures. AP measures how well
the continuous scores rank these rare cases by weighting precision at each
increase in recall.

| Configuration | Evaluation | Mean fold AP | Sample SD |
| --- | --- | ---: | ---: |
| Prior dummy | Fixed | 0.033833 | 0.000456 |
| Logistic regression, unweighted | Fixed | 0.458571 | 0.042855 |
| Logistic regression, balanced | Fixed sensitivity | 0.439010 | 0.022426 |
| Random forest 11, unweighted | Selection CV | 0.747959 | 0.077190 |
| Histogram boosting 10, balanced | Selection CV | 0.759824 | 0.048779 |

The selected HGB uses learning rate 0.1, 100 iterations, up to 31 leaves,
L2 regularization 1.0 and balanced class weights. Its fold APs were 0.701921,
0.782883, 0.732319, 0.829294 and 0.752706. The best forest uses 300 trees,
depth 16 and minimum leaf size 1. HGB exceeded that forest by 0.011865 AP on
average. The second-ranked configuration, balanced HGB 12, scored 0.758559.
The 0.001266 difference, calculated before rounding, decides the prespecified ranking. These five
overlapping training fits do not establish statistical superiority.

Across matched configurations, balancing changed logistic mean AP by −0.019561
and reduced forest mean AP in all six pairs (differences −0.128594 to −0.064412).
For boosting, three pairs gained and three lost (−0.051701 to +0.019394).
The largest boosting gain occurred at the selected structure. These are
conditional effects at the sampled settings. Reweighting changes the training
loss and can shift the distribution and calibration of scores. AP evaluates
ranking. Probability calibration would require a separate assessment.

Selecting the highest score after comparing candidates on the same folds can
inflate the apparent performance of the winner. Fold SD measures dispersion
and supplies no confidence interval. Day 3 uses continuous scores throughout.
The Day 2 diagnostic cutoff stays at 0.5. There is no selected alert threshold,
post-selection OOF analysis or held-out evaluation. Test performance remains
unknown. The simulator provides no verified equipment identities, time horizon
or operational costs, so these observations cannot establish forecasting
performance, field transfer or the value of acting on an alert.

## Evidence and reproduction

The run evaluated 24 searched and three fixed configurations, completing all
135 planned fits with zero failures. Each successful scorer call recorded the
digest of its evaluation UDIs and checked it against the published Day 2
membership. Both fixed Day 2 configurations reproduced their reference fold
scores within an absolute tolerance of 1e-12.

- [All candidate and fold APs](../results/day3/model_comparison.md) and
  [full estimator parameters, aggregate fit/score seconds and errors](../results/day3/candidate_results.csv).
- [Fold sizes, positive counts, UDI digests and scores](../results/day3/candidate_fold_metrics.csv)
  and [matched weight contrasts by fold](../results/day3/class_weight_analysis.csv).
- [Sampled candidates and search settings](../results/day3/search_config.json),
  [complete ranking](../results/day3/model_comparison.json) and
  [provisional configuration](../results/day3/selected_configuration.json).
- [Family comparison](../results/day3/figures/model_family_ap.png),
  [candidate search](../results/day3/figures/candidate_search.png) and
  [class-weight sensitivity](../results/day3/figures/weight_sensitivity.png).
- [Runtime, warning history, versions and SHA-256 inventory](../results/day3/execution.json).

The reproducibility script copies only the frozen training CSV, the two
configuration files, the lockfile, Day 1 metadata, the fold manifest and the
published Day 2 metrics into an isolated directory. It verifies file digests,
recomputes summaries and contrasts from fold scores, and repeats the search.
Elapsed times depend on system load. Numerical libraries and figure rendering
can vary across platforms. The isolated Windows replay reproduced all 135 fold
scores and all 13 paired contrasts. Nine of the eleven output files matched
byte-for-byte, including the fold ledger, selection record and three figures.
The candidate table and execution record include measured times. The recorded
comparison took 97.55 seconds with Python 3.12.16 and scikit-learn 1.9.1.
See the [replay commands](../README.md#day-3-model-selection).
