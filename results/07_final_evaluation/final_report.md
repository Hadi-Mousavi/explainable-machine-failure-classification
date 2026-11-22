# Explainable Machine Learning for Equipment Failure Classification

## A. Research objective

Estimate how well a frozen classifier ranks and identifies the machine-failure label of a single synthetic operation.

## B. Dataset provenance and limitations

The UCI AI4I 2020 dataset contains 10,000 synthetic observations. Product type, air and process temperature, rotational speed, torque and tool wear were the six permitted measurements. UDI identified observations. Equipment identity, timestamps and a forecasting horizon are unavailable. [Dataset provenance](../../docs/dataset.md)

## C. Experimental methodology

The frozen stratified split allocated 6,000 observations to training and 2,000 each to validation and test. All training comparisons used training observations. Validation determined one primary cutoff. The final evaluation used both persisted pipelines once on the same test observations. All metrics and intervals used the original class-1 scores. [Approved protocol](../../docs/07_final_evaluation_protocol.md)

## D. Baseline model comparison

The prior-only baseline achieved mean training-fold AP of 0.033833. Logistic regression achieved 0.458571. These are means over five training folds. [Baseline results](../02_baseline_evaluation/baseline_summary.md)

## E. Hyperparameter selection

On the six original predictors, HGB-10 achieved mean training-fold AP of 0.759824 and RF-11 achieved 0.747959. HGB-10 supplied the original descriptive reference. [Model selection](../../docs/03_model_selection.md)

## F. Feature-ablation analysis

Adding temperature difference and mechanical power increased mean training-fold AP to 0.848176 for HGB-10 and 0.869823 for RF-11. Both features overlap with synthetic failure mechanisms. [Feature ablation](../../docs/04_feature_ablation.md)

## G. Explainability findings

Torque yielded the largest mean permutation loss across the four configurations. Tool wear led the fold-1 mean absolute SHAP rankings for both models examined. These findings describe training-fold behavior. [Explainability study](../../docs/05_explainability.md)

## H. Robustness analysis

RF-11 with both engineered features led the mean AP rankings across three random seeds and the mean over three ordered blocks. HGB with both features scored higher on two individual ordered blocks. UDI order has no established temporal interpretation. [Robustness study](../../docs/06_robustness_analysis.md)

## I. Prespecified final model and threshold

RF-11 + A4 was designated primary before validation. The threshold of 0.15 was the unique F2 maximum across 19 approved validation cutoffs. It detected 61 of 68 validation failures with 32 false positives. Validation F2 was 0.835616. HGB-10 + A0 remained the descriptive reference at 0.50. [Validation decision](validation_decision.json)

## J. Independent held-out evaluation

The test partition contained 2,000 observations, including 68 positives and 1,932 negatives, for a prevalence of 3.40%. AP uses the continuous ranking and therefore has the same value at both primary cutoffs.

| System and cutoff | AP | Precision | Recall | F1 | F2 | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RF-11 + A4, frozen 0.15 | 0.879927 | 0.580000 | 0.852941 | 0.690476 | 0.779570 | 1890 | 42 | 10 | 58 |
| RF-11 + A4, diagnostic 0.50 | 0.879927 | 0.980000 | 0.720588 | 0.830508 | 0.760870 | 1931 | 1 | 19 | 49 |
| HGB-10 + A0, reference 0.50 | 0.830214 | 0.706667 | 0.779412 | 0.741259 | 0.763689 | 1910 | 22 | 15 | 53 |

At 0.15, the test-minus-validation differences were -0.0759 in precision, -0.0441 in recall and -0.0560 in F2. The validation estimate describes the sample used to select the cutoff. Test performance describes its application to the reserved sample. [Full metrics](test_metrics.json) · [Precision–recall curve](primary_pr_curve.png)

## K. Uncertainty analysis

The primary percentile 95% interval was [0.810543, 0.941731] for AP and [0.764706, 0.926471] for recall at 0.15. Each of 2,000 paired bootstrap replicates retained the observed positive and negative counts. A single PCG64 generator used seed 42 and drew positives first, then negatives, from ascending UDI order. Endpoints use linear interpolation. [Bootstrap record](bootstrap_intervals.json)

## L. Limitations and future work

The intervals condition on observed prevalence, the particular fitted forest and the selected threshold. Limited positive support constrains recall precision. Dependence between synthetic observations could narrow the intervals. The bootstrap omits fitting and selection uncertainty, including optimism from repeated training comparisons. The findings support contemporaneous classification on this synthetic sample. Industrial generalization, failure forecasting and maintenance utility require independent field evidence.

## M. Reproducibility and software availability

The frozen evaluation revision was `47cc09cd1e91035472df965507ce5e4dee61527a`. The manifest records the release identity, software environment, score-vector digests, access sequence and output hashes. Full-precision estimates and all curve coordinates accompany this report. [Execution manifest](execution_manifest.json) · [Fit manifest](fit_manifest.json) · [Curve coordinates](primary_pr_curve.csv)
