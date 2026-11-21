# Baseline evaluation

How much ranking information do the six original predictors provide beyond failure prevalence? On five training folds, mean average precision (AP) was **0.033833** for the prior-only dummy and **0.458571** for unweighted logistic regression. Sample standard deviations across folds were 0.000456 and 0.042855. [Baseline results](../../results/02_baseline_evaluation/baseline_summary.md)

## Method and findings

Both classifiers used identical shuffled, stratified folds. Each training row received one out-of-fold score per model, giving 12,000 scores for 6,000 rows. The logistic pipeline learned its category encoding and numeric scaling within each fit. AP uses continuous scores. At the prespecified diagnostic cutoff of 0.5, logistic regression detected 41 of 203 failures and generated 15 false alarms. Operational alert costs and an appropriate decision threshold remain unknown. [Methods](../../docs/02_baseline_evaluation.md) · [Precision–recall curves](../../results/02_baseline_evaluation/figures/precision_recall.png) · [Diagnostic confusion matrices](../../results/02_baseline_evaluation/figures/diagnostic_confusion.png)

- [Configuration](../../configs/02_baseline_evaluation.toml), [baseline implementation](../../src/ai4i_pm/baselines.py), and [model specifications](../../results/02_baseline_evaluation/model_specs.json)
- [Shared fold assignments](../../results/02_baseline_evaluation/cv_manifest.json), [fold metrics](../../results/02_baseline_evaluation/fold_metrics.csv), and [out-of-fold predictions](../../results/02_baseline_evaluation/oof_predictions.csv)

## Reproduction

With the frozen training export available, run the verifier from the project root. It checks row and fold identities and compares all eleven baseline artifacts against an isolated replay. [Verifier](../../scripts/verify_02_baselines.py) · [Execution record](../../results/02_baseline_evaluation/execution.json)

```sh
uv run --locked python scripts/verify_02_baselines.py
```
