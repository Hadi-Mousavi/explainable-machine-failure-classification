# Day 2 training cross-validation

6,000 observations, 203 failures (3.38%). Each model scored every training UDI once.

| Model | Mean fold AP | Sample SD across folds |
| --- | ---: | ---: |
| Dummy prior | 0.033833 | 0.000456 |
| Logistic regression | 0.458571 | 0.042855 |

AP weights precision by increments in recall. The SD describes the five fold scores. Diagnostics use score ≥ 0.5 and assign zero to undefined ratios. The threshold was fixed before fitting.

These results describe synthetic training observations. See [fold metrics](fold_metrics.csv) and the [Day 2 protocol](../../docs/day2_baselines.md) for interpretation.
