# Training-only exploratory analysis

Scope: 6000 frozen training observations. No validation/test values were read.
Dataset SHA-256: `dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e`.
Split manifest SHA-256: `4ac9dea7352d9da8a9e7a52d241325e3e9df593aab1423c587e880d79b2fd310`.

## Measured observations

- Training failures: 203/6000 (3.38%); rare positives motivate the prespecified imbalance-aware evaluation plan.
- Air/process temperature Pearson r = 0.876; speed/torque r = -0.876. These are descriptive associations, not evidence of physical causality.
- Training missing cells across the allowed predictors and target: 0.
- IQR fences flag distribution tails for review; they are not domain-validity boundaries. All observations were retained and no imputation or feature selection was performed.
- Target-conditioned boxplots describe unequal group sizes; they are neither model performance estimates nor causal effects.

## Potentially unusual values (training IQR rule)

| Feature | Minimum | Maximum | Lower fence | Upper fence | Outside fences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Air temperature [K] | 295.30 | 304.50 | 293.50 | 306.30 | 0 |
| Process temperature [K] | 305.70 | 313.80 | 305.50 | 314.30 | 0 |
| Rotational speed [rpm] | 1168.00 | 2874.00 | 1137.00 | 1897.00 | 237 |
| Torque [Nm] | 4.20 | 76.60 | 12.80 | 67.20 | 41 |
| Tool wear [min] | 0.00 | 253.00 | -114.50 | 329.50 | 0 |

## Figures

- [figures/01_train_class_distribution.png](figures/01_train_class_distribution.png)
- [figures/02_train_type_distribution.png](figures/02_train_type_distribution.png)
- [figures/03_train_numeric_distributions.png](figures/03_train_numeric_distributions.png)
- [figures/04_train_missingness.png](figures/04_train_missingness.png)
- [figures/05_train_numeric_correlations.png](figures/05_train_numeric_correlations.png)
- [figures/06_train_feature_relationships.png](figures/06_train_feature_relationships.png)
- [figures/07_train_features_by_target.png](figures/07_train_features_by_target.png)

These synthetic observation-level patterns establish neither real-world generalization
nor future failure forecasting. No failure-mode comparisons, models, or thresholds
were used.
