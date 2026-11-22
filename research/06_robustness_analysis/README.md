# Robustness analysis

How sensitive are the fixed models to reassignment of training folds and to the ordering of synthetic observations? All three random CV seeds produced the same mean average precision (AP) ranking. On the three UDI-ordered blocks, mean AP fell for all four configurations and RF overtook HGB on the original predictors. Adding temperature difference and mechanical power increased mean AP in both families under every assignment. [Scientific report, tables and reproduction commands](../../docs/06_robustness_analysis.md)

| Evidence | Contents |
| --- | --- |
| [Protocol](../../configs/06_robustness_analysis.toml) and [resolved configuration](../../results/06_robustness_analysis/robustness_config.json) | Frozen input hashes, model parameters, feature sets, splits, metrics and aggregation |
| [Alternative fold assignments](../../results/06_robustness_analysis/alternative_cv_manifest.json) and [fold metrics](../../results/06_robustness_analysis/random_split_fold_metrics.csv) | Two exact five-fold assignments by UDI, 40 new evaluations and 20 seed-42 reference rows |
| [Ordered memberships](../../results/06_robustness_analysis/ordered_split_manifest.json) and [block metrics](../../results/06_robustness_analysis/ordered_block_metrics.csv) | Three expanding windows and all 12 evaluations with class support, prevalence and AP |
| [Summary tables](../../results/06_robustness_analysis/robustness_summary.md) and [numerical summary](../../results/06_robustness_analysis/robustness_summary.json) | Fold and block dispersion, rankings, paired feature contrasts and between-strategy differences |
| [Figures](../../results/06_robustness_analysis/figures) and [execution inventory](../../results/06_robustness_analysis/execution.json) | Four scientific figures, software versions, input and output digests |

All comparisons use the original 6,000 training observations. UDI has no verified temporal or equipment interpretation. The [final evaluation](../07_final_evaluation/README.md) reports the separately reserved validation and test results.
