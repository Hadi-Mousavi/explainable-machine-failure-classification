# Training-partition robustness results

Equal-weight mean fold or block average precision (AP). Δ is the paired A4 minus A0 mean.

| Assignment | M1 | M2 | M3 | M4 | HGB Δ | RF Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CV seed 7 | 0.772925 | 0.752304 | 0.863365 | 0.871172 | 0.090441 | 0.118868 |
| CV seed 42 | 0.759824 | 0.747959 | 0.848176 | 0.869823 | 0.088351 | 0.121864 |
| CV seed 123 | 0.746293 | 0.732310 | 0.860728 | 0.880047 | 0.114435 | 0.147737 |
| UDI order | 0.522726 | 0.568382 | 0.710596 | 0.738921 | 0.187870 | 0.170539 |

| Ordered block | Fit positives / rows | Evaluation positives / rows | Prevalence | M1 AP | M2 AP | M3 AP | M4 AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 57/2400 | 98/1200 | 0.081667 | 0.402072 | 0.471986 | 0.632831 | 0.764631 |
| 2 | 155/3600 | 26/1200 | 0.021667 | 0.523121 | 0.590908 | 0.816796 | 0.778639 |
| 3 | 181/4800 | 22/1200 | 0.018333 | 0.642986 | 0.642253 | 0.682162 | 0.673494 |

The random assignments reuse training observations. Ordered blocks vary in fit size, evaluation composition and prevalence. UDI has no verified time or equipment interpretation.

[Fold metrics](random_split_fold_metrics.csv) · [Ordered metrics](ordered_block_metrics.csv) · [Dispersion, rankings and contrasts](robustness_summary.json)
