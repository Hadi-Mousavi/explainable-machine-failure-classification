# Leakage controls

The study separated training-fold comparisons, validation cutoff selection and held-out evaluation.

| Risk | Control | Evidence or evaluation requirement |
| --- | --- | --- |
| Target-related TWF/HDF/PWF/OSF/RNF | Exclude by explicit allowlist | Rejection tests; no relationship analysis |
| UDI/Product ID encoding | Use UDI only as row identity; omit both from predictors | Central schema and export checks |
| Inadvertent held-out EDA | Export training rows and analysis columns only | EDA loader hashes export/descriptor; isolation test observes all CSV reads |
| Silent dataset drift | Preserve original bytes, verify SHA-256 | Provenance and loader tests |
| Silent split drift | Canonical UDI ordering; immutable manifest | Determinism, overlap, completeness, size, checksum and override-history tests |
| Global preprocessing | Fit learned operations inside training CV pipelines | Pipeline fitted separately on each training fold |
| Imbalance resampling contamination | Restrict any resampling to fit rows | All completed fits used observed rows |
| Validation overuse | Prespecify threshold utility and limit feedback | [Sealed validation decision](../results/07_final_evaluation/validation_decision.json) |
| Test-driven model/threshold selection | Freeze decisions before one final test evaluation | [Execution freeze](../configs/07b_execution_freeze.json) and [one-time test record](../results/07_final_evaluation/execution_manifest.json) |
| Duplicate or related observations across splits | Audit three duplicate definitions; discuss dependence | Do not claim IID equipment sampling; review ordered sensitivity |
| Recovering simulator rules | Disclose engineered-feature overlap | Paired feature ablations; no causal claims |
| Error-analysis feedback after test | Treat test findings as descriptive | Any new hypothesis needs new validation evidence |

The explanation study used held-out folds within training. The ordered sensitivity study also stayed within the training partition. After validation selected the cutoff, the study carried both training-fitted pipelines into the final evaluation. [Explanation design](05_explainability.md) · [Ordered sensitivity](06_robustness_analysis.md) · [Final evaluation](../research/07_final_evaluation/README.md)
