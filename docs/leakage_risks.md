# Leakage controls

The same observation and preprocessing boundaries apply across the four completed phases.

| Risk | Control | Evidence or evaluation requirement |
| --- | --- | --- |
| Target-related TWF/HDF/PWF/OSF/RNF | Exclude by explicit allowlist | Rejection tests; no relationship analysis |
| UDI/Product ID encoding | Use UDI only as row identity; omit both from predictors | Central schema and export checks |
| Inadvertent held-out EDA | Export training rows and analysis columns only | EDA loader hashes export/descriptor; isolation test observes all CSV reads |
| Silent dataset drift | Preserve original bytes, verify SHA-256 | Provenance and loader tests |
| Silent split drift | Canonical UDI ordering; immutable manifest | Determinism, overlap, completeness, size, checksum and override-history tests |
| Global preprocessing | Fit learned operations inside training CV pipelines | Pipeline fitted separately on each training fold |
| Imbalance resampling contamination | Restrict any resampling to fit rows | All completed fits used observed rows |
| Validation overuse | Prespecify threshold utility and limit feedback | Record full decision history later |
| Test-driven model/threshold selection | Freeze decisions before one final test evaluation | All reported model scores come from training folds |
| Duplicate or related observations across splits | Audit three duplicate definitions; discuss dependence | Do not claim IID equipment sampling; review ordered sensitivity |
| Recovering simulator rules | Disclose engineered-feature overlap | Paired feature ablations; no causal claims |
| Error-analysis feedback after test | Treat test findings as descriptive | Any new hypothesis needs new validation evidence |

Planned explanations are descriptive. The ordered sensitivity study requires a prespecified UDI-ordering protocol. Final evaluation requires a frozen model, feature set, threshold and refitting strategy. Any post-prediction failure-mode analysis requires separate approval.
