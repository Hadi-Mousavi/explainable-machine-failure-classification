# Phase 05 · Explainability

All four fixed classifier and feature combinations were fitted on the five frozen training folds. Permuting torque caused the largest mean AP loss in each configuration. In a separate 300-row fold-1 sample, tool wear had the largest mean absolute SHAP contribution for HGB-10 + A0 and RF-11 + A4. The two methods measure different properties of the fitted classifiers. [Scientific report and reproduction commands](../../docs/05_explainability.md)

| Evidence | Contents |
| --- | --- |
| [Frozen design](../../configs/05_explainability.toml) | Input digests, seeds, feature interventions, background and sampling rules, output units |
| [Measured summary](../../results/05_explainability/explanation_summary.md) | All six original-predictor rankings, fold variation, global attributions and local decisions |
| [Figures](../../results/05_explainability/figures) | Five-fold permutation results and model-specific attribution, effect and diagnostic-case plots |
| [SHAP metadata](../../results/05_explainability/shap_metadata.json) | Exact sample IDs, positive-class coordinates, category counts and reconstruction checks |
| [Execution inventory](../../results/05_explainability/execution.json) | Runtime versions and artifact checksums |

The label concerns a single synthetic observation. The 0.5 cutoff defines diagnostic cases. The provisional HGB selection and exploratory forest result remain open for supervisor review.
