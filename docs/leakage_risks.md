# Leakage risks and seven-day controls

Only Day 1 is implemented. The later-day sequence is a proposed control plan,
subject to research-supervisor approval, not a claim of completed experiments.

| Risk | Control | Day 1 evidence / later obligation |
| --- | --- | --- |
| Target-related TWF/HDF/PWF/OSF/RNF | Exclude by explicit allowlist | Rejection tests; no relationship analysis |
| UDI/Product ID encoding | Use UDI only as row identity; omit both from predictors | Central schema and export checks |
| Inadvertent held-out EDA | Export training rows and analysis columns only | EDA loader hashes export/descriptor; isolation test observes all CSV reads |
| Silent dataset drift | Preserve original bytes, verify SHA-256 | Provenance and loader tests |
| Silent split drift | Canonical UDI ordering; immutable manifest | Determinism, overlap, completeness, size, checksum and override-history tests |
| Global preprocessing | Fit learned operations inside training CV pipelines | Mandatory for later modeling |
| Imbalance resampling contamination | Resample only within CV training folds | No Day 1 resampling |
| Validation overuse | Prespecify threshold utility and limit feedback | Record full decision history later |
| Test-driven model/threshold selection | Freeze decisions before one final test evaluation | No test EDA/model scores on Day 1 |
| Duplicate or related observations across splits | Audit three duplicate definitions; discuss dependence | Do not claim IID equipment sampling; review ordered sensitivity |
| Recovering simulator rules | Disclose engineered-feature overlap | Later prespecified ablations; no causal claims |
| Error-analysis feedback after test | Treat test findings as descriptive | Any new hypothesis needs new validation evidence |

## Proposed seven-day sequence

1. **Day 1:** official source, immutable provenance, full-source integrity audit,
   frozen primary assignments, training-only EDA, tests, and protocol. No models.
2. **Day 2:** after approval, establish training CV folds and baseline pipelines.
   Fit preprocessing only within folds; keep identifiers and failure modes out.
3. **Day 3:** prespecified model comparisons and imbalance strategies using only
   the training CV design. Record trials and seeds, including unsuccessful trials.
4. **Day 4:** approved interpretability and physics-feature ablations on training
   folds; avoid choosing features based on validation/test patterns. Explanations
   remain descriptive and are checked for correlation and synthetic-rule effects.
5. **Day 5:** preregister and run the separate UDI-ordered sensitivity protocol.
   Preserve the primary split; never label UDI a verified time or equipment axis.
6. **Day 6:** freeze the model choice; select the threshold on validation under
   the agreed utility; freeze code, features, preprocessing, refit strategy,
   artifacts, and analysis plan. Validation feedback must not become hidden tuning.
7. **Day 7:** evaluate the frozen primary system once on the test partition,
   report uncertainty/limitations and all prespecified outcomes. Failure-mode
   indicators may be used for approved descriptive post-prediction error analysis,
   never as inputs or to revise reported test decisions. No such analysis is
   implemented or performed on Day 1.

The raw CSV and full manifest remain accessible locally. These controls prevent
accidental workflow leakage; they are not a secure data enclave. The small
held-out positive count and the synthetic observations limit generalization even
when every technical leakage check passes. A successful test suite does not
establish absence of dependence, realistic labels, or scientific validity by itself.
