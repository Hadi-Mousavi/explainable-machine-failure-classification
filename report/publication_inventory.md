# Publication artifact inventory

This inventory identifies the documents and evidence for the completed AI4I study as of 22 November 2026. Links are relative to this inventory. The recorded evaluation revision is `47cc09cd1e91035472df965507ce5e4dee61527a`.

## Reader documents

| Item | Role |
| --- | --- |
| [Project README](../README.md) | Study abstract, main findings, seven-phase navigation and software checks |
| [Technical report](final_report.md) · [Typeset PDF](final_report.pdf) | Academic synthesis with four original figures, held-out metric table and references |
| [Phase 07 guide](../research/07_final_evaluation/README.md) | Decision history and portable checksum verification |
| [Scientific integrity audit](scientific_integrity_audit.md) | Scope, independent checks and review limitations |
| [Source citation](../CITATION.cff) · [Software license](../LICENSE) · [Data attribution](../docs/dataset.md) | Software authorship and MIT terms, dataset DOI and CC BY 4.0 terms |

## Training evidence and figures

| Phase | Source record | Selected visual |
| --- | --- | --- |
| 01 | [Source provenance](../results/01_data_exploration/provenance.json) · [Training EDA](../results/01_data_exploration/eda.md) · [Migration ledger and digests](../docs/migration/artifacts.csv) | [Training class distribution](../results/01_data_exploration/figures/01_train_class_distribution.png) |
| 02 | [Baseline summary](../results/02_baseline_evaluation/baseline_summary.md) · [Execution](../results/02_baseline_evaluation/execution.json) | [Report Fig. 1, baseline AP](../results/02_baseline_evaluation/figures/baseline_ap.png) |
| 03 | [Candidate comparison](../results/03_model_selection/model_comparison.md) · [Execution](../results/03_model_selection/execution.json) | [Family comparison](../results/03_model_selection/figures/model_family_ap.png) |
| 04 | [Paired ablations](../results/04_feature_ablation/ablation_summary.md) · [Execution](../results/04_feature_ablation/execution.json) | [Report Fig. 2, ablation AP](../results/04_feature_ablation/figures/ap_comparison.png) |
| 05 | [Explanation summary](../results/05_explainability/explanation_summary.md) · [Execution](../results/05_explainability/execution.json) | [Raw-input permutation losses](../results/05_explainability/figures/permutation.png) · [RF fold-1 SHAP](../results/05_explainability/figures/M4_global.png) |
| 06 | [Strategy and block metrics](../results/06_robustness_analysis/robustness_summary.md) · [Execution](../results/06_robustness_analysis/execution.json) | [Report Fig. 3, strategy comparison](../results/06_robustness_analysis/figures/strategy_ap.png) |

## Final evaluation evidence

The [approved design](../configs/07_final_evaluation.toml), [approval record](../results/07_final_evaluation/approved_protocol_record.json), [fit manifest](../results/07_final_evaluation/fit_manifest.json), [validation grid](../results/07_final_evaluation/validation_threshold_grid.csv), [sealed decision](../results/07_final_evaluation/validation_decision.json) and [execution freeze](../configs/07b_execution_freeze.json) precede the test. The fit and validation start markers and [decision checksum](../results/07_final_evaluation/validation_decision.sha256) are retained alongside them.

| Final release file | Contents | Verified SHA-256 |
| --- | --- | --- |
| [test_started.json](../results/07_final_evaluation/test_started.json) | Test reservation and frozen revision | `ed4f4708a6ef3655f2d1a987799d038a1eb514b270ac0f45d9fd0211b7b7156d` |
| [test_metrics.json](../results/07_final_evaluation/test_metrics.json) | Exact estimates and counts underlying Table 1 | `4bca19c67d15407bc6e74ff2ef62587859832b0b303b3dd674b9441c04517066` |
| [primary_pr_curve.csv](../results/07_final_evaluation/primary_pr_curve.csv) | Aggregate curve coordinates | `a4cc4f821edcb93445c87de7f2be3d30f087cc478f4237481fc184e291bea5a4` |
| [primary_pr_curve.png](../results/07_final_evaluation/primary_pr_curve.png) | Report Fig. 4, primary held-out precision–recall curve | `dccae83bbba3eab9227b9c6f8fd33ae98e61a5271a1f79e02841945dd80335d3` |
| [bootstrap_intervals.json](../results/07_final_evaluation/bootstrap_intervals.json) | Percentiles, point estimates and resampling specification | `14895399ca1b0154d2718d80911fb536ff90cb9fb0f50e0338f7df893312f70e` |
| [final_report.md](../results/07_final_evaluation/final_report.md) | Original execution-generated report | `16a30d5e9415a0f82ddf17bd176badbfd6162c23051531265b832f464a630c44` |
| [execution_manifest.json](../results/07_final_evaluation/execution_manifest.json) | Run identity, event sequence, environment and six output digests | `8216a55872cee966412090e88acf52b1ce46cd42d5e63e6b1db84c120e98ffc8` |

The first six digests match the manifest. The last is the SHA-256 observed during this audit. The original archive and CSV, derived observation exports, validation scores and fitted pipelines are local ignored inputs or outputs. They are outside the publication inventory. Training out-of-fold predictions and training-fold SHAP records are identified by phase in the original inventories.

## Publication document checksums

These SHA-256 digests identify the reader documents in this release. The experiment manifests above retain their original digests.

| Document | SHA-256 |
| --- | --- |
| [README.md](../README.md) | `ad5f41814be20fa9d43d49f3ad8bab9067f24d6142d2e344030efdc0c2233125` |
| [CITATION.cff](../CITATION.cff) | `f5e9a5314f3582f714a49894e506580a26c613b56772d853068306d0e839ebf3` |
| [report/final_report.md](final_report.md) | `55bc18f035d115da5ffef2b487de4452f4519ba45400a054d8b46953e207cb35` |
| [report/final_report.pdf](final_report.pdf) | `2464d486c24e19fc312c4798377185a076874eee8ca268880676c154c77a0aaf` |
| [research/07_final_evaluation/README.md](../research/07_final_evaluation/README.md) | `cc395a48f65a29b3e7b6e623fe3dfde354a286a4cfbcaf425fee983766be4e56` |
| [report/scientific_integrity_audit.md](scientific_integrity_audit.md) | `e340a51e60122e04f14aaffb88120cd2c78087f71d00c7ed84a86c5d75214259` |
