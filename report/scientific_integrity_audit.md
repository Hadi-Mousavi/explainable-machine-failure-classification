# Scientific integrity audit

**Review date:** 22 November 2026. The audit covered the active working tree, the local Git history, saved result aggregates and publication documents for the AI4I equipment failure classification study.

## Git provenance and scientific freeze

Before the evaluation outputs were prepared for publication, `main`, local `origin/main` and the revision in the final execution manifest all identified `47cc09cd1e91035472df965507ce5e4dee61527a`. It contains the software freeze for test execution. Its parent `5fd17fe` froze the validation decision, following the protocol commit `7c2232a`. The robustness, explanation, ablation, model-search, baseline and foundation commits precede those decisions.

The expanded report lives outside the result tree because the original execution report is one of the six checksum-bound outputs. No model selection, threshold search, scientific replay, bootstrap resampling or test observation access was part of this audit. The saved aggregate curve and confusion counts are sufficient to check several results independently.

## Independent scientific checks

| Check | Result |
| --- | --- |
| Earlier result integrity | All 56 output digests listed in the Phase 02–06 execution records matched active artifact bytes (10, 10, 9, 16 and 11 respectively), including the training-study figures reused in the report. |
| Final release | All six listed output hashes matched. The independently measured manifest digest and the six original digest values appear in the [inventory](publication_inventory.md). The manifest has no self-digest. |
| Decision and code chain | The approval, fit manifest, validation decision, split manifest, protocol and execution-freeze files matched their recorded hashes. The separate decision seal, validation grid, three frozen code/test files and both serialized pipeline hashes also matched. Pipelines were hashed as bytes. Source identity was compared with saved provenance and the archive digest. The raw CSV was left unopened. The audit also verified the migration ledger against its historical revision, the saved training membership digests and the model specifications. |
| Ranking | Summing stored precisions weighted by successive recall differences in the saved aggregate curve reproduced primary AP 0.8799273420656948 within 10⁻¹². Reference AP 0.8302142198101026 was transcribed from the checksum-verified metric record. |
| Counts and operating metrics | For validation at 0.15 and each of the three test rows, counts yielded 68 positives and 1,932 negatives. Precision, recall, F1 and F2 agreed with direct arithmetic on those counts. The validation grid has 19 rows. The sealed decision names 0.15 as its unique F2 maximum. |
| Uncertainty | Both point estimates in the interval record match the corresponding metrics. The AP interval is [0.810543, 0.941731] and the recall interval is [0.764706, 0.926471]. Endpoints were verified against the saved record. The test score-vector digests remain provenance from the execution. The test vectors were not persisted. |

## Publication and security review

The review inventoried tracked and non-ignored paths and checked ignored data and model locations by filename. The original archive and raw CSV reside under the ignored raw-data directory. The processed training export, private training and validation exports, private validation scores and both fitted pipelines are also ignored. The pipelines are 4,574,034 and 375,129 bytes. No row-level test predictions or test observation export appeared in the publication inventory. The test curve is an aggregate coordinate table. The Phase 02 out-of-fold predictions and Phase 05 explanation archive describe training observations. The exploration notebook has zero saved cell outputs.

Filename and text-pattern checks covered common provider tokens, private-key headers, credential-bearing URLs, secret assignments and machine-specific paths. No matching credential or machine-local path was found in the reviewed publication text. This is a scoped screen of the active files. It does not certify every historical Git object or every possible secret encoding. Local virtual environments, bytecode and tool caches stay outside the publication set.

The approved Phase 07A protocol records the decision boundary at the time of its approval. The [Phase 07 guide](../research/07_final_evaluation/README.md) links it to the subsequent execution freeze and held-out evaluation.

## Software and documentation checks

The full pytest suite ran from a temporary copy of the publication files, with the package imported from the copied source. The copy excluded observation files, private score exports and serialized models. Model-fitting tests used artificial fixtures. All 145 tests passed in 61.51 seconds. Four deprecation warnings came from Seaborn/Matplotlib in the training-EDA fixture. Ruff lint passed and the format check reported 77 files already formatted.

The documentation check resolved relative links, image paths and referenced Markdown heading anchors across the root, phase guides, results, report and exploration notebook. The report bibliography was checked against the UCI record, IEEE, JMLR, the NeurIPS proceedings, Springer and the scikit-learn documentation. UCI identifies Matzka’s 2020 conference paper as the introductory publication. IEEE confirms the title, conference and DOI, and the author’s university confirms pages 69–74. The DOI identifiers were confirmed at their landing pages. The original dataset archive was not fetched. The six-page PDF was rendered for visual review, including its figure captions and test table.

## Interpretation and release status

The primary result concerns synthetic, contemporaneous classification under one reserved split. Reused folds, simulator-aligned features, uncertain row dependence and 68 held-out failures limit inference. The saved intervals condition on the chosen system and class counts. The evidence supplies neither a validated warning horizon nor an estimate of maintenance benefit. The UCI dataset is licensed under CC BY 4.0. The source code is licensed under the MIT License. The final publication review covered the seven original execution records and the separate Markdown and PDF editions of the expanded report.
