# Phase 06 · Sensitivity to training partitions and UDI order

All three stratified assignments gave the same mean average precision (AP) ordering, RF with A4, HGB with A4, HGB with A0, then RF with A0. Under the three UDI-ordered evaluations, the original-feature models exchanged places. All four ordered means were lower. Both families gained from the two engineered features under every assignment. [Measured summary](../results/06_robustness_analysis/robustness_summary.md)

## Preregistration and data boundary

Phase 06 began on clean `main` at Phase 05 commit `4debe0c69bc020206aa2f0e10591f6a1e7ce6048`. Before fitting, the protocol fixed the models, input digests, seeds, exact alternative membership digests, cut points, class diagnostics, runtime settings, aggregation and treatment of undefined metrics. Its SHA-256 is `a50f7fbda76b3eb32d87dc6cae9824c13be74ade02c19aadd18093e672e23d6b`. [Preregistered protocol](../configs/06_robustness_analysis.toml)

The 6,000 approved training observations contained 203 positives and 5,797 negatives. The loader verified the source CSV identity against frozen provenance, hashed the primary split manifest as opaque bytes, and verified the training CSV, its UDI membership and the original CV manifest. The original source CSV and primary validation and test observations were outside the replay directory. The only observation input was the training export. [Execution and input inventory](../results/06_robustness_analysis/execution.json)

| Frozen input | SHA-256 |
| --- | --- |
| Source CSV, via verified provenance | `dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e` |
| Training CSV | `a919b8884167b94edd8f538878db122c4d0a27817aee5cff6e303894eb130900` |
| Primary split manifest | `4ac9dea7352d9da8a9e7a52d241325e3e9df593aab1423c587e880d79b2fd310` |
| Original CV manifest | `1ddf0178f0561ec5b03f61931ba0db84a7e49a625eb7e4e9d36d137e1def1636` |

M1 and M3 use HGB-10 with balanced class weights, learning rate 0.1, 100 iterations, up to 31 leaves, L2 regularization 1.0 and early stopping disabled. M2 and M4 use RF-11 with 300 bootstrap trees, depth 16, minimum leaf size 1, square-root feature sampling, one worker and unweighted observations. Both use estimator seed 42 and one native thread. Every fit restores the complete saved parameter dictionary. Type encoding is learned inside each fit.

A0 supplies Type, air and process temperatures, rotational speed, torque and tool wear. A4 adds process minus air temperature in kelvins and mechanical power in watts (torque × rpm × 2π/60). UDI supplies row identity and ordering. The six-column input guard excludes identifiers, the target and failure-mode indicators. [Complete estimator parameters and ordered columns](../results/06_robustness_analysis/robustness_config.json) · [Feature definitions](../src/ai4i_pm/feature_engineering.py)

## Random-split sensitivity

For seeds 7 and 123, shuffled stratified five-fold assignments started from ascending numeric UDI. Each fit used the complement of the 1,200 evaluation UDIs. Each training observation received one evaluation assignment per seed, shared by all four configurations. The 20 seed-42 scores came directly from the checksum-verified Phase 04 fold table. Only the other 40 scores required new fits. Folds 1–5 had 40, 40, 41, 41 and 41 positives for each seed, or prevalence of 3.333% to 3.417%. The saved UDI lists identify individual observations even when fold supports coincide. [Seed-42 membership](../results/02_baseline_evaluation/cv_manifest.json) · [Exact alternative memberships](../results/06_robustness_analysis/alternative_cv_manifest.json) · [All 60 fold scores, counts, hashes and available timings](../results/06_robustness_analysis/random_split_fold_metrics.csv)

| Assignment | M1 HGB A0 | M2 RF A0 | M3 HGB A4 | M4 RF A4 | HGB A4 − A0 | RF A4 − A0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Seed 42, reference | .759824 (.048779) | .747959 (.077190) | .848176 (.054924) | .869823 (.046892) | +.088351 | +.121864 |
| Seed 7 | .772925 (.066004) | .752304 (.070997) | .863365 (.024395) | .871172 (.027263) | +.090441 | +.118868 |
| Seed 123 | .746293 (.088080) | .732310 (.074927) | .860728 (.039357) | .880047 (.042220) | +.114435 | +.147737 |
| Range of seed means | .026632 | .019994 | .015190 | .010224 | | |

Entries give equal-weight mean AP with the sample SD of the five fold scores in parentheses. Within-seed fold SD exceeded the range of seed means for each configuration. Seed 123 gave the largest A4 gain in both families. The seeds reuse the same 6,000 synthetic observations and overlapping fits. SD describes dispersion and supplies no independent-sample confidence interval. [Random-seed figure](../results/06_robustness_analysis/figures/random_seed_ap.png)

## UDI-ordered sensitivity

The separate expanding-window design retained ascending numeric UDI. Four identical model comparisons used each pair of slices. Row positions below are one-based and inclusive. UDI intervals give endpoints of the training observations in each evaluation slice. The exact manifests preserve the gaps between training UDIs. [All fit and evaluation UDI lists](../results/06_robustness_analysis/ordered_split_manifest.json) · [All 12 scores and runtimes](../results/06_robustness_analysis/ordered_block_metrics.csv)

| Block | Fit positions | Fit + / − | Evaluation positions | Evaluation UDI endpoints | Eval + / − | Prevalence | M1 AP | M2 AP | M3 AP | M4 AP |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1–2400 | 57 / 2343 | 2401–3600 | 3996–6003 | 98 / 1102 | 8.167% | .402072 | .471986 | .632831 | .764631 |
| 2 | 1–3600 | 155 / 3445 | 3601–4800 | 6004–8004 | 26 / 1174 | 2.167% | .523121 | .590908 | .816796 | .778639 |
| 3 | 1–4800 | 181 / 4619 | 4801–6000 | 8005–9999 | 22 / 1178 | 1.833% | .642986 | .642253 | .682162 | .673494 |

Every planned AP was defined. AP uses continuous class-1 scores and depends on both ranking and positive prevalence. A constant score gives AP equal to the prevalence of that evaluation set. Block 1 contained 98 of the 146 ordered evaluation positives. It also had the smallest fit set and the lowest A0 APs, despite having the highest prevalence. The comparisons preserve the differences in case mix. [Ordered block figure and prevalence baseline](../results/06_robustness_analysis/figures/ordered_block_ap.png)

## Comparison and scientific limits

| Strategy summary | M1 | M2 | M3 | M4 |
| --- | ---: | ---: | ---: | ---: |
| Mean of three random-seed means | .759681 | .744191 | .857423 | .873681 |
| Mean of three ordered blocks | .522726 | .568382 | .710596 | .738921 |
| Ordered minus random | −.236955 | −.175809 | −.146827 | −.134759 |

On A0, HGB led RF for every random seed, while RF led HGB by .045656 in the ordered mean. On A4, RF had the higher mean under both strategies. HGB A4 nevertheless scored higher than RF A4 on ordered blocks 2 and 3. RF's larger advantage on block 1 accounted for its higher three-block mean. For the ordered strategy, A4 increased mean AP by .187870 for HGB and .170539 for RF. These contrasts pair scores on the same evaluation memberships. [Strategy comparison](../results/06_robustness_analysis/figures/strategy_ap.png) · [Paired feature differences](../results/06_robustness_analysis/figures/feature_set_differences.png) · [Machine-readable dispersion, rankings and differences](../results/06_robustness_analysis/robustness_summary.json)

Temperature difference and mechanical power occur in the simulator's failure rules. Explicitly providing those quantities may help the trees recover synthetic label structure across partitions. Ordered evaluation also changes training size, fit composition, evaluation prevalence and which observations receive scores. The first 2,400 rows contribute only to fitting in the ordered design. Those differences prevent attributing the AP change to any single mechanism. [Dataset methodology and attribution](dataset.md)

UDI has no verified timestamp or equipment identity. These comparisons cannot establish temporal shift, independence of equipment observations, prospective forecasting, physical causality, statistical superiority or transfer to industrial equipment. Phase 03 chose candidates using the original folds. Subsequent analysis on overlapping observations carries selection optimism. Model adoption, operational thresholds and held-out performance remain open.

## Reproduction and acceptance evidence

Scientific reproduction requires the matching training export and frozen repository inputs. The CLI creates the Phase 06 output on its first run and checks the existing inventory on subsequent calls. The verifier independently refits all 52 evaluations in a fresh directory.

```sh
uv sync --locked
uv run --locked ai4i-pm robustness
uv run --locked python scripts/verify_06_robustness.py
```

The isolated replay reproduced all 52 new APs and all 20 historical references with maximum absolute AP difference 0 (acceptance tolerance 10⁻¹²). Nine of 12 result files matched byte for byte, including both manifests, both summaries and all four figures. Measured runtimes differed in the two metric CSVs, changing their hashes in the execution record. Reference rows identify their Phase 04 origin and leave unavailable per-fit runtimes empty. [Replay verifier](../scripts/verify_06_robustness.py) · [Boundary, membership, scoring and replay tests](../tests/test_robustness.py)

### Post-implementation software verification

The November 21, 2026 check ran from a fresh temporary copy of all 152 tracked and non-ignored working files. Each copied file matched its source SHA-256. The only entries under `data/` were the two empty `.gitkeep` files. Tests that fit models used artificial fixtures. The installed package resolved to the copied source tree. The environment used uv 0.13.9, Python 3.12.16, SHAP 0.54.1, pytest 9.2.0 and Ruff 0.17.6.

```sh
uv sync --locked
uv pip install --require-hashes -r configs/05_shap_requirements.txt
uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked ruff format --check .
```

All five commands exited with status 0. Pytest reported **127 passed, 0 failed, 0 skipped, 0 errors and 4 warnings in 60.80 seconds**. All four warnings came from the artificial Phase 01 EDA test `test_eda_reads_only_training_values`. Seaborn emitted one pending deprecation warning for the colormap method `set_bad` and three Matplotlib deprecation warnings for the boxplot argument `vert`. Ruff lint passed and Ruff reported all 65 files already formatted.

After the suite, the frozen TOML and all 12 Phase 06 result files matched their pre-run SHA-256 digests in both copies. The 11 artifact hashes recorded inside `execution.json` also matched. The execution record itself retained SHA-256 `16b80a331053d2ccb4b38128318e0aabe7696da21b7c491e54998f9d401e5041`.

| Supervisor acceptance check | Evidence |
| --- | --- |
| Training isolation and preregistration | Frozen input digests and model specifications, pre-fit membership digests for both seeds, predefined ordered cuts |
| Paired comparisons | Exact UDI lists, per-fit and per-evaluation class counts, 40 random fits and 12 ordered fits, 20 historical reference rows |
| Scientific reproducibility | 72/72 scores agreed exactly, 9/12 files byte-identical, four figures inspected |
| Software quality | Complete suite passed in the observation-free copy (127 passed, 4 plotting-library warnings). Repository-wide Ruff lint and formatting passed. |
| Scientific artifact integrity after software checks | All 13 Phase 06 scientific files matched their pre-run SHA-256 digests in the working tree and the test copy. |
| Publication boundary | The supervisor authorized Phase 06 publication on private main. Final evaluation requires separate approval. |

AI4I 2020 Predictive Maintenance Dataset (2020), UCI Machine Learning Repository, CC BY 4.0. The tables, memberships and figures derive from the training partition. [Source and license](dataset.md)
