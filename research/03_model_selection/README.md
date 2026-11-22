# Model selection

Which prespecified configuration ranks synthetic failures most accurately on the original inputs? Across 27 candidates and 135 successful fold evaluations, balanced histogram gradient boosting (HGB-10) ranked first at **0.759824** mean average precision (AP). The best random forest (RF-11) reached **0.747959**. The next boosting candidate trailed the leader by only 0.001266 AP before rounding. This original-feature comparison established HGB-10 as the reference for the later held-out evaluation. [Candidate rankings](../../results/03_model_selection/model_comparison.md) · [Provisional specification](../../results/03_model_selection/selected_configuration.json)

## Method and findings

The study sampled six structures per tree family with seed 42 and evaluated each with equal and balanced class weights. Three fixed comparators completed the design. Every candidate used the baseline folds and fold-local preprocessing. The selection rule chose the highest unrounded mean fold AP among candidates with five successful scores. Balancing reduced mean AP for all six forest structures and improved it for three of six boosting structures. Reusing the folds for selection limits the interpretation of the winning score. [Methods and tie-breaking rule](../../docs/03_model_selection.md) · [Matched weight contrasts](../../results/03_model_selection/class_weight_analysis.csv)

- [Search configuration](../../configs/03_model_selection.toml), [comparison runner](../../src/ai4i_pm/comparison.py), and [search implementation](../../src/ai4i_pm/model_search.py)
- [Fold ledger](../../results/03_model_selection/candidate_fold_metrics.csv), [model family comparison](../../results/03_model_selection/figures/model_family_ap.png), and [class-weight sensitivity figure](../../results/03_model_selection/figures/weight_sensitivity.png)

## Reproduction

From the project root with the frozen training export, replay the recorded design and folds. The verifier compares fold scores, paired contrasts, and the selected configuration. Measured timings can vary between runs. [Verifier](../../scripts/verify_03_model_selection.py) · [Execution record](../../results/03_model_selection/execution.json)

```sh
uv run --locked python scripts/verify_03_model_selection.py
```
