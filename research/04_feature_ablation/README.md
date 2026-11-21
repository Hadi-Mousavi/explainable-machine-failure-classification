# Feature ablation

How much do two fixed tree ensembles depend on tool wear and on direct representations of the simulator’s failure mechanisms? Six prespecified conditions used identical training folds and the saved HGB-10 and RF-11 parameters. The reference condition reproduced all ten model-selection fold scores exactly. [Study methods](../../docs/04_feature_ablation.md) · [Paired fold results](../../results/04_feature_ablation/ablation_summary.md)

| Feature condition | HGB-10 mean AP | RF-11 mean AP |
| --- | ---: | ---: |
| A0 · Six original predictors | 0.759824 | 0.747959 |
| A1 · Remove tool wear | 0.575906 | 0.607668 |
| A2 · Add temperature difference | 0.802469 | 0.818515 |
| A3 · Add mechanical power | 0.811426 | 0.832828 |
| A4 · Add both derived features | 0.848176 | 0.869823 |
| A5 · Add both and remove tool wear | 0.675385 | 0.705205 |

Removing wear reduced average precision (AP) in every paired fold for both models. Each derived feature improved AP on every fold when added to the six originals. Process minus air temperature and torque times angular speed give a tree direct access to quantities used in the simulator’s heat-dissipation and power-failure rules. The A4 forest score is exploratory. The study reused selection folds and supplies no independent evidence of industrial generalization. [UCI generator description](https://doi.org/10.24432/C5HS5C) · [Paired differences figure](../../results/04_feature_ablation/figures/paired_differences.png)

## Design and reproduction

- [Prespecified feature sets](../../configs/04_feature_ablation.toml), [feature transformations](../../src/ai4i_pm/feature_engineering.py), and [ablation runner](../../src/ai4i_pm/ablations.py)
- [Complete parameters and fold descriptors](../../results/04_feature_ablation/ablation_config.json), [fold scores](../../results/04_feature_ablation/ablation_fold_metrics.csv), and [AP comparison figure](../../results/04_feature_ablation/figures/ap_comparison.png)

With the frozen training export available, run the verifier from the project root. It repeats the 60 fixed evaluations in a temporary directory and checks scores, summaries, and paired differences against the saved artifacts. [Verifier](../../scripts/verify_04_feature_ablation.py) · [Execution record and hashes](../../results/04_feature_ablation/execution.json)

```sh
uv run --locked python scripts/verify_04_feature_ablation.py
```
