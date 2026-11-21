# Day 4 feature ablation on frozen training folds

Adding temperature difference and mechanical power to the six original predictors
raised mean Average Precision (AP) from 0.759824 to 0.848176 for histogram gradient
boosting and from 0.747959 to 0.869823 for random forest. Both models gained on all
five paired folds. Removing Tool wear reduced AP on every fold for both models.

**The simulator uses temperature difference and mechanical power in its failure
rules.** Giving a tree either quantity directly can make a label-generation rule
easier to approximate. The experiment measures how these representations affect the
ranking of synthetic observations. It supplies no field evidence about physical
causality, prospective failure prediction, or industrial generalization.
[UCI dataset documentation](https://doi.org/10.24432/C5HS5C)

## Prespecified design

Day 4 began on clean commit `eca2f5e855750d8473e827de3396b711e2f55d6c`.
The configuration was written before fitting and retained SHA-256
`2216505b89179c8f5856dbd65cade068218121f171e12268e5e4b6a63ebe11bf`.
It specifies the exact ordered columns, the two candidates, the baseline,
input digests and paired analysis. [Frozen configuration](../configs/day4.toml)

The original inputs are Type, air temperature (K), process temperature (K),
rotational speed (rpm), torque (N·m) and Tool wear (min). The only derived inputs
are the process temperature minus air temperature in kelvins (ΔT) and mechanical
power in watts, calculated as torque × rotational speed × 2π/60.
The factor converts revolutions per minute to radians per second. Each transform
operates on measurements from a single row and preserves its UDI index.

| Condition | Original measurements | Added measurements |
| --- | --- | --- |
| A0, reference | All six | None |
| A1 | All except Tool wear | None |
| A2 | All six | ΔT |
| A3 | All six | Mechanical power |
| A4 | All six | ΔT and mechanical power |
| A5 | All except Tool wear | ΔT and mechanical power |

The primary estimator is the provisional Day 3 HGB-10. It uses learning rate 0.1,
100 boosting iterations, up to 31 leaves, L2 regularization 1.0, balanced class
weights, disabled early stopping and seed 42. Secondary RF-11 uses 300 trees,
depth 16, minimum leaf size 1, bootstrap sampling, square-root feature sampling,
equal class weights, one worker and seed 42. The runner restores every saved
estimator argument, including defaults, and checks the parameter inventory against
the locked estimator. Both use dense, fold-local one-hot encoding of Type with unseen
categories ignored, and numeric passthrough. Fits are serial with one native thread.
[Complete specifications and fold descriptors](../results/day4/ablation_config.json)

The source CSV digest was verified through frozen Day 1 provenance and the approved
training descriptor. Day 4 hashed the primary split manifest, read the checksum-verified
training export, and mapped the Day 2 evaluation UDIs to rows. The complement of
each evaluation fold supplied its 4,800 fit rows. The same 1,200 rows were scored for
each model and condition on that fold. Across all 6,000 training rows there were
203 positive labels. Folds 1–5 contained 40, 40, 41, 41 and 41 positives.
The primary validation and test observations, raw source and failure-mode indicators
were outside the input inventory. [Input and output hashes](../results/day4/execution.json)

## Paired results

AP uses continuous scores for Machine failure = 1. Each reported mean gives the
five folds equal weight. Each difference compares one condition with A0 for the
same model on the same evaluation UDIs. All ten A0 fold scores matched the Day 3
scores exactly under an absolute tolerance of 1e-12.

| Set | HGB mean AP | Sample SD | Mean Δ | Folds above A0 | RF mean AP | Sample SD | Mean Δ | Folds above A0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 | 0.759824 | 0.048779 | 0 | reference | 0.747959 | 0.077190 | 0 | reference |
| A1 | 0.575906 | 0.089596 | −0.183919 | 0/5 | 0.607668 | 0.095211 | −0.140291 | 0/5 |
| A2 | 0.802469 | 0.043064 | +0.042645 | 5/5 | 0.818515 | 0.044842 | +0.070556 | 5/5 |
| A3 | 0.811426 | 0.062891 | +0.051601 | 5/5 | 0.832828 | 0.063159 | +0.084868 | 5/5 |
| A4 | 0.848176 | 0.054924 | +0.088351 | 5/5 | 0.869823 | 0.046892 | +0.121864 | 5/5 |
| A5 | 0.675385 | 0.089986 | −0.084440 | 0/5 | 0.705205 | 0.085461 | −0.042755 | 2/5 |

The full paired table gives each AP, each difference and the positive counts.
Values above are rounded after computing all differences.
[Fold-by-fold AP and paired differences](../results/day4/ablation_summary.md)

- **A1, remove Tool wear.** Every fold lost AP. HGB showed the larger mean drop.
  Wear contributed ranking information that the remaining inputs did not recover
  at these fixed model settings.
- **A2, add ΔT.** Both models improved on every fold. Fold 1 accounted for the
  largest gain in each family (+0.140513 HGB, +0.166490 RF). For HGB the other
  gains ranged from +0.009182 to +0.037792, so the mean hides substantial variation.
- **A3, add power.** Both models gained on every fold, with a larger mean gain for RF.
  A tree can split directly on a torque–speed product when it is supplied as one input.
  Using the two original inputs can require several axis-aligned splits.
- **A4, add both.** This condition produced the highest observed mean for each
  fixed model. RF gained more than HGB on average. The pattern is consistent with
  easier access to two generator relationships. It does not identify their independent
  causal contributions.
- **A5, add both and remove wear.** Mean AP fell below A0 for both models.
  HGB lost on all five folds. RF gained on folds 1 and 4 and lost on the other
  three. The two derived quantities therefore only partly offset the average
  loss observed when wear was removed. A5 is a combined sensitivity condition.

Both engineered features are deterministic functions of existing measurements and
supply no independent measurement. Tree ensembles can already approximate differences
and products with nonlinear combinations of splits. How much an explicit feature
helps depends on the sample and estimator. For RF, keeping the square-root sampling
rule fixed also means that changing the available columns changes which predictors
can compete at a split. The experiment cannot separate these mechanisms.

There are only 40 or 41 failures in an evaluation fold, so changes in the ranks of a
few positives can affect AP. Sample SD uses denominator four and describes fold
dispersion. The fits overlap, and Day 3 chose the two models using these same folds.
Selection optimism and dependence limit inference about generalization. The five
differences provide no significance test or confidence interval. HGB-10 retains
its provisional status. Supervisor review is needed before adopting a feature set,
changing the primary model, or authorizing another phase.

## Figures and reproduction

- [AP by condition and model](../results/day4/figures/ap_comparison.png)
- [Paired fold differences from A0](../results/day4/figures/paired_differences.png)
- [Mean feature effects across models](../results/day4/figures/cross_model_effects.png)

From the project root with the approved training export, use the locked environment.

```sh
uv run --locked ai4i-pm ablations
uv run --locked python scripts/verify_day4.py
uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked ruff format --check .
```

The verifier copies only the training CSV, frozen metadata, study configurations
and lockfile into an isolated root and repeats all 60 fits. It checks the artifact
inventory and summarizes byte matches as well as numerical agreement. The runner
preserves existing results if a repeat differs. Software versions and platform are
recorded because numerical and rendering behavior can vary across environments.
[Execution record](../results/day4/execution.json)

## Local verification

With Python 3.12.16 and scikit-learn 1.9.1, the isolated Day 4 repeat reproduced
all 60 AP values and paired comparisons. All ten output files, including the three
figures and execution record, were byte-identical on this machine. Day 2 reproduced
all eleven artifacts. The Day 3 replay after sharing the tree preprocessing code
reproduced all 135 fold scores and nine of eleven files byte-for-byte. The two
timing-bearing files vary by run.

The complete local suite passed 109 tests. Four plotting-library deprecation
warnings came from the existing Day 1 EDA test. Both Ruff checks passed.
Unit tests use artificial data, and all experimental values above come from the
approved training export.
