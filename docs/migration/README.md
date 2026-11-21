# Provenance of the phase migration

The four experiments were first recorded under implementation-day identifiers.
Each row below identifies the commit that introduced the evidence for a phase.
The complete approved checkpoint is `92f72a0db9fef8dca30197833a719c9708bef00e`.

| Original identifier | Current phase | Original commit |
| --- | --- | --- |
| day1 | 01_data_exploration | `5779ab41adb0aa5911fb6b4a12070485b8214b05` |
| day2 | 02_baseline_evaluation | `8a59b0172d8b1a69d4872f94ed97a2c8ecb70a65` |
| day3 | 03_model_selection | `eca2f5e855750d8473e827de3396b711e2f55d6c` |
| day4 | 04_feature_ablation | `92f72a0db9fef8dca30197833a719c9708bef00e` |

The ledger covers the four configurations, 47 result files, two source files,
the training export and the environment lock. It lists each original path and
SHA-256 beside the migrated path, digest and reason for any byte change. The
original tracked bytes are available with `git show <checkpoint>:<original_path>`.
The provenance record identifies the official source for the locally restored
archive and CSV. [Artifact ledger](artifacts.csv) · [Dataset provenance](../../results/01_data_exploration/provenance.json)

## Dependencies

1. Data exploration binds the original CSV to a seed-42 primary split and an
   ordered training export. The baseline configuration pins all three digests.
2. The baseline fold manifest assigns evaluation UDIs and defines each fit set
   by complement. The search configuration pins that manifest, the baseline
   configuration and the baseline fold scores. All candidate fits share these folds.
3. The ablation configuration pins the selection, candidate table and fold
   ledger, in addition to the training inputs. It identifies HGB-10 and RF-11
   and specifies the ordered predictors for A0 through A5.
4. Each execution record inventories the inputs and outputs for its experiment.
   The migration updates dependent digests after changes to a path, configuration
   field name or report label. Original timings, environment versions, acquisition
   time and observations are preserved. [Baseline record](../../results/02_baseline_evaluation/execution.json) · [Selection record](../../results/03_model_selection/execution.json) · [Ablation record](../../results/04_feature_ablation/execution.json)

## Interpreting a replay

All score tables, predictions, UDI assignments, predictor specifications,
figures and the provisional selection retain their approved bytes. The baseline
verifier compares the full output by SHA-256. The search and ablation verifiers
use the unrounded fold tables with zero relative tolerance and an absolute
tolerance of `1e-12` for floating-point comparisons. This allows small numerical
round-off at unit-scale AP. Candidate identities, row assignments, feature
ordering and parameter values require exact agreement. Search timings are
measured afresh and rendering can depend on the platform. [Replay commands](../../README.md#replay-the-published-training-studies)
