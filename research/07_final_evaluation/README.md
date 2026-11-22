# Phase 07 · Final held-out evaluation

The primary RF-11 forest with both engineered features (A4) achieved AP 0.879927 on the reserved 2,000 operations. At the approved cutoff of 0.15, it detected 58 of 68 failures with 42 false positives. The sample also contained 1,890 true negatives and ten missed failures. HGB-10 with the original inputs (A0) served as the descriptive reference at 0.50. [Technical report](../../report/final_report.md) · [Exact results for both systems and the forest diagnostic](../../results/07_final_evaluation/test_metrics.json)

| Stage | Decision and evidence |
| --- | --- |
| Design | The protocol names the primary and reference systems, comparison metric, threshold rule and uncertainty procedure. Its approval record documents the earlier 07A scope. [Protocol](../../docs/07_final_evaluation_protocol.md) · [Approval record](../../results/07_final_evaluation/approved_protocol_record.json) |
| Training | Both pipelines were fitted on the 6,000 original training operations. The fit manifest binds the feature lists, parameters and serialized model digests. [Fit manifest](../../results/07_final_evaluation/fit_manifest.json) |
| Validation | The primary cutoff of 0.15 uniquely maximized F2 on the approved 19-point grid. It gave 61 true positives, 32 false positives and seven false negatives in 2,000 validation operations. [Grid](../../results/07_final_evaluation/validation_threshold_grid.csv) · [Decision](../../results/07_final_evaluation/validation_decision.json) · [Decision digest](../../results/07_final_evaluation/validation_decision.sha256) |
| Test | The execution freeze was committed at `47cc09cd1e91035472df965507ce5e4dee61527a`. The manifest records one score pass per persisted pipeline and the SHA-256 digests of six output files. [Execution freeze](../../configs/07b_execution_freeze.json) · [Execution manifest](../../results/07_final_evaluation/execution_manifest.json) |
| Uncertainty | The 2,000 class-stratified paired bootstrap replicates gave 95% percentile intervals of [0.810543, 0.941731] for primary AP and [0.764706, 0.926471] for primary recall. These condition on the fitted forest, cutoff and class counts. [Bootstrap record](../../results/07_final_evaluation/bootstrap_intervals.json) |

The generated [execution report](../../results/07_final_evaluation/final_report.md) is itself checksum-listed. The expanded [portfolio report](../../report/final_report.pdf) interprets the results across all seven phases.

## Verify the saved release

Run this read-only check from the repository root with Python 3.12. It checks all six recorded output hashes and the validation decision against the execution manifest. The independently observed digest of the manifest is in the [publication inventory](../../report/publication_inventory.md).

```python
import hashlib
import json
from pathlib import Path

folder = Path("results/07_final_evaluation")
manifest = json.loads((folder / "execution_manifest.json").read_text(encoding="utf-8"))
expected = dict(manifest["output_sha256"])
expected["validation_decision.json"] = manifest["decision_sha256"]
for name, digest in expected.items():
    actual = hashlib.sha256((folder / name).read_bytes()).hexdigest()
    if actual != digest:
        raise ValueError(f"SHA-256 mismatch for {name}")
print(f"Verified {len(expected)} files")
```

The [curve coordinates](../../results/07_final_evaluation/primary_pr_curve.csv) describe aggregate precision and recall at score thresholds. The [original figure](../../results/07_final_evaluation/primary_pr_curve.png) displays the continuous ranking on the test partition. The saved files support inspection without access to test observations.
