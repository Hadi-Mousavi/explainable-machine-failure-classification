# Training-fold explanation results

Each AP drop is the unperturbed AP minus AP after shuffling one original predictor. The pipeline recomputes the A4 features on every perturbation. The SD below describes five fold means. Within-fold repeat variation is reported in the CSV tables.

| Model | Original predictor | Mean AP drop | Fold SD | Fold range |
| --- | --- | ---: | ---: | ---: |
| M1 | Torque [Nm] | 0.54592 | 0.06297 | 0.47070 to 0.61856 |
| M1 | Air temperature [K] | 0.41142 | 0.03050 | 0.37724 to 0.45714 |
| M1 | Rotational speed [rpm] | 0.31939 | 0.04858 | 0.27806 to 0.37581 |
| M1 | Tool wear [min] | 0.30473 | 0.05359 | 0.24527 to 0.38033 |
| M1 | Process temperature [K] | 0.23216 | 0.05611 | 0.15501 to 0.29839 |
| M1 | Type | 0.02030 | 0.01792 | -0.00488 to 0.04383 |
| M2 | Torque [Nm] | 0.48402 | 0.08429 | 0.36880 to 0.55940 |
| M2 | Rotational speed [rpm] | 0.32157 | 0.03179 | 0.28060 to 0.36492 |
| M2 | Air temperature [K] | 0.28959 | 0.03054 | 0.25759 to 0.32585 |
| M2 | Tool wear [min] | 0.26045 | 0.02776 | 0.21595 to 0.28373 |
| M2 | Process temperature [K] | 0.14143 | 0.03288 | 0.08561 to 0.16934 |
| M2 | Type | 0.02954 | 0.02222 | -0.00779 to 0.05030 |
| M3 | Torque [Nm] | 0.64778 | 0.07104 | 0.52324 to 0.70236 |
| M3 | Rotational speed [rpm] | 0.52295 | 0.03644 | 0.49318 to 0.58328 |
| M3 | Air temperature [K] | 0.48037 | 0.04465 | 0.43769 to 0.54220 |
| M3 | Process temperature [K] | 0.30790 | 0.03494 | 0.27476 to 0.34842 |
| M3 | Tool wear [min] | 0.22644 | 0.04934 | 0.15326 to 0.27522 |
| M3 | Type | 0.01868 | 0.00935 | 0.01076 to 0.03188 |
| M4 | Torque [Nm] | 0.75515 | 0.04743 | 0.68784 to 0.81902 |
| M4 | Rotational speed [rpm] | 0.55226 | 0.06428 | 0.48664 to 0.65557 |
| M4 | Air temperature [K] | 0.47628 | 0.04726 | 0.42727 to 0.53709 |
| M4 | Process temperature [K] | 0.30788 | 0.03956 | 0.25639 to 0.34092 |
| M4 | Tool wear [min] | 0.21902 | 0.04485 | 0.15821 to 0.26250 |
| M4 | Type | 0.00876 | 0.00598 | 0.00389 to 0.01591 |

## Fold 1 attributions

The global sample was drawn without conditioning on the label. Type indicators have separate contributions. Each model uses its own additive output scale.

### M1

Mean absolute contribution in class-1 log-odds, largest first. Tool wear [min] (0.8274), Torque [Nm] (0.8136), Air temperature [K] (0.5866), Rotational speed [rpm] (0.3446), Process temperature [K] (0.3394), Type = L (0.1573), Type = H (0.0693), Type = M (0.0362).

The global sample includes 4 positives. Maximum absolute reconstruction error across global and local observations was 2.22e-14.

At the diagnostic cutoff of 0.5, the full evaluation fold contains 33 TP, 28 FP, 7 FN, 1132 TN.

| UDI | Category | Score | Baseline → output | Two largest signed contributions |
| ---: | --- | ---: | ---: | --- |
| 1168 | TP | 0.80259 | -6.6338 → 1.4026 | Tool wear [min] +5.6714, Torque [Nm] +1.6134 |
| 1497 | TP | 0.99694 | -6.6338 → 5.7864 | Torque [Nm] +11.6289, Tool wear [min] +0.5971 |
| 345 | FP | 0.93303 | -6.6338 → 2.6342 | Torque [Nm] +7.0991, Rotational speed [rpm] +2.1835 |
| 760 | FP | 0.50936 | -6.6338 → 0.0375 | Tool wear [min] +6.9568, Torque [Nm] -0.6171 |
| 1764 | FN | 0.02159 | -6.6338 → -3.8138 | Tool wear [min] +3.8024, Torque [Nm] -0.7866 |
| 3530 | FN | 0.01609 | -6.6338 → -4.1133 | Tool wear [min] +3.5134, Torque [Nm] -1.0400 |
| 11 | TN | 0.00110 | -6.6338 → -6.8132 | Type = H +0.4691, Torque [Nm] -0.4244 |
| 19 | TN | 0.00037 | -6.6338 → -7.8969 | Air temperature [K] -1.4265, Tool wear [min] -0.7076 |

Positive contributions raise the output from its baseline and negative contributions lower it. The local CSV retains every signed term and original measurement. These smallest-UDI cases describe selected errors and successes.

### M4

Mean absolute contribution in class-1 probability, largest first. Tool wear [min] (0.0190), Torque [Nm] (0.0149), Rotational speed [rpm] (0.0141), temperature_difference_K (0.0117), mechanical_power_W (0.0083), Air temperature [K] (0.0054), Process temperature [K] (0.0022), Type = L (0.0009), Type = M (0.0005), Type = H (0.0004).

The global sample includes 4 positives. Maximum absolute reconstruction error across global and local observations was 4.55e-15.

At the diagnostic cutoff of 0.5, the full evaluation fold contains 29 TP, 3 FP, 11 FN, 1157 TN.

| UDI | Category | Score | Baseline → output | Two largest signed contributions |
| ---: | --- | ---: | ---: | --- |
| 1497 | TP | 0.98333 | 0.0340 → 0.9833 | mechanical_power_W +0.5878, Torque [Nm] +0.2971 |
| 2859 | TP | 0.56333 | 0.0340 → 0.5633 | Tool wear [min] +0.4388, Torque [Nm] +0.0673 |
| 4119 | FP | 0.54667 | 0.0340 → 0.5467 | Rotational speed [rpm] +0.1882, temperature_difference_K +0.1310 |
| 5153 | FP | 0.52667 | 0.0340 → 0.5267 | Rotational speed [rpm] +0.2763, Torque [Nm] +0.2482 |
| 1168 | FN | 0.33352 | 0.0340 → 0.3335 | Tool wear [min] +0.1911, Rotational speed [rpm] +0.0725 |
| 1584 | FN | 0.38667 | 0.0340 → 0.3867 | Torque [Nm] +0.1398, Tool wear [min] +0.1250 |
| 11 | TN | 0.00000 | 0.0340 → 0.0000 | Torque [Nm] -0.0109, Tool wear [min] -0.0077 |
| 19 | TN | 0.00333 | 0.0340 → 0.0033 | Rotational speed [rpm] +0.0297, temperature_difference_K -0.0276 |

Positive contributions raise the output from its baseline and negative contributions lower it. The local CSV retains every signed term and original measurement. These smallest-UDI cases describe selected errors and successes.
