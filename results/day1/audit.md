# Original dataset integrity audit

Scope: full-source integrity verification only. Analytical EDA uses training only.
Failure-mode relationships were not examined. No observations were altered.

- CSV SHA-256: `dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e`
- Dimensions: 10000 rows x 14 columns
- Class counts: {'0': 9661, '1': 339}
- Missing cells: 0
- Invalid targets: 0
- Identifier uniqueness: {'UDI': True, 'Product ID': True}
- Type counts: {'L': 6000, 'M': 2997, 'H': 1003}
- Negative / positive ratio: 28.499

## Columns and missingness

| Column | dtype | Missing |
| --- | --- | ---: |
| UDI | int64 | 0 |
| Product ID | object | 0 |
| Type | object | 0 |
| Air temperature [K] | float64 | 0 |
| Process temperature [K] | float64 | 0 |
| Rotational speed [rpm] | int64 | 0 |
| Torque [Nm] | float64 | 0 |
| Tool wear [min] | int64 | 0 |
| Machine failure | int64 | 0 |
| TWF | int64 | 0 |
| HDF | int64 | 0 |
| PWF | int64 | 0 |
| OSF | int64 | 0 |
| RNF | int64 | 0 |

## Duplicate checks

number of repeated rows after the first occurrence

- complete_rows: 0
- predictor_combinations: 0
- predictor_plus_target: 0

## Numeric ranges (integrity)

| Feature | Min | Max |
| --- | ---: | ---: |
| Air temperature [K] | 295.3 | 304.5 |
| Process temperature [K] | 305.7 | 313.8 |
| Rotational speed [rpm] | 1168.0 | 2886.0 |
| Torque [Nm] | 3.8 | 76.6 |
| Tool wear [min] | 0.0 | 253.0 |

## Unexpected values

Positive Kelvin temperatures, speed, torque; nonnegative tool wear; finite numeric values; L/M/H Type; unique positive integer UDI; Product ID pattern [LMH]\d{5}; binary labels. No data-driven full-source outlier thresholds.

No violations of the checks above were detected.

## Leakage controls

Explicit predictors: Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min].
Excluded: UDI, Product ID, TWF, HDF, PWF, OSF, RNF.
Source-level counts/ranges are integrity checks, not inputs to feature selection.
No duplicate removal, outlier removal, imputation, or other cleaning was performed.
See docs/leakage_risks.md and audit.json for complete evidence.
