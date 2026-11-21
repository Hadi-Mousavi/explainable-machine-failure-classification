# Dataset attribution and provenance

AI4I 2020 Predictive Maintenance Dataset [Dataset]. (2020).
UCI Machine Learning Repository.
[DOI 10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C).

Official landing page:
[UCI dataset 601](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset).
Official archive:
[UCI download](https://archive.ics.uci.edu/static/public/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset.zip).
Only this official source is supported; no third-party mirrors are used.

Dataset license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The repository's MIT license applies to the software. The dataset attribution
follows the citation supplied by UCI.

The original CSV is not rewritten. Acquisition retains the original archive,
reads its sole `ai4i2020.csv` member in memory, verifies names/order/dimensions,
and records exact original-file SHA-256 hashes and acquisition UTC time in
`results/01_data_exploration/provenance.json`. Acquisition restores missing local files only
when their digests match the recorded source. Existing conflicting files are
refused. Verification preserves the original acquisition record.
Manual imports explicitly record user-attested official provenance; local
structure and checksums alone cannot authenticate a manually supplied source.

Raw data and the training-only CSV export are ignored by Git. Audit summaries,
split manifests, and EDA figures are derived project outputs.
The training export preserves
selected values but changes row selection and serialization; it is not the
original raw CSV. No cleaning, deduplication, imputation, or outlier removal is
performed. See the [integrity audit](../results/01_data_exploration/audit.md) for source counts.

The dataset is synthetic. Its six predictors approximate industrial measurements
but do not demonstrate equipment-level generalization or future forecasting.
The documentation sometimes names the identifier UID; the original CSV's UDI is
the enforced source spelling. The original failure-mode columns remain in raw
storage but are excluded from the training export and from Phase 01 EDA.

The acquired file has Type counts L=6000, M=2997, H=1003. The UCI narrative
describes nominal proportions L/M/H=50/30/20%, whereas the verified CSV has
60/29.97/10.03%. This documentation discrepancy is recorded rather than
"corrected"; the original file and all observations remain unchanged. These
counts are source-integrity evidence and are not a feature-selection criterion.
