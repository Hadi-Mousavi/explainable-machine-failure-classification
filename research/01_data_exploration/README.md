# Data exploration

Can the source support a reproducible comparison of rare-failure classifiers? The original integrity audit found 10,000 rows, 339 failures, zero missing cells, and unique row identifiers. The split assigns 6,000 rows to training and 2,000 each to validation and test. Of the training rows, 203 carry a failure label (3.38%). [Source audit](../../results/01_data_exploration/audit.md) · [Training EDA](../../results/01_data_exploration/eda.md)

## Method and findings

Acquisition preserves the source bytes. The split orders rows by UDI and stratifies twice by the binary target with seed 42. Exploration uses the checksum-verified training export. Air and process temperature had a Pearson correlation of 0.876. Rotational speed and torque had a correlation of −0.876. All observations were retained. These descriptive associations apply to the synthetic observations and establish no physical causal relationship. [Split settings](../../configs/01_data_exploration.toml) · [Split implementation](../../src/ai4i_pm/splits.py) · [EDA implementation](../../src/ai4i_pm/eda.py)

- [Source provenance](../../results/01_data_exploration/provenance.json), [split assignments](../../results/01_data_exploration/split_manifest.json), and [training descriptor](../../results/01_data_exploration/training_partition.json)
- [Notebook](../../notebooks/01_eda.ipynb), [training correlations figure](../../results/01_data_exploration/figures/05_train_numeric_correlations.png), and [all seven figures](../../results/01_data_exploration/figures)
- [Dataset methodology](../../docs/dataset.md) and [published reproducibility evidence](../../results/01_data_exploration/reproducibility.json)

## Reproduction

Use a separate reproduction checkout for the full sequence. Acquisition, audit, and split preparation require the original source. The final verifier repeats those operations, checks split invariance under reordering, and compares the resulting evidence with the published record in a temporary directory. For training-only reproduction with an existing export and manifests, the EDA command can run on its own.

```sh
uv run --locked ai4i-pm acquire
uv run --locked ai4i-pm audit
uv run --locked ai4i-pm split
uv run --locked ai4i-pm eda
uv run --locked python scripts/verify_01_data_exploration.py
```

If automatic acquisition is unavailable, supply the matching official archive using `uv run --locked ai4i-pm acquire --manual PATH_TO_ZIP`. For the optional notebook, run `uv run --locked --group notebook python scripts/execute_notebook.py`. [Acquisition implementation](../../src/ai4i_pm/data.py) · [Notebook runner](../../scripts/execute_notebook.py)
