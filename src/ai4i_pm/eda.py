"""Reproducible descriptive EDA with a frozen training-only entry point."""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from ai4i_pm.audit import counts
from ai4i_pm.schema import NUMERIC_FEATURES, PREDICTORS, TARGET, predictor_matrix
from ai4i_pm.splits import load_training
from ai4i_pm.utils import environment_versions, sha256_file, write_json

PALETTE = ["#256A8A", "#C75A3A"]


def generate_eda(root: Path) -> dict[str, Any]:
    """Load only the training artifact, create figures and save measured observations."""
    train = load_training(root)
    features = predictor_matrix(train)
    output = root / "results/day1/figures"
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15, palette=PALETTE)
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "axes.titleweight": "bold"})
    figure_names: list[str] = []

    def save(fig: plt.Figure, name: str) -> None:
        fig.savefig(output / name, dpi=300, bbox_inches="tight", metadata={"Software": "ai4i-pm"})
        plt.close(fig)
        figure_names.append(name)

    fig, ax = plt.subplots(figsize=(6.8, 4.3), layout="constrained")
    class_counts = train[TARGET].value_counts().reindex([0, 1], fill_value=0)
    bars = ax.bar(["No failure (0)", "Failure (1)"], class_counts, color=PALETTE)
    ax.bar_label(bars, labels=[f"{n:,} ({n / len(train):.2%})" for n in class_counts], padding=5)
    ax.set(
        ylim=(0, class_counts.max() * 1.16),
        ylabel="Training observations",
        title="Training class distribution",
    )
    save(fig, "01_train_class_distribution.png")

    fig, ax = plt.subplots(figsize=(6.8, 4.3), layout="constrained")
    type_counts = features["Type"].value_counts().reindex(["L", "M", "H"], fill_value=0)
    bars = ax.bar(type_counts.index, type_counts, color="#256A8A")
    ax.bar_label(bars, padding=4)
    ax.set(
        xlabel="Product quality type",
        ylabel="Training observations",
        title="Training Type distribution",
        ylim=(0, type_counts.max() * 1.12),
    )
    save(fig, "02_train_type_distribution.png")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), layout="constrained")
    for ax, name in zip(axes.flat, NUMERIC_FEATURES, strict=False):
        sns.histplot(features, x=name, bins=35, ax=ax, color=PALETTE[0])
        ax.set(ylabel="Training observations", title=name.split(" [")[0])
    axes.flat[-1].set_visible(False)
    fig.suptitle("Numeric feature distributions — training only")
    save(fig, "03_train_numeric_distributions.png")

    analysis_columns = [*PREDICTORS, TARGET]
    missing = train.loc[:, analysis_columns].isna().sum()
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    bars = ax.barh(missing.index, missing, color=PALETTE[0])
    ax.bar_label(bars, padding=4)
    ax.set(
        xlabel="Missing training values",
        xlim=(0, max(1, int(missing.max()) * 1.15)),
        title="Training missingness (predictors and target)",
    )
    save(fig, "04_train_missingness.png")

    correlation = features.loc[:, list(NUMERIC_FEATURES)].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(8.7, 7), layout="constrained")
    sns.heatmap(
        correlation,
        vmin=-1,
        vmax=1,
        center=0,
        cmap="vlag",
        annot=True,
        fmt=".2f",
        square=True,
        ax=ax,
        cbar_kws={"label": "Pearson r"},
    )
    ax.set(title="Training numeric correlations (descriptive associations)")
    save(fig, "05_train_numeric_correlations.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    sns.scatterplot(
        features,
        x="Air temperature [K]",
        y="Process temperature [K]",
        s=9,
        alpha=0.25,
        linewidth=0,
        ax=axes[0],
    )
    sns.scatterplot(
        features,
        x="Rotational speed [rpm]",
        y="Torque [Nm]",
        s=9,
        alpha=0.25,
        linewidth=0,
        ax=axes[1],
    )
    fig.suptitle("Prespecified feature relationships — training only")
    save(fig, "06_train_feature_relationships.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), layout="constrained")
    for ax, name in zip(
        axes, ("Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"), strict=True
    ):
        sns.boxplot(train, x=TARGET, y=name, order=[0, 1], color=PALETTE[0], fliersize=2, ax=ax)
        ax.set(xlabel="Machine failure (0 = no, 1 = yes)")
    fig.suptitle("Selected feature distributions by target — training only")
    save(fig, "07_train_features_by_target.png")

    numeric_summary = features.loc[:, list(NUMERIC_FEATURES)].describe().to_dict()
    unusual = {}
    for name in NUMERIC_FEATURES:
        values = features[name]
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        unusual[name] = {
            "lower_fence": float(lower),
            "upper_fence": float(upper),
            "outside_fences": int(((values < lower) | (values > upper)).sum()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    # Only numeric allowed predictors are correlated; no failure-mode columns are loaded.
    temp_r = float(correlation.loc["Air temperature [K]", "Process temperature [K]"])
    speed_torque_r = float(correlation.loc["Rotational speed [rpm]", "Torque [Nm]"])
    observations = [
        f"Training failures: {int(class_counts.loc[1])}/{len(train)} ({train[TARGET].mean():.2%}); "
        "rare positives motivate the prespecified imbalance-aware evaluation plan.",
        f"Air/process temperature Pearson r = {temp_r:.3f}; speed/torque r = {speed_torque_r:.3f}. "
        "These are descriptive associations, not evidence of physical causality.",
        f"Training missing cells across the allowed predictors and target: {int(missing.sum())}.",
        "IQR fences flag distribution tails for review; they are not domain-validity boundaries. "
        "All observations were retained and no imputation or feature selection was performed.",
        "Target-conditioned boxplots describe unequal group sizes; they are neither model "
        "performance estimates nor causal effects.",
    ]
    summary = {
        **train.attrs,
        "environment": environment_versions(),
        "rows": len(train),
        "predictors": list(PREDICTORS),
        "class_counts": counts(train[TARGET]),
        "type_counts": counts(features["Type"]),
        "missing_by_column": {name: int(n) for name, n in missing.items()},
        "numeric_summary": numeric_summary,
        "pearson_correlations": correlation.to_dict(),
        "unusual_values_iqr_review": unusual,
        "observations": observations,
        "figures": {name: sha256_file(output / name) for name in figure_names},
        "failure_modes_examined": False,
        "models_trained": False,
    }
    write_json(root / "results/day1/eda_summary.json", summary)
    lines = [
        "# Training-only exploratory analysis",
        "",
        f"Scope: {len(train)} frozen training observations. No validation/test values were read.",
        f"Dataset SHA-256: `{train.attrs['dataset_sha256']}`.",
        f"Split manifest SHA-256: `{train.attrs['split_manifest_sha256']}`.",
        "",
        "## Measured observations",
        "",
        *(f"- {line}" for line in observations),
        "",
        "## Potentially unusual values (training IQR rule)",
        "",
        "| Feature | Minimum | Maximum | Lower fence | Upper fence | Outside fences |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in unusual.items():
        lines.append(
            f"| {name} | {values['min']:.2f} | {values['max']:.2f} | "
            f"{values['lower_fence']:.2f} | {values['upper_fence']:.2f} | "
            f"{values['outside_fences']} |"
        )
    lines += [
        "",
        "## Figures",
        "",
        *(f"- [figures/{name}](figures/{name})" for name in figure_names),
    ]
    lines += [
        "",
        "These synthetic observation-level patterns establish neither real-world generalization",
        "nor future failure forecasting. No failure-mode comparisons, models, or thresholds",
        "were used.",
    ]
    (root / "results/day1/eda.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    return summary
