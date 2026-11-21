"""Training-only baseline figures and the exact fold-level PR curve coordinates."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import precision_recall_curve

from ai4i_pm.schema import TARGET

NAMES = {"dummy_prior": "Dummy prior", "logistic_regression": "Logistic regression"}
COLORS = ["#365f93", "#bd6325"]


def save_figures(directory: Path, metrics: pd.DataFrame, oof: pd.DataFrame) -> None:
    figures = directory / "figures"
    figures.mkdir()
    fig, ax = plt.subplots(figsize=(6.8, 3.4), layout="constrained")
    for row, (name, color) in enumerate(zip(NAMES, COLORS, strict=True)):
        values = metrics.loc[metrics.model == name, "average_precision"]
        ax.scatter(
            values,
            row + np.linspace(-0.10, 0.10, len(values)),
            color=color,
            alpha=0.65,
            label="Evaluation fold" if row == 0 else None,
        )
        ax.scatter(
            values.mean(),
            row,
            marker="D",
            s=65,
            color="black",
            label="Five-fold mean" if row == 0 else None,
        )
    ax.set(
        yticks=[0, 1],
        yticklabels=list(NAMES.values()),
        ylim=(-0.4, 1.5),
        xlim=(0, 1),
        xlabel="Average precision",
        title="Failure ranking across training folds",
    )
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="lower right")
    fig.savefig(figures / "baseline_ap.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(9, 4.1), layout="constrained")
    curves = []
    for ax, (model, label) in zip(axes, NAMES.items(), strict=True):
        for fold, rows in oof[oof.model == model].groupby("fold"):
            p, r, thresholds = precision_recall_curve(rows[TARGET], rows.positive_score)
            curves.append(
                pd.DataFrame(
                    {
                        "model": model,
                        "fold": fold,
                        "precision": p,
                        "recall": r,
                        "threshold": np.r_[thresholds, np.nan],
                    }
                )
            )
            ax.step(r, p, where="post", linewidth=1.3, alpha=0.8, label=f"Fold {fold}")
        ax.axhline(oof[TARGET].mean(), color="black", linestyle=":", label="Train prevalence")
        ax.set(title=label, xlabel="Recall", xlim=(-0.015, 1.015), ylim=(-0.015, 1.035))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel("Precision")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Precision–recall curves on the five training evaluation folds")
    fig.savefig(figures / "precision_recall.png", dpi=180)
    plt.close(fig)
    pd.concat(curves).to_csv(directory / "pr_curves.csv", index=False, lineterminator="\n")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
    for ax, (model, label) in zip(axes, NAMES.items(), strict=True):
        counts = metrics.loc[metrics.model == model, ["tn", "fp", "fn", "tp"]].sum()
        matrix = counts.to_numpy().reshape(2, 2)
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=oof[TARGET].eq(0).sum() / len(NAMES))
        for (i, j), count in np.ndenumerate(matrix):
            ax.text(
                j,
                i,
                f"{count:,}",
                ha="center",
                va="center",
                fontsize=14,
                color="white" if count > matrix.max() / 2 else "#172533",
            )
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["No failure", "Failure"],
            yticklabels=["No failure", "Failure"],
            xlabel="Predicted label",
            title=label,
        )
    axes[0].set_ylabel("Actual label")
    threshold = metrics.diagnostic_threshold.iloc[0]
    fig.suptitle(f"Training OOF counts at the fixed diagnostic threshold of {threshold}")
    fig.savefig(figures / "diagnostic_confusion.png", dpi=180)
    plt.close(fig)
