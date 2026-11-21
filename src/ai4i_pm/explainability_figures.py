"""Figures and measured summaries for the prespecified explanation study."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK = "#245b73"
WARM = "#b45232"


def display_name(name: str) -> str:
    return name.removeprefix("numeric__").removeprefix("type__").replace("Type_", "Type = ")


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, metadata={"Software": "ai4i-pm"})
    plt.close(fig)


def permutation_figure(folder: Path, folds: pd.DataFrame, overall: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, layout="constrained")
    for ax, (key, table) in zip(axes.flat, overall.groupby("model"), strict=True):
        names = table.predictor.tolist()
        for i, name in enumerate(names):
            values = folds.loc[(folds.model == key) & (folds.predictor == name), "mean_drop"]
            ax.scatter(values, i + np.linspace(-0.16, 0.16, len(values)), s=15, color="#90b3c0")
        ax.scatter(table.mean_drop, range(len(names)), marker="D", color=INK, s=27)
        ax.axvline(0, color="0.7", lw=1)
        ax.set(
            yticks=range(len(names)),
            yticklabels=names,
            title={
                "M1": "M1 · HGB-10 + A0",
                "M2": "M2 · RF-11 + A0",
                "M3": "M3 · HGB-10 + A4",
                "M4": "M4 · RF-11 + A4",
            }[key],
        )
        ax.invert_yaxis()
    fig.suptitle("Loss of average precision after permuting an original predictor")
    fig.supxlabel("AP drop  ·  dots = five fold means  /  diamond = overall mean")
    save(fig, folder / "permutation.png")


def shap_figures(
    folder: Path, key: str, batch: dict, ranking: pd.DataFrame, local: pd.DataFrame
) -> None:
    scale = ranking.scale.iloc[0]
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.barh([display_name(x) for x in ranking.feature], ranking.mean_abs_shap, color=INK)
    ax.invert_yaxis()
    ax.set(
        title=f"{key} · Global attribution on {batch['global_mask'].sum()} fold-1 observations",
        xlabel=f"Mean absolute SHAP contribution ({scale})",
    )
    save(fig, folder / f"{key}_global.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout="constrained")
    mask = batch["global_mask"]
    for ax, name in zip(axes, ranking.feature.iloc[:2], strict=True):
        j = batch["features"].tolist().index(name)
        ax.scatter(
            batch["transformed"][mask, j], batch["values"][mask, j], color=INK, s=12, alpha=0.55
        )
        ax.axhline(0, lw=1, color="0.7")
        ax.set_xlabel(display_name(name))
    axes[0].set_ylabel(f"Signed SHAP contribution ({scale})")
    fig.suptitle(f"{key} · Attributions at observed feature values")
    save(fig, folder / f"{key}_effects.png")
    groups = list(local.groupby("UDI", sort=False))
    fig, axes = plt.subplots(
        (len(groups) + 1) // 2,
        2,
        figsize=(12, 2.25 * ((len(groups) + 1) // 2)),
        layout="constrained",
        squeeze=False,
    )
    for ax in axes.flat[len(groups) :]:
        ax.set_visible(False)
    for ax, (udi, group) in zip(axes.flat, groups, strict=False):
        top = group.loc[group.signed_shap.abs().nlargest(3).index].sort_values("signed_shap")
        ax.barh(
            [display_name(x) for x in top.feature],
            top.signed_shap,
            color=[WARM if x > 0 else INK for x in top.signed_shap],
        )
        ax.axvline(0, color="0.6", lw=0.8)
        row = group.iloc[0]
        ax.set_title(f"{row.category} · UDI {udi} · p = {row.score:.3f}", loc="left", fontsize=10)
    fig.suptitle(f"{key} · Three largest signed contributions per diagnostic case")
    fig.supxlabel(f"Contribution to {scale} (individual panel scales)")
    save(fig, folder / f"{key}_local.png")


def write_summary(
    folder: Path, overall: pd.DataFrame, meta: dict, rankings: list, cases: list
) -> None:
    lines = [
        "# Training-fold explanation results",
        "",
        "Each AP drop is the unperturbed AP minus AP after shuffling one original predictor. "
        "The pipeline recomputes the A4 features on every perturbation. The SD below describes "
        "five fold means. Within-fold repeat variation is reported in the CSV tables.",
        "",
        "| Model | Original predictor | Mean AP drop | Fold SD | Fold range |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in overall.itertuples():
        lines.append(
            f"| {row.model} | {row.predictor} | {row.mean_drop:.5f} | {row.fold_sd:.5f} | "
            f"{row.fold_min:.5f} to {row.fold_max:.5f} |"
        )
    lines.extend(
        [
            "",
            "## Fold 1 attributions",
            "",
            "The global sample was drawn without conditioning on the label. Type indicators "
            "have separate contributions. Each model uses its own additive output scale.",
        ]
    )
    for key, record in meta.items():
        if record["status"] == "BLOCKED":
            lines.extend(["", f"**{key} BLOCKED.** {record['error_type']}: {record['error']}"])
    for ranking, local in zip(rankings, cases, strict=True):
        key = ranking.model.iloc[0]
        record = meta[key]
        lines.extend(
            [
                "",
                f"### {key}",
                "",
                f"Mean absolute contribution in {record['scale']}, largest first. "
                + ", ".join(
                    f"{display_name(r.feature)} ({r.mean_abs_shap:.4f})"
                    for r in ranking.itertuples()
                )
                + ".",
                "",
                f"The global sample includes {record['global_positives']} positives. "
                f"Maximum absolute reconstruction error across global and local observations "
                f"was {record['max_absolute_error']:.3g}.",
                "",
                "At the diagnostic cutoff of 0.5, the full evaluation fold contains "
                + ", ".join(f"{n} {category}" for category, n in record["confusion_counts"].items())
                + ".",
                "",
                "| UDI | Category | Score | Baseline → output | Two largest signed contributions |",
                "| ---: | --- | ---: | ---: | --- |",
            ]
        )
        for udi, group in local.groupby("UDI", sort=False):
            row = group.iloc[0]
            top = group.loc[group.signed_shap.abs().nlargest(2).index]
            effects = ", ".join(
                f"{display_name(r.feature)} {r.signed_shap:+.4f}" for r in top.itertuples()
            )
            lines.append(
                f"| {udi} | {row.category} | {row.score:.5f} | {row.base:.4f} → "
                f"{row.output:.4f} | {effects} |"
            )
        lines.extend(
            [
                "",
                "Positive contributions raise the output from its baseline and negative "
                "contributions lower it. The local CSV retains every signed term and original "
                "measurement. These smallest-UDI cases describe selected errors and successes.",
            ]
        )
    (folder / "explanation_summary.md").write_bytes(("\n".join(lines) + "\n").encode())
