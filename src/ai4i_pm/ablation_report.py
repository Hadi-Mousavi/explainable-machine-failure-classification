"""Paired descriptive summaries and figures for the six training conditions."""

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ai4i_pm.comparison_figures import BLUE, ORANGE

LABELS = ("Original", "Remove wear", "Add ΔT", "Add power", "Add both", "Both without wear")
MODELS = {"hist_gradient_boosting-10": ("HGB-10", BLUE), "random_forest-11": ("RF-11", ORANGE)}


def paired_summary(metrics, reference, ddof):
    keys = ["candidate_id", "fold", "eval_udis_sha256", "eval_positives"]
    baseline = metrics[metrics.condition == reference][keys + ["ap"]].rename(
        columns={"ap": "reference_ap"}
    )
    pairs = metrics.merge(baseline, on=keys, how="left", validate="many_to_one")
    if (
        pairs.reference_ap.isna().any()
        or pairs.duplicated(["condition", "candidate_id", "fold"]).any()
    ):
        raise ValueError("Each score needs a unique matched A0 fold")
    pairs["delta_ap"] = pairs.ap - pairs.reference_ap
    rows = []
    for (condition, candidate), part in pairs.groupby(["condition", "candidate_id"], sort=False):
        part = part.sort_values("fold")
        if part.fold.tolist() != list(range(1, 6)) or not np.isfinite(part.ap).all():
            raise ValueError("Expected five finite AP measurements per condition and model")
        rows.append(
            {
                "condition": condition,
                "candidate_id": candidate,
                "mean_ap": part.ap.mean(),
                "sample_sd": part.ap.std(ddof=ddof),
                "mean_delta_ap": part.delta_ap.mean(),
                "folds_higher": int((part.delta_ap > 0).sum()),
                "folds_equal": int((part.delta_ap == 0).sum()),
                "folds_lower": int((part.delta_ap < 0).sum()),
                **{f"ap_fold_{r.fold}": r.ap for r in part.itertuples()},
                **{f"delta_fold_{r.fold}": r.delta_ap for r in part.itertuples()},
                **{f"positives_fold_{r.fold}": r.eval_positives for r in part.itertuples()},
            }
        )
    return rows, pairs


def write_markdown(path, rows):
    lines = [
        "# Training CV ablations",
        "",
        "Each fold has equal weight. Differences compare the same model and evaluation UDIs "
        "with A0. Sample SD describes dispersion across the five folds.",
        "",
        "| Set | Model | F1 AP | F2 AP | F3 AP | F4 AP | F5 AP | Mean AP | Sample SD | Mean Δ |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        scores = [row[f"ap_fold_{k}"] for k in range(1, 6)]
        values = [*scores, row["mean_ap"], row["sample_sd"]]
        cells = [row["condition"], MODELS[row["candidate_id"]][0], *[f"{v:.6f}" for v in values]]
        lines.append("| " + " | ".join([*cells, f"{row['mean_delta_ap']:+.6f}"]) + " |")
    lines += [
        "",
        "## Paired AP differences from A0",
        "",
        "| Set | Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        cells = [row["condition"], MODELS[row["candidate_id"]][0]]
        cells.extend(f"{row[f'delta_fold_{k}']:+.6f}" for k in range(1, 6))
        lines.append("| " + " | ".join(cells) + " |")
    counts = ", ".join(str(rows[0][f"positives_fold_{k}"]) for k in range(1, 6))
    lines += ["", f"Positive observations in folds 1–5, respectively, were {counts}."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def save_figures(folder, pairs, summary):
    folder.mkdir()
    ticks = [f"A{i}  {label}" for i, label in enumerate(LABELS)]
    for column, title, xlabel, name in [
        ("ap", "Average precision by feature set", "Average precision", "ap_comparison.png"),
        ("delta_ap", "Paired fold differences from A0", "AP difference", "paired_differences.png"),
    ]:
        fig = Figure(figsize=(10, 4.5), layout="constrained")
        FigureCanvasAgg(fig)
        axes = fig.subplots(1, 2, sharex=True, sharey=True)
        for ax, (candidate, (label, color)) in zip(axes, MODELS.items(), strict=True):
            for i, (_, part) in enumerate(
                pairs[pairs.candidate_id == candidate].groupby("condition")
            ):
                values = part[column].to_numpy()
                ax.scatter(
                    values,
                    i + np.linspace(-0.16, 0.16, len(values)),
                    s=17,
                    color="#8195a5",
                    label="Five folds" if i == 0 else None,
                )
                ax.scatter(
                    values.mean(),
                    i,
                    marker="D",
                    color=color,
                    s=42,
                    label="Mean" if i == 0 else None,
                )
            if column == "delta_ap":
                ax.axvline(0, color="#555555", lw=1)
            else:
                ax.set_xlim(0, 1)
            ax.set(title=label, xlabel=xlabel, yticks=range(6), yticklabels=ticks)
            ax.grid(axis="x", alpha=0.2)
            ax.set_axisbelow(True)
        axes[0].invert_yaxis()
        fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=2)
        fig.suptitle(f"{title} · Training CV\nA0 uses the six original predictors")
        fig.savefig(folder / name, dpi=150)

    fig = Figure(figsize=(8.5, 4.2), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.axvline(0, color="#555555", lw=1)
    for shift, (candidate, (label, color)) in zip((-0.10, 0.10), MODELS.items(), strict=True):
        values = summary[summary.candidate_id == candidate].sort_values("condition").mean_delta_ap
        ax.scatter(values[1:], np.arange(1, 6) + shift, color=color, label=label, s=50)
    ax.set(
        yticks=range(1, 6),
        yticklabels=ticks[1:],
        xlabel="Mean paired AP difference from A0",
        title="Response to each feature change · Training CV",
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    ax.legend()
    fig.savefig(folder / "cross_model_effects.png", dpi=150)
