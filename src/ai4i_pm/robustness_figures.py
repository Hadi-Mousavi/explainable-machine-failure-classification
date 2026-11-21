"""Descriptive summaries and figures for the fixed robustness design."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ai4i_pm.utils import write_json

LABELS = {"M1": "HGB · A0", "M2": "RF · A0", "M3": "HGB · A4", "M4": "RF · A4"}
COLORS = {"M1": "#245b73", "M2": "#a14f31", "M3": "#15766c", "M4": "#704397"}


def describe(group):
    means = {}
    stats = {}
    for key, entries in group.groupby("model"):
        ap = entries.ap
        complete = ap.notna().all()
        means[key] = float(ap.mean()) if complete else None
        stats[key] = {
            "mean": means[key],
            "sd": float(ap.std(ddof=1)) if complete else None,
            "min": float(ap.min()) if complete else None,
            "max": float(ap.max()) if complete else None,
            "defined": int(ap.notna().sum()),
            "planned": len(ap),
        }
    ranks = pd.Series(means, dtype=float).rank(method="min", ascending=False)
    return {
        "models": stats,
        "rank": {k: int(v) if pd.notna(v) else None for k, v in ranks.items()},
        "a4_minus_a0": {
            family: means[b] - means[a] if None not in (means[b], means[a]) else None
            for family, a, b in (("HGB", "M1", "M3"), ("RF", "M2", "M4"))
        },
    }


def display(value):
    return "undefined" if value is None or pd.isna(value) else f"{value:.6f}"


def summarize(random, blocks, folder):
    groups = {str(seed): describe(part) for seed, part in random.groupby("seed")}
    ordered = describe(blocks)
    comparisons = {}
    for key in LABELS:
        means = [group["models"][key]["mean"] for group in groups.values()]
        all_cv = float(np.mean(means)) if None not in means else None
        ordered_ap = ordered["models"][key]["mean"]
        comparisons[key] = {
            "seed_mean_range": max(means) - min(means) if None not in means else None,
            "mean_across_seeds": all_cv,
            "ordered_minus_random": ordered_ap - all_cv
            if None not in (ordered_ap, all_cv)
            else None,
        }
    summary = {"random_cv": groups, "udi_ordered": ordered, "comparisons": comparisons}
    write_json(folder / "robustness_summary.json", summary)
    lines = [
        "# Training-partition robustness results",
        "",
        "Equal-weight mean fold or block average precision (AP). Δ is the paired A4 minus A0 mean.",
        "",
        "| Assignment | M1 | M2 | M3 | M4 | HGB Δ | RF Δ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, group in [*((f"CV seed {s}", g) for s, g in groups.items()), ("UDI order", ordered)]:
        values = [group["models"][k]["mean"] for k in LABELS] + list(group["a4_minus_a0"].values())
        lines.append("| " + label + " | " + " | ".join(map(display, values)) + " |")
    lines += [
        "",
        "| Ordered block | Fit positives / rows | Evaluation positives / rows | Prevalence | "
        "M1 AP | M2 AP | M3 AP | M4 AP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, part in blocks.groupby("fold_or_block"):
        row = part.iloc[0]
        scores = part.set_index("model").ap
        lines.append(
            f"| {i} | {row.fit_positives}/{row.fit_rows} | {row.eval_positives}/{row.eval_rows} | "
            + " | ".join(map(display, [row.prevalence_baseline, *scores.loc[list(LABELS)]]))
            + " |"
        )
    lines += [
        "",
        "The random assignments reuse training observations. Ordered blocks vary in fit size, "
        "evaluation composition and prevalence. UDI has no verified time or equipment "
        "interpretation.",
        "",
        "[Fold metrics](random_split_fold_metrics.csv) · "
        "[Ordered metrics](ordered_block_metrics.csv) · "
        "[Dispersion, rankings and contrasts](robustness_summary.json)",
    ]
    (folder / "robustness_summary.md").write_bytes(("\n".join(lines) + "\n").encode())
    return summary


def figures(random, blocks, summary, folder):
    output = folder / "figures"
    output.mkdir()

    def save(fig, name):
        fig.savefig(output / name, dpi=160, metadata={"Software": "ai4i-pm"})
        plt.close(fig)

    seeds = sorted(random.seed.unique())
    fig, axes = plt.subplots(1, 4, sharey=True, figsize=(11, 4), layout="constrained")
    for ax, (key, label) in zip(axes, LABELS.items(), strict=True):
        for j, seed in enumerate(seeds):
            values = random.loc[(random.model == key) & (random.seed == seed), "ap"]
            ax.scatter(
                j + np.linspace(-0.12, 0.12, len(values)), values, color=COLORS[key], alpha=0.5
            )
            ax.plot(j, summary["random_cv"][str(seed)]["models"][key]["mean"], "D", color="#19252b")
        ax.set(title=label, xticks=range(len(seeds)), xticklabels=seeds, xlabel="CV seed")
        ax.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("Average precision")
    fig.suptitle("Five fold scores per assignment · diamonds show means")
    save(fig, "random_seed_ap.png")

    fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
    for key, label in LABELS.items():
        part = blocks.loc[blocks.model == key]
        ax.plot(part.fold_or_block, part.ap, marker="o", color=COLORS[key], label=label)
    ax.plot(
        part.fold_or_block, part.prevalence_baseline, "s--", color=".35", label="Block prevalence"
    )
    ax.set(
        xticks=part.fold_or_block,
        xlabel="Ordered evaluation block",
        ylabel="Average precision",
        ylim=(0, 1.02),
        title="Ranking performance across UDI-ordered blocks",
    )
    ax.legend(ncol=3)
    save(fig, "ordered_block_ap.png")

    fig, ax = plt.subplots(figsize=(8, 4.4), layout="constrained")
    for key, label in LABELS.items():
        values = [
            summary["comparisons"][key]["mean_across_seeds"],
            summary["udi_ordered"]["models"][key]["mean"],
        ]
        ax.plot([0, 1], values, "o-", color=COLORS[key], label=label)
    ax.set(
        xticks=[0, 1],
        xticklabels=["Random CV\n3 seed means", "UDI order\n3 block scores"],
        ylabel="Mean average precision",
        title="Sensitivity to evaluation strategy",
        xlim=(-0.25, 1.25),
    )
    ax.legend(ncol=2)
    save(fig, "strategy_ap.png")

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    labels = [f"Seed {s}" for s in seeds] + ["UDI order"]
    groups = [summary["random_cv"][str(seed)] for seed in seeds] + [summary["udi_ordered"]]
    for offset, family, color in ((-0.18, "HGB", COLORS["M1"]), (0.18, "RF", COLORS["M2"])):
        ax.bar(
            np.arange(len(groups)) + offset,
            [
                np.nan if g["a4_minus_a0"][family] is None else g["a4_minus_a0"][family]
                for g in groups
            ],
            width=0.34,
            color=color,
            label=family,
        )
    ax.axhline(0, color=".35", lw=1)
    ax.set(
        xticks=range(len(labels)),
        xticklabels=labels,
        ylabel="Mean paired AP difference (A4 − A0)",
        title="Change after adding temperature difference and mechanical power",
    )
    ax.legend()
    save(fig, "feature_set_differences.png")
