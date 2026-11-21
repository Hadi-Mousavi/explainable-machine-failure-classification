"""Figures for the paired training-fold comparison."""

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

LABELS = {
    "dummy_prior": "Prior dummy",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "hist_gradient_boosting": "Histogram boosting",
}
BLUE, ORANGE = "#175787", "#ac4a16"


def panel(size):
    fig = Figure(figsize=size, layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.grid(axis="x", color="#d9e1e7", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["right", "top"]].set_visible(False)
    return fig, ax


def save_figures(output, comparison, metrics, pairs):
    output.mkdir()
    fig, ax = panel((8.5, 4))
    for i, row in enumerate(comparison):
        subset = metrics[metrics.candidate_id == row["candidate_id"]]
        ax.scatter(
            subset.ap,
            [i] * len(subset),
            color="#a4bac8",
            s=30,
            label="Evaluation fold" if i == 0 else None,
        )
        ax.scatter(
            row["mean_ap"],
            i,
            color=BLUE,
            marker="D",
            s=45,
            label="Five-fold mean" if i == 0 else None,
        )
    ax.set_yticks(range(len(comparison)), [r["label"] for r in comparison])
    ax.set(
        xlim=(0, 1),
        xlabel="Average precision",
        title="Fixed baselines and selected tree configurations",
    )
    ax.legend(loc="lower right")
    fig.savefig(output / "model_family_ap.png", dpi=160)

    fig = Figure(figsize=(9, 4.5), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2, sharex=True)
    for ax, family in zip(axes, list(LABELS)[2:], strict=True):
        records = pairs[pairs.family == family].reset_index(drop=True)
        for i, r in records.iterrows():
            ax.plot([r.unweighted_ap, r.balanced_ap], [i, i], color="#a4bac8")
        ax.scatter(records.unweighted_ap, records.index, color=BLUE, label="Unweighted")
        ax.scatter(records.balanced_ap, records.index, color=ORANGE, marker="^", label="Balanced")
        ax.set_yticks(
            records.index,
            [
                a[-2:] + " / " + b[-2:]
                for a, b in zip(records.unweighted_id, records.balanced_id, strict=True)
            ],
        )
        ax.set(title=LABELS[family], xlabel="Mean fold AP", ylabel="Candidate IDs")
        ax.grid(axis="x", alpha=0.25)
    axes[0].legend()
    fig.suptitle("All tree candidates on the selection folds")
    fig.savefig(output / "candidate_search.png", dpi=160)

    fig, ax = panel((8.5, 5.5))
    labels = [
        LABELS[r.family] + "  " + r.unweighted_id[-2:] + " / " + r.balanced_id[-2:]
        for r in pairs.itertuples()
    ]
    ax.axvline(0, color="#616d73", lw=1)
    for i, r in pairs.reset_index(drop=True).iterrows():
        deltas = [r[f"delta_fold_{k}"] for k in range(1, 6)]
        ax.scatter(deltas, [i] * 5, color="#a4bac8", s=18, label="Fold" if i == 0 else None)
        ax.scatter(r.delta_ap, i, color=BLUE, marker="D", s=35, label="Mean" if i == 0 else None)
    ax.set_yticks(range(len(labels)), labels)
    ax.set(
        xlabel="AP difference (balanced minus unweighted)",
        title="Class weighting at matched settings on training folds",
    )
    ax.legend()
    fig.savefig(output / "weight_sensitivity.png", dpi=160)
