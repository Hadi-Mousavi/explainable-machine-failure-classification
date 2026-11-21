"""Phase 07A selection on one fixed score vector."""

from fractions import Fraction

import numpy as np
from sklearn.metrics import confusion_matrix

GRID = tuple(i / 20 for i in range(1, 20))


def f2_ratio(row):
    numerator = 5 * row["tp"]
    denominator = numerator + 4 * row["fn"] + row["fp"]
    return Fraction(numerator, denominator or 1)


def threshold_grid(labels, scores, cutoffs):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if tuple(cutoffs) != GRID:
        raise ValueError("Expected the complete approved grid in order")
    if labels.ndim != 1 or scores.shape != labels.shape or set(labels) != {0, 1}:
        raise ValueError("Expected aligned vectors and both binary labels")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Invalid class-1 scores")
    rows = []
    for cutoff in cutoffs:
        tn, fp, fn, tp = map(int, confusion_matrix(labels, scores >= cutoff, labels=[0, 1]).ravel())
        row = dict(threshold=cutoff, tn=tn, fp=fp, fn=fn, tp=tp)
        row.update(
            precision=tp / (tp + fp or 1),
            recall=tp / (tp + fn or 1),
            f1=2 * tp / (2 * tp + fp + fn or 1),
            f2=float(f2_ratio(row)),
        )
        rows.append(row)
    return rows


def decision_from_grid(rows):
    if tuple(r["threshold"] for r in rows) != GRID:
        raise ValueError("Incomplete or reordered threshold grid")
    ratios = [f2_ratio(row) for row in rows]
    best = max(ratios)
    maxima = [r for r, value in zip(rows, ratios, strict=True) if value == best]
    diagnostic = rows[GRID.index(0.5)]
    selected = maxima[-1] if best else diagnostic
    reason = "Unique maximum F2 on the approved grid"
    if not best:
        reason = "All F2 values were zero; validation did not establish a useful operating point"
    elif len(maxima) > 1:
        reason = "Highest cutoff among exact maximum F2 ties"
    support = selected["fn"] + selected["tp"]
    negatives = selected["tn"] + selected["fp"]
    return dict(
        selected=selected,
        diagnostic_050=diagnostic,
        reason=reason,
        selected_f2_exact=str(f2_ratio(selected)),
        fallback=not best,
        maximizing_thresholds=[r["threshold"] for r in maxima],
        positives=support,
        negatives=negatives,
        rows=support + negatives,
        prevalence=support / (support + negatives),
    )


def verify_decision(rows, decision):
    expected = decision_from_grid(rows)
    if any(decision.get(k) != v for k, v in expected.items()):
        raise ValueError("Decision disagrees with the preserved grid")
    for r in rows:
        tp, fp, tn, fn = (r[k] for k in ("tp", "fp", "tn", "fn"))
        if (
            min(tp, fp, tn, fn) < 0
            or tp + fn != expected["positives"]
            or tn + fp != expected["negatives"]
            or r["f2"] != float(f2_ratio(r))
            or r["recall"] != tp / (tp + fn or 1)
            or r["precision"] != tp / (tp + fp or 1)
            or r["f1"] != 2 * tp / (2 * tp + fp + fn or 1)
        ):
            raise ValueError("Grid metrics disagree with counts")
