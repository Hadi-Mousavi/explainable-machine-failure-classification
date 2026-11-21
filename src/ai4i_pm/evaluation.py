"""Ranking and fixed operating-point diagnostics on cross-fitted scores."""

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)


def fold_metrics(actual, scores, metrics: dict) -> dict:
    positive = metrics["positive_class"]
    labels = (scores >= metrics["diagnostic_threshold"]).astype(int)
    options = {"pos_label": positive, "zero_division": metrics["zero_division"]}
    tn, fp, fn, tp = confusion_matrix(actual, labels, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(actual, scores, pos_label=positive)),
        "diagnostic_threshold": metrics["diagnostic_threshold"],
        "precision": float(precision_score(actual, labels, **options)),
        "recall": float(recall_score(actual, labels, **options)),
        "f1": float(f1_score(actual, labels, **options)),
        "f2": float(fbeta_score(actual, labels, beta=2, **options)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
