"""UDI-indexed stratified folds and paired out-of-fold evaluation."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from ai4i_pm.evaluation import fold_metrics
from ai4i_pm.models import positive_scores
from ai4i_pm.schema import IDENTIFIER, TARGET, predictor_matrix


def assign_folds(target: pd.Series, cv: dict) -> pd.Series:
    target = target.sort_index()
    if not target.index.is_unique:
        raise ValueError("Training IDs must be unique")
    folds = pd.Series(0, index=target.index, name="fold")
    for fold, (fit, evaluate) in enumerate(StratifiedKFold(**cv).split(target, target), start=1):
        if np.intersect1d(fit, evaluate).size or (folds.iloc[evaluate] != 0).any():
            raise ValueError("Overlapping cross-validation memberships")
        if any(set(target.iloc[subset]) != {0, 1} for subset in (fit, evaluate)):
            raise ValueError("Both classes are required in each fold partition")
        folds.iloc[evaluate] = fold
    if set(folds) != set(range(1, cv["n_splits"] + 1)):
        raise ValueError("Incomplete cross-validation memberships")
    return folds


def cross_validate(training, pipelines, cv, metrics):
    ordered = training.set_index(IDENTIFIER, verify_integrity=True).sort_index()
    target = ordered[TARGET]
    features = predictor_matrix(ordered)
    folds = assign_folds(target, cv)
    measurements, predictions = [], []
    for fold in sorted(folds.unique()):
        held = folds == fold
        fit_ids, eval_ids = folds.index[~held], folds.index[held]
        actual = target.loc[eval_ids]
        fit_positives = int(target.loc[fit_ids].sum())
        counts = {
            "fit_positives": fit_positives,
            "fit_negatives": len(fit_ids) - fit_positives,
            "eval_positives": int(actual.sum()),
            "eval_negatives": len(actual) - int(actual.sum()),
            "eval_prevalence": float(actual.mean()),
        }
        for name, template in pipelines.items():
            model = clone(template).fit(features.loc[fit_ids], target.loc[fit_ids])
            scores = positive_scores(model, features.loc[eval_ids], metrics["positive_class"])
            measurements.append(
                {"model": name, "fold": fold, **counts, **fold_metrics(actual, scores, metrics)}
            )
            predictions.append(
                pd.DataFrame(
                    {
                        IDENTIFIER: eval_ids,
                        "fold": fold,
                        "model": name,
                        TARGET: actual.to_numpy(),
                        "positive_score": scores,
                    }
                )
            )
    oof = pd.concat(predictions, ignore_index=True).sort_values(["model", IDENTIFIER])
    for _, group in oof.groupby("model"):
        observed = group.set_index(IDENTIFIER)
        if not observed["fold"].equals(folds) or not observed[TARGET].equals(target):
            raise ValueError("Out-of-fold assignments or labels differ from the training manifest")
    return folds, pd.DataFrame(measurements), oof
