"""Prespecified Phase 02 pipelines. Both models receive the same six predictors."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ai4i_pm.schema import NUMERIC_FEATURES, PREDICTORS


def check_model_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """Reject extra, missing, duplicated or reordered columns at fit and score time."""
    if tuple(frame.columns) != PREDICTORS:
        raise ValueError(f"Expected exactly the Phase 02 predictors in order: {PREDICTORS}")
    return frame


def baseline_pipelines(models: dict) -> dict[str, Pipeline]:
    logistic = models["logistic_regression"].copy()
    if logistic.pop("class_weight") != "none":
        raise ValueError("Phase 02 requires unweighted logistic regression")
    estimators = {
        "dummy_prior": DummyClassifier(**models["dummy_prior"]),
        "logistic_regression": LogisticRegression(class_weight=None, **logistic),
    }
    return {
        name: Pipeline(
            [
                ("allowlist", FunctionTransformer(check_model_inputs)),
                (
                    "features",
                    ColumnTransformer(
                        [
                            ("type", OneHotEncoder(handle_unknown="ignore"), ["Type"]),
                            ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
                        ],
                        remainder="drop",
                    ),
                ),
                ("classifier", classifier),
            ]
        )
        for name, classifier in estimators.items()
    }


def positive_scores(model, features: pd.DataFrame, positive_class: int = 1) -> np.ndarray:
    """Look up the failure column using the fitted label order."""
    columns = np.flatnonzero(model.classes_ == positive_class)
    if len(columns) != 1:
        raise ValueError("Expected one probability column for the positive class")
    scores = model.predict_proba(features)[:, columns.item()]
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Invalid positive-class probabilities")
    return scores
