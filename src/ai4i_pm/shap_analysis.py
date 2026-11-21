"""Fold-local Tree SHAP on the fitted estimator's transformed coordinates."""

from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline

from ai4i_pm.models import positive_scores


def sample_ids(index: pd.Index, size: int, seed: int) -> np.ndarray:
    ordered = np.sort(index.to_numpy())
    return np.sort(
        np.random.default_rng(seed).choice(ordered, min(size, len(ordered)), replace=False)
    )


def diagnostic_cases(actual: pd.Series, scores: np.ndarray, cutoff: float, limit: int) -> tuple:
    predicted = scores >= cutoff
    selected, counts = {}, {}
    for category, label, decision in (
        ("TP", 1, True),
        ("FP", 0, True),
        ("FN", 1, False),
        ("TN", 0, False),
    ):
        eligible = actual.index[(actual == label) & (predicted == decision)].sort_values()
        counts[category] = len(eligible)
        selected[category] = eligible[:limit].tolist()
    return selected, counts


@dataclass
class Attribution:
    names: list[str]
    matrix: np.ndarray
    values: np.ndarray
    base: np.ndarray
    output: np.ndarray
    scale: str
    shape: tuple

    @property
    def residual(self) -> np.ndarray:
        return self.base + self.values.sum(axis=1) - self.output


def checked_attribution(
    model: Pipeline, explainer: Any, rows: pd.DataFrame, atol: float
) -> Attribution:
    """Validate feature coordinates, class axis and additive reconstruction on each row."""
    names = model["features"].get_feature_names_out().tolist()
    matrix = model[:-1].transform(rows)
    estimator = model["classifier"]
    if matrix.shape != (len(rows), len(names)) or len(set(names)) != len(names):
        raise ValueError("Transformed feature names and matrix do not align")
    explanation = explainer(matrix)
    if list(explanation.feature_names) != names or not np.array_equal(explanation.data, matrix):
        raise ValueError("Explanation does not identify the fitted transformed coordinates")
    values, base = np.asarray(explanation.values), np.asarray(explanation.base_values)
    shape = values.shape
    classes = estimator.classes_.tolist()
    if sorted(classes) != [0, 1]:
        raise ValueError(f"Unexpected classifier labels {classes}")
    positive = classes.index(1)
    probability = positive_scores(model, rows)
    if isinstance(estimator, RandomForestClassifier):
        if shape != (*matrix.shape, 2) or base.shape != (len(rows), 2):
            raise ValueError(f"Unexpected forest SHAP dimensions {shape}, {base.shape}")
        values, base = values[:, :, positive], base[:, positive]
        output, scale = probability, "class-1 probability"
    elif isinstance(estimator, HistGradientBoostingClassifier):
        if positive != 1 or shape != matrix.shape or base.size != len(rows):
            raise ValueError(
                f"Unexpected binary margin dimensions or classes: {shape}, {base.shape}, {classes}"
            )
        base = base.reshape(-1)
        output, scale = estimator.decision_function(matrix), "class-1 log-odds"
        if not np.allclose(expit(output), probability, atol=1e-12, rtol=0):
            raise ValueError("Sigmoid of HGB decision margin differs from class-1 probability")
    else:
        raise ValueError(f"Unsupported classifier {type(estimator).__name__}")
    batch = Attribution(names, matrix, values, base, output, scale, shape)
    if not np.isfinite(batch.residual).all() or np.abs(batch.residual).max() > atol:
        raise ValueError(
            f"SHAP additivity exceeds {atol:g}: max error {np.abs(batch.residual).max()}"
        )
    return batch


def preflight(model: Pipeline, held: pd.DataFrame, settings: dict) -> tuple:
    import shap

    if version("shap") != settings["version"]:
        raise ValueError(f"Expected SHAP {settings['version']}, installed {version('shap')}")
    names = model["features"].get_feature_names_out().tolist()
    explainer = shap.TreeExplainer(
        model["classifier"],
        feature_names=names,
        feature_perturbation=settings["feature_perturbation"],
        model_output=settings["model_output"],
    )
    rows = held.iloc[: settings["preflight_size"]]
    batch = checked_attribution(model, explainer, rows, settings["atol"])
    return explainer, {
        "preflight_udis": rows.index.tolist(),
        "feature_names": names,
        "classes": model.classes_.tolist(),
        "positive_index": model.classes_.tolist().index(1),
        "native_shape": batch.shape,
        "scale": batch.scale,
        "preflight_max_absolute_error": float(abs(batch.residual).max()),
    }


def explain_sample(
    key: str,
    model: Pipeline,
    explainer: Any,
    held: pd.DataFrame,
    actual: pd.Series,
    ids: np.ndarray,
    cfg: dict,
    spec: list[str],
) -> tuple:
    scores = pd.Series(positive_scores(model, held), index=held.index)
    selected, counts = diagnostic_cases(
        actual, scores.to_numpy(), cfg["cutoff"], cfg["cases_per_category"]
    )
    cases = {udi: category for category, members in selected.items() for udi in members}
    indices = np.union1d(ids, list(cases)).astype(int)
    rows = held.loc[indices]
    batch = checked_attribution(model, explainer, rows, cfg["atol"])
    global_mask = np.isin(indices, ids)
    magnitude = np.abs(batch.values[global_mask]).mean(axis=0)
    ranking = pd.DataFrame(
        {"model": key, "feature": batch.names, "mean_abs_shap": magnitude, "scale": batch.scale}
    ).sort_values("mean_abs_shap", ascending=False, kind="stable")
    arrays = {
        "udi": indices,
        "global_mask": global_mask,
        "features": np.array(batch.names),
        "transformed": batch.matrix,
        "values": batch.values,
        "base": batch.base,
        "output": batch.output,
        "score": scores.loc[indices].to_numpy(),
        "actual": actual.loc[indices].to_numpy(),
    }
    measurements = model["allowlist"].transform(rows)
    local = []
    for udi, category in cases.items():
        i = rows.index.get_loc(udi)
        for j, name in enumerate(batch.names):
            local.append(
                {
                    "model": key,
                    "candidate": spec[0],
                    "condition": spec[1],
                    "fold": cfg["fold"],
                    "UDI": udi,
                    "actual": int(actual.at[udi]),
                    "score": scores.at[udi],
                    "prediction": int(scores.at[udi] >= cfg["cutoff"]),
                    "category": category,
                    **measurements.loc[udi].to_dict(),
                    "base": batch.base[i],
                    "output": batch.output[i],
                    "scale": batch.scale,
                    "additivity_residual": batch.residual[i],
                    "feature": name,
                    "transformed_value": batch.matrix[i, j],
                    "signed_shap": batch.values[i, j],
                }
            )
    metadata = {
        "global_positives": int(actual.loc[ids].sum()),
        "selected_cases": selected,
        "confusion_counts": counts,
        "explained_union_rows": len(indices),
        "max_absolute_error": float(abs(batch.residual).max()),
    }
    return metadata, arrays, ranking, pd.DataFrame(local)
