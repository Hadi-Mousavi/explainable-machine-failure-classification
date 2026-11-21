"""Paired, bounded searches using the published UDI fold memberships."""

import json
import warnings
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from ai4i_pm.evaluation import fold_average_precision
from ai4i_pm.models import baseline_pipelines, check_model_inputs, positive_scores
from ai4i_pm.schema import NUMERIC_FEATURES
from ai4i_pm.utils import json_bytes, sha256_bytes

FAMILIES = ("dummy_prior", "logistic_regression", "random_forest", "hist_gradient_boosting")


def frozen_splits(target, manifest):
    """Translate manifest UDIs into positional indices for this exact dataframe."""
    memberships = manifest["evaluation_udis"]
    if (
        manifest["primary_partition"] != "train"
        or set(memberships) != {str(f) for f in range(1, 6)}
        or not target.index.is_unique
    ):
        raise ValueError("Expected five training folds with unique training IDs")
    ids = [udi for members in memberships.values() for udi in members]
    if len(ids) != len(target) or set(ids) != set(target.index):
        raise ValueError("CV memberships must cover the training IDs exactly once")
    splits, records = [], []
    for fold in range(1, 6):
        members = memberships[str(fold)]
        evaluate = target.index.get_indexer(members)
        fit = np.flatnonzero(~target.index.isin(members))
        if any(set(target.iloc[subset]) != {0, 1} for subset in (fit, evaluate)):
            raise ValueError("Each fold partition requires both classes")
        splits.append((fit, evaluate))
        records.append(
            {
                "fold": fold,
                "eval_udis_sha256": sha256_bytes(json_bytes(target.index[evaluate].tolist())),
                "fit_rows": len(fit),
                "fit_positives": int(target.iloc[fit].sum()),
                "eval_rows": len(evaluate),
                "eval_positives": int(target.iloc[evaluate].sum()),
            }
        )
    return splits, records


def pipelines(settings, seed):
    models = baseline_pipelines(settings)
    for family, estimator in (
        ("random_forest", RandomForestClassifier(random_state=seed, n_jobs=1)),
        (
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(random_state=seed, early_stopping=False),
        ),
    ):
        models[family] = Pipeline(
            [
                ("allowlist", FunctionTransformer(check_model_inputs)),
                (
                    "features",
                    ColumnTransformer(
                        [
                            (
                                "type",
                                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                                ["Type"],
                            ),
                            ("numeric", "passthrough", list(NUMERIC_FEATURES)),
                        ]
                    ),
                ),
                ("classifier", estimator),
            ]
        )
    return models


def candidate_plan(config):
    weights = [None if w == "none" else w for w in config["class_weights"]]
    if weights != [None, "balanced"]:
        raise ValueError("Paired analysis requires unweighted and balanced configurations")
    plan = {"dummy_prior": [{}], "logistic_regression": [{"class_weight": w} for w in weights]}
    for name in FAMILIES[2:]:
        space = {k: [None if x == "none" else x for x in v] for k, v in config[name].items()}
        draws = ParameterSampler(
            space, n_iter=config["structures_per_family"], random_state=config["seed"]
        )
        plan[name] = [{**draw, "class_weight": w} for draw in draws for w in weights]
    return plan


def candidate_id(family, ordinal):
    return f"{family}-{ordinal + 1:02d}"


def score_ap(model, features, target):
    return fold_average_precision(target, positive_scores(model, features))


def diagnose_fold(template, params, features, target, split):
    """A failed fold is ineligible even when a diagnostic retry succeeds."""
    fit, evaluate = split
    try:
        model = clone(template).set_params(**params).fit(features.iloc[fit], target.iloc[fit])
        score_ap(model, features.iloc[evaluate], target.iloc[evaluate])
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return "Failure did not recur on diagnostic retry; see original search messages"


def evaluate_family(family, template, candidates, features, target, splits, fold_records, seed):
    grids = [{"classifier__" + key: [v] for key, v in p.items()} for p in candidates]
    # Record the UDIs actually passed to the scorer, alongside the planned split.
    keys = tuple(candidates[0])
    observed = {}

    def audited_ap(model, evaluation, labels):
        ap = score_ap(model, evaluation, labels)
        parameters = model["classifier"].get_params(deep=False)
        signature = tuple(parameters[k] for k in keys)
        digest = sha256_bytes(json_bytes(evaluation.index.tolist()))
        observed.setdefault(signature, []).append(digest)
        return ap

    search = RandomizedSearchCV(
        template,
        grids,
        n_iter=len(grids),
        random_state=seed,
        scoring=audited_ap,
        cv=splits,
        n_jobs=1,
        refit=False,
        error_score=np.nan,
    )
    start = perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            search.fit(features, target)
            results = search.cv_results_
            messages = []
        except ValueError as exc:
            # sklearn raises instead of returning cv_results_ when every fit fails.
            if "fits failed" not in str(exc) or "All the" not in str(exc):
                raise
            results = None
            messages = [str(exc)]
    elapsed = perf_counter() - start
    messages.extend(f"{w.category.__name__}: {w.message}" for w in captured)
    rows, measurements = [], []
    positions = (
        {}
        if results is None
        else {json.dumps(p, sort_keys=True): i for i, p in enumerate(results["params"])}
    )
    for i, spec in enumerate(candidates):
        ident = candidate_id(family, i)
        params = {"classifier__" + k: v for k, v in spec.items()}
        index = positions[json.dumps(params, sort_keys=True)] if results is not None else None
        scores, errors = [], []
        for j, record in enumerate(fold_records):
            ap = results[f"split{j}_test_score"][index] if results is not None else np.nan
            error = ""
            if not np.isfinite(ap):
                error = diagnose_fold(template, params, features, target, splits[j])
                errors.append(f"fold {j + 1}: {error}")
            scores.append(ap)
            measurements.append({"candidate_id": ident, **record, "ap": ap, "error": error})
        expected_digests = [
            record["eval_udis_sha256"]
            for record, ap in zip(fold_records, scores, strict=True)
            if np.isfinite(ap)
        ]
        if observed.get(tuple(spec[k] for k in keys), []) != expected_digests:
            raise ValueError(f"Observed scoring UDIs differ from the saved folds for {ident}")
        full_params = clone(template).set_params(**params)["classifier"].get_params(deep=False)
        rows.append(
            {
                "candidate_id": ident,
                "family": family,
                "evaluation": "selection CV" if family in FAMILIES[2:] else "fixed CV",
                "settings": json.dumps(spec, sort_keys=True),
                "classifier_parameters": json.dumps(full_params, sort_keys=True),
                "mean_ap": float(np.mean(scores)),
                "sample_sd": float(np.std(scores, ddof=1)),
                "training_prevalence": float(target.mean()),
                "fit_score_seconds": (
                    len(splits)
                    * (results["mean_fit_time"][index] + results["mean_score_time"][index])
                    if results is not None
                    else None
                ),
                "failed_folds": len(errors),
                "error": " | ".join(errors),
            }
        )
    return rows, measurements, {"search_seconds": elapsed, "messages": messages}


def ranking_key(row):
    family = row["family"]
    p = json.loads(row["classifier_parameters"])
    structural = ()
    if family == "random_forest":
        structural = (p["max_depth"] or np.inf, -p["min_samples_leaf"], p["n_estimators"])
    elif family == "hist_gradient_boosting":
        structural = (
            p["max_leaf_nodes"],
            p["max_iter"],
            -p["l2_regularization"],
            p["learning_rate"],
        )
    return (
        -row["mean_ap"],
        FAMILIES.index(family),
        structural,
        p.get("class_weight") is not None,
        row["candidate_id"],
    )


def ranked_candidates(rows):
    return sorted(
        (r for r in rows if not r["failed_folds"] and np.isfinite(r["mean_ap"])), key=ranking_key
    )


def weight_pairs(rows, fold_metrics):
    fold_table = pd.DataFrame(fold_metrics).pivot(index="candidate_id", columns="fold", values="ap")
    pairs = {}
    for row in rows:
        if row["family"] == "dummy_prior":
            continue
        settings = json.loads(row["settings"])
        weight = settings.pop("class_weight")
        key = (row["family"], json.dumps(settings, sort_keys=True))
        pairs.setdefault(key, {})[weight] = row
    output = []
    for (family, settings), variants in pairs.items():
        equal, balanced = variants[None], variants["balanced"]
        delta = fold_table.loc[balanced["candidate_id"]] - fold_table.loc[equal["candidate_id"]]
        output.append(
            {
                "family": family,
                "settings": settings,
                "unweighted_id": equal["candidate_id"],
                "balanced_id": balanced["candidate_id"],
                "unweighted_ap": equal["mean_ap"],
                "balanced_ap": balanced["mean_ap"],
                "delta_ap": balanced["mean_ap"] - equal["mean_ap"],
                **{f"delta_fold_{k}": v for k, v in delta.items()},
            }
        )
    return pd.DataFrame(output)
