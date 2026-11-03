"""Deterministic statistical helpers for frozen label-risk analyses.

This module contains reusable mechanics only.  It does not inspect project
outputs, select data-driven thresholds, mutate labels, or make error claims.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


DEFAULT_SEED = 20_260_830
DEFAULT_BOOTSTRAP_REPLICATES = 2_000
DEFAULT_PERMUTATIONS = 2_000
REPORT_FOLD_SALT = "rsna-knee-label-risk-v2|report-group-fold"
TOP_TIE_SALT = "rsna-knee-label-risk-v2|top-quantile-tie"


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _finite_numeric(values: Any, *, name: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _binary(values: Any, *, name: str) -> np.ndarray:
    result = _finite_numeric(values, name=name)
    if not np.isin(result, (0.0, 1.0)).all():
        raise ValueError(f"{name} must contain only 0/1 values")
    return result.astype(np.int8)


def _nonempty_text(values: Any, *, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if pd.isna(raw).any():
        raise ValueError(f"{name} contains null values")
    result = np.asarray([str(value) for value in raw], dtype=object)
    if np.any(result == ""):
        raise ValueError(f"{name} contains empty values")
    return result


def _strict_index_alignment(
    reference: pd.Index,
    other: pd.Index,
    *,
    name: str,
) -> None:
    if not reference.equals(other):
        raise ValueError(f"{name} index does not exactly match the outcome index")


def _normal_probability_above(mean: float, sd: float, threshold: float) -> float:
    if sd < 0.0 or not all(np.isfinite([mean, sd, threshold])):
        raise ValueError("normal probability inputs must be finite with non-negative sd")
    if sd == 0.0:
        return float(mean > threshold)
    z = (threshold - mean) / sd
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


def empirical_bayes_normal_shrinkage(
    frame: pd.DataFrame,
    *,
    effect_column: str,
    standard_error_column: str,
    family_columns: Sequence[str],
    threshold: float = 0.0,
    minimum_family_size: int = 2,
) -> pd.DataFrame:
    """Apply zero-centred normal-normal empirical-Bayes shrinkage by family.

    For each declared family, the model is
    ``effect_hat | theta ~ N(theta, se**2)`` and
    ``theta ~ N(0, tau**2)``.  ``tau**2`` is the non-negative method-of-moments
    estimate ``mean(effect_hat**2 - se**2)``.  This fixed estimator keeps the
    helper deterministic and makes the shrinkage target explicit rather than
    learning a favourable non-zero family mean from the same effects.
    """

    families = tuple(family_columns)
    if not families:
        raise ValueError("family_columns must declare at least one family key")
    if len(families) != len(set(families)):
        raise ValueError("family_columns contains duplicates")
    if not isinstance(minimum_family_size, (int, np.integer)) or int(
        minimum_family_size
    ) < 2:
        raise ValueError("minimum_family_size must be an integer of at least two")
    if not np.isfinite(float(threshold)):
        raise ValueError("threshold must be finite")
    required = [effect_column, standard_error_column, *families]
    _require_columns(frame, required, name="empirical-Bayes input")
    if not frame.index.is_unique or frame.index.hasnans:
        raise ValueError("empirical-Bayes row index must be unique and non-null")
    if frame[list(families)].isna().any().any():
        raise ValueError("family keys contain null values")

    effects = _finite_numeric(frame[effect_column], name=effect_column)
    standard_errors = _finite_numeric(
        frame[standard_error_column], name=standard_error_column
    )
    if np.any(standard_errors <= 0.0):
        raise ValueError("standard errors must be strictly positive")

    result = frame.copy()
    result["eb_tau2"] = np.nan
    result["eb_posterior_mean"] = np.nan
    result["eb_posterior_se"] = np.nan
    result["eb_probability_positive"] = np.nan
    result["eb_probability_above_threshold"] = np.nan
    result["eb_threshold"] = float(threshold)
    result["eb_family_size"] = 0

    grouped_positions = result.groupby(
        list(families), sort=False, dropna=False
    ).indices
    for family_key, raw_positions in grouped_positions.items():
        positions = np.asarray(raw_positions, dtype=np.int64)
        if len(positions) < int(minimum_family_size):
            raise ValueError(
                f"empirical-Bayes family {family_key!r} has {len(positions)} rows; "
                f"minimum is {minimum_family_size}"
            )
        family_effects = effects[positions]
        family_se = standard_errors[positions]
        tau2 = float(max(0.0, np.mean(family_effects**2 - family_se**2)))
        if tau2 == 0.0:
            posterior_mean = np.zeros(len(positions), dtype=np.float64)
            posterior_se = np.zeros(len(positions), dtype=np.float64)
        else:
            observation_variance = family_se**2
            shrinkage = tau2 / (tau2 + observation_variance)
            posterior_mean = shrinkage * family_effects
            posterior_se = np.sqrt(
                tau2 * observation_variance / (tau2 + observation_variance)
            )
        probability_positive = np.asarray(
            [
                _normal_probability_above(float(mean), float(sd), 0.0)
                for mean, sd in zip(posterior_mean, posterior_se, strict=True)
            ],
            dtype=np.float64,
        )
        probability_threshold = np.asarray(
            [
                _normal_probability_above(float(mean), float(sd), float(threshold))
                for mean, sd in zip(posterior_mean, posterior_se, strict=True)
            ],
            dtype=np.float64,
        )
        index = result.index[positions]
        result.loc[index, "eb_tau2"] = tau2
        result.loc[index, "eb_posterior_mean"] = posterior_mean
        result.loc[index, "eb_posterior_se"] = posterior_se
        result.loc[index, "eb_probability_positive"] = probability_positive
        result.loc[index, "eb_probability_above_threshold"] = probability_threshold
        result.loc[index, "eb_family_size"] = int(len(positions))

    numeric_outputs = result[
        [
            "eb_tau2",
            "eb_posterior_mean",
            "eb_posterior_se",
            "eb_probability_positive",
            "eb_probability_above_threshold",
        ]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric_outputs).all():
        raise AssertionError("empirical-Bayes shrinkage produced non-finite output")
    return result


def deterministic_report_group_folds(
    study_keys: Sequence[Any] | np.ndarray | pd.Series,
    report_groups: Sequence[Any] | np.ndarray | pd.Series,
    *,
    n_splits: int,
    salt: str = REPORT_FOLD_SALT,
    require_all_folds: bool = True,
) -> np.ndarray:
    """Map whole report groups to cohort-independent SHA-256 K-folds."""

    if not isinstance(n_splits, (int, np.integer)) or int(n_splits) < 2:
        raise ValueError("n_splits must be an integer of at least two")
    if not isinstance(salt, str) or not salt:
        raise ValueError("salt must be a non-empty string")
    studies = _nonempty_text(study_keys, name="study_keys")
    groups = _nonempty_text(report_groups, name="report_groups")
    if len(studies) != len(groups):
        raise ValueError("study_keys and report_groups have different lengths")
    if len(studies) == 0:
        raise ValueError("fold input is empty")
    mapping = pd.DataFrame({"study": studies, "report_group": groups})
    per_study = mapping.groupby("study", sort=False)["report_group"].nunique()
    if per_study.gt(1).any():
        example = str(per_study.loc[per_study.gt(1)].index[0])
        raise ValueError(f"study key maps to multiple report groups: {example}")

    cache: dict[str, int] = {}
    folds = np.empty(len(groups), dtype=np.int16)
    for index, group in enumerate(groups):
        fold = cache.get(group)
        if fold is None:
            digest = hashlib.sha256(f"{salt}|{group}".encode("utf-8")).digest()
            fold = int(int.from_bytes(digest[:8], "big") % int(n_splits))
            cache[group] = fold
        folds[index] = fold
    if require_all_folds and set(folds.tolist()) != set(range(int(n_splits))):
        raise ValueError("SHA-256 report-group assignment produced an empty fold")
    if (
        pd.DataFrame({"report_group": groups, "fold": folds})
        .groupby("report_group", sort=False)["fold"]
        .nunique()
        .max()
        != 1
    ):
        raise AssertionError("a report group was split across folds")
    return folds


def _fixed_top_membership(
    frame: pd.DataFrame,
    *,
    risk_column: str,
    row_key_columns: Sequence[str],
    top_fraction: float,
) -> np.ndarray:
    if not 0.0 < float(top_fraction) < 1.0:
        raise ValueError("top_fraction must be strictly between 0 and 1")
    keys = tuple(row_key_columns)
    if not keys:
        raise ValueError("row_key_columns must not be empty")
    _require_columns(frame, keys, name="risk metric input")
    if frame[list(keys)].isna().any().any():
        raise ValueError("row keys contain null values")
    if any(frame[column].astype(str).eq("").any() for column in keys):
        raise ValueError("row keys contain empty values")
    if frame.duplicated(list(keys)).any():
        raise ValueError("row keys are not unique")
    risk = _finite_numeric(frame[risk_column], name=risk_column)
    tie_keys: list[str] = []
    for values in frame.loc[:, list(keys)].itertuples(index=False, name=None):
        payload = "|".join([TOP_TIE_SALT, *(str(value) for value in values)])
        tie_keys.append(hashlib.sha256(payload.encode("utf-8")).hexdigest())
    order = sorted(range(len(frame)), key=lambda idx: (-risk[idx], tie_keys[idx]))
    top_count = max(1, int(math.ceil(float(top_fraction) * len(frame))))
    membership = np.zeros(len(frame), dtype=bool)
    membership[np.asarray(order[:top_count], dtype=np.int64)] = True
    return membership


def _risk_metrics(
    outcome: np.ndarray,
    risk: np.ndarray,
    top: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    weights = (
        np.ones(len(outcome), dtype=np.float64)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64)
    )
    if weights.shape != outcome.shape or not np.isfinite(weights).all() or np.any(
        weights < 0.0
    ):
        raise ValueError("sample weights are invalid")
    positive_weight = float(weights[outcome == 1].sum())
    negative_weight = float(weights[outcome == 0].sum())
    total_weight = positive_weight + negative_weight
    if total_weight <= 0.0:
        raise ValueError("sample weights have zero total")
    auroc = (
        float(roc_auc_score(outcome, risk, sample_weight=weights))
        if positive_weight > 0.0 and negative_weight > 0.0
        else np.nan
    )
    average_precision = (
        float(average_precision_score(outcome, risk, sample_weight=weights))
        if positive_weight > 0.0
        else np.nan
    )
    brier = float(np.average((risk - outcome) ** 2, weights=weights))
    overall_rate = float(np.average(outcome, weights=weights))
    top_weight = float(weights[top].sum())
    top_rate = (
        float(np.average(outcome[top], weights=weights[top]))
        if top_weight > 0.0
        else np.nan
    )
    enrichment_ratio = (
        float(top_rate / overall_rate)
        if np.isfinite(top_rate) and overall_rate > 0.0
        else np.nan
    )
    enrichment_delta = (
        float(top_rate - overall_rate) if np.isfinite(top_rate) else np.nan
    )
    return {
        "auroc": auroc,
        "average_precision": average_precision,
        "brier": brier,
        "overall_error_rate": overall_rate,
        "top_error_rate": top_rate,
        "top_enrichment_ratio": enrichment_ratio,
        "top_enrichment_delta": enrichment_delta,
    }


def _ranking_metrics(
    outcome: np.ndarray,
    risk_rank: np.ndarray,
    top: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute discrimination and fixed-top ranking metrics without calibration."""

    weights = (
        np.ones(len(outcome), dtype=np.float64)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64)
    )
    if weights.shape != outcome.shape or not np.isfinite(weights).all() or np.any(
        weights < 0.0
    ):
        raise ValueError("sample weights are invalid")
    positive_weight = float(weights[outcome == 1].sum())
    negative_weight = float(weights[outcome == 0].sum())
    total_weight = positive_weight + negative_weight
    if total_weight <= 0.0:
        raise ValueError("sample weights have zero total")
    prevalence = positive_weight / total_weight
    auroc = (
        float(roc_auc_score(outcome, risk_rank, sample_weight=weights))
        if positive_weight > 0.0 and negative_weight > 0.0
        else np.nan
    )
    average_precision = (
        float(average_precision_score(outcome, risk_rank, sample_weight=weights))
        if positive_weight > 0.0
        else np.nan
    )
    ap_prevalence_lift = (
        float(average_precision / prevalence)
        if np.isfinite(average_precision) and prevalence > 0.0
        else np.nan
    )
    top_weight = float(weights[top].sum())
    top_positive_weight = float(weights[top & (outcome == 1)].sum())
    top_error_rate = top_positive_weight / top_weight if top_weight > 0.0 else np.nan
    top_enrichment = (
        float(top_error_rate / prevalence)
        if np.isfinite(top_error_rate) and prevalence > 0.0
        else np.nan
    )
    top_error_capture = (
        float(top_positive_weight / positive_weight)
        if positive_weight > 0.0
        else np.nan
    )
    return {
        "auroc": auroc,
        "average_precision": average_precision,
        "error_prevalence": float(prevalence),
        "average_precision_prevalence_lift": ap_prevalence_lift,
        "top_error_rate": float(top_error_rate),
        "top_enrichment_ratio": top_enrichment,
        "top_error_capture": top_error_capture,
    }


def clustered_bootstrap_binary_risk_metrics(
    frame: pd.DataFrame,
    *,
    outcome_column: str,
    risk_column: str,
    study_column: str,
    row_key_columns: Sequence[str],
    cluster_unit: Literal["study", "report_group"] = "study",
    report_group_column: str | None = None,
    top_fraction: float = 0.10,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Evaluate a frozen binary risk score with a cluster bootstrap.

    Top-quantile membership is fixed once on the original rows using exact
    top-k selection and SHA-256 tie-breaking, then retained in every bootstrap.
    Resampling is over studies or report groups, never over target cells.
    """

    if not isinstance(replicates, (int, np.integer)) or int(replicates) < 1:
        raise ValueError("replicates must be a positive integer")
    if not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    if cluster_unit not in {"study", "report_group"}:
        raise ValueError("cluster_unit must be study or report_group")
    required = [outcome_column, risk_column, study_column, *row_key_columns]
    if report_group_column is not None:
        required.append(report_group_column)
    _require_columns(frame, required, name="risk metric input")
    if len(frame) == 0:
        raise ValueError("risk metric input is empty")

    outcome = _binary(frame[outcome_column], name=outcome_column)
    risk = _finite_numeric(frame[risk_column], name=risk_column)
    if np.any((risk < 0.0) | (risk > 1.0)):
        raise ValueError("risk score must be within [0, 1] for Brier evaluation")
    studies = _nonempty_text(frame[study_column], name=study_column)
    if report_group_column is not None:
        report_groups = _nonempty_text(
            frame[report_group_column], name=report_group_column
        )
        mapping = pd.DataFrame({"study": studies, "report_group": report_groups})
        per_study = mapping.groupby("study", sort=False)["report_group"].nunique()
        if per_study.gt(1).any():
            example = str(per_study.loc[per_study.gt(1)].index[0])
            raise ValueError(f"study key maps to multiple report groups: {example}")
    else:
        report_groups = None
    if cluster_unit == "report_group" and report_groups is None:
        raise ValueError("report_group clustering requires report_group_column")

    top = _fixed_top_membership(
        frame,
        risk_column=risk_column,
        row_key_columns=row_key_columns,
        top_fraction=top_fraction,
    )
    clusters = studies if cluster_unit == "study" else report_groups
    assert clusters is not None
    cluster_codes, cluster_values = pd.factorize(clusters, sort=False)
    cluster_count = int(len(cluster_values))
    if cluster_count < 2:
        raise ValueError("at least two bootstrap clusters are required")

    point = _risk_metrics(outcome, risk, top)
    metric_names = tuple(point)
    distributions = {
        metric: np.full(int(replicates), np.nan, dtype=np.float64)
        for metric in metric_names
    }
    generator = np.random.default_rng(int(seed))
    for replicate in range(int(replicates)):
        sampled = generator.integers(0, cluster_count, size=cluster_count)
        cluster_weights = np.bincount(sampled, minlength=cluster_count).astype(
            np.float64
        )
        row_weights = cluster_weights[cluster_codes]
        metrics = _risk_metrics(outcome, risk, top, sample_weight=row_weights)
        for metric, value in metrics.items():
            distributions[metric][replicate] = value

    intervals: dict[str, list[float]] = {}
    valid_counts: dict[str, int] = {}
    for metric in metric_names:
        finite = distributions[metric][np.isfinite(distributions[metric])]
        valid_counts[metric] = int(len(finite))
        intervals[metric] = (
            [
                float(np.quantile(finite, float(alpha) / 2.0)),
                float(np.quantile(finite, 1.0 - float(alpha) / 2.0)),
            ]
            if len(finite)
            else [np.nan, np.nan]
        )
    undefined = sorted(metric for metric, value in point.items() if not np.isfinite(value))
    partial_bootstrap = any(count < int(replicates) for count in valid_counts.values())
    status = (
        "PASS_WITH_UNDEFINED_CLASS_METRICS"
        if undefined
        else "PASS_WITH_PARTIAL_BOOTSTRAP_METRICS"
        if partial_bootstrap
        else "PASS"
    )
    return {
        "status": status,
        "row_count": int(len(frame)),
        "study_count": int(pd.Series(studies).nunique()),
        "cluster_unit": cluster_unit,
        "cluster_count": cluster_count,
        "positive_count": int(outcome.sum()),
        "negative_count": int(len(outcome) - outcome.sum()),
        "top_fraction": float(top_fraction),
        "top_row_count": int(top.sum()),
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
        "bootstrap_alpha": float(alpha),
        "point": point,
        "bootstrap_interval": intervals,
        "valid_bootstrap_replicates": valid_counts,
        "undefined_point_metrics": undefined,
    }


def clustered_bootstrap_ranking_metrics(
    frame: pd.DataFrame,
    *,
    outcome_column: str,
    risk_rank_column: str,
    study_column: str,
    row_key_columns: Sequence[str],
    cluster_unit: Literal["study", "report_group"] = "study",
    report_group_column: str | None = None,
    top_fraction: float = 0.10,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Evaluate a frozen, potentially uncalibrated risk ranking by cluster.

    The risk rank may use any finite numeric scale.  Metrics are invariant to a
    strictly increasing transformation of that scale and intentionally exclude
    Brier score or any other probability-calibration metric.  Top membership is
    selected exactly once with fixed top-k and SHA-256 tie-breaking, then held
    fixed while studies or report groups are bootstrapped.
    """

    if not isinstance(replicates, (int, np.integer)) or int(replicates) < 1:
        raise ValueError("replicates must be a positive integer")
    if not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    if cluster_unit not in {"study", "report_group"}:
        raise ValueError("cluster_unit must be study or report_group")
    required = [outcome_column, risk_rank_column, study_column, *row_key_columns]
    if report_group_column is not None:
        required.append(report_group_column)
    _require_columns(frame, required, name="ranking metric input")
    if len(frame) == 0:
        raise ValueError("ranking metric input is empty")

    outcome = _binary(frame[outcome_column], name=outcome_column)
    risk_rank = _finite_numeric(frame[risk_rank_column], name=risk_rank_column)
    studies = _nonempty_text(frame[study_column], name=study_column)
    if report_group_column is not None:
        report_groups = _nonempty_text(
            frame[report_group_column], name=report_group_column
        )
        mapping = pd.DataFrame({"study": studies, "report_group": report_groups})
        per_study = mapping.groupby("study", sort=False)["report_group"].nunique()
        if per_study.gt(1).any():
            example = str(per_study.loc[per_study.gt(1)].index[0])
            raise ValueError(f"study key maps to multiple report groups: {example}")
    else:
        report_groups = None
    if cluster_unit == "report_group" and report_groups is None:
        raise ValueError("report_group clustering requires report_group_column")

    top = _fixed_top_membership(
        frame,
        risk_column=risk_rank_column,
        row_key_columns=row_key_columns,
        top_fraction=top_fraction,
    )
    clusters = studies if cluster_unit == "study" else report_groups
    assert clusters is not None
    cluster_codes, cluster_values = pd.factorize(clusters, sort=False)
    cluster_count = int(len(cluster_values))
    if cluster_count < 2:
        raise ValueError("at least two bootstrap clusters are required")

    point = _ranking_metrics(outcome, risk_rank, top)
    metric_names = tuple(point)
    distributions = {
        metric: np.full(int(replicates), np.nan, dtype=np.float64)
        for metric in metric_names
    }
    generator = np.random.default_rng(int(seed))
    for replicate in range(int(replicates)):
        sampled = generator.integers(0, cluster_count, size=cluster_count)
        cluster_weights = np.bincount(sampled, minlength=cluster_count).astype(
            np.float64
        )
        row_weights = cluster_weights[cluster_codes]
        metrics = _ranking_metrics(
            outcome, risk_rank, top, sample_weight=row_weights
        )
        for metric, value in metrics.items():
            distributions[metric][replicate] = value

    intervals: dict[str, list[float]] = {}
    valid_counts: dict[str, int] = {}
    for metric in metric_names:
        finite = distributions[metric][np.isfinite(distributions[metric])]
        valid_counts[metric] = int(len(finite))
        intervals[metric] = (
            [
                float(np.quantile(finite, float(alpha) / 2.0)),
                float(np.quantile(finite, 1.0 - float(alpha) / 2.0)),
            ]
            if len(finite)
            else [np.nan, np.nan]
        )
    undefined = sorted(metric for metric, value in point.items() if not np.isfinite(value))
    partial_bootstrap = any(count < int(replicates) for count in valid_counts.values())
    status = (
        "PASS_WITH_UNDEFINED_CLASS_METRICS"
        if undefined
        else "PASS_WITH_PARTIAL_BOOTSTRAP_METRICS"
        if partial_bootstrap
        else "PASS"
    )
    return {
        "status": status,
        "row_count": int(len(frame)),
        "study_count": int(pd.Series(studies).nunique()),
        "cluster_unit": cluster_unit,
        "cluster_count": cluster_count,
        "positive_count": int(outcome.sum()),
        "negative_count": int(len(outcome) - outcome.sum()),
        "top_fraction": float(top_fraction),
        "top_row_count": int(top.sum()),
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
        "bootstrap_alpha": float(alpha),
        "score_semantics": "uncalibrated_finite_risk_rank_higher_is_riskier",
        "point": point,
        "bootstrap_interval": intervals,
        "valid_bootstrap_replicates": valid_counts,
        "undefined_point_metrics": undefined,
    }


def _membership_matrix(frame: pd.DataFrame) -> np.ndarray:
    if frame.shape[1] == 0:
        raise ValueError("membership matrix has no hypotheses")
    if not frame.columns.is_unique:
        raise ValueError("membership matrix has duplicate hypothesis names")
    if frame.columns.hasnans or any(str(column) == "" for column in frame.columns):
        raise ValueError("membership matrix has null or empty hypothesis names")
    raw = frame.to_numpy()
    if pd.isna(raw).any():
        raise ValueError("membership matrix contains null values")
    try:
        numeric = raw.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("membership matrix must be boolean or 0/1") from error
    if not np.isin(numeric, (0.0, 1.0)).all():
        raise ValueError("membership matrix must be boolean or 0/1")
    return numeric.astype(bool)


def _effect_and_welch_t(
    outcome: np.ndarray,
    membership: np.ndarray,
    *,
    statistic: Literal["welch_t", "mean_difference"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_total, hypothesis_count = membership.shape
    n_segment = membership.sum(axis=0).astype(np.int64)
    n_complement = n_total - n_segment
    sum_total = float(outcome.sum())
    sum_segment = membership.T.astype(np.float64) @ outcome
    mean_segment = sum_segment / n_segment
    mean_complement = (sum_total - sum_segment) / n_complement
    effect = mean_segment - mean_complement
    if statistic == "mean_difference":
        standard_error = np.full(hypothesis_count, np.nan, dtype=np.float64)
        return effect, standard_error, effect.copy()

    squared = outcome**2
    squared_total = float(squared.sum())
    squared_segment = membership.T.astype(np.float64) @ squared
    variance_segment = (
        squared_segment - n_segment * mean_segment**2
    ) / (n_segment - 1)
    variance_complement = (
        (squared_total - squared_segment) - n_complement * mean_complement**2
    ) / (n_complement - 1)
    variance_segment = np.maximum(variance_segment, 0.0)
    variance_complement = np.maximum(variance_complement, 0.0)
    standard_error = np.sqrt(
        variance_segment / n_segment + variance_complement / n_complement
    )
    if np.any(standard_error <= 0.0) or not np.isfinite(standard_error).all():
        raise ValueError("Welch statistic has a non-positive or non-finite standard error")
    return effect, standard_error, effect / standard_error


def max_t_group_permutation(
    outcomes: pd.Series,
    membership: pd.DataFrame,
    strata: pd.Series | pd.DataFrame,
    *,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
    minimum_groups_per_arm: int = 5,
    statistic: Literal["welch_t", "mean_difference"] = "welch_t",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run a fixed-family group-level, within-stratum maxT permutation test.

    Each input row is one unique permutation group.  Outcomes are shuffled only
    among groups with the exact same supplied stratum tuple.  The boolean
    membership matrix is fixed throughout.  Raw and family-wise maxT p-values
    use the standard add-one empirical correction and two-sided absolute test
    statistics.
    """

    if not isinstance(outcomes, pd.Series):
        raise TypeError("outcomes must be a pandas Series indexed by group key")
    if not isinstance(membership, pd.DataFrame):
        raise TypeError("membership must be a pandas DataFrame indexed by group key")
    if not isinstance(strata, (pd.Series, pd.DataFrame)):
        raise TypeError("strata must be a pandas Series or DataFrame indexed by group key")
    if not outcomes.index.is_unique or outcomes.index.hasnans:
        raise ValueError("outcome group keys must be unique and non-null")
    _strict_index_alignment(outcomes.index, membership.index, name="membership")
    _strict_index_alignment(outcomes.index, strata.index, name="strata")
    if not isinstance(permutations, (int, np.integer)) or int(permutations) < 1:
        raise ValueError("permutations must be a positive integer")
    if not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(minimum_groups_per_arm, (int, np.integer)) or int(
        minimum_groups_per_arm
    ) < 2:
        raise ValueError("minimum_groups_per_arm must be at least two")
    if statistic not in {"welch_t", "mean_difference"}:
        raise ValueError("statistic must be welch_t or mean_difference")

    outcome = _finite_numeric(outcomes.to_numpy(), name="outcomes")
    matrix = _membership_matrix(membership)
    n_segment = matrix.sum(axis=0).astype(np.int64)
    n_complement = len(outcome) - n_segment
    if np.any(n_segment < int(minimum_groups_per_arm)) or np.any(
        n_complement < int(minimum_groups_per_arm)
    ):
        raise ValueError("a hypothesis fails the minimum group count in one arm")

    strata_frame = strata.to_frame("stratum") if isinstance(strata, pd.Series) else strata
    if strata_frame.shape[1] == 0 or not strata_frame.columns.is_unique:
        raise ValueError("strata must have at least one uniquely named column")
    if strata_frame.isna().any().any():
        raise ValueError("strata contains null values")
    stratum_rows = tuple(
        np.asarray(positions, dtype=np.int64)
        for positions in strata_frame.groupby(
            list(strata_frame.columns), sort=False, dropna=False
        ).indices.values()
    )
    if any(len(positions) < 2 for positions in stratum_rows):
        raise ValueError("every permutation stratum must contain at least two groups")

    effect, standard_error, observed_statistic = _effect_and_welch_t(
        outcome, matrix, statistic=statistic
    )
    absolute_observed = np.abs(observed_statistic)
    raw_exceedances = np.zeros(matrix.shape[1], dtype=np.int64)
    max_exceedances = np.zeros(matrix.shape[1], dtype=np.int64)
    generator = np.random.default_rng(int(seed))
    for _ in range(int(permutations)):
        permuted = outcome.copy()
        for positions in stratum_rows:
            permuted[positions] = outcome[positions][generator.permutation(len(positions))]
        _, _, permuted_statistic = _effect_and_welch_t(
            permuted, matrix, statistic=statistic
        )
        absolute_permuted = np.abs(permuted_statistic)
        raw_exceedances += absolute_permuted >= absolute_observed
        maximum = float(absolute_permuted.max())
        max_exceedances += maximum >= absolute_observed

    raw_p = (raw_exceedances + 1.0) / (int(permutations) + 1.0)
    max_t_p = (max_exceedances + 1.0) / (int(permutations) + 1.0)
    if np.any(max_t_p + 1e-15 < raw_p):
        raise AssertionError("maxT adjusted p-value is smaller than raw p-value")
    results = pd.DataFrame(
        {
            "hypothesis": membership.columns.to_list(),
            "segment_group_count": n_segment,
            "complement_group_count": n_complement,
            "effect": effect,
            "standard_error": standard_error,
            "test_statistic": observed_statistic,
            "raw_p_value": raw_p,
            "max_t_p_value": max_t_p,
        }
    )
    diagnostics = {
        "status": "PASS",
        "group_count": int(len(outcome)),
        "hypothesis_count": int(matrix.shape[1]),
        "stratum_count": int(len(stratum_rows)),
        "permutations": int(permutations),
        "seed": int(seed),
        "minimum_groups_per_arm": int(minimum_groups_per_arm),
        "statistic": statistic,
        "permutation_unit": "unique_outcome_index_group",
        "permutation_scope": "within_exact_supplied_stratum",
    }
    return results, diagnostics
