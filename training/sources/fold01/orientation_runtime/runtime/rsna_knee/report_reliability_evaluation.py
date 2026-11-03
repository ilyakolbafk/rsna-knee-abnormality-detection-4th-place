"""Generic deterministic evaluation mechanics for report-reliability analyses.

This module contains formulas only.  It does not load project artifacts, choose
features or segments, derive permutation strata, inspect Gold labels, or assign
meaning to supplied events.  Callers must provide already-frozen predictions,
binary proxy events, report groups, and permutation strata.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Sequence

import numpy as np
import pandas as pd


DEFAULT_ALIGNMENT_PERMUTATIONS = 999
DEFAULT_ALIGNMENT_SEED = 20260831
DEFAULT_REVIEW_FRACTIONS: tuple[float, ...] = (0.05, 0.10, 0.20)
DEFAULT_REVIEW_TIE_SALT = "rsna-knee-report-reliability-v3|review-fraction-tie"


def _require_columns(
    frame: pd.DataFrame, columns: Sequence[str], *, name: str
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


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
    raw = np.asarray(values, dtype=object)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if pd.isna(raw).any():
        raise ValueError(f"{name} contains null values")
    result = np.asarray([str(value).strip() for value in raw], dtype=object)
    if np.any(result == ""):
        raise ValueError(f"{name} contains empty values")
    return result


def _stable_mean(values: pd.Series) -> float:
    """Float64 mean with an input-order-independent summation order."""

    numeric = np.asarray(values, dtype=np.float64)
    return float(np.mean(np.sort(numeric), dtype=np.float64))


def _report_group_losses(
    frame: pd.DataFrame,
    *,
    outcome_column: str,
    prediction_column: str,
    report_group_column: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    _require_columns(
        frame,
        (outcome_column, prediction_column, report_group_column),
        name="regression metric input",
    )
    if frame.empty:
        raise ValueError("regression metric input is empty")
    outcome = _finite_numeric(frame[outcome_column], name=outcome_column)
    prediction = _finite_numeric(frame[prediction_column], name=prediction_column)
    groups = _nonempty_text(frame[report_group_column], name=report_group_column)
    working = pd.DataFrame(
        {
            "_group": groups,
            "_squared_error": np.square(outcome - prediction),
            "_absolute_error": np.abs(outcome - prediction),
        }
    )
    per_group = working.groupby("_group", sort=True, observed=True)[
        ["_squared_error", "_absolute_error"]
    ].agg(_stable_mean)
    squared = per_group["_squared_error"].to_numpy(dtype=np.float64)
    absolute = per_group["_absolute_error"].to_numpy(dtype=np.float64)
    if not np.isfinite(squared).all() or not np.isfinite(absolute).all():
        raise ValueError("report-group loss finite-value gate failed")
    return squared, absolute, int(len(per_group))


def report_group_weighted_regression_metrics(
    frame: pd.DataFrame,
    *,
    outcome_column: str,
    prediction_column: str,
    report_group_column: str = "report_group",
) -> dict[str, int | float | str]:
    """Return equal-report-group MSE and MAE.

    Each report group contributes the mean loss of its rows, after which group
    losses are averaged with equal weight.  This is equivalent to assigning
    every row inverse eligible-row-count weight within its report group.
    """

    squared, absolute, group_count = _report_group_losses(
        frame,
        outcome_column=outcome_column,
        prediction_column=prediction_column,
        report_group_column=report_group_column,
    )
    result: dict[str, int | float | str] = {
        "status": "PASS",
        "row_count": int(len(frame)),
        "report_group_count": group_count,
        "report_group_weighted_mse": float(np.mean(squared, dtype=np.float64)),
        "report_group_weighted_mae": float(np.mean(absolute, dtype=np.float64)),
    }
    if not np.isfinite(
        [
            result["report_group_weighted_mse"],
            result["report_group_weighted_mae"],
        ]
    ).all():
        raise ValueError("report-group metric finite-value gate failed")
    return result


def relative_mse_improvement(context_mse: float, full_mse: float) -> float:
    """Return ``(context_mse - full_mse) / context_mse`` fail-closed."""

    values = _finite_numeric([context_mse, full_mse], name="MSE inputs")
    if np.any(values < 0.0):
        raise ValueError("MSE inputs must be non-negative")
    if values[0] == 0.0:
        raise ValueError("zero context MSE makes relative improvement undefined")
    return float((values[0] - values[1]) / values[0])


def compare_report_group_weighted_predictions(
    frame: pd.DataFrame,
    *,
    outcome_column: str,
    context_prediction_column: str,
    full_prediction_column: str,
    report_group_column: str = "report_group",
) -> dict[str, int | float | str]:
    """Evaluate frozen context and full predictions on identical rows/groups."""

    context = report_group_weighted_regression_metrics(
        frame,
        outcome_column=outcome_column,
        prediction_column=context_prediction_column,
        report_group_column=report_group_column,
    )
    full = report_group_weighted_regression_metrics(
        frame,
        outcome_column=outcome_column,
        prediction_column=full_prediction_column,
        report_group_column=report_group_column,
    )
    if context["report_group_count"] != full["report_group_count"]:
        raise AssertionError("context/full report-group coverage drift")
    context_mse = float(context["report_group_weighted_mse"])
    full_mse = float(full["report_group_weighted_mse"])
    return {
        "status": "PASS",
        "row_count": int(len(frame)),
        "report_group_count": int(context["report_group_count"]),
        "context_report_group_weighted_mse": context_mse,
        "full_report_group_weighted_mse": full_mse,
        "context_report_group_weighted_mae": float(
            context["report_group_weighted_mae"]
        ),
        "full_report_group_weighted_mae": float(
            full["report_group_weighted_mae"]
        ),
        "mse_increment": float(context_mse - full_mse),
        "relative_mse_improvement": relative_mse_improvement(
            context_mse, full_mse
        ),
    }


@dataclass(frozen=True)
class AlignmentPermutationResult:
    """Immutable result of one supplied-strata alignment permutation test."""

    status: str
    report_group_count: int
    stratum_count: int
    permutations: int
    seed: int
    context_mse: float
    observed_full_mse: float
    observed_statistic: float
    null_exceedance_count: int
    add_one_p_value: float
    null_statistics: tuple[float, ...]


def _aggregate_alignment_groups(
    frame: pd.DataFrame,
    *,
    endpoint_column: str,
    context_prediction_column: str,
    contribution_column: str,
    report_group_column: str,
    strata_columns: Sequence[str],
) -> pd.DataFrame:
    strata = tuple(strata_columns)
    if not strata:
        raise ValueError("strata_columns must not be empty")
    if len(set(strata)) != len(strata):
        raise ValueError("strata_columns contains duplicates")
    required = (
        endpoint_column,
        context_prediction_column,
        contribution_column,
        report_group_column,
        *strata,
    )
    _require_columns(frame, required, name="alignment permutation input")
    if frame.empty:
        raise ValueError("alignment permutation input is empty")

    working = pd.DataFrame(
        {
            "_report_group": _nonempty_text(
                frame[report_group_column], name=report_group_column
            ),
            "_endpoint": _finite_numeric(
                frame[endpoint_column], name=endpoint_column
            ),
            "_context": _finite_numeric(
                frame[context_prediction_column], name=context_prediction_column
            ),
            "_contribution": _finite_numeric(
                frame[contribution_column], name=contribution_column
            ),
        }
    )
    for index, column in enumerate(strata):
        working[f"_stratum_{index}"] = _nonempty_text(
            frame[column], name=column
        )

    stratum_internal = [f"_stratum_{index}" for index in range(len(strata))]
    drift = working.groupby("_report_group", sort=True, observed=True)[
        stratum_internal
    ].nunique(dropna=False)
    if (drift > 1).any().any():
        bad_groups = int((drift > 1).any(axis=1).sum())
        raise ValueError(
            f"precomputed strata drift within {bad_groups} report groups"
        )

    means = working.groupby("_report_group", sort=True, observed=True)[
        ["_endpoint", "_context", "_contribution"]
    ].agg(_stable_mean)
    strata_by_group = working.groupby(
        "_report_group", sort=True, observed=True
    )[stratum_internal].first()
    aggregated = means.join(strata_by_group, how="left", validate="one_to_one")
    aggregated["_stratum_key"] = [
        json.dumps(
            [str(value) for value in row],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for row in aggregated[stratum_internal].itertuples(index=False, name=None)
    ]
    return aggregated.reset_index(drop=False)


def alignment_permutation_test(
    frame: pd.DataFrame,
    *,
    endpoint_column: str,
    context_prediction_column: str,
    contribution_column: str,
    report_group_column: str = "report_group",
    strata_columns: Sequence[str],
    permutations: int = DEFAULT_ALIGNMENT_PERMUTATIONS,
    seed: int = DEFAULT_ALIGNMENT_SEED,
) -> AlignmentPermutationResult:
    """Permute whole report-group contributions within supplied strata.

    The input is aggregated to one row per report group using means for the
    endpoint, fixed context prediction, and morphology contribution.  Strata
    are never derived or altered here and must be homogeneous within each
    report group.  Call this once per target-direction permutation family.
    """

    if not isinstance(permutations, (int, np.integer)) or int(permutations) < 1:
        raise ValueError("permutations must be a positive integer")
    if not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    aggregated = _aggregate_alignment_groups(
        frame,
        endpoint_column=endpoint_column,
        context_prediction_column=context_prediction_column,
        contribution_column=contribution_column,
        report_group_column=report_group_column,
        strata_columns=strata_columns,
    )
    endpoint = aggregated["_endpoint"].to_numpy(dtype=np.float64)
    context = aggregated["_context"].to_numpy(dtype=np.float64)
    contribution = aggregated["_contribution"].to_numpy(dtype=np.float64)
    context_mse = float(np.mean(np.square(endpoint - context), dtype=np.float64))
    observed_full_mse = float(
        np.mean(np.square(endpoint - (context + contribution)), dtype=np.float64)
    )
    observed_statistic = float(context_mse - observed_full_mse)

    stratum_positions = [
        group.index.to_numpy(dtype=np.int64)
        for _key, group in aggregated.groupby(
            "_stratum_key", sort=True, observed=True
        )
    ]
    generator = np.random.default_rng(int(seed))
    null_statistics = np.empty(int(permutations), dtype=np.float64)
    for permutation_index in range(int(permutations)):
        permuted = contribution.copy()
        for positions in stratum_positions:
            order = generator.permutation(len(positions))
            permuted[positions] = contribution[positions][order]
        permuted_full_mse = float(
            np.mean(np.square(endpoint - (context + permuted)), dtype=np.float64)
        )
        null_statistics[permutation_index] = context_mse - permuted_full_mse

    if not np.isfinite(null_statistics).all() or not np.isfinite(
        [context_mse, observed_full_mse, observed_statistic]
    ).all():
        raise ValueError("alignment permutation finite-value gate failed")
    exceedance_count = int(np.count_nonzero(null_statistics >= observed_statistic))
    p_value = float((exceedance_count + 1) / (int(permutations) + 1))
    return AlignmentPermutationResult(
        status="PASS",
        report_group_count=int(len(aggregated)),
        stratum_count=int(aggregated["_stratum_key"].nunique()),
        permutations=int(permutations),
        seed=int(seed),
        context_mse=context_mse,
        observed_full_mse=observed_full_mse,
        observed_statistic=observed_statistic,
        null_exceedance_count=exceedance_count,
        add_one_p_value=p_value,
        null_statistics=tuple(float(value) for value in null_statistics),
    )


def benjamini_hochberg_24(
    frame: pd.DataFrame,
    *,
    p_value_column: str,
    target_column: str = "target",
    direction_column: str = "direction",
    alpha: float = 0.10,
    q_value_column: str = "bh_q_value",
    reject_column: str = "bh_reject",
) -> pd.DataFrame:
    """Apply Benjamini-Hochberg to exactly 24 unique target-direction rows."""

    _require_columns(
        frame,
        (p_value_column, target_column, direction_column),
        name="BH input",
    )
    if len(frame) != 24:
        raise ValueError(f"BH input must contain exactly 24 rows; got {len(frame)}")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    if q_value_column in frame or reject_column in frame:
        raise ValueError("BH output columns already exist")
    if frame[[target_column, direction_column]].isna().any().any():
        raise ValueError("target-direction keys contain null values")
    if frame.duplicated([target_column, direction_column]).any():
        raise ValueError("target-direction rows are not unique")
    p_values = _finite_numeric(frame[p_value_column], name=p_value_column)
    if np.any((p_values < 0.0) | (p_values > 1.0)):
        raise ValueError("p-values must be within [0, 1]")

    hypothesis_count = len(p_values)
    order = np.argsort(p_values, kind="mergesort")
    ordered_p = p_values[order]
    raw_adjusted = ordered_p * hypothesis_count / np.arange(
        1, hypothesis_count + 1, dtype=np.float64
    )
    ordered_adjusted = np.minimum.accumulate(raw_adjusted[::-1])[::-1]
    ordered_adjusted = np.clip(ordered_adjusted, 0.0, 1.0)
    adjusted = np.empty(hypothesis_count, dtype=np.float64)
    adjusted[order] = ordered_adjusted

    result = frame.copy()
    result[q_value_column] = adjusted
    result[reject_column] = adjusted <= float(alpha)
    return result


def _stable_row_key_hash(
    values: Sequence[object], *, tie_salt: str
) -> str:
    payload = json.dumps(
        [tie_salt, *[str(value) for value in values]],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def review_fraction_capture_enrichment(
    frame: pd.DataFrame,
    *,
    score_column: str,
    event_column: str,
    row_key_columns: Sequence[str],
    review_fractions: Sequence[float] = DEFAULT_REVIEW_FRACTIONS,
    tie_salt: str = DEFAULT_REVIEW_TIE_SALT,
) -> pd.DataFrame:
    """Evaluate top-review capture/enrichment for supplied binary events.

    Exact top-k membership uses ``ceil(fraction * row_count)`` and SHA-256
    row-key tie-breaking.  Scores and events are accepted as supplied; this
    helper neither derives events nor chooses a review fraction.
    """

    keys = tuple(row_key_columns)
    if not keys:
        raise ValueError("row_key_columns must not be empty")
    if len(set(keys)) != len(keys):
        raise ValueError("row_key_columns contains duplicates")
    _require_columns(
        frame,
        (score_column, event_column, *keys),
        name="review metric input",
    )
    if frame.empty:
        raise ValueError("review metric input is empty")
    if not isinstance(tie_salt, str) or not tie_salt:
        raise ValueError("tie_salt must be a non-empty string")
    if frame[list(keys)].isna().any().any():
        raise ValueError("row keys contain null values")
    if any(frame[column].astype(str).str.strip().eq("").any() for column in keys):
        raise ValueError("row keys contain empty values")
    if frame.duplicated(list(keys)).any():
        raise ValueError("row keys are not unique")

    scores = _finite_numeric(frame[score_column], name=score_column)
    events = _binary(frame[event_column], name=event_column)
    fractions = [float(value) for value in review_fractions]
    if not fractions:
        raise ValueError("review_fractions must not be empty")
    if not np.isfinite(fractions).all() or any(
        value <= 0.0 or value > 1.0 for value in fractions
    ):
        raise ValueError("review fractions must be finite within (0, 1]")
    if len(set(fractions)) != len(fractions):
        raise ValueError("review_fractions contains duplicates")

    tie_hashes = [
        _stable_row_key_hash(values, tie_salt=tie_salt)
        for values in frame[list(keys)].itertuples(index=False, name=None)
    ]
    order = sorted(
        range(len(frame)), key=lambda index: (-scores[index], tie_hashes[index])
    )
    population_count = int(len(frame))
    event_count = int(events.sum())
    prevalence = float(event_count / population_count)
    rows: list[dict[str, int | float | str]] = []
    for fraction in sorted(fractions):
        review_count = min(
            population_count,
            max(1, int(math.ceil(fraction * population_count))),
        )
        selected = np.asarray(order[:review_count], dtype=np.int64)
        reviewed_event_count = int(events[selected].sum())
        reviewed_event_rate = float(reviewed_event_count / review_count)
        capture = (
            float(reviewed_event_count / event_count) if event_count else np.nan
        )
        enrichment = (
            float(reviewed_event_rate / prevalence) if event_count else np.nan
        )
        rows.append(
            {
                "status": "PASS" if event_count else "PASS_WITH_NO_EVENTS",
                "review_fraction": fraction,
                "population_count": population_count,
                "review_count": review_count,
                "event_count": event_count,
                "reviewed_event_count": reviewed_event_count,
                "event_prevalence": prevalence,
                "reviewed_event_rate": reviewed_event_rate,
                "event_capture": capture,
                "top_risk_enrichment": enrichment,
                "selection_sha256": hashlib.sha256(
                    "\n".join(sorted(tie_hashes[index] for index in selected)).encode(
                        "utf-8"
                    )
                ).hexdigest(),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "AlignmentPermutationResult",
    "DEFAULT_ALIGNMENT_PERMUTATIONS",
    "DEFAULT_ALIGNMENT_SEED",
    "DEFAULT_REVIEW_FRACTIONS",
    "DEFAULT_REVIEW_TIE_SALT",
    "alignment_permutation_test",
    "benjamini_hochberg_24",
    "compare_report_group_weighted_predictions",
    "relative_mse_improvement",
    "report_group_weighted_regression_metrics",
    "review_fraction_capture_enrichment",
]
