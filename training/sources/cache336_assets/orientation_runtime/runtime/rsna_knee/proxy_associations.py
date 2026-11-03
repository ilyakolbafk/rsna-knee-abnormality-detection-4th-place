"""Gold-blind grouped-OOF descriptive associations for report/source proxies.

The estimands in this module are incremental out-of-fold predictability and
concentration of base-model residuals.  They are not causal effects and do not identify
institutions, regions, or individual readers.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder

from .constants import ID_COLUMN


SEED = 20260829
N_SPLITS = 5
RIDGE_ALPHA = 10.0
LOGISTIC_C = 1.0
MINIMUM_STUDIES = 80
MISSING_CATEGORY = "__missing__"

TARGET_COLUMN = "target"
GROUP_COLUMN = "report_group"
STYLE_NESTED_COLUMN = "style_cluster_nested_within_language"

BASE_WITHOUT_LANGUAGE = (TARGET_COLUMN, "text_state", "criteria_state")
BASE_WITH_LANGUAGE = (*BASE_WITHOUT_LANGUAGE, "language_proxy")

CONTINUOUS_OUTCOMES = (
    "llm_abs_disagreement",
    "adjusted_gap_residual",
    "text_image_rank_gap_abs",
)
BINARY_OUTCOMES = (
    "flag__nlp_conflict",
    "flag__rubric_mismatch",
    "flag__borderline_or_unclear",
)
OUTCOME_TYPES = {
    **{name: "continuous" for name in CONTINUOUS_OUTCOMES},
    **{name: "binary" for name in BINARY_OUTCOMES},
}

RULE_DERIVED_OUTCOME_REASONS = {
    "flag__nlp_conflict": (
        "The flag includes contradictory rule evidence and rule-versus-consensus logic."
    ),
    "flag__rubric_mismatch": (
        "The flag is directly constructed from rule-derived text_state and criteria_state."
    ),
    "flag__borderline_or_unclear": (
        "The flag includes rule-derived uncertainty and criteria_state logic."
    ),
}


@dataclass(frozen=True)
class ProxySpecification:
    """One fixed proxy addition and its adjustment set."""

    name: str
    source_column: str
    model_column: str
    base_columns: tuple[str, ...]


PROXY_SPECIFICATIONS = (
    ProxySpecification(
        name="language",
        source_column="language_proxy",
        model_column="language_proxy",
        base_columns=BASE_WITHOUT_LANGUAGE,
    ),
    ProxySpecification(
        name="semantic_rule_only_cluster",
        source_column="semantic_rule_only__cluster_id",
        model_column="semantic_rule_only__cluster_id",
        base_columns=BASE_WITH_LANGUAGE,
    ),
    ProxySpecification(
        name="style_cluster_nested_within_language",
        source_column="style_residual_within_language__cluster_id",
        model_column=STYLE_NESTED_COLUMN,
        base_columns=BASE_WITH_LANGUAGE,
    ),
    ProxySpecification(
        name="acquisition_profile",
        source_column="acquisition_profile",
        model_column="acquisition_profile",
        base_columns=BASE_WITH_LANGUAGE,
    ),
    ProxySpecification(
        name="scanner_proxy",
        source_column="scanner_proxy",
        model_column="scanner_proxy",
        base_columns=BASE_WITH_LANGUAGE,
    ),
)

INPUT_COLUMNS = (
    ID_COLUMN,
    TARGET_COLUMN,
    GROUP_COLUMN,
    "text_state",
    "criteria_state",
    "language_proxy",
    "semantic_rule_only__cluster_id",
    "style_residual_within_language__cluster_id",
    "acquisition_profile",
    "scanner_proxy",
    *CONTINUOUS_OUTCOMES,
    *BINARY_OUTCOMES,
)


def deterministic_group_folds(
    groups: pd.Series | np.ndarray,
    *,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
) -> np.ndarray:
    """Assign complete report groups to deterministic, size-balanced folds."""

    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    values = pd.Series(groups, copy=False)
    if values.isna().any():
        raise ValueError("report_group contains null values")
    values = values.astype(str)
    counts = values.value_counts(sort=False)
    if len(counts) < n_splits:
        raise ValueError("fewer report groups than folds")

    def tie_key(value: str) -> str:
        payload = f"adjusted-proxy-association-v1|{seed}|{value}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    ordered = sorted(counts.index, key=lambda value: (-int(counts[value]), tie_key(value)))
    totals = np.zeros(n_splits, dtype=np.int64)
    mapping: dict[str, int] = {}
    for value in ordered:
        fold = int(np.flatnonzero(totals == totals.min())[0])
        mapping[str(value)] = fold
        totals[fold] += int(counts[value])
    result = values.map(mapping).to_numpy(dtype=np.int16)
    if set(result.tolist()) != set(range(n_splits)):
        raise AssertionError("deterministic fold assignment produced an empty fold")
    if pd.DataFrame({"group": values, "fold": result}).groupby("group")["fold"].nunique().max() != 1:
        raise AssertionError("report_group was split across folds")
    return result


def _scanner_availability(frame: pd.DataFrame) -> tuple[bool, int, int, str]:
    scanner = frame[[ID_COLUMN, "scanner_proxy"]].copy()
    per_study_nunique = scanner.groupby(ID_COLUMN, sort=False)["scanner_proxy"].nunique(
        dropna=False
    )
    if per_study_nunique.gt(1).any():
        raise ValueError("scanner_proxy varies within a study")
    per_study = scanner.drop_duplicates(ID_COLUMN)
    total = int(len(per_study))
    covered = int(per_study["scanner_proxy"].notna().sum())
    available = total > 0 and covered == total
    reason = (
        "complete_study_coverage"
        if available
        else f"partial_study_coverage:{covered}/{total}"
    )
    return available, covered, total, reason


def prepare_panel(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate and isolate the fixed Gold-blind analysis columns."""

    missing = sorted(set(INPUT_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"proxy association panel is missing columns: {missing}")
    scanner_available, scanner_covered, scanner_total, scanner_reason = (
        _scanner_availability(frame)
    )
    result = frame.loc[:, list(INPUT_COLUMNS)].copy()
    if result[[ID_COLUMN, TARGET_COLUMN, GROUP_COLUMN]].isna().any().any():
        raise ValueError("study, target, and report group keys must be non-null")
    result[ID_COLUMN] = result[ID_COLUMN].astype(str)
    result[TARGET_COLUMN] = result[TARGET_COLUMN].astype(str)
    result[GROUP_COLUMN] = result[GROUP_COLUMN].astype(str)
    if result.duplicated([ID_COLUMN, TARGET_COLUMN]).any():
        raise ValueError("proxy association panel has duplicate study-target rows")

    required_non_null = (
        "text_state",
        "criteria_state",
        "language_proxy",
        "semantic_rule_only__cluster_id",
        "style_residual_within_language__cluster_id",
        "acquisition_profile",
    )
    if result[list(required_non_null)].isna().any().any():
        raise ValueError("a required non-scanner proxy contains null values")
    categorical = set(BASE_WITH_LANGUAGE)
    categorical.update(spec.source_column for spec in PROXY_SPECIFICATIONS)
    for column in categorical:
        result[column] = result[column].fillna(MISSING_CATEGORY).astype(str)
    result[STYLE_NESTED_COLUMN] = (
        result["language_proxy"]
        + "||"
        + result["style_residual_within_language__cluster_id"]
    )

    for outcome in CONTINUOUS_OUTCOMES:
        result[outcome] = pd.to_numeric(result[outcome], errors="raise").astype(float)
        if not np.isfinite(result[outcome].to_numpy()).all():
            raise ValueError(f"continuous outcome {outcome} contains non-finite values")
    for outcome in BINARY_OUTCOMES:
        values = pd.to_numeric(result[outcome], errors="raise").to_numpy(float)
        if not np.isin(values, [0.0, 1.0]).all():
            raise ValueError(f"binary outcome {outcome} must contain only 0/1")
        result[outcome] = values.astype(np.uint8)

    folds = deterministic_group_folds(result[GROUP_COLUMN])
    result["oof_fold"] = folds
    diagnostics = {
        "rows": int(len(result)),
        "studies": int(result[ID_COLUMN].nunique()),
        "report_groups": int(result[GROUP_COLUMN].nunique()),
        "scanner_available": bool(scanner_available),
        "scanner_covered_studies": scanner_covered,
        "scanner_total_studies": scanner_total,
        "scanner_availability_reason": scanner_reason,
        "gold_columns_loaded": False,
        "loaded_columns": list(INPUT_COLUMNS),
    }
    return result, diagnostics


def _oof_predictions(
    frame: pd.DataFrame,
    *,
    outcome: str,
    columns: tuple[str, ...],
    outcome_type: str,
) -> np.ndarray:
    prediction = np.full(len(frame), np.nan, dtype=np.float64)
    y = frame[outcome].to_numpy(dtype=float)
    for fold in range(N_SPLITS):
        valid = frame["oof_fold"].to_numpy() == fold
        train = ~valid
        if not valid.any() or not train.any():
            raise ValueError(f"empty train/validation partition for fold {fold}")
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
            dtype=np.float64,
        )
        x_train = encoder.fit_transform(frame.loc[train, list(columns)])
        x_valid = encoder.transform(frame.loc[valid, list(columns)])
        if outcome_type == "continuous":
            model = Ridge(
                alpha=RIDGE_ALPHA,
                fit_intercept=True,
                solver="lsqr",
                tol=1e-8,
            )
            model.fit(x_train, y[train])
            prediction[valid] = model.predict(x_valid)
        elif outcome_type == "binary":
            if np.unique(y[train]).size != 2:
                raise ValueError(f"binary outcome {outcome} has one class in fold {fold}")
            model = LogisticRegression(
                C=LOGISTIC_C,
                solver="liblinear",
                fit_intercept=True,
                max_iter=2000,
                tol=1e-8,
                random_state=SEED,
            )
            model.fit(x_train, y[train].astype(np.uint8))
            prediction[valid] = model.predict_proba(x_valid)[:, 1]
        else:
            raise ValueError(f"unknown outcome type: {outcome_type}")
    if not np.isfinite(prediction).all():
        raise AssertionError("OOF prediction coverage is incomplete")
    if outcome_type == "binary":
        prediction = np.clip(prediction, 1e-7, 1.0 - 1e-7)
    return prediction


def _safe_auc(y: np.ndarray, probability: np.ndarray) -> float:
    return (
        float(roc_auc_score(y, probability))
        if np.unique(y).size == 2
        else float("nan")
    )


def _metric_values(
    y: np.ndarray,
    base: np.ndarray,
    augmented: np.ndarray,
    *,
    outcome_type: str,
) -> dict[str, float | str]:
    if outcome_type == "continuous":
        base_rmse = float(np.sqrt(mean_squared_error(y, base)))
        augmented_rmse = float(np.sqrt(mean_squared_error(y, augmented)))
        base_mae = float(mean_absolute_error(y, base))
        augmented_mae = float(mean_absolute_error(y, augmented))
        base_r2 = float(r2_score(y, base)) if np.unique(y).size > 1 else float("nan")
        augmented_r2 = (
            float(r2_score(y, augmented)) if np.unique(y).size > 1 else float("nan")
        )
        return {
            "primary_metric": "rmse",
            "base_primary": base_rmse,
            "augmented_primary": augmented_rmse,
            "primary_improvement": base_rmse - augmented_rmse,
            "base_rmse": base_rmse,
            "augmented_rmse": augmented_rmse,
            "rmse_improvement": base_rmse - augmented_rmse,
            "base_mae": base_mae,
            "augmented_mae": augmented_mae,
            "mae_improvement": base_mae - augmented_mae,
            "base_r2": base_r2,
            "augmented_r2": augmented_r2,
            "r2_improvement": augmented_r2 - base_r2,
            "base_log_loss": float("nan"),
            "augmented_log_loss": float("nan"),
            "log_loss_improvement": float("nan"),
            "base_brier": float("nan"),
            "augmented_brier": float("nan"),
            "brier_improvement": float("nan"),
            "base_roc_auc": float("nan"),
            "augmented_roc_auc": float("nan"),
            "roc_auc_improvement": float("nan"),
        }
    base_loss = float(log_loss(y, base, labels=[0, 1]))
    augmented_loss = float(log_loss(y, augmented, labels=[0, 1]))
    base_brier = float(np.mean((base - y) ** 2))
    augmented_brier = float(np.mean((augmented - y) ** 2))
    base_auc = _safe_auc(y, base)
    augmented_auc = _safe_auc(y, augmented)
    return {
        "primary_metric": "log_loss",
        "base_primary": base_loss,
        "augmented_primary": augmented_loss,
        "primary_improvement": base_loss - augmented_loss,
        "base_rmse": float("nan"),
        "augmented_rmse": float("nan"),
        "rmse_improvement": float("nan"),
        "base_mae": float("nan"),
        "augmented_mae": float("nan"),
        "mae_improvement": float("nan"),
        "base_r2": float("nan"),
        "augmented_r2": float("nan"),
        "r2_improvement": float("nan"),
        "base_log_loss": base_loss,
        "augmented_log_loss": augmented_loss,
        "log_loss_improvement": base_loss - augmented_loss,
        "base_brier": base_brier,
        "augmented_brier": augmented_brier,
        "brier_improvement": base_brier - augmented_brier,
        "base_roc_auc": base_auc,
        "augmented_roc_auc": augmented_auc,
        "roc_auc_improvement": augmented_auc - base_auc,
    }


def _circularity_fields(proxy: str, outcome: str) -> dict[str, Any]:
    outcome_rule_derived = outcome in RULE_DERIVED_OUTCOME_REASONS
    direct = proxy == "semantic_rule_only_cluster" and outcome_rule_derived
    shared_adjustment = (
        proxy == "semantic_rule_only_cluster" and outcome == "adjusted_gap_residual"
    )
    if direct:
        level = "direct_or_partial_rule_construction"
        reason = RULE_DERIVED_OUTCOME_REASONS[outcome]
    elif shared_adjustment:
        level = "shared_text_adjustment_fields"
        reason = (
            "The signed gap outcome is target/text_state/criteria_state adjusted, while the "
            "semantic proxy is built from the same rule-evidence family."
        )
    elif outcome_rule_derived:
        level = "rule_derived_outcome_only"
        reason = RULE_DERIVED_OUTCOME_REASONS[outcome]
    else:
        level = "none_identified"
        reason = (
            "No direct shared deterministic construction was identified; correlated text "
            "components may still remain."
        )
    return {
        "outcome_rule_derived": bool(outcome_rule_derived),
        "proxy_outcome_circularity_flag": bool(direct or shared_adjustment),
        "circularity_level": level,
        "circularity_reason": reason,
    }


def _unavailable_metric_row(
    spec: ProxySpecification,
    outcome: str,
    *,
    rows: int,
    studies: int,
    groups: int,
    reason: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "proxy": spec.name,
        "status": "unavailable",
        "availability_reason": reason,
        "outcome": outcome,
        "outcome_type": OUTCOME_TYPES[outcome],
        "evaluation_scope": "overall",
        "fold": -1,
        "rows": rows,
        "studies": studies,
        "report_groups": groups,
        "base_columns_json": json.dumps(list(spec.base_columns), separators=(",", ":")),
        "added_proxy_columns_json": json.dumps([spec.model_column], separators=(",", ":")),
        "model": "Ridge" if OUTCOME_TYPES[outcome] == "continuous" else "LogisticRegression",
        "ridge_alpha": RIDGE_ALPHA,
        "logistic_c": LOGISTIC_C,
        "primary_metric": "rmse" if OUTCOME_TYPES[outcome] == "continuous" else "log_loss",
    }
    for key in (
        "base_primary",
        "augmented_primary",
        "primary_improvement",
        "relative_primary_improvement",
        "base_rmse",
        "augmented_rmse",
        "rmse_improvement",
        "base_mae",
        "augmented_mae",
        "mae_improvement",
        "base_r2",
        "augmented_r2",
        "r2_improvement",
        "base_log_loss",
        "augmented_log_loss",
        "log_loss_improvement",
        "base_brier",
        "augmented_brier",
        "brier_improvement",
        "base_roc_auc",
        "augmented_roc_auc",
        "roc_auc_improvement",
    ):
        row[key] = float("nan")
    row.update(_circularity_fields(spec.name, outcome))
    row["inference_strength"] = "descriptive_or_directional"
    return row


def _residual_summary_rows(
    frame: pd.DataFrame,
    *,
    spec: ProxySpecification,
    outcome: str,
    base: np.ndarray,
    augmented: np.ndarray,
    minimum_studies: int,
) -> list[dict[str, Any]]:
    y = frame[outcome].to_numpy(dtype=float)
    working = frame[
        [ID_COLUMN, GROUP_COLUMN, TARGET_COLUMN, spec.model_column]
    ].copy()
    working["observed"] = y
    working["base_prediction"] = base
    working["augmented_prediction"] = augmented
    working["base_residual"] = y - base
    working["augmented_residual"] = y - augmented
    circularity = _circularity_fields(spec.name, outcome)
    rows: list[dict[str, Any]] = []
    for scope in ("overall", "target"):
        keys = [spec.model_column] if scope == "overall" else [TARGET_COLUMN, spec.model_column]
        for key, part in working.groupby(keys, sort=True, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            target = "__all__" if scope == "overall" else str(key[0])
            proxy_value = str(key[-1])
            studies = int(part[ID_COLUMN].nunique())
            suppressed = studies < minimum_studies
            reference = working if scope == "overall" else working.loc[
                working[TARGET_COLUMN].eq(target)
            ]
            outcome_sd = float(reference["observed"].std(ddof=0))
            residual_sd = float(reference["base_residual"].std(ddof=0))
            row: dict[str, Any] = {
                "proxy": spec.name,
                "outcome": outcome,
                "outcome_type": OUTCOME_TYPES[outcome],
                "scope": scope,
                "target": target,
                "proxy_value": proxy_value,
                "cells": int(len(part)),
                "studies": studies,
                "report_groups": int(part[GROUP_COLUMN].nunique()),
                "minimum_studies": int(minimum_studies),
                "association_suppressed": bool(suppressed),
                "inference_strength": "descriptive_or_directional",
                **circularity,
            }
            estimates = {
                "observed_mean": float(part["observed"].mean()),
                "base_prediction_mean": float(part["base_prediction"].mean()),
                "augmented_prediction_mean": float(part["augmented_prediction"].mean()),
                "base_residual_mean": float(part["base_residual"].mean()),
                "augmented_residual_mean": float(part["augmented_residual"].mean()),
                "base_residual_mean_absolute": float(part["base_residual"].abs().mean()),
                "augmented_residual_mean_absolute": float(
                    part["augmented_residual"].abs().mean()
                ),
                "standardized_base_residual_outcome_sd": (
                    float(part["base_residual"].mean() / outcome_sd)
                    if outcome_sd > 0
                    else float("nan")
                ),
                "standardized_base_residual_residual_sd": (
                    float(part["base_residual"].mean() / residual_sd)
                    if residual_sd > 0
                    else float("nan")
                ),
            }
            if suppressed:
                estimates = {name: float("nan") for name in estimates}
            row.update(estimates)
            rows.append(row)
    return rows


def run_adjusted_proxy_audit(
    frame: pd.DataFrame,
    *,
    minimum_studies: int = MINIMUM_STUDIES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run the fixed grouped-OOF proxy audit and return its four artifacts."""

    if minimum_studies <= 0:
        raise ValueError("minimum_studies must be positive")
    panel, diagnostics = prepare_panel(frame)
    scanner_available = bool(diagnostics["scanner_available"])
    base_cache: dict[tuple[tuple[str, ...], str], np.ndarray] = {}
    metric_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []

    for spec in PROXY_SPECIFICATIONS:
        available = spec.name != "scanner_proxy" or scanner_available
        if not available:
            for outcome in OUTCOME_TYPES:
                metric_rows.append(
                    _unavailable_metric_row(
                        spec,
                        outcome,
                        rows=len(panel),
                        studies=panel[ID_COLUMN].nunique(),
                        groups=panel[GROUP_COLUMN].nunique(),
                        reason=str(diagnostics["scanner_availability_reason"]),
                    )
                )
            continue
        augmented_columns = (*spec.base_columns, spec.model_column)
        if len(augmented_columns) != len(spec.base_columns) + 1:
            raise AssertionError("proxy model must add exactly one categorical column")
        for outcome, outcome_type in OUTCOME_TYPES.items():
            cache_key = (spec.base_columns, outcome)
            if cache_key not in base_cache:
                base_cache[cache_key] = _oof_predictions(
                    panel,
                    outcome=outcome,
                    columns=spec.base_columns,
                    outcome_type=outcome_type,
                )
            base = base_cache[cache_key]
            augmented = _oof_predictions(
                panel,
                outcome=outcome,
                columns=augmented_columns,
                outcome_type=outcome_type,
            )
            for fold in (-1, *range(N_SPLITS)):
                mask = (
                    np.ones(len(panel), dtype=bool)
                    if fold == -1
                    else panel["oof_fold"].to_numpy() == fold
                )
                values = _metric_values(
                    panel.loc[mask, outcome].to_numpy(dtype=float),
                    base[mask],
                    augmented[mask],
                    outcome_type=outcome_type,
                )
                relative = (
                    float(values["primary_improvement"] / values["base_primary"])
                    if float(values["base_primary"]) > 0
                    else float("nan")
                )
                metric_rows.append(
                    {
                        "proxy": spec.name,
                        "status": "available",
                        "availability_reason": "complete_required_coverage",
                        "outcome": outcome,
                        "outcome_type": outcome_type,
                        "evaluation_scope": "overall" if fold == -1 else "fold",
                        "fold": fold,
                        "rows": int(mask.sum()),
                        "studies": int(panel.loc[mask, ID_COLUMN].nunique()),
                        "report_groups": int(panel.loc[mask, GROUP_COLUMN].nunique()),
                        "base_columns_json": json.dumps(
                            list(spec.base_columns), separators=(",", ":")
                        ),
                        "added_proxy_columns_json": json.dumps(
                            [spec.model_column], separators=(",", ":")
                        ),
                        "model": "Ridge" if outcome_type == "continuous" else "LogisticRegression",
                        "ridge_alpha": RIDGE_ALPHA,
                        "logistic_c": LOGISTIC_C,
                        "relative_primary_improvement": relative,
                        **values,
                        **_circularity_fields(spec.name, outcome),
                        "inference_strength": "descriptive_or_directional",
                    }
                )
            residual_rows.extend(
                _residual_summary_rows(
                    panel,
                    spec=spec,
                    outcome=outcome,
                    base=base,
                    augmented=augmented,
                    minimum_studies=minimum_studies,
                )
            )

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["proxy", "outcome", "evaluation_scope", "fold"], kind="mergesort"
    ).reset_index(drop=True)
    residuals = pd.DataFrame(residual_rows).sort_values(
        ["proxy", "outcome", "scope", "target", "proxy_value"], kind="mergesort"
    ).reset_index(drop=True)
    folds = (
        panel.groupby([GROUP_COLUMN, "oof_fold"], as_index=False)
        .agg(
            cells=(ID_COLUMN, "size"),
            studies=(ID_COLUMN, "nunique"),
        )
        .sort_values(["oof_fold", GROUP_COLUMN], kind="mergesort")
        .reset_index(drop=True)
    )
    diagnostics.update(
        {
            "outcomes": dict(OUTCOME_TYPES),
            "n_splits": N_SPLITS,
            "split_group": GROUP_COLUMN,
            "fold_assignment": "deterministic size-balanced report-group assignment",
            "ridge_alpha": RIDGE_ALPHA,
            "logistic_c": LOGISTIC_C,
            "regularization_selected_from_outcomes": False,
            "minimum_studies": int(minimum_studies),
            "metric_rows": int(len(metrics)),
            "residual_summary_rows": int(len(residuals)),
            "proxy_availability": {
                spec.name: (
                    {
                        "available": False,
                        "reason": diagnostics["scanner_availability_reason"],
                    }
                    if spec.name == "scanner_proxy" and not scanner_available
                    else {"available": True, "reason": "complete_required_coverage"}
                )
                for spec in PROXY_SPECIFICATIONS
            },
            "primary_outcomes_include_gold": False,
            "inference_strength": "descriptive_or_directional",
            "effect_size_definition": (
                "mean base OOF residual divided by outcome SD and by base-residual SD; "
                "estimates suppressed below 80 unique studies"
            ),
            "interpretation_boundary": (
                "OOF metric deltas measure incremental predictability and residual summaries "
                "measure descriptive concentration. Neither is a causal, reader-identity, "
                "institution, or regional effect."
            ),
            "forbidden_claims": [
                "causal language, scanner, acquisition, semantic, or style effect",
                "specific radiologist or doctor identity",
                "regional attribution without an external validated key",
                "independent image adjudication",
            ],
        }
    )
    return metrics, residuals, folds, diagnostics
