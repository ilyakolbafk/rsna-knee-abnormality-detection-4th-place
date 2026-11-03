"""Deterministic helpers for report-morphology error-risk analysis.

The module deliberately treats image-versus-label distance as a descriptive
anomaly proxy.  It never rewrites labels and it never interprets language
proxies as countries, sites, or readers.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .constants import ID_COLUMN, TARGET_COLUMNS


PRIMARY_SCORE = "public_image_oof_mean4"
SENSITIVITY_SCORES = (
    "dino_public20_equal_oof",
    "public_rad_oof_mean3",
    "directional_v45_final_rank",
)
CONTEXT_SCORE_COLUMNS = ("dinov3_rank_mean", "coatnet_v45_no_swin_rank")
RAD_COMPONENT_COLUMNS = (
    "radimagenet_oof",
    "radimagenet_v52_oof",
    "radimagenet_e11_public_oof",
)
SOURCE_SCORE_COLUMNS = (
    PRIMARY_SCORE,
    "public_image_oof_std4",
    "dino_public20_equal_oof",
    *RAD_COMPONENT_COLUMNS,
    "directional_v45_final_rank",
    *CONTEXT_SCORE_COLUMNS,
)
ALL_SCORE_COLUMNS = (PRIMARY_SCORE, *SENSITIVITY_SCORES, *CONTEXT_SCORE_COLUMNS)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_deterministic_gzip_csv(frame: pd.DataFrame, path: str | Path) -> None:
    """Write byte-stable UTF-8 CSV in gzip with no path or time in the header."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text_buffer = io.StringIO(newline="")
    frame.to_csv(text_buffer, index=False, lineterminator="\n")
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=9,
            fileobj=raw,
            mtime=0,
        ) as compressed:
            compressed.write(text_buffer.getvalue().encode("utf-8"))
    os.replace(temporary, destination)


def deterministic_uid_fold(uid: str, n_folds: int = 5) -> int:
    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    token = hashlib.sha256(str(uid).encode("utf-8")).hexdigest()
    return int(token[:16], 16) % n_folds


def _validate_probability_column(frame: pd.DataFrame, column: str) -> None:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{column} contains missing or non-finite values")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError(f"{column} contains values outside [0, 1]")


def build_morphology_error_panel(
    manifest: pd.DataFrame,
    scores: pd.DataFrame,
) -> pd.DataFrame:
    """Construct the exact study-target panel and preregistered gap outcomes."""

    required_manifest = {
        ID_COLUMN,
        "report_group",
        "is_gold",
        *TARGET_COLUMNS,
        *(f"confidence__{target}" for target in TARGET_COLUMNS),
    }
    missing_manifest = sorted(required_manifest.difference(manifest.columns))
    if missing_manifest:
        raise ValueError(f"manifest columns missing: {missing_manifest}")
    if manifest[ID_COLUMN].duplicated().any():
        raise ValueError("manifest StudyInstanceUID values are not unique")

    required_scores = {
        ID_COLUMN,
        "target",
        "gold",
        "is_gold",
        "llm_v4_blend",
        *SOURCE_SCORE_COLUMNS,
    }
    missing_scores = sorted(required_scores.difference(scores.columns))
    if missing_scores:
        raise ValueError(f"score columns missing: {missing_scores}")
    if scores.duplicated([ID_COLUMN, "target"]).any():
        raise ValueError("score panel study-target keys are not unique")

    label_long = manifest[
        [ID_COLUMN, "report_group", "is_gold", *TARGET_COLUMNS]
    ].melt(
        id_vars=[ID_COLUMN, "report_group", "is_gold"],
        value_vars=list(TARGET_COLUMNS),
        var_name="target",
        value_name="canonical_label_probability",
    )
    confidence_wide = manifest[
        [ID_COLUMN, *(f"confidence__{target}" for target in TARGET_COLUMNS)]
    ].rename(columns={f"confidence__{target}": target for target in TARGET_COLUMNS})
    confidence_long = confidence_wide.melt(
        id_vars=[ID_COLUMN],
        value_vars=list(TARGET_COLUMNS),
        var_name="target",
        value_name="canonical_confidence",
    )
    label_long = label_long.merge(
        confidence_long,
        on=[ID_COLUMN, "target"],
        how="left",
        validate="one_to_one",
    )

    score_columns = [
        ID_COLUMN,
        "target",
        "gold",
        "is_gold",
        "llm_v4_blend",
        *SOURCE_SCORE_COLUMNS,
    ]
    score_subset = scores[score_columns].rename(columns={"is_gold": "score_is_gold"})
    panel = label_long.merge(
        score_subset,
        on=[ID_COLUMN, "target"],
        how="left",
        validate="one_to_one",
    )
    if not panel["is_gold"].astype(bool).equals(panel["score_is_gold"].astype(bool)):
        raise ValueError("Gold identity differs between manifest and score panel")
    panel = panel.drop(columns="score_is_gold")

    panel["public_rad_oof_mean3"] = panel[list(RAD_COMPONENT_COLUMNS)].mean(axis=1)
    if panel[list(ALL_SCORE_COLUMNS)].isna().any().any():
        raise ValueError("score join left missing values")

    _validate_probability_column(panel, "canonical_label_probability")
    _validate_probability_column(panel, "canonical_confidence")
    _validate_probability_column(panel, "llm_v4_blend")
    for column in (*SOURCE_SCORE_COLUMNS, "public_rad_oof_mean3"):
        _validate_probability_column(panel, column)

    gold_mask = panel["is_gold"].astype(bool)
    gold_values = panel.loc[gold_mask, "canonical_label_probability"].to_numpy(float)
    if not np.isin(gold_values, [0.0, 1.0]).all():
        raise ValueError("canonical Gold labels are not exact 0/1")
    score_gold = panel.loc[gold_mask, "gold"].to_numpy(float)
    if not np.array_equal(gold_values, score_gold):
        raise ValueError("canonical Gold override differs from official Gold")

    panel["analysis_primary_eligible"] = gold_mask | (
        (panel["canonical_confidence"] >= 0.10)
        & panel["canonical_label_probability"].ne(0.5)
    )
    panel["canonical_hard_label"] = (
        panel["canonical_label_probability"] > 0.5
    ).astype("int8")
    panel["full_hard_label_ge_0_5"] = (
        panel["canonical_label_probability"] >= 0.5
    ).astype("int8")
    panel["legacy_llm_v4_hard_label"] = (panel["llm_v4_blend"] >= 0.5).astype(
        "int8"
    )
    panel["decisive_label"] = (
        (panel["canonical_label_probability"] <= 0.25)
        | (panel["canonical_label_probability"] >= 0.75)
    )
    panel["legacy_llm_v4_hard_error_vs_gold"] = np.where(
        gold_mask,
        (
            panel["legacy_llm_v4_hard_label"].to_numpy()
            != panel["gold"].to_numpy()
        ).astype(float),
        np.nan,
    )

    strata = ["target", "canonical_hard_label"]
    eligible = panel["analysis_primary_eligible"].astype(bool)
    for score_column in ALL_SCORE_COLUMNS:
        gap = (panel[score_column] - panel["canonical_hard_label"]).abs()
        panel[f"abs_gap__{score_column}"] = gap
        panel[f"signed_residual__{score_column}"] = (
            panel[score_column] - panel["canonical_hard_label"]
        )
        panel[f"hard_disagreement__{score_column}"] = (
            (panel[score_column] >= 0.5) != panel["canonical_hard_label"].astype(bool)
        ).astype("int8")
        panel[f"continuous_gap__{score_column}"] = (
            panel[score_column] - panel["canonical_label_probability"]
        ).abs()
        adjusted = pd.Series(np.nan, index=panel.index, dtype=float)
        eligible_gap = gap.loc[eligible]
        adjusted.loc[eligible] = eligible_gap - eligible_gap.groupby(
            [panel.loc[eligible, column] for column in strata], sort=False
        ).transform("mean")
        panel[f"adjusted_gap__{score_column}"] = adjusted

    full_gap = (panel[PRIMARY_SCORE] - panel["full_hard_label_ge_0_5"]).abs()
    panel["full_hardening_gap_sensitivity"] = full_gap
    panel["full_hardening_adjusted_gap_sensitivity"] = full_gap - full_gap.groupby(
        [panel[column] for column in ["target", "full_hard_label_ge_0_5"]],
        sort=False,
    ).transform("mean")

    panel["primary_gap"] = panel[f"abs_gap__{PRIMARY_SCORE}"]
    panel["primary_adjusted_gap"] = panel[f"adjusted_gap__{PRIMARY_SCORE}"]
    panel["primary_continuous_gap"] = panel[f"continuous_gap__{PRIMARY_SCORE}"]
    panel["primary_hard_disagreement"] = panel[
        f"hard_disagreement__{PRIMARY_SCORE}"
    ]
    panel["primary_image_family_dispersion"] = panel["public_image_oof_std4"]
    panel = panel.sort_values([ID_COLUMN, "target"], kind="mergesort").reset_index(
        drop=True
    )
    return panel


def add_secondary_adjustment(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add a fixed weighted linear residual sensitivity.

    Each target-by-hard-label stratum contributes the same total weight.  This
    is descriptive adjustment, not a fitted error model used for deployment.
    """

    required = {
        "target",
        "canonical_hard_label",
        "language_proxy",
        "report_char_count",
        "canonical_confidence",
        "acquisition_profile",
        "primary_gap",
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"secondary adjustment columns missing: {missing}")
    if panel[list(required)].isna().any().any():
        raise ValueError("secondary adjustment inputs contain missing values")

    if "analysis_primary_eligible" not in panel:
        raise ValueError("analysis_primary_eligible is required")
    eligible = panel["analysis_primary_eligible"].astype(bool)
    fit_panel = panel.loc[eligible].reset_index(drop=True)

    categorical = pd.get_dummies(
        fit_panel[["target", "language_proxy", "acquisition_profile"]].astype(str),
        drop_first=True,
        dtype=float,
    )
    numeric = pd.DataFrame(
        {
            "intercept": np.ones(len(fit_panel), dtype=float),
            "canonical_hard_label": fit_panel["canonical_hard_label"].to_numpy(float),
            "canonical_confidence": fit_panel["canonical_confidence"].to_numpy(float),
            "log1p_report_char_count": np.log1p(
                fit_panel["report_char_count"].to_numpy(float)
            ),
        }
    )
    design = pd.concat(
        [numeric.reset_index(drop=True), categorical.reset_index(drop=True)], axis=1
    )
    stratum_size = fit_panel.groupby(
        ["target", "canonical_hard_label"], sort=False
    )[ID_COLUMN].transform("size")
    weights = 1.0 / stratum_size.to_numpy(float)
    weights *= len(weights) / weights.sum()
    root_weights = np.sqrt(weights)
    matrix = design.to_numpy(float)
    outcome = fit_panel["primary_gap"].to_numpy(float)
    coefficients, *_ = np.linalg.lstsq(
        matrix * root_weights[:, None],
        outcome * root_weights,
        rcond=None,
    )
    fitted = matrix @ coefficients
    result = panel.copy()
    result["secondary_adjusted_gap"] = np.nan
    result.loc[eligible, "secondary_adjusted_gap"] = outcome - fitted
    diagnostics = {
        "design_columns": design.columns.tolist(),
        "design_rank": int(np.linalg.matrix_rank(matrix)),
        "coefficient_count": int(len(coefficients)),
        "weighted_residual_mean": float(
            np.average(result.loc[eligible, "secondary_adjusted_gap"], weights=weights)
        ),
        "eligible_cell_count": int(eligible.sum()),
        "scope": "whole-panel descriptive weighted linear residual sensitivity",
    }
    return result, diagnostics


def empirical_quintile(
    values: pd.Series,
    *,
    prefix: str,
) -> tuple[pd.Series, list[float]]:
    """Return deterministic empirical quintile labels and the used edges."""

    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"{prefix} contains missing or non-numeric values")
    array = numeric.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{prefix} contains non-finite values")
    edges = np.unique(np.quantile(array, np.linspace(0.0, 1.0, 6)))
    if len(edges) == 1:
        return pd.Series(f"{prefix}_Q1", index=values.index), [float(edges[0])]
    labels = [f"{prefix}_Q{index + 1}" for index in range(len(edges) - 1)]
    binned = pd.cut(
        numeric,
        bins=edges,
        labels=labels,
        include_lowest=True,
        duplicates="drop",
    ).astype("string")
    if binned.isna().any():
        raise ValueError(f"{prefix} quintile binning produced missing values")
    return binned, [float(value) for value in edges]


def build_segment_memberships(
    features: pd.DataFrame,
    *,
    categorical_features: Mapping[str, str],
    continuous_features: Mapping[str, str],
    binary_features: Mapping[str, str],
    interactions: Sequence[tuple[str, str, str]],
) -> tuple[pd.DataFrame, dict[str, list[float]], pd.DataFrame]:
    """Materialize preregistered UID-to-segment memberships.

    Mapping values are feature-family names.  Interactions are
    ``(left_column, right_column, family)`` and operate on already materialized
    categorical values or quintile columns.
    """

    if features[ID_COLUMN].duplicated().any():
        raise ValueError("report feature rows are not unique by StudyInstanceUID")
    working = features.copy()
    rows: list[pd.DataFrame] = []
    bin_edges: dict[str, list[float]] = {}

    def append_feature(column: str, family: str, values: pd.Series) -> None:
        safe_values = values.astype("string").fillna("missing")
        local = pd.DataFrame(
            {
                ID_COLUMN: working[ID_COLUMN].astype(str),
                "segment_family": family,
                "segment_feature": column,
                "segment_value": safe_values,
            }
        )
        local["segment_id"] = (
            local["segment_family"]
            + "|"
            + local["segment_feature"]
            + "="
            + local["segment_value"]
        )
        rows.append(local)

    for column, family in categorical_features.items():
        if column not in working:
            raise ValueError(f"categorical segment feature missing: {column}")
        append_feature(column, family, working[column])

    for column, family in continuous_features.items():
        if column not in working:
            raise ValueError(f"continuous segment feature missing: {column}")
        quintile_column = f"bin__{column}"
        working[quintile_column], bin_edges[column] = empirical_quintile(
            working[column], prefix=column
        )
        append_feature(quintile_column, family, working[quintile_column])

    for column, family in binary_features.items():
        if column not in working:
            raise ValueError(f"binary segment feature missing: {column}")
        values = working[column]
        if values.isna().any():
            raise ValueError(f"binary segment feature {column} contains missing values")
        normalized = values.astype(bool).map({True: "true", False: "false"})
        append_feature(column, family, normalized)

    for left, right, family in interactions:
        if left not in working or right not in working:
            raise ValueError(f"interaction inputs missing: {left}, {right}")
        column = f"interaction__{left}__x__{right}"
        working[column] = (
            working[left].astype("string").fillna("missing")
            + "__x__"
            + working[right].astype("string").fillna("missing")
        )
        append_feature(column, family, working[column])

    memberships = pd.concat(rows, ignore_index=True)
    if memberships.duplicated([ID_COLUMN, "segment_feature"]).any():
        raise ValueError("a study has multiple values for the same segment feature")
    memberships = memberships.sort_values(
        ["segment_family", "segment_feature", "segment_value", ID_COLUMN],
        kind="mergesort",
    ).reset_index(drop=True)
    return memberships, bin_edges, working


def _make_preprocessor(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> ColumnTransformer:
    numeric = [column for column in columns if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in columns if column not in numeric]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        ),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        raise ValueError("at least one feature is required")
    return ColumnTransformer(transformers, remainder="drop")


def _cross_validated_ridge_predictions(
    frame: pd.DataFrame,
    outcome: np.ndarray,
    columns: Sequence[str],
    *,
    n_folds: int,
    group_column: str,
) -> np.ndarray:
    if group_column not in frame:
        raise ValueError(f"cross-validation group column missing: {group_column}")
    if frame[group_column].isna().any():
        raise ValueError(f"cross-validation group column {group_column} has missing values")
    folds = frame[group_column].map(
        lambda value: deterministic_uid_fold(value, n_folds)
    )
    predictions = np.full(len(frame), np.nan, dtype=float)
    for fold in range(n_folds):
        train = folds.ne(fold).to_numpy()
        valid = folds.eq(fold).to_numpy()
        if not train.any() or not valid.any():
            raise ValueError(f"empty deterministic fold {fold}")
        estimator = Pipeline(
            [
                ("preprocess", _make_preprocessor(frame.loc[train], columns)),
                ("ridge", Ridge(alpha=10.0, solver="lsqr")),
            ]
        )
        estimator.fit(frame.loc[train, columns], outcome[train])
        predictions[valid] = estimator.predict(frame.loc[valid, columns])
    if not np.isfinite(predictions).all():
        raise ValueError("cross-validated predictions are not complete and finite")
    return predictions


def cross_validated_feature_family_ablation(
    features: pd.DataFrame,
    study_outcome: pd.DataFrame,
    family_columns: Mapping[str, Sequence[str]],
    *,
    n_folds: int = 5,
    group_column: str = "report_group",
) -> pd.DataFrame:
    """Measure fixed 5-fold predictive contribution by family ablation."""

    outcome_required = {ID_COLUMN, "study_primary_adjusted_gap"}
    if not outcome_required.issubset(study_outcome.columns):
        raise ValueError("study outcome columns are incomplete")
    if group_column not in features:
        raise ValueError(f"feature-importance group column missing: {group_column}")
    merged = features.merge(
        study_outcome[[ID_COLUMN, "study_primary_adjusted_gap"]],
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
    ).sort_values(ID_COLUMN, kind="mergesort").reset_index(drop=True)
    if len(merged) != len(features):
        raise ValueError("feature-importance outcome join lost studies")
    all_columns = sorted({column for columns in family_columns.values() for column in columns})
    missing = sorted(set(all_columns).difference(merged.columns))
    if missing:
        raise ValueError(f"feature-importance columns missing: {missing}")
    outcome = merged["study_primary_adjusted_gap"].to_numpy(float)
    full_prediction = _cross_validated_ridge_predictions(
        merged, outcome, all_columns, n_folds=n_folds, group_column=group_column
    )
    full_mae = float(mean_absolute_error(outcome, full_prediction))
    full_mse = float(mean_squared_error(outcome, full_prediction))
    full_r2 = float(r2_score(outcome, full_prediction))

    rows: list[dict[str, Any]] = []
    for family, removed_columns in sorted(family_columns.items()):
        keep = [column for column in all_columns if column not in set(removed_columns)]
        if not keep:
            raise ValueError(f"ablating {family} removes all features")
        prediction = _cross_validated_ridge_predictions(
            merged, outcome, keep, n_folds=n_folds, group_column=group_column
        )
        ablated_mae = float(mean_absolute_error(outcome, prediction))
        ablated_mse = float(mean_squared_error(outcome, prediction))
        rows.append(
            {
                "feature_family": family,
                "feature_count": int(len(removed_columns)),
                "full_oof_mae": full_mae,
                "ablated_oof_mae": ablated_mae,
                "delta_mae_ablated_minus_full": ablated_mae - full_mae,
                "full_oof_mse": full_mse,
                "ablated_oof_mse": ablated_mse,
                "delta_mse_ablated_minus_full": ablated_mse - full_mse,
                "full_oof_r2": full_r2,
                "ablated_oof_r2": float(r2_score(outcome, prediction)),
                "interpretation": "descriptive fixed-model ablation; positive delta means unique predictive contribution",
            }
        )
    return pd.DataFrame(rows).sort_values(
        "delta_mse_ablated_minus_full", ascending=False, kind="mergesort"
    ).reset_index(drop=True)


def materialize_candidate_confidence_multipliers(
    panel: pd.DataFrame,
    memberships: pd.DataFrame,
    overall_summary: pd.DataFrame,
    target_summary: pd.DataFrame,
    *,
    floor: float = 0.65,
) -> pd.DataFrame:
    """Materialize, but do not apply, robust-segment confidence candidates."""

    robust_overall = overall_summary.loc[
        overall_summary["robust_candidate"].astype(bool),
        ["segment_id", "candidate_multiplier"],
    ].copy()
    robust_overall["target"] = "__all__"
    robust_target = target_summary.loc[
        target_summary["robust_candidate"].astype(bool),
        ["target", "segment_id", "candidate_multiplier"],
    ].copy()
    robust = pd.concat([robust_overall, robust_target], ignore_index=True)

    triggers = memberships[[ID_COLUMN, "segment_id"]].merge(
        robust,
        on="segment_id",
        how="inner",
        validate="many_to_many",
    )
    all_target_triggers = triggers.loc[triggers["target"].eq("__all__")].drop(
        columns="target"
    )
    if not all_target_triggers.empty:
        target_frame = pd.DataFrame({"target": list(TARGET_COLUMNS)})
        all_target_triggers = all_target_triggers.merge(target_frame, how="cross")
    specific_triggers = triggers.loc[~triggers["target"].eq("__all__")]
    triggers = pd.concat([all_target_triggers, specific_triggers], ignore_index=True)

    keys = [ID_COLUMN, "target"]
    if triggers.empty:
        aggregated = pd.DataFrame(
            columns=[*keys, "candidate_confidence_multiplier", "trigger_segment_count", "trigger_segment_ids"]
        )
    else:
        aggregated = (
            triggers.groupby(keys, sort=False)
            .agg(
                candidate_confidence_multiplier=("candidate_multiplier", "min"),
                trigger_segment_count=("segment_id", "nunique"),
                trigger_segment_ids=(
                    "segment_id",
                    lambda values: ";".join(sorted(set(map(str, values)))),
                ),
            )
            .reset_index()
        )
    output = panel[
        [
            ID_COLUMN,
            "target",
            "is_gold",
            "analysis_primary_eligible",
            "canonical_label_probability",
            "canonical_confidence",
        ]
    ].merge(aggregated, on=keys, how="left", validate="one_to_one")
    output["candidate_confidence_multiplier"] = output[
        "candidate_confidence_multiplier"
    ].fillna(1.0).clip(lower=floor, upper=1.0)
    output["trigger_segment_count"] = output["trigger_segment_count"].fillna(0).astype(int)
    output["trigger_segment_ids"] = output["trigger_segment_ids"].fillna("")
    gold = output["is_gold"].astype(bool)
    inactive = ~output["analysis_primary_eligible"].astype(bool)
    output.loc[inactive, "candidate_confidence_multiplier"] = 1.0
    output.loc[inactive, "trigger_segment_count"] = 0
    output.loc[inactive, "trigger_segment_ids"] = ""
    output.loc[gold, "candidate_confidence_multiplier"] = 1.0
    output.loc[gold, "trigger_segment_count"] = 0
    output.loc[gold, "trigger_segment_ids"] = ""
    output["candidate_confidence_if_tested"] = (
        output["canonical_confidence"]
        * output["candidate_confidence_multiplier"]
    )
    output["label_probability_changed"] = False
    output["deployment_status"] = "NOT_APPLIED_REQUIRES_GROUPED_OOF"
    return output.sort_values(keys, kind="mergesort").reset_index(drop=True)
