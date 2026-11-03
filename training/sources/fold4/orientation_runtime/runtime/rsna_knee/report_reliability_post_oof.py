"""Proxy-only post-OOF mechanics for the frozen report-reliability contract.

All functions accept explicit in-memory frames.  They do not read Gold, select
features, invent strata, or alter predictions.  The acquisition profile,
LOTO-count binning, hierarchical stratum fallback, component endpoints, and
strict proxy event below are direct translations of contract v3.4.0.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import json
from typing import Any

import numpy as np
import pandas as pd

from .constants import ID_COLUMN
from .report_reliability_evaluation import (
    compare_report_group_weighted_predictions,
    review_fraction_capture_enrichment,
)


REPORT_GROUP_COLUMN = "report_group"
TARGET_COLUMN = "target"
DIRECTION_COLUMN = "direction"
OUTER_FOLD_COLUMN = "outer_fold"
PROFILE_COLUMN = "acquisition_profile_pooled"
LOTO_BIN_COLUMN = "identity_loto_positive_count_bin"
PERMUTATION_STRATUM_COLUMN = "permutation_stratum"

ACQUISITION_COLUMNS: tuple[str, ...] = (
    "acquisition_manufacturer_mode",
    "acquisition_model_mode",
    "acquisition_field_strength_mode",
)


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
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be one-dimensional and finite")
    return result


def _nonempty_text(values: Any, *, name: str) -> np.ndarray:
    raw = np.asarray(values, dtype=object)
    if raw.ndim != 1 or pd.isna(raw).any():
        raise ValueError(f"{name} must be one-dimensional and non-null")
    result = np.asarray([str(value).strip() for value in raw], dtype=object)
    if np.any(result == ""):
        raise ValueError(f"{name} contains empty values")
    return result


def _stable_mean(values: pd.Series) -> float:
    numeric = np.sort(np.asarray(values, dtype=np.float64))
    return float(np.mean(numeric, dtype=np.float64))


def _lexicographic_mode(values: pd.Series) -> str:
    normalized = values.astype(str)
    counts = normalized.value_counts(dropna=False, sort=False)
    maximum = int(counts.max())
    return min(str(value) for value in counts.index[counts.eq(maximum)])


def _smallest_integer_mode(values: pd.Series) -> int:
    numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("other_positive_count must contain finite integers")
    integer = pd.Series(numeric.astype(np.int64))
    counts = integer.value_counts(dropna=False, sort=False)
    maximum = int(counts.max())
    return int(min(int(value) for value in counts.index[counts.eq(maximum)]))


def _missing_profile_part(value: object) -> str:
    if value is None or pd.isna(value):
        return "__MISSING__"
    normalized = str(value).strip()
    return normalized if normalized else "__MISSING__"


def build_pooled_acquisition_profile_map(
    feature_frame: pd.DataFrame,
    *,
    eligible_report_groups: Iterable[str],
    minimum_profile_groups: int = 50,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the frozen outcome-blind acquisition profile per report group."""

    if not isinstance(minimum_profile_groups, (int, np.integer)) or int(
        minimum_profile_groups
    ) < 1:
        raise ValueError("minimum_profile_groups must be a positive integer")
    required = (ID_COLUMN, REPORT_GROUP_COLUMN, *ACQUISITION_COLUMNS)
    _require_columns(feature_frame, required, name="acquisition feature input")
    eligible = frozenset(str(value) for value in eligible_report_groups)
    if not eligible:
        raise ValueError("eligible_report_groups must not be empty")

    source = feature_frame.loc[
        feature_frame[REPORT_GROUP_COLUMN].astype(str).isin(eligible), list(required)
    ].copy()
    source[ID_COLUMN] = _nonempty_text(source[ID_COLUMN], name=ID_COLUMN)
    source[REPORT_GROUP_COLUMN] = _nonempty_text(
        source[REPORT_GROUP_COLUMN], name=REPORT_GROUP_COLUMN
    )
    if set(source[REPORT_GROUP_COLUMN]) != set(eligible):
        missing = len(set(eligible).difference(source[REPORT_GROUP_COLUMN]))
        raise ValueError(f"eligible acquisition group coverage missing={missing}")

    # The feature panel is target-long.  Acquisition fields and report group
    # must be invariant within study before target duplicates are removed.
    invariant_columns = [REPORT_GROUP_COLUMN, *ACQUISITION_COLUMNS]
    drift = source.groupby(ID_COLUMN, sort=True, observed=True)[
        invariant_columns
    ].nunique(dropna=False)
    if (drift > 1).any().any():
        raise ValueError("acquisition fields drift within StudyInstanceUID")
    studies = source.sort_values(ID_COLUMN, kind="mergesort").drop_duplicates(
        ID_COLUMN, keep="first"
    )
    studies["_acquisition_profile"] = [
        "|".join(_missing_profile_part(value) for value in values)
        for values in studies[list(ACQUISITION_COLUMNS)].itertuples(
            index=False, name=None
        )
    ]
    group_profiles = (
        studies.groupby(REPORT_GROUP_COLUMN, sort=True, observed=True)[
            "_acquisition_profile"
        ]
        .agg(_lexicographic_mode)
        .rename("acquisition_profile")
        .reset_index()
    )
    counts = group_profiles["acquisition_profile"].value_counts(
        sort=False, dropna=False
    )
    common = frozenset(
        str(profile)
        for profile, count in counts.items()
        if int(count) >= int(minimum_profile_groups)
    )
    group_profiles[PROFILE_COLUMN] = group_profiles["acquisition_profile"].where(
        group_profiles["acquisition_profile"].isin(common), "__OTHER_RARE__"
    )
    group_profiles = group_profiles.sort_values(
        REPORT_GROUP_COLUMN, kind="mergesort"
    ).reset_index(drop=True)
    diagnostics = {
        "status": "PASS",
        "eligible_report_groups": int(len(group_profiles)),
        "eligible_studies": int(len(studies)),
        "raw_profile_count": int(len(counts)),
        "common_profile_count": int(len(common)),
        "pooled_rare_report_groups": int(
            group_profiles[PROFILE_COLUMN].eq("__OTHER_RARE__").sum()
        ),
        "minimum_profile_groups": int(minimum_profile_groups),
        "outcome_columns_used": False,
    }
    return group_profiles, diagnostics


def loto_positive_count_bin(value: object) -> str:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric != math_floor(numeric):
        raise ValueError("other_positive_count must be a finite integer")
    count = int(numeric)
    if count == 0:
        return "ZERO"
    if count == 1:
        return "ONE"
    if 2 <= count <= 3:
        return "TWO_TO_THREE"
    if 4 <= count <= 6:
        return "FOUR_TO_SIX"
    if 7 <= count <= 11:
        return "SEVEN_TO_ELEVEN"
    raise ValueError(f"other_positive_count outside frozen 0..11 range: {count}")


def math_floor(value: float) -> int:
    # Kept local to avoid accepting numpy's silent non-finite casts.
    return int(np.floor(value))


def build_direction_group_frame(
    oof_frame: pd.DataFrame,
    feature_frame: pd.DataFrame,
    acquisition_profile_map: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate one target-direction OOF frame to frozen report-group rows."""

    oof_required = (
        ID_COLUMN,
        REPORT_GROUP_COLUMN,
        TARGET_COLUMN,
        DIRECTION_COLUMN,
        OUTER_FOLD_COLUMN,
        "observed",
        "context_oof",
        "context_plus_morphology_oof",
    )
    feature_required = (
        ID_COLUMN,
        TARGET_COLUMN,
        REPORT_GROUP_COLUMN,
        "other_positive_count",
    )
    _require_columns(oof_frame, oof_required, name="OOF direction input")
    _require_columns(feature_frame, feature_required, name="LOTO feature input")
    _require_columns(
        acquisition_profile_map,
        (REPORT_GROUP_COLUMN, PROFILE_COLUMN),
        name="acquisition profile map",
    )
    if oof_frame.empty:
        raise ValueError("OOF direction input is empty")
    if oof_frame[[TARGET_COLUMN, DIRECTION_COLUMN]].drop_duplicates().shape[0] != 1:
        raise ValueError("build_direction_group_frame requires one target-direction")
    if oof_frame.duplicated([ID_COLUMN, TARGET_COLUMN]).any():
        raise ValueError("OOF direction keys are not unique")

    safe_features = feature_frame[list(feature_required)].copy()
    if safe_features.duplicated([ID_COLUMN, TARGET_COLUMN]).any():
        raise ValueError("LOTO feature keys are not unique")
    merged = oof_frame[list(oof_required)].merge(
        safe_features,
        on=[ID_COLUMN, TARGET_COLUMN],
        how="left",
        validate="one_to_one",
        suffixes=("", "_feature"),
    )
    if merged["other_positive_count"].isna().any():
        raise ValueError("LOTO feature join is incomplete")
    if not merged[REPORT_GROUP_COLUMN].eq(
        merged[f"{REPORT_GROUP_COLUMN}_feature"]
    ).all():
        raise ValueError("OOF/feature report_group identity mismatch")
    merged = merged.drop(columns=f"{REPORT_GROUP_COLUMN}_feature")
    merged = merged.merge(
        acquisition_profile_map[[REPORT_GROUP_COLUMN, PROFILE_COLUMN]],
        on=REPORT_GROUP_COLUMN,
        how="left",
        validate="many_to_one",
    )
    if merged[PROFILE_COLUMN].isna().any():
        raise ValueError("acquisition profile join is incomplete")

    homogeneity_columns = [
        TARGET_COLUMN,
        DIRECTION_COLUMN,
        OUTER_FOLD_COLUMN,
        PROFILE_COLUMN,
    ]
    drift = merged.groupby(REPORT_GROUP_COLUMN, sort=True, observed=True)[
        homogeneity_columns
    ].nunique(dropna=False)
    if (drift > 1).any().any():
        raise ValueError("frozen stratum identity drifts within report_group")

    numeric = ["observed", "context_oof", "context_plus_morphology_oof"]
    means = merged.groupby(REPORT_GROUP_COLUMN, sort=True, observed=True)[
        numeric
    ].agg(_stable_mean)
    identities = merged.groupby(
        REPORT_GROUP_COLUMN, sort=True, observed=True
    )[homogeneity_columns].first()
    loto = (
        merged.groupby(REPORT_GROUP_COLUMN, sort=True, observed=True)[
            "other_positive_count"
        ]
        .agg(_smallest_integer_mode)
        .rename("other_positive_count_mode")
    )
    result = means.join(identities, validate="one_to_one").join(
        loto, validate="one_to_one"
    )
    result["contribution"] = (
        result["context_plus_morphology_oof"] - result["context_oof"]
    )
    result[LOTO_BIN_COLUMN] = result["other_positive_count_mode"].map(
        loto_positive_count_bin
    )
    return result.reset_index(drop=False)


def _json_stratum(level: str, values: Sequence[object]) -> str:
    return json.dumps(
        [level, *[str(value) for value in values]],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def assign_hierarchical_permutation_strata(
    group_frame: pd.DataFrame,
    *,
    minimum_stratum_groups: int = 20,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the frozen full -> no-profile -> outer-fold fallback.

    If any full profile cell is undersized, every group sharing its reduced
    outer-fold/LOTO cell is collapsed together.  If that collapsed cell remains
    undersized, its entire outer-fold cell is collapsed.  This avoids leaving a
    nominal fallback permutation cell below the frozen minimum.
    """

    if not isinstance(minimum_stratum_groups, (int, np.integer)) or int(
        minimum_stratum_groups
    ) < 1:
        raise ValueError("minimum_stratum_groups must be a positive integer")
    required = (
        REPORT_GROUP_COLUMN,
        TARGET_COLUMN,
        DIRECTION_COLUMN,
        OUTER_FOLD_COLUMN,
        PROFILE_COLUMN,
        LOTO_BIN_COLUMN,
    )
    _require_columns(group_frame, required, name="group stratum input")
    if group_frame.empty or group_frame[REPORT_GROUP_COLUMN].duplicated().any():
        raise ValueError("group stratum input must have unique report_group rows")
    if group_frame[[TARGET_COLUMN, DIRECTION_COLUMN]].drop_duplicates().shape[0] != 1:
        raise ValueError("group stratum input must contain one target-direction")
    result = group_frame.copy().sort_values(
        REPORT_GROUP_COLUMN, kind="mergesort"
    ).reset_index(drop=True)
    full_columns = [OUTER_FOLD_COLUMN, PROFILE_COLUMN, LOTO_BIN_COLUMN]
    reduced_columns = [OUTER_FOLD_COLUMN, LOTO_BIN_COLUMN]
    outer_columns = [OUTER_FOLD_COLUMN]

    full_sizes = result.groupby(full_columns, sort=True, observed=True)[
        REPORT_GROUP_COLUMN
    ].transform("size")
    small_reduced_keys = {
        tuple(row)
        for row in result.loc[full_sizes < int(minimum_stratum_groups), reduced_columns]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    reduced_key = [
        tuple(row)
        for row in result[reduced_columns].itertuples(index=False, name=None)
    ]
    level = np.asarray(
        ["NO_PROFILE" if key in small_reduced_keys else "FULL" for key in reduced_key],
        dtype=object,
    )
    provisional = [
        _json_stratum(
            str(level[index]),
            (
                result.loc[index, full_columns].tolist()
                if level[index] == "FULL"
                else result.loc[index, reduced_columns].tolist()
            ),
        )
        for index in range(len(result))
    ]
    provisional_sizes = pd.Series(provisional).groupby(provisional, sort=True).transform(
        "size"
    )
    small_outer_keys = {
        tuple(row)
        for row in result.loc[
            provisional_sizes.to_numpy() < int(minimum_stratum_groups), outer_columns
        ]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    final: list[str] = []
    final_level: list[str] = []
    for index in range(len(result)):
        outer_key = tuple(result.loc[index, outer_columns].tolist())
        if outer_key in small_outer_keys:
            final_level.append("OUTER_ONLY")
            final.append(_json_stratum("OUTER_ONLY", outer_key))
        elif level[index] == "NO_PROFILE":
            final_level.append("NO_PROFILE")
            final.append(_json_stratum("NO_PROFILE", reduced_key[index]))
        else:
            final_level.append("FULL")
            final.append(
                _json_stratum("FULL", result.loc[index, full_columns].tolist())
            )
    result["permutation_stratum_level"] = final_level
    result[PERMUTATION_STRATUM_COLUMN] = final
    final_sizes = result.groupby(PERMUTATION_STRATUM_COLUMN, sort=True)[
        REPORT_GROUP_COLUMN
    ].transform("size")
    if (final_sizes < int(minimum_stratum_groups)).any():
        minimum = int(final_sizes.min())
        raise ValueError(f"hierarchical stratum remains undersized: minimum={minimum}")
    diagnostics = {
        "status": "PASS",
        "report_groups": int(len(result)),
        "strata": int(result[PERMUTATION_STRATUM_COLUMN].nunique()),
        "minimum_stratum_groups": int(final_sizes.min()),
        "maximum_stratum_groups": int(final_sizes.max()),
        "level_group_counts": {
            str(key): int(value)
            for key, value in result["permutation_stratum_level"]
            .value_counts()
            .sort_index()
            .items()
        },
    }
    return result, diagnostics


def component_and_high_agreement_robustness(
    oof_direction: pd.DataFrame,
) -> dict[str, int | float | bool | str]:
    """Re-score identical predictions against frozen component endpoints."""

    required = (
        REPORT_GROUP_COLUMN,
        TARGET_COLUMN,
        DIRECTION_COLUMN,
        "observed",
        "context_oof",
        "context_plus_morphology_oof",
        "dino_target_ecdf",
        "rad_target_ecdf",
        "image_high_agreement",
    )
    _require_columns(oof_direction, required, name="robustness input")
    identity = oof_direction[[TARGET_COLUMN, DIRECTION_COLUMN]].drop_duplicates()
    if len(identity) != 1:
        raise ValueError("robustness input must contain one target-direction")
    direction = str(identity.iloc[0][DIRECTION_COLUMN])
    if direction not in {"omission", "contradiction"}:
        raise ValueError(f"unexpected direction: {direction}")
    result: dict[str, int | float | bool | str] = {
        TARGET_COLUMN: str(identity.iloc[0][TARGET_COLUMN]),
        DIRECTION_COLUMN: direction,
        "rows": int(len(oof_direction)),
        "report_groups": int(oof_direction[REPORT_GROUP_COLUMN].nunique()),
    }
    working = oof_direction.copy()
    for label, source in (
        ("primary", "observed"),
        ("dino_only", "dino_target_ecdf"),
        ("rad_only", "rad_target_ecdf"),
    ):
        raw = _finite_numeric(working[source], name=source)
        working[f"_{label}_endpoint"] = (
            raw
            if label == "primary" or direction == "omission"
            else 1.0 - raw
        )
        metrics = compare_report_group_weighted_predictions(
            working,
            outcome_column=f"_{label}_endpoint",
            context_prediction_column="context_oof",
            full_prediction_column="context_plus_morphology_oof",
            report_group_column=REPORT_GROUP_COLUMN,
        )
        result[f"{label}_context_mse"] = float(
            metrics["context_report_group_weighted_mse"]
        )
        result[f"{label}_full_mse"] = float(
            metrics["full_report_group_weighted_mse"]
        )
        result[f"{label}_mse_increment"] = float(metrics["mse_increment"])
        result[f"{label}_relative_mse_improvement"] = float(
            metrics["relative_mse_improvement"]
        )

    high = working.loc[working["image_high_agreement"].astype(bool)].copy()
    if high.empty:
        raise ValueError("high-agreement robustness subset is empty")
    high_metrics = compare_report_group_weighted_predictions(
        high,
        outcome_column="_primary_endpoint",
        context_prediction_column="context_oof",
        full_prediction_column="context_plus_morphology_oof",
        report_group_column=REPORT_GROUP_COLUMN,
    )
    result["high_agreement_rows"] = int(len(high))
    result["high_agreement_report_groups"] = int(
        high[REPORT_GROUP_COLUMN].nunique()
    )
    result["high_agreement_mse_increment"] = float(high_metrics["mse_increment"])
    result["high_agreement_sign"] = bool(
        float(high_metrics["mse_increment"]) > 0.0
    )
    return result


def strict_proxy_review_metrics(
    oof_direction: pd.DataFrame,
    *,
    review_fractions: Sequence[float] = (0.05, 0.10, 0.20),
) -> pd.DataFrame:
    """Evaluate supplied OOF risk against the frozen strict-10% proxy event."""

    required = (
        ID_COLUMN,
        TARGET_COLUMN,
        DIRECTION_COLUMN,
        "observed",
        "image_high_agreement",
        "risk_percentile",
    )
    _require_columns(oof_direction, required, name="review input")
    identity = oof_direction[[TARGET_COLUMN, DIRECTION_COLUMN]].drop_duplicates()
    if len(identity) != 1:
        raise ValueError("review input must contain one target-direction")
    working = oof_direction.copy()
    observed = _finite_numeric(working["observed"], name="observed")
    if np.any((observed < 0.0) | (observed > 1.0)):
        raise ValueError("observed directional endpoint must lie within [0, 1]")
    working["strict_proxy_event"] = (
        (observed >= 0.90 - 1e-12)
        & working["image_high_agreement"].astype(bool).to_numpy()
    ).astype(np.int8)
    metrics = review_fraction_capture_enrichment(
        working,
        score_column="risk_percentile",
        event_column="strict_proxy_event",
        row_key_columns=(ID_COLUMN, TARGET_COLUMN),
        review_fractions=review_fractions,
    )
    metrics.insert(0, DIRECTION_COLUMN, str(identity.iloc[0][DIRECTION_COLUMN]))
    metrics.insert(0, TARGET_COLUMN, str(identity.iloc[0][TARGET_COLUMN]))
    metrics["event_definition"] = (
        "observed_directional_endpoint>=0.90_and_image_family_disagreement<=0.10"
    )
    metrics["score_definition"] = "outer_oof_full_morphology_risk_percentile"
    return metrics


__all__ = [
    "ACQUISITION_COLUMNS",
    "DIRECTION_COLUMN",
    "LOTO_BIN_COLUMN",
    "OUTER_FOLD_COLUMN",
    "PERMUTATION_STRATUM_COLUMN",
    "PROFILE_COLUMN",
    "REPORT_GROUP_COLUMN",
    "TARGET_COLUMN",
    "assign_hierarchical_permutation_strata",
    "build_direction_group_frame",
    "build_pooled_acquisition_profile_map",
    "component_and_high_agreement_robustness",
    "loto_positive_count_bin",
    "strict_proxy_review_metrics",
]
