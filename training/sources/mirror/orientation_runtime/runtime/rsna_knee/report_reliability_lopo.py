"""Deterministic surviving-only leave-one-acquisition-profile-out helpers.

This module implements the contract-v3.4 LOPO mechanics without reading project
artifacts.  A held-out acquisition profile is excluded from every model,
family-ECDF, preprocessing, and fold-fitted orthographic reference.  The
resulting report-group residual effects are directional robustness checks only;
they do not confirm report error and cannot change labels or weights.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from typing import Any, Literal

import numpy as np
import pandas as pd

from .constants import ID_COLUMN
from .foldfit_orthographic_features import (
    FEATURE_COLUMNS as FOLDFIT_FEATURE_COLUMNS,
    FoldFittedOrthographicAnomalyTransformer,
)
from .report_reliability_segments import HYPOTHESIS_ID_COLUMN, signed_group_effect
from .report_reliability_stats import (
    FoldFittedTransformerProvider,
    _fit_predict_ridge,
    fit_family_balanced_image_reference,
    inverse_report_group_weights,
    select_final_directional_ridge_alpha,
)


Direction = Literal["omission", "contradiction"]

PROFILE_COLUMN = "acquisition_profile"
PROFILE_SOURCE_COLUMNS: tuple[str, ...] = (
    "acquisition_manufacturer_mode",
    "acquisition_model_mode",
    "acquisition_field_strength_mode",
)
MINIMUM_PROFILE_REPORT_GROUPS = 100
MINIMUM_EFFECT_REPORT_GROUPS_PER_ARM = 100

SURVIVOR_REQUIRED_COLUMNS: tuple[str, ...] = (
    "target",
    "direction",
    HYPOTHESIS_ID_COLUMN,
    "adjustment_key",
    "adjustment_excluded_fields_json",
    "segment_effect",
    "core_maxT_survivor",
    "dino_segment_effect",
    "rad_segment_effect",
    "dino_rad_same_sign",
    "high_agreement_support_eligible",
    "high_agreement_segment_effect",
    "high_agreement_sign_retained",
    "common_support_profile_count",
    "common_support_eligible",
    "common_support_segment_effect",
    "common_support_sign_retained",
)
PROFILE_EFFECT_COLUMNS: tuple[str, ...] = (
    "target",
    "direction",
    HYPOTHESIS_ID_COLUMN,
    "adjustment_key",
    PROFILE_COLUMN,
    "profile_total_report_groups",
    "fit_status",
    "selected_alpha",
    "train_rows",
    "train_report_groups",
    "heldout_rows",
    "heldout_report_groups",
    "segment_report_groups",
    "complement_report_groups",
    "support_eligible",
    "primary_segment_effect",
    "lopo_segment_effect",
    "lopo_sign_matches_primary",
)
SURVIVOR_SUMMARY_COLUMNS: tuple[str, ...] = (
    "target",
    "direction",
    HYPOTHESIS_ID_COLUMN,
    "adjustment_key",
    "primary_segment_effect",
    "lopo_profiles_required",
    "lopo_profiles_completed",
    "lopo_profiles_supported",
    "lopo_all_effects_same_sign",
    "precomputed_dino_rad_same_sign",
    "precomputed_high_agreement_support_eligible",
    "precomputed_high_agreement_sign_retained",
    "precomputed_common_support_eligible",
    "precomputed_common_support_sign_retained",
    "precomputed_robustness_guard_pass",
    "lopo_status",
    "proxy_segment_status",
)


@dataclass(frozen=True)
class LopoProfileFit:
    """Training-only alpha trace and held-out profile row predictions."""

    scores: pd.DataFrame
    alpha_trace: pd.DataFrame
    diagnostics: Mapping[str, Any]


def _require(frame: pd.DataFrame, columns: Sequence[str], *, name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def _assert_no_gold_columns(frame: pd.DataFrame, *, name: str) -> None:
    forbidden = [column for column in frame.columns if "gold" in str(column).casefold()]
    if forbidden:
        raise ValueError(f"{name} contains forbidden Gold columns: {forbidden}")


def _nonempty_strings(values: pd.Series, *, name: str) -> pd.Series:
    if values.isna().any():
        raise ValueError(f"{name} contains nulls")
    result = values.astype(str).str.strip()
    if result.eq("").any():
        raise ValueError(f"{name} contains empty strings")
    return result


def _strict_boolean(values: pd.Series, *, name: str) -> pd.Series:
    if values.isna().any() or not values.isin([True, False, 0, 1]).all():
        raise ValueError(f"{name} must contain only non-null booleans")
    return values.astype(bool)


def _profile_component(value: object) -> str:
    if value is None or pd.isna(value):
        return "__MISSING__"
    result = str(value).strip()
    return result if result else "__MISSING__"


def _lexicographic_mode(values: pd.Series) -> str:
    normalized = _nonempty_strings(values, name="profile mode values")
    counts = normalized.value_counts(sort=False)
    maximum = int(counts.max())
    return sorted(str(value) for value in counts.index[counts.eq(maximum)])[0]


def _stable_mean(values: pd.Series) -> float:
    numeric = np.sort(pd.to_numeric(values, errors="raise").to_numpy(float))
    if not np.isfinite(numeric).all():
        raise ValueError("LOPO residual aggregation contains non-finite values")
    return float(np.mean(numeric, dtype=np.float64))


def build_report_group_acquisition_profiles(
    frame: pd.DataFrame,
    *,
    group_column: str = "report_group",
    key_column: str = ID_COLUMN,
    profile_source_columns: Sequence[str] = PROFILE_SOURCE_COLUMNS,
) -> pd.DataFrame:
    """Build the frozen unpooled report-group acquisition-profile map.

    Study-level components are concatenated after ``__MISSING__`` substitution.
    When one report group contains multiple study profiles, the lexicographic
    frequency mode is used, matching the contract's report-group aggregation.
    """

    columns = tuple(str(column) for column in profile_source_columns)
    if len(columns) != 3 or len(set(columns)) != 3:
        raise ValueError("profile_source_columns must contain three unique fields")
    _require(frame, [key_column, group_column, *columns], name="profile source")
    _assert_no_gold_columns(frame, name="profile source")
    source = frame[[key_column, group_column, *columns]].drop_duplicates().copy()
    source[key_column] = _nonempty_strings(source[key_column], name=key_column)
    source[group_column] = _nonempty_strings(source[group_column], name=group_column)
    if source.duplicated(key_column).any():
        raise ValueError("one study maps to multiple acquisition profiles or report groups")
    source[PROFILE_COLUMN] = source.apply(
        lambda row: "|".join(_profile_component(row[column]) for column in columns),
        axis=1,
    )
    grouped = (
        source.groupby(group_column, sort=True, observed=True)
        .agg(
            **{
                PROFILE_COLUMN: (PROFILE_COLUMN, _lexicographic_mode),
                "profile_group_studies": (key_column, "size"),
            }
        )
        .reset_index()
    )
    if grouped[group_column].duplicated().any():
        raise AssertionError("report-group profile map is not unique")
    return grouped.sort_values(group_column, kind="mergesort").reset_index(drop=True)


def required_acquisition_profiles(
    report_group_profiles: pd.DataFrame,
    *,
    minimum_report_groups: int = MINIMUM_PROFILE_REPORT_GROUPS,
    group_column: str = "report_group",
) -> pd.DataFrame:
    """Return every frozen profile represented by at least 100 report groups."""

    if int(minimum_report_groups) != MINIMUM_PROFILE_REPORT_GROUPS:
        raise ValueError("minimum_report_groups is frozen at 100")
    _require(
        report_group_profiles,
        [group_column, PROFILE_COLUMN],
        name="report-group profiles",
    )
    if report_group_profiles[group_column].duplicated().any():
        raise ValueError("report-group profiles contain duplicate groups")
    counts = (
        report_group_profiles.groupby(PROFILE_COLUMN, sort=True, observed=True)[
            group_column
        ]
        .nunique()
        .rename("profile_total_report_groups")
        .reset_index()
    )
    result = counts.loc[
        counts["profile_total_report_groups"].ge(MINIMUM_PROFILE_REPORT_GROUPS)
    ].copy()
    return result.sort_values(PROFILE_COLUMN, kind="mergesort").reset_index(drop=True)


def validate_survivor_inputs(
    effects: pd.DataFrame,
    shortlist: pd.DataFrame,
    metadata: pd.DataFrame,
    cache_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Validate the effects/shortlist/universe/cache survivor identity chain."""

    for frame, name in (
        (effects, "segment effects"),
        (shortlist, "segment shortlist"),
        (metadata, "segment metadata"),
        (cache_metadata, "adjustment cache metadata"),
    ):
        _assert_no_gold_columns(frame, name=name)
    _require(effects, SURVIVOR_REQUIRED_COLUMNS, name="segment effects")
    _require(shortlist, SURVIVOR_REQUIRED_COLUMNS, name="segment shortlist")
    metadata_columns = (
        HYPOTHESIS_ID_COLUMN,
        "adjustment_key",
        "adjustment_excluded_fields_json",
    )
    _require(metadata, metadata_columns, name="segment metadata")
    _require(
        cache_metadata,
        ("adjustment_key", "adjustment_excluded_fields_json"),
        name="adjustment cache metadata",
    )
    key = ["target", "direction", HYPOTHESIS_ID_COLUMN]
    if effects.duplicated(key).any() or shortlist.duplicated(key).any():
        raise ValueError("segment effects/shortlist keys are duplicated")
    survivor_mask = _strict_boolean(
        effects["core_maxT_survivor"], name="core_maxT_survivor"
    )
    survivors = effects.loc[survivor_mask, list(SURVIVOR_REQUIRED_COLUMNS)].copy()
    shortlist_survivor = _strict_boolean(
        shortlist["core_maxT_survivor"], name="shortlist core_maxT_survivor"
    )
    if not shortlist_survivor.all():
        raise ValueError("segment shortlist contains a non-survivor")
    left = survivors.sort_values(key, kind="mergesort").reset_index(drop=True)
    right = shortlist.loc[:, list(SURVIVOR_REQUIRED_COLUMNS)].sort_values(
        key, kind="mergesort"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=True)

    meta = metadata.loc[:, list(metadata_columns)].copy()
    if meta[HYPOTHESIS_ID_COLUMN].duplicated().any():
        raise ValueError("segment metadata hypothesis IDs are duplicated")
    cache = cache_metadata.loc[
        :, ["adjustment_key", "adjustment_excluded_fields_json"]
    ].drop_duplicates()
    if cache["adjustment_key"].duplicated().any():
        raise ValueError("adjustment cache key maps to multiple exclusions")
    universe_mapping = meta[
        ["adjustment_key", "adjustment_excluded_fields_json"]
    ].drop_duplicates()
    if universe_mapping["adjustment_key"].duplicated().any():
        raise ValueError("segment metadata adjustment key maps to multiple exclusions")
    pd.testing.assert_frame_equal(
        universe_mapping.sort_values("adjustment_key", kind="mergesort").reset_index(
            drop=True
        ),
        cache.sort_values("adjustment_key", kind="mergesort").reset_index(drop=True),
        check_dtype=False,
        check_exact=True,
    )
    joined = survivors.merge(
        meta,
        on=HYPOTHESIS_ID_COLUMN,
        how="left",
        suffixes=("_effect", "_metadata"),
        validate="many_to_one",
    )
    if len(joined) != len(survivors):
        raise AssertionError("survivor metadata join changed row count")
    for column in ("adjustment_key", "adjustment_excluded_fields_json"):
        if not joined[f"{column}_effect"].astype(str).equals(
            joined[f"{column}_metadata"].astype(str)
        ):
            raise ValueError(f"survivor {column} differs from frozen metadata")
    numeric_effect = pd.to_numeric(survivors["segment_effect"], errors="coerce")
    if len(survivors) and (
        numeric_effect.isna().any()
        or not np.isfinite(numeric_effect).all()
        or numeric_effect.eq(0.0).any()
    ):
        raise ValueError("core survivors require finite nonzero primary segment effects")
    normalized_guards: dict[str, pd.Series] = {}
    for column in (
        "dino_rad_same_sign",
        "high_agreement_support_eligible",
        "high_agreement_sign_retained",
        "common_support_eligible",
        "common_support_sign_retained",
    ):
        normalized_guards[column] = _strict_boolean(survivors[column], name=column)
    common_profiles = pd.to_numeric(
        survivors["common_support_profile_count"], errors="coerce"
    )
    if len(survivors) and (
        common_profiles.isna().any()
        or not np.isfinite(common_profiles).all()
        or (common_profiles < 0).any()
        or not np.array_equal(
            common_profiles.to_numpy(float),
            common_profiles.to_numpy(dtype=np.int64),
        )
    ):
        raise ValueError("common_support_profile_count must be a nonnegative integer")
    common_available = common_profiles.gt(0).reset_index(drop=True)
    common_eligible = normalized_guards["common_support_eligible"].reset_index(drop=True)
    if not common_available.equals(common_eligible):
        raise ValueError(
            "common_support_profile_count > 0 must be equivalent to "
            "common_support_eligible"
        )
    high_support = normalized_guards["high_agreement_support_eligible"]
    high_sign = normalized_guards["high_agreement_sign_retained"]
    if (high_sign & ~high_support).any():
        raise ValueError("high-agreement sign retention requires eligible support")
    common_sign = normalized_guards["common_support_sign_retained"]
    if (common_sign & ~normalized_guards["common_support_eligible"]).any():
        raise ValueError("common-support sign retention requires eligible support")
    return survivors.sort_values(key, kind="mergesort").reset_index(drop=True)


def _merge_generated_features(
    model_frame: pd.DataFrame,
    generated: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    key_column: str,
) -> pd.DataFrame:
    columns = tuple(str(column) for column in feature_columns)
    _require(generated, [key_column, *columns], name="generated split features")
    if generated[key_column].isna().any() or generated[key_column].duplicated().any():
        raise ValueError("generated split-feature keys are null or duplicated")
    working = model_frame.drop(columns=[c for c in columns if c in model_frame], errors="ignore")
    merged = working.merge(
        generated[[key_column, *columns]],
        on=key_column,
        how="left",
        validate="many_to_one",
    )
    if merged[list(columns)].isna().any().any():
        raise ValueError("generated split-feature coverage is incomplete")
    merged.index = model_frame.index
    return merged


def fit_score_lopo_profile(
    train_target_reference: pd.DataFrame,
    heldout_direction_frame: pd.DataFrame,
    *,
    training_report_reference: pd.DataFrame,
    heldout_report_reference: pd.DataFrame,
    target: str,
    direction: Direction,
    dino_column: str,
    rad_component_columns: Sequence[str],
    context_columns: Sequence[str],
    morphology_columns: Sequence[str],
    split_feature_columns: Sequence[str] = FOLDFIT_FEATURE_COLUMNS,
    transformer_factory: Callable[[], Any] = FoldFittedOrthographicAnomalyTransformer,
    target_column: str = "target",
    teacher_label_column: str = "teacher_hard_label",
    group_column: str = "report_group",
    key_column: str = ID_COLUMN,
) -> LopoProfileFit:
    """Fit on non-profile rows and score one held-out profile exactly once."""

    if direction not in ("omission", "contradiction"):
        raise ValueError("direction must be omission or contradiction")
    for frame, name in (
        (train_target_reference, "LOPO training target reference"),
        (heldout_direction_frame, "LOPO held-out direction frame"),
        (training_report_reference, "LOPO training reports"),
        (heldout_report_reference, "LOPO held-out reports"),
    ):
        _assert_no_gold_columns(frame, name=name)
    rad_columns = tuple(str(column) for column in rad_component_columns)
    features = (*tuple(context_columns), *tuple(morphology_columns))
    declared_dynamic = tuple(str(column) for column in split_feature_columns)
    if len(set(declared_dynamic)) != len(declared_dynamic):
        raise ValueError("split_feature_columns contains duplicates")
    # A segment-adjustment exclusion may remove one or both orthographic
    # predictors.  Never materialize an excluded feature back into the fitted
    # model: the split-local provider is active only for declared predictors
    # that survive this hypothesis' exact exclusion set.
    dynamic = tuple(column for column in declared_dynamic if column in features)
    required = [
        key_column,
        target_column,
        teacher_label_column,
        group_column,
        dino_column,
        *rad_columns,
    ]
    _require(train_target_reference, required, name="LOPO training target reference")
    _require(heldout_direction_frame, required, name="LOPO held-out direction frame")
    if train_target_reference.empty or heldout_direction_frame.empty:
        raise ValueError("LOPO training and held-out model frames must be non-empty")
    train_groups = set(
        _nonempty_strings(train_target_reference[group_column], name="training groups")
    )
    heldout_groups = set(
        _nonempty_strings(heldout_direction_frame[group_column], name="held-out groups")
    )
    if train_groups.intersection(heldout_groups):
        raise ValueError("held-out acquisition-profile groups entered model training")
    report_train_groups = set(
        _nonempty_strings(training_report_reference[group_column], name="training report groups")
    )
    report_heldout_groups = set(
        _nonempty_strings(heldout_report_reference[group_column], name="held-out report groups")
    )
    if report_train_groups.intersection(report_heldout_groups):
        raise ValueError("held-out acquisition-profile groups entered typo training")
    if not train_groups.issubset(report_train_groups):
        raise ValueError("training report reference omits model-training groups")
    if not heldout_groups.issubset(report_heldout_groups):
        raise ValueError("held-out report reference omits scored groups")
    if set(train_target_reference[target_column].astype(str)) != {str(target)}:
        raise ValueError("LOPO training target reference contains another target")
    if set(heldout_direction_frame[target_column].astype(str)) != {str(target)}:
        raise ValueError("LOPO held-out frame contains another target")
    labels = pd.to_numeric(train_target_reference[teacher_label_column], errors="coerce")
    if labels.isna().any() or not labels.isin([0.0, 1.0]).all():
        raise ValueError("LOPO training target reference must be decisive")
    required_label = 0.0 if direction == "omission" else 1.0
    heldout_labels = pd.to_numeric(
        heldout_direction_frame[teacher_label_column], errors="coerce"
    )
    if heldout_labels.isna().any() or not heldout_labels.eq(required_label).all():
        raise ValueError("LOPO held-out frame contains the wrong teacher polarity")
    training_direction = train_target_reference.loc[labels.eq(required_label)].copy()
    if training_direction.empty:
        raise ValueError("LOPO training direction has no rows")

    provider = (
        FoldFittedTransformerProvider(
            reference_frame=training_report_reference.copy(),
            transformer_factory=transformer_factory,
            feature_columns=dynamic,
            group_column=group_column,
            key_column=key_column,
        )
        if dynamic
        else None
    )
    selection = select_final_directional_ridge_alpha(
        training_direction,
        ecdf_reference_frame=train_target_reference,
        direction=direction,
        dino_column=dino_column,
        rad_component_columns=rad_columns,
        context_columns=context_columns,
        morphology_columns=morphology_columns,
        target_column=target_column,
        teacher_label_column=teacher_label_column,
        group_column=group_column,
        split_feature_provider=provider,
        split_feature_columns=dynamic,
        split_feature_key_column=key_column,
    )

    transformer = None
    training_direction_augmented = training_direction.copy()
    heldout_augmented = heldout_direction_frame.copy()
    if dynamic:
        transformer = transformer_factory()
        train_generated = transformer.fit_transform(training_report_reference.copy())
        heldout_generated = transformer.transform(heldout_report_reference.copy())
        training_direction_augmented = _merge_generated_features(
            training_direction,
            train_generated,
            feature_columns=dynamic,
            key_column=key_column,
        )
        heldout_augmented = _merge_generated_features(
            heldout_direction_frame,
            heldout_generated,
            feature_columns=dynamic,
            key_column=key_column,
        )
    image_reference = fit_family_balanced_image_reference(
        train_target_reference,
        dino_column=dino_column,
        rad_component_columns=rad_columns,
        target_column=target_column,
    )
    train_evidence = image_reference.transform(training_direction_augmented)
    heldout_evidence = image_reference.transform(heldout_augmented)
    train_outcome = train_evidence["image_consensus_rank"].to_numpy(float)
    heldout_outcome = heldout_evidence["image_consensus_rank"].to_numpy(float)
    if direction == "contradiction":
        train_outcome = 1.0 - train_outcome
        heldout_outcome = 1.0 - heldout_outcome
    weights = inverse_report_group_weights(training_direction_augmented[group_column])
    prediction = _fit_predict_ridge(
        training_direction_augmented,
        heldout_augmented,
        columns=features,
        train_outcome=train_outcome,
        train_weights=weights,
        alpha=selection.selected_alpha,
    )
    scores = heldout_direction_frame[[key_column, group_column]].copy()
    scores.insert(1, "target", str(target))
    scores.insert(2, "direction", str(direction))
    scores["observed"] = heldout_outcome
    scores["prediction"] = prediction
    scores["residual"] = heldout_outcome - prediction
    if not np.isfinite(scores[["observed", "prediction", "residual"]].to_numpy()).all():
        raise AssertionError("LOPO scoring produced non-finite values")
    trace = selection.trace.copy()
    trace.insert(0, "direction", str(direction))
    trace.insert(0, "target", str(target))
    diagnostics = {
        "selected_alpha": float(selection.selected_alpha),
        "train_rows": int(len(training_direction)),
        "train_report_groups": int(training_direction[group_column].nunique()),
        "heldout_rows": int(len(heldout_direction_frame)),
        "heldout_report_groups": int(heldout_direction_frame[group_column].nunique()),
        "training_reference_rows": int(len(train_target_reference)),
        "training_reference_report_groups": int(
            train_target_reference[group_column].nunique()
        ),
        "typo_training_report_groups": int(
            training_report_reference[group_column].nunique()
        ),
        "typo_heldout_report_groups": int(
            heldout_report_reference[group_column].nunique()
        ),
        "active_split_feature_columns": list(dynamic),
        "typo_reference_sha256": getattr(transformer, "reference_sha256_", None),
    }
    return LopoProfileFit(
        scores=scores.reset_index(drop=True),
        alpha_trace=trace.reset_index(drop=True),
        diagnostics=diagnostics,
    )


def profile_segment_effect(
    scores: pd.DataFrame,
    membership: pd.Series,
    *,
    primary_segment_effect: float,
    group_column: str = "report_group",
    minimum_report_groups_per_arm: int = MINIMUM_EFFECT_REPORT_GROUPS_PER_ARM,
) -> dict[str, int | float | bool]:
    """Aggregate held-out residuals by report group and test frozen sign retention."""

    if int(minimum_report_groups_per_arm) != MINIMUM_EFFECT_REPORT_GROUPS_PER_ARM:
        raise ValueError("minimum_report_groups_per_arm is frozen at 100")
    _require(scores, [group_column, "residual"], name="LOPO scores")
    if scores.empty:
        return {
            "segment_report_groups": 0,
            "complement_report_groups": 0,
            "support_eligible": False,
            "effect": np.nan,
            "sign_matches_primary": False,
        }
    if scores[group_column].isna().any():
        raise ValueError("LOPO scores contain null report groups")
    residual = (
        scores.groupby(group_column, sort=True, observed=True)["residual"]
        .agg(_stable_mean)
        .sort_index(kind="mergesort")
    )
    if membership.index.has_duplicates or membership.isna().any():
        raise ValueError("LOPO membership index is duplicated or null")
    selected = membership.reindex(residual.index)
    if selected.isna().any():
        raise ValueError("LOPO membership does not cover held-out report groups")
    effect = signed_group_effect(
        residual,
        selected.astype(bool),
        minimum_report_groups_per_arm=MINIMUM_EFFECT_REPORT_GROUPS_PER_ARM,
    )
    primary = float(primary_segment_effect)
    if not np.isfinite(primary) or primary == 0.0:
        raise ValueError("primary_segment_effect must be finite and nonzero")
    value = float(effect["effect"])
    same = bool(effect["support_eligible"] and np.isfinite(value) and primary * value > 0)
    return {**effect, "sign_matches_primary": same}


def summarize_survivor_lopo(
    survivors: pd.DataFrame,
    profile_effects: pd.DataFrame,
    *,
    required_profile_count: int,
) -> pd.DataFrame:
    """Apply the frozen all-supported-profile sign gate to every survivor."""

    _require(survivors, SURVIVOR_REQUIRED_COLUMNS, name="LOPO survivors")
    _require(profile_effects, PROFILE_EFFECT_COLUMNS, name="LOPO profile effects")
    if int(required_profile_count) < 0:
        raise ValueError("required_profile_count must be nonnegative")
    key = ["target", "direction", HYPOTHESIS_ID_COLUMN]
    if survivors.duplicated(key).any() or profile_effects.duplicated(
        [*key, PROFILE_COLUMN]
    ).any():
        raise ValueError("LOPO survivor/profile-effect keys are duplicated")
    rows: list[dict[str, Any]] = []
    for survivor in survivors.itertuples(index=False):
        identity = {
            "target": str(survivor.target),
            "direction": str(survivor.direction),
            HYPOTHESIS_ID_COLUMN: str(getattr(survivor, HYPOTHESIS_ID_COLUMN)),
        }
        part = profile_effects.copy()
        for column, value in identity.items():
            part = part.loc[part[column].astype(str).eq(value)]
        if len(part) != int(required_profile_count):
            raise ValueError("LOPO profile coverage differs from the required profile set")
        if not part["adjustment_key"].astype(str).eq(str(survivor.adjustment_key)).all():
            raise ValueError("LOPO profile effect adjustment key differs from survivor")
        primary_values = pd.to_numeric(
            part["primary_segment_effect"], errors="coerce"
        ).to_numpy(float)
        if not np.isfinite(primary_values).all() or not np.array_equal(
            primary_values,
            np.full(len(part), float(survivor.segment_effect), dtype=np.float64),
        ):
            raise ValueError("LOPO profile effects differ from the frozen primary effect")
        completed = int(part["fit_status"].eq("PASS_SCORED").sum())
        support = _strict_boolean(part["support_eligible"], name="support_eligible")
        supported = part.loc[support]
        sign = _strict_boolean(
            supported["lopo_sign_matches_primary"],
            name="lopo_sign_matches_primary",
        )
        required = int(required_profile_count)
        complete_all = bool(required > 0 and completed == required)
        support_all = bool(required > 0 and len(supported) == required)
        all_same = bool(support_all and sign.all())
        def strict_guard(column: str) -> bool:
            return bool(
                _strict_boolean(
                    pd.Series([getattr(survivor, column)]), name=column
                ).iloc[0]
            )

        dino_rad_same = strict_guard("dino_rad_same_sign")
        high_support = strict_guard("high_agreement_support_eligible")
        high_sign = strict_guard("high_agreement_sign_retained")
        common_eligible = strict_guard("common_support_eligible")
        common_sign = strict_guard("common_support_sign_retained")
        common_profile_count = float(survivor.common_support_profile_count)
        if (
            not np.isfinite(common_profile_count)
            or common_profile_count < 0
            or common_profile_count != int(common_profile_count)
        ):
            raise ValueError(
                "common_support_profile_count must be a nonnegative integer"
            )
        common_support = common_profile_count > 0
        if common_eligible != common_support:
            raise ValueError(
                "common_support_profile_count > 0 must be equivalent to "
                "common_support_eligible"
            )
        if high_sign and not high_support:
            raise ValueError("high-agreement sign retention requires eligible support")
        if common_sign and not common_support:
            raise ValueError("common-support sign retention requires eligible support")
        guard_pass = bool(
            dino_rad_same
            and high_support
            and high_sign
            and common_support
            and common_sign
        )
        if not complete_all:
            status = "BLOCKED_INCOMPLETE_REQUIRED_PROFILE_SCORING"
            proxy_status = "DIRECTIONAL_ONLY_LOPO_UNRESOLVED"
        elif not support_all:
            status = "DEMOTED_UNSUPPORTED_REQUIRED_PROFILE"
            proxy_status = "DIRECTIONAL_ONLY_LOPO_DEMOTED"
        elif not all_same:
            status = "DEMOTED_REQUIRED_PROFILE_SIGN_REVERSAL"
            proxy_status = "DIRECTIONAL_ONLY_LOPO_DEMOTED"
        elif not guard_pass:
            status = "DEMOTED_PRECOMPUTED_ROBUSTNESS_GUARD_FAILED"
            proxy_status = "DIRECTIONAL_ONLY_LOPO_DEMOTED"
        else:
            status = "PASS_ALL_REQUIRED_PROFILE_AND_PRECOMPUTED_GUARDS"
            proxy_status = "CORE_SURVIVOR_LOPO_PASS"
        rows.append(
            {
                **identity,
                "adjustment_key": str(survivor.adjustment_key),
                "primary_segment_effect": float(survivor.segment_effect),
                "lopo_profiles_required": required,
                "lopo_profiles_completed": completed,
                "lopo_profiles_supported": int(len(supported)),
                "lopo_all_effects_same_sign": all_same,
                "precomputed_dino_rad_same_sign": dino_rad_same,
                "precomputed_high_agreement_support_eligible": high_support,
                "precomputed_high_agreement_sign_retained": high_sign,
                "precomputed_common_support_eligible": common_support,
                "precomputed_common_support_sign_retained": common_sign,
                "precomputed_robustness_guard_pass": guard_pass,
                "lopo_status": status,
                "proxy_segment_status": proxy_status,
            }
        )
    return pd.DataFrame(rows, columns=list(SURVIVOR_SUMMARY_COLUMNS)).sort_values(
        key, kind="mergesort"
    ).reset_index(drop=True)


def empty_profile_effects() -> pd.DataFrame:
    return pd.DataFrame(columns=list(PROFILE_EFFECT_COLUMNS))


def empty_survivor_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=list(SURVIVOR_SUMMARY_COLUMNS))


__all__ = [
    "LopoProfileFit",
    "MINIMUM_EFFECT_REPORT_GROUPS_PER_ARM",
    "MINIMUM_PROFILE_REPORT_GROUPS",
    "PROFILE_COLUMN",
    "PROFILE_EFFECT_COLUMNS",
    "PROFILE_SOURCE_COLUMNS",
    "SURVIVOR_REQUIRED_COLUMNS",
    "SURVIVOR_SUMMARY_COLUMNS",
    "build_report_group_acquisition_profiles",
    "empty_profile_effects",
    "empty_survivor_summary",
    "fit_score_lopo_profile",
    "profile_segment_effect",
    "required_acquisition_profiles",
    "summarize_survivor_lopo",
    "validate_survivor_inputs",
]
