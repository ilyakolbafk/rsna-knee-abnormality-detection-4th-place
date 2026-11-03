"""Deterministic descriptive statistics for report-morphology error analysis.

The helpers in this module implement the frozen statistical mechanics in the
``rsna-knee-morphology-error-risk-v1`` analysis contract.  They deliberately do
not choose feature thresholds, mutate labels, or turn model/label disagreement
into an observed-error claim.  All effects are segment-minus-complement
descriptive associations.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .constants import ID_COLUMN


BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 20_260_830
BOOTSTRAP_ALPHA = 0.05
MINIMUM_STUDIES = 100
SPLIT_HALF_SALT = "rsna-knee-morphology-error-risk-v1|split-half"


def _as_one_dimensional(values: Any, *, name: str) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return result


def _validated_binary(values: Any, *, name: str) -> np.ndarray:
    raw = _as_one_dimensional(values, name=name)
    if pd.isna(raw).any():
        raise ValueError(f"{name} contains null values")
    if raw.dtype == np.bool_:
        return raw.astype(bool, copy=False)
    try:
        numeric = raw.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain only boolean or 0/1 values") from error
    if not np.isin(numeric, (0.0, 1.0)).all():
        raise ValueError(f"{name} must contain only boolean or 0/1 values")
    return numeric.astype(bool)


def deterministic_seed(base_seed: int, *identity: Any) -> int:
    """Derive an order-independent uint32 seed from an immutable analysis key."""

    if not isinstance(base_seed, (int, np.integer)) or int(base_seed) < 0:
        raise ValueError("base_seed must be a non-negative integer")
    payload = "|".join([str(int(base_seed)), *(str(item) for item in identity)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def deterministic_sha256_halves(
    study_uids: Sequence[Any] | pd.Series | np.ndarray,
    *,
    salt: str = SPLIT_HALF_SALT,
) -> np.ndarray:
    """Assign every study UID to deterministic SHA-256 half 0 or 1.

    Repeated occurrences of a study always receive the same assignment.  The
    function does not rebalance the halves, because rebalancing would make the
    assignment depend on the analysis cohort.
    """

    values = _as_one_dimensional(study_uids, name="study_uids")
    if pd.isna(values).any():
        raise ValueError("study_uids contains null values")
    if not isinstance(salt, str) or not salt:
        raise ValueError("salt must be a non-empty string")
    result = np.empty(len(values), dtype=np.int8)
    cache: dict[str, int] = {}
    for index, raw_uid in enumerate(values):
        uid = str(raw_uid)
        if not uid:
            raise ValueError("study_uids contains an empty value")
        assignment = cache.get(uid)
        if assignment is None:
            digest = hashlib.sha256(f"{salt}|{uid}".encode("utf-8")).digest()
            assignment = int(digest[0] & 1)
            cache[uid] = assignment
        result[index] = assignment
    return result


def benjamini_hochberg(p_values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted q-values, preserving missing values."""

    raw = _as_one_dimensional(p_values, name="p_values").astype(np.float64)
    finite = np.isfinite(raw)
    if ((raw[finite] < 0.0) | (raw[finite] > 1.0)).any():
        raise ValueError("finite p_values must be within [0, 1]")
    invalid = ~finite & ~np.isnan(raw)
    if invalid.any():
        raise ValueError("p_values may be finite or NaN, but not infinite")

    result = np.full(raw.shape, np.nan, dtype=np.float64)
    valid_positions = np.flatnonzero(finite)
    count = len(valid_positions)
    if count == 0:
        return result

    # Mergesort makes tied p-value handling reproducible across platforms.
    order_within_valid = np.argsort(raw[valid_positions], kind="mergesort")
    ordered_positions = valid_positions[order_within_valid]
    ordered_p = raw[ordered_positions]
    ranks = np.arange(1, count + 1, dtype=np.float64)
    adjusted = ordered_p * float(count) / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result[ordered_positions] = np.clip(adjusted, 0.0, 1.0)
    return result


def add_bh_qvalues(
    frame: pd.DataFrame,
    *,
    p_value_column: str = "p_value",
    family_columns: Sequence[str] = (),
    q_value_column: str = "q_value",
    q_threshold: float = 0.10,
    reject_column: str = "q_value_pass",
) -> pd.DataFrame:
    """Add BH q-values separately within each explicitly supplied family."""

    required = {p_value_column, *family_columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"BH input is missing columns: {missing}")
    if not 0.0 <= float(q_threshold) <= 1.0:
        raise ValueError("q_threshold must be within [0, 1]")

    result = frame.copy()
    q_values = np.full(len(result), np.nan, dtype=np.float64)
    if family_columns:
        grouped_positions = result.groupby(
            list(family_columns), sort=False, dropna=False
        ).indices.values()
    else:
        grouped_positions = (np.arange(len(result), dtype=np.int64),)
    for positions in grouped_positions:
        positions_array = np.asarray(positions, dtype=np.int64)
        q_values[positions_array] = benjamini_hochberg(
            result.iloc[positions_array][p_value_column].to_numpy()
        )
    result[q_value_column] = q_values
    result[reject_column] = np.isfinite(q_values) & (q_values <= float(q_threshold))
    return result


def residualize_target_label(
    frame: pd.DataFrame,
    value_column: str,
    *,
    target_column: str = "target",
    label_column: str = "hard_label",
    output_name: str | None = None,
) -> pd.Series:
    """Subtract the target-by-hard-label mean from a finite numeric outcome."""

    required = {value_column, target_column, label_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"residualization input is missing columns: {missing}")
    if frame[target_column].isna().any():
        raise ValueError(f"{target_column} contains null values")
    labels = _validated_binary(frame[label_column].to_numpy(), name=label_column)
    values = pd.to_numeric(frame[value_column], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(values).all():
        raise ValueError(f"{value_column} must contain only finite numeric values")

    working = pd.DataFrame(
        {
            "_target": frame[target_column].astype(str).to_numpy(),
            "_label": labels.astype(np.int8),
            "_value": values,
        },
        index=frame.index,
    )
    means = working.groupby(["_target", "_label"], sort=False, observed=True)[
        "_value"
    ].transform("mean")
    residual = pd.Series(
        values - means.to_numpy(dtype=np.float64),
        index=frame.index,
        name=output_name or f"{value_column}__target_label_residual",
        dtype=np.float64,
    )
    if not np.isfinite(residual.to_numpy()).all():
        raise AssertionError("target-label residualization produced non-finite values")
    return residual


def _study_level_values(
    values: Any,
    segment: Any,
    study_uids: Any,
) -> pd.DataFrame:
    numeric_values = _as_one_dimensional(values, name="values").astype(np.float64)
    binary_segment = _validated_binary(segment, name="segment")
    uids = _as_one_dimensional(study_uids, name="study_uids")
    if not (len(numeric_values) == len(binary_segment) == len(uids)):
        raise ValueError("values, segment, and study_uids must have the same length")
    if len(uids) == 0:
        raise ValueError("bootstrap input is empty")
    if not np.isfinite(numeric_values).all():
        raise ValueError("values must contain only finite numbers")
    if pd.isna(uids).any() or np.any(np.asarray([str(uid) == "" for uid in uids])):
        raise ValueError("study_uids must be non-null and non-empty")

    rows = pd.DataFrame(
        {
            "study_uid": np.asarray([str(uid) for uid in uids], dtype=object),
            "value": numeric_values,
            "segment": binary_segment,
        }
    )
    grouped = rows.groupby("study_uid", sort=False, observed=True)
    segment_nunique = grouped["segment"].nunique(dropna=False)
    if segment_nunique.gt(1).any():
        example = str(segment_nunique.loc[segment_nunique.gt(1)].index[0])
        raise ValueError(f"segment membership varies within study {example}")
    result = grouped.agg(value=("value", "mean"), segment=("segment", "first"))
    result["segment"] = result["segment"].astype(bool)
    return result.reset_index()


def _direction(value: float, *, zero_tolerance: float = 0.0) -> int:
    if not np.isfinite(value):
        return 0
    if value > zero_tolerance:
        return 1
    if value < -zero_tolerance:
        return -1
    return 0


def cluster_bootstrap_difference(
    values: Any,
    segment: Any,
    study_uids: Any,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = BOOTSTRAP_ALPHA,
    minimum_segment_studies: int = MINIMUM_STUDIES,
    minimum_complement_studies: int | None = None,
) -> dict[str, Any]:
    """Estimate a segment-minus-complement effect with a study bootstrap.

    Rows are first collapsed to one mean outcome per study.  Segment and
    complement studies are then independently resampled with replacement,
    retaining the original number of studies in each arm.  This is a cluster
    bootstrap even when the input has one row per target per study.
    """

    if not isinstance(replicates, (int, np.integer)) or int(replicates) < 1:
        raise ValueError("replicates must be a positive integer")
    if not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    if int(minimum_segment_studies) < 1:
        raise ValueError("minimum_segment_studies must be positive")
    if minimum_complement_studies is None:
        minimum_complement_studies = int(minimum_segment_studies)
    if int(minimum_complement_studies) < 1:
        raise ValueError("minimum_complement_studies must be positive")

    studies = _study_level_values(values, segment, study_uids)
    segment_values = studies.loc[studies["segment"], "value"].to_numpy(
        dtype=np.float64
    )
    complement_values = studies.loc[~studies["segment"], "value"].to_numpy(
        dtype=np.float64
    )
    n_segment = int(len(segment_values))
    n_complement = int(len(complement_values))
    point_effect = (
        float(segment_values.mean() - complement_values.mean())
        if n_segment and n_complement
        else np.nan
    )
    minimum_pass = (
        n_segment >= int(minimum_segment_studies)
        and n_complement >= int(minimum_complement_studies)
    )
    base: dict[str, Any] = {
        "status": "available" if minimum_pass else "suppressed_minimum_studies",
        "study_count": int(len(studies)),
        "segment_study_count": n_segment,
        "complement_study_count": n_complement,
        "minimum_segment_studies": int(minimum_segment_studies),
        "minimum_complement_studies": int(minimum_complement_studies),
        "minimum_studies_pass": bool(minimum_pass),
        "effect": point_effect,
        "effect_direction": _direction(point_effect),
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "p_value": np.nan,
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
        "bootstrap_alpha": float(alpha),
    }
    if not minimum_pass:
        return base

    generator = np.random.default_rng(int(seed))
    distribution = np.empty(int(replicates), dtype=np.float64)
    chunk_size = 256
    for start in range(0, int(replicates), chunk_size):
        stop = min(start + chunk_size, int(replicates))
        count = stop - start
        segment_draws = generator.integers(
            0, n_segment, size=(count, n_segment), endpoint=False
        )
        complement_draws = generator.integers(
            0, n_complement, size=(count, n_complement), endpoint=False
        )
        distribution[start:stop] = (
            segment_values[segment_draws].mean(axis=1)
            - complement_values[complement_draws].mean(axis=1)
        )

    lower, upper = np.quantile(
        distribution, [float(alpha) / 2.0, 1.0 - float(alpha) / 2.0]
    )
    lower_tail = (float(np.count_nonzero(distribution <= 0.0)) + 1.0) / (
        int(replicates) + 1.0
    )
    upper_tail = (float(np.count_nonzero(distribution >= 0.0)) + 1.0) / (
        int(replicates) + 1.0
    )
    base.update(
        {
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "p_value": float(min(1.0, 2.0 * min(lower_tail, upper_tail))),
        }
    )
    return base


def segment_vs_complement_effect_summary(
    frame: pd.DataFrame,
    *,
    outcome_column: str,
    segment_column: str,
    segment_value: Any = True,
    study_column: str = ID_COLUMN,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = BOOTSTRAP_ALPHA,
    minimum_segment_studies: int = MINIMUM_STUDIES,
    minimum_complement_studies: int | None = None,
    split_salt: str = SPLIT_HALF_SALT,
) -> dict[str, Any]:
    """Summarize one frozen segment against its complement and split halves."""

    required = {outcome_column, segment_column, study_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"segment summary input is missing columns: {missing}")
    if frame[segment_column].isna().any():
        raise ValueError(f"{segment_column} contains null values")
    membership = frame[segment_column].eq(segment_value).to_numpy(dtype=bool)
    analysis_seed = deterministic_seed(
        int(seed), outcome_column, segment_column, repr(segment_value)
    )
    result = cluster_bootstrap_difference(
        frame[outcome_column].to_numpy(),
        membership,
        frame[study_column].to_numpy(),
        replicates=replicates,
        seed=analysis_seed,
        alpha=alpha,
        minimum_segment_studies=minimum_segment_studies,
        minimum_complement_studies=minimum_complement_studies,
    )
    result.update(
        {
            "outcome": outcome_column,
            "segment_column": segment_column,
            "segment_value": segment_value,
            "split_half_salt": split_salt,
        }
    )

    studies = _study_level_values(
        frame[outcome_column].to_numpy(),
        membership,
        frame[study_column].to_numpy(),
    )
    halves = deterministic_sha256_halves(studies["study_uid"], salt=split_salt)
    half_directions: list[int] = []
    for half in (0, 1):
        half_rows = studies.loc[halves == half]
        segment_values = half_rows.loc[half_rows["segment"], "value"]
        complement_values = half_rows.loc[~half_rows["segment"], "value"]
        effect = (
            float(segment_values.mean() - complement_values.mean())
            if len(segment_values) and len(complement_values)
            else np.nan
        )
        direction = _direction(effect)
        half_directions.append(direction)
        result[f"split_half_{half}_study_count"] = int(len(half_rows))
        result[f"split_half_{half}_segment_study_count"] = int(len(segment_values))
        result[f"split_half_{half}_complement_study_count"] = int(
            len(complement_values)
        )
        result[f"split_half_{half}_effect"] = effect
        result[f"split_half_{half}_direction"] = direction

    split_stable = half_directions[0] != 0 and half_directions[0] == half_directions[1]
    result["split_half_direction_stable"] = bool(split_stable)
    result["split_half_both_positive"] = bool(half_directions == [1, 1])
    return result


def score_family_direction_stability(
    effects: Mapping[str, float],
    *,
    primary_family: str,
    sensitivity_families: Sequence[str],
    minimum_matching_sensitivities: int = 2,
    zero_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Check frozen primary-versus-sensitivity effect-direction stability."""

    if primary_family not in effects:
        raise ValueError(f"primary family is missing from effects: {primary_family}")
    families = tuple(sensitivity_families)
    if len(families) != len(set(families)):
        raise ValueError("sensitivity_families contains duplicates")
    if primary_family in families:
        raise ValueError("primary_family cannot also be a sensitivity family")
    if not 0 <= int(minimum_matching_sensitivities) <= len(families):
        raise ValueError("minimum_matching_sensitivities is outside the family count")
    missing = sorted(set(families).difference(effects))
    if missing:
        raise ValueError(f"sensitivity effects are missing: {missing}")
    selected = (primary_family, *families)
    non_finite = [family for family in selected if not np.isfinite(effects[family])]
    if non_finite:
        raise ValueError(f"score-family effects must be finite: {non_finite}")
    if float(zero_tolerance) < 0.0:
        raise ValueError("zero_tolerance must be non-negative")

    directions = {
        family: _direction(float(effects[family]), zero_tolerance=float(zero_tolerance))
        for family in selected
    }
    primary_direction = directions[primary_family]
    matching = sum(
        directions[family] == primary_direction and primary_direction != 0
        for family in families
    )
    positive = sum(directions[family] == 1 for family in families)
    negative = sum(directions[family] == -1 for family in families)
    return {
        "primary_family": primary_family,
        "primary_effect": float(effects[primary_family]),
        "primary_direction": int(primary_direction),
        "sensitivity_family_count": len(families),
        "minimum_matching_sensitivities": int(minimum_matching_sensitivities),
        "matching_sensitivity_count": int(matching),
        "positive_sensitivity_count": int(positive),
        "negative_sensitivity_count": int(negative),
        "directions": directions,
        "direction_stability_pass": bool(
            primary_direction != 0 and matching >= int(minimum_matching_sensitivities)
        ),
        "positive_candidate_gate_pass": bool(
            primary_direction == 1
            and positive >= int(minimum_matching_sensitivities)
        ),
    }

