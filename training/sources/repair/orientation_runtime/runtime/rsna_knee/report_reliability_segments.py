"""Outcome-blind segment membership and maxT mechanics for reliability v3.

The functions in this module operate only on supplied morphology, proxy
residual, report-group, and frozen-stratum frames.  They never read files or
inspect Gold outcomes.  The implementation preserves the joint dependence of
the frozen segment universe by applying one common report-group permutation to
all segment membership columns inside every frozen stratum.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from .constants import ID_COLUMN


REPORT_GROUP_COLUMN = "report_group"
HYPOTHESIS_ID_COLUMN = "hypothesis_id"


def canonical_exclusion_key(fields: Sequence[str]) -> str:
    """Return a stable compact key for one frozen adjustment exclusion set."""

    normalized = tuple(sorted(str(field) for field in fields))
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("adjustment exclusion fields must be non-empty and unique")
    payload = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _required_columns(
    frame: pd.DataFrame, columns: Sequence[str], *, name: str
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def _normalize_category(series: pd.Series) -> pd.Series:
    return series.where(series.notna(), "__MISSING__").astype(str)


def _lexicographic_mode(series: pd.Series) -> str:
    values = _normalize_category(series)
    counts = values.value_counts(sort=False, dropna=False)
    maximum = int(counts.max())
    return min(str(value) for value in counts.index[counts.eq(maximum)])


def _study_level_source(
    frame: pd.DataFrame, *, fields: Sequence[str]
) -> pd.DataFrame:
    columns = [ID_COLUMN, REPORT_GROUP_COLUMN, *fields]
    _required_columns(frame, columns, name="segment morphology input")
    if frame.empty:
        raise ValueError("segment morphology input is empty")
    if frame[[ID_COLUMN, REPORT_GROUP_COLUMN]].isna().any().any():
        raise ValueError("segment morphology identities contain nulls")
    identity = frame[[ID_COLUMN, REPORT_GROUP_COLUMN]].astype(str)
    if identity[ID_COLUMN].str.strip().eq("").any() or identity[
        REPORT_GROUP_COLUMN
    ].str.strip().eq("").any():
        raise ValueError("segment morphology identities contain empty strings")
    drift = frame.groupby(ID_COLUMN, sort=True, observed=True)[
        [REPORT_GROUP_COLUMN, *fields]
    ].nunique(dropna=False)
    if (drift > 1).any().any():
        raise ValueError("segment morphology fields drift within StudyInstanceUID")
    return (
        frame[columns]
        .sort_values(ID_COLUMN, kind="mergesort")
        .drop_duplicates(ID_COLUMN, keep="first")
        .reset_index(drop=True)
    )


def build_frozen_segment_memberships(
    morphology_frame: pd.DataFrame,
    universe: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Reconstruct and validate the exact frozen report-group memberships.

    The aggregation and threshold rules are read from the already-frozen
    universe.  Every reconstructed arm count must match the frozen count,
    making this a fail-closed reconstruction rather than a new segment search.
    """

    hypotheses = list(universe.get("hypotheses", []))
    if not hypotheses:
        raise ValueError("frozen segment universe contains no hypotheses")
    categorical_fields: set[str] = set()
    binary_fields: set[str] = set()
    continuous_fields: set[str] = set()
    for item in hypotheses:
        kind = str(item.get("kind"))
        if kind.startswith("categorical_"):
            categorical_fields.add(str(item["field"]))
        elif kind == "binary_true_vs_false":
            binary_fields.add(str(item["field"]))
        elif kind == "continuous_decile_vs_rest":
            continuous_fields.add(str(item["field"]))
        elif kind == "language_by_length_decile_vs_rest":
            categorical_fields.add("language_proxy")
            continuous_fields.add(str(item["length_field"]))
        else:
            raise ValueError(f"unsupported frozen segment kind: {kind}")
    fields = tuple(
        sorted(categorical_fields | binary_fields | continuous_fields)
    )
    studies = _study_level_source(morphology_frame, fields=fields)
    grouped = studies.groupby(REPORT_GROUP_COLUMN, sort=True, observed=True)
    group_index = pd.Index(
        sorted(studies[REPORT_GROUP_COLUMN].astype(str).unique()),
        name=REPORT_GROUP_COLUMN,
    )
    aggregate = pd.DataFrame(index=group_index)
    for field in sorted(categorical_fields):
        aggregate[field] = grouped[field].agg(_lexicographic_mode).reindex(group_index)
    for field in sorted(binary_fields):
        values = pd.to_numeric(studies[field], errors="raise")
        if not values.isin([0, 1]).all():
            raise ValueError(f"binary segment field is not 0/1: {field}")
        aggregate[field] = (
            values.groupby(studies[REPORT_GROUP_COLUMN], sort=True)
            .mean()
            .ge(0.5)
            .astype(np.int8)
            .reindex(group_index)
        )
    for field in sorted(continuous_fields):
        values = pd.to_numeric(studies[field], errors="raise")
        numeric = values.to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise ValueError(f"continuous segment field contains non-finite values: {field}")
        aggregate[field] = (
            values.groupby(studies[REPORT_GROUP_COLUMN], sort=True)
            .median()
            .reindex(group_index)
        )
    if aggregate.isna().any().any():
        raise ValueError("report-group segment aggregation produced nulls")

    continuous_metadata = universe.get("continuous_metadata", {})
    deciles: dict[str, np.ndarray] = {}
    for field in sorted(continuous_fields):
        metadata = continuous_metadata.get(field)
        if not isinstance(metadata, Mapping):
            raise ValueError(f"missing frozen continuous metadata for {field}")
        cutpoints = np.asarray(metadata.get("cutpoints"), dtype=np.float64)
        if cutpoints.shape != (11,) or not np.isfinite(cutpoints).all():
            raise ValueError(f"invalid frozen decile cutpoints for {field}")
        if np.any(np.diff(cutpoints) < 0.0):
            raise ValueError(f"non-monotone frozen decile cutpoints for {field}")
        deciles[field] = np.searchsorted(
            cutpoints[1:-1],
            aggregate[field].to_numpy(dtype=np.float64),
            side="right",
        ).astype(np.int8)

    membership_columns: dict[str, np.ndarray] = {}
    metadata_rows: list[dict[str, Any]] = []
    for item in hypotheses:
        hypothesis_id = str(item["id"])
        kind = str(item["kind"])
        if hypothesis_id in membership_columns:
            raise ValueError(f"duplicate frozen hypothesis id: {hypothesis_id}")
        if kind == "categorical_one_vs_rest":
            selected = _normalize_category(aggregate[str(item["field"])]).eq(
                str(item["value"])
            )
        elif kind == "categorical_rare_pool_vs_rest":
            selected = _normalize_category(aggregate[str(item["field"])]).isin(
                [str(value) for value in item["members"]]
            )
        elif kind == "binary_true_vs_false":
            selected = pd.Series(
                aggregate[str(item["field"])].to_numpy(dtype=np.int8)
                == int(item["value"]),
                index=group_index,
            )
        elif kind == "continuous_decile_vs_rest":
            selected = pd.Series(
                deciles[str(item["field"])] == int(item["decile"]),
                index=group_index,
            )
        else:
            length_field = str(item["length_field"])
            selected = _normalize_category(aggregate["language_proxy"]).eq(
                str(item["language"])
            ) & pd.Series(
                deciles[length_field] == int(item["length_decile"]),
                index=group_index,
            )
        values = selected.to_numpy(dtype=bool)
        segment_count = int(values.sum())
        complement_count = int(len(values) - segment_count)
        if segment_count != int(item["segment_report_groups"]) or complement_count != int(
            item["complement_report_groups"]
        ):
            raise ValueError(
                f"frozen membership count drift for {hypothesis_id}: "
                f"observed={segment_count}/{complement_count} "
                f"expected={item['segment_report_groups']}/{item['complement_report_groups']}"
            )
        membership_columns[hypothesis_id] = values
        excluded = tuple(str(field) for field in item["adjustment_excluded_fields"])
        metadata_rows.append(
            {
                HYPOTHESIS_ID_COLUMN: hypothesis_id,
                "scope": str(item["scope"]),
                "kind": kind,
                "adjustment_excluded_fields_json": json.dumps(
                    list(excluded), ensure_ascii=False, separators=(",", ":")
                ),
                "adjustment_key": canonical_exclusion_key(excluded),
                "frozen_segment_report_groups": segment_count,
                "frozen_complement_report_groups": complement_count,
            }
        )
    memberships = pd.DataFrame(membership_columns, index=group_index, dtype=bool)
    metadata = pd.DataFrame(metadata_rows).sort_values(
        HYPOTHESIS_ID_COLUMN, kind="mergesort"
    ).reset_index(drop=True)
    memberships = memberships.loc[:, metadata[HYPOTHESIS_ID_COLUMN].tolist()]
    diagnostics = {
        "status": "PASS",
        "eligible_studies": int(studies[ID_COLUMN].nunique()),
        "eligible_report_groups": int(len(group_index)),
        "hypotheses": int(len(metadata)),
        "unique_adjustment_exclusion_sets": int(metadata["adjustment_key"].nunique()),
        "outcome_columns_used": [],
    }
    return memberships, metadata, diagnostics


def _welch_effect_and_t(
    residuals: np.ndarray,
    membership: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized signed mean effect and absolute Welch t statistic."""

    values = np.asarray(residuals, dtype=np.float64)
    selected = np.asarray(membership, dtype=bool)
    if values.ndim != 2 or selected.shape != values.shape:
        raise ValueError("residual and membership matrices must have identical 2D shapes")
    if not np.isfinite(values).all():
        raise ValueError("segment residual matrix contains non-finite values")
    n_total = values.shape[0]
    n1 = selected.sum(axis=0).astype(np.int64)
    n0 = n_total - n1
    if np.any(n1 < 2) or np.any(n0 < 2):
        raise ValueError("Welch statistic requires at least two groups per arm")
    selected_float = selected.astype(np.float64)
    total_sum = values.sum(axis=0, dtype=np.float64)
    total_sumsq = np.square(values).sum(axis=0, dtype=np.float64)
    sum1 = np.einsum("nh,nh->h", selected_float, values, optimize=True)
    sumsq1 = np.einsum(
        "nh,nh->h", selected_float, np.square(values), optimize=True
    )
    sum0 = total_sum - sum1
    sumsq0 = total_sumsq - sumsq1
    mean1 = sum1 / n1
    mean0 = sum0 / n0
    var1 = np.maximum((sumsq1 - np.square(sum1) / n1) / (n1 - 1), 0.0)
    var0 = np.maximum((sumsq0 - np.square(sum0) / n0) / (n0 - 1), 0.0)
    denominator = np.sqrt(var1 / n1 + var0 / n0)
    effect = mean1 - mean0
    invalid = denominator <= 0.0
    if np.any(invalid & (np.abs(effect) > 1e-15)):
        raise ValueError("non-zero segment effect has zero Welch denominator")
    statistic = np.divide(
        np.abs(effect),
        denominator,
        out=np.zeros_like(effect),
        where=~invalid,
    )
    if not np.isfinite(effect).all() or not np.isfinite(statistic).all():
        raise ValueError("segment effect or Welch statistic is non-finite")
    return effect, statistic, n1, n0


def _permuted_welch_t(
    residuals: np.ndarray,
    permuted_membership: np.ndarray,
    *,
    n1: np.ndarray,
    n0: np.ndarray,
) -> np.ndarray:
    values = np.asarray(residuals, dtype=np.float64)
    selected = np.asarray(permuted_membership, dtype=bool)
    if selected.ndim != 3 or selected.shape[1:] != values.shape:
        raise ValueError("batched membership shape differs from residual matrix")
    selected_float = selected.astype(np.float64)
    total_sum = values.sum(axis=0, dtype=np.float64)
    total_sumsq = np.square(values).sum(axis=0, dtype=np.float64)
    sum1 = np.einsum("pnh,nh->ph", selected_float, values, optimize=True)
    sumsq1 = np.einsum(
        "pnh,nh->ph", selected_float, np.square(values), optimize=True
    )
    sum0 = total_sum[None, :] - sum1
    sumsq0 = total_sumsq[None, :] - sumsq1
    mean1 = sum1 / n1[None, :]
    mean0 = sum0 / n0[None, :]
    var1 = np.maximum(
        (sumsq1 - np.square(sum1) / n1[None, :]) / (n1[None, :] - 1),
        0.0,
    )
    var0 = np.maximum(
        (sumsq0 - np.square(sum0) / n0[None, :]) / (n0[None, :] - 1),
        0.0,
    )
    denominator = np.sqrt(var1 / n1[None, :] + var0 / n0[None, :])
    effect = mean1 - mean0
    invalid = denominator <= 0.0
    if np.any(invalid & (np.abs(effect) > 1e-15)):
        raise ValueError("permuted non-zero effect has zero Welch denominator")
    statistic = np.divide(
        np.abs(effect), denominator, out=np.zeros_like(effect), where=~invalid
    )
    if not np.isfinite(statistic).all():
        raise ValueError("permuted Welch statistic is non-finite")
    return statistic


def max_t_segment_family(
    residuals: pd.DataFrame,
    memberships: pd.DataFrame,
    strata: pd.Series,
    *,
    minimum_report_groups_per_arm: int = 100,
    permutations: int = 999,
    seed: int = 20260830,
    permutation_chunk_size: int = 16,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run one target-direction joint-label permutation/maxT family.

    ``residuals`` and ``memberships`` must have the same report-group index and
    the same hypothesis columns.  A single within-stratum row permutation is
    applied to every membership column, preserving dependence among the 127
    frozen hypotheses for the maxT null.
    """

    if not isinstance(residuals, pd.DataFrame) or not isinstance(
        memberships, pd.DataFrame
    ):
        raise ValueError("residuals and memberships must be pandas DataFrames")
    if not residuals.index.is_unique or not memberships.index.is_unique:
        raise ValueError("segment family indices must be unique")
    if not residuals.index.equals(memberships.index):
        raise ValueError("residual and membership report-group indices differ")
    if list(residuals.columns) != list(memberships.columns):
        raise ValueError("residual and membership hypothesis columns differ")
    if not strata.index.equals(residuals.index):
        raise ValueError("permutation strata index differs from segment family index")
    if strata.isna().any():
        raise ValueError("permutation strata contain nulls")
    if int(minimum_report_groups_per_arm) < 2:
        raise ValueError("minimum report-group support must be at least two")
    if int(permutations) < 1 or int(permutation_chunk_size) < 1:
        raise ValueError("permutation count and chunk size must be positive")
    values = residuals.to_numpy(dtype=np.float64)
    membership = memberships.to_numpy(dtype=bool)
    if not np.isfinite(values).all():
        raise ValueError("segment family residuals contain non-finite values")
    n1_all = membership.sum(axis=0).astype(np.int64)
    n0_all = len(membership) - n1_all
    eligible = (n1_all >= int(minimum_report_groups_per_arm)) & (
        n0_all >= int(minimum_report_groups_per_arm)
    )
    if not eligible.any():
        raise ValueError("segment family has no support-eligible hypotheses")

    eligible_values = values[:, eligible]
    eligible_membership = membership[:, eligible]
    effect, observed_t, n1, n0 = _welch_effect_and_t(
        eligible_values, eligible_membership
    )
    raw_exceedance = np.zeros(len(observed_t), dtype=np.int64)
    max_null_statistics = np.empty(int(permutations), dtype=np.float64)
    stratum_positions = [
        positions.to_numpy(dtype=np.int64)
        for _, positions in pd.Series(
            np.arange(len(strata), dtype=np.int64), index=strata.astype(str).to_numpy()
        )
        .groupby(level=0, sort=True)
    ]
    rng = np.random.default_rng(int(seed))
    cursor = 0
    while cursor < int(permutations):
        chunk = min(int(permutation_chunk_size), int(permutations) - cursor)
        indices = np.tile(np.arange(len(strata), dtype=np.int64), (chunk, 1))
        for row in range(chunk):
            for positions in stratum_positions:
                indices[row, positions] = rng.permutation(positions)
        permuted = eligible_membership[indices]
        null_t = _permuted_welch_t(
            eligible_values, permuted, n1=n1, n0=n0
        )
        raw_exceedance += np.sum(null_t >= observed_t[None, :] - 1e-15, axis=0)
        max_null_statistics[cursor : cursor + chunk] = np.max(null_t, axis=1)
        cursor += chunk

    max_exceedance = np.sum(
        max_null_statistics[:, None] >= observed_t[None, :] - 1e-15, axis=0
    ).astype(np.int64)
    observed_family_max = float(np.max(observed_t))
    family_exceedance = int(
        np.sum(max_null_statistics >= observed_family_max - 1e-15)
    )

    output = pd.DataFrame(
        {
            HYPOTHESIS_ID_COLUMN: list(residuals.columns),
            "direction_segment_report_groups": n1_all,
            "direction_complement_report_groups": n0_all,
            "direction_support_eligible": eligible,
            "segment_effect": np.nan,
            "absolute_welch_t": np.nan,
            "raw_permutation_exceedance_count": np.nan,
            "raw_permutation_p_value": np.nan,
            "maxT_exceedance_count": np.nan,
            "maxT_adjusted_p_value": np.nan,
        }
    )
    eligible_positions = np.flatnonzero(eligible)
    output.loc[eligible_positions, "segment_effect"] = effect
    output.loc[eligible_positions, "absolute_welch_t"] = observed_t
    output.loc[
        eligible_positions, "raw_permutation_exceedance_count"
    ] = raw_exceedance
    output.loc[eligible_positions, "raw_permutation_p_value"] = (
        1.0 + raw_exceedance
    ) / (1.0 + int(permutations))
    output.loc[eligible_positions, "maxT_exceedance_count"] = max_exceedance
    output.loc[eligible_positions, "maxT_adjusted_p_value"] = (
        1.0 + max_exceedance
    ) / (1.0 + int(permutations))
    family = {
        "eligible_hypotheses": int(eligible.sum()),
        "total_hypotheses": int(len(eligible)),
        "observed_family_max_absolute_welch_t": observed_family_max,
        "family_maxT_exceedance_count": family_exceedance,
        "family_maxT_p_value": float(
            (1.0 + family_exceedance) / (1.0 + int(permutations))
        ),
        "permutations": int(permutations),
        "seed": int(seed),
        "joint_membership_permutation": True,
    }
    return output, family


def signed_group_effect(
    residual: pd.Series,
    membership: pd.Series,
    *,
    minimum_report_groups_per_arm: int = 2,
) -> dict[str, int | float | bool]:
    """Compute one signed report-group mean contrast with finite gates."""

    if not residual.index.equals(membership.index):
        raise ValueError("signed-effect residual and membership indices differ")
    valid = residual.notna() & membership.notna()
    values = pd.to_numeric(residual.loc[valid], errors="raise").to_numpy(float)
    selected = membership.loc[valid].astype(bool).to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("signed-effect residual contains non-finite values")
    n1 = int(selected.sum())
    n0 = int(len(selected) - n1)
    supported = min(n1, n0) >= int(minimum_report_groups_per_arm)
    effect = (
        float(values[selected].mean() - values[~selected].mean())
        if supported
        else np.nan
    )
    return {
        "segment_report_groups": n1,
        "complement_report_groups": n0,
        "support_eligible": bool(supported),
        "effect": effect,
    }


__all__ = [
    "HYPOTHESIS_ID_COLUMN",
    "REPORT_GROUP_COLUMN",
    "build_frozen_segment_memberships",
    "canonical_exclusion_key",
    "max_t_segment_family",
    "signed_group_effect",
]
