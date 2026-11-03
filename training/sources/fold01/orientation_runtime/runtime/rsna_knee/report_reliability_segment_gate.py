"""Fail-closed verifier for the frozen reliability-v3 segment screen.

This module never computes segment effects and never reads Gold outcomes.  It
checks that a supplied ``segment_effects_v3.csv.gz`` is exactly the Cartesian
product of the 127 outcome-blind hypotheses and 24 target-direction families,
that frozen membership/adjustment identities are preserved, and that the
permutation, maxT, and cross-family BH bookkeeping is internally exact.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .constants import TARGET_COLUMNS


EXPECTED_UNIVERSE_SHA256 = (
    "0a508fd5d7d27b2f24774636e78aaadb8fad6401f693593c331664cb4ac7b7e4"
)
EXPECTED_HYPOTHESES_SHA256 = (
    "9d50052e95340547c419eba3019ec6f1686cbaca4267979938ebc8763d6efd79"
)
EXPECTED_CONTRACT_VERSION = "3.4.0"
EXPECTED_CONTRACT_SEMANTIC_HASH = (
    "f00d67756c1d840dfeb5d6738e172d03eee452f2e091c29f4a3366ff4ec30e31"
)
EXPECTED_HYPOTHESES = 127
EXPECTED_FAMILIES = 24
EXPECTED_ROWS = EXPECTED_HYPOTHESES * EXPECTED_FAMILIES
EXPECTED_PERMUTATIONS = 999
EXPECTED_SEED = 20260830
MINIMUM_REPORT_GROUPS_PER_ARM = 100
BH_Q = 0.10
DIRECTIONS: tuple[str, str] = ("contradiction", "omission")


CORE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "target",
    "direction",
    "hypothesis_id",
    "scope",
    "kind",
    "adjustment_key",
    "adjustment_excluded_fields_json",
    "frozen_segment_report_groups",
    "frozen_complement_report_groups",
    "direction_segment_report_groups",
    "direction_complement_report_groups",
    "direction_support_eligible",
    "segment_effect",
    "absolute_welch_t",
    "raw_permutation_exceedance_count",
    "raw_permutation_p_value",
    "maxT_exceedance_count",
    "maxT_adjusted_p_value",
    "eligible_hypotheses",
    "total_hypotheses",
    "observed_family_max_absolute_welch_t",
    "family_maxT_exceedance_count",
    "family_maxT_p_value",
    "permutations",
    "seed",
    "joint_membership_permutation",
    "family_bh_q_value",
    "family_bh_reject",
    "core_maxT_survivor",
)

ROBUSTNESS_REQUIRED_COLUMNS: tuple[str, ...] = (
    "dino_segment_effect",
    "rad_segment_effect",
    "dino_rad_same_sign",
    "high_agreement_segment_report_groups",
    "high_agreement_complement_report_groups",
    "high_agreement_support_eligible",
    "high_agreement_segment_effect",
    "high_agreement_sign_retained",
    "common_support_profile_count",
    "common_support_profiles_json",
    "common_support_segment_report_groups",
    "common_support_complement_report_groups",
    "common_support_eligible",
    "common_support_segment_effect",
    "common_support_sign_retained",
)

LOPO_REQUIRED_COLUMNS: tuple[str, ...] = (
    "lopo_status",
    "lopo_profiles_required",
    "lopo_profiles_completed",
    "lopo_all_effects_same_sign",
    "proxy_segment_status",
)

SEGMENT_EFFECT_COLUMNS: tuple[str, ...] = (
    *CORE_REQUIRED_COLUMNS,
    *ROBUSTNESS_REQUIRED_COLUMNS,
    *LOPO_REQUIRED_COLUMNS,
)

SEGMENT_FAMILY_COLUMNS: tuple[str, ...] = (
    "target",
    "direction",
    "eligible_hypotheses",
    "total_hypotheses",
    "observed_family_max_absolute_welch_t",
    "family_maxT_exceedance_count",
    "family_maxT_p_value",
    "permutations",
    "seed",
    "joint_membership_permutation",
    "family_bh_q_value",
    "family_bh_reject",
)

SEGMENT_CORE_OUTPUT_FILENAMES: tuple[str, ...] = (
    "segment_effects_v3.csv.gz",
    "segment_family_maxT_v3.csv",
    "segment_proxy_shortlist_v3.csv",
    "SEGMENT_EFFECTS_AUDIT.json",
)


@dataclass(frozen=True)
class SegmentEffectsVerification:
    effects: pd.DataFrame
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class SegmentEffectsDeterminismVerification:
    diagnostics: Mapping[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_exclusion_key(fields: Sequence[str]) -> str:
    normalized = tuple(sorted(str(field) for field in fields))
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("adjustment exclusion fields must be non-empty and unique")
    payload = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _strict_boolean(values: pd.Series, *, name: str) -> pd.Series:
    if values.isna().any() or not values.isin([True, False, 0, 1]).all():
        raise ValueError(f"{name} must contain only non-null booleans")
    return values.astype(bool)


def _integer_series(
    values: pd.Series,
    *,
    name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric).all():
        raise ValueError(f"{name} must contain only finite integers")
    if not np.equal(numeric.to_numpy(float), np.floor(numeric.to_numpy(float))).all():
        raise ValueError(f"{name} must contain only finite integers")
    result = numeric.astype(np.int64)
    if result.lt(minimum).any() or (
        maximum is not None and result.gt(maximum).any()
    ):
        raise ValueError(f"{name} lies outside its frozen bounds")
    return result


def _numeric_series(
    values: pd.Series,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric).all():
        raise ValueError(f"{name} must contain only finite numeric values")
    if minimum is not None and numeric.lt(minimum).any():
        raise ValueError(f"{name} lies below its frozen bound")
    if maximum is not None and numeric.gt(maximum).any():
        raise ValueError(f"{name} lies above its frozen bound")
    return numeric.astype(np.float64)


def _conditional_numeric(
    values: pd.Series,
    applicable: pd.Series,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> pd.Series:
    nonnull = values.notna()
    if not nonnull.eq(applicable).all():
        raise ValueError(f"{name} applicability differs from frozen support")
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric[nonnull].isna().any() or not np.isfinite(numeric[nonnull]).all():
        raise ValueError(f"{name} contains nonnumeric/nonfinite values")
    if minimum is not None and numeric[nonnull].lt(minimum).any():
        raise ValueError(f"{name} lies below its frozen bound")
    if maximum is not None and numeric[nonnull].gt(maximum).any():
        raise ValueError(f"{name} lies above its frozen bound")
    return numeric.astype(np.float64)


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("BH p-values must be a finite one-dimensional array")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("BH p-values lie outside [0, 1]")
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def _validate_universe(
    universe_path: Path, *, expected_universe_sha256: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if universe_path.name != "SEGMENT_UNIVERSE.json":
        raise ValueError("segment universe filename must be SEGMENT_UNIVERSE.json")
    if not universe_path.is_file():
        raise FileNotFoundError(universe_path)
    if _sha256_file(universe_path) != expected_universe_sha256:
        raise ValueError("segment universe SHA-256 mismatch")
    universe = json.loads(universe_path.read_text(encoding="utf-8"))
    if not isinstance(universe, dict):
        raise ValueError("segment universe root must be an object")
    expected_scalars = {
        "version": "1.0.0",
        "status": "FROZEN_OUTCOME_BLIND",
        "contract_version": EXPECTED_CONTRACT_VERSION,
        "contract_semantic_hash": EXPECTED_CONTRACT_SEMANTIC_HASH,
        "minimum_report_groups_per_arm": MINIMUM_REPORT_GROUPS_PER_ARM,
        "primary_hypothesis_count": EXPECTED_HYPOTHESES,
        "sensitivity_hypothesis_count": 0,
        "hypotheses_sha256": EXPECTED_HYPOTHESES_SHA256,
        "gold_outcomes_loaded": False,
    }
    for key, expected in expected_scalars.items():
        if universe.get(key) != expected:
            raise ValueError(f"segment universe {key} drift")
    if universe.get("outcome_columns_used") != []:
        raise ValueError("segment universe is not outcome-blind")
    eligible = universe.get("eligible_proxy")
    if eligible != {
        "studies": 4314,
        "report_groups": 4189,
        "excluded_report_group_count": 11,
    }:
        raise ValueError("segment universe eligible-proxy identity drift")
    hypotheses = universe.get("hypotheses")
    if not isinstance(hypotheses, list) or len(hypotheses) != EXPECTED_HYPOTHESES:
        raise ValueError("segment universe hypothesis count drift")
    if _canonical_hash(hypotheses) != EXPECTED_HYPOTHESES_SHA256:
        raise ValueError("segment universe canonical hypothesis hash mismatch")
    ids = [str(item.get("id")) for item in hypotheses]
    if len(set(ids)) != EXPECTED_HYPOTHESES or ids != sorted(ids):
        raise ValueError("segment hypothesis ids are duplicate or unsorted")
    for item in hypotheses:
        if item.get("scope") != "PRIMARY":
            raise ValueError("non-primary hypothesis escaped the frozen universe")
        n1 = int(item.get("segment_report_groups", -1))
        n0 = int(item.get("complement_report_groups", -1))
        if min(n1, n0) < MINIMUM_REPORT_GROUPS_PER_ARM or n1 + n0 != 4189:
            raise ValueError(f"frozen membership support drift for {item.get('id')}")
        fields = item.get("adjustment_excluded_fields")
        if not isinstance(fields, list) or not fields:
            raise ValueError(f"missing adjustment exclusion for {item.get('id')}")
        _canonical_exclusion_key(fields)
    return universe, hypotheses


def _expected_keys(hypotheses: Sequence[Mapping[str, Any]]) -> list[tuple[str, str, str]]:
    hypothesis_ids = sorted(str(item["id"]) for item in hypotheses)
    return [
        (target, direction, hypothesis_id)
        for target in sorted(TARGET_COLUMNS)
        for direction in sorted(DIRECTIONS)
        for hypothesis_id in hypothesis_ids
    ]


def _validate_identity(
    effects: pd.DataFrame, hypotheses: Sequence[Mapping[str, Any]]
) -> None:
    key_columns = ["target", "direction", "hypothesis_id"]
    if effects[key_columns].isna().any().any():
        raise ValueError("segment effect keys contain nulls")
    for column in key_columns:
        effects[column] = effects[column].astype(str)
        if effects[column].str.strip().eq("").any():
            raise ValueError("segment effect keys contain empty strings")
    if effects.duplicated(key_columns).any():
        raise ValueError("segment effect keys are duplicate")
    observed = list(effects[key_columns].itertuples(index=False, name=None))
    if observed != _expected_keys(hypotheses):
        raise ValueError("segment effects are not the exact sorted 127 x 24 key set")

    frozen_n1 = _integer_series(
        effects["frozen_segment_report_groups"], name="frozen segment support"
    )
    frozen_n0 = _integer_series(
        effects["frozen_complement_report_groups"], name="frozen complement support"
    )
    metadata = {str(item["id"]): item for item in hypotheses}
    for position, row in enumerate(effects.itertuples(index=False)):
        item = metadata[str(row.hypothesis_id)]
        excluded = [str(field) for field in item["adjustment_excluded_fields"]]
        expected_json = json.dumps(
            excluded, ensure_ascii=False, separators=(",", ":")
        )
        if str(row.scope) != str(item["scope"]):
            raise ValueError(f"scope drift for {row.hypothesis_id}")
        if str(row.kind) != str(item["kind"]):
            raise ValueError(f"kind drift for {row.hypothesis_id}")
        if str(row.adjustment_excluded_fields_json) != expected_json:
            raise ValueError(f"adjustment exclusion drift for {row.hypothesis_id}")
        if str(row.adjustment_key) != _canonical_exclusion_key(excluded):
            raise ValueError(f"adjustment key drift for {row.hypothesis_id}")
        if int(frozen_n1.iloc[position]) != int(item["segment_report_groups"]) or int(
            frozen_n0.iloc[position]
        ) != int(item["complement_report_groups"]):
            raise ValueError(f"frozen membership support drift for {row.hypothesis_id}")


def _validate_core_statistics(effects: pd.DataFrame) -> None:
    n1 = _integer_series(
        effects["direction_segment_report_groups"],
        name="direction segment support",
    )
    n0 = _integer_series(
        effects["direction_complement_report_groups"],
        name="direction complement support",
    )
    eligible = _strict_boolean(
        effects["direction_support_eligible"], name="direction support eligibility"
    )
    expected_eligible = n1.ge(MINIMUM_REPORT_GROUPS_PER_ARM) & n0.ge(
        MINIMUM_REPORT_GROUPS_PER_ARM
    )
    if not eligible.eq(expected_eligible).all():
        raise ValueError("direction support eligibility differs from arm counts")
    frozen_n1 = _integer_series(
        effects["frozen_segment_report_groups"],
        name="frozen segment support",
    )
    frozen_n0 = _integer_series(
        effects["frozen_complement_report_groups"],
        name="frozen complement support",
    )
    if n1.gt(frozen_n1).any() or n0.gt(frozen_n0).any():
        raise ValueError("direction support exceeds the frozen membership cohort")
    family_totals = pd.DataFrame(
        {
            "target": effects["target"].astype(str),
            "direction": effects["direction"].astype(str),
            "total": n1 + n0,
        }
    ).groupby(["target", "direction"], sort=True, observed=True)["total"]
    if family_totals.nunique().ne(1).any():
        raise ValueError("direction membership totals drift within a family")

    _conditional_numeric(
        effects["segment_effect"], eligible, name="segment effect"
    )
    observed_t = _conditional_numeric(
        effects["absolute_welch_t"],
        eligible,
        name="absolute Welch statistic",
        minimum=0.0,
    )
    raw_count = _conditional_numeric(
        effects["raw_permutation_exceedance_count"],
        eligible,
        name="raw permutation exceedance count",
        minimum=0.0,
        maximum=EXPECTED_PERMUTATIONS,
    )
    max_count = _conditional_numeric(
        effects["maxT_exceedance_count"],
        eligible,
        name="maxT exceedance count",
        minimum=0.0,
        maximum=EXPECTED_PERMUTATIONS,
    )
    if not np.equal(raw_count[eligible], np.floor(raw_count[eligible])).all() or not np.equal(
        max_count[eligible], np.floor(max_count[eligible])
    ).all():
        raise ValueError("permutation exceedance counts must be integers")
    raw_p = _conditional_numeric(
        effects["raw_permutation_p_value"],
        eligible,
        name="raw permutation p-value",
        minimum=0.0,
        maximum=1.0,
    )
    max_p = _conditional_numeric(
        effects["maxT_adjusted_p_value"],
        eligible,
        name="maxT adjusted p-value",
        minimum=0.0,
        maximum=1.0,
    )
    np.testing.assert_allclose(
        raw_p[eligible].to_numpy(float),
        (1.0 + raw_count[eligible].to_numpy(float)) / 1000.0,
        rtol=0.0,
        atol=1e-15,
        err_msg="raw permutation p-value does not use add-one / 999",
    )
    np.testing.assert_allclose(
        max_p[eligible].to_numpy(float),
        (1.0 + max_count[eligible].to_numpy(float)) / 1000.0,
        rtol=0.0,
        atol=1e-15,
        err_msg="maxT p-value does not use add-one / 999",
    )
    if max_p[eligible].lt(raw_p[eligible] - 1e-15).any():
        raise ValueError("maxT adjusted p-value is smaller than raw p-value")
    if observed_t[eligible].isna().any():
        raise AssertionError("eligible Welch statistic unexpectedly missing")


def _validate_family_statistics(effects: pd.DataFrame) -> None:
    family_columns = ["target", "direction"]
    repeated_columns = [
        "eligible_hypotheses",
        "total_hypotheses",
        "observed_family_max_absolute_welch_t",
        "family_maxT_exceedance_count",
        "family_maxT_p_value",
        "permutations",
        "seed",
        "joint_membership_permutation",
        "family_bh_q_value",
        "family_bh_reject",
    ]
    if (
        effects.groupby(family_columns, sort=True, observed=True)[repeated_columns]
        .nunique(dropna=False)
        .ne(1)
        .any()
        .any()
    ):
        raise ValueError("family-level fields are not constant within family")
    families = effects.drop_duplicates(family_columns, keep="first").copy()
    if len(families) != EXPECTED_FAMILIES:
        raise ValueError("segment effects do not contain exactly 24 families")
    eligible_hypotheses = _integer_series(
        families["eligible_hypotheses"], name="family eligible hypotheses"
    )
    if not _integer_series(
        families["total_hypotheses"], name="family total hypotheses"
    ).eq(EXPECTED_HYPOTHESES).all():
        raise ValueError("family total hypothesis count drift")
    family_exceedance = _integer_series(
        families["family_maxT_exceedance_count"],
        name="family maxT exceedance count",
        maximum=EXPECTED_PERMUTATIONS,
    )
    family_count_by_key = {
        (str(row.target), str(row.direction)): int(count)
        for row, count in zip(
            families.itertuples(index=False), family_exceedance, strict=True
        )
    }
    eligible_count_by_key = {
        (str(row.target), str(row.direction)): int(count)
        for row, count in zip(
            families.itertuples(index=False), eligible_hypotheses, strict=True
        )
    }
    if not _integer_series(
        families["permutations"], name="family permutations"
    ).eq(EXPECTED_PERMUTATIONS).all():
        raise ValueError("family permutation count drift")
    if not _integer_series(families["seed"], name="family seed").eq(
        EXPECTED_SEED
    ).all():
        raise ValueError("family permutation seed drift")
    if not _strict_boolean(
        families["joint_membership_permutation"],
        name="joint membership permutation",
    ).all():
        raise ValueError("segment labels were not permuted jointly")

    for (target, direction), part in effects.groupby(
        family_columns, sort=True, observed=True
    ):
        support = _strict_boolean(
            part["direction_support_eligible"], name="direction support eligibility"
        )
        eligible_count = int(support.sum())
        if eligible_count < 1 or eligible_count_by_key[(str(target), str(direction))] != eligible_count:
            raise ValueError(f"eligible hypothesis count drift for {target}/{direction}")
        observed_max = float(
            pd.to_numeric(part.loc[support, "absolute_welch_t"]).max()
        )
        family_max = float(part["observed_family_max_absolute_welch_t"].iloc[0])
        if not np.isfinite(family_max) or abs(family_max - observed_max) > 1e-15:
            raise ValueError(f"observed family max drift for {target}/{direction}")
        maximizer = part.loc[
            support
            & pd.to_numeric(part["absolute_welch_t"]).sub(observed_max).abs().le(1e-15)
        ]
        family_count = family_count_by_key[(str(target), str(direction))]
        max_counts = pd.to_numeric(maximizer["maxT_exceedance_count"])
        if not max_counts.eq(family_count).all():
            raise ValueError(f"family maxT count differs from maximizer for {target}/{direction}")
        family_p = float(part["family_maxT_p_value"].iloc[0])
        expected_p = (1.0 + family_count) / 1000.0
        if not np.isfinite(family_p) or abs(family_p - expected_p) > 1e-15:
            raise ValueError(f"family maxT p-value drift for {target}/{direction}")

    family_p = _numeric_series(
        families["family_maxT_p_value"],
        name="family maxT p-value",
        minimum=0.0,
        maximum=1.0,
    )
    expected_q = _bh_adjust(family_p.to_numpy(float))
    observed_q = _numeric_series(
        families["family_bh_q_value"],
        name="family BH q-value",
        minimum=0.0,
        maximum=1.0,
    )
    np.testing.assert_allclose(
        observed_q,
        expected_q,
        rtol=0.0,
        atol=1e-15,
        err_msg="family BH q-values differ from the exact 24-family adjustment",
    )
    observed_reject = _strict_boolean(
        families["family_bh_reject"], name="family BH reject"
    )
    if not observed_reject.eq(observed_q.le(BH_Q)).all():
        raise ValueError("family BH reject differs from q<=0.10")

    family_reject = effects.merge(
        families[[*family_columns, "family_bh_reject"]].rename(
            columns={"family_bh_reject": "_expected_family_bh_reject"}
        ),
        on=family_columns,
        how="left",
        validate="many_to_one",
    )["_expected_family_bh_reject"].astype(bool)
    support = _strict_boolean(
        effects["direction_support_eligible"], name="direction support eligibility"
    )
    max_p = pd.to_numeric(effects["maxT_adjusted_p_value"], errors="coerce")
    expected_survivor = support & max_p.le(BH_Q) & family_reject.to_numpy(bool)
    survivor = _strict_boolean(
        effects["core_maxT_survivor"], name="core maxT survivor"
    )
    if not survivor.eq(expected_survivor).all():
        raise ValueError("core survivor rule differs from support/maxT/family-BH")


def _same_nonzero_sign(left: pd.Series, right: pd.Series) -> pd.Series:
    left_numeric = pd.to_numeric(left, errors="coerce")
    right_numeric = pd.to_numeric(right, errors="coerce")
    return (
        left_numeric.notna()
        & right_numeric.notna()
        & np.isfinite(left_numeric)
        & np.isfinite(right_numeric)
        & left_numeric.mul(right_numeric).gt(0.0)
    )


def _validate_robustness(effects: pd.DataFrame) -> None:
    core_support = _strict_boolean(
        effects["direction_support_eligible"], name="direction support eligibility"
    )
    primary = pd.to_numeric(effects["segment_effect"], errors="coerce")
    dino_nonnull = effects["dino_segment_effect"].notna()
    rad_nonnull = effects["rad_segment_effect"].notna()
    dino = pd.to_numeric(effects["dino_segment_effect"], errors="coerce")
    rad = pd.to_numeric(effects["rad_segment_effect"], errors="coerce")
    if (
        dino[dino_nonnull].isna().any()
        or rad[rad_nonnull].isna().any()
        or not np.isfinite(dino[dino_nonnull]).all()
        or not np.isfinite(rad[rad_nonnull]).all()
    ):
        raise ValueError("DINO/Rad robustness effects contain nonnumeric/nonfinite values")
    if dino[core_support].isna().any() or rad[core_support].isna().any():
        raise ValueError("support-eligible row is missing DINO/Rad robustness effects")
    observed_component_sign = _strict_boolean(
        effects["dino_rad_same_sign"], name="DINO/Rad same-sign guardrail"
    )
    expected_component_sign = _same_nonzero_sign(dino, rad)
    if not observed_component_sign.eq(expected_component_sign).all():
        raise ValueError("DINO/Rad same-sign guardrail differs from component effects")

    high_n1 = _integer_series(
        effects["high_agreement_segment_report_groups"],
        name="high-agreement segment support",
    )
    high_n0 = _integer_series(
        effects["high_agreement_complement_report_groups"],
        name="high-agreement complement support",
    )
    high_eligible = _strict_boolean(
        effects["high_agreement_support_eligible"],
        name="high-agreement support eligibility",
    )
    expected_high_eligible = high_n1.ge(MINIMUM_REPORT_GROUPS_PER_ARM) & high_n0.ge(
        MINIMUM_REPORT_GROUPS_PER_ARM
    )
    if not high_eligible.eq(expected_high_eligible).all():
        raise ValueError("high-agreement eligibility differs from arm counts")
    direction_n1 = _integer_series(
        effects["direction_segment_report_groups"], name="direction segment support"
    )
    direction_n0 = _integer_series(
        effects["direction_complement_report_groups"],
        name="direction complement support",
    )
    if high_n1.gt(direction_n1).any() or high_n0.gt(direction_n0).any():
        raise ValueError("high-agreement support exceeds the direction cohort")
    high_effect = _conditional_numeric(
        effects["high_agreement_segment_effect"],
        high_eligible,
        name="high-agreement segment effect",
    )
    high_sign = _strict_boolean(
        effects["high_agreement_sign_retained"],
        name="high-agreement sign-retained guardrail",
    )
    expected_high_sign = high_eligible & _same_nonzero_sign(primary, high_effect)
    if not high_sign.eq(expected_high_sign).all():
        raise ValueError("high-agreement sign guardrail differs from effects/support")

    profile_count = _integer_series(
        effects["common_support_profile_count"],
        name="common-support profile count",
    )
    for index, (encoded, expected_count) in enumerate(
        zip(effects["common_support_profiles_json"], profile_count, strict=True)
    ):
        try:
            profiles = json.loads(str(encoded))
        except json.JSONDecodeError as error:
            raise ValueError("common-support profile JSON is invalid") from error
        if not isinstance(profiles, list) or any(
            not isinstance(value, str) or not value for value in profiles
        ):
            raise ValueError("common-support profiles must be nonempty strings")
        if profiles != sorted(set(profiles)) or len(profiles) != int(expected_count):
            raise ValueError(
                f"common-support profile identity/count drift at row {index}"
            )
    common_n1 = _integer_series(
        effects["common_support_segment_report_groups"],
        name="common-support segment support",
    )
    common_n0 = _integer_series(
        effects["common_support_complement_report_groups"],
        name="common-support complement support",
    )
    common_eligible = _strict_boolean(
        effects["common_support_eligible"], name="common-support eligibility"
    )
    expected_common_eligible = common_n1.ge(
        MINIMUM_REPORT_GROUPS_PER_ARM
    ) & common_n0.ge(MINIMUM_REPORT_GROUPS_PER_ARM)
    if not common_eligible.eq(expected_common_eligible).all():
        raise ValueError("common-support eligibility differs from arm counts")
    if common_n1.gt(direction_n1).any() or common_n0.gt(direction_n0).any():
        raise ValueError("common support exceeds the direction cohort")
    common_effect = _conditional_numeric(
        effects["common_support_segment_effect"],
        common_eligible,
        name="common-support segment effect",
    )
    common_sign = _strict_boolean(
        effects["common_support_sign_retained"],
        name="common-support sign-retained guardrail",
    )
    expected_common_sign = common_eligible & _same_nonzero_sign(
        primary, common_effect
    )
    if not common_sign.eq(expected_common_sign).all():
        raise ValueError("common-support sign guardrail differs from effects/support")


def _validate_pre_lopo_boundary(effects: pd.DataFrame) -> int:
    survivor = _strict_boolean(
        effects["core_maxT_survivor"], name="core maxT survivor"
    )
    survivor_count = int(survivor.sum())
    expected_lopo = np.where(
        survivor,
        "PENDING_REQUIRED_CORE_SURVIVOR",
        "NOT_APPLICABLE_NOT_CORE_SURVIVOR",
    )
    if not np.array_equal(effects["lopo_status"].astype(str), expected_lopo):
        raise ValueError("pre-LOPO status differs from core-survivor identity")
    expected_proxy = np.where(
        survivor,
        "CORE_SURVIVOR_PENDING_LOPO",
        "DIRECTIONAL_ONLY_NOT_CORE_SURVIVOR",
    )
    if not np.array_equal(effects["proxy_segment_status"].astype(str), expected_proxy):
        raise ValueError("proxy segment status may only inherit/demote core survivors")
    for column in ("lopo_profiles_required", "lopo_profiles_completed"):
        values = pd.to_numeric(effects[column], errors="coerce")
        if values[~survivor].isna().any() or not values[~survivor].eq(0).all():
            raise ValueError(f"{column} must be zero for non-survivors")
        if values[survivor].notna().any():
            raise ValueError(f"{column} must remain pending for core survivors")
    lopo_sign = effects["lopo_all_effects_same_sign"]
    if lopo_sign[~survivor].isna().any() or not lopo_sign[~survivor].isin(
        [False, 0]
    ).all():
        raise ValueError("LOPO sign must be false for non-survivors")
    if lopo_sign[survivor].notna().any():
        raise ValueError("LOPO sign must remain pending for core survivors")
    return survivor_count


def validate_segment_effect_frame(
    effects: pd.DataFrame,
    hypotheses: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Validate an already-loaded pre-Gold segment effects frame."""

    if any("gold" in str(column).casefold() for column in effects.columns):
        raise ValueError("segment effects contain forbidden Gold columns")
    missing = sorted(set(SEGMENT_EFFECT_COLUMNS).difference(effects.columns))
    if missing:
        raise ValueError(f"segment effects missing columns: {missing}")
    extras = sorted(set(effects.columns).difference(SEGMENT_EFFECT_COLUMNS))
    if extras:
        raise ValueError(f"segment effects contain non-contract columns: {extras}")
    if len(effects) != EXPECTED_ROWS:
        raise ValueError("segment effects row count is not 127 x 24")
    result = effects.copy()
    _validate_identity(result, hypotheses)
    _validate_core_statistics(result)
    _validate_family_statistics(result)
    _validate_robustness(result)
    _validate_pre_lopo_boundary(result)
    return result


def verify_segment_effects(
    universe_path: str | Path,
    effects_path: str | Path,
    *,
    expected_universe_sha256: str = EXPECTED_UNIVERSE_SHA256,
) -> SegmentEffectsVerification:
    """Verify the exact frozen universe and future pre-Gold effect artifact."""

    universe_file = Path(universe_path).resolve()
    effects_file = Path(effects_path).resolve()
    _, hypotheses = _validate_universe(
        universe_file, expected_universe_sha256=expected_universe_sha256
    )
    if effects_file.name != "segment_effects_v3.csv.gz":
        raise ValueError("segment effect filename must be segment_effects_v3.csv.gz")
    if not effects_file.is_file():
        raise FileNotFoundError(effects_file)
    header = pd.read_csv(effects_file, nrows=0).columns.tolist()
    forbidden = [column for column in header if "gold" in str(column).casefold()]
    if forbidden:
        raise ValueError(f"segment effects contain forbidden Gold columns: {forbidden}")
    effects = pd.read_csv(effects_file)
    verified = validate_segment_effect_frame(effects, hypotheses)
    survivor_count = _validate_pre_lopo_boundary(verified)
    diagnostics = {
        "status": (
            "PASS_PRE_GOLD_SEGMENT_EFFECTS_ZERO_SURVIVORS_LOPO_NOT_REQUIRED"
            if survivor_count == 0
            else "BLOCKED_PRE_GOLD_CORE_SURVIVORS_REQUIRE_EXACT_LOPO"
        ),
        "segment_universe_sha256": _sha256_file(universe_file),
        "hypotheses_sha256": EXPECTED_HYPOTHESES_SHA256,
        "segment_effects_sha256": _sha256_file(effects_file),
        "rows": int(len(verified)),
        "hypotheses": EXPECTED_HYPOTHESES,
        "families": EXPECTED_FAMILIES,
        "permutations_per_family": EXPECTED_PERMUTATIONS,
        "core_maxT_survivors": survivor_count,
        "gold_columns_loaded": [],
        "gold_outcomes_loaded": False,
    }
    return SegmentEffectsVerification(effects=verified, diagnostics=diagnostics)


def _assert_frame_equal(left: pd.DataFrame, right: pd.DataFrame, *, name: str) -> None:
    try:
        pd.testing.assert_frame_equal(
            left.reset_index(drop=True),
            right.reset_index(drop=True),
            check_dtype=False,
            # CSV round-tripping can represent the same frozen decimal as
            # 0.418 or 0.4179999999999999.  The artifact bytes are verified
            # separately below; this recomputation check therefore permits
            # only sub-ulp parsing noise, never a material numeric change.
            check_exact=False,
            rtol=0.0,
            atol=1e-15,
        )
    except AssertionError as error:
        raise ValueError(f"{name} differs from independent recomputation") from error


def verify_segment_effects_determinism(
    root: str | Path,
    canonical_directory: str | Path,
    reference_directories: Sequence[str | Path],
    *,
    expected_universe_sha256: str = EXPECTED_UNIVERSE_SHA256,
) -> SegmentEffectsDeterminismVerification:
    """Verify canonical segment outputs against exactly two independent runs."""

    root_path = Path(root).resolve()
    canonical = Path(canonical_directory)
    canonical = canonical.resolve() if canonical.is_absolute() else (root_path / canonical).resolve()
    references = [
        path.resolve() if (path := Path(value)).is_absolute() else (root_path / path).resolve()
        for value in reference_directories
    ]
    if len(references) != 2 or len(set(references)) != 2:
        raise ValueError("exactly two distinct segment reference directories are required")
    run_boundary = (root_path / ".codex/runs/rsna-knee-report-reliability-v3").resolve()
    if not canonical.is_relative_to(run_boundary) or any(
        not path.is_relative_to(run_boundary) for path in references
    ):
        raise ValueError("segment determinism directories must remain inside the frozen run")
    universe_path = run_boundary / "SEGMENT_UNIVERSE.json"

    verifications: list[SegmentEffectsVerification] = []
    hashes_by_directory: dict[str, dict[str, str]] = {}
    for directory in [canonical, *references]:
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        verification = verify_segment_effects(
            universe_path,
            directory / "segment_effects_v3.csv.gz",
            expected_universe_sha256=expected_universe_sha256,
        )
        verifications.append(verification)
        hashes: dict[str, str] = {}
        for filename in SEGMENT_CORE_OUTPUT_FILENAMES:
            path = directory / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            hashes[filename] = _sha256_file(path)
        hashes_by_directory[str(directory.relative_to(root_path))] = hashes

        effects = verification.effects
        family = pd.read_csv(directory / "segment_family_maxT_v3.csv")
        if family.columns.tolist() != list(SEGMENT_FAMILY_COLUMNS):
            raise ValueError("segment family schema/order drift")
        expected_family = effects.loc[:, list(SEGMENT_FAMILY_COLUMNS)].drop_duplicates(
            ["target", "direction"], keep="first"
        ).reset_index(drop=True)
        _assert_frame_equal(family, expected_family, name="segment family table")

        shortlist = pd.read_csv(directory / "segment_proxy_shortlist_v3.csv")
        missing = sorted(set(SEGMENT_EFFECT_COLUMNS).difference(shortlist.columns))
        extras = sorted(set(shortlist.columns).difference(SEGMENT_EFFECT_COLUMNS))
        if missing or extras:
            raise ValueError(
                f"segment core shortlist schema drift: missing={missing}, extras={extras}"
            )
        expected_shortlist = effects.loc[
            _strict_boolean(effects["core_maxT_survivor"], name="core maxT survivor"),
            list(SEGMENT_EFFECT_COLUMNS),
        ].reset_index(drop=True)
        observed_shortlist = shortlist.loc[:, list(SEGMENT_EFFECT_COLUMNS)]
        _assert_frame_equal(
            observed_shortlist,
            expected_shortlist,
            name="segment core shortlist",
        )

        audit_path = directory / "SEGMENT_EFFECTS_AUDIT.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if not isinstance(audit, dict):
            raise ValueError("segment effects audit must be an object")
        survivors = int(verification.diagnostics["core_maxT_survivors"])
        expected_status = (
            "PASS_SEGMENT_CORE_ZERO_SURVIVORS_LOPO_NOT_REQUIRED_GOLD_NOT_LOADED"
            if survivors == 0
            else "BLOCKED_SEGMENT_CORE_SURVIVORS_REQUIRE_EXACT_LOPO"
        )
        if audit.get("status") != expected_status:
            raise ValueError("segment effects audit terminal status drift")
        for key in ("gold_columns_loaded", "gold_files_loaded", "is_gold_columns_loaded"):
            if audit.get(key) != []:
                raise ValueError(f"segment effects audit does not attest empty {key}")
        expected_outputs = {
            filename: hashes[filename]
            for filename in SEGMENT_CORE_OUTPUT_FILENAMES
            if filename != "SEGMENT_EFFECTS_AUDIT.json"
        }
        if audit.get("output_hashes") != expected_outputs:
            raise ValueError("segment effects audit output hash map drift")

    canonical_hashes = hashes_by_directory[str(canonical.relative_to(root_path))]
    for directory in references:
        if hashes_by_directory[str(directory.relative_to(root_path))] != canonical_hashes:
            raise ValueError("segment core outputs are not byte-identical across runs")
    diagnostics = {
        "status": (
            "PASS_INDEPENDENT_SEGMENT_EFFECTS_EXACT_TWO_RUNS_PENDING_LOPO_GOLD_NOT_LOADED"
            if int(verifications[0].diagnostics["core_maxT_survivors"]) > 0
            else "PASS_INDEPENDENT_SEGMENT_EFFECTS_EXACT_TWO_RUNS_LOPO_NOT_REQUIRED_GOLD_NOT_LOADED"
        ),
        "contract_version": EXPECTED_CONTRACT_VERSION,
        "contract_semantic_hash": EXPECTED_CONTRACT_SEMANTIC_HASH,
        "canonical_directory": str(canonical.relative_to(root_path)),
        "reference_directories": [str(path.relative_to(root_path)) for path in references],
        "core_output_hashes": canonical_hashes,
        "segment_gate_diagnostics": dict(verifications[0].diagnostics),
        "gold_columns_loaded": [],
        "gold_files_loaded": [],
        "is_gold_columns_loaded": [],
        "gold_outcomes_loaded": False,
        "weights_or_labels_modified": False,
    }
    return SegmentEffectsDeterminismVerification(diagnostics=diagnostics)


__all__ = [
    "CORE_REQUIRED_COLUMNS",
    "EXPECTED_HYPOTHESES_SHA256",
    "EXPECTED_UNIVERSE_SHA256",
    "LOPO_REQUIRED_COLUMNS",
    "ROBUSTNESS_REQUIRED_COLUMNS",
    "SEGMENT_EFFECT_COLUMNS",
    "SEGMENT_CORE_OUTPUT_FILENAMES",
    "SEGMENT_FAMILY_COLUMNS",
    "SegmentEffectsDeterminismVerification",
    "SegmentEffectsVerification",
    "validate_segment_effect_frame",
    "verify_segment_effects",
    "verify_segment_effects_determinism",
]
