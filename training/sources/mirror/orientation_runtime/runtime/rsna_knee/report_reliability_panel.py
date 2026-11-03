"""Deterministic presentation rules for the report-reliability panel.

The functions in this module do not fit models and do not read project files.
They translate already frozen proxy scores and image-family evidence into the
contract's review-priority fields.  The direction-specific proxy percentile is
fixed before Gold and is never changed by image-family disagreement.  The
agreement guardrail affects evidence strength and routing only; it cannot
increase, decrease, or suppress the frozen report-review percentile.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


REASON_PRECEDENCE: tuple[str, ...] = (
    "TEACHER_ABSTAIN",
    "IMAGE_OOF_INVALID",
    "IMAGE_FAMILY_DISAGREEMENT",
    "EXTERNAL_LLM_AUDIT_DISAGREEMENT",
    "TEACHER_COMPONENT_SELF_CONFLICT",
    "LLM_SILENCE_INSTABILITY",
    "TEACHER_NEGATIVE_IMAGE_CONSENSUS_HIGH",
    "TEACHER_POSITIVE_IMAGE_CONSENSUS_LOW",
)

ALLOWED_DIRECTION_MODEL_STATUSES: frozenset[str] = frozenset(
    {"PASS_INCREMENT", "DIRECTIONAL_ONLY", "INSUFFICIENT_SUPPORT"}
)

ALLOWED_IMAGE_STATES: frozenset[str] = frozenset(
    {
        "BOTH_HIGH",
        "BOTH_LOW",
        "DINO_ONLY_EXTREME",
        "RAD_ONLY_EXTREME",
        "MID_AMBIGUOUS",
        "OOF_INVALID",
    }
)


def _require(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"panel input missing columns: {missing}")


def _strict_boolean(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column]
    if values.isna().any() or not values.isin([True, False, 0, 1]).all():
        raise ValueError(f"{column} must contain only non-null booleans")
    return values.astype(bool)


def _optional_unit_interval(values: pd.Series, *, name: str) -> pd.Series:
    nonnull = values.notna()
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric[nonnull].isna().any() or not np.isfinite(numeric[nonnull]).all():
        raise ValueError(f"{name} must be numeric and finite when populated")
    if ((numeric[nonnull] < 0.0) | (numeric[nonnull] > 1.0)).any():
        raise ValueError(f"{name} lies outside [0, 1]")
    return numeric


def image_evidence_strength(
    *, valid: bool, dino_valid: bool, rad_valid: bool, disagreement: float
) -> str:
    """Return the frozen image-evidence tier for one row."""

    if not valid:
        if rad_valid:
            return "LOW"
        return "UNAVAILABLE"
    if not np.isfinite(disagreement):
        return "UNAVAILABLE"
    if dino_valid and rad_valid and disagreement <= 0.10 + 1e-12:
        return "HIGH"
    if dino_valid and rad_valid and disagreement <= 0.20 + 1e-12:
        return "MODERATE"
    return "LOW"


def risk_strength(value: float) -> str:
    """Map a frozen percentile to a descriptive tier, never a probability."""

    if pd.isna(value):
        return "NOT_APPLICABLE"
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
        raise ValueError("risk percentile must be finite within [0, 1] or null")
    if numeric >= 0.95:
        return "VERY_HIGH"
    if numeric >= 0.90:
        return "HIGH"
    if numeric >= 0.80:
        return "MODERATE"
    return "LOW"


def _reason_codes(row: pd.Series) -> tuple[str, ...]:
    teacher = str(row["teacher_state"])
    codes: list[str] = []
    if teacher == "ABSTAIN":
        codes.append("TEACHER_ABSTAIN")
    if not bool(row["image_oof_valid"]):
        codes.append("IMAGE_OOF_INVALID")
    disagreement = row["image_family_disagreement"]
    if (
        bool(row["image_oof_valid"])
        and pd.notna(disagreement)
        and float(disagreement) > 0.10 + 1e-12
    ):
        codes.append("IMAGE_FAMILY_DISAGREEMENT")
    if bool(row["external_audit_disagreement"]):
        codes.append("EXTERNAL_LLM_AUDIT_DISAGREEMENT")
    if bool(row["teacher_component_self_conflict"]):
        codes.append("TEACHER_COMPONENT_SELF_CONFLICT")
    if bool(row["family_silence_instability"]):
        codes.append("LLM_SILENCE_INSTABILITY")
    high_omission = teacher == "NEGATIVE" and str(row["image_state"]) == "BOTH_HIGH"
    high_contradiction = teacher == "POSITIVE" and str(row["image_state"]) == "BOTH_LOW"
    if high_omission:
        codes.append("TEACHER_NEGATIVE_IMAGE_CONSENSUS_HIGH")
    if high_contradiction:
        codes.append("TEACHER_POSITIVE_IMAGE_CONSENSUS_LOW")
    if bool(row["external_audit_disagreement"]) and (high_omission or high_contradiction):
        codes.append("COMPOUND_CONFLICT")
    ordered = [code for code in REASON_PRECEDENCE if code in codes]
    if "COMPOUND_CONFLICT" in codes:
        ordered.append("COMPOUND_CONFLICT")
    return tuple(ordered)


def build_proxy_presentation_fields(
    frame: pd.DataFrame,
    *,
    model_status_by_direction: Mapping[tuple[str, str], str],
) -> pd.DataFrame:
    """Build frozen proxy-only status, tier and reason fields.

    ``report_risk_percentile_proxy_reference`` is the direction-specific
    morphology rank and is copied without modification into the applicable
    output risk fields.
    ``guardrailed_review_risk_percentile`` is a diagnostic integrity field: it
    must equal the raw rank for decisive high-agreement rows and be null
    otherwise.  It never replaces the frozen output rank.
    """

    required = (
        "StudyInstanceUID",
        "target",
        "teacher_state",
        "family_state_summary",
        "other_positive_count",
        "other_negative_count",
        "other_abstain_count",
        "direction",
        "report_risk_percentile_proxy_reference",
        "guardrailed_review_risk_percentile",
        "image_oof_valid",
        "dino_oof_valid",
        "rad_oof_valid",
        "image_consensus_rank",
        "image_family_disagreement",
        "rad_internal_dispersion",
        "image_state",
        "external_audit_disagreement",
        "teacher_component_self_conflict",
        "family_silence_instability",
    )
    _require(frame, required)
    result = frame.copy()
    if result["StudyInstanceUID"].isna().any() or result["target"].isna().any():
        raise ValueError("panel keys contain null values")
    if result["StudyInstanceUID"].astype(str).eq("").any() or result[
        "target"
    ].astype(str).eq("").any():
        raise ValueError("panel keys contain empty values")
    if result.duplicated(["StudyInstanceUID", "target"]).any():
        raise ValueError("panel keys are not unique")

    teacher = result["teacher_state"].astype(str)
    direction = result["direction"].astype(str)
    expected_direction = teacher.map(
        {
            "NEGATIVE": "omission",
            "POSITIVE": "contradiction",
            "ABSTAIN": "not_applicable",
        }
    )
    if expected_direction.isna().any() or not expected_direction.eq(direction).all():
        raise ValueError("teacher state and direction are inconsistent")

    raw = _optional_unit_interval(
        result["report_risk_percentile_proxy_reference"],
        name="proxy-reference risk percentile",
    )
    guarded = _optional_unit_interval(
        result["guardrailed_review_risk_percentile"],
        name="guardrailed review risk percentile",
    )
    decisive = teacher.isin(["NEGATIVE", "POSITIVE"])
    if raw[decisive].isna().any() or raw[~decisive].notna().any():
        raise ValueError("raw risk applicability differs from teacher polarity")
    image_valid = _strict_boolean(result, "image_oof_valid")
    dino_valid = _strict_boolean(result, "dino_oof_valid")
    rad_valid = _strict_boolean(result, "rad_oof_valid")
    if not image_valid.eq(dino_valid & rad_valid).all():
        raise ValueError("image_oof_valid differs from DINO/Rad validity")
    disagreement = _optional_unit_interval(
        result["image_family_disagreement"], name="image family disagreement"
    )
    consensus = _optional_unit_interval(
        result["image_consensus_rank"], name="image consensus rank"
    )
    _optional_unit_interval(
        result["rad_internal_dispersion"], name="Rad internal dispersion"
    )
    if disagreement.notna().ne(image_valid).any():
        raise ValueError("image disagreement applicability differs from OOF validity")
    if consensus.notna().ne(image_valid).any():
        raise ValueError("image consensus applicability differs from OOF validity")
    states = result["image_state"].astype(str)
    if not states.isin(ALLOWED_IMAGE_STATES).all():
        raise ValueError("image_state contains an unknown value")
    if states.eq("OOF_INVALID").ne(~image_valid).any():
        raise ValueError("OOF_INVALID state differs from image validity")

    high_agreement = image_valid & disagreement.le(0.10 + 1e-12)
    if (
        states.eq("BOTH_HIGH")
        & (~high_agreement | consensus.lt(0.90 - 1e-12))
    ).any():
        raise ValueError("BOTH_HIGH escaped the frozen extreme/agreement gate")
    if (
        states.eq("BOTH_LOW")
        & (~high_agreement | consensus.gt(0.10 + 1e-12))
    ).any():
        raise ValueError("BOTH_LOW escaped the frozen extreme/agreement gate")

    guarded_applicable = decisive & high_agreement
    if not guarded.notna().eq(guarded_applicable).all():
        raise ValueError("guardrailed risk applicability differs from the agreement gate")
    if not np.allclose(
        guarded[guarded.notna()].to_numpy(float),
        raw[guarded.notna()].to_numpy(float),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("guardrailed risk differs from the frozen raw percentile")

    result["report_risk_percentile_proxy_reference"] = raw
    result["omission_like_risk_percentile"] = np.where(
        teacher.eq("NEGATIVE"), raw, np.nan
    )
    result["contradiction_like_risk_percentile"] = np.where(
        teacher.eq("POSITIVE"), raw, np.nan
    )
    result["label_review_risk_percentile"] = raw
    result["image_evidence_strength"] = [
        image_evidence_strength(
            valid=bool(row.image_oof_valid),
            dino_valid=bool(row.dino_oof_valid),
            rad_valid=bool(row.rad_oof_valid),
            disagreement=float(row.image_family_disagreement)
            if pd.notna(row.image_family_disagreement)
            else np.nan,
        )
        for row in result.itertuples(index=False)
    ]
    statuses: list[str] = []
    for row in result.itertuples(index=False):
        if row.direction == "not_applicable":
            statuses.append("NOT_APPLICABLE")
        else:
            key = (str(row.target), str(row.direction))
            if key not in model_status_by_direction:
                raise ValueError(f"missing model status for {key}")
            status = str(model_status_by_direction[key])
            if status not in ALLOWED_DIRECTION_MODEL_STATUSES:
                raise ValueError(f"invalid model status for {key}: {status}")
            statuses.append(status)
    result["proxy_model_status"] = statuses
    reasons = result.apply(_reason_codes, axis=1)
    result["candidate_reason_codes"] = reasons.map(lambda values: "|".join(values))
    has_high_extreme = reasons.map(
        lambda values: bool(
            {"TEACHER_NEGATIVE_IMAGE_CONSENSUS_HIGH", "TEACHER_POSITIVE_IMAGE_CONSENSUS_LOW"}
            .intersection(values)
        )
    )
    result["proxy_evidence_tier"] = np.where(
        result["external_audit_disagreement"].astype(bool) & ~has_high_extreme,
        "TEXT_INTERNAL",
        np.where(raw.notna() | has_high_extreme, "PROXY_ONLY", "UNRESOLVED"),
    )
    result["external_llm_audit_disagreement"] = _strict_boolean(
        result, "external_audit_disagreement"
    )
    result["llm_silence_instability"] = _strict_boolean(
        result, "family_silence_instability"
    )
    result["risk_strength"] = raw.map(risk_strength)
    result["candidate_weight_status"] = (
        "UNAPPLIED_REQUIRES_SEPARATE_GROUPED_OOF_RETRAINING"
    )
    return result
