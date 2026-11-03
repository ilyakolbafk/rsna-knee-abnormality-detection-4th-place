"""Deterministic, local-only finalization of report-reliability v3.

The finalizer is deliberately downstream of the immutable pre-Gold freeze and
the append-only Gold58 evaluation.  It cannot fit a model, select a feature,
change a proxy rank, or turn Gold into a routing rule.  Its policy output is a
manual-review queue made only from the already frozen risk tier, image context,
and proxy model status.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import ID_COLUMN, TARGET_COLUMNS
from .report_reliability_panel import risk_strength as frozen_risk_strength


EXPECTED_PROXY_FREEZE_MANIFEST_SHA256 = (
    "24c91b2ca80ef4fce71ef6a87af4147c9beb647c14b4bd662368100cda1f5218"
)
EXPECTED_PROXY_FREEZE_BINDING_PLAN_SHA256 = (
    "9f874fab17ab2e6b60fd18c88c20308cd7d15190752737b981a2651d480cdfa4"
)

POLICY_SOURCE_COLUMNS: tuple[str, ...] = (
    ID_COLUMN,
    "target",
    "teacher_state",
    "image_state",
    "image_evidence_strength",
    "proxy_model_status",
    "candidate_reason_codes",
    "proxy_evidence_tier",
    "risk_strength",
    "label_review_risk_percentile",
    "report_risk_percentile_proxy_reference",
)

POLICY_COLUMNS: tuple[str, ...] = (
    ID_COLUMN,
    "target",
    "manual_review_priority",
    "manual_review_route",
    "risk_strength",
    "label_review_risk_percentile",
    "report_risk_percentile_proxy_reference",
    "teacher_state",
    "image_state",
    "image_evidence_strength",
    "image_review_context",
    "proxy_model_status",
    "model_evidence_context",
    "candidate_reason_codes",
    "proxy_evidence_tier",
    "policy_action",
    "selection_basis",
    "gold_used_for_routing",
    "automatic_label_change",
    "automatic_confidence_change",
    "automatic_weight_change",
    "requires_blinded_manual_report_image_adjudication",
)

RANKING_REQUIRED_COLUMNS: tuple[str, ...] = (
    "scope",
    "level",
    "status",
    "statistic",
    "point",
    "bootstrap_ci_lower",
    "bootstrap_ci_upper",
    "bootstrap_valid_replicates",
    "bootstrap_requested_replicates",
    "studies",
    "report_groups",
    "cells",
    "events",
    "agreements",
    "event_prevalence",
    "claim_strength",
)

GOLD_ANNOTATION_COLUMNS: tuple[str, ...] = (
    ID_COLUMN,
    "target",
    "gold",
    "gold_descriptive_reason_codes",
    "gold_descriptive_evidence_tier",
    "teacher_reference_disagreement",
    "attribution_status",
    "automatic_label_or_weight_action",
)

_ALLOWED_IMAGE_STATES = frozenset(
    {
        "BOTH_HIGH",
        "BOTH_LOW",
        "DINO_ONLY_EXTREME",
        "RAD_ONLY_EXTREME",
        "MID_AMBIGUOUS",
        "OOF_INVALID",
    }
)
_ALLOWED_IMAGE_STRENGTHS = frozenset({"HIGH", "MODERATE", "LOW", "UNAVAILABLE"})
_ALLOWED_MODEL_STATUSES = frozenset(
    {"PASS_INCREMENT", "DIRECTIONAL_ONLY", "INSUFFICIENT_SUPPORT", "NOT_APPLICABLE"}
)
_IMAGE_CONTEXT = {
    "BOTH_HIGH": "IMAGE_FAMILIES_AGREE_HIGH",
    "BOTH_LOW": "IMAGE_FAMILIES_AGREE_LOW",
    "DINO_ONLY_EXTREME": "DINO_ONLY_EXTREME",
    "RAD_ONLY_EXTREME": "RAD_ONLY_EXTREME",
    "MID_AMBIGUOUS": "IMAGE_MID_OR_FAMILY_DISAGREEMENT",
    "OOF_INVALID": "IMAGE_OOF_INVALID",
}
_MODEL_CONTEXT = {
    "PASS_INCREMENT": "PROXY_INCREMENT_GATE_PASSED",
    "DIRECTIONAL_ONLY": "PROXY_DIRECTIONAL_ONLY",
    "INSUFFICIENT_SUPPORT": "PROXY_INSUFFICIENT_SUPPORT",
    "NOT_APPLICABLE": "PROXY_NOT_APPLICABLE",
}
_PRIORITY = {"VERY_HIGH": "P1", "HIGH": "P2"}

_TARGET_SUPPORT: Mapping[str, tuple[int, int]] = {
    "ACL": (58, 7),
    "Baker's": (58, 6),
    "Contusion": (58, 15),
    "Effusion": (58, 16),
    "Fracture": (58, 11),
    "Lateral Meniscus": (57, 12),
    "Lateral OA": (58, 10),
    "MCL": (57, 8),
    "Medial Meniscus": (55, 6),
    "Medial OA": (58, 8),
    "PF OA": (58, 12),
    "Synovitis": (58, 18),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def columns_sha256(columns: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(str(value) for value in columns).encode("utf-8")).hexdigest()


def _require(frame: pd.DataFrame, columns: Sequence[str], *, name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def _strict_unit_interval(values: pd.Series, *, name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError(f"{name} must be finite numeric values")
    if ((numeric < 0.0) | (numeric > 1.0)).any():
        raise ValueError(f"{name} lies outside [0, 1]")
    return numeric.astype(float)


def build_policy_shortlist(panel: pd.DataFrame) -> pd.DataFrame:
    """Build the frozen-tier manual-review queue without consulting Gold.

    The only selection predicate is membership in the already frozen
    ``HIGH``/``VERY_HIGH`` risk tiers.  Image and model fields are copied into
    deterministic context labels; neither can add/remove a row or modify its
    rank.  Gold/post-freeze columns are intentionally absent from the read
    allowlist and from the output schema.
    """

    _require(panel, POLICY_SOURCE_COLUMNS, name="complete frozen panel")
    source = panel.loc[:, list(POLICY_SOURCE_COLUMNS)].copy()
    if source[[ID_COLUMN, "target"]].isna().any().any():
        raise ValueError("complete frozen panel contains null keys")
    if source.duplicated([ID_COLUMN, "target"]).any():
        raise ValueError("complete frozen panel contains duplicate keys")
    if not source["target"].isin(TARGET_COLUMNS).all():
        raise ValueError("complete frozen panel contains an unknown target")

    states = source["teacher_state"].astype(str)
    if not states.isin(["NEGATIVE", "POSITIVE", "ABSTAIN"]).all():
        raise ValueError("complete frozen panel contains an invalid teacher state")
    image_states = source["image_state"].astype(str)
    if not image_states.isin(_ALLOWED_IMAGE_STATES).all():
        raise ValueError("complete frozen panel contains an invalid image state")
    image_strength = source["image_evidence_strength"].astype(str)
    if not image_strength.isin(_ALLOWED_IMAGE_STRENGTHS).all():
        raise ValueError("complete frozen panel contains an invalid image evidence strength")
    model_status = source["proxy_model_status"].astype(str)
    if not model_status.isin(_ALLOWED_MODEL_STATUSES).all():
        raise ValueError("complete frozen panel contains an invalid proxy model status")

    tier = source["risk_strength"].astype(str)
    allowed_tiers = {"VERY_HIGH", "HIGH", "MODERATE", "LOW", "NOT_APPLICABLE"}
    if not tier.isin(allowed_tiers).all():
        raise ValueError("complete frozen panel contains an invalid risk tier")
    risk = pd.to_numeric(source["label_review_risk_percentile"], errors="coerce")
    proxy_reference = pd.to_numeric(
        source["report_risk_percentile_proxy_reference"], errors="coerce"
    )
    decisive = states.isin(["NEGATIVE", "POSITIVE"])
    if risk.notna().ne(decisive).any() or proxy_reference.notna().ne(decisive).any():
        raise ValueError("risk applicability differs from teacher state")
    if decisive.any():
        risk.loc[decisive] = _strict_unit_interval(
            risk.loc[decisive], name="label review risk percentile"
        )
        proxy_reference.loc[decisive] = _strict_unit_interval(
            proxy_reference.loc[decisive], name="proxy-reference percentile"
        )
    if not np.allclose(
        risk.loc[decisive].to_numpy(float),
        proxy_reference.loc[decisive].to_numpy(float),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("frozen output rank differs from proxy-reference percentile")
    expected_tier = risk.map(frozen_risk_strength)
    if not expected_tier.eq(tier).all():
        raise ValueError("risk tier differs from the frozen percentile rule")

    selected = source.loc[tier.isin(_PRIORITY)].copy()
    if selected.empty:
        raise ValueError("frozen HIGH/VERY_HIGH manual-review queue is empty")
    if not selected["teacher_state"].isin(["NEGATIVE", "POSITIVE"]).all():
        raise ValueError("an abstain row escaped into the manual-review queue")

    selected["manual_review_priority"] = selected["risk_strength"].map(_PRIORITY)
    selected["image_review_context"] = selected["image_state"].map(_IMAGE_CONTEXT)
    selected["model_evidence_context"] = selected["proxy_model_status"].map(_MODEL_CONTEXT)
    if selected[
        ["manual_review_priority", "image_review_context", "model_evidence_context"]
    ].isna().any().any():
        raise AssertionError("manual-review routing contains an unmapped frozen state")
    selected["manual_review_route"] = (
        selected["manual_review_priority"].astype(str)
        + "__"
        + selected["image_review_context"].astype(str)
        + "__"
        + selected["model_evidence_context"].astype(str)
    )
    selected["policy_action"] = "MANUAL_REVIEW_ONLY"
    selected["selection_basis"] = "FROZEN_RISK_TIER_HIGH_OR_VERY_HIGH_PRE_GOLD"
    selected["gold_used_for_routing"] = False
    selected["automatic_label_change"] = False
    selected["automatic_confidence_change"] = False
    selected["automatic_weight_change"] = False
    selected["requires_blinded_manual_report_image_adjudication"] = True

    priority_order = pd.Categorical(
        selected["manual_review_priority"], categories=["P1", "P2"], ordered=True
    )
    selected = selected.assign(
        __priority=priority_order,
        __risk_numeric=pd.to_numeric(
            selected["label_review_risk_percentile"], errors="raise"
        ),
    ).sort_values(
        ["__priority", "__risk_numeric", ID_COLUMN, "target"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    result = selected.drop(columns=["__priority", "__risk_numeric"]).loc[
        :, list(POLICY_COLUMNS)
    ].reset_index(drop=True)
    if result.duplicated([ID_COLUMN, "target"]).any():
        raise AssertionError("manual-review queue contains duplicate keys")
    if any(column in result for column in ("gold", "teacher_reference_disagreement")):
        raise AssertionError("Gold-derived columns escaped into the policy queue")
    return result


def validate_gold_ranking_metrics(metrics: pd.DataFrame) -> dict[str, Any]:
    """Validate the descriptive Gold metric artifact and return report facts."""

    _require(metrics, RANKING_REQUIRED_COLUMNS, name="Gold ranking metrics")
    frame = metrics.loc[:, list(RANKING_REQUIRED_COLUMNS)].copy()
    if frame.duplicated(["scope", "level", "statistic"]).any():
        raise ValueError("Gold ranking metrics contain duplicate metric keys")
    if not frame["claim_strength"].eq("DESCRIPTIVE_GOLD58_ONLY").all():
        raise ValueError("Gold ranking metrics overstate claim strength")
    requested = pd.to_numeric(frame["bootstrap_requested_replicates"], errors="raise")
    if not requested.eq(999).all():
        raise ValueError("Gold ranking metrics do not use the frozen 999 bootstraps")

    pooled = frame.loc[
        frame["scope"].eq("POOLED_TARGET_STRATIFIED") & frame["level"].eq("ALL")
    ].copy()
    expected_pooled = {
        "AUROC",
        "AVERAGE_PRECISION",
        "AVERAGE_PRECISION_OVER_PREVALENCE",
        "ERROR_CAPTURE@0.05",
        "ERROR_CAPTURE@0.10",
        "ERROR_CAPTURE@0.20",
        "TOP_RISK_ENRICHMENT@0.05",
        "TOP_RISK_ENRICHMENT@0.10",
        "TOP_RISK_ENRICHMENT@0.20",
    }
    if set(pooled["statistic"]) != expected_pooled or len(pooled) != len(expected_pooled):
        raise ValueError("Gold ranking metrics lack the exact pooled metric set")
    if len(frame) != 117:
        raise ValueError("Gold ranking metrics must contain exactly 117 rows")
    if not pooled["status"].eq("PASS_DESCRIPTIVE").all():
        raise ValueError("pooled Gold ranking metrics are not descriptive PASS")
    if not (
        pooled["studies"].eq(58).all()
        and pooled["report_groups"].eq(58).all()
        and pooled["cells"].eq(691).all()
        and pooled["events"].eq(129).all()
        and pooled["agreements"].eq(562).all()
    ):
        raise ValueError("pooled Gold ranking cohort identity drift")

    passed = frame.loc[frame["status"].eq("PASS_DESCRIPTIVE")].copy()
    for column in ("point", "bootstrap_ci_lower", "bootstrap_ci_upper"):
        values = pd.to_numeric(passed[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(float)).all():
            raise ValueError(f"Gold ranking metric {column} is not finite")
    if (passed["bootstrap_ci_lower"] > passed["point"]).any() or (
        passed["point"] > passed["bootstrap_ci_upper"]
    ).any():
        raise ValueError("Gold metric point lies outside its bootstrap CI")
    valid = pd.to_numeric(passed["bootstrap_valid_replicates"], errors="raise")
    if ((valid < 900) | (valid > 999)).any():
        raise ValueError("Gold metric has too few or too many valid bootstrap replicates")

    for level, (cells, events) in _TARGET_SUPPORT.items():
        rows = frame.loc[frame["scope"].eq("TARGET") & frame["level"].eq(level)]
        if set(rows["statistic"]) != expected_pooled or len(rows) != len(expected_pooled):
            raise ValueError(f"target metric set is not exact for {level}")
        expected_valid = 994 if level == "Medial Meniscus" else 999
        if not (
            rows["status"].eq("PASS_DESCRIPTIVE").all()
            and rows["cells"].eq(cells).all()
            and rows["studies"].eq(cells).all()
            and rows["report_groups"].eq(cells).all()
            and rows["events"].eq(events).all()
            and rows["agreements"].eq(cells - events).all()
            and rows["bootstrap_valid_replicates"].eq(expected_valid).all()
        ):
            raise ValueError(f"target support/bootstrap identity drift for {level}")

    target = frame.loc[
        frame["scope"].eq("TARGET")
        & frame["statistic"].isin(["AUROC", "AVERAGE_PRECISION"])
    ].copy()
    if set(target["level"]) != set(TARGET_COLUMNS) or len(target) != 2 * len(TARGET_COLUMNS):
        raise ValueError("target-level Gold AUROC/AP coverage is not exact")
    if not target["status"].eq("PASS_DESCRIPTIVE").all():
        raise ValueError("a target-level AUROC/AP row lacks descriptive support")

    def metric_row(scope: str, level: str, statistic: str) -> dict[str, Any]:
        row = frame.loc[
            frame["scope"].eq(scope)
            & frame["level"].eq(level)
            & frame["statistic"].eq(statistic)
        ]
        if len(row) != 1:
            raise ValueError(f"missing unique metric {scope}|{level}|{statistic}")
        return row.iloc[0].to_dict()

    target_rows: list[dict[str, Any]] = []
    for label in TARGET_COLUMNS:
        auc = metric_row("TARGET", label, "AUROC")
        ap = metric_row("TARGET", label, "AVERAGE_PRECISION")
        if int(auc["events"]) != int(ap["events"]) or int(auc["cells"]) != int(ap["cells"]):
            raise ValueError(f"target metric support mismatch for {label}")
        target_rows.append(
            {
                "target": label,
                "events": int(auc["events"]),
                "cells": int(auc["cells"]),
                "auroc": float(auc["point"]),
                "auroc_ci_lower": float(auc["bootstrap_ci_lower"]),
                "auroc_ci_upper": float(auc["bootstrap_ci_upper"]),
                "average_precision": float(ap["point"]),
                "average_precision_ci_lower": float(ap["bootstrap_ci_lower"]),
                "average_precision_ci_upper": float(ap["bootstrap_ci_upper"]),
            }
        )
    return {
        "pooled_auroc": metric_row("POOLED_TARGET_STRATIFIED", "ALL", "AUROC"),
        "pooled_average_precision": metric_row(
            "POOLED_TARGET_STRATIFIED", "ALL", "AVERAGE_PRECISION"
        ),
        "pooled_average_precision_over_prevalence": metric_row(
            "POOLED_TARGET_STRATIFIED",
            "ALL",
            "AVERAGE_PRECISION_OVER_PREVALENCE",
        ),
        "pooled_capture_0_05": metric_row(
            "POOLED_TARGET_STRATIFIED", "ALL", "ERROR_CAPTURE@0.05"
        ),
        "pooled_capture_0_10": metric_row(
            "POOLED_TARGET_STRATIFIED", "ALL", "ERROR_CAPTURE@0.10"
        ),
        "pooled_capture_0_20": metric_row(
            "POOLED_TARGET_STRATIFIED", "ALL", "ERROR_CAPTURE@0.20"
        ),
        "pooled_enrichment_0_10": metric_row(
            "POOLED_TARGET_STRATIFIED", "ALL", "TOP_RISK_ENRICHMENT@0.10"
        ),
        "pooled_enrichment_0_05": metric_row(
            "POOLED_TARGET_STRATIFIED", "ALL", "TOP_RISK_ENRICHMENT@0.05"
        ),
        "pooled_enrichment_0_20": metric_row(
            "POOLED_TARGET_STRATIFIED", "ALL", "TOP_RISK_ENRICHMENT@0.20"
        ),
        "target_rows": target_rows,
    }


def validate_gold_annotations(annotations: pd.DataFrame) -> dict[str, int]:
    """Validate the token-exact append-only Gold sidecar schema and cohort."""

    if list(annotations.columns) != list(GOLD_ANNOTATION_COLUMNS):
        raise ValueError("Gold annotation sidecar columns are not exact append-only schema")
    if annotations[[ID_COLUMN, "target"]].isna().any().any() or annotations.duplicated(
        [ID_COLUMN, "target"]
    ).any():
        raise ValueError("Gold annotation sidecar contains null or duplicate keys")
    if len(annotations) != 696 or annotations[ID_COLUMN].nunique() != 58:
        raise ValueError("Gold annotation sidecar is not the exact 58 x 12 cohort")
    if set(annotations["target"].astype(str)) != set(TARGET_COLUMNS):
        raise ValueError("Gold annotation sidecar target coverage is not exact")
    per_study = annotations.groupby(ID_COLUMN, sort=False)["target"].agg(
        lambda values: frozenset(str(value) for value in values)
    )
    if not per_study.map(lambda values: set(values) == set(TARGET_COLUMNS)).all():
        raise ValueError("Gold annotation sidecar is not a 12-target Cartesian cohort")
    gold = pd.to_numeric(annotations["gold"], errors="coerce")
    if gold.isna().any() or not gold.isin([0, 1]).all():
        raise ValueError("Gold annotation sidecar contains non-binary Gold values")
    if not annotations["automatic_label_or_weight_action"].eq("NONE_FORBIDDEN").all():
        raise ValueError("Gold annotation sidecar contains an automatic action")
    disagreement = annotations["teacher_reference_disagreement"]
    populated = disagreement.notna() & disagreement.astype(str).ne("")
    normalized = disagreement.loc[populated].map(
        {True: True, False: False, 1: True, 0: False, "True": True, "False": False, "1": True, "0": False}
    )
    if normalized.isna().any():
        raise ValueError("Gold annotation sidecar contains an invalid disagreement state")
    return {
        "rows": int(len(annotations)),
        "studies": int(annotations[ID_COLUMN].nunique()),
        "targets": int(annotations["target"].nunique()),
        "assessable_cells": int(populated.sum()),
        "disagreements": int(normalized.sum()),
    }


def _metric_text(row: Mapping[str, Any]) -> str:
    return (
        f"{float(row['point']):.3f} "
        f"(95% cluster-bootstrap CI {float(row['bootstrap_ci_lower']):.3f}–"
        f"{float(row['bootstrap_ci_upper']):.3f})"
    )


def render_report(
    *,
    metric_facts: Mapping[str, Any],
    policy: pd.DataFrame,
    manifest_sha256: str,
    independent_gold_audit_sha256: str,
    core_segment_candidates: int,
    lopo_profile_fits_completed: int,
    lopo_support_eligible_profiles: int,
    lopo_demoted_candidates: int,
    robust_segment_effects: int,
) -> str:
    """Render the deterministic concise final report."""

    if manifest_sha256 != EXPECTED_PROXY_FREEZE_MANIFEST_SHA256:
        raise ValueError("report manifest SHA differs from the immutable pre-Gold freeze")
    if len(independent_gold_audit_sha256) != 64:
        raise ValueError("independent Gold audit SHA-256 is invalid")
    if int(robust_segment_effects) != 0:
        raise ValueError("report finalizer requires the frozen empty robust segment shortlist")
    if not (
        int(core_segment_candidates) == 3
        and int(lopo_profile_fits_completed) == 42
        and int(lopo_support_eligible_profiles) == 0
        and int(lopo_demoted_candidates) == 3
    ):
        raise ValueError("report finalizer segment/LOPO counts drift")
    _require(policy, POLICY_COLUMNS, name="policy shortlist")

    priority_counts = policy["manual_review_priority"].value_counts().to_dict()
    model_counts = policy["proxy_model_status"].value_counts().to_dict()
    image_counts = policy["image_review_context"].value_counts().to_dict()
    targets = metric_facts["target_rows"]
    target_lines = [
        (
            f"| {row['target']} | {row['events']}/{row['cells']} | "
            f"{row['auroc']:.3f} [{row['auroc_ci_lower']:.3f}, {row['auroc_ci_upper']:.3f}] | "
            f"{row['average_precision']:.3f} "
            f"[{row['average_precision_ci_lower']:.3f}, {row['average_precision_ci_upper']:.3f}] |"
        )
        for row in targets
    ]
    text = f"""# Report reliability v3 — final local summary

## Design

The model, features, alphas, proxy ranks, risk tiers, candidate lists, and 127 × 24 segment hypotheses were frozen before Gold under manifest `{manifest_sha256}`. Gold58 was opened only after that freeze and is used here for append-only descriptive evaluation. It did not change selection, model fitting, ranks, tiers, routing rules, labels, confidence, or weights. The independent post-freeze Gold audit is bound by SHA-256 `{independent_gold_audit_sha256}`. Queue rows are rebuilt from the three manifest-bound frozen presentation panels and the authoritative token-exact `gold58_annotations_v3.csv` sidecar. The earlier `label_review_risk_panel_v3.csv.gz` is deterministic but non-authoritative convenience output because its reserialization changed some frozen decimal tokens at approximately 1e-16.

## Gold58 descriptive check

There are 691 assessable teacher/reference target cells from 58 report groups, including 129 disagreements (prevalence {float(metric_facts['pooled_auroc']['event_prevalence']):.3f}). After pooling the already target-specific percentile ranks across targets (this is not a macro-average or a conditional AUROC), descriptive AUROC is {_metric_text(metric_facts['pooled_auroc'])}; average precision is {_metric_text(metric_facts['pooled_average_precision'])}; AP/prevalence is {_metric_text(metric_facts['pooled_average_precision_over_prevalence'])}. Reviewing the top frozen 5% of target cells captures {_metric_text(metric_facts['pooled_capture_0_05'])} of disagreements with enrichment {_metric_text(metric_facts['pooled_enrichment_0_05'])}; the top 10% captures {_metric_text(metric_facts['pooled_capture_0_10'])} with enrichment {_metric_text(metric_facts['pooled_enrichment_0_10'])}; and the top 20% captures {_metric_text(metric_facts['pooled_capture_0_20'])} with enrichment {_metric_text(metric_facts['pooled_enrichment_0_20'])}. These fractions count target cells, not reports. Bootstrap resampling uses whole report groups (999 requested replicates; 994 valid for Medial Meniscus and 999 for every other scope).

Target-level values are descriptive only. Each target has just 6–18 observed disagreements, and the wide, overlapping intervals do not support target-specific policy thresholds or comparative claims. Gold58 is a small descriptive panel and does not establish transportability to the full training cohort.

| Target | disagreements / cells | AUROC [95% CI] | AP [95% CI] |
|---|---:|---:|---:|
{chr(10).join(target_lines)}

## Morphology segments

The frozen segment branch screened 127 hypotheses across 24 target-direction families. {int(core_segment_candidates)} core proxy candidates reached the pre-LOPO gate. All {int(lopo_profile_fits_completed)}/42 required LOPO profile fits completed, but {int(lopo_support_eligible_profiles)}/42 profiles met the frozen ≥100-per-arm support gate; {int(lopo_demoted_candidates)} candidates were therefore demoted as unsupported and {int(robust_segment_effects)} retained. The final robust segment shortlist contains **{int(robust_segment_effects)} effects**. This does not prove reversal, no effect, or invariance; it means no tested language, acquisition-profile, typo, length, or template segment passed the frozen robustness gates. No verified country, site, hospital, or reader identifier exists; language/scanner/acquisition fields cannot be interpreted as country or reader proxies.

## Fixed-tier manual-review queue

The local fixed-tier filtered view contains {len(policy):,} study-target rows from {policy[ID_COLUMN].nunique():,} studies: {int(priority_counts.get('P1', 0)):,} frozen `VERY_HIGH` rows routed to P1 and {int(priority_counts.get('P2', 0)):,} frozen `HIGH` rows routed to P2. Selection uses only those pre-Gold tiers. Image context and the frozen model status are carried as routing annotations, not as score modifiers ({int(model_counts.get('PASS_INCREMENT', 0)):,} `PASS_INCREMENT`; {int(model_counts.get('DIRECTIONAL_ONLY', 0)):,} `DIRECTIONAL_ONLY`; {int(image_counts.get('IMAGE_FAMILIES_AGREE_HIGH', 0) + image_counts.get('IMAGE_FAMILIES_AGREE_LOW', 0)):,} rows with both image families at the same extreme). Gold58 descriptively evaluates the frozen ranking but does not validate this view as an operating policy.

## Action boundary

`POLICY_SHORTLIST_v3.csv` is a deterministic **manual-review-only fixed-tier queue**, not an outcome-validated operating policy. It authorizes no automatic label flip, confidence reduction, sample-weight change, retraining, learned threshold, or submission change. A suspected omission/contradiction remains unconfirmed until blinded manual report-and-image adjudication. Any later policy experiment requires a new grouped-OOF contract and separate approval.
"""
    return text.rstrip() + "\n"


def stable_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = [
    "EXPECTED_PROXY_FREEZE_BINDING_PLAN_SHA256",
    "EXPECTED_PROXY_FREEZE_MANIFEST_SHA256",
    "GOLD_ANNOTATION_COLUMNS",
    "POLICY_COLUMNS",
    "POLICY_SOURCE_COLUMNS",
    "RANKING_REQUIRED_COLUMNS",
    "build_policy_shortlist",
    "columns_sha256",
    "render_report",
    "sha256_file",
    "stable_json_sha256",
    "validate_gold_ranking_metrics",
    "validate_gold_annotations",
]
