"""Deterministic descriptive outputs after the exact proxy freeze passes.

This module does not select features, models, segments, thresholds, ranks, or
weights.  It only combines byte-frozen pre-Gold presentation rows with the
append-only Gold annotations produced by :mod:`report_reliability_gold_gate`
and computes descriptive teacher-reference-disagreement diagnostics.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .constants import ID_COLUMN, TARGET_COLUMNS
from .report_reliability_evaluation import review_fraction_capture_enrichment
from .report_reliability_gold_gate import EXTERNAL_REQUIRED_COLUMNS


KEYS: tuple[str, str] = (ID_COLUMN, "target")
REPORT_GROUP_COLUMN = "report_group"
GOLD_APPEND_COLUMNS: tuple[str, ...] = (
    "gold",
    "gold_descriptive_reason_codes",
    "gold_descriptive_evidence_tier",
    "teacher_reference_disagreement",
    "attribution_status",
    "automatic_label_or_weight_action",
)
REVIEW_FRACTIONS: tuple[float, ...] = (0.05, 0.10, 0.20)
BOOTSTRAP_REPLICATES = 999
BOOTSTRAP_SEED = 20260830


def _require(frame: pd.DataFrame, columns: Sequence[str], *, name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def _exact_base(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    _require(frame, EXTERNAL_REQUIRED_COLUMNS, name=name)
    result = frame.loc[:, list(EXTERNAL_REQUIRED_COLUMNS)].copy()
    if result[list(KEYS)].isna().any().any() or result.duplicated(list(KEYS)).any():
        raise ValueError(f"{name} contains null or duplicate cell keys")
    return result


def _validated_gold_cell_cohort(
    frame: pd.DataFrame,
    *,
    expected_gold_studies: int,
    name: str,
) -> None:
    _require(frame, KEYS, name=name)
    if frame[list(KEYS)].isna().any().any() or frame.duplicated(list(KEYS)).any():
        raise ValueError(f"{name} contains null or duplicate cell keys")
    if (
        len(frame) != int(expected_gold_studies) * len(TARGET_COLUMNS)
        or frame[ID_COLUMN].nunique() != int(expected_gold_studies)
    ):
        raise ValueError(f"{name} coverage differs from expected studies x 12")
    expected_targets = set(TARGET_COLUMNS)
    target_sets = frame.groupby(ID_COLUMN, sort=False)["target"].agg(
        lambda values: frozenset(str(value) for value in values)
    )
    if not target_sets.map(lambda values: set(values) == expected_targets).all():
        raise ValueError(f"{name} is not an exact 12-target Cartesian cohort")


def _strict_binary_events(values: pd.Series, *, name: str) -> pd.Series:
    if values.isna().any() or not values.isin([True, False, 0, 1]).all():
        raise ValueError(f"{name} must contain exact non-null booleans/0/1")
    return values.astype(bool)


def _validated_study_cluster_map(
    gold_panel: pd.DataFrame,
    study_cluster_map: pd.DataFrame,
) -> pd.DataFrame:
    _require(
        study_cluster_map,
        (ID_COLUMN, REPORT_GROUP_COLUMN),
        name="Gold study cluster map",
    )
    mapping = study_cluster_map.loc[:, [ID_COLUMN, REPORT_GROUP_COLUMN]].copy()
    if mapping.isna().any().any() or mapping[ID_COLUMN].duplicated().any():
        raise ValueError("Gold study cluster map contains null or duplicate study keys")
    mapping[ID_COLUMN] = mapping[ID_COLUMN].astype(str).str.strip()
    mapping[REPORT_GROUP_COLUMN] = mapping[REPORT_GROUP_COLUMN].astype(str).str.strip()
    if mapping[ID_COLUMN].eq("").any() or mapping[REPORT_GROUP_COLUMN].eq("").any():
        raise ValueError("Gold study cluster map contains empty keys")
    expected_uids = set(gold_panel[ID_COLUMN].astype(str))
    selected = mapping.loc[mapping[ID_COLUMN].isin(expected_uids)].copy()
    if set(selected[ID_COLUMN]) != expected_uids:
        raise ValueError("Gold study cluster map does not exactly cover Gold studies")
    return selected.sort_values(ID_COLUMN, kind="mergesort").reset_index(drop=True)


def assemble_complete_cell_panel(
    proxy_decisive: pd.DataFrame,
    proxy_abstain: pd.DataFrame,
    external: pd.DataFrame,
    gold_panel: pd.DataFrame,
    *,
    expected_studies: int = 4_407,
    expected_cells: int = 52_884,
    expected_gold_studies: int = 58,
) -> pd.DataFrame:
    """Combine the frozen three-way cohort and append Gold annotations only."""

    decisive = _exact_base(proxy_decisive, name="proxy decisive")
    abstain = _exact_base(proxy_abstain, name="proxy abstain")
    external_base = _exact_base(external, name="external")
    if not abstain["teacher_state"].eq("ABSTAIN").all():
        raise ValueError("proxy abstain artifact contains decisive teacher rows")
    if abstain["label_review_risk_percentile"].notna().any():
        raise ValueError("proxy abstain artifact contains report-risk values")

    base = pd.concat([decisive, abstain, external_base], ignore_index=True)
    if base.duplicated(list(KEYS)).any():
        raise ValueError("frozen proxy/external cohorts overlap")
    if len(base) != int(expected_cells) or base[ID_COLUMN].nunique() != int(expected_studies):
        raise ValueError("complete frozen cell cohort count drift")
    if not base.groupby(ID_COLUMN, sort=False)["target"].nunique().eq(len(TARGET_COLUMNS)).all():
        raise ValueError("complete cell panel is not a 12-target Cartesian cohort")

    _require(gold_panel, (*EXTERNAL_REQUIRED_COLUMNS, *GOLD_APPEND_COLUMNS), name="Gold panel")
    gold = gold_panel.loc[:, [*EXTERNAL_REQUIRED_COLUMNS, *GOLD_APPEND_COLUMNS]].copy()
    if gold[list(KEYS)].isna().any().any() or gold.duplicated(list(KEYS)).any():
        raise ValueError("Gold panel contains null or duplicate cell keys")
    if (
        len(gold) != int(expected_gold_studies) * len(TARGET_COLUMNS)
        or gold[ID_COLUMN].nunique() != int(expected_gold_studies)
    ):
        raise ValueError("Gold panel coverage differs from 58 x 12")
    _validated_gold_cell_cohort(
        gold,
        expected_gold_studies=expected_gold_studies,
        name="Gold panel",
    )
    gold_values = pd.to_numeric(gold["gold"], errors="coerce")
    if gold_values.isna().any() or not gold_values.isin([0, 1]).all():
        raise ValueError("Gold panel gold must contain exact binary 0/1 values")

    external_gold = external_base.merge(
        gold.loc[:, list(KEYS)], on=list(KEYS), how="inner", validate="one_to_one"
    ).sort_values(list(KEYS), kind="mergesort").reset_index(drop=True)
    gold_base = gold.loc[:, list(EXTERNAL_REQUIRED_COLUMNS)].sort_values(
        list(KEYS), kind="mergesort"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(external_gold, gold_base, check_exact=True)

    frozen_snapshot = base.copy(deep=True).sort_values(list(KEYS), kind="mergesort").reset_index(drop=True)
    append = gold.loc[:, [*KEYS, *GOLD_APPEND_COLUMNS]].copy()
    result = base.merge(append, on=list(KEYS), how="left", validate="one_to_one")
    non_gold = result["gold"].isna()
    result["gold"] = pd.array(result["gold"], dtype="Int8")
    result.loc[non_gold, "gold_descriptive_reason_codes"] = ""
    result.loc[non_gold, "gold_descriptive_evidence_tier"] = "NOT_APPLICABLE"
    result["teacher_reference_disagreement"] = pd.array(
        result["teacher_reference_disagreement"], dtype="boolean"
    )
    result.loc[non_gold, "attribution_status"] = "NOT_GOLD_ADJUDICATED"
    result.loc[non_gold, "automatic_label_or_weight_action"] = "NONE_FORBIDDEN"

    target_order = {target: index for index, target in enumerate(TARGET_COLUMNS)}
    result["__target_order"] = result["target"].map(target_order)
    if result["__target_order"].isna().any():
        raise ValueError("complete panel contains an unknown target")
    result = result.sort_values(
        [ID_COLUMN, "__target_order"], kind="mergesort"
    ).drop(columns="__target_order").reset_index(drop=True)

    observed_frozen = result.loc[:, list(EXTERNAL_REQUIRED_COLUMNS)].sort_values(
        list(KEYS), kind="mergesort"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(frozen_snapshot, observed_frozen, check_exact=True)
    if not result["automatic_label_or_weight_action"].eq("NONE_FORBIDDEN").all():
        raise AssertionError("complete panel contains a label/weight action")
    return result


def build_gold_cross_tab(
    gold_panel: pd.DataFrame,
    *,
    expected_gold_studies: int = 58,
) -> pd.DataFrame:
    """Return fixed, long-form descriptive Gold cross-tabs."""

    required = (
        *KEYS,
        "teacher_state",
        "image_state",
        "image_evidence_strength",
        "proxy_model_status",
        "risk_strength",
        "teacher_reference_disagreement",
        "gold_descriptive_evidence_tier",
    )
    _require(gold_panel, required, name="Gold panel")
    _validated_gold_cell_cohort(
        gold_panel,
        expected_gold_studies=expected_gold_studies,
        name="Gold panel",
    )

    dimensions: tuple[tuple[str, ...], ...] = (
        (),
        ("target",),
        ("teacher_state",),
        ("image_state",),
        ("image_evidence_strength",),
        ("proxy_model_status",),
        ("risk_strength",),
        ("target", "teacher_state"),
    )
    rows: list[dict[str, Any]] = []
    for columns in dimensions:
        if columns:
            iterator = gold_panel.groupby(list(columns), sort=True, dropna=False, observed=True)
        else:
            iterator = [((), gold_panel)]
        for key, part in iterator:
            key_tuple = key if isinstance(key, tuple) else (key,)
            level = "ALL" if not columns else "|".join(
                f"{column}={value}" for column, value in zip(columns, key_tuple)
            )
            assessable = part["teacher_reference_disagreement"].notna()
            events = _strict_binary_events(
                part.loc[assessable, "teacher_reference_disagreement"],
                name="teacher_reference_disagreement",
            )
            rows.append(
                {
                    "dimension": "ALL" if not columns else "+".join(columns),
                    "level": level,
                    "cells": int(len(part)),
                    "studies": int(part[ID_COLUMN].nunique()),
                    "assessable_cells": int(assessable.sum()),
                    "teacher_reference_disagreements": int(events.sum()),
                    "teacher_reference_disagreement_rate": (
                        float(events.mean()) if len(events) else np.nan
                    ),
                    "gold_anchored_cells": int(
                        part["gold_descriptive_evidence_tier"].eq("GOLD_ANCHORED").sum()
                    ),
                    "claim_strength": "DESCRIPTIVE_GOLD58_ONLY",
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["dimension", "level"], kind="mergesort"
    ).reset_index(drop=True)


def _point_metrics(frame: pd.DataFrame) -> dict[str, float]:
    score = pd.to_numeric(frame["label_review_risk_percentile"], errors="raise").to_numpy(float)
    event = _strict_binary_events(
        frame["teacher_reference_disagreement"],
        name="teacher_reference_disagreement",
    ).to_numpy(np.int8)
    if (
        not np.isfinite(score).all()
        or np.any((score < 0.0) | (score > 1.0))
        or len(np.unique(event)) != 2
    ):
        raise ValueError(
            "ranking metrics require [0,1] finite scores and both event classes"
        )
    prevalence = float(event.mean())
    metrics = {
        "AUROC": float(roc_auc_score(event, score)),
        "AVERAGE_PRECISION": float(average_precision_score(event, score)),
        "AVERAGE_PRECISION_OVER_PREVALENCE": float(
            average_precision_score(event, score) / prevalence
        ),
    }
    review = review_fraction_capture_enrichment(
        frame,
        score_column="label_review_risk_percentile",
        event_column="teacher_reference_disagreement",
        row_key_columns=KEYS,
        review_fractions=REVIEW_FRACTIONS,
    )
    for row in review.itertuples(index=False):
        suffix = f"{float(row.review_fraction):.2f}"
        metrics[f"TOP_RISK_ENRICHMENT@{suffix}"] = float(row.top_risk_enrichment)
        metrics[f"ERROR_CAPTURE@{suffix}"] = float(row.event_capture)
    return metrics


def _bootstrap_metric_draws(
    frame: pd.DataFrame,
    *,
    scope_key: str,
    replicates: int,
    seed: int,
) -> dict[str, list[float]]:
    groups = sorted(frame[REPORT_GROUP_COLUMN].astype(str).unique().tolist())
    if not groups:
        raise ValueError("bootstrap scope contains no report groups")
    local_seed = int.from_bytes(
        hashlib.sha256(f"{seed}|{scope_key}".encode("utf-8")).digest()[:8],
        "big",
    )
    generator = np.random.default_rng(local_seed)
    by_group = {
        group: frame.loc[frame[REPORT_GROUP_COLUMN].astype(str).eq(group)].copy()
        for group in groups
    }
    draws: dict[str, list[float]] = {}
    for replicate in range(int(replicates)):
        sampled = generator.choice(groups, size=len(groups), replace=True)
        parts: list[pd.DataFrame] = []
        for copy_index, group in enumerate(sampled):
            part = by_group[str(group)].copy()
            part[ID_COLUMN] = [
                f"{copy_index:04d}|{uid}"
                for uid in part[ID_COLUMN].astype(str)
            ]
            part[REPORT_GROUP_COLUMN] = f"{copy_index:04d}|{group}"
            parts.append(part)
        boot = pd.concat(parts, ignore_index=True)
        events = _strict_binary_events(
            boot["teacher_reference_disagreement"],
            name="bootstrap teacher_reference_disagreement",
        )
        if events.nunique() != 2:
            continue
        values = _point_metrics(boot)
        for statistic, value in values.items():
            if np.isfinite(value):
                draws.setdefault(statistic, []).append(float(value))
    return draws


def build_gold_ranking_metrics(
    gold_panel: pd.DataFrame,
    study_cluster_map: pd.DataFrame,
    *,
    expected_gold_studies: int = 58,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Evaluate frozen ranks against Gold teacher-reference disagreement.

    Pooled scores are already target/direction-specific percentiles.  Target
    metrics are computed only when the frozen >=5 event and >=5 agreement gate
    passes.  Bootstrap resampling is by whole report_group, retaining all Gold
    studies and target rows belonging to every sampled group copy.
    """

    required = (*KEYS, "label_review_risk_percentile", "teacher_reference_disagreement")
    _require(gold_panel, required, name="Gold panel")
    _validated_gold_cell_cohort(
        gold_panel,
        expected_gold_studies=expected_gold_studies,
        name="Gold panel",
    )
    cluster_map = _validated_study_cluster_map(gold_panel, study_cluster_map)
    assessable = gold_panel.loc[
        gold_panel["teacher_reference_disagreement"].notna()
        & gold_panel["label_review_risk_percentile"].notna()
    ].copy()
    if assessable.empty:
        raise ValueError("Gold panel has no assessable frozen-risk rows")
    if assessable.duplicated(list(KEYS)).any():
        raise ValueError("Gold ranking rows contain duplicate keys")
    assessable[ID_COLUMN] = assessable[ID_COLUMN].astype(str)
    assessable = assessable.merge(
        cluster_map,
        on=ID_COLUMN,
        how="left",
        validate="many_to_one",
    )
    if assessable[REPORT_GROUP_COLUMN].isna().any():
        raise ValueError("Gold ranking rows are missing report-group clusters")

    scopes: list[tuple[str, str, pd.DataFrame]] = [
        ("POOLED_TARGET_STRATIFIED", "ALL", assessable)
    ]
    for target in TARGET_COLUMNS:
        scopes.append(("TARGET", target, assessable.loc[assessable["target"].eq(target)].copy()))

    rows: list[dict[str, Any]] = []
    for scope, level, part in scopes:
        events = _strict_binary_events(
            part["teacher_reference_disagreement"],
            name="teacher_reference_disagreement",
        )
        event_count = int(events.sum())
        agreement_count = int(len(events) - event_count)
        support = bool(event_count >= 5 and agreement_count >= 5)
        if not support:
            rows.append(
                {
                    "scope": scope,
                    "level": level,
                    "status": "INSUFFICIENT_SUPPORT",
                    "statistic": "NOT_COMPUTED",
                    "point": np.nan,
                    "bootstrap_ci_lower": np.nan,
                    "bootstrap_ci_upper": np.nan,
                    "bootstrap_valid_replicates": 0,
                    "bootstrap_requested_replicates": int(bootstrap_replicates),
                    "studies": int(part[ID_COLUMN].nunique()),
                    "report_groups": int(part[REPORT_GROUP_COLUMN].nunique()),
                    "cells": int(len(part)),
                    "events": event_count,
                    "agreements": agreement_count,
                    "event_prevalence": float(events.mean()) if len(events) else np.nan,
                    "claim_strength": "DESCRIPTIVE_GOLD58_ONLY",
                }
            )
            continue

        points = _point_metrics(part)
        draws = _bootstrap_metric_draws(
            part,
            scope_key=f"{scope}|{level}",
            replicates=int(bootstrap_replicates),
            seed=int(bootstrap_seed),
        )
        for statistic in sorted(points):
            values = np.asarray(draws.get(statistic, []), dtype=np.float64)
            if len(values) < max(1, int(math.floor(0.90 * bootstrap_replicates))):
                raise ValueError(
                    f"too few valid whole-report-group bootstrap replicates for "
                    f"{scope}/{level}/{statistic}"
                )
            lower, upper = np.quantile(values, [0.025, 0.975], method="linear")
            rows.append(
                {
                    "scope": scope,
                    "level": level,
                    "status": "PASS_DESCRIPTIVE",
                    "statistic": statistic,
                    "point": float(points[statistic]),
                    "bootstrap_ci_lower": float(lower),
                    "bootstrap_ci_upper": float(upper),
                    "bootstrap_valid_replicates": int(len(values)),
                    "bootstrap_requested_replicates": int(bootstrap_replicates),
                    "studies": int(part[ID_COLUMN].nunique()),
                    "report_groups": int(part[REPORT_GROUP_COLUMN].nunique()),
                    "cells": int(len(part)),
                    "events": event_count,
                    "agreements": agreement_count,
                    "event_prevalence": float(events.mean()),
                    "claim_strength": "DESCRIPTIVE_GOLD58_ONLY",
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["scope", "level", "statistic"], kind="mergesort"
    ).reset_index(drop=True)


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "GOLD_APPEND_COLUMNS",
    "REPORT_GROUP_COLUMN",
    "assemble_complete_cell_panel",
    "build_gold_cross_tab",
    "build_gold_ranking_metrics",
]
