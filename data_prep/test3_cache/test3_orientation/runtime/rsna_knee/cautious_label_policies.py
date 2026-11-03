"""Frozen cautious report/image reliability policies for the full train grid."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rsna_knee.constants import ID_COLUMN, TARGET_COLUMNS
from rsna_knee.gold58_reliability import add_image_families, add_report_stances


KEYS = [ID_COLUMN, "target"]
POLICY_COLUMNS = {
    "baseline": (
        "train_baseline_probability",
        "train_baseline_weight",
    ),
    "concordance_continuous_v1": (
        "train_concordance_continuous_v1_probability",
        "train_concordance_continuous_v1_weight",
    ),
    "extreme_bucket_v1": (
        "train_extreme_bucket_v1_probability",
        "train_extreme_bucket_v1_weight",
    ),
    "extreme_soft10_v1": (
        "train_extreme_soft10_v1_probability",
        "train_extreme_soft10_v1_weight",
    ),
}


def build_cautious_policy_cells(evidence: pd.DataFrame) -> pd.DataFrame:
    """Apply the predeclared policies without changing canonical Gold supervision."""

    required = {
        *KEYS,
        "gold",
        "is_gold",
        "text_state",
        "criteria_state",
        "llm_v4_blend",
        "llm_abs_disagreement",
        "dino_oof",
        "radimagenet_oof",
        "radimagenet_v52_oof",
        "radimagenet_e11_public_oof",
        "train_baseline_probability",
        "train_baseline_weight",
    }
    missing = sorted(required.difference(evidence.columns))
    if missing:
        raise ValueError(f"Evidence panel missing policy inputs: {missing}")
    if evidence.duplicated(KEYS).any():
        raise ValueError("Evidence panel must be unique by study and target")
    if (
        len(evidence) != 4407 * len(TARGET_COLUMNS)
        or evidence[ID_COLUMN].nunique() != 4407
        or set(evidence["target"].astype(str)) != set(TARGET_COLUMNS)
    ):
        raise ValueError("Evidence coverage differs from 4,407 studies x 12 targets")

    result = add_image_families(add_report_stances(evidence))
    baseline_probability = result["train_baseline_probability"].astype(float)
    baseline_weight = result["train_baseline_weight"].astype(float)
    is_gold = result["is_gold"].astype(bool)
    stance_available = result["rubric_report_stance"].notna()
    report_positive = result["rubric_report_stance"].eq(1).fillna(False)
    report_negative = result["rubric_report_stance"].eq(0).fillna(False)

    primary_image = result["public_image_oof_mean4"].astype(float)
    dino = result["dino_oof"].astype(float)
    rad = result["rad_family_probability"].astype(float)
    family_agreement = result["image_family_agreement"].astype(float)
    image_confidence = 2.0 * (primary_image - 0.5).abs()
    llm_support = np.where(
        report_positive,
        result["llm_v4_blend"].astype(float),
        1.0 - result["llm_v4_blend"].astype(float),
    )
    report_confidence = 0.5 + 0.5 * np.maximum(0.0, 2.0 * llm_support - 1.0) * (
        1.0 - result["llm_abs_disagreement"].astype(float)
    )
    alignment = np.where(report_positive, primary_image, 1.0 - primary_image)
    continuous_signal = (
        (2.0 * alignment - 1.0)
        * image_confidence
        * family_agreement
        * report_confidence
    )
    continuous_multiplier = np.where(
        stance_available,
        np.clip(1.0 + 0.25 * continuous_signal, 0.75, 1.25),
        1.0,
    )

    strong_image_positive = dino.ge(0.80) & rad.ge(0.80)
    strong_image_negative = dino.le(0.20) & rad.le(0.20)
    report_image_strong_agree = (
        (report_positive & strong_image_positive)
        | (report_negative & strong_image_negative)
    )
    report_image_strong_conflict = (
        (report_positive & strong_image_negative)
        | (report_negative & strong_image_positive)
    )
    extreme_multiplier = np.select(
        [report_image_strong_agree, report_image_strong_conflict],
        [1.20, 0.70],
        default=1.0,
    )

    result["policy__rubric_stance_available"] = stance_available
    result["policy__strong_image_positive"] = strong_image_positive
    result["policy__strong_image_negative"] = strong_image_negative
    result["policy__report_image_strong_agree"] = report_image_strong_agree
    result["policy__report_image_strong_conflict"] = report_image_strong_conflict
    result["policy__continuous_signal"] = continuous_signal
    result["policy__continuous_multiplier"] = continuous_multiplier
    result["policy__extreme_multiplier"] = extreme_multiplier

    result["train_concordance_continuous_v1_probability"] = baseline_probability
    result["train_concordance_continuous_v1_weight"] = np.minimum(
        1.0, baseline_weight * continuous_multiplier
    )
    result["train_extreme_bucket_v1_probability"] = baseline_probability
    result["train_extreme_bucket_v1_weight"] = np.minimum(
        1.0, baseline_weight * extreme_multiplier
    )
    result["train_extreme_soft10_v1_weight"] = result[
        "train_extreme_bucket_v1_weight"
    ].astype(float)
    stance_probability = result["rubric_report_stance"].astype("Float64").fillna(0.5)
    consensus_probability = (stance_probability.astype(float) + dino + rad) / 3.0
    soft_eligible = (~is_gold) & report_image_strong_agree
    result["train_extreme_soft10_v1_probability"] = np.where(
        soft_eligible,
        0.90 * baseline_probability + 0.10 * consensus_probability,
        baseline_probability,
    )
    result["policy__soft10_eligible"] = soft_eligible

    # Canonical Gold is the immutable supervision anchor under every policy.
    for policy, (probability_column, weight_column) in POLICY_COLUMNS.items():
        if policy == "baseline":
            continue
        result.loc[is_gold, probability_column] = baseline_probability.loc[is_gold]
        result.loc[is_gold, weight_column] = 1.0

    for probability_column, weight_column in POLICY_COLUMNS.values():
        probability = result[probability_column].astype(float).to_numpy()
        weight = result[weight_column].astype(float).to_numpy()
        if (
            not np.isfinite(probability).all()
            or not np.isfinite(weight).all()
            or ((probability < 0.0) | (probability > 1.0)).any()
            or ((weight < 0.0) | (weight > 1.0)).any()
        ):
            raise ValueError(
                f"Policy values are non-finite or unbounded: {probability_column}"
            )
        if not np.array_equal(
            result.loc[is_gold, probability_column].to_numpy(float),
            baseline_probability.loc[is_gold].to_numpy(float),
        ):
            raise ValueError(f"Policy changes Gold probability: {probability_column}")
        if not result.loc[is_gold, weight_column].eq(1.0).all():
            raise ValueError(f"Policy changes Gold weight: {weight_column}")

    output_columns = [
        *KEYS,
        "gold",
        "is_gold",
        "report_group",
        "text_state",
        "criteria_state",
        "rubric_report_stance",
        "llm_v4_blend",
        "llm_abs_disagreement",
        "dino_oof",
        "rad_family_probability",
        "public_image_oof_mean4",
        "image_family_agreement",
        "policy__rubric_stance_available",
        "policy__strong_image_positive",
        "policy__strong_image_negative",
        "policy__report_image_strong_agree",
        "policy__report_image_strong_conflict",
        "policy__continuous_signal",
        "policy__continuous_multiplier",
        "policy__extreme_multiplier",
        "policy__soft10_eligible",
        *[column for pair in POLICY_COLUMNS.values() for column in pair],
    ]
    return result[output_columns].sort_values(KEYS, kind="mergesort").reset_index(
        drop=True
    )


def summarize_cautious_policy_cells(cells: pd.DataFrame) -> dict[str, Any]:
    """Create exact change counts and distribution diagnostics by policy/target."""

    is_gold = cells["is_gold"].astype(bool)
    pseudo = ~is_gold
    baseline_probability = cells["train_baseline_probability"].astype(float)
    baseline_weight = cells["train_baseline_weight"].astype(float)
    policies: dict[str, Any] = {}
    for policy, (probability_column, weight_column) in POLICY_COLUMNS.items():
        probability = cells[probability_column].astype(float)
        weight = cells[weight_column].astype(float)
        probability_delta = (probability - baseline_probability).abs()
        weight_delta = weight - baseline_weight
        by_target: dict[str, Any] = {}
        for target, part in cells.groupby("target", sort=True):
            target_pseudo = ~part["is_gold"].astype(bool)
            target_probability_delta = (
                part[probability_column].astype(float)
                - part["train_baseline_probability"].astype(float)
            ).abs()
            target_weight_delta = (
                part[weight_column].astype(float)
                - part["train_baseline_weight"].astype(float)
            )
            by_target[str(target)] = {
                "pseudo_cells": int(target_pseudo.sum()),
                "changed_probability_cells": int(
                    (target_pseudo & target_probability_delta.gt(0)).sum()
                ),
                "changed_weight_cells": int(
                    (target_pseudo & target_weight_delta.ne(0)).sum()
                ),
                "downweighted_cells": int(
                    (target_pseudo & target_weight_delta.lt(0)).sum()
                ),
                "upweighted_cells": int(
                    (target_pseudo & target_weight_delta.gt(0)).sum()
                ),
                "mean_weight": float(part.loc[target_pseudo, weight_column].mean()),
            }
        policies[policy] = {
            "changed_probability_cells": int((pseudo & probability_delta.gt(0)).sum()),
            "changed_weight_cells": int((pseudo & weight_delta.ne(0)).sum()),
            "downweighted_cells": int((pseudo & weight_delta.lt(0)).sum()),
            "upweighted_cells": int((pseudo & weight_delta.gt(0)).sum()),
            "probability_l1_change": float(probability_delta.loc[pseudo].sum()),
            "weight_l1_change": float(weight_delta.loc[pseudo].abs().sum()),
            "pseudo_weight_sum": float(weight.loc[pseudo].sum()),
            "pseudo_weight_retained_vs_baseline": float(
                weight.loc[pseudo].sum() / baseline_weight.loc[pseudo].sum()
            ),
            "by_target": by_target,
        }
    return {
        "studies": int(cells[ID_COLUMN].nunique()),
        "targets": int(cells["target"].nunique()),
        "cells": int(len(cells)),
        "gold_cells": int(is_gold.sum()),
        "pseudo_cells": int(pseudo.sum()),
        "rubric_assessable_cells": int(cells["policy__rubric_stance_available"].sum()),
        "strong_image_positive_cells": int(cells["policy__strong_image_positive"].sum()),
        "strong_image_negative_cells": int(cells["policy__strong_image_negative"].sum()),
        "strong_report_image_agree_cells": int(
            cells["policy__report_image_strong_agree"].sum()
        ),
        "strong_report_image_conflict_cells": int(
            cells["policy__report_image_strong_conflict"].sum()
        ),
        "soft10_eligible_pseudo_cells": int(cells["policy__soft10_eligible"].sum()),
        "policies": policies,
        "gates": {
            "complete_4407_x_12_grid": len(cells) == 4407 * len(TARGET_COLUMNS),
            "unique_study_target_keys": not cells.duplicated(KEYS).any(),
            "gold_probability_unchanged": all(
                np.array_equal(
                    cells.loc[is_gold, probability_column].to_numpy(float),
                    baseline_probability.loc[is_gold].to_numpy(float),
                )
                for probability_column, _ in POLICY_COLUMNS.values()
            ),
            "gold_weight_one": all(
                cells.loc[is_gold, weight_column].eq(1.0).all()
                for _, weight_column in POLICY_COLUMNS.values()
            ),
            "all_values_bounded": True,
        },
        "claim_strength": "mechanical policy materialization; outcome value not yet known",
    }
