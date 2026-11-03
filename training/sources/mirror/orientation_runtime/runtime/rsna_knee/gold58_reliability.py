"""Gold58 report-versus-score conflict and label-reliability diagnostics."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from rsna_knee.constants import ID_COLUMN, REPORT_COLUMN, TARGET_COLUMNS


KEYS = [ID_COLUMN, "target"]
PROBABILITY_SCORE_COLUMNS = [
    "dino_oof",
    "radimagenet_oof",
    "radimagenet_v52_oof",
    "radimagenet_e11_public_oof",
    "rad_family_probability",
    "family_balanced_probability",
    "public_image_oof_mean4",
    "public_image_oof_median4",
]
RANK_LIKE_SCORE_COLUMNS = ["dino_public20_equal_oof"]
REFERENCE_SCORE_COLUMNS = [
    "baseline_consensus_score",
    "llm_v4_blend",
]


def _clip_probability(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)


def add_report_stances(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the two Gold-blind report stance definitions frozen in the contract."""

    result = frame.copy()
    text_state = result["text_state"].astype(str)
    criteria_state = result["criteria_state"].astype(str)

    mention = pd.Series(pd.NA, index=result.index, dtype="Int8")
    mention.loc[text_state.eq("explicit_positive")] = 1
    mention.loc[text_state.eq("explicit_negative")] = 0
    result["mention_report_stance"] = mention

    rubric = pd.Series(pd.NA, index=result.index, dtype="Int8")
    rubric.loc[
        text_state.eq("explicit_positive") & criteria_state.eq("meets_threshold")
    ] = 1
    rubric.loc[
        text_state.eq("explicit_negative")
        | (
            text_state.eq("explicit_positive")
            & criteria_state.eq("below_threshold")
        )
    ] = 0
    result["rubric_report_stance"] = rubric
    return result


def add_image_families(frame: pd.DataFrame) -> pd.DataFrame:
    """Build probability-only ensembles and preserve the legacy rank-mixed score."""

    result = frame.copy()
    if "balanced_image_probability" in result:
        result["legacy_rank_mixed_balanced_score"] = result[
            "balanced_image_probability"
        ].astype(float)
    result["rad_family_probability"] = result[
        [
            "radimagenet_oof",
            "radimagenet_v52_oof",
            "radimagenet_e11_public_oof",
        ]
    ].median(axis=1)
    result["family_balanced_probability"] = 0.5 * (
        result["dino_oof"].astype(float) + result["rad_family_probability"]
    )
    ordinary_probability_views = result[
        [
            "dino_oof",
            "radimagenet_oof",
            "radimagenet_v52_oof",
            "radimagenet_e11_public_oof",
        ]
    ].astype(float)
    result["public_image_oof_mean4"] = ordinary_probability_views.mean(axis=1)
    result["public_image_oof_median4"] = ordinary_probability_views.median(axis=1)
    result["image_family_agreement"] = 1.0 - (
        result["dino_oof"].astype(float) - result["rad_family_probability"]
    ).abs()
    return result


def add_conflict_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """Add report/Gold conflicts and model direction on every assessable cell."""

    result = frame.copy()
    gold = result["gold"].astype(int)
    for definition in ("mention", "rubric"):
        stance_column = f"{definition}_report_stance"
        assessable = result[stance_column].notna()
        result[f"{definition}_report_assessable"] = assessable
        result[f"{definition}_report_gold_conflict"] = (
            assessable & result[stance_column].astype("Float64").ne(gold)
        )
    for score in (
        PROBABILITY_SCORE_COLUMNS
        + RANK_LIKE_SCORE_COLUMNS
        + REFERENCE_SCORE_COLUMNS
        + ["coatnet_v4_probability"]
    ):
        result[f"direction__{score}"] = (result[score].astype(float) >= 0.5).astype(
            np.uint8
        )
        probability = result[score].astype(float)
        result[f"gold_support__{score}"] = np.where(
            gold.eq(1), probability, 1.0 - probability
        )
    return result


def build_gold58_panel(evidence: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Build the exact 58-study x 12-target post-freeze diagnostic panel."""

    required = {
        *KEYS,
        "gold",
        "is_gold",
        "text_state",
        "criteria_state",
        "criteria_reason",
        "evidence_clause_normalized",
        "baseline_consensus_score",
        "baseline_consensus_weight",
        "llm_v4_blend",
        "llm_abs_disagreement",
        "dino_oof",
        "dino_public20_equal_oof",
        "radimagenet_oof",
        "radimagenet_v52_oof",
        "radimagenet_e11_public_oof",
        "coatnet_v4_probability",
        "report_group",
    }
    missing = sorted(required.difference(evidence.columns))
    if missing:
        raise ValueError(f"Evidence panel missing columns: {missing}")
    if evidence.duplicated(KEYS).any():
        raise ValueError("Evidence panel keys are not unique")
    gold = evidence.loc[evidence["is_gold"].astype(bool)].copy()
    if (
        len(gold) != 58 * len(TARGET_COLUMNS)
        or gold[ID_COLUMN].nunique() != 58
        or set(gold["target"]) != set(TARGET_COLUMNS)
        or not gold.groupby("target").size().eq(58).all()
        or gold["gold"].isna().any()
    ):
        raise ValueError("Gold58 coverage differs from 58 studies x 12 targets")

    if train[ID_COLUMN].duplicated().any() or REPORT_COLUMN not in train:
        raise ValueError("Official train report table is not unique or lacks Report")
    gold = gold.merge(
        train[[ID_COLUMN, REPORT_COLUMN]],
        on=ID_COLUMN,
        how="left",
        validate="many_to_one",
    )
    if gold[REPORT_COLUMN].isna().any():
        raise ValueError("Official report text does not cover Gold58")
    gold = add_report_stances(gold)
    gold = add_image_families(gold)
    gold = add_conflict_fields(gold)
    gold["pseudo_hard_label"] = (
        gold["baseline_consensus_score"].astype(float) >= 0.5
    ).astype(np.uint8)
    gold["pseudo_hard_correct"] = gold["pseudo_hard_label"].eq(
        gold["gold"].astype(int)
    )
    gold["pseudo_absolute_error"] = (
        gold["baseline_consensus_score"].astype(float) - gold["gold"].astype(float)
    ).abs()
    return gold.sort_values(KEYS, kind="mergesort").reset_index(drop=True)


def score_metrics(frame: pd.DataFrame, score_column: str) -> dict[str, Any]:
    """Compute macro target metrics plus pooled calibration metrics."""

    per_target: dict[str, Any] = {}
    for target, part in frame.groupby("target", sort=True):
        y = part["gold"].astype(int).to_numpy()
        p = _clip_probability(part[score_column])
        if len(np.unique(y)) != 2:
            raise ValueError(f"Target {target} does not have both Gold classes")
        per_target[str(target)] = {
            "roc_auc": float(roc_auc_score(y, p)),
            "average_precision": float(average_precision_score(y, p)),
            "brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
        }
    return {
        "macro_roc_auc": float(
            np.mean([values["roc_auc"] for values in per_target.values()])
        ),
        "macro_average_precision": float(
            np.mean([values["average_precision"] for values in per_target.values()])
        ),
        "brier": float(np.mean((_clip_probability(frame[score_column]) - frame["gold"]) ** 2)),
        "log_loss": float(
            log_loss(frame["gold"].astype(int), _clip_probability(frame[score_column]), labels=[0, 1])
        ),
        "per_target": per_target,
    }


def rank_score_metrics(frame: pd.DataFrame, score_column: str) -> dict[str, Any]:
    """Compute ranking metrics only for a rank-like OOF score."""

    per_target: dict[str, Any] = {}
    for target, part in frame.groupby("target", sort=True):
        y = part["gold"].astype(int).to_numpy()
        score = part[score_column].astype(float).to_numpy()
        per_target[str(target)] = {
            "roc_auc": float(roc_auc_score(y, score)),
            "average_precision": float(average_precision_score(y, score)),
        }
    return {
        "macro_roc_auc": float(
            np.mean([values["roc_auc"] for values in per_target.values()])
        ),
        "macro_average_precision": float(
            np.mean([values["average_precision"] for values in per_target.values()])
        ),
        "brier": None,
        "log_loss": None,
        "per_target": per_target,
        "probability_calibration_interpretable": False,
    }


def _conflict_score_summary(
    conflict: pd.DataFrame,
    stance_column: str,
    score_column: str,
) -> dict[str, Any]:
    direction = conflict[f"direction__{score_column}"].astype(int)
    gold = conflict["gold"].astype(int)
    stance = conflict[stance_column].astype(int)
    probability = conflict[score_column].astype(float)
    margin = 2.0 * (probability - 0.5).abs()
    strong = margin.ge(0.5)
    toward_gold = direction.eq(gold)
    toward_report = direction.eq(stance)
    return {
        "toward_gold_cells": int(toward_gold.sum()),
        "toward_report_cells": int(toward_report.sum()),
        "toward_gold_rate": float(toward_gold.mean()) if len(conflict) else None,
        "mean_gold_support": float(conflict[f"gold_support__{score_column}"].mean())
        if len(conflict)
        else None,
        "mean_normalized_margin": float(margin.mean()) if len(conflict) else None,
        "strong_cells": int(strong.sum()),
        "strong_toward_gold_cells": int((strong & toward_gold).sum()),
    }


def summarize_gold58_panel(panel: pd.DataFrame) -> dict[str, Any]:
    """Summarize all Gold cells and both frozen conflict definitions."""

    summary: dict[str, Any] = {
        "cells": int(len(panel)),
        "studies": int(panel[ID_COLUMN].nunique()),
        "targets": int(panel["target"].nunique()),
        "gold_positive_cells": int(panel["gold"].sum()),
        "score_metrics": {
            score: score_metrics(panel, score)
            for score in PROBABILITY_SCORE_COLUMNS + REFERENCE_SCORE_COLUMNS
        },
        "rank_like_score_metrics": {
            score: rank_score_metrics(panel, score)
            for score in RANK_LIKE_SCORE_COLUMNS
        },
        "diagnostic_in_sample_metrics": {
            "coatnet_v4_probability": score_metrics(panel, "coatnet_v4_probability")
        },
        "conflicts": {},
        "boundaries": {
            "coatnet_used_in_primary": False,
            "coatnet_gold58_is_heldout_diagnostic": True,
            "coatnet_other_4349_train_scores_are_in_sample": True,
            "full_0936_ensemble_score_present": False,
            "dino_public20_equal_oof_used_as_probability": False,
            "can_rewrite_gold": False,
            "claim_strength": "descriptive/directional only",
        },
    }
    for definition in ("mention", "rubric"):
        stance_column = f"{definition}_report_stance"
        assessable = panel[panel[f"{definition}_report_assessable"]]
        conflict = panel[panel[f"{definition}_report_gold_conflict"]]
        by_target: dict[str, Any] = {}
        for target, target_all in panel.groupby("target", sort=True):
            target_assessable = target_all[target_all[f"{definition}_report_assessable"]]
            target_conflict = target_all[target_all[f"{definition}_report_gold_conflict"]]
            by_target[str(target)] = {
                "assessable_cells": int(len(target_assessable)),
                "conflict_cells": int(len(target_conflict)),
                "report_positive_gold_negative": int(
                    (
                        target_conflict[stance_column].astype(int).eq(1)
                        & target_conflict["gold"].astype(int).eq(0)
                    ).sum()
                ),
                "report_negative_gold_positive": int(
                    (
                        target_conflict[stance_column].astype(int).eq(0)
                        & target_conflict["gold"].astype(int).eq(1)
                    ).sum()
                ),
                "scores": {
                    score: _conflict_score_summary(
                        target_conflict, stance_column, score
                    )
                    for score in (
                        "dino_oof",
                        "rad_family_probability",
                        "family_balanced_probability",
                        "public_image_oof_mean4",
                        "coatnet_v4_probability",
                    )
                },
            }
        summary["conflicts"][definition] = {
            "assessable_cells": int(len(assessable)),
            "conflict_cells": int(len(conflict)),
            "conflict_studies": int(conflict[ID_COLUMN].nunique()),
            "conflict_rate_full_grid": float(len(conflict) / len(panel)),
            "conflict_rate_assessable": float(len(conflict) / len(assessable))
            if len(assessable)
            else None,
            "report_positive_gold_negative": int(
                (
                    conflict[stance_column].astype(int).eq(1)
                    & conflict["gold"].astype(int).eq(0)
                ).sum()
            ),
            "report_negative_gold_positive": int(
                (
                    conflict[stance_column].astype(int).eq(0)
                    & conflict["gold"].astype(int).eq(1)
                ).sum()
            ),
            "scores": {
                score: _conflict_score_summary(conflict, stance_column, score)
                for score in (
                    "dino_oof",
                    "rad_family_probability",
                    "family_balanced_probability",
                    "public_image_oof_mean4",
                    "coatnet_v4_probability",
                )
            },
            "by_target": by_target,
        }
    mention_conflict = panel["mention_report_gold_conflict"]
    rubric_conflict = panel["rubric_report_gold_conflict"]
    summary["conflict_set_overlap"] = {
        "both": int((mention_conflict & rubric_conflict).sum()),
        "mention_only": int((mention_conflict & ~rubric_conflict).sum()),
        "rubric_only": int((~mention_conflict & rubric_conflict).sum()),
        "union": int((mention_conflict | rubric_conflict).sum()),
    }
    return summary


def conflict_case_columns(panel: pd.DataFrame) -> Iterable[str]:
    base = [
        ID_COLUMN,
        "target",
        "gold",
        "mention_report_stance",
        "rubric_report_stance",
        "mention_report_gold_conflict",
        "rubric_report_gold_conflict",
        "text_state",
        "criteria_state",
        "criteria_reason",
        "evidence_clause_normalized",
        "baseline_consensus_score",
        "llm_v4_blend",
        "llm_abs_disagreement",
        "dino_oof",
        "rad_family_probability",
        "family_balanced_probability",
        "public_image_oof_mean4",
        "dino_public20_equal_oof",
        "image_family_agreement",
        "coatnet_v4_probability",
        REPORT_COLUMN,
    ]
    return [column for column in base if column in panel]
