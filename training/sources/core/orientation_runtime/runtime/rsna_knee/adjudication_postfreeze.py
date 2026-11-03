"""Post-freeze diagnostic joins for report-only adjudication results."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rsna_knee.constants import ID_COLUMN


KEYS = [ID_COLUMN, "target"]
EVIDENCE_COLUMNS = [
    *KEYS,
    "gold",
    "is_gold",
    "llm_steven_v2",
    "llm_lixin_sol56",
    "llm_lixin_confidence",
    "llm_v4_blend",
    "dino_oof",
    "dino_public20_equal_oof",
    "radimagenet_oof",
    "radimagenet_v52_oof",
    "radimagenet_e11_public_oof",
    "balanced_image_probability",
    "train_baseline_probability",
    "train_baseline_weight",
    "primary_candidate_type",
    "text_state",
    "criteria_state",
    "language_proxy",
]


def build_postfreeze_panel(
    results: pd.DataFrame,
    evidence: pd.DataFrame,
) -> pd.DataFrame:
    """Reveal labels and public OOF diagnostics only after reader decisions are frozen."""

    required_results = {
        *KEYS,
        "case_key",
        "frozen_report_only_decision",
        "reader_assertions_agree",
        "reader_criteria_agree",
    }
    missing_results = sorted(required_results.difference(results.columns))
    missing_evidence = sorted(set(EVIDENCE_COLUMNS).difference(evidence.columns))
    if missing_results or missing_evidence:
        raise ValueError(
            f"Missing post-freeze columns: results={missing_results}, evidence={missing_evidence}"
        )
    if results.duplicated(KEYS).any() or results["case_key"].duplicated().any():
        raise ValueError("Adjudication result keys are not unique")
    if evidence.duplicated(KEYS).any():
        raise ValueError("Evidence-panel keys are not unique")

    panel = results.merge(
        evidence[EVIDENCE_COLUMNS],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    if len(panel) != len(results) or panel["train_baseline_probability"].isna().any():
        raise ValueError("Evidence panel does not cover every adjudication result")
    panel["baseline_threshold_decision"] = np.where(
        panel["train_baseline_probability"].astype(float) >= 0.5,
        "positive",
        "negative",
    )
    binary = panel["frozen_report_only_decision"].isin(["positive", "negative"])
    panel["report_binary_decision_available"] = binary
    panel["report_vs_baseline_threshold_disagreement"] = pd.array(
        [
            bool(report != baseline) if is_binary else pd.NA
            for report, baseline, is_binary in zip(
                panel["frozen_report_only_decision"],
                panel["baseline_threshold_decision"],
                binary,
                strict=True,
            )
        ],
        dtype="boolean",
    )
    gold_hard = np.where(panel["gold"].fillna(0).astype(float) >= 0.5, "positive", "negative")
    panel["gold_threshold_decision"] = pd.Series(gold_hard, index=panel.index).where(
        panel["is_gold"].astype(bool), pd.NA
    )
    panel["report_vs_gold_threshold_match"] = pd.array(
        [
            bool(report == gold_decision)
            if is_binary and is_gold
            else pd.NA
            for report, gold_decision, is_binary, is_gold in zip(
                panel["frozen_report_only_decision"],
                panel["gold_threshold_decision"],
                binary,
                panel["is_gold"].astype(bool),
                strict=True,
            )
        ],
        dtype="boolean",
    )
    panel["post_freeze_reveal"] = True
    panel["can_estimate_population_error_rate"] = False
    panel["can_rewrite_training_labels"] = False
    return panel


def summarize_postfreeze_panel(panel: pd.DataFrame) -> dict[str, Any]:
    """Return bounded descriptive counts for the intentionally enriched audit sample."""

    binary = panel[panel["report_binary_decision_available"].astype(bool)]
    gold_binary = binary[binary["is_gold"].astype(bool)]
    by_target: dict[str, Any] = {}
    for target, part in panel.groupby("target", sort=True):
        part_binary = part[part["report_binary_decision_available"].astype(bool)]
        by_target[str(target)] = {
            "cells": int(len(part)),
            "reader_assertion_agreement_cells": int(part["reader_assertions_agree"].sum()),
            "clear_report_binary_cells": int(len(part_binary)),
            "report_vs_baseline_threshold_disagreement_cells": int(
                part_binary["report_vs_baseline_threshold_disagreement"].sum()
            ),
            "frozen_decision_counts": {
                str(key): int(value)
                for key, value in part["frozen_report_only_decision"].value_counts().items()
            },
        }
    return {
        "cells": int(len(panel)),
        "frozen_decision_counts": {
            str(key): int(value)
            for key, value in panel["frozen_report_only_decision"].value_counts().items()
        },
        "reader_assertion_agreement_cells": int(panel["reader_assertions_agree"].sum()),
        "reader_assertion_agreement_rate": float(panel["reader_assertions_agree"].mean()),
        "clear_report_binary_cells": int(len(binary)),
        "report_vs_baseline_threshold_disagreement_cells": int(
            binary["report_vs_baseline_threshold_disagreement"].sum()
        ),
        "gold_cells_in_selected_sample": int(panel["is_gold"].sum()),
        "gold_clear_report_binary_cells": int(len(gold_binary)),
        "gold_clear_report_binary_match_cells": int(
            gold_binary["report_vs_gold_threshold_match"].sum()
        ),
        "by_target": by_target,
        "interpretation_boundary": {
            "sample_is_probability_sample": False,
            "sample_is_conflict_enriched": True,
            "can_estimate_population_error_rate": False,
            "report_reading_is_image_ground_truth": False,
            "can_rewrite_training_labels": False,
            "public_oof_images_are_diagnostic_only": True,
            "in_sample_coatnet_loaded": False,
        },
    }
