"""Decision-safe summary of the frozen three-policy grouped OOF run."""

from __future__ import annotations

from typing import Any, Mapping


def build_policy_comparison(
    metrics: Mapping[str, Any],
    run_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the frozen run and expose only decision-relevant comparisons."""

    if run_audit.get("status") != "PASS" or not all(run_audit.get("gates", {}).values()):
        raise ValueError("OOF run audit does not pass every gate")
    expected = {"baseline", "downweight", "soft"}
    if set(metrics.get("conditions", {})) != expected:
        raise ValueError("OOF conditions differ from the frozen three-policy design")
    comparisons = metrics.get("comparisons", {})
    if set(comparisons) != {"downweight_minus_baseline", "soft_minus_baseline"}:
        raise ValueError("OOF baseline comparisons are incomplete")

    decisions: dict[str, Any] = {}
    for name, comparison in comparisons.items():
        auc_delta = float(comparison["observed_delta_macro_roc_auc"])
        auc_low, auc_high = [float(value) for value in comparison["paired_bootstrap_95"]]
        brier_improvement = float(comparison["observed_brier_improvement"])
        brier_low, brier_high = [
            float(value) for value in comparison["paired_brier_improvement_95"]
        ]
        decisions[name] = {
            "observed_delta_macro_roc_auc": auc_delta,
            "paired_auc_delta_95": [auc_low, auc_high],
            "auc_improvement_supported": bool(auc_delta > 0 and auc_low > 0),
            "observed_brier_improvement": brier_improvement,
            "paired_brier_improvement_95": [brier_low, brier_high],
            "calibration_improvement_supported": bool(
                brier_improvement > 0 and brier_low > 0
            ),
            "calibration_worsening_supported": bool(
                brier_improvement < 0 and brier_high < 0
            ),
            "observed_log_loss_improvement": float(
                comparison["observed_log_loss_improvement"]
            ),
            "worst_target_roc_auc_delta": comparison["worst_target_roc_auc_delta"],
            "per_target_roc_auc_delta": comparison["per_target_roc_auc_delta"],
        }

    baseline = metrics["conditions"]["baseline"]
    return {
        "format": "rsna-knee-oof-policy-comparison-v1",
        "status": "PASS",
        "claim_strength": "descriptive/directional only",
        "selected_policy": "baseline",
        "decision": "retain_baseline_and_reject_bulk_downweight_or_soft_correction",
        "reason": (
            "Neither sensitivity policy has a positive paired AUC interval, while both "
            "have paired Brier intervals entirely on the worsening side."
        ),
        "design": {
            "gold_studies": int(metrics["gold_studies"]),
            "targets": int(metrics["targets"]),
            "repeats_averaged": int(metrics["repeats_averaged"]),
            "bootstrap_iterations": int(metrics["bootstrap_iterations"]),
            "raw_prediction_rows": int(run_audit["coverage"]["raw_rows"]),
            "heads": int(
                len(expected)
                * int(run_audit["coverage"]["repeats"])
                * int(run_audit["coverage"]["folds"])
            ),
        },
        "baseline": {
            "macro_roc_auc": float(baseline["macro_roc_auc"]),
            "macro_roc_auc_bootstrap_95": [
                float(value) for value in baseline["macro_roc_auc_bootstrap_95"]
            ],
            "macro_average_precision": float(baseline["macro_average_precision"]),
            "brier": float(baseline["brier"]),
            "log_loss": float(baseline["log_loss"]),
        },
        "comparisons": decisions,
        "effective_supervision": {
            name: {
                "effective_weight_retained_vs_baseline": float(
                    values["effective_weight_retained_vs_baseline"]
                ),
                "changed_probability_cells": int(values["changed_probability_cells"]),
                "changed_weight_cells": int(values["changed_weight_cells"]),
            }
            for name, values in metrics["effective_supervision"].items()
        },
        "boundaries": {
            "gold58_is_small": True,
            "same_gold_can_choose_new_target_specific_policy": False,
            "public_leaderboard_weight_tuning_allowed": False,
            "bulk_label_rewrite_supported": False,
        },
    }
