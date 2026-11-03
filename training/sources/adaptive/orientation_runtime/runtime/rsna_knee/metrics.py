"""Robust competition metric utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from .constants import TARGET_COLUMNS


def macro_roc_auc(
    labels: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if labels.shape != predictions.shape or labels.ndim != 2:
        raise ValueError("labels and predictions must have the same 2D shape")
    if labels.shape[1] != len(TARGET_COLUMNS):
        raise ValueError(f"Expected {len(TARGET_COLUMNS)} target columns")
    if mask is None:
        mask = np.isfinite(labels) & np.isfinite(predictions)
    else:
        mask = np.asarray(mask, dtype=bool) & np.isfinite(labels) & np.isfinite(predictions)

    per_target: dict[str, float | None] = {}
    counts: dict[str, dict[str, int]] = {}
    valid_scores: list[float] = []
    for index, target in enumerate(TARGET_COLUMNS):
        # ROC AUC requires authoritative binary outcomes. Soft pseudo-labels are
        # training targets, not validation truth, and must not enter the metric.
        binary = (labels[:, index] == 0.0) | (labels[:, index] == 1.0)
        valid = mask[:, index] & binary
        target_labels = labels[valid, index]
        target_predictions = predictions[valid, index]
        counts[target] = {
            "cells": int(valid.sum()),
            "positives": int((target_labels == 1.0).sum()),
            "negatives": int((target_labels == 0.0).sum()),
            "excluded_nonbinary_cells": int((mask[:, index] & ~binary).sum()),
        }
        if len(target_labels) == 0 or np.unique(target_labels).size < 2:
            per_target[target] = None
            continue
        score = float(roc_auc_score(target_labels, target_predictions))
        per_target[target] = score
        valid_scores.append(score)
    return {
        "macro_auc": float(np.mean(valid_scores)) if valid_scores else None,
        "valid_target_count": len(valid_scores),
        "per_target": per_target,
        "counts": counts,
        "protocol": "strict_binary_observed; soft labels excluded; source is not inferred from binary values",
    }
