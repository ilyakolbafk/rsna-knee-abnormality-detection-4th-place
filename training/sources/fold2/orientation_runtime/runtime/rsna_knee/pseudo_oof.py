"""Primary validation protocol: report-grouped OOF scored on pseudo-labels.

Why this module exists
----------------------
Only 58 of 4,407 training studies carry image-derived gold labels.  Scoring a
model on those 58 alone (the historical project gate) has a paired 95%
interval of roughly +-0.015 macro AUC, so it cannot distinguish the 0.003-0.01
effects that separate leaderboard positions.  Competitors who report that
their CV tracks the leaderboard score their out-of-fold predictions against
the *report-derived soft labels of every held-out study*; the seed-to-seed
noise floor of that estimate is about 0.002 macro AUC.

This module implements that protocol without any torch or sklearn dependency
so it can run inside minimal environments (CI, a laptop without CUDA, a
Kaggle CPU kernel):

* ``roc_auc`` - rank-based AUC with tie handling (Mann-Whitney U).
* ``pseudo_macro_auc`` - macro AUC where each held-out soft label is binarised
  at ``positive_threshold`` and cells below ``min_confidence`` (or below an
  abstention band around the threshold) are excluded.  Gold cells always
  count with their exact binary label.
* ``gold_macro_auc`` - the legacy 58-study sanity metric.
* ``evaluate_oof_frame`` - one call that returns the whole scorecard from an
  OOF prediction frame.
* ``study_bootstrap_delta`` - paired study-level bootstrap for A-vs-B deltas.
* ``seed_noise_floor`` - mean absolute paired difference between seeds of the
  same recipe; an improvement smaller than this is noise by construction.

OOF frame contract
------------------
One row per held-out study with at least::

    StudyInstanceUID, is_gold, fold,
    <target>            soft or gold label in [0, 1] or NaN
    confidence__<target> teacher confidence in [0, 1] (1.0 for gold)
    pred__<target>      model probability

Missing ``confidence__`` columns default to 1.0.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .constants import ID_COLUMN, TARGET_COLUMNS

DEFAULT_MIN_CONFIDENCE = 0.5
DEFAULT_POSITIVE_THRESHOLD = 0.5
DEFAULT_ABSTENTION_BAND = 0.0


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Rank-based ROC AUC with average ranks for ties (equals sklearn)."""

    labels = np.asarray(labels, dtype=float).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have the same length")
    if len(labels) == 0:
        return None
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    positives = labels == 1.0
    negatives = labels == 0.0
    if not (positives | negatives).all():
        raise ValueError("labels must be exactly 0 or 1 for AUC")
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    # Average ranks over tied groups.
    index = 0
    while index < len(sorted_scores):
        stop = index
        while stop + 1 < len(sorted_scores) and sorted_scores[stop + 1] == sorted_scores[index]:
            stop += 1
        average_rank = 0.5 * (index + stop) + 1.0
        ranks[order[index : stop + 1]] = average_rank
        index = stop + 1
    rank_sum_positive = float(ranks[positives].sum())
    auc = (rank_sum_positive - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _target_arrays(
    frame: pd.DataFrame, prefix: str = "pred__"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = frame[list(TARGET_COLUMNS)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    predictions = np.stack(
        [pd.to_numeric(frame[f"{prefix}{target}"], errors="coerce").to_numpy(float) for target in TARGET_COLUMNS],
        axis=1,
    )
    confidence = np.stack(
        [
            (
                pd.to_numeric(frame[f"confidence__{target}"], errors="coerce").to_numpy(float)
                if f"confidence__{target}" in frame
                else np.ones(len(frame), dtype=float)
            )
            for target in TARGET_COLUMNS
        ],
        axis=1,
    )
    confidence = np.where(np.isfinite(confidence), confidence, 0.0)
    is_gold = frame.get("is_gold", pd.Series(False, index=frame.index)).astype(bool).to_numpy()
    return labels, predictions, confidence, is_gold


def binarise_pseudo_labels(
    labels: np.ndarray,
    confidence: np.ndarray,
    is_gold: np.ndarray,
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    positive_threshold: float = DEFAULT_POSITIVE_THRESHOLD,
    abstention_band: float = DEFAULT_ABSTENTION_BAND,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(binary_labels, usable_mask)`` for pseudo-label scoring.

    Gold rows are used verbatim (they are already binary).  Pseudo rows are
    usable when the label is finite, teacher confidence is at least
    ``min_confidence`` and the soft label is outside the abstention band
    ``[threshold - band, threshold + band]``.
    """

    labels = np.asarray(labels, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    is_gold = np.asarray(is_gold, dtype=bool)
    finite = np.isfinite(labels)
    gold_mask = finite & is_gold[:, None] & ((labels == 0.0) | (labels == 1.0))
    distance = np.abs(labels - positive_threshold)
    pseudo_mask = (
        finite
        & ~is_gold[:, None]
        & (confidence >= min_confidence)
        & (distance > abstention_band)
    )
    binary = np.where(labels >= positive_threshold, 1.0, 0.0)
    binary = np.where(gold_mask, labels, binary)
    return binary, gold_mask | pseudo_mask


def macro_auc_from_arrays(
    binary_labels: np.ndarray, predictions: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    per_target: dict[str, float | None] = {}
    counts: dict[str, dict[str, int]] = {}
    scores: list[float] = []
    for index, target in enumerate(TARGET_COLUMNS):
        valid = mask[:, index] & np.isfinite(predictions[:, index])
        target_labels = binary_labels[valid, index]
        auc = roc_auc(target_labels, predictions[valid, index]) if valid.any() else None
        per_target[target] = auc
        counts[target] = {
            "cells": int(valid.sum()),
            "positives": int((target_labels == 1.0).sum()),
        }
        if auc is not None:
            scores.append(auc)
    return {
        "macro_auc": float(np.mean(scores)) if scores else None,
        "valid_target_count": len(scores),
        "per_target": per_target,
        "counts": counts,
    }


def pseudo_macro_auc(
    frame: pd.DataFrame,
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    positive_threshold: float = DEFAULT_POSITIVE_THRESHOLD,
    abstention_band: float = DEFAULT_ABSTENTION_BAND,
    prediction_prefix: str = "pred__",
) -> dict[str, Any]:
    labels, predictions, confidence, is_gold = _target_arrays(frame, prediction_prefix)
    binary, mask = binarise_pseudo_labels(
        labels,
        confidence,
        is_gold,
        min_confidence=min_confidence,
        positive_threshold=positive_threshold,
        abstention_band=abstention_band,
    )
    result = macro_auc_from_arrays(binary, predictions, mask)
    result["protocol"] = {
        "min_confidence": float(min_confidence),
        "positive_threshold": float(positive_threshold),
        "abstention_band": float(abstention_band),
        "studies": int(len(frame)),
    }
    return result


def gold_macro_auc(frame: pd.DataFrame, *, prediction_prefix: str = "pred__") -> dict[str, Any]:
    labels, predictions, _, is_gold = _target_arrays(frame, prediction_prefix)
    mask = is_gold[:, None] & np.isfinite(labels) & ((labels == 0.0) | (labels == 1.0))
    return macro_auc_from_arrays(labels, predictions, mask)


def soft_calibration(frame: pd.DataFrame, *, prediction_prefix: str = "pred__") -> dict[str, float | None]:
    """Brier score and log loss against the soft labels (all finite cells)."""

    labels, predictions, _, _ = _target_arrays(frame, prediction_prefix)
    mask = np.isfinite(labels) & np.isfinite(predictions)
    if not mask.any():
        return {"brier": None, "log_loss": None}
    p = np.clip(predictions[mask], 1e-6, 1 - 1e-6)
    y = labels[mask]
    brier = float(np.mean((p - y) ** 2))
    log_loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return {"brier": brier, "log_loss": log_loss}


def evaluate_oof_frame(
    frame: pd.DataFrame,
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    positive_threshold: float = DEFAULT_POSITIVE_THRESHOLD,
    abstention_band: float = DEFAULT_ABSTENTION_BAND,
    prediction_prefix: str = "pred__",
) -> dict[str, Any]:
    """Full scorecard: pseudo OOF macro AUC (primary), gold AUC, calibration."""

    required = {ID_COLUMN, *TARGET_COLUMNS, *(f"{prediction_prefix}{t}" for t in TARGET_COLUMNS)}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"OOF frame is missing columns: {missing}")
    if frame[ID_COLUMN].duplicated().any():
        raise ValueError("OOF frame must contain one row per StudyInstanceUID")
    pseudo = pseudo_macro_auc(
        frame,
        min_confidence=min_confidence,
        positive_threshold=positive_threshold,
        abstention_band=abstention_band,
        prediction_prefix=prediction_prefix,
    )
    gold = gold_macro_auc(frame, prediction_prefix=prediction_prefix)
    result: dict[str, Any] = {
        "primary_metric": "pseudo_macro_auc",
        "pseudo": pseudo,
        "gold": gold,
        "calibration": soft_calibration(frame, prediction_prefix=prediction_prefix),
        "studies": int(len(frame)),
        "gold_studies": int(frame.get("is_gold", pd.Series(False, index=frame.index)).astype(bool).sum()),
    }
    if "fold" in frame:
        per_fold: dict[str, float | None] = {}
        for fold, part in frame.groupby("fold", sort=True):
            per_fold[str(int(fold))] = pseudo_macro_auc(
                part,
                min_confidence=min_confidence,
                positive_threshold=positive_threshold,
                abstention_band=abstention_band,
                prediction_prefix=prediction_prefix,
            )["macro_auc"]
        result["pseudo_per_fold"] = per_fold
    return result


def _aligned(frame_a: pd.DataFrame, frame_b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = frame_a.set_index(ID_COLUMN).sort_index()
    b = frame_b.set_index(ID_COLUMN).sort_index()
    if not a.index.equals(b.index):
        raise ValueError("paired frames must contain the same StudyInstanceUID set")
    return a.reset_index(), b.reset_index()


def study_bootstrap_delta(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    *,
    iterations: int = 2000,
    seed: int = 0,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    positive_threshold: float = DEFAULT_POSITIVE_THRESHOLD,
    abstention_band: float = DEFAULT_ABSTENTION_BAND,
) -> dict[str, Any]:
    """Paired study-level bootstrap of ``pseudo_macro_auc(A) - pseudo_macro_auc(B)``."""

    a, b = _aligned(frame_a, frame_b)
    labels, pred_a, confidence, is_gold = _target_arrays(a)
    _, pred_b, _, _ = _target_arrays(b)
    binary, mask = binarise_pseudo_labels(
        labels,
        confidence,
        is_gold,
        min_confidence=min_confidence,
        positive_threshold=positive_threshold,
        abstention_band=abstention_band,
    )
    point_a = macro_auc_from_arrays(binary, pred_a, mask)["macro_auc"]
    point_b = macro_auc_from_arrays(binary, pred_b, mask)["macro_auc"]
    rng = np.random.default_rng(seed)
    n = len(a)
    deltas: list[float] = []
    for _ in range(int(iterations)):
        sample = rng.integers(0, n, size=n)
        auc_a = macro_auc_from_arrays(binary[sample], pred_a[sample], mask[sample])["macro_auc"]
        auc_b = macro_auc_from_arrays(binary[sample], pred_b[sample], mask[sample])["macro_auc"]
        if auc_a is None or auc_b is None:
            continue
        deltas.append(auc_a - auc_b)
    deltas_array = np.asarray(deltas, dtype=float)
    return {
        "point_a": point_a,
        "point_b": point_b,
        "delta": None if point_a is None or point_b is None else float(point_a - point_b),
        "ci_low": float(np.quantile(deltas_array, 0.025)) if len(deltas_array) else None,
        "ci_high": float(np.quantile(deltas_array, 0.975)) if len(deltas_array) else None,
        "probability_positive": float(np.mean(deltas_array > 0)) if len(deltas_array) else None,
        "iterations": int(len(deltas_array)),
    }


def seed_noise_floor(frames: Iterable[pd.DataFrame], **kwargs: Any) -> dict[str, Any]:
    """Mean absolute pairwise difference of the primary metric across seeds."""

    values: list[float] = []
    for frame in frames:
        score = pseudo_macro_auc(frame, **kwargs)["macro_auc"]
        if score is None:
            raise ValueError("cannot compute a noise floor from an undefined metric")
        values.append(score)
    if len(values) < 2:
        raise ValueError("at least two seeds are required for a noise floor")
    array = np.asarray(values, dtype=float)
    pairwise = [abs(array[i] - array[j]) for i in range(len(array)) for j in range(i + 1, len(array))]
    return {
        "seed_scores": values,
        "mean_abs_pairwise_delta": float(np.mean(pairwise)),
        "max_abs_pairwise_delta": float(np.max(pairwise)),
        "std": float(np.std(array, ddof=1)),
    }


def summarize_scorecard(scorecard: dict[str, Any]) -> str:
    pseudo = scorecard["pseudo"]["macro_auc"]
    gold = scorecard["gold"]["macro_auc"]
    calibration = scorecard["calibration"]
    parts = [
        f"pseudo_macro_auc={pseudo:.5f}" if pseudo is not None else "pseudo_macro_auc=NA",
        f"gold_macro_auc={gold:.5f}" if gold is not None else "gold_macro_auc=NA",
        f"brier={calibration['brier']:.5f}" if calibration["brier"] is not None else "brier=NA",
        f"studies={scorecard['studies']}",
    ]
    return " ".join(parts)
