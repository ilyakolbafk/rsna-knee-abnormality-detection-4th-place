"""Evaluation utilities for report-derived weak labels on the 58 gold studies."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .constants import ID_COLUMN, REPORT_COLUMN, TARGET_COLUMNS
from .weak_labels import READER_ID, extract_report_frame


def select_complete_gold(train: pd.DataFrame) -> pd.DataFrame:
    """Return rows with all twelve authoritative image labels present."""

    required = {ID_COLUMN, REPORT_COLUMN, *TARGET_COLUMNS}
    missing = sorted(required.difference(train.columns))
    if missing:
        raise ValueError(f"Training frame is missing columns: {missing}")
    numeric = train[TARGET_COLUMNS].apply(pd.to_numeric, errors="coerce")
    gold = train.loc[numeric.notna().all(axis=1)].copy()
    gold[TARGET_COLUMNS] = numeric.loc[gold.index].astype(np.uint8)
    if gold[ID_COLUMN].duplicated().any():
        raise ValueError("Gold frame contains duplicate StudyInstanceUID values")
    return gold.reset_index(drop=True)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _bootstrap_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    rows = len(labels)
    macro_samples: list[float] = []
    target_samples: list[list[float]] = [[] for _ in TARGET_COLUMNS]
    for _ in range(iterations):
        sample = rng.integers(0, rows, size=rows)
        target_auc: list[float] = []
        for target_index in range(len(TARGET_COLUMNS)):
            y = labels[sample, target_index]
            if np.unique(y).size < 2:
                continue
            value = float(roc_auc_score(y, scores[sample, target_index]))
            target_samples[target_index].append(value)
            target_auc.append(value)
        if target_auc:
            macro_samples.append(float(np.mean(target_auc)))

    def interval(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"p025": None, "median": None, "p975": None, "valid_samples": 0}
        array = np.asarray(values, dtype=float)
        return {
            "p025": float(np.quantile(array, 0.025)),
            "median": float(np.quantile(array, 0.5)),
            "p975": float(np.quantile(array, 0.975)),
            "valid_samples": int(len(array)),
        }

    return {
        "iterations": int(iterations),
        "seed": int(seed),
        "macro_auc": interval(macro_samples),
        "per_target": {
            target: interval(target_samples[index])
            for index, target in enumerate(TARGET_COLUMNS)
        },
    }


def evaluate_report_teacher(
    train: pd.DataFrame,
    *,
    bootstrap_iterations: int = 2000,
    seed: int = 20260829,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Generate teacher scores and evaluate them against complete gold labels.

    Returns the all-train teacher frame, a long gold decision table, and a JSON-safe
    summary.  The public deterministic reader was developed with access to these
    gold rows, so this is a descriptive, selection-biased audit. Any future
    calibration or model comparison must use an outer grouped split.
    """

    teacher = extract_report_frame(train[[ID_COLUMN, REPORT_COLUMN]])
    gold = select_complete_gold(train)
    gold_ids = set(gold[ID_COLUMN].astype(str))
    teacher_ids = set(teacher[ID_COLUMN].astype(str))
    if not gold_ids.issubset(teacher_ids):
        raise ValueError("Teacher output is missing gold studies")

    merged = gold.merge(
        teacher,
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
        suffixes=("__gold", ""),
    )
    long_rows: list[dict[str, Any]] = []
    per_target: dict[str, dict[str, Any]] = {}
    labels_matrix = merged[[f"{target}__gold" for target in TARGET_COLUMNS]].to_numpy(
        dtype=np.uint8
    )
    scores_matrix = merged[TARGET_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(scores_matrix).all():
        raise ValueError("Teacher scores contain non-finite values")

    total_tp = total_fp = total_fn = total_tn = 0
    auc_values: list[float] = []
    for target_index, target in enumerate(TARGET_COLUMNS):
        labels = labels_matrix[:, target_index]
        scores = scores_matrix[:, target_index]
        predicted = scores >= 0.5
        positives = labels == 1
        negatives = ~positives
        tp = int(np.sum(predicted & positives))
        fp = int(np.sum(predicted & negatives))
        fn = int(np.sum(~predicted & positives))
        tn = int(np.sum(~predicted & negatives))
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn
        auc = float(roc_auc_score(labels, scores))
        auc_values.append(auc)
        state_column = f"evidence_state__{target}"
        explicit = merged[state_column].isin(
            ["explicit_positive", "explicit_negative", "mixed_explicit"]
        )
        per_target[target] = {
            "positives": int(positives.sum()),
            "negatives": int(negatives.sum()),
            "roc_auc": auc,
            "brier": float(np.mean((scores - labels) ** 2)),
            "agreement_at_0_5": float(np.mean(predicted == positives)),
            "precision_at_0_5": _safe_ratio(tp, tp + fp),
            "recall_at_0_5": _safe_ratio(tp, tp + fn),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "explicit_evidence_coverage": float(explicit.mean()),
            "mean_confidence": float(merged[f"confidence__{target}"].mean()),
            "evidence_states": {
                str(key): int(value)
                for key, value in merged[state_column].value_counts().items()
            },
        }
        for row_index, row in merged.iterrows():
            hard = int(predicted[row_index])
            gold_value = int(labels[row_index])
            if hard == 1 and gold_value == 1:
                outcome = "TP"
            elif hard == 1:
                outcome = "FP"
            elif gold_value == 1:
                outcome = "FN"
            else:
                outcome = "TN"
            long_rows.append(
                {
                    ID_COLUMN: str(row[ID_COLUMN]),
                    "report_group": str(row["report_group"]),
                    "dominant_script": str(row["dominant_script"]),
                    "target": target,
                    "gold": gold_value,
                    "score": float(scores[row_index]),
                    "confidence": float(row[f"confidence__{target}"]),
                    "evidence_state": str(row[state_column]),
                    "positive_evidence_count": int(
                        row[f"positive_evidence_count__{target}"]
                    ),
                    "negative_evidence_count": int(
                        row[f"negative_evidence_count__{target}"]
                    ),
                    "hard_prediction_at_0_5": hard,
                    "outcome_at_0_5": outcome,
                    "correct_at_0_5": bool(hard == gold_value),
                }
            )

    decisions = pd.DataFrame(long_rows)
    summary: dict[str, Any] = {
        "status": "DESCRIPTIVE_BASELINE",
        "reader_id": READER_ID,
        "train_studies": int(len(train)),
        "gold_studies": int(len(gold)),
        "gold_decisions": int(len(gold) * len(TARGET_COLUMNS)),
        "unique_gold_report_groups": int(merged["report_group"].nunique()),
        "gold_script_counts": {
            str(key): int(value)
            for key, value in merged["dominant_script"].value_counts().items()
        },
        "macro_auc": float(np.mean(auc_values)),
        "overall_at_0_5": {
            "agreement": float((total_tp + total_tn) / len(decisions)),
            "precision": _safe_ratio(total_tp, total_tp + total_fp),
            "recall": _safe_ratio(total_tp, total_tp + total_fn),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "tn": total_tn,
        },
        "per_target": per_target,
        "bootstrap": _bootstrap_auc(
            labels_matrix,
            scores_matrix,
            iterations=bootstrap_iterations,
            seed=seed,
        ),
        "interpretation_boundary": (
            "Descriptive 58-study audit only; no claim about private leaderboard, "
            "clinical validity, or image-model gain."
        ),
    }
    return teacher, decisions, summary
