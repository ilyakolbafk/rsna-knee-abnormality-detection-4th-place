"""Stable project wrapper around the frozen multilingual report rules."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from .constants import ID_COLUMN, REPORT_COLUMN, TARGET_COLUMNS
from .report_rules import TARGETS, extract
from .reports import dominant_script, report_hash


TEACHER_VERSION = "public_multilingual_rules_v2_2026-08-29"
READER_ID = TEACHER_VERSION


def _validate_report_frame(frame: pd.DataFrame) -> None:
    missing = sorted({ID_COLUMN, REPORT_COLUMN}.difference(frame.columns))
    if missing:
        raise ValueError(f"Report frame is missing columns: {missing}")
    if frame[ID_COLUMN].duplicated().any():
        raise ValueError("Report frame must be unique by StudyInstanceUID")
    if frame[REPORT_COLUMN].isna().any():
        raise ValueError("Report frame contains missing reports")
    if TARGETS != TARGET_COLUMNS:
        raise AssertionError("Rule-reader target order differs from the official schema")


def _state(positive: int, negative: int, score: float) -> str:
    if positive and negative:
        return "mixed_explicit"
    if positive:
        return "explicit_positive"
    if negative:
        return "explicit_negative"
    if not np.isclose(score, 0.28):
        return "implicit_or_uncertain"
    return "not_mentioned"


def run_rule_teacher(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return pseudo-label and evidence tables in stable input row order."""

    _validate_report_frame(frame)
    result_cache: dict[str, dict[str, Any]] = {}
    pseudo_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []

    for uid, report in zip(frame[ID_COLUMN], frame[REPORT_COLUMN], strict=True):
        group = report_hash(report)
        result = result_cache.setdefault(group, extract(str(report)))
        pseudo: dict[str, Any] = {ID_COLUMN: str(uid)}
        evidence: dict[str, Any] = {ID_COLUMN: str(uid)}
        confidences: list[float] = []
        for target in TARGET_COLUMNS:
            score = float(result[target])
            confidence = float(result[f"{target}__conf"])
            positive = int(result[f"{target}__npos"])
            negative = int(result[f"{target}__nneg"])
            if not 0.0 <= score <= 1.0 or not 0.0 <= confidence <= 1.0:
                raise ValueError(f"Invalid rule-reader output for {uid} / {target}")
            pseudo[target] = score
            pseudo[f"confidence__{target}"] = confidence
            confidences.append(confidence)
            evidence[f"positive_mentions__{target}"] = positive
            evidence[f"negative_mentions__{target}"] = negative
            evidence[f"silent__{target}"] = positive == 0 and negative == 0
            evidence[f"contradictory__{target}"] = positive > 0 and negative > 0
        pseudo["pseudo_confidence"] = float(np.mean(confidences))
        pseudo["teacher_version"] = TEACHER_VERSION
        pseudo_rows.append(pseudo)
        evidence_rows.append(evidence)

    pseudo_frame = pd.DataFrame(pseudo_rows)
    evidence_frame = pd.DataFrame(evidence_rows)
    expected_ids = frame[ID_COLUMN].astype(str).tolist()
    if pseudo_frame[ID_COLUMN].tolist() != expected_ids:
        raise AssertionError("Rule teacher changed report row order")
    if evidence_frame[ID_COLUMN].tolist() != expected_ids:
        raise AssertionError("Rule evidence changed report row order")
    return pseudo_frame, evidence_frame


def extract_report_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the richer reader table used by the 58-study audit."""

    pseudo, evidence = run_rule_teacher(frame)
    output = pseudo.merge(evidence, on=ID_COLUMN, validate="one_to_one")
    metadata = frame[[ID_COLUMN, REPORT_COLUMN]].copy()
    metadata[ID_COLUMN] = metadata[ID_COLUMN].astype(str)
    metadata["report_group"] = metadata[REPORT_COLUMN].map(report_hash)
    metadata["dominant_script"] = metadata[REPORT_COLUMN].map(dominant_script)
    output = metadata[[ID_COLUMN, "report_group", "dominant_script"]].merge(
        output, on=ID_COLUMN, validate="one_to_one"
    )
    output["reader_id"] = READER_ID
    for target in TARGET_COLUMNS:
        positive = output[f"positive_mentions__{target}"].astype(int)
        negative = output[f"negative_mentions__{target}"].astype(int)
        output[f"evidence_state__{target}"] = [
            _state(int(pos), int(neg), float(score))
            for pos, neg, score in zip(
                positive, negative, output[target], strict=True
            )
        ]
        output[f"positive_evidence_count__{target}"] = positive
        output[f"negative_evidence_count__{target}"] = negative
    return output


def target_confidence_columns(
    targets: Iterable[str] = TARGET_COLUMNS,
) -> list[str]:
    return [f"confidence__{target}" for target in targets]


def _bootstrap_target_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float | None, float | None, int]:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        indices = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size < 2:
            continue
        values.append(float(roc_auc_score(sampled_labels, scores[indices])))
    if not values:
        return None, None, 0
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
        len(values),
    )


def evaluate_rule_teacher(
    train: pd.DataFrame,
    pseudo: pd.DataFrame,
    evidence: pd.DataFrame,
    *,
    n_bootstrap: int = 2000,
    seed: int = 20260829,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Audit a frozen rule teacher on complete gold rows.

    The public rules were iterated with access to these same 58 studies.  The
    resulting metrics are descriptive and selection-biased, not an independent
    validation estimate.
    """

    from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

    _validate_report_frame(train)
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be non-negative")
    for name, frame in (("pseudo", pseudo), ("evidence", evidence)):
        if ID_COLUMN not in frame:
            raise ValueError(f"{name} frame is missing {ID_COLUMN}")
        if frame[ID_COLUMN].duplicated().any():
            raise ValueError(f"{name} frame contains duplicate study identifiers")

    numeric_gold = train[TARGET_COLUMNS].apply(pd.to_numeric, errors="coerce")
    complete = numeric_gold.notna().all(axis=1)
    gold = train.loc[complete, [ID_COLUMN]].copy()
    for target in TARGET_COLUMNS:
        gold[f"gold__{target}"] = numeric_gold.loc[complete, target].astype(np.uint8)
    merged = gold.merge(pseudo, on=ID_COLUMN, how="left", validate="one_to_one")
    merged = merged.merge(evidence, on=ID_COLUMN, how="left", validate="one_to_one")
    all_teacher = pseudo.merge(evidence, on=ID_COLUMN, how="left", validate="one_to_one")
    if len(merged) == 0:
        raise ValueError("No complete gold studies are available for teacher evaluation")

    targets: dict[str, dict[str, Any]] = {}
    error_rows: list[dict[str, Any]] = []
    auc_values: list[float] = []
    ap_values: list[float] = []
    brier_values: list[float] = []
    silence_values: list[float] = []
    confidence_values: list[float] = []

    for target_index, target in enumerate(TARGET_COLUMNS):
        required = [
            target,
            f"confidence__{target}",
            f"silent__{target}",
            f"contradictory__{target}",
        ]
        missing = [column for column in required if column not in merged]
        if missing:
            raise ValueError(f"Teacher evaluation is missing columns: {missing}")
        labels = merged[f"gold__{target}"].to_numpy(dtype=np.uint8)
        scores = pd.to_numeric(merged[target], errors="coerce").to_numpy(float)
        confidences = pd.to_numeric(
            merged[f"confidence__{target}"], errors="coerce"
        ).to_numpy(float)
        if not np.isfinite(scores).all() or not np.isfinite(confidences).all():
            raise ValueError(f"Non-finite teacher values for {target}")
        if np.unique(labels).size < 2:
            raise ValueError(f"Gold labels contain one class only for {target}")
        auc = float(roc_auc_score(labels, scores))
        average_precision = float(average_precision_score(labels, scores))
        brier = float(np.mean((scores - labels) ** 2))
        clipped = np.clip(scores, 1e-7, 1.0 - 1e-7)
        predicted = scores >= 0.5
        positive = labels == 1
        tp = int(np.sum(predicted & positive))
        fp = int(np.sum(predicted & ~positive))
        fn = int(np.sum(~predicted & positive))
        bootstrap_low, bootstrap_high, valid_samples = _bootstrap_target_auc(
            labels,
            scores,
            samples=n_bootstrap,
            seed=seed + target_index,
        )
        all_silent = pd.to_numeric(
            all_teacher[f"silent__{target}"], errors="coerce"
        ).to_numpy(float)
        all_confidence = pd.to_numeric(
            all_teacher[f"confidence__{target}"], errors="coerce"
        ).to_numpy(float)
        all_scores = pd.to_numeric(all_teacher[target], errors="coerce").to_numpy(float)
        all_contradictory = pd.to_numeric(
            all_teacher[f"contradictory__{target}"], errors="coerce"
        ).to_numpy(float)
        silence_rate = float(np.mean(all_silent))
        mean_confidence = float(np.mean(all_confidence))
        silence_values.append(silence_rate)
        confidence_values.append(mean_confidence)
        auc_values.append(auc)
        ap_values.append(average_precision)
        brier_values.append(brier)
        targets[target] = {
            "gold_positive": int(positive.sum()),
            "gold_negative": int((~positive).sum()),
            "auc": auc,
            "auc_bootstrap_95_low": bootstrap_low,
            "auc_bootstrap_95_high": bootstrap_high,
            "valid_bootstrap_samples": int(valid_samples),
            "average_precision": average_precision,
            "brier": brier,
            "log_loss": float(log_loss(labels, clipped, labels=[0, 1])),
            "threshold_0_5_precision": float(tp / (tp + fp)) if tp + fp else None,
            "threshold_0_5_recall": float(tp / (tp + fn)) if tp + fn else None,
            "threshold_0_5_predicted_positive": int(predicted.sum()),
            "gold_mean_confidence": float(np.mean(confidences)),
            "gold_silent": int(merged[f"silent__{target}"].sum()),
            "gold_contradictory": int(merged[f"contradictory__{target}"].sum()),
            "all_studies_silence_rate": silence_rate,
            "all_studies_contradiction_rate": float(np.mean(all_contradictory)),
            "all_studies_derived_positive_rate": float(np.mean(all_scores >= 0.5)),
            "all_studies_mean_confidence": mean_confidence,
        }
        for row_index in np.flatnonzero(predicted != positive):
            error_rows.append(
                {
                    ID_COLUMN: str(merged.iloc[row_index][ID_COLUMN]),
                    "target": target,
                    "gold": int(labels[row_index]),
                    "score": float(scores[row_index]),
                    "confidence": float(confidences[row_index]),
                    "error": "FP" if predicted[row_index] else "FN",
                    "silent": int(merged.iloc[row_index][f"silent__{target}"]),
                    "contradictory": int(
                        merged.iloc[row_index][f"contradictory__{target}"]
                    ),
                }
            )

    diagnostics: dict[str, Any] = {
        "teacher_version": TEACHER_VERSION,
        "upstream_notebook": (
            "https://www.kaggle.com/code/prvsiyan/"
            "rsna-knee-read-the-report-then-the-knee"
        ),
        "upstream_source_sha256": (
            "b51dd390fcf3fb3fea4791c00c65d64aa2ba9a04f95c457ebbde5d481b7ace83"
        ),
        "studies": int(len(train)),
        "gold_studies": int(complete.sum()),
        "pseudo_studies_without_complete_gold": int((~complete).sum()),
        "target_count": len(TARGET_COLUMNS),
        "macro_auc_descriptive_on_gold58": float(np.mean(auc_values)),
        "macro_average_precision_descriptive_on_gold58": float(np.mean(ap_values)),
        "macro_brier_descriptive_on_gold58": float(np.mean(brier_values)),
        "mean_silence_rate_all_studies": float(np.mean(silence_values)),
        "mean_confidence_all_studies": float(np.mean(confidence_values)),
        "critical_limitation": (
            "Descriptive only: the public rules were developed with access to the "
            "same 58 gold cases, so this is not an independent or cross-validated "
            "estimate."
        ),
        "targets": targets,
    }
    return diagnostics, pd.DataFrame(error_rows)
