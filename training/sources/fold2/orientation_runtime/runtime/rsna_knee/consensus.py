"""Pinned three-teacher consensus recovered from public Kaggle artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .constants import ID_COLUMN, TARGET_COLUMNS
from .weak_labels import _bootstrap_target_auc


CONSENSUS_VERSION = "public_three_teacher_consensus_v1_2026-08-29"
EVIDENCE_GATED_VERSION = (
    "public_three_teacher_consensus_evidence_gated_v2_2026-08-29"
)
PUBLIC_TEACHERS: dict[str, dict[str, str]] = {
    "pilkwang": {
        "dataset": "pilkwang/rsna-knee-llm-labels",
        "filename": "report_labels_v2.csv",
        "sha256": "6f704a7bdb2f894cc49445b19ba7c4378c3f548d3449e00361e10044bee40920",
    },
    "steven_v2": {
        "dataset": "stevenleehans/rsna-knee-llm-report-labels",
        "filename": "llm_labels_v2.csv",
        "sha256": "3c082a987939528a55202028f655bd6f9dc5c9e28eff5058bbae7ef088a9f144",
    },
    "lixin_sol56": {
        "dataset": "lixin73/rsna-knee-llm-report-labels-sol56",
        "filename": "labels_llm_gpt56sol.csv",
        "sha256": "a79b05e609336811d51c97f9a48f00ea8004244a1a80fe886504a62bb39867ae",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_public_teacher(path: str | Path, teacher_name: str) -> pd.DataFrame:
    if teacher_name not in PUBLIC_TEACHERS:
        raise ValueError(f"Unknown public teacher: {teacher_name}")
    path = Path(path)
    actual_hash = _sha256(path)
    expected_hash = PUBLIC_TEACHERS[teacher_name]["sha256"]
    if actual_hash != expected_hash:
        raise ValueError(
            f"Hash mismatch for {teacher_name}: expected {expected_hash}, got {actual_hash}"
        )
    frame = pd.read_csv(path, dtype={ID_COLUMN: str})
    _validate_teacher_frame(frame, teacher_name)
    return frame


def _validate_teacher_frame(frame: pd.DataFrame, teacher_name: str) -> None:
    missing = sorted({ID_COLUMN, *TARGET_COLUMNS}.difference(frame.columns))
    if missing:
        raise ValueError(f"{teacher_name} is missing columns: {missing}")
    if frame[ID_COLUMN].duplicated().any():
        raise ValueError(f"{teacher_name} contains duplicate StudyInstanceUID values")
    scores = frame[TARGET_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if scores.isna().any().any():
        raise ValueError(f"{teacher_name} contains missing or non-numeric scores")
    if not ((scores >= 0.0) & (scores <= 1.0)).all().all():
        raise ValueError(f"{teacher_name} contains scores outside [0, 1]")


def build_public_consensus(
    train: pd.DataFrame,
    teachers: dict[str, pd.DataFrame],
    *,
    weight_mode: str = "public_v52",
) -> pd.DataFrame:
    """Return public-v52 mean scores and selected training weights.

    ``public_v52`` reproduces the notebook exactly. ``evidence_gated`` keeps
    those scores but multiplies the weight by Pilkwang's explicit
    YES/NO/UNK confidence, preventing shared UNK scores from becoming strong
    negatives.
    """

    expected_names = set(PUBLIC_TEACHERS)
    if set(teachers) != expected_names:
        raise ValueError(
            f"Expected teachers {sorted(expected_names)}, got {sorted(teachers)}"
        )
    if ID_COLUMN not in train or train[ID_COLUMN].duplicated().any():
        raise ValueError("Train frame needs unique StudyInstanceUID values")
    for name, frame in teachers.items():
        _validate_teacher_frame(frame, name)
    if weight_mode not in {"public_v52", "evidence_gated"}:
        raise ValueError("weight_mode must be 'public_v52' or 'evidence_gated'")

    study_ids = train[[ID_COLUMN]].copy()
    study_ids[ID_COLUMN] = study_ids[ID_COLUMN].astype(str)
    aligned: list[np.ndarray] = []
    for name in PUBLIC_TEACHERS:
        source = teachers[name][[ID_COLUMN, *TARGET_COLUMNS]].copy()
        source[ID_COLUMN] = source[ID_COLUMN].astype(str)
        merged = study_ids.merge(source, on=ID_COLUMN, how="left", validate="one_to_one")
        aligned.append(merged[TARGET_COLUMNS].to_numpy(dtype=float))
    cube = np.stack(aligned, axis=2)
    availability = np.isfinite(cube).sum(axis=2)
    if np.any(availability < 2):
        target_index = np.argwhere(availability < 2)[0]
        raise ValueError(
            "Fewer than two public teachers for "
            f"{study_ids.iloc[target_index[0]][ID_COLUMN]} / "
            f"{TARGET_COLUMNS[target_index[1]]}"
        )

    mean_score = np.nanmean(cube, axis=2)
    disagreement = np.nanmean(np.abs(cube - mean_score[:, :, None]), axis=2)
    agreement = np.clip(1.0 - 2.0 * disagreement, 0.0, 1.0)
    certainty = np.clip(2.0 * np.abs(mean_score - 0.5), 0.0, 1.0)
    public_weight = 0.15 + 0.85 * (0.65 * agreement + 0.35 * certainty)
    evidence_gate = np.ones_like(public_weight)
    version = CONSENSUS_VERSION
    if weight_mode == "evidence_gated":
        pilkwang = teachers["pilkwang"]
        required_confidence = [f"{target}__conf" for target in TARGET_COLUMNS]
        missing_confidence = sorted(set(required_confidence).difference(pilkwang.columns))
        if missing_confidence:
            raise ValueError(
                f"Pilkwang teacher is missing confidence columns: {missing_confidence}"
            )
        confidence_source = pilkwang[[ID_COLUMN, *required_confidence]].copy()
        confidence_source[ID_COLUMN] = confidence_source[ID_COLUMN].astype(str)
        confidence_aligned = study_ids.merge(
            confidence_source,
            on=ID_COLUMN,
            how="left",
            validate="one_to_one",
        )
        evidence_gate = confidence_aligned[required_confidence].to_numpy(dtype=float)
        # The one missing Pilkwang row is a gold case and is overridden later.
        evidence_gate = np.nan_to_num(evidence_gate, nan=0.0)
        if not ((evidence_gate >= 0.0) & (evidence_gate <= 1.0)).all():
            raise ValueError("Pilkwang evidence confidence lies outside [0, 1]")
        version = EVIDENCE_GATED_VERSION
    confidence = public_weight * evidence_gate

    output = study_ids.copy()
    for target_index, target in enumerate(TARGET_COLUMNS):
        output[target] = mean_score[:, target_index]
        output[f"confidence__{target}"] = confidence[:, target_index]
        output[f"public_weight__{target}"] = public_weight[:, target_index]
        output[f"evidence_gate__{target}"] = evidence_gate[:, target_index]
        output[f"teacher_count__{target}"] = availability[:, target_index]
        output[f"agreement__{target}"] = agreement[:, target_index]
        output[f"certainty__{target}"] = certainty[:, target_index]
    output["pseudo_confidence"] = confidence.mean(axis=1)
    output["teacher_version"] = version
    return output


def _score_audit(
    train: pd.DataFrame,
    scores: pd.DataFrame,
) -> dict[str, Any]:
    numeric_gold = train[TARGET_COLUMNS].apply(pd.to_numeric, errors="coerce")
    complete = numeric_gold.notna().all(axis=1)
    gold = train.loc[complete, [ID_COLUMN]].copy()
    for target in TARGET_COLUMNS:
        gold[f"gold__{target}"] = numeric_gold.loc[complete, target].astype(np.uint8)
    merged = gold.merge(
        scores[[ID_COLUMN, *TARGET_COLUMNS]],
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
    )
    per_target: dict[str, dict[str, float | int]] = {}
    auc_values: list[float] = []
    ap_values: list[float] = []
    for target in TARGET_COLUMNS:
        labels = merged[f"gold__{target}"].to_numpy(dtype=np.uint8)
        predictions = merged[target].to_numpy(dtype=float)
        auc = float(roc_auc_score(labels, predictions))
        ap = float(average_precision_score(labels, predictions))
        auc_values.append(auc)
        ap_values.append(ap)
        per_target[target] = {"auc": auc, "average_precision": ap}
    return {
        "gold_rows_available": int(len(merged)),
        "macro_auc": float(np.mean(auc_values)),
        "macro_average_precision": float(np.mean(ap_values)),
        "targets": per_target,
    }


def evaluate_public_consensus(
    train: pd.DataFrame,
    consensus: pd.DataFrame,
    teachers: dict[str, pd.DataFrame],
    *,
    n_bootstrap: int = 2000,
    seed: int = 20260829,
) -> dict[str, Any]:
    consensus_audit = _score_audit(train, consensus)
    numeric_gold = train[TARGET_COLUMNS].apply(pd.to_numeric, errors="coerce")
    complete = numeric_gold.notna().all(axis=1)
    gold = train.loc[complete, [ID_COLUMN]].copy()
    for target in TARGET_COLUMNS:
        gold[f"gold__{target}"] = numeric_gold.loc[complete, target].astype(np.uint8)
    merged = gold.merge(consensus, on=ID_COLUMN, how="left", validate="one_to_one")
    for target_index, target in enumerate(TARGET_COLUMNS):
        labels = merged[f"gold__{target}"].to_numpy(dtype=np.uint8)
        scores = merged[target].to_numpy(dtype=float)
        low, high, valid = _bootstrap_target_auc(
            labels,
            scores,
            samples=n_bootstrap,
            seed=seed + target_index,
        )
        consensus_audit["targets"][target].update(
            {
                "auc_bootstrap_95_low": low,
                "auc_bootstrap_95_high": high,
                "valid_bootstrap_samples": int(valid),
                "mean_training_weight": float(
                    consensus[f"confidence__{target}"].mean()
                ),
            }
        )

    source_audits = {
        name: {
            **_score_audit(train, frame),
            "rows": int(len(frame)),
            "dataset": PUBLIC_TEACHERS[name]["dataset"],
            "filename": PUBLIC_TEACHERS[name]["filename"],
            "sha256": PUBLIC_TEACHERS[name]["sha256"],
        }
        for name, frame in teachers.items()
    }
    teacher_counts = np.stack(
        [
            consensus[f"teacher_count__{target}"].to_numpy(dtype=int)
            for target in TARGET_COLUMNS
        ],
        axis=1,
    )
    confidence = consensus[
        [f"confidence__{target}" for target in TARGET_COLUMNS]
    ].to_numpy(dtype=float)
    teacher_versions = consensus["teacher_version"].astype(str).unique().tolist()
    if len(teacher_versions) != 1:
        raise ValueError("Consensus output contains multiple teacher versions")
    weight_mode = (
        "evidence_gated"
        if teacher_versions[0] == EVIDENCE_GATED_VERSION
        else "public_v52"
    )
    return {
        "teacher_version": teacher_versions[0],
        "weight_mode": weight_mode,
        "studies": int(len(consensus)),
        "targets": len(TARGET_COLUMNS),
        "minimum_teacher_availability": int(teacher_counts.min()),
        "cells_with_fewer_than_two_teachers": int((teacher_counts < 2).sum()),
        "training_weight_min": float(confidence.min()),
        "training_weight_mean": float(confidence.mean()),
        "training_weight_max": float(confidence.max()),
        "consensus_descriptive_gold58": consensus_audit,
        "source_teachers_descriptive_gold": source_audits,
        "weight_formula": (
            "public_weight = 0.15 + 0.85 * (0.65 * agreement + 0.35 * "
            "certainty); agreement=clip(1-2*mean(abs(score-teacher_mean)),0,1); "
            "certainty=clip(2*abs(teacher_mean-0.5),0,1); training_weight = "
            + (
                "public_weight * Pilkwang YES/NO/UNK confidence"
                if weight_mode == "evidence_gated"
                else "public_weight"
            )
        ),
        "critical_limitations": [
            "The same 58 gold cases were used by public authors for comparison and iteration; AUC is descriptive and selection-biased.",
            "Steven and Lixin publish label CSVs but not fully reproducible generation pipelines.",
            "Agreement is not correctness, and the public weight formula can assign substantial weight when teachers share the same error.",
            "The evidence-gated mode is a local conservative adaptation, not a published leaderboard-validated formula.",
            "This audit does not establish an image-model or private-leaderboard gain.",
        ],
    }
