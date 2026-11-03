"""Deterministic, study-paired gates for staged image-model experiments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .constants import ID_COLUMN, TARGET_COLUMNS


FROZEN_GOLD_COLUMNS = [f"gold__{target}" for target in TARGET_COLUMNS]
FROZEN_PREDICTION_COLUMNS = [f"prediction__{target}" for target in TARGET_COLUMNS]
FINETUNE_PREDICTION_COLUMNS = [f"pred__{target}" for target in TARGET_COLUMNS]


def _require_columns(frame: pd.DataFrame, columns: list[str], *, source: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")


def _numeric_matrix(
    frame: pd.DataFrame, columns: list[str], *, source: str
) -> np.ndarray:
    matrix = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{source} contains non-finite values in {columns}")
    return matrix


def _validate_binary_gold(gold: np.ndarray, *, source: str) -> None:
    if not np.isin(gold, [0.0, 1.0]).all():
        raise ValueError(f"{source} gold labels must be exactly binary")


def _validate_probabilities(probability: np.ndarray, *, source: str) -> None:
    if ((probability < 0.0) | (probability > 1.0)).any():
        raise ValueError(f"{source} predictions must lie in [0, 1]")


def _macro_auc(gold: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    per_target: dict[str, float] = {}
    for target_index, target in enumerate(TARGET_COLUMNS):
        labels = gold[:, target_index]
        if np.unique(labels).size != 2:
            raise ValueError(f"Target {target!r} does not contain both classes")
        per_target[target] = float(roc_auc_score(labels, probability[:, target_index]))
    return {
        "macro_roc_auc": float(np.mean(list(per_target.values()))),
        "valid_targets": len(per_target),
        "per_target_roc_auc": per_target,
    }


def _fold_map(repeated_folds: pd.DataFrame) -> pd.DataFrame:
    required = [ID_COLUMN, "repeat", "fold"]
    _require_columns(repeated_folds, required, source="repeated folds")
    result = repeated_folds[required].copy()
    result[ID_COLUMN] = result[ID_COLUMN].astype(str)
    result["repeat"] = pd.to_numeric(result["repeat"], errors="raise").astype(int)
    result["fold"] = pd.to_numeric(result["fold"], errors="raise").astype(int)
    if result.duplicated([ID_COLUMN, "repeat"]).any():
        raise ValueError("Repeated folds contain duplicate study/repeat rows")
    if set(result["repeat"]) != {0, 1, 2}:
        raise ValueError("Repeated folds must contain repeats 0, 1, and 2")
    counts = result.groupby("repeat")[ID_COLUMN].nunique()
    if not counts.eq(58).all():
        raise ValueError(f"Repeated folds must cover 58 studies per repeat: {counts.to_dict()}")
    if set(result["fold"]) != {0, 1, 2, 3}:
        raise ValueError("Repeated folds must contain folds 0 through 3")
    return result


def _normalize_frozen(
    frame: pd.DataFrame,
    *,
    arm: str,
    folds: pd.DataFrame,
) -> pd.DataFrame:
    required = [
        "condition",
        "repeat",
        "fold",
        ID_COLUMN,
        *FROZEN_GOLD_COLUMNS,
        *FROZEN_PREDICTION_COLUMNS,
    ]
    _require_columns(frame, required, source=f"frozen arm {arm}")
    result = frame[required].copy()
    result[ID_COLUMN] = result[ID_COLUMN].astype(str)
    result["repeat"] = pd.to_numeric(result["repeat"], errors="raise").astype(int)
    result["fold"] = pd.to_numeric(result["fold"], errors="raise").astype(int)
    conditions = set(result["condition"].astype(str))
    if conditions != {"evidence_gated"}:
        raise ValueError(f"Frozen arm {arm} has unexpected conditions: {conditions}")
    if set(result["repeat"]) != {0, 1, 2}:
        raise ValueError(f"Frozen arm {arm} must contain repeats 0, 1, and 2")
    if result.duplicated([ID_COLUMN, "repeat"]).any():
        raise ValueError(f"Frozen arm {arm} contains duplicate study/repeat rows")
    counts = result.groupby("repeat")[ID_COLUMN].nunique()
    if not counts.eq(58).all():
        raise ValueError(
            f"Frozen arm {arm} must cover 58 studies per repeat: {counts.to_dict()}"
        )

    expected = folds.rename(columns={"fold": "expected_fold"})
    checked = result.merge(
        expected,
        on=[ID_COLUMN, "repeat"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not checked["_merge"].eq("both").all():
        raise ValueError(f"Frozen arm {arm} UID/repeat coverage differs from frozen folds")
    if not checked["fold"].eq(checked["expected_fold"]).all():
        raise ValueError(f"Frozen arm {arm} fold assignments drifted")

    gold = _numeric_matrix(result, FROZEN_GOLD_COLUMNS, source=f"frozen arm {arm}")
    probability = _numeric_matrix(
        result, FROZEN_PREDICTION_COLUMNS, source=f"frozen arm {arm}"
    )
    _validate_binary_gold(gold, source=f"frozen arm {arm}")
    _validate_probabilities(probability, source=f"frozen arm {arm}")
    return result.sort_values(["repeat", ID_COLUMN]).reset_index(drop=True)


def _normalize_last1(
    fold_frames: Mapping[int, pd.DataFrame], *, folds: pd.DataFrame
) -> pd.DataFrame:
    if set(fold_frames) != {0, 1, 2, 3}:
        raise ValueError("last1 inputs must contain exactly folds 0 through 3")
    expected_repeat = folds.loc[folds["repeat"] == 0, [ID_COLUMN, "fold"]]
    parts: list[pd.DataFrame] = []
    required = [ID_COLUMN, "is_gold", *TARGET_COLUMNS, *FINETUNE_PREDICTION_COLUMNS]
    for fold, frame in sorted(fold_frames.items()):
        source = f"last1 fold {fold}"
        _require_columns(frame, required, source=source)
        part = frame[required].copy()
        part[ID_COLUMN] = part[ID_COLUMN].astype(str)
        if part[ID_COLUMN].duplicated().any():
            raise ValueError(f"{source} contains duplicate studies")
        is_gold = part["is_gold"].astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )
        if is_gold.isna().any() or not is_gold.all():
            raise ValueError(f"{source} must contain gold validation rows only")
        expected_ids = set(
            expected_repeat.loc[expected_repeat["fold"] == fold, ID_COLUMN].astype(str)
        )
        if set(part[ID_COLUMN]) != expected_ids:
            raise ValueError(f"{source} UID coverage differs from frozen repeat-0 fold")
        gold = _numeric_matrix(part, TARGET_COLUMNS, source=source)
        probability = _numeric_matrix(part, FINETUNE_PREDICTION_COLUMNS, source=source)
        _validate_binary_gold(gold, source=source)
        _validate_probabilities(probability, source=source)
        part.insert(1, "fold", int(fold))
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    if len(result) != 58 or result[ID_COLUMN].nunique() != 58:
        raise ValueError("Combined last1 OOF must contain exactly 58 unique studies")
    return result.sort_values(ID_COLUMN).reset_index(drop=True)


def evaluate_last4_gate(
    frozen_frames: Mapping[str, pd.DataFrame],
    last1_fold_frames: Mapping[int, pd.DataFrame],
    repeated_folds: pd.DataFrame,
    *,
    bootstrap_iterations: int = 5000,
    bootstrap_seed: int = 20260829,
    minimum_observed_delta: float = 0.01,
    minimum_probability_positive: float = 0.80,
) -> dict[str, Any]:
    """Select the frozen incumbent and evaluate the preregistered last4 gate."""

    if not frozen_frames:
        raise ValueError("At least one frozen arm is required")
    if bootstrap_iterations <= 0:
        raise ValueError("bootstrap_iterations must be positive")
    if not 0.0 <= minimum_probability_positive <= 1.0:
        raise ValueError("minimum_probability_positive must lie in [0, 1]")

    folds = _fold_map(repeated_folds)
    frozen = {
        str(arm): _normalize_frozen(frame, arm=str(arm), folds=folds)
        for arm, frame in frozen_frames.items()
    }
    if len(frozen) != len(frozen_frames):
        raise ValueError("Frozen arm names must be unique strings")

    reference_arm = sorted(frozen)[0]
    reference = frozen[reference_arm]
    reference_gold = reference[["repeat", ID_COLUMN, *FROZEN_GOLD_COLUMNS]]
    for arm, frame in frozen.items():
        candidate = frame[["repeat", ID_COLUMN, *FROZEN_GOLD_COLUMNS]]
        if not candidate.equals(reference_gold):
            raise ValueError(f"Frozen arm {arm} is not aligned on identical gold rows")

    frozen_metrics: dict[str, Any] = {}
    for arm, frame in frozen.items():
        averaged = frame.groupby(ID_COLUMN, sort=True)[FROZEN_PREDICTION_COLUMNS].mean()
        gold = (
            frame.groupby(ID_COLUMN, sort=True)[FROZEN_GOLD_COLUMNS]
            .first()
            .to_numpy(float)
        )
        frozen_metrics[arm] = _macro_auc(gold, averaged.to_numpy(float))
    best_score = max(item["macro_roc_auc"] for item in frozen_metrics.values())
    incumbent = min(
        arm
        for arm, item in frozen_metrics.items()
        if best_score - item["macro_roc_auc"] <= 1e-12
    )

    last1 = _normalize_last1(last1_fold_frames, folds=folds)
    incumbent_repeat0 = frozen[incumbent].loc[
        frozen[incumbent]["repeat"] == 0
    ].sort_values(ID_COLUMN)
    if last1[ID_COLUMN].tolist() != incumbent_repeat0[ID_COLUMN].tolist():
        raise ValueError("last1 and frozen incumbent are not UID-aligned")
    last1_gold = _numeric_matrix(last1, TARGET_COLUMNS, source="combined last1 OOF")
    incumbent_gold = _numeric_matrix(
        incumbent_repeat0,
        FROZEN_GOLD_COLUMNS,
        source="frozen incumbent repeat 0",
    )
    if not np.array_equal(last1_gold, incumbent_gold):
        raise ValueError("last1 and frozen incumbent gold labels differ")
    last1_probability = _numeric_matrix(
        last1, FINETUNE_PREDICTION_COLUMNS, source="combined last1 OOF"
    )
    incumbent_probability = _numeric_matrix(
        incumbent_repeat0,
        FROZEN_PREDICTION_COLUMNS,
        source="frozen incumbent repeat 0",
    )
    last1_metrics = _macro_auc(last1_gold, last1_probability)
    incumbent_repeat0_metrics = _macro_auc(last1_gold, incumbent_probability)
    observed_delta = float(
        last1_metrics["macro_roc_auc"]
        - incumbent_repeat0_metrics["macro_roc_auc"]
    )

    rng = np.random.default_rng(bootstrap_seed)
    deltas: list[float] = []
    for _ in range(bootstrap_iterations):
        sample = rng.integers(0, len(last1_gold), size=len(last1_gold))
        sampled_gold = last1_gold[sample]
        last1_scores: list[float] = []
        incumbent_scores: list[float] = []
        for target_index in range(len(TARGET_COLUMNS)):
            labels = sampled_gold[:, target_index]
            if np.unique(labels).size != 2:
                continue
            last1_scores.append(
                float(
                    roc_auc_score(
                        labels, last1_probability[sample, target_index]
                    )
                )
            )
            incumbent_scores.append(
                float(
                    roc_auc_score(
                        labels, incumbent_probability[sample, target_index]
                    )
                )
            )
        if not last1_scores:
            raise RuntimeError("Bootstrap sample contains no valid binary targets")
        deltas.append(float(np.mean(last1_scores) - np.mean(incumbent_scores)))
    distribution = np.asarray(deltas)
    probability_positive = float(np.mean(distribution > 0.0))
    passes_delta = observed_delta >= minimum_observed_delta
    passes_probability = probability_positive >= minimum_probability_positive
    gate_pass = bool(passes_delta and passes_probability)

    return {
        "status": "PASS",
        "decision": "START_LAST4" if gate_pass else "STOP_AFTER_LAST1",
        "gate_pass": gate_pass,
        "claim_strength": "operational/directional only",
        "coverage": {
            "gold_studies": int(len(last1)),
            "targets": len(TARGET_COLUMNS),
            "frozen_repeats": 3,
            "last1_folds": 4,
        },
        "frozen_selection": {
            "incumbent": incumbent,
            "tie_tolerance": 1e-12,
            "arms": frozen_metrics,
        },
        "last1": last1_metrics,
        "incumbent_repeat0": incumbent_repeat0_metrics,
        "comparison": {
            "observed_delta_macro_roc_auc": observed_delta,
            "paired_bootstrap_95": [
                float(np.quantile(distribution, 0.025)),
                float(np.quantile(distribution, 0.975)),
            ],
            "bootstrap_probability_positive": probability_positive,
            "bootstrap_iterations": int(bootstrap_iterations),
            "bootstrap_seed": int(bootstrap_seed),
        },
        "thresholds": {
            "minimum_observed_delta_macro_roc_auc": float(
                minimum_observed_delta
            ),
            "minimum_probability_delta_positive": float(
                minimum_probability_positive
            ),
            "passes_observed_delta": bool(passes_delta),
            "passes_probability": bool(passes_probability),
        },
    }
