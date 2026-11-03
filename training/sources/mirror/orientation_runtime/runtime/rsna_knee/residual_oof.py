"""Pseudo-only weak image head and cross-fitted gold residual correction."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .constants import ID_COLUMN, TARGET_COLUMNS
from .feature_oof import (
    _feature_head,
    _manifest_arrays,
    _normalized_target_loss,
    _seed_everything,
    _torch,
    evaluate_repeated_oof,
)


WEAK_CONDITION = "pseudo_only_weak"
RESIDUAL_CONDITION = "pseudo_only_weak_plus_residual"


def _cache_indices(
    feature_cache: dict[str, np.ndarray], frame: pd.DataFrame
) -> np.ndarray:
    uid_to_index = {
        str(uid): index for index, uid in enumerate(feature_cache["uids"].astype(str))
    }
    uids = frame[ID_COLUMN].astype(str).tolist()
    missing = [uid for uid in uids if uid not in uid_to_index]
    if missing:
        raise ValueError(f"Feature cache is missing {len(missing)} studies")
    return np.asarray([uid_to_index[uid] for uid in uids], dtype=int)


def train_pseudo_only_weak_head(
    feature_cache: dict[str, np.ndarray],
    manifest: pd.DataFrame,
    evaluation_manifest: pd.DataFrame,
    config: dict[str, Any],
    *,
    seed: int,
    device_name: str,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Train one fixed-step image head without consuming any gold target value.

    The caller may pass a mixed manifest.  Gold rows are filtered before labels are
    materialized, which makes accidental gold supervision fail visibly in diagnostics.
    Returned values are logits, not probabilities, because they are the frozen offset
    for the subsequent residual fit.
    """

    torch = _torch()
    if manifest[ID_COLUMN].duplicated().any():
        raise ValueError("Weak manifest must be unique by StudyInstanceUID")
    if evaluation_manifest[ID_COLUMN].duplicated().any():
        raise ValueError("Evaluation manifest must be unique by StudyInstanceUID")
    if "is_gold" not in manifest:
        raise ValueError("Weak manifest must contain is_gold")
    pseudo = manifest.loc[~manifest["is_gold"].astype(bool)].copy()
    excluded_gold_rows = int(manifest["is_gold"].astype(bool).sum())
    if pseudo.empty:
        raise ValueError("Pseudo-only weak training requires at least one pseudo row")

    _seed_everything(seed)
    pseudo_indices = _cache_indices(feature_cache, pseudo)
    evaluation_indices = _cache_indices(feature_cache, evaluation_manifest)
    arrays = _manifest_arrays(pseudo)
    train_config = config["train"]
    minimum_confidence = float(train_config["pseudo_min_confidence"])
    effective_mask = arrays["mask"] & (
        arrays["confidence"] >= minimum_confidence
    )
    observed_by_target = effective_mask.sum(axis=0)
    if np.any(observed_by_target == 0):
        missing_targets = [
            TARGET_COLUMNS[index]
            for index, count in enumerate(observed_by_target)
            if count == 0
        ]
        raise ValueError(f"Pseudo labels are absent for targets: {missing_targets}")

    features_np = feature_cache["features"]
    token_mask_np = feature_cache["token_mask"]
    slot_ids_np = feature_cache["slot_ids"]
    device = torch.device(device_name)
    head = _feature_head(
        feature_dim=features_np.shape[-1],
        num_slots=len(config["data"]["slots"]),
        dropout=float(config["model"].get("dropout", 0.2)),
    ).to(device)
    steps = int(train_config["fixed_steps"])
    if steps <= 0:
        raise ValueError("fixed_steps must be positive")
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    generator = np.random.default_rng(seed)
    batch_size = int(train_config["pseudo_batch_size"])
    loss_history: list[float] = []

    head.train()
    for step in range(steps):
        rows = generator.choice(
            len(pseudo), size=batch_size, replace=len(pseudo) < batch_size
        )
        cache_rows = pseudo_indices[rows]
        features = torch.from_numpy(
            features_np[cache_rows].astype(np.float32)
        ).to(device)
        token_mask = torch.from_numpy(token_mask_np[cache_rows]).to(device)
        slot_ids = torch.from_numpy(
            slot_ids_np[cache_rows].astype(np.int64)
        ).to(device)
        labels = torch.from_numpy(arrays["labels"][rows]).to(device)
        label_mask = torch.from_numpy(effective_mask[rows]).to(device)
        confidence = torch.from_numpy(arrays["confidence"][rows]).to(device)
        weight = 0.2 + 0.8 * confidence
        logits = head(features, token_mask, slot_ids)
        loss = _normalized_target_loss(logits, labels, label_mask, weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            head.parameters(), float(train_config["gradient_clip"])
        )
        optimizer.step()
        scheduler.step()
        loss_history.append(float(loss.detach().cpu()))
        if progress is not None and (step + 1 == steps or (step + 1) % 25 == 0):
            progress(
                {
                    "step": step + 1,
                    "steps": steps,
                    "loss": loss_history[-1],
                }
            )

    head.eval()
    with torch.inference_mode():
        evaluation_features = torch.from_numpy(
            features_np[evaluation_indices].astype(np.float32)
        ).to(device)
        evaluation_token_mask = torch.from_numpy(
            token_mask_np[evaluation_indices]
        ).to(device)
        evaluation_slot_ids = torch.from_numpy(
            slot_ids_np[evaluation_indices].astype(np.int64)
        ).to(device)
        weak_logits = head(
            evaluation_features, evaluation_token_mask, evaluation_slot_ids
        ).cpu().numpy()

    diagnostics = {
        "seed": int(seed),
        "fixed_steps": steps,
        "pseudo_train_rows": int(len(pseudo)),
        "gold_rows_excluded_before_training": excluded_gold_rows,
        "evaluation_rows": int(len(evaluation_manifest)),
        "observed_pseudo_labels_by_target": {
            target: int(observed_by_target[index])
            for index, target in enumerate(TARGET_COLUMNS)
        },
        "initial_loss_mean": float(np.mean(loss_history[: min(10, steps)])),
        "final_loss_mean": float(np.mean(loss_history[-min(10, steps) :])),
        "minimum_loss": float(np.min(loss_history)),
    }
    return weak_logits, diagnostics


def apply_bias_free_residual(
    weak_logits: np.ndarray,
    weight: np.ndarray,
    *,
    cap: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply ``weak + cap*tanh(tanh(weak) @ W.T / cap)``."""

    weak = np.asarray(weak_logits, dtype=np.float64)
    matrix = np.asarray(weight, dtype=np.float64)
    targets = len(TARGET_COLUMNS)
    if weak.ndim != 2 or weak.shape[1] != targets:
        raise ValueError(f"weak_logits must have shape [rows, {targets}]")
    if matrix.shape != (targets, targets):
        raise ValueError(f"weight must have shape [{targets}, {targets}]")
    if cap <= 0:
        raise ValueError("cap must be positive")
    raw_delta = np.tanh(weak) @ matrix.T
    delta = cap * np.tanh(raw_delta / cap)
    return weak + delta, delta


def fit_bias_free_residual(
    train_weak_logits: np.ndarray,
    train_gold: np.ndarray,
    validation_weak_logits: np.ndarray,
    *,
    steps: int,
    learning_rate: float,
    weight_penalty: float,
    delta_penalty: float,
    cap: float,
    minimum_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit the zero-initialized 12x12 residual on outer-train gold only."""

    torch = _torch()
    _seed_everything(seed)
    weak = np.asarray(train_weak_logits, dtype=np.float64)
    gold = np.asarray(train_gold, dtype=np.float64)
    validation = np.asarray(validation_weak_logits, dtype=np.float64)
    targets = len(TARGET_COLUMNS)
    if weak.shape != gold.shape or weak.ndim != 2 or weak.shape[1] != targets:
        raise ValueError("train weak logits and gold must have matching [rows, 12] shape")
    if validation.ndim != 2 or validation.shape[1] != targets:
        raise ValueError("validation weak logits must have shape [rows, 12]")
    if not np.isin(gold, [0.0, 1.0]).all():
        raise ValueError("Residual gold labels must be fully observed binary values")
    if steps <= 0 or learning_rate <= 0:
        raise ValueError("Residual steps and learning rate must be positive")
    positives = gold.sum(axis=0).astype(int)
    negatives = (len(gold) - gold.sum(axis=0)).astype(int)
    eligible = (positives >= minimum_per_class) & (
        negatives >= minimum_per_class
    )
    if not eligible.any():
        raise ValueError("No residual target meets minimum_per_class")

    weak_tensor = torch.tensor(weak, dtype=torch.float64)
    gold_tensor = torch.tensor(gold, dtype=torch.float64)
    eligible_tensor = torch.tensor(eligible, dtype=torch.float64)
    weight = torch.nn.Parameter(torch.zeros((targets, targets), dtype=torch.float64))
    optimizer = torch.optim.Adam([weight], lr=learning_rate)
    loss_history: list[float] = []

    for _ in range(steps):
        effective_weight = weight * eligible_tensor[:, None]
        raw_delta = torch.tanh(weak_tensor) @ effective_weight.T
        delta = cap * torch.tanh(raw_delta / cap)
        final_logits = weak_tensor + delta
        raw_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            final_logits, gold_tensor, reduction="none"
        )
        data_loss = raw_loss[:, eligible].mean()
        # Average the full row L2 over eligible targets.  Using a per-parameter
        # mean would silently weaken lambda by another factor of 12.
        l2_weight = (
            effective_weight[eligible].square().sum()
            / eligible_tensor.sum()
        )
        l2_delta = delta[:, eligible].square().mean()
        loss = (
            data_loss
            + weight_penalty * l2_weight
            + delta_penalty * l2_delta
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.detach()))

    with torch.no_grad():
        effective_weight = weight * eligible_tensor[:, None]
        raw_delta = torch.tanh(weak_tensor) @ effective_weight.T
        delta_tensor = cap * torch.tanh(raw_delta / cap)
        final_tensor = weak_tensor + delta_tensor
        final_data_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            final_tensor, gold_tensor, reduction="none"
        )[:, eligible].mean()
        final_weight_l2 = (
            effective_weight[eligible].square().sum()
            / eligible_tensor.sum()
        )
        final_delta_l2 = delta_tensor[:, eligible].square().mean()
        final_objective = (
            final_data_loss
            + weight_penalty * final_weight_l2
            + delta_penalty * final_delta_l2
        )
    fitted_weight = (weight.detach().numpy() * eligible[:, None]).astype(np.float64)
    corrected, validation_delta = apply_bias_free_residual(
        validation, fitted_weight, cap=cap
    )
    train_corrected, train_delta = apply_bias_free_residual(
        weak, fitted_weight, cap=cap
    )
    diagnostics = {
        "seed": int(seed),
        "steps": int(steps),
        "learning_rate": float(learning_rate),
        "weight_penalty": float(weight_penalty),
        "delta_penalty": float(delta_penalty),
        "cap": float(cap),
        "minimum_per_class": int(minimum_per_class),
        "train_rows": int(len(gold)),
        "eligible_targets": [
            target for target, enabled in zip(TARGET_COLUMNS, eligible) if enabled
        ],
        "positive_by_target": {
            target: int(positives[index])
            for index, target in enumerate(TARGET_COLUMNS)
        },
        "negative_by_target": {
            target: int(negatives[index])
            for index, target in enumerate(TARGET_COLUMNS)
        },
        "initial_objective": float(loss_history[0]),
        "final_objective": float(final_objective),
        "weight_frobenius_norm": float(np.linalg.norm(fitted_weight)),
        "maximum_absolute_train_delta": float(np.abs(train_delta).max()),
        "maximum_absolute_validation_delta": float(
            np.abs(validation_delta).max()
        ),
        "train_final_logit_finite": bool(np.isfinite(train_corrected).all()),
        "weight": fitted_weight.tolist(),
    }
    return corrected, validation_delta, fitted_weight, diagnostics


def run_cross_fitted_residual(
    gold_manifest: pd.DataFrame,
    weak_logits: pd.DataFrame,
    repeated_folds: pd.DataFrame,
    residual_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Cross-fit the residual while keeping one frozen weak model for all folds."""

    if gold_manifest[ID_COLUMN].duplicated().any():
        raise ValueError("Gold manifest must be unique by StudyInstanceUID")
    gold = gold_manifest.copy()
    gold[ID_COLUMN] = gold[ID_COLUMN].astype(str)
    if "report_group" not in gold:
        raise ValueError("Gold manifest must contain report_group")
    if gold["report_group"].astype(str).duplicated().any():
        raise ValueError("Gold report groups must be atomic and unique in this pilot")
    labels = gold[TARGET_COLUMNS].apply(pd.to_numeric, errors="raise").to_numpy(float)
    weak_columns = [f"weak_logit__{target}" for target in TARGET_COLUMNS]
    if not isinstance(weak_logits, pd.DataFrame):
        raise TypeError("weak_logits must be a UID-keyed DataFrame")
    required_weak = {ID_COLUMN, *weak_columns}
    missing_weak = sorted(required_weak.difference(weak_logits.columns))
    if missing_weak:
        raise ValueError(f"Weak-logit frame is missing columns: {missing_weak}")
    weak_frame = weak_logits[[ID_COLUMN, *weak_columns]].copy()
    weak_frame[ID_COLUMN] = weak_frame[ID_COLUMN].astype(str)
    if weak_frame[ID_COLUMN].duplicated().any():
        raise ValueError("Weak-logit frame contains duplicate StudyInstanceUID values")
    if set(weak_frame[ID_COLUMN]) != set(gold[ID_COLUMN]):
        raise ValueError("Weak-logit and gold UID sets differ")
    weak = (
        weak_frame.set_index(ID_COLUMN)
        .loc[gold[ID_COLUMN], weak_columns]
        .to_numpy(dtype=np.float64)
    )
    if weak.shape != labels.shape:
        raise ValueError("weak_logits must align exactly with gold_manifest")
    uid_to_row = {uid: index for index, uid in enumerate(gold[ID_COLUMN])}
    if len(uid_to_row) != len(gold):
        raise ValueError("Gold StudyInstanceUID values are not unique")

    outputs: list[pd.DataFrame] = []
    diagnostics: dict[str, Any] = {}
    expected_uids = set(gold[ID_COLUMN])
    gold_group_by_uid = gold.set_index(ID_COLUMN)["report_group"].astype(str)
    for repeat, repeat_rows in repeated_folds.groupby("repeat", sort=True):
        repeat_uids = set(repeat_rows[ID_COLUMN].astype(str))
        if repeat_uids != expected_uids or len(repeat_rows) != len(gold):
            raise ValueError(f"Repeat {repeat} does not cover every gold study")
        fold_groups = repeat_rows.set_index(ID_COLUMN)["report_group"].astype(str)
        if not fold_groups.sort_index().equals(gold_group_by_uid.sort_index()):
            raise ValueError(f"Repeat {repeat} report groups disagree with gold manifest")
        for fold, validation_rows in repeat_rows.groupby("fold", sort=True):
            validation_ids = validation_rows[ID_COLUMN].astype(str).tolist()
            validation_index = np.asarray(
                [uid_to_row[uid] for uid in validation_ids], dtype=int
            )
            train_index = np.asarray(
                sorted(set(range(len(gold))).difference(validation_index)), dtype=int
            )
            validation_groups = set(validation_rows["report_group"].astype(str))
            train_groups = set(
                gold.iloc[train_index]["report_group"].astype(str)
            )
            if train_groups.intersection(validation_groups):
                raise AssertionError("Gold report-group leakage in residual split")
            corrected, delta, _, fold_diagnostics = fit_bias_free_residual(
                weak[train_index],
                labels[train_index],
                weak[validation_index],
                steps=int(residual_config["steps"]),
                learning_rate=float(residual_config["learning_rate"]),
                weight_penalty=float(residual_config["weight_penalty"]),
                delta_penalty=float(residual_config["delta_penalty"]),
                cap=float(residual_config["cap"]),
                minimum_per_class=int(residual_config["minimum_per_class"]),
                seed=int(validation_rows["seed"].iloc[0]) + int(fold),
            )
            for condition, logits, condition_delta in (
                (WEAK_CONDITION, weak[validation_index], np.zeros_like(delta)),
                (RESIDUAL_CONDITION, corrected, delta),
            ):
                frame = pd.DataFrame(
                    {
                        "condition": condition,
                        "repeat": int(repeat),
                        "fold": int(fold),
                        ID_COLUMN: validation_ids,
                    }
                )
                probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
                for target_index, target in enumerate(TARGET_COLUMNS):
                    frame[f"gold__{target}"] = labels[
                        validation_index, target_index
                    ]
                    frame[f"weak_logit__{target}"] = weak[
                        validation_index, target_index
                    ]
                    frame[f"delta_logit__{target}"] = condition_delta[
                        :, target_index
                    ]
                    frame[f"prediction__{target}"] = probability[:, target_index]
                outputs.append(frame)
            fold_diagnostics["validation_rows"] = int(len(validation_index))
            fold_diagnostics["train_uids"] = gold.iloc[train_index][
                ID_COLUMN
            ].tolist()
            fold_diagnostics["validation_uids"] = validation_ids
            diagnostics[f"repeat_{int(repeat)}_fold_{int(fold)}"] = fold_diagnostics

    result = pd.concat(outputs, ignore_index=True)
    expected = len(gold)
    counts = result.groupby(["condition", "repeat"])[ID_COLUMN].nunique()
    if not counts.eq(expected).all():
        raise RuntimeError("Residual OOF does not cover every gold study once per repeat")
    duplicates = result.groupby(["condition", "repeat", ID_COLUMN]).size()
    if not duplicates.eq(1).all():
        raise RuntimeError("Residual OOF contains duplicate held-out predictions")
    return result, diagnostics


def evaluate_residual_oof(
    predictions: pd.DataFrame,
    *,
    bootstrap_iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate weak versus residual with a paired study bootstrap."""

    averaged, summary = evaluate_repeated_oof(
        predictions,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )
    by_condition = {
        condition: frame.sort_values(ID_COLUMN).reset_index(drop=True)
        for condition, frame in averaged.groupby("condition", sort=False)
    }
    if WEAK_CONDITION not in by_condition or RESIDUAL_CONDITION not in by_condition:
        raise ValueError("Both weak and residual conditions are required")
    weak = by_condition[WEAK_CONDITION]
    residual = by_condition[RESIDUAL_CONDITION]
    if weak[ID_COLUMN].tolist() != residual[ID_COLUMN].tolist():
        raise ValueError("Weak and residual conditions are not study-aligned")
    gold_columns = [f"gold__{target}" for target in TARGET_COLUMNS]
    prediction_columns = [f"prediction__{target}" for target in TARGET_COLUMNS]
    y = weak[gold_columns].to_numpy(float)
    weak_probability = weak[prediction_columns].to_numpy(float)
    residual_probability = residual[prediction_columns].to_numpy(float)
    rng = np.random.default_rng(seed)
    auc_deltas: list[float] = []
    for _ in range(bootstrap_iterations):
        sample = rng.integers(0, len(y), size=len(y))
        target_deltas = []
        for target_index in range(len(TARGET_COLUMNS)):
            target_y = y[sample, target_index]
            if np.unique(target_y).size != 2:
                continue
            target_deltas.append(
                roc_auc_score(
                    target_y, residual_probability[sample, target_index]
                )
                - roc_auc_score(target_y, weak_probability[sample, target_index])
            )
        auc_deltas.append(float(np.mean(target_deltas)))
    delta = np.asarray(auc_deltas)
    weak_metrics = summary["conditions"][WEAK_CONDITION]
    residual_metrics = summary["conditions"][RESIDUAL_CONDITION]
    per_repeat: dict[str, Any] = {}
    for repeat, repeat_frame in predictions.groupby("repeat", sort=True):
        repeat_scores: dict[str, float] = {}
        for condition in (WEAK_CONDITION, RESIDUAL_CONDITION):
            condition_frame = repeat_frame.loc[
                repeat_frame["condition"] == condition
            ].sort_values(ID_COLUMN)
            repeat_y = condition_frame[gold_columns].to_numpy(float)
            repeat_probability = condition_frame[prediction_columns].to_numpy(float)
            target_scores = [
                roc_auc_score(repeat_y[:, index], repeat_probability[:, index])
                for index in range(len(TARGET_COLUMNS))
            ]
            repeat_scores[condition] = float(np.mean(target_scores))
        per_repeat[str(int(repeat))] = {
            "weak_macro_roc_auc": repeat_scores[WEAK_CONDITION],
            "residual_macro_roc_auc": repeat_scores[RESIDUAL_CONDITION],
            "delta_macro_roc_auc": float(
                repeat_scores[RESIDUAL_CONDITION]
                - repeat_scores[WEAK_CONDITION]
            ),
        }
    per_target_delta = {
        target: float(
            residual_metrics["per_target_roc_auc"][target]
            - weak_metrics["per_target_roc_auc"][target]
        )
        for target in TARGET_COLUMNS
    }
    positive_repeats = sum(
        item["delta_macro_roc_auc"] > 0 for item in per_repeat.values()
    )
    summary["comparisons"][f"{RESIDUAL_CONDITION}_minus_{WEAK_CONDITION}"] = {
        "observed_delta_macro_roc_auc": float(
            residual_metrics["macro_roc_auc"] - weak_metrics["macro_roc_auc"]
        ),
        "paired_bootstrap_95": [
            float(np.quantile(delta, 0.025)),
            float(np.quantile(delta, 0.975)),
        ],
        "bootstrap_probability_positive": float(np.mean(delta > 0)),
        "observed_delta_macro_average_precision": float(
            residual_metrics["macro_average_precision"]
            - weak_metrics["macro_average_precision"]
        ),
        "observed_delta_brier": float(
            residual_metrics["brier"] - weak_metrics["brier"]
        ),
        "observed_delta_log_loss": float(
            residual_metrics["log_loss"] - weak_metrics["log_loss"]
        ),
        "per_target_delta_roc_auc": per_target_delta,
        "per_repeat": per_repeat,
        "positive_repeat_count": int(positive_repeats),
        "repeat_count": int(len(per_repeat)),
        "bootstrap_note": "Study bootstrap of frozen OOF predictions; residual models are not refit inside bootstrap.",
    }
    return averaged, summary
