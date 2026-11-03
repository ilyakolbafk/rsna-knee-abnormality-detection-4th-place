"""Deterministic, distribution-matched study selection for the image OOF pilot."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .constants import ID_COLUMN, TARGET_COLUMNS


def _selection_features(
    frame: pd.DataFrame,
    *,
    minimum_confidence: float,
) -> tuple[np.ndarray, list[str]]:
    scores = frame[TARGET_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    confidence = frame[
        [f"confidence__{target}" for target in TARGET_COLUMNS]
    ].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    active = confidence >= minimum_confidence
    positive = active & (scores >= 0.5)
    weighted_score = np.where(active, scores, 0.0)
    features = np.concatenate(
        [weighted_score, active.astype(float), positive.astype(float), confidence],
        axis=1,
    )
    names = (
        [f"weighted_score__{target}" for target in TARGET_COLUMNS]
        + [f"active__{target}" for target in TARGET_COLUMNS]
        + [f"positive__{target}" for target in TARGET_COLUMNS]
        + [f"confidence__{target}" for target in TARGET_COLUMNS]
    )
    return features, names


def _objective(
    selected_sum: np.ndarray,
    *,
    count: int,
    population_mean: np.ndarray,
    scale: np.ndarray,
    positive_slices: list[slice],
) -> float:
    selected_mean = selected_sum / count
    standardized = (selected_mean - population_mean) / scale
    value = float(np.mean(standardized**2))
    expected_positive = np.concatenate(
        [population_mean[positive_slice] for positive_slice in positive_slices]
    ) * count
    selected_positive = np.concatenate(
        [selected_sum[positive_slice] for positive_slice in positive_slices]
    )
    required = np.where(expected_positive >= 1.0, 1.0, 0.0)
    missing = np.maximum(required - selected_positive, 0.0)
    return value + float(2.0 * np.mean(missing**2))


def _match_population(
    features: np.ndarray,
    count: int,
    *,
    seed: int,
    restarts: int = 12,
    iterations: int = 2500,
) -> tuple[np.ndarray, float]:
    if not 0 < count <= len(features):
        raise ValueError("Selection count must be between one and population size")
    if count == len(features):
        return np.arange(len(features)), 0.0
    rng = np.random.default_rng(seed)
    population_mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 0.05)
    target_count = len(TARGET_COLUMNS)
    feature_block = 4 * target_count
    if features.shape[1] % feature_block:
        raise ValueError("Selection features must contain complete teacher blocks")
    positive_slices = [
        slice(offset + 2 * target_count, offset + 3 * target_count)
        for offset in range(0, features.shape[1], feature_block)
    ]
    best_indices: np.ndarray | None = None
    best_value = float("inf")

    for _ in range(restarts):
        selected = np.zeros(len(features), dtype=bool)
        selected[rng.choice(len(features), size=count, replace=False)] = True
        selected_sum = features[selected].sum(axis=0)
        current = _objective(
            selected_sum,
            count=count,
            population_mean=population_mean,
            scale=scale,
            positive_slices=positive_slices,
        )
        for _ in range(iterations):
            selected_positions = np.flatnonzero(selected)
            available_positions = np.flatnonzero(~selected)
            remove = int(rng.choice(selected_positions))
            add = int(rng.choice(available_positions))
            candidate_sum = selected_sum - features[remove] + features[add]
            candidate = _objective(
                candidate_sum,
                count=count,
                population_mean=population_mean,
                scale=scale,
                positive_slices=positive_slices,
            )
            if candidate < current:
                selected[remove] = False
                selected[add] = True
                selected_sum = candidate_sum
                current = candidate
        if current < best_value:
            best_value = current
            best_indices = np.flatnonzero(selected)
    if best_indices is None:
        raise AssertionError("Population matching failed to produce a selection")
    return best_indices, best_value


def select_pilot_studies(
    manifest: pd.DataFrame,
    *,
    balance_manifests: dict[str, pd.DataFrame] | None = None,
    pseudo_count: int = 300,
    minimum_confidence: float = 0.10,
    seed: int = 20260829,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        ID_COLUMN,
        "fold",
        "report_group",
        "is_gold",
        *TARGET_COLUMNS,
        *[f"confidence__{target}" for target in TARGET_COLUMNS],
    }
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"Pilot manifest is missing columns: {missing}")
    if manifest[ID_COLUMN].duplicated().any():
        raise ValueError("Pilot manifest must be unique by StudyInstanceUID")
    if pseudo_count <= 0:
        raise ValueError("pseudo_count must be positive")
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must lie in [0, 1]")

    gold = manifest.loc[manifest["is_gold"].astype(bool)].copy()
    pseudo = manifest.loc[~manifest["is_gold"].astype(bool)].copy()
    balancing = balance_manifests or {"primary": manifest}
    feature_parts: list[np.ndarray] = []
    feature_names: list[str] = []
    pseudo_ids = pseudo[ID_COLUMN].astype(str).tolist()
    for name, balance_manifest in balancing.items():
        if balance_manifest[ID_COLUMN].duplicated().any():
            raise ValueError(f"Balance manifest {name!r} is not unique by {ID_COLUMN}")
        indexed = balance_manifest.assign(
            **{ID_COLUMN: balance_manifest[ID_COLUMN].astype(str)}
        ).set_index(ID_COLUMN)
        missing_ids = sorted(set(pseudo_ids).difference(indexed.index))
        if missing_ids:
            raise ValueError(
                f"Balance manifest {name!r} is missing {len(missing_ids)} pseudo studies"
            )
        aligned = indexed.loc[pseudo_ids].reset_index()
        features, names = _selection_features(
            aligned, minimum_confidence=minimum_confidence
        )
        feature_parts.append(features)
        feature_names.extend(f"{name}::{feature}" for feature in names)
    pseudo_features = np.concatenate(feature_parts, axis=1)
    pseudo = pseudo.assign(
        _pilot_any_active=np.concatenate(
            [
                part[:, len(TARGET_COLUMNS) : 2 * len(TARGET_COLUMNS)]
                for part in feature_parts
            ],
            axis=1,
        ).any(axis=1)
    )
    eligible_mask = pseudo["_pilot_any_active"].to_numpy(bool)
    pseudo = pseudo.loc[pseudo["_pilot_any_active"]].copy()
    pseudo_features = pseudo_features[eligible_mask]
    if pseudo_count > len(pseudo):
        raise ValueError(
            f"Requested {pseudo_count} pseudo studies but only {len(pseudo)} have active labels"
        )

    folds = sorted(int(value) for value in manifest["fold"].unique())
    base, remainder = divmod(pseudo_count, len(folds))
    quotas = {
        fold: base + (1 if position < remainder else 0)
        for position, fold in enumerate(folds)
    }
    selected_parts: list[pd.DataFrame] = []
    fold_objectives: dict[str, float] = {}
    for fold in folds:
        candidates = pseudo.loc[pseudo["fold"].astype(int) == fold].copy()
        candidate_mask = pseudo["fold"].astype(int).to_numpy() == fold
        features = pseudo_features[candidate_mask]
        indices, objective = _match_population(
            features,
            quotas[fold],
            seed=seed + fold,
        )
        selected_parts.append(candidates.iloc[indices].copy())
        fold_objectives[str(fold)] = float(objective)

    selected_pseudo = pd.concat(selected_parts, ignore_index=True)
    selected = pd.concat([gold, selected_pseudo], ignore_index=True)
    selected["pilot_role"] = np.where(selected["is_gold"], "gold", "pseudo")
    selected = selected.sort_values(
        ["pilot_role", "fold", ID_COLUMN], ascending=[True, True, True]
    ).reset_index(drop=True)
    output = selected[
        [ID_COLUMN, "pilot_role", "fold", "report_group", "is_gold"]
    ].rename(columns={"fold": "selection_fold"})

    population_features = pseudo_features
    selected_ids = set(selected_pseudo[ID_COLUMN].astype(str))
    chosen_features = pseudo_features[
        pseudo[ID_COLUMN].astype(str).isin(selected_ids).to_numpy()
    ]
    population_mean = population_features.mean(axis=0)
    chosen_mean = chosen_features.mean(axis=0)
    balance = {
        feature_names[index]: {
            "population_mean": float(population_mean[index]),
            "selected_mean": float(chosen_mean[index]),
            "absolute_delta": float(abs(chosen_mean[index] - population_mean[index])),
        }
        for index in range(len(feature_names))
    }
    diagnostics: dict[str, Any] = {
        "studies": int(len(output)),
        "gold_studies": int((output["pilot_role"] == "gold").sum()),
        "pseudo_studies": int((output["pilot_role"] == "pseudo").sum()),
        "minimum_confidence": float(minimum_confidence),
        "seed": int(seed),
        "balance_manifests": list(balancing),
        "selection_folds": {
            str(fold): {
                "gold": int(
                    ((output["selection_fold"] == fold) & (output["pilot_role"] == "gold")).sum()
                ),
                "pseudo": int(
                    ((output["selection_fold"] == fold) & (output["pilot_role"] == "pseudo")).sum()
                ),
                "matching_objective": fold_objectives[str(fold)],
            }
            for fold in folds
        },
        "selected_report_groups": int(output["report_group"].nunique()),
        "duplicate_selected_report_groups": int(
            (output.groupby("report_group").size() > 1).sum()
        ),
        "maximum_absolute_feature_mean_delta": float(
            np.max(np.abs(chosen_mean - population_mean))
        ),
        "feature_balance": balance,
    }
    return output, diagnostics
