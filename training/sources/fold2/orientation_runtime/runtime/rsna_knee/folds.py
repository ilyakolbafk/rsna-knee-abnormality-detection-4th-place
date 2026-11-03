"""Deterministic report-grouped multilabel fold generation."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from .constants import ID_COLUMN, REPORT_COLUMN, TARGET_COLUMNS
from .reports import add_report_groups


def _stable_tiebreak(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def prepare_training_manifest(
    train: pd.DataFrame,
    pseudo: pd.DataFrame | None = None,
) -> pd.DataFrame:
    required = {ID_COLUMN, REPORT_COLUMN, *TARGET_COLUMNS}
    missing = sorted(required.difference(train.columns))
    if missing:
        raise ValueError(f"Training metadata is missing columns: {missing}")
    if train[ID_COLUMN].duplicated().any():
        raise ValueError("Training metadata must be unique by StudyInstanceUID")

    result = add_report_groups(train).reset_index(drop=True)
    gold = result[TARGET_COLUMNS].apply(pd.to_numeric, errors="coerce")

    if pseudo is None:
        effective = gold.copy()
        pseudo_confidence = pd.Series(1.0, index=result.index, dtype=float)
        target_confidence = pd.DataFrame(
            1.0, index=result.index, columns=TARGET_COLUMNS, dtype=float
        )
    else:
        pseudo_required = {ID_COLUMN, *TARGET_COLUMNS}
        pseudo_missing = sorted(pseudo_required.difference(pseudo.columns))
        if pseudo_missing:
            raise ValueError(f"Pseudo-label file is missing columns: {pseudo_missing}")
        if pseudo[ID_COLUMN].duplicated().any():
            raise ValueError("Pseudo-label file must be unique by StudyInstanceUID")
        pseudo_indexed = pseudo.set_index(ID_COLUMN)
        has_pseudo = result[ID_COLUMN].isin(pseudo_indexed.index)
        needs_pseudo = ~gold.notna().all(axis=1)
        missing_ids = result.loc[~has_pseudo & needs_pseudo, ID_COLUMN]
        if len(missing_ids):
            raise ValueError(
                f"Pseudo-label file is missing {len(missing_ids)} studies without complete gold labels"
            )
        aligned = pseudo_indexed.reindex(result[ID_COLUMN])
        effective = aligned[TARGET_COLUMNS].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        if "pseudo_confidence" in aligned:
            pseudo_confidence = pd.to_numeric(
                aligned["pseudo_confidence"], errors="coerce"
            ).fillna(1.0).clip(0.0, 1.0).reset_index(drop=True)
        else:
            pseudo_confidence = pd.Series(1.0, index=result.index, dtype=float)
        target_confidence = pd.DataFrame(index=result.index, columns=TARGET_COLUMNS, dtype=float)
        for target in TARGET_COLUMNS:
            confidence_column = f"confidence__{target}"
            if confidence_column in aligned:
                values = pd.to_numeric(
                    aligned[confidence_column], errors="coerce"
                ).reset_index(drop=True)
                target_confidence[target] = values.fillna(pseudo_confidence).clip(0.0, 1.0)
            else:
                target_confidence[target] = pseudo_confidence
        effective = effective.where(gold.isna(), gold)

    invalid = (~effective.isin([0, 1]) & effective.notna())
    if invalid.any().any():
        # Soft pseudo-labels in [0, 1] are valid; reject only values outside that interval.
        outside = ((effective < 0) | (effective > 1)) & effective.notna()
        if outside.any().any():
            raise ValueError("Labels and pseudo-labels must lie in [0, 1]")

    result[TARGET_COLUMNS] = effective.to_numpy(dtype=float)
    result["is_gold"] = gold.notna().all(axis=1)
    result["has_any_gold"] = gold.notna().any(axis=1)
    result["pseudo_confidence"] = pseudo_confidence.to_numpy(dtype=float)
    result["label_source"] = np.where(
        result["is_gold"], "gold", np.where(effective.notna().any(axis=1), "pseudo", "missing")
    )
    for target in TARGET_COLUMNS:
        result[f"label_mask__{target}"] = effective[target].notna().astype(np.uint8)
        result[f"confidence__{target}"] = np.where(
            gold[target].notna(),
            1.0,
            target_confidence[target].to_numpy(dtype=float),
        )
    return result


def assign_grouped_folds(
    manifest: pd.DataFrame,
    *,
    n_splits: int = 5,
    seed: int = 20260828,
) -> pd.DataFrame:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    required = {ID_COLUMN, "report_group", *TARGET_COLUMNS}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")

    numeric = manifest[TARGET_COLUMNS].apply(pd.to_numeric, errors="coerce")
    work = manifest[[ID_COLUMN, "report_group"]].copy()
    gold_indicator = manifest.get("is_gold", pd.Series(False, index=manifest.index)).astype(bool)
    for target in TARGET_COLUMNS:
        work[f"sum__{target}"] = numeric[target].fillna(0.0)
        work[f"obs__{target}"] = numeric[target].notna().astype(float)
        work[f"gold_sum__{target}"] = numeric[target].fillna(0.0) * gold_indicator.astype(float)
    work["rows"] = 1.0
    work["gold_rows"] = gold_indicator.astype(float)

    aggregation = {"rows": "sum", "gold_rows": "sum"}
    aggregation.update({f"sum__{target}": "sum" for target in TARGET_COLUMNS})
    aggregation.update({f"obs__{target}": "sum" for target in TARGET_COLUMNS})
    aggregation.update({f"gold_sum__{target}": "sum" for target in TARGET_COLUMNS})
    groups = work.groupby("report_group", sort=False).agg(aggregation).reset_index()
    if len(groups) < n_splits:
        raise ValueError(f"Need at least {n_splits} report groups, found {len(groups)}")

    target_sums = groups[[f"sum__{target}" for target in TARGET_COLUMNS]].to_numpy(float)
    target_obs = groups[[f"obs__{target}" for target in TARGET_COLUMNS]].to_numpy(float)
    gold_rows = groups["gold_rows"].to_numpy(float)
    gold_sums = groups[[f"gold_sum__{target}" for target in TARGET_COLUMNS]].to_numpy(float)
    rows = groups["rows"].to_numpy(float)
    total_sums = target_sums.sum(axis=0)
    total_obs = target_obs.sum(axis=0)
    ideal_sums = total_sums / n_splits
    ideal_obs = total_obs / n_splits
    ideal_rows = rows.sum() / n_splits
    ideal_gold_rows = gold_rows.sum() / n_splits
    ideal_gold_sums = gold_sums.sum(axis=0) / n_splits

    rarity = np.divide(
        target_sums,
        np.maximum(total_sums, 1e-6),
        out=np.zeros_like(target_sums),
        where=np.maximum(total_sums, 1e-6) > 0,
    ).sum(axis=1)
    gold_rarity = np.divide(
        gold_sums,
        np.maximum(gold_sums.sum(axis=0), 1e-6),
        out=np.zeros_like(gold_sums),
        where=np.maximum(gold_sums.sum(axis=0), 1e-6) > 0,
    ).sum(axis=1)
    observed_count = target_obs.sum(axis=1)
    order = sorted(
        range(len(groups)),
        key=lambda index: (
            -gold_rows[index],
            -gold_rarity[index],
            -observed_count[index],
            -rarity[index],
            -rows[index],
            _stable_tiebreak(str(groups.iloc[index]["report_group"]), seed),
        ),
    )

    fold_rows = np.zeros(n_splits, dtype=float)
    fold_sums = np.zeros((n_splits, len(TARGET_COLUMNS)), dtype=float)
    fold_obs = np.zeros((n_splits, len(TARGET_COLUMNS)), dtype=float)
    fold_gold_rows = np.zeros(n_splits, dtype=float)
    fold_gold_sums = np.zeros((n_splits, len(TARGET_COLUMNS)), dtype=float)
    group_fold: dict[str, int] = {}

    for position, group_index in enumerate(order):
        group_name = str(groups.iloc[group_index]["report_group"])
        if position < n_splits:
            chosen = position
        else:
            costs: list[tuple[float, float, int]] = []
            for fold in range(n_splits):
                candidate_rows = fold_rows.copy()
                candidate_sums = fold_sums.copy()
                candidate_obs = fold_obs.copy()
                candidate_gold_rows = fold_gold_rows.copy()
                candidate_gold_sums = fold_gold_sums.copy()
                candidate_rows[fold] += rows[group_index]
                candidate_sums[fold] += target_sums[group_index]
                candidate_obs[fold] += target_obs[group_index]
                candidate_gold_rows[fold] += gold_rows[group_index]
                candidate_gold_sums[fold] += gold_sums[group_index]

                row_cost = np.mean(
                    ((candidate_rows - ideal_rows) / max(ideal_rows, 1.0)) ** 2
                )
                sum_scale = np.maximum(ideal_sums, 1.0)
                obs_scale = np.maximum(ideal_obs, 1.0)
                sum_cost = np.mean(
                    ((candidate_sums - ideal_sums[None, :]) / sum_scale[None, :]) ** 2
                )
                obs_cost = np.mean(
                    ((candidate_obs - ideal_obs[None, :]) / obs_scale[None, :]) ** 2
                )
                gold_row_cost = np.mean(
                    (
                        (candidate_gold_rows - ideal_gold_rows)
                        / max(ideal_gold_rows, 1.0)
                    )
                    ** 2
                )
                gold_sum_scale = np.maximum(ideal_gold_sums, 1.0)
                gold_sum_cost = np.mean(
                    (
                        (candidate_gold_sums - ideal_gold_sums[None, :])
                        / gold_sum_scale[None, :]
                    )
                    ** 2
                )
                total_cost = (
                    2.0 * row_cost
                    + sum_cost
                    + 0.25 * obs_cost
                    + 4.0 * gold_row_cost
                    + 3.0 * gold_sum_cost
                )
                costs.append((float(total_cost), float(fold_rows[fold]), fold))
            chosen = min(costs)[2]

        group_fold[group_name] = chosen
        fold_rows[chosen] += rows[group_index]
        fold_sums[chosen] += target_sums[group_index]
        fold_obs[chosen] += target_obs[group_index]
        fold_gold_rows[chosen] += gold_rows[group_index]
        fold_gold_sums[chosen] += gold_sums[group_index]

    result = manifest.copy()
    result["fold"] = result["report_group"].map(group_fold).astype(int)
    if result.groupby("report_group")["fold"].nunique().max() != 1:
        raise AssertionError("Report group leakage detected after assignment")
    return result


def fold_diagnostics(manifest: pd.DataFrame) -> dict[str, Any]:
    if "fold" not in manifest:
        raise ValueError("Manifest has no fold column")
    diagnostics: dict[str, Any] = {
        "studies": int(len(manifest)),
        "report_groups": int(manifest["report_group"].nunique()),
        "folds": {},
        "report_group_leakage": bool(
            (manifest.groupby("report_group")["fold"].nunique() > 1).any()
        ),
    }
    for fold, part in manifest.groupby("fold", sort=True):
        labels = part[TARGET_COLUMNS].apply(pd.to_numeric, errors="coerce")
        diagnostics["folds"][str(int(fold))] = {
            "studies": int(len(part)),
            "report_groups": int(part["report_group"].nunique()),
            "gold_studies": int(part.get("is_gold", pd.Series(False, index=part.index)).sum()),
            "target_means": {
                target: (float(labels[target].mean()) if labels[target].notna().any() else None)
                for target in TARGET_COLUMNS
            },
            "target_observed": {
                target: int(labels[target].notna().sum()) for target in TARGET_COLUMNS
            },
        }
    return diagnostics
