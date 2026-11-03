"""Teacher-neutral manifests and repeated gold-only outer OOF partitions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .constants import ID_COLUMN, TARGET_COLUMNS


def prepare_condition_manifests(
    pilot_ids: pd.DataFrame,
    conditions: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Align every teacher condition to the exact same frozen study subset."""

    if ID_COLUMN not in pilot_ids or pilot_ids[ID_COLUMN].duplicated().any():
        raise ValueError("Pilot IDs must be unique by StudyInstanceUID")
    wanted = pilot_ids[ID_COLUMN].astype(str).tolist()
    output: dict[str, pd.DataFrame] = {}
    gold_reference: pd.DataFrame | None = None
    for name, frame in conditions.items():
        if frame[ID_COLUMN].duplicated().any():
            raise ValueError(f"Condition {name!r} has duplicate StudyInstanceUID values")
        indexed = frame.assign(**{ID_COLUMN: frame[ID_COLUMN].astype(str)}).set_index(
            ID_COLUMN
        )
        missing = sorted(set(wanted).difference(indexed.index))
        if missing:
            raise ValueError(f"Condition {name!r} is missing {len(missing)} pilot studies")
        aligned = indexed.loc[wanted].reset_index()
        # Source manifests may carry an unrelated historical split.  All image
        # experiments use the separately frozen repeated_gold_folds.csv instead.
        aligned = aligned.drop(columns=["fold"], errors="ignore")
        if len(aligned) != len(wanted):
            raise AssertionError("Condition alignment changed the pilot row count")
        output[name] = aligned
        gold = aligned.loc[aligned["is_gold"].astype(bool), [ID_COLUMN, *TARGET_COLUMNS]]
        gold = gold.sort_values(ID_COLUMN).reset_index(drop=True)
        if gold_reference is None:
            gold_reference = gold
        else:
            pd.testing.assert_frame_equal(gold, gold_reference, check_dtype=False)
    if gold_reference is None or len(gold_reference) == 0:
        raise ValueError("No gold studies were found in the condition manifests")
    output["gold_only"] = next(iter(output.values())).loc[
        lambda frame: frame["is_gold"].astype(bool)
    ].reset_index(drop=True)
    return output


def _partition_cost(labels: np.ndarray, assignment: np.ndarray, n_splits: int) -> float:
    global_mean = labels.mean(axis=0)
    variance = np.maximum(global_mean * (1.0 - global_mean), 0.02)
    costs = []
    for fold in range(n_splits):
        part = labels[assignment == fold]
        if len(part) == 0:
            return float("inf")
        positives = part.sum(axis=0)
        if np.any(positives <= 0) or np.any(positives >= len(part)):
            return float("inf")
        costs.append(np.mean(((part.mean(axis=0) - global_mean) ** 2) / variance))
    return float(np.mean(costs))


def make_repeated_gold_folds(
    gold_manifest: pd.DataFrame,
    *,
    n_splits: int = 4,
    seeds: tuple[int, ...] = (20260829, 20260929, 20261029),
    trials: int = 20_000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Random-search balanced partitions without using model predictions.

    The current competition has one report group per gold study.  The function
    still validates and keeps report groups atomic so it fails closed if that
    assumption changes.
    """

    required = {ID_COLUMN, "report_group", "is_gold", *TARGET_COLUMNS}
    missing = sorted(required.difference(gold_manifest.columns))
    if missing:
        raise ValueError(f"Gold manifest is missing columns: {missing}")
    gold = gold_manifest.loc[gold_manifest["is_gold"].astype(bool)].copy()
    if len(gold) < n_splits * 2:
        raise ValueError("Too few gold studies for repeated outer folds")
    if gold[ID_COLUMN].duplicated().any():
        raise ValueError("Gold manifest must be unique by StudyInstanceUID")
    group_sizes = gold.groupby("report_group").size()
    if (group_sizes > 1).any():
        raise ValueError(
            "Repeated pilot splitter currently requires one gold study per report group"
        )
    labels = gold[TARGET_COLUMNS].apply(pd.to_numeric, errors="raise").to_numpy(float)
    if not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("Gold labels must be fully observed binary values")
    if trials <= 0:
        raise ValueError("trials must be positive")

    fold_sizes = np.full(n_splits, len(gold) // n_splits, dtype=int)
    fold_sizes[: len(gold) % n_splits] += 1
    template = np.concatenate(
        [np.full(size, fold, dtype=np.int16) for fold, size in enumerate(fold_sizes)]
    )
    parts: list[pd.DataFrame] = []
    diagnostics: dict[str, Any] = {"n_splits": n_splits, "repeats": {}}
    assignments: list[np.ndarray] = []
    for repeat, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        best_cost = float("inf")
        best_assignment: np.ndarray | None = None
        for _ in range(trials):
            candidate = template[rng.permutation(len(template))]
            cost = _partition_cost(labels, candidate, n_splits)
            if cost < best_cost:
                best_cost = cost
                best_assignment = candidate.copy()
        if best_assignment is None or not np.isfinite(best_cost):
            raise RuntimeError(
                f"Could not find a fully evaluable {n_splits}-fold partition for seed {seed}"
            )
        assignments.append(best_assignment)
        repeat_frame = gold[[ID_COLUMN, "report_group"]].copy()
        repeat_frame["repeat"] = repeat
        repeat_frame["seed"] = seed
        repeat_frame["fold"] = best_assignment.astype(int)
        parts.append(repeat_frame)
        fold_payload: dict[str, Any] = {}
        for fold in range(n_splits):
            mask = best_assignment == fold
            fold_payload[str(fold)] = {
                "gold_studies": int(mask.sum()),
                "target_positives": {
                    target: int(labels[mask, index].sum())
                    for index, target in enumerate(TARGET_COLUMNS)
                },
            }
        diagnostics["repeats"][str(repeat)] = {
            "seed": int(seed),
            "balance_cost": best_cost,
            "folds": fold_payload,
        }

    pairwise: dict[str, float] = {}
    for right in range(len(assignments)):
        for left in range(right):
            left_same = assignments[left][:, None] == assignments[left][None, :]
            right_same = assignments[right][:, None] == assignments[right][None, :]
            upper = np.triu_indices(len(gold), k=1)
            both = (left_same & right_same)[upper].sum()
            either = (left_same | right_same)[upper].sum()
            pairwise[f"{left}-{right}"] = float(both / max(either, 1))
    diagnostics["pairwise_same_fold_jaccard"] = pairwise
    result = pd.concat(parts, ignore_index=True)
    if (result.groupby(["repeat", "report_group"])["fold"].nunique() > 1).any():
        raise AssertionError("Report-group leakage in repeated folds")
    return result, diagnostics


def attach_outer_folds_for_training(
    supervision_manifest: pd.DataFrame,
    repeated_folds: pd.DataFrame,
    *,
    repeat: int,
    n_splits: int,
) -> pd.DataFrame:
    """Attach one frozen outer split and exclude report-group twins from training.

    Gold studies receive their frozen validation fold.  A pseudo-labelled study
    normally keeps the training-only sentinel ``-1``; when its normalized report
    group is also present among the gold studies, it receives that group's fold.
    ``train_fold`` then excludes the pseudo row from the matching outer fold while
    retaining it for the other folds.  This preserves report-group isolation
    without treating weakly labelled rows as validation observations.
    """

    required = {ID_COLUMN, "is_gold", "report_group"}
    missing = sorted(required.difference(supervision_manifest.columns))
    if missing:
        raise ValueError(f"Supervision manifest is missing columns: {missing}")
    if supervision_manifest[ID_COLUMN].duplicated().any():
        raise ValueError("Supervision manifest must be unique by StudyInstanceUID")
    selected = repeated_folds.loc[
        pd.to_numeric(repeated_folds["repeat"], errors="raise").astype(int) == repeat
    ].copy()
    if selected[ID_COLUMN].duplicated().any():
        raise ValueError(f"Repeat {repeat} assigns a gold study more than once")
    observed_folds = set(
        pd.to_numeric(selected["fold"], errors="raise").astype(int).tolist()
    )
    expected_folds = set(range(n_splits))
    if observed_folds != expected_folds:
        raise ValueError(
            f"Repeat {repeat} fold set mismatch: {sorted(observed_folds)} != "
            f"{sorted(expected_folds)}"
        )
    manifest = supervision_manifest.drop(columns=["fold"], errors="ignore").copy()
    manifest[ID_COLUMN] = manifest[ID_COLUMN].astype(str)
    assignment = selected[[ID_COLUMN, "report_group", "fold"]].copy()
    assignment[ID_COLUMN] = assignment[ID_COLUMN].astype(str)
    assignment["report_group"] = assignment["report_group"].astype(str)
    if (assignment.groupby("report_group")["fold"].nunique() > 1).any():
        raise ValueError("A frozen report group is assigned to multiple outer folds")
    group_assignment = (
        assignment[["report_group", "fold"]]
        .drop_duplicates("report_group")
        .set_index("report_group")["fold"]
    )
    assignment = assignment.rename(columns={"report_group": "_outer_report_group"})
    manifest = manifest.merge(
        assignment,
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    gold = manifest["is_gold"].astype(bool)
    if manifest.loc[gold, "fold"].isna().any():
        raise ValueError("At least one gold study lacks a frozen outer-fold assignment")
    if manifest.loc[~gold, "fold"].notna().any():
        raise ValueError("Repeated outer folds unexpectedly contain pseudo-labelled studies")
    if (
        manifest.loc[gold, "report_group"].astype(str)
        != manifest.loc[gold, "_outer_report_group"].astype(str)
    ).any():
        raise ValueError("Report-group identity drift between manifest and frozen folds")
    report_group_fold = manifest["report_group"].astype(str).map(group_assignment)
    if not np.array_equal(
        manifest.loc[gold, "fold"].astype(int).to_numpy(),
        report_group_fold.loc[gold].astype(int).to_numpy(),
    ):
        raise ValueError("Gold UID and report-group fold assignments disagree")
    manifest.loc[~gold, "fold"] = report_group_fold.loc[~gold]
    manifest["fold"] = manifest["fold"].fillna(-1).astype(int)
    manifest = manifest.drop(columns=["_outer_report_group"])
    if set(manifest.loc[gold, "fold"]) != expected_folds:
        raise ValueError("Gold rows do not cover every configured outer fold")
    return manifest
