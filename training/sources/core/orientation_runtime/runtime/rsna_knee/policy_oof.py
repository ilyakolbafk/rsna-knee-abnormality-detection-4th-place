"""Fail-closed policy manifests and paired feature-cache OOF evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .constants import ID_COLUMN, TARGET_COLUMNS
from .feature_oof import evaluate_repeated_oof
from .oof import prepare_condition_manifests


POLICY_MANIFEST_FORMAT = "rsna-knee-policy-manifests-v2"
POLICY_OOF_FORMAT = "rsna-knee-policy-feature-oof-v1"


@dataclass(frozen=True)
class PolicySpec:
    """Map one training condition to long-form probability and weight columns."""

    name: str
    probability_column: str
    weight_column: str


POLICY_SPECS: tuple[PolicySpec, ...] = (
    PolicySpec(
        "baseline",
        "train_baseline_probability",
        "train_baseline_weight",
    ),
    PolicySpec(
        "downweight",
        "train_downweight_probability",
        "train_downweight_weight",
    ),
    PolicySpec(
        "soft",
        "train_soft_probability",
        "train_soft_weight",
    ),
)


def _as_bool(series: pd.Series, *, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.casefold()
    result = normalized.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        }
    )
    if result.isna().any():
        examples = sorted(set(series.loc[result.isna()].astype(str)))[:3]
        raise ValueError(f"{name} contains invalid boolean values: {examples}")
    return result.astype(bool)


def _bounded_numeric(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    source: str,
) -> pd.DataFrame:
    numeric = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"{source} contains non-finite values in {list(columns)}")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError(f"{source} values must lie in [0, 1]")
    return numeric


def _canonical_long(manifest: pd.DataFrame) -> pd.DataFrame:
    blocks: list[pd.DataFrame] = []
    for target in TARGET_COLUMNS:
        block = manifest[
            [
                ID_COLUMN,
                "is_gold",
                target,
                f"label_mask__{target}",
                f"confidence__{target}",
            ]
        ].copy()
        block = block.rename(
            columns={
                target: "canonical_probability",
                f"label_mask__{target}": "canonical_mask",
                f"confidence__{target}": "canonical_weight",
            }
        )
        block["target"] = target
        blocks.append(block)
    result = pd.concat(blocks, ignore_index=True)
    return result.set_index([ID_COLUMN, "target"]).sort_index()


def _validate_baseline_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    required = {
        ID_COLUMN,
        "is_gold",
        "report_group",
        *TARGET_COLUMNS,
        *(f"label_mask__{target}" for target in TARGET_COLUMNS),
        *(f"confidence__{target}" for target in TARGET_COLUMNS),
    }
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"Baseline manifest is missing columns: {missing}")
    result = manifest.copy()
    if result[ID_COLUMN].isna().any():
        raise ValueError("Baseline manifest contains a null StudyInstanceUID")
    result[ID_COLUMN] = result[ID_COLUMN].astype(str)
    if result[ID_COLUMN].duplicated().any():
        raise ValueError("Baseline manifest must be unique by StudyInstanceUID")
    if result["report_group"].isna().any():
        raise ValueError("Baseline manifest contains a null report_group")
    result["report_group"] = result["report_group"].astype(str)
    result["is_gold"] = _as_bool(result["is_gold"], name="baseline is_gold")

    probabilities = _bounded_numeric(
        result,
        TARGET_COLUMNS,
        source="baseline probabilities",
    )
    result.loc[:, TARGET_COLUMNS] = probabilities
    for target in TARGET_COLUMNS:
        mask_column = f"label_mask__{target}"
        weight_column = f"confidence__{target}"
        result[mask_column] = _as_bool(
            result[mask_column], name=f"baseline {mask_column}"
        ).astype(int)
        result[weight_column] = _bounded_numeric(
            result,
            [weight_column],
            source=f"baseline {weight_column}",
        )[weight_column]

    gold = result["is_gold"]
    gold_values = result.loc[gold, TARGET_COLUMNS].to_numpy(float)
    if not np.isin(gold_values, [0.0, 1.0]).all():
        raise ValueError("Canonical Gold labels must be exactly binary")
    for target in TARGET_COLUMNS:
        if not result.loc[gold, f"confidence__{target}"].eq(1.0).all():
            raise ValueError("Canonical Gold confidence must be exactly 1")
    return result


def _validate_policy_cells(
    canonical: pd.DataFrame,
    policy_cells: pd.DataFrame,
    specs: Sequence[PolicySpec],
) -> pd.DataFrame:
    names = [spec.name for spec in specs]
    if not names or len(set(names)) != len(names):
        raise ValueError("Policy condition names must be non-empty and unique")
    if any(not name.replace("_", "").isalnum() for name in names):
        raise ValueError("Policy condition names may contain letters, digits, and underscores")
    required = {
        ID_COLUMN,
        "target",
        "gold",
        "is_gold",
        *(spec.probability_column for spec in specs),
        *(spec.weight_column for spec in specs),
    }
    missing = sorted(required.difference(policy_cells.columns))
    if missing:
        raise ValueError(f"Policy cells are missing columns: {missing}")

    cells = policy_cells.copy()
    if cells[[ID_COLUMN, "target"]].isna().any().any():
        raise ValueError("Policy cells contain a null StudyInstanceUID or target")
    cells[ID_COLUMN] = cells[ID_COLUMN].astype(str)
    cells["target"] = cells["target"].astype(str)
    if cells.duplicated([ID_COLUMN, "target"]).any():
        raise ValueError("Policy cells must be unique by StudyInstanceUID and target")

    unknown_targets = sorted(set(cells["target"]).difference(TARGET_COLUMNS))
    if unknown_targets:
        raise ValueError(f"Policy cells contain unknown targets: {unknown_targets}")

    expected_index = pd.MultiIndex.from_product(
        [canonical[ID_COLUMN].astype(str), TARGET_COLUMNS],
        names=[ID_COLUMN, "target"],
    )
    observed_index = pd.MultiIndex.from_frame(cells[[ID_COLUMN, "target"]])
    missing_index = expected_index.difference(observed_index)
    extra_index = observed_index.difference(expected_index)
    if len(missing_index) or len(extra_index):
        raise ValueError(
            "Policy cell coverage differs from the canonical study-target grid: "
            f"missing={len(missing_index)}, extra={len(extra_index)}"
        )
    cells = cells.set_index([ID_COLUMN, "target"]).loc[expected_index].copy()
    cells["is_gold"] = _as_bool(cells["is_gold"], name="policy is_gold")

    canonical_long = _canonical_long(canonical).loc[expected_index]
    if not np.array_equal(
        cells["is_gold"].to_numpy(bool),
        canonical_long["is_gold"].to_numpy(bool),
    ):
        raise ValueError("Policy is_gold values differ from the canonical manifest")

    gold_values = pd.to_numeric(cells["gold"], errors="coerce")
    gold_mask = canonical_long["is_gold"].to_numpy(bool)
    if gold_values.loc[~gold_mask].notna().any():
        raise ValueError("Policy gold must be null for non-Gold cells")
    if gold_values.loc[gold_mask].isna().any():
        raise ValueError("Policy gold is missing for at least one Gold cell")
    if not np.array_equal(
        gold_values.loc[gold_mask].to_numpy(float),
        canonical_long.loc[gold_mask, "canonical_probability"].to_numpy(float),
    ):
        raise ValueError("Policy gold values differ from canonical Gold labels")

    value_columns: list[str] = []
    for spec in specs:
        value_columns.extend([spec.probability_column, spec.weight_column])
    numeric = _bounded_numeric(cells, value_columns, source="policy cells")
    cells.loc[:, value_columns] = numeric

    baseline_spec = next((spec for spec in specs if spec.name == "baseline"), None)
    if baseline_spec is None:
        raise ValueError("Policy specs must contain a baseline condition")
    if not np.array_equal(
        cells[baseline_spec.probability_column].to_numpy(float),
        canonical_long["canonical_probability"].to_numpy(float),
    ):
        raise ValueError("Policy baseline probabilities do not reproduce the manifest")
    if not np.array_equal(
        cells[baseline_spec.weight_column].to_numpy(float),
        canonical_long["canonical_weight"].to_numpy(float),
    ):
        raise ValueError("Policy baseline weights do not reproduce the manifest")

    for spec in specs:
        if not np.array_equal(
            cells.loc[gold_mask, spec.probability_column].to_numpy(float),
            canonical_long.loc[gold_mask, "canonical_probability"].to_numpy(float),
        ):
            raise ValueError(f"Condition {spec.name!r} changes canonical Gold labels")
        if not cells.loc[gold_mask, spec.weight_column].eq(1.0).all():
            raise ValueError(f"Condition {spec.name!r} changes canonical Gold weights")
    return cells


def _condition_diagnostics(
    manifest: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    minimum_confidence: float,
) -> dict[str, Any]:
    pseudo = ~manifest["is_gold"].astype(bool).to_numpy()
    changed_probability = 0
    changed_weight = 0
    probability_l1 = 0.0
    weight_l1 = 0.0
    active_cells = 0
    effective_weight_sum = 0.0
    raw_weight_sum = 0.0
    per_target: dict[str, Any] = {}
    for target in TARGET_COLUMNS:
        probability = manifest[target].to_numpy(float)
        baseline_probability = baseline[target].to_numpy(float)
        weight = manifest[f"confidence__{target}"].to_numpy(float)
        baseline_weight = baseline[f"confidence__{target}"].to_numpy(float)
        label_mask = manifest[f"label_mask__{target}"].astype(bool).to_numpy()
        probability_delta = np.abs(probability - baseline_probability)
        weight_delta = np.abs(weight - baseline_weight)
        effective = pseudo & label_mask & (weight >= minimum_confidence)
        target_active = int(effective.sum())
        target_effective_weight = float(weight[effective].sum())
        target_raw_weight = float(weight[pseudo].sum())
        per_target[target] = {
            "pseudo_cells": int(pseudo.sum()),
            "changed_probability_cells": int((pseudo & (probability_delta > 0.0)).sum()),
            "changed_weight_cells": int((pseudo & (weight_delta > 0.0)).sum()),
            "active_pseudo_cells": target_active,
            "inactive_pseudo_cells": int(pseudo.sum()) - target_active,
            "raw_weight_sum": target_raw_weight,
            "effective_weight_sum": target_effective_weight,
        }
        changed_probability += per_target[target]["changed_probability_cells"]
        changed_weight += per_target[target]["changed_weight_cells"]
        probability_l1 += float(probability_delta[pseudo].sum())
        weight_l1 += float(weight_delta[pseudo].sum())
        active_cells += target_active
        effective_weight_sum += target_effective_weight
        raw_weight_sum += target_raw_weight

    pseudo_cells = int(pseudo.sum()) * len(TARGET_COLUMNS)
    return {
        "pseudo_cells": pseudo_cells,
        "changed_probability_cells": int(changed_probability),
        "changed_weight_cells": int(changed_weight),
        "probability_l1_change": probability_l1,
        "mean_absolute_probability_change": probability_l1 / max(pseudo_cells, 1),
        "weight_l1_change": weight_l1,
        "active_pseudo_cells": int(active_cells),
        "inactive_pseudo_cells": int(pseudo_cells - active_cells),
        "raw_weight_sum": raw_weight_sum,
        "effective_weight_sum": effective_weight_sum,
        "per_target": per_target,
    }


def materialize_policy_manifests(
    baseline_manifest: pd.DataFrame,
    policy_cells: pd.DataFrame,
    *,
    specs: Sequence[PolicySpec] = POLICY_SPECS,
    minimum_confidence: float = 0.10,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Materialize three exact-schema manifests from a complete long policy grid."""

    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must lie in [0, 1]")
    canonical = _validate_baseline_manifest(baseline_manifest)
    cells = _validate_policy_cells(canonical, policy_cells, specs)
    expected_columns = canonical.columns.tolist()
    uids = canonical[ID_COLUMN].astype(str).tolist()
    manifests: dict[str, pd.DataFrame] = {}
    for spec in specs:
        manifest = canonical.copy()
        for target in TARGET_COLUMNS:
            target_cells = cells.xs(target, level="target").loc[uids]
            manifest[target] = target_cells[spec.probability_column].to_numpy(float)
            manifest[f"confidence__{target}"] = target_cells[
                spec.weight_column
            ].to_numpy(float)
        if manifest.columns.tolist() != expected_columns:
            raise AssertionError("Policy materialization changed the manifest schema")
        manifests[spec.name] = manifest

    aligned = prepare_condition_manifests(
        canonical[[ID_COLUMN]],
        manifests,
    )
    manifests = {spec.name: aligned[spec.name] for spec in specs}
    immutable_columns = [
        column
        for column in expected_columns
        if column not in TARGET_COLUMNS
        and not column.startswith("confidence__")
    ]
    canonical_immutable = canonical[immutable_columns].reset_index(drop=True)
    for name, manifest in manifests.items():
        if manifest.columns.tolist() != expected_columns:
            raise AssertionError(f"Condition {name!r} changed the manifest schema")
        pd.testing.assert_frame_equal(
            manifest[immutable_columns].reset_index(drop=True),
            canonical_immutable,
            check_dtype=False,
            check_exact=True,
        )

    diagnostics = {
        "format": POLICY_MANIFEST_FORMAT,
        "status": "PASS",
        "studies": int(len(canonical)),
        "cells": int(len(canonical) * len(TARGET_COLUMNS)),
        "gold_studies": int(canonical["is_gold"].sum()),
        "pseudo_studies": int((~canonical["is_gold"]).sum()),
        "targets": len(TARGET_COLUMNS),
        "minimum_confidence": float(minimum_confidence),
        "conditions": {},
        "gates": {
            "complete_study_target_grid": True,
            "canonical_uid_order_preserved": True,
            "canonical_schema_preserved": True,
            "canonical_gold_preserved": True,
            "canonical_report_groups_preserved": True,
            "baseline_round_trip_exact": True,
            "all_probabilities_and_weights_bounded": True,
        },
    }
    for spec in specs:
        diagnostics["conditions"][spec.name] = _condition_diagnostics(
            manifests[spec.name],
            manifests["baseline"],
            minimum_confidence=minimum_confidence,
        )
    baseline_effective = diagnostics["conditions"]["baseline"][
        "effective_weight_sum"
    ]
    for condition in diagnostics["conditions"].values():
        condition["effective_weight_retained_vs_baseline"] = (
            float(condition["effective_weight_sum"] / baseline_effective)
            if baseline_effective > 0
            else None
        )
    return manifests, diagnostics


def _validate_policy_predictions(
    predictions: pd.DataFrame,
    *,
    conditions: Sequence[str],
) -> pd.DataFrame:
    gold_columns = [f"gold__{target}" for target in TARGET_COLUMNS]
    prediction_columns = [f"prediction__{target}" for target in TARGET_COLUMNS]
    required = {
        "condition",
        "repeat",
        "fold",
        ID_COLUMN,
        *gold_columns,
        *prediction_columns,
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Policy OOF predictions are missing columns: {missing}")
    frame = predictions.copy()
    if frame[["condition", "repeat", "fold", ID_COLUMN]].isna().any().any():
        raise ValueError("Policy OOF predictions contain null key fields")
    frame["condition"] = frame["condition"].astype(str)
    frame[ID_COLUMN] = frame[ID_COLUMN].astype(str)
    frame["repeat"] = pd.to_numeric(frame["repeat"], errors="raise").astype(int)
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    expected_conditions = list(conditions)
    if len(set(expected_conditions)) != len(expected_conditions):
        raise ValueError("Expected policy conditions must be unique")
    observed_conditions = set(frame["condition"])
    if observed_conditions != set(expected_conditions):
        raise ValueError(
            "Policy OOF condition set mismatch: "
            f"{sorted(observed_conditions)} != {sorted(expected_conditions)}"
        )
    if frame.duplicated(["condition", "repeat", ID_COLUMN]).any():
        raise ValueError("Policy OOF predictions duplicate a condition/repeat/study row")
    frame.loc[:, gold_columns] = _bounded_numeric(
        frame,
        gold_columns,
        source="policy OOF gold labels",
    )
    gold_values = frame[gold_columns].to_numpy(float)
    if not np.isin(gold_values, [0.0, 1.0]).all():
        raise ValueError("Policy OOF gold labels must be exactly binary")
    frame.loc[:, prediction_columns] = _bounded_numeric(
        frame,
        prediction_columns,
        source="policy OOF predictions",
    )

    reference_name = expected_conditions[0]
    reference = frame.loc[frame["condition"] == reference_name].sort_values(
        ["repeat", ID_COLUMN]
    )
    reference_keys = reference[["repeat", "fold", ID_COLUMN]].reset_index(drop=True)
    reference_gold = reference[gold_columns].reset_index(drop=True)
    for condition in expected_conditions[1:]:
        candidate = frame.loc[frame["condition"] == condition].sort_values(
            ["repeat", ID_COLUMN]
        )
        if not candidate[["repeat", "fold", ID_COLUMN]].reset_index(drop=True).equals(
            reference_keys
        ):
            raise ValueError(f"Condition {condition!r} is not aligned to the reference folds")
        if not np.array_equal(
            candidate[gold_columns].to_numpy(float),
            reference_gold.to_numpy(float),
        ):
            raise ValueError(f"Condition {condition!r} changes OOF Gold labels")
    return frame


def evaluate_policy_repeated_oof(
    predictions: pd.DataFrame,
    *,
    reference_condition: str = "baseline",
    candidate_conditions: Sequence[str] = ("downweight", "soft"),
    bootstrap_iterations: int = 5000,
    seed: int = 20260829,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Average repeats and compare every candidate with one aligned baseline."""

    if bootstrap_iterations <= 0:
        raise ValueError("bootstrap_iterations must be positive")
    conditions = [reference_condition, *candidate_conditions]
    frame = _validate_policy_predictions(predictions, conditions=conditions)
    averaged, summary = evaluate_repeated_oof(
        frame,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )
    gold_columns = [f"gold__{target}" for target in TARGET_COLUMNS]
    prediction_columns = [f"prediction__{target}" for target in TARGET_COLUMNS]
    reference = averaged.loc[
        averaged["condition"].astype(str) == reference_condition
    ].sort_values(ID_COLUMN)
    reference_gold = reference[gold_columns].to_numpy(float)
    reference_probability = reference[prediction_columns].to_numpy(float)
    comparisons: dict[str, Any] = {}
    rng = np.random.default_rng(seed)
    bootstrap_samples = [
        rng.integers(0, len(reference_gold), size=len(reference_gold))
        for _ in range(bootstrap_iterations)
    ]
    for condition in candidate_conditions:
        candidate = averaged.loc[
            averaged["condition"].astype(str) == condition
        ].sort_values(ID_COLUMN)
        if candidate[ID_COLUMN].astype(str).tolist() != reference[
            ID_COLUMN
        ].astype(str).tolist():
            raise ValueError(f"Condition {condition!r} repeat means are not UID-aligned")
        candidate_gold = candidate[gold_columns].to_numpy(float)
        if not np.array_equal(candidate_gold, reference_gold):
            raise ValueError(f"Condition {condition!r} repeat means change Gold labels")
        candidate_probability = candidate[prediction_columns].to_numpy(float)
        reference_metrics = summary["conditions"][reference_condition]
        candidate_metrics = summary["conditions"][condition]
        per_target_delta = {
            target: float(
                candidate_metrics["per_target_roc_auc"][target]
                - reference_metrics["per_target_roc_auc"][target]
            )
            for target in TARGET_COLUMNS
        }
        worst_target = min(TARGET_COLUMNS, key=per_target_delta.__getitem__)
        auc_deltas: list[float] = []
        brier_improvements: list[float] = []
        for sample in bootstrap_samples:
            sampled_gold = reference_gold[sample]
            reference_auc: list[float] = []
            candidate_auc: list[float] = []
            for target_index in range(len(TARGET_COLUMNS)):
                labels = sampled_gold[:, target_index]
                if np.unique(labels).size != 2:
                    continue
                reference_auc.append(
                    float(
                        roc_auc_score(
                            labels,
                            reference_probability[sample, target_index],
                        )
                    )
                )
                candidate_auc.append(
                    float(
                        roc_auc_score(
                            labels,
                            candidate_probability[sample, target_index],
                        )
                    )
                )
            if not reference_auc:
                raise RuntimeError("Bootstrap sample contains no evaluable targets")
            auc_deltas.append(float(np.mean(candidate_auc) - np.mean(reference_auc)))
            reference_brier = float(
                np.mean((reference_probability[sample] - sampled_gold) ** 2)
            )
            candidate_brier = float(
                np.mean((candidate_probability[sample] - sampled_gold) ** 2)
            )
            brier_improvements.append(reference_brier - candidate_brier)

        auc_distribution = np.asarray(auc_deltas)
        brier_distribution = np.asarray(brier_improvements)
        comparisons[f"{condition}_minus_{reference_condition}"] = {
            "observed_delta_macro_roc_auc": float(
                candidate_metrics["macro_roc_auc"]
                - reference_metrics["macro_roc_auc"]
            ),
            "observed_delta_macro_average_precision": float(
                candidate_metrics["macro_average_precision"]
                - reference_metrics["macro_average_precision"]
            ),
            "observed_brier_improvement": float(
                reference_metrics["brier"] - candidate_metrics["brier"]
            ),
            "observed_log_loss_improvement": float(
                reference_metrics["log_loss"] - candidate_metrics["log_loss"]
            ),
            "per_target_roc_auc_delta": per_target_delta,
            "worst_target_roc_auc_delta": {
                "target": worst_target,
                "delta": per_target_delta[worst_target],
            },
            "paired_bootstrap_95": [
                float(np.quantile(auc_distribution, 0.025)),
                float(np.quantile(auc_distribution, 0.975)),
            ],
            "bootstrap_probability_positive": float(
                np.mean(auc_distribution > 0.0)
            ),
            "paired_brier_improvement_95": [
                float(np.quantile(brier_distribution, 0.025)),
                float(np.quantile(brier_distribution, 0.975)),
            ],
            "bootstrap_probability_brier_improvement_positive": float(
                np.mean(brier_distribution > 0.0)
            ),
        }

    summary.update(
        {
            "format": POLICY_OOF_FORMAT,
            "reference_condition": reference_condition,
            "comparisons": comparisons,
            "claim_strength": "descriptive/directional only",
        }
    )
    return averaged, summary
