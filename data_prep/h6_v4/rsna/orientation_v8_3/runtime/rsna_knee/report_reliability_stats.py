"""Frozen statistical mechanics for report-reliability proxy analysis.

The helpers in this module implement only the deterministic mechanics declared
by the ``rsna-knee-report-reliability-v3`` analysis contract.  They do not read
project artifacts, inspect Gold labels, choose thresholds from outcomes, or
interpret report/image discordance as a confirmed error.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


OUTER_FOLDS = 5
INNER_FOLDS = 3
FOLD_SEED = 20_260_830
FINAL_ALPHA_FOLD_SEED = 20_260_930
RIDGE_ALPHA_GRID = (0.1, 1.0, 10.0, 100.0)
ALPHA_TIE_TOLERANCE = 1e-12
FROZEN_TAIL_FRACTIONS = (0.05, 0.10, 0.20)
DERIVED_IMAGE_EVIDENCE_COLUMNS = frozenset(
    {
        "image_oof_valid",
        "dino_oof_valid",
        "rad_oof_valid",
        "dino_target_ecdf",
        "rad_probability",
        "rad_target_ecdf",
        "image_consensus_rank",
        "image_family_disagreement",
        "rad_internal_dispersion",
        "image_high_agreement",
        "image_state",
        "omission_like_continuous",
        "contradiction_like_continuous",
    }
)

Direction = Literal["omission", "contradiction"]


@dataclass(frozen=True)
class NestedRidgeComparison:
    """OOF predictions, alpha-search trace, and weighted comparison metrics."""

    oof: pd.DataFrame
    selections: pd.DataFrame
    metrics: dict[str, float | int]


@dataclass(frozen=True)
class DirectionalNestedRidgeResult:
    """Target-by-direction nested OOF results and audited exclusions."""

    oof: pd.DataFrame
    selections: pd.DataFrame
    metrics: pd.DataFrame
    exclusions: pd.DataFrame


@dataclass(frozen=True)
class FinalAlphaSelection:
    """Frozen proxy-only final-CV alpha trace and selected value."""

    trace: pd.DataFrame
    selected_alpha: float


@dataclass(frozen=True)
class FoldFittedTransformerProvider:
    """Adapter from a fit/transform object to the split-provider protocol.

    ``reference_frame`` should be the outcome-blind, all-polarity/all-target
    eligible report corpus.  With frozen fold columns, its fit corpus is the
    complete corresponding global training folds.  Without fold columns (the
    fresh final 3-fold selection), it is restricted to the training report-group
    keys supplied by the caller.  A new transformer is created and fitted on
    every invocation.
    """

    reference_frame: pd.DataFrame
    transformer_factory: Any
    feature_columns: tuple[str, ...]
    group_column: str = "report_group"
    key_column: str = "StudyInstanceUID"
    outer_fold_column: str | None = None
    inner_fold_columns: Mapping[int, str] | None = None

    def __call__(
        self,
        *,
        train_frame: pd.DataFrame,
        valid_frame: pd.DataFrame,
        train_report_groups: tuple[str, ...],
        valid_report_groups: tuple[str, ...],
        outer_fold: int,
        inner_fold: int | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not callable(self.transformer_factory):
            raise ValueError("transformer_factory must be callable")
        columns = tuple(str(column) for column in self.feature_columns)
        if not columns or len(set(columns)) != len(columns):
            raise ValueError("feature_columns must be non-empty and unique")
        _required_columns(
            self.reference_frame, [self.group_column, self.key_column]
        )
        reference_groups = _nonempty_string_keys(
            self.reference_frame[self.group_column],
            name=f"reference {self.group_column}",
        )
        if self.outer_fold_column is None:
            fit_mask = np.isin(reference_groups, np.asarray(train_report_groups))
        else:
            _required_columns(self.reference_frame, [self.outer_fold_column])
            outer_values = _finite_numeric(
                self.reference_frame[self.outer_fold_column],
                name=self.outer_fold_column,
            )
            fit_mask = outer_values != int(outer_fold)
            if inner_fold is not None:
                if self.inner_fold_columns is None or outer_fold not in self.inner_fold_columns:
                    raise ValueError(
                        "frozen-fold provider needs an inner column for every outer fold"
                    )
                inner_column = str(self.inner_fold_columns[outer_fold])
                _required_columns(self.reference_frame, [inner_column])
                inner_values = _finite_numeric(
                    self.reference_frame[inner_column], name=inner_column
                )
                fit_mask &= inner_values != int(inner_fold)
        fit_reference = self.reference_frame.loc[fit_mask].copy()
        if fit_reference.empty:
            raise ValueError("split transformer fit corpus is empty")
        fit_groups = set(
            _nonempty_string_keys(
                fit_reference[self.group_column], name=f"fit {self.group_column}"
            ).tolist()
        )
        if not set(train_report_groups).issubset(fit_groups):
            raise ValueError("split transformer fit corpus omits model training groups")
        if fit_groups.intersection(valid_report_groups):
            raise ValueError("split transformer fit corpus contains held-out groups")

        transformer = self.transformer_factory()
        fit_transform = getattr(transformer, "fit_transform", None)
        transform = getattr(transformer, "transform", None)
        if fit_transform is None or transform is None:
            raise ValueError(
                "transformer must expose fit_transform and transform methods"
            )
        fitted_generated = fit_transform(fit_reference.copy())
        aligned_fit = _align_generated_split_features(
            fit_reference,
            fitted_generated,
            feature_columns=columns,
            key_column=self.key_column,
            label="global training reference",
        )
        fit_feature_reference = fit_reference[[self.key_column]].copy()
        for column in columns:
            fit_feature_reference[column] = aligned_fit[column].to_numpy()
        training_augmented = _merge_feature_reference_subset(
            train_frame,
            fit_feature_reference,
            feature_columns=columns,
            key_column=self.key_column,
        )
        heldout_generated = transform(valid_frame.copy())
        heldout_aligned = _align_generated_split_features(
            valid_frame,
            heldout_generated,
            feature_columns=columns,
            key_column=self.key_column,
            label="held-out",
        )
        return (
            training_augmented.loc[:, list(columns)].copy(),
            heldout_aligned.loc[:, list(columns)].copy(),
        )


@dataclass(frozen=True)
class TargetwiseAverageTieECDFReference:
    """Frozen target-wise sorted reference values for held-out ECDF mapping."""

    target_column: str
    value_columns: tuple[str, ...]
    sorted_values: dict[tuple[str, str], np.ndarray]

    def transform(
        self,
        frame: pd.DataFrame,
        *,
        allow_nonfinite: bool = False,
    ) -> pd.DataFrame:
        """Apply the frozen reference without using evaluation-row outcomes.

        For reference size ``n``, and insertion bounds ``left`` and ``right``,
        the frozen percentile is ``(left + right) / (2 * n)``.  This is the
        contract's average-tie mid-ECDF: values below/above the reference map to
        exactly zero/one, and evaluation rows never alter the reference.
        """

        _required_columns(frame, [self.target_column, *self.value_columns])
        targets = _nonempty_string_keys(
            frame[self.target_column].to_numpy(), name=self.target_column
        )
        output: dict[str, np.ndarray] = {}
        known_targets = {target for target, _ in self.sorted_values}
        unknown = sorted(set(targets.tolist()).difference(known_targets))
        if unknown:
            raise ValueError(f"ECDF evaluation contains unknown targets: {unknown}")
        for column in self.value_columns:
            numeric = pd.to_numeric(frame[column], errors="coerce").to_numpy(
                dtype=np.float64
            )
            invalid = ~np.isfinite(numeric)
            if invalid.any() and not allow_nonfinite:
                raise ValueError(f"{column} must contain only finite values")
            transformed = np.full(len(frame), np.nan, dtype=np.float64)
            for target in sorted(set(targets.tolist())):
                positions = np.flatnonzero((targets == target) & ~invalid)
                if len(positions) == 0:
                    continue
                reference = self.sorted_values[(target, column)]
                left = np.searchsorted(reference, numeric[positions], side="left")
                right = np.searchsorted(reference, numeric[positions], side="right")
                rank = (left + right) / (2.0 * float(len(reference)))
                transformed[positions] = np.clip(rank, 0.0, 1.0)
            output[f"{column}__target_ecdf"] = transformed
        return pd.DataFrame(output, index=frame.index)


@dataclass(frozen=True)
class FamilyBalancedImageReference:
    """Frozen DINO/Rad reference used for proxy-only and external scoring."""

    target_column: str
    dino_column: str
    rad_component_columns: tuple[str, str, str]
    rad_probability_column: str
    ecdf_reference: TargetwiseAverageTieECDFReference
    disagreement_gate: float

    def transform(
        self,
        frame: pd.DataFrame,
        *,
        allow_oof_invalid: bool = False,
    ) -> pd.DataFrame:
        """Build family-balanced evidence, marking invalid external OOF rows."""

        required = [
            self.target_column,
            self.dino_column,
            *self.rad_component_columns,
        ]
        _required_columns(frame, required)
        working = frame[required].copy()
        numeric_columns = [self.dino_column, *self.rad_component_columns]
        numeric = np.column_stack(
            [
                pd.to_numeric(working[column], errors="coerce").to_numpy(
                    dtype=np.float64
                )
                for column in numeric_columns
            ]
        )
        dino_valid = np.isfinite(numeric[:, 0])
        rad_valid = np.isfinite(numeric[:, 1:]).all(axis=1)
        valid = dino_valid & rad_valid
        finite_values = numeric[np.isfinite(numeric)]
        if np.any((finite_values < 0.0) | (finite_values > 1.0)):
            raise ValueError("finite image evidence must lie within [0, 1]")
        if (~valid).any() and not allow_oof_invalid:
            raise ValueError("image evidence contains non-finite OOF values")
        rad_raw = numeric[:, 1:]
        rad_probability = np.full(len(frame), np.nan, dtype=np.float64)
        rad_probability[rad_valid] = np.median(rad_raw[rad_valid], axis=1)
        working[self.rad_probability_column] = rad_probability
        ranks = self.ecdf_reference.transform(
            working,
            allow_nonfinite=allow_oof_invalid,
        )
        dino_rank = ranks[f"{self.dino_column}__target_ecdf"].to_numpy(float)
        rad_rank = ranks[f"{self.rad_probability_column}__target_ecdf"].to_numpy(
            float
        )
        rad_component_ranks = np.column_stack(
            [
                ranks[f"{column}__target_ecdf"].to_numpy(float)
                for column in self.rad_component_columns
            ]
        )
        consensus = 0.5 * dino_rank + 0.5 * rad_rank
        disagreement = np.abs(dino_rank - rad_rank)
        dispersion = np.full(len(frame), np.nan, dtype=np.float64)
        dispersion[rad_valid] = np.std(rad_component_ranks[rad_valid], axis=1, ddof=0)
        high_agreement = valid & (disagreement <= self.disagreement_gate + 1e-12)

        state = np.full(len(frame), "OOF_INVALID", dtype=object)
        if valid.any():
            valid_states = strict_image_states(
                dino_rank[valid],
                rad_rank[valid],
                tail_fraction=0.10,
                disagreement_gate=self.disagreement_gate,
            )
            state[valid] = valid_states.to_numpy(dtype=object)
        return pd.DataFrame(
            {
                "image_oof_valid": valid,
                "dino_oof_valid": dino_valid,
                "rad_oof_valid": rad_valid,
                "dino_target_ecdf": dino_rank,
                "rad_probability": rad_probability,
                "rad_target_ecdf": rad_rank,
                "image_consensus_rank": consensus,
                "image_family_disagreement": disagreement,
                "rad_internal_dispersion": dispersion,
                "image_high_agreement": high_agreement,
                "image_state": state,
            },
            index=frame.index,
        )


@dataclass(frozen=True)
class FinalDirectionalRidgeModel:
    """All-proxy final model plus proxy-only external percentile reference."""

    target: str
    direction: Direction
    target_column: str
    teacher_label_column: str
    feature_columns: tuple[str, ...]
    estimator: Pipeline
    image_reference: FamilyBalancedImageReference
    proxy_risk_reference: np.ndarray
    fitted_feature_transformer: Any | None = None
    split_feature_columns: tuple[str, ...] = ()
    split_feature_key_column: str = "StudyInstanceUID"

    def score_external(
        self,
        frame: pd.DataFrame,
        *,
        allow_oof_invalid: bool = True,
    ) -> pd.DataFrame:
        """Score external rows without fitting anything on them.

        Morphology-only risk remains available for audit when image OOF values
        are invalid, but ``guardrailed_review_risk_percentile`` is null unless
        the image-family agreement gate passes.  Consequently disagreement can
        never boost the externally exposed review-risk rank.
        """

        working = _apply_fitted_feature_transformer(
            frame,
            transformer=self.fitted_feature_transformer,
            feature_columns=self.split_feature_columns,
            key_column=self.split_feature_key_column,
            training_reference=False,
        )
        _required_columns(
            working,
            [self.target_column, self.teacher_label_column, *self.feature_columns],
        )
        targets = _nonempty_string_keys(
            working[self.target_column].to_numpy(), name=self.target_column
        )
        if set(targets.tolist()) != {self.target}:
            raise ValueError(
                f"external frame must contain only fitted target {self.target}"
            )
        required_label = 0.0 if self.direction == "omission" else 1.0
        labels = _finite_numeric(
            working[self.teacher_label_column], name=self.teacher_label_column
        )
        if not np.all(labels == required_label):
            raise ValueError("external frame contains the wrong teacher polarity")
        _validate_feature_columns(
            working, self.feature_columns, label="external feature columns"
        )
        raw = np.asarray(
            self.estimator.predict(working[list(self.feature_columns)]),
            dtype=np.float64,
        )
        if not np.isfinite(raw).all():
            raise ValueError("final Ridge produced non-finite external scores")
        risk = np.clip(raw, 0.0, 1.0)
        percentile = _apply_sorted_average_tie_reference(
            self.proxy_risk_reference, risk
        )
        evidence = self.image_reference.transform(
            working, allow_oof_invalid=allow_oof_invalid
        )
        guardrail = (
            evidence["image_oof_valid"].to_numpy(bool)
            & evidence["image_high_agreement"].to_numpy(bool)
        )
        guarded = np.where(guardrail, percentile, np.nan)
        result = evidence.copy()
        result.insert(0, "direction", self.direction)
        result.insert(0, "target", self.target)
        result["report_risk_score"] = risk
        result["report_risk_percentile_proxy_reference"] = percentile
        result["image_agreement_guardrail_pass"] = guardrail
        result["guardrailed_review_risk_percentile"] = guarded
        return result


def _required_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"input is missing columns: {missing}")


def _finite_numeric(values: Any, *, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    try:
        result = raw.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _nonempty_string_keys(values: Any, *, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if pd.isna(raw).any():
        raise ValueError(f"{name} contains null values")
    result = np.asarray([str(value) for value in raw], dtype=object)
    if np.any(result == ""):
        raise ValueError(f"{name} contains empty values")
    # String normalization must not silently collapse distinct source keys.
    mapping = pd.DataFrame({"raw": raw, "text": result}).groupby(
        "text", sort=False, dropna=False
    )["raw"].nunique(dropna=False)
    if mapping.gt(1).any():
        raise ValueError(f"{name} contains distinct keys with the same string form")
    return result


def targetwise_average_tie_ecdf(
    frame: pd.DataFrame,
    *,
    value_column: str,
    target_column: str = "target",
    output_name: str | None = None,
) -> pd.Series:
    """Return target-wise empirical-CDF ranks with average handling of ties.

    The caller supplies the already frozen eligible non-Gold cohort.  For a
    target containing ``n`` rows, the result is ``(left + right) / (2*n)``.
    Thus ties receive their mid-ECDF value and the result is order invariant.
    """

    _required_columns(frame, [target_column, value_column])
    if frame.empty:
        raise ValueError("ECDF input is empty")
    targets = _nonempty_string_keys(frame[target_column].to_numpy(), name=target_column)
    values = _finite_numeric(frame[value_column].to_numpy(), name=value_column)
    working = pd.DataFrame(
        {"_target": targets, "_value": values}, index=frame.index
    )
    grouped = working.groupby("_target", sort=False, observed=True)["_value"]
    ranks = (grouped.rank(method="average") - 0.5) / grouped.transform("size")
    result = pd.Series(
        ranks.to_numpy(dtype=np.float64),
        index=frame.index,
        name=output_name or f"{value_column}__target_ecdf",
        dtype=np.float64,
    )
    if not np.isfinite(result.to_numpy()).all() or not (
        result.ge(0.0).all() and result.le(1.0).all()
    ):
        raise AssertionError("target-wise ECDF produced invalid ranks")
    return result


def _apply_sorted_average_tie_reference(
    sorted_reference: np.ndarray, values: Any
) -> np.ndarray:
    reference = _finite_numeric(sorted_reference, name="sorted_reference")
    if len(reference) == 0:
        raise ValueError("sorted_reference is empty")
    if np.any(reference[:-1] > reference[1:]):
        raise ValueError("sorted_reference is not sorted")
    numeric = _finite_numeric(values, name="evaluation values")
    left = np.searchsorted(reference, numeric, side="left")
    right = np.searchsorted(reference, numeric, side="right")
    rank = (left + right) / (2.0 * float(len(reference)))
    return np.clip(rank, 0.0, 1.0)


def fit_targetwise_average_tie_ecdf_reference(
    frame: pd.DataFrame,
    *,
    value_columns: Sequence[str],
    target_column: str = "target",
) -> TargetwiseAverageTieECDFReference:
    """Fit a reusable target-wise ECDF reference on training rows only."""

    columns = tuple(str(column) for column in value_columns)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("value_columns must contain unique column names")
    _required_columns(frame, [target_column, *columns])
    if frame.empty:
        raise ValueError("ECDF reference input is empty")
    targets = _nonempty_string_keys(frame[target_column].to_numpy(), name=target_column)
    sorted_values: dict[tuple[str, str], np.ndarray] = {}
    for column in columns:
        values = _finite_numeric(frame[column].to_numpy(), name=column)
        for target in sorted(set(targets.tolist())):
            reference = np.sort(values[targets == target], kind="mergesort")
            if len(reference) == 0:
                raise AssertionError(f"empty ECDF reference for target {target}")
            reference.setflags(write=False)
            sorted_values[(target, column)] = reference
    return TargetwiseAverageTieECDFReference(
        target_column=target_column,
        value_columns=columns,
        sorted_values=sorted_values,
    )


def crossfit_targetwise_average_tie_ecdf(
    frame: pd.DataFrame,
    *,
    value_columns: Sequence[str],
    fold_column: str,
    target_column: str = "target",
) -> pd.DataFrame:
    """Fit target-wise ECDFs on fold-training rows and apply held-out only."""

    _required_columns(frame, [target_column, fold_column, *value_columns])
    folds_raw = _finite_numeric(frame[fold_column].to_numpy(), name=fold_column)
    folds = folds_raw.astype(np.int64)
    if not np.array_equal(folds_raw, folds.astype(np.float64)):
        raise ValueError(f"{fold_column} must contain integer fold ids")
    unique_folds = sorted(set(folds.tolist()))
    if len(unique_folds) < 2:
        raise ValueError("cross-fitted ECDF requires at least two folds")
    result = pd.DataFrame(
        np.nan,
        index=frame.index,
        columns=[f"{column}__target_ecdf" for column in value_columns],
        dtype=np.float64,
    )
    for fold in unique_folds:
        train = folds != fold
        valid = folds == fold
        reference = fit_targetwise_average_tie_ecdf_reference(
            frame.loc[train],
            value_columns=value_columns,
            target_column=target_column,
        )
        transformed = reference.transform(frame.loc[valid])
        result.loc[valid, transformed.columns] = transformed.to_numpy()
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError(
            "cross-fitted ECDF is incomplete; every held-out target needs training support"
        )
    return result


def fit_family_balanced_image_reference(
    frame: pd.DataFrame,
    *,
    dino_column: str,
    rad_component_columns: Sequence[str],
    target_column: str = "target",
    disagreement_gate: float = 0.10,
) -> FamilyBalancedImageReference:
    """Fit the frozen equal-family DINO/median-Rad image-score reference."""

    rad_columns = tuple(str(column) for column in rad_component_columns)
    if len(rad_columns) != 3 or len(set(rad_columns)) != 3:
        raise ValueError("rad_component_columns must contain exactly three columns")
    if str(dino_column) in set(rad_columns):
        raise ValueError("DINO and Rad component columns must be distinct")
    gate = _frozen_tail_fraction(disagreement_gate)
    required = [target_column, dino_column, *rad_columns]
    _required_columns(frame, required)
    if frame.empty:
        raise ValueError("family-balanced image reference input is empty")
    working = frame[required].copy()
    for column in (dino_column, *rad_columns):
        _validate_unit_interval(working[column], name=column)
    rad_numeric = np.column_stack(
        [_finite_numeric(working[column], name=column) for column in rad_columns]
    )
    rad_probability_column = "__rad_probability_median"
    if rad_probability_column in frame.columns:
        raise ValueError(f"reserved column already exists: {rad_probability_column}")
    working[rad_probability_column] = np.median(rad_numeric, axis=1)
    reference = fit_targetwise_average_tie_ecdf_reference(
        working,
        value_columns=[dino_column, *rad_columns, rad_probability_column],
        target_column=target_column,
    )
    return FamilyBalancedImageReference(
        target_column=target_column,
        dino_column=dino_column,
        rad_component_columns=(rad_columns[0], rad_columns[1], rad_columns[2]),
        rad_probability_column=rad_probability_column,
        ecdf_reference=reference,
        disagreement_gate=gate,
    )


def crossfit_family_balanced_image_evidence(
    frame: pd.DataFrame,
    *,
    dino_column: str,
    rad_component_columns: Sequence[str],
    fold_column: str,
    target_column: str = "target",
    disagreement_gate: float = 0.10,
) -> pd.DataFrame:
    """Create family-balanced evidence with fold-local training references."""

    _required_columns(
        frame,
        [target_column, fold_column, dino_column, *rad_component_columns],
    )
    folds_raw = _finite_numeric(frame[fold_column].to_numpy(), name=fold_column)
    folds = folds_raw.astype(np.int64)
    if not np.array_equal(folds_raw, folds.astype(np.float64)):
        raise ValueError(f"{fold_column} must contain integer fold ids")
    unique_folds = sorted(set(folds.tolist()))
    if len(unique_folds) < 2:
        raise ValueError("cross-fitted image evidence requires at least two folds")
    parts: list[pd.DataFrame] = []
    for fold in unique_folds:
        train = folds != fold
        valid = folds == fold
        reference = fit_family_balanced_image_reference(
            frame.loc[train],
            dino_column=dino_column,
            rad_component_columns=rad_component_columns,
            target_column=target_column,
            disagreement_gate=disagreement_gate,
        )
        part = reference.transform(frame.loc[valid])
        part["_source_position"] = np.flatnonzero(valid)
        parts.append(part)
    result = pd.concat(parts).sort_values("_source_position", kind="mergesort")
    result = result.drop(columns="_source_position")
    if not result.index.equals(frame.index):
        # Preserve arbitrary unique source indices while retaining source order.
        result = result.reindex(frame.index)
    if not result["image_oof_valid"].all():
        raise AssertionError("proxy cross-fit unexpectedly produced OOF-invalid rows")
    return result


def _validate_unit_interval(values: Any, *, name: str) -> np.ndarray:
    result = _finite_numeric(values, name=name)
    if np.any((result < 0.0) | (result > 1.0)):
        raise ValueError(f"{name} must lie within [0, 1]")
    return result


def _frozen_tail_fraction(value: float) -> float:
    tail = float(value)
    if not any(abs(tail - frozen) <= 1e-12 for frozen in FROZEN_TAIL_FRACTIONS):
        raise ValueError(
            "tail_fraction must be one of the frozen values 0.05, 0.10, or 0.20"
        )
    return tail


def strict_image_states(
    dino_target_ecdf: Any,
    rad_target_ecdf: Any,
    *,
    tail_fraction: float = 0.10,
    disagreement_gate: float = 0.10,
) -> pd.Series:
    """Classify the two correlated image views at frozen tail quantiles.

    ``BOTH_HIGH`` and ``BOTH_LOW`` require an extreme equal-family consensus
    together with the frozen DINO/Rad agreement gate.  A single extreme family
    is identified as a sensitivity state; opposite extremes and non-extreme
    pairs remain ``MID_AMBIGUOUS``.  Invalid OOF values fail closed rather than
    being converted to the ``OOF_INVALID`` state.
    """

    tail = _frozen_tail_fraction(tail_fraction)
    gate = _frozen_tail_fraction(disagreement_gate)
    dino = _validate_unit_interval(dino_target_ecdf, name="dino_target_ecdf")
    rad = _validate_unit_interval(rad_target_ecdf, name="rad_target_ecdf")
    if len(dino) != len(rad):
        raise ValueError("DINO and Rad ECDF arrays must have the same length")

    high = 1.0 - tail
    dino_high = dino >= high
    rad_high = rad >= high
    dino_low = dino <= tail
    rad_low = rad <= tail
    dino_extreme = dino_high | dino_low
    rad_extreme = rad_high | rad_low

    state = np.full(len(dino), "MID_AMBIGUOUS", dtype=object)
    state[dino_extreme & ~rad_extreme] = "DINO_ONLY_EXTREME"
    state[rad_extreme & ~dino_extreme] = "RAD_ONLY_EXTREME"
    consensus = 0.5 * dino + 0.5 * rad
    high_agreement = np.abs(dino - rad) <= gate + 1e-12
    state[(consensus <= tail + 1e-12) & high_agreement] = "BOTH_LOW"
    state[(consensus >= high - 1e-12) & high_agreement] = "BOTH_HIGH"
    return pd.Series(state, dtype="string", name="image_state")


def directional_teacher_image_endpoints(
    teacher_score: Any,
    dino_target_ecdf: Any,
    rad_target_ecdf: Any,
    *,
    tail_fraction: float = 0.10,
    disagreement_gate: float = 0.10,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """Create omission/contradiction proxy endpoints by teacher polarity.

    Teacher score ``0.5`` is an abstention.  Continuous omission-like values
    exist only for teacher-negative rows and continuous contradiction-like
    values only for teacher-positive rows.  Boolean strict endpoints are false
    outside their declared teacher cohort.
    """

    teacher = _validate_unit_interval(teacher_score, name="teacher_score")
    dino = _validate_unit_interval(dino_target_ecdf, name="dino_target_ecdf")
    rad = _validate_unit_interval(rad_target_ecdf, name="rad_target_ecdf")
    if not (len(teacher) == len(dino) == len(rad)):
        raise ValueError("teacher and image arrays must have the same length")
    if index is None:
        index = pd.RangeIndex(len(teacher))
    if len(index) != len(teacher):
        raise ValueError("index length differs from endpoint arrays")

    gate = _frozen_tail_fraction(disagreement_gate)
    state = strict_image_states(
        dino,
        rad,
        tail_fraction=tail_fraction,
        disagreement_gate=gate,
    )
    state.index = index
    negative = teacher < 0.5
    positive = teacher > 0.5
    decisive = negative | positive
    hard_label = np.full(len(teacher), np.nan, dtype=np.float64)
    hard_label[negative] = 0.0
    hard_label[positive] = 1.0

    omission = np.full(len(teacher), np.nan, dtype=np.float64)
    omission[negative] = (dino[negative] + rad[negative]) / 2.0
    contradiction = np.full(len(teacher), np.nan, dtype=np.float64)
    contradiction[positive] = (
        (1.0 - dino[positive]) + (1.0 - rad[positive])
    ) / 2.0

    both_high = state.eq("BOTH_HIGH").to_numpy()
    both_low = state.eq("BOTH_LOW").to_numpy()
    return pd.DataFrame(
        {
            "teacher_state": np.where(
                negative, "NEGATIVE", np.where(positive, "POSITIVE", "ABSTAIN")
            ),
            "teacher_hard_label": hard_label,
            "teacher_decisive": decisive,
            "dino_target_ecdf": dino,
            "rad_target_ecdf": rad,
            "image_state": state.to_numpy(dtype=object),
            "image_consensus_rank": (dino + rad) / 2.0,
            "image_family_disagreement": np.abs(dino - rad),
            "image_high_agreement": np.abs(dino - rad) <= gate + 1e-12,
            "omission_like_continuous": omission,
            "contradiction_like_continuous": contradiction,
            "strict_omission": negative & both_high,
            "strict_contradiction": positive & both_low,
            "agreement_negative": negative & both_low,
            "agreement_positive": positive & both_high,
        },
        index=index,
    )


def targetwise_directional_endpoints(
    frame: pd.DataFrame,
    *,
    teacher_score_column: str,
    dino_score_column: str,
    rad_score_column: str,
    target_column: str = "target",
    tail_fraction: float = 0.10,
    disagreement_gate: float = 0.10,
) -> pd.DataFrame:
    """Rank both image families by target and construct directional endpoints."""

    _required_columns(
        frame,
        [target_column, teacher_score_column, dino_score_column, rad_score_column],
    )
    dino = targetwise_average_tie_ecdf(
        frame,
        value_column=dino_score_column,
        target_column=target_column,
        output_name="dino_target_ecdf",
    )
    rad = targetwise_average_tie_ecdf(
        frame,
        value_column=rad_score_column,
        target_column=target_column,
        output_name="rad_target_ecdf",
    )
    return directional_teacher_image_endpoints(
        frame[teacher_score_column].to_numpy(),
        dino.to_numpy(),
        rad.to_numpy(),
        tail_fraction=tail_fraction,
        disagreement_gate=disagreement_gate,
        index=frame.index,
    )


def inverse_report_group_weights(report_groups: Any) -> np.ndarray:
    """Return inverse eligible-row counts, giving every report group weight one."""

    groups = _nonempty_string_keys(report_groups, name="report_groups")
    if len(groups) == 0:
        raise ValueError("report_groups is empty")
    counts = pd.Series(groups).groupby(groups, sort=False).transform("size")
    weights = 1.0 / counts.to_numpy(dtype=np.float64)
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise AssertionError("invalid inverse report-group weights")
    check = pd.DataFrame({"group": groups, "weight": weights}).groupby(
        "group", sort=False
    )["weight"].sum()
    if not np.allclose(check.to_numpy(), 1.0, rtol=0.0, atol=1e-12):
        raise AssertionError("report-group weights do not sum to one")
    return weights


def _shuffled_group_fold_ids(
    report_groups: Any,
    *,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    """Version-stable equivalent of shuffled ``GroupKFold`` group assignment."""

    groups = _nonempty_string_keys(report_groups, name="report_groups")
    if not isinstance(n_splits, (int, np.integer)) or int(n_splits) < 2:
        raise ValueError("n_splits must be an integer of at least two")
    if not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    unique_groups = np.unique(groups)
    if len(unique_groups) < int(n_splits):
        raise ValueError(
            f"need at least {int(n_splits)} report groups, found {len(unique_groups)}"
        )
    shuffled = np.random.RandomState(int(seed)).permutation(unique_groups)
    mapping: dict[str, int] = {}
    for fold, fold_groups in enumerate(np.array_split(shuffled, int(n_splits))):
        mapping.update({str(group): fold for group in fold_groups})
    result = np.asarray([mapping[str(group)] for group in groups], dtype=np.int16)
    if set(result.tolist()) != set(range(int(n_splits))):
        raise AssertionError("shuffled GroupKFold assignment is incomplete")
    return result


def _validated_outer_folds(
    frame: pd.DataFrame,
    *,
    group_column: str,
    outer_fold_column: str | None,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    groups = _nonempty_string_keys(frame[group_column].to_numpy(), name=group_column)
    if outer_fold_column is None:
        return _shuffled_group_fold_ids(groups, n_splits=n_splits, seed=seed)
    _required_columns(frame, [outer_fold_column])
    numeric = _finite_numeric(frame[outer_fold_column].to_numpy(), name=outer_fold_column)
    folds = numeric.astype(np.int16)
    if not np.array_equal(numeric, folds.astype(np.float64)):
        raise ValueError(f"{outer_fold_column} must contain integer fold ids")
    if set(folds.tolist()) != set(range(int(n_splits))):
        raise ValueError(
            f"{outer_fold_column} must contain every fold 0 through {int(n_splits) - 1}"
        )
    check = pd.DataFrame({"group": groups, "fold": folds}).groupby(
        "group", sort=False
    )["fold"].nunique()
    if not check.eq(1).all():
        example = str(check.loc[check.gt(1)].index[0])
        raise ValueError(f"report group {example} spans multiple outer folds")
    return folds


def _materialize_contract_nested_fold_columns(
    frame: pd.DataFrame,
    *,
    group_column: str,
    outer_fold_column: str | None,
    inner_fold_columns: Mapping[int, str] | None,
    n_outer_folds: int,
    n_inner_folds: int,
    seed: int,
) -> tuple[pd.DataFrame, str, dict[int, str]]:
    """Return one global group-fold map shared by every target/direction.

    Actual contract execution supplies the frozen fold-map columns.  The
    deterministic materialization branch exists for synthetic tests and new
    frozen-map construction: importantly it is run once on the full eligible
    frame, never independently on a teacher-polarity subset.
    """

    working = frame.copy()
    if outer_fold_column is None:
        resolved_outer = "__contract_outer_fold"
        if resolved_outer in working.columns:
            raise ValueError(f"reserved column already exists: {resolved_outer}")
        working[resolved_outer] = _shuffled_group_fold_ids(
            working[group_column], n_splits=n_outer_folds, seed=seed
        )
    else:
        resolved_outer = str(outer_fold_column)
    outer = _validated_outer_folds(
        working,
        group_column=group_column,
        outer_fold_column=resolved_outer,
        n_splits=n_outer_folds,
        seed=seed,
    )

    expected_outer = set(range(int(n_outer_folds)))
    if inner_fold_columns is None:
        resolved_inner: dict[int, str] = {}
        for outer_fold in range(int(n_outer_folds)):
            column = f"__contract_inner_fold_when_outer_{outer_fold}"
            if column in working.columns:
                raise ValueError(f"reserved column already exists: {column}")
            values = np.full(len(working), -1, dtype=np.int16)
            outer_train = outer != outer_fold
            values[outer_train] = _shuffled_group_fold_ids(
                working.loc[outer_train, group_column],
                n_splits=n_inner_folds,
                seed=int(seed) + outer_fold,
            )
            working[column] = values
            resolved_inner[outer_fold] = column
    else:
        if set(inner_fold_columns) != expected_outer:
            raise ValueError(
                "inner_fold_columns must map every outer fold exactly once"
            )
        resolved_inner = {
            int(outer_fold): str(column)
            for outer_fold, column in inner_fold_columns.items()
        }
        if len(set(resolved_inner.values())) != len(resolved_inner):
            raise ValueError("inner_fold_columns must contain distinct columns")
        _required_columns(working, list(resolved_inner.values()))

    for outer_fold, inner_column in resolved_inner.items():
        outer_train = outer != outer_fold
        _validated_outer_folds(
            working.loc[outer_train],
            group_column=group_column,
            outer_fold_column=inner_column,
            n_splits=n_inner_folds,
            seed=int(seed) + outer_fold,
        )
    return working, resolved_outer, resolved_inner


def _validate_feature_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    label: str,
) -> tuple[str, ...]:
    normalized = tuple(str(column) for column in columns)
    if not normalized:
        raise ValueError(f"{label} must contain at least one feature")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} contains duplicate columns")
    _required_columns(frame, normalized)
    for column in normalized:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
            if np.isinf(values).any():
                raise ValueError(f"numeric feature {column} contains infinite values")
    return normalized


def _normalized_split_feature_columns(
    split_feature_columns: Sequence[str],
    *,
    model_columns: Sequence[str],
    provider: Any | None,
) -> tuple[str, ...]:
    columns = tuple(str(column) for column in split_feature_columns)
    if len(set(columns)) != len(columns):
        raise ValueError("split_feature_columns contains duplicate columns")
    if provider is None and columns:
        raise ValueError(
            "split_feature_columns requires a split_feature_provider"
        )
    if provider is not None and not columns:
        raise ValueError(
            "split_feature_provider requires declared split_feature_columns"
        )
    unknown = sorted(set(columns).difference(model_columns))
    if unknown:
        raise ValueError(
            f"split feature columns are not declared model predictors: {unknown}"
        )
    return columns


def _align_generated_split_features(
    base: pd.DataFrame,
    generated: Any,
    *,
    feature_columns: Sequence[str],
    key_column: str,
    label: str,
) -> pd.DataFrame:
    if not isinstance(generated, pd.DataFrame):
        raise ValueError(f"{label} split features must be a pandas DataFrame")
    _required_columns(generated, feature_columns)
    if not generated.index.is_unique:
        raise ValueError(f"{label} split-feature index must be unique")
    if generated.index.equals(base.index):
        aligned = generated.loc[:, list(feature_columns)].copy()
    elif len(generated) == len(base) and set(generated.index) == set(base.index):
        aligned = generated.reindex(base.index).loc[:, list(feature_columns)].copy()
    else:
        _required_columns(base, [key_column])
        base_keys = _nonempty_string_keys(base[key_column], name=key_column)
        if key_column in generated.columns:
            generated_keys = _nonempty_string_keys(
                generated[key_column], name=f"generated {key_column}"
            )
        elif generated.index.name == key_column:
            generated_keys = _nonempty_string_keys(
                generated.index.to_numpy(), name=f"generated {key_column} index"
            )
        else:
            raise ValueError(
                f"{label} split features cannot be aligned by index or {key_column}"
            )
        if len(set(generated_keys.tolist())) != len(generated_keys):
            raise ValueError(f"{label} generated keys must be unique")
        if len(set(base_keys.tolist())) != len(base_keys):
            raise ValueError(f"{label} base keys must be unique")
        if set(generated_keys.tolist()) != set(base_keys.tolist()):
            raise ValueError(f"{label} generated keys differ from requested keys")
        lookup = pd.DataFrame(
            generated.loc[:, list(feature_columns)].to_numpy(),
            index=generated_keys,
            columns=list(feature_columns),
        )
        aligned = lookup.loc[base_keys].copy()
        aligned.index = base.index
    _validate_feature_columns(aligned, feature_columns, label=label)
    return aligned


def _materialize_split_feature_frames(
    frame: pd.DataFrame,
    *,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    provider: Any | None,
    feature_columns: Sequence[str],
    key_column: str,
    group_column: str,
    outer_fold: int,
    inner_fold: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_frame = frame.iloc[train_indices].copy()
    valid_frame = frame.iloc[valid_indices].copy()
    if provider is None:
        return train_frame, valid_frame
    train_groups = tuple(
        sorted(
            set(
                _nonempty_string_keys(
                    train_frame[group_column], name=f"train {group_column}"
                ).tolist()
            )
        )
    )
    valid_groups = tuple(
        sorted(
            set(
                _nonempty_string_keys(
                    valid_frame[group_column], name=f"valid {group_column}"
                ).tolist()
            )
        )
    )
    if set(train_groups).intersection(valid_groups):
        raise AssertionError("split feature train/validation groups overlap")
    try:
        generated = provider(
            train_frame=train_frame.copy(),
            valid_frame=valid_frame.copy(),
            train_report_groups=train_groups,
            valid_report_groups=valid_groups,
            outer_fold=int(outer_fold),
            inner_fold=None if inner_fold is None else int(inner_fold),
        )
    except TypeError as error:
        raise ValueError(
            "split_feature_provider must accept the declared keyword protocol"
        ) from error
    if not isinstance(generated, tuple) or len(generated) != 2:
        raise ValueError(
            "split_feature_provider must return (train_features, valid_features)"
        )
    train_generated = _align_generated_split_features(
        train_frame,
        generated[0],
        feature_columns=feature_columns,
        key_column=key_column,
        label="training",
    )
    valid_generated = _align_generated_split_features(
        valid_frame,
        generated[1],
        feature_columns=feature_columns,
        key_column=key_column,
        label="validation",
    )
    for column in feature_columns:
        train_frame[column] = train_generated[column].to_numpy()
        valid_frame[column] = valid_generated[column].to_numpy()
    return train_frame, valid_frame


def _apply_fitted_feature_transformer(
    frame: pd.DataFrame,
    *,
    transformer: Any | None,
    feature_columns: Sequence[str],
    key_column: str,
    training_reference: bool,
) -> pd.DataFrame:
    """Apply an already-fitted final transformer without any external refit."""

    if transformer is None:
        return frame
    columns = tuple(str(column) for column in feature_columns)
    if not columns:
        raise ValueError("fitted transformer requires split_feature_columns")
    if training_reference:
        method = getattr(transformer, "transform_training", None)
        if method is None:
            raise ValueError(
                "fitted transformer must expose transform_training for fit rows"
            )
    else:
        method = getattr(transformer, "transform", None)
        if method is None:
            raise ValueError("fitted transformer must expose transform")
    generated = method(frame.copy())
    aligned = _align_generated_split_features(
        frame,
        generated,
        feature_columns=columns,
        key_column=key_column,
        label="final training" if training_reference else "external",
    )
    result = frame.copy()
    for column in columns:
        result[column] = aligned[column].to_numpy()
    return result


def _merge_feature_reference_subset(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    key_column: str,
) -> pd.DataFrame:
    """Attach a feature-reference subset without changing source row order."""

    _required_columns(reference, feature_columns)
    if not frame.index.is_unique or not reference.index.is_unique:
        raise ValueError("feature-reference indices must be unique")
    use_index = frame.index.isin(reference.index).all()
    if use_index and key_column in frame.columns and key_column in reference.columns:
        candidate_keys = _nonempty_string_keys(
            reference.loc[frame.index, key_column], name=f"reference {key_column}"
        )
        requested_keys = _nonempty_string_keys(frame[key_column], name=key_column)
        use_index = np.array_equal(candidate_keys, requested_keys)
    if use_index:
        aligned = reference.loc[frame.index, list(feature_columns)].copy()
        aligned.index = frame.index
    else:
        _required_columns(frame, [key_column])
        _required_columns(reference, [key_column])
        frame_keys = _nonempty_string_keys(frame[key_column], name=key_column)
        reference_keys = _nonempty_string_keys(
            reference[key_column], name=f"reference {key_column}"
        )
        if len(set(frame_keys.tolist())) != len(frame_keys):
            raise ValueError("final-fit feature keys must be unique")
        if len(set(reference_keys.tolist())) != len(reference_keys):
            raise ValueError("feature-reference keys must be unique")
        missing = sorted(set(frame_keys.tolist()).difference(reference_keys.tolist()))
        if missing:
            raise ValueError(f"feature reference is missing requested keys: {missing[:3]}")
        lookup = pd.DataFrame(
            reference.loc[:, list(feature_columns)].to_numpy(),
            index=reference_keys,
            columns=list(feature_columns),
        )
        aligned = lookup.loc[frame_keys].copy()
        aligned.index = frame.index
    result = frame.copy()
    for column in feature_columns:
        result[column] = aligned[column].to_numpy()
    return result


def _preprocessor(frame: pd.DataFrame, columns: Sequence[str]) -> ColumnTransformer:
    numeric = [
        column
        for column in columns
        if pd.api.types.is_numeric_dtype(frame[column])
        or pd.api.types.is_bool_dtype(frame[column])
    ]
    categorical = [column for column in columns if column not in numeric]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(
                                strategy="median",
                                add_indicator=True,
                                keep_empty_features=True,
                            ),
                        ),
                        (
                            "scale",
                            StandardScaler(with_mean=True, with_std=True),
                        ),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="__MISSING__",
                                keep_empty_features=True,
                            ),
                        ),
                        (
                            "one_hot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=True,
                                dtype=np.float64,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        raise ValueError("at least one model feature is required")
    return ColumnTransformer(transformers, remainder="drop")


def _fit_predict_ridge(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    train_outcome: np.ndarray,
    train_weights: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if len(train_outcome) != len(train_frame):
        raise ValueError("train_outcome length differs from training frame")
    if len(train_weights) != len(train_frame):
        raise ValueError("train_weights length differs from training frame")
    if not np.isfinite(train_outcome).all():
        raise ValueError("train_outcome contains non-finite values")
    _validate_feature_columns(train_frame, columns, label="split training features")
    _validate_feature_columns(valid_frame, columns, label="split validation features")
    estimator = Pipeline(
        [
            ("preprocess", _preprocessor(train_frame, columns)),
            (
                "ridge",
                Ridge(
                    alpha=float(alpha),
                    fit_intercept=True,
                    solver="lsqr",
                    tol=1e-8,
                    max_iter=10_000,
                ),
            ),
        ]
    )
    estimator.fit(
        train_frame[list(columns)],
        train_outcome,
        ridge__sample_weight=train_weights,
    )
    prediction = estimator.predict(valid_frame[list(columns)])
    prediction = np.asarray(prediction, dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise ValueError("Ridge produced non-finite predictions")
    return np.clip(prediction, 0.0, 1.0)


def _weighted_mse(
    observed: np.ndarray, prediction: np.ndarray, weights: np.ndarray
) -> float:
    if not (
        len(observed) == len(prediction) == len(weights)
        and len(observed) > 0
    ):
        raise ValueError("weighted MSE arrays must have equal non-zero length")
    if not (
        np.isfinite(observed).all()
        and np.isfinite(prediction).all()
        and np.isfinite(weights).all()
        and np.all(weights > 0.0)
    ):
        raise ValueError("weighted MSE inputs must be finite with positive weights")
    return float(np.average((observed - prediction) ** 2, weights=weights))


def _nested_group_ridge_oof_comparison(
    frame: pd.DataFrame,
    *,
    outcome_column: str | None,
    context_columns: Sequence[str],
    morphology_columns: Sequence[str],
    group_column: str = "report_group",
    outer_fold_column: str | None = "outer_fold",
    n_outer_folds: int = OUTER_FOLDS,
    n_inner_folds: int = INNER_FOLDS,
    seed: int = FOLD_SEED,
    alpha_grid: Sequence[float] = RIDGE_ALPHA_GRID,
    inner_fold_columns: Mapping[int, str] | None = None,
    split_outcome_provider: Any | None = None,
    split_feature_provider: Any | None = None,
    split_feature_columns: Sequence[str] = (),
    split_feature_key_column: str = "StudyInstanceUID",
) -> NestedRidgeComparison:
    """Compare context and context-plus-morphology with nested grouped OOF.

    This function operates on one already selected target-by-direction cohort.
    Preprocessing is fitted inside every inner/outer training partition.  Ridge
    fitting and MSE use inverse report-group-size weights.  Predictions are
    clipped to the contract's risk-score range before both alpha selection and
    reported OOF evaluation.  The largest alpha wins all MSE ties within
    ``1e-12``.
    """

    if frame.empty:
        raise ValueError("nested OOF input is empty")
    if not frame.index.is_unique:
        raise ValueError("nested OOF frame index must be unique")
    required = [group_column]
    if split_outcome_provider is None:
        if outcome_column is None:
            raise ValueError("outcome_column is required without a split provider")
        required.append(outcome_column)
    _required_columns(frame, required)
    if split_feature_provider is not None and not callable(split_feature_provider):
        raise ValueError("split_feature_provider must be callable")
    declared_model_columns = (*tuple(context_columns), *tuple(morphology_columns))
    dynamic_columns = _normalized_split_feature_columns(
        split_feature_columns,
        model_columns=declared_model_columns,
        provider=split_feature_provider,
    )
    if dynamic_columns:
        frame = frame.copy()
        for column in dynamic_columns:
            if column not in frame.columns:
                frame[column] = np.nan
        _required_columns(frame, [split_feature_key_column])
    context = _validate_feature_columns(frame, context_columns, label="context_columns")
    morphology = _validate_feature_columns(
        frame, morphology_columns, label="morphology_columns"
    )
    overlap = sorted(set(context).intersection(morphology))
    if overlap:
        raise ValueError(f"context and morphology columns overlap: {overlap}")

    fixed_outcome = (
        _validate_unit_interval(frame[outcome_column].to_numpy(), name=outcome_column)
        if split_outcome_provider is None and outcome_column is not None
        else None
    )

    def split_outcomes(
        train_indices: np.ndarray,
        valid_indices: np.ndarray,
        *,
        outer_fold: int,
        inner_fold: int | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if split_outcome_provider is None:
            if fixed_outcome is None:
                raise AssertionError("fixed outcome was not initialized")
            return fixed_outcome[train_indices], fixed_outcome[valid_indices]
        train_values, valid_values = split_outcome_provider(
            train_indices.copy(),
            valid_indices.copy(),
            outer_fold,
            inner_fold,
        )
        train_array = _validate_unit_interval(train_values, name="split train outcome")
        valid_array = _validate_unit_interval(valid_values, name="split valid outcome")
        if len(train_array) != len(train_indices) or len(valid_array) != len(valid_indices):
            raise ValueError("split outcome provider returned incorrect lengths")
        return train_array, valid_array
    groups = _nonempty_string_keys(frame[group_column].to_numpy(), name=group_column)
    weights = inverse_report_group_weights(groups)
    outer_folds = _validated_outer_folds(
        frame,
        group_column=group_column,
        outer_fold_column=outer_fold_column,
        n_splits=n_outer_folds,
        seed=seed,
    )
    if not isinstance(n_inner_folds, (int, np.integer)) or int(n_inner_folds) < 2:
        raise ValueError("n_inner_folds must be an integer of at least two")

    alphas = np.asarray(tuple(alpha_grid), dtype=np.float64)
    if (
        alphas.ndim != 1
        or len(alphas) == 0
        or not np.isfinite(alphas).all()
        or np.any(alphas <= 0.0)
        or len(np.unique(alphas)) != len(alphas)
    ):
        raise ValueError("alpha_grid must contain unique finite positive values")
    alphas = np.sort(alphas)

    model_columns = {
        "context": context,
        "context_plus_morphology": (*context, *morphology),
    }
    predictions = {
        model_name: np.full(len(frame), np.nan, dtype=np.float64)
        for model_name in model_columns
    }
    observed_oof = np.full(len(frame), np.nan, dtype=np.float64)
    trace: list[dict[str, float | int | str | bool]] = []

    for outer_fold in range(int(n_outer_folds)):
        outer_valid = np.flatnonzero(outer_folds == outer_fold)
        outer_train = np.flatnonzero(outer_folds != outer_fold)
        if len(outer_valid) == 0 or len(outer_train) == 0:
            raise ValueError(f"outer fold {outer_fold} has an empty partition")
        if inner_fold_columns is None:
            inner_folds = _shuffled_group_fold_ids(
                groups[outer_train],
                n_splits=int(n_inner_folds),
                seed=int(seed) + outer_fold,
            )
        else:
            expected_keys = set(range(int(n_outer_folds)))
            if set(inner_fold_columns) != expected_keys:
                raise ValueError(
                    "inner_fold_columns must map every outer fold exactly once"
                )
            inner_column = str(inner_fold_columns[outer_fold])
            inner_folds = _validated_outer_folds(
                frame.iloc[outer_train],
                group_column=group_column,
                outer_fold_column=inner_column,
                n_splits=int(n_inner_folds),
                seed=int(seed) + outer_fold,
            )
        outer_train_outcome, outer_valid_outcome = split_outcomes(
            outer_train,
            outer_valid,
            outer_fold=outer_fold,
            inner_fold=None,
        )
        outer_train_frame, outer_valid_frame = _materialize_split_feature_frames(
            frame,
            train_indices=outer_train,
            valid_indices=outer_valid,
            provider=split_feature_provider,
            feature_columns=dynamic_columns,
            key_column=split_feature_key_column,
            group_column=group_column,
            outer_fold=outer_fold,
            inner_fold=None,
        )
        observed_oof[outer_valid] = outer_valid_outcome
        inner_splits: list[
            tuple[
                np.ndarray,
                np.ndarray,
                pd.DataFrame,
                pd.DataFrame,
                np.ndarray,
                np.ndarray,
            ]
        ] = []
        for inner_fold in range(int(n_inner_folds)):
            local_valid = np.flatnonzero(inner_folds == inner_fold)
            local_train = np.flatnonzero(inner_folds != inner_fold)
            if len(local_valid) == 0 or len(local_train) == 0:
                raise ValueError(
                    f"inner fold {inner_fold} for outer fold {outer_fold} is empty"
                )
            global_valid = outer_train[local_valid]
            global_train = outer_train[local_train]
            inner_train_outcome, inner_valid_outcome = split_outcomes(
                global_train,
                global_valid,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
            )
            inner_train_frame, inner_valid_frame = _materialize_split_feature_frames(
                frame,
                train_indices=global_train,
                valid_indices=global_valid,
                provider=split_feature_provider,
                feature_columns=dynamic_columns,
                key_column=split_feature_key_column,
                group_column=group_column,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
            )
            inner_splits.append(
                (
                    local_valid,
                    global_train,
                    inner_train_frame,
                    inner_valid_frame,
                    inner_train_outcome,
                    inner_valid_outcome,
                )
            )

        for model_name, columns in model_columns.items():
            mse_by_alpha: dict[float, float] = {}
            for alpha in alphas:
                inner_prediction = np.full(len(outer_train), np.nan, dtype=np.float64)
                inner_observed = np.full(len(outer_train), np.nan, dtype=np.float64)
                for (
                    local_valid,
                    global_train,
                    inner_train_frame,
                    inner_valid_frame,
                    inner_train_outcome,
                    inner_valid_outcome,
                ) in inner_splits:
                    inner_prediction[local_valid] = _fit_predict_ridge(
                        inner_train_frame,
                        inner_valid_frame,
                        columns=columns,
                        train_outcome=inner_train_outcome,
                        train_weights=weights[global_train],
                        alpha=float(alpha),
                    )
                    inner_observed[local_valid] = inner_valid_outcome
                if not (
                    np.isfinite(inner_prediction).all()
                    and np.isfinite(inner_observed).all()
                ):
                    raise AssertionError("inner OOF outcome/prediction coverage is incomplete")
                mse_by_alpha[float(alpha)] = _weighted_mse(
                    inner_observed, inner_prediction, weights[outer_train]
                )

            best_mse = min(mse_by_alpha.values())
            tied = [
                alpha
                for alpha, mse in mse_by_alpha.items()
                if mse <= best_mse + ALPHA_TIE_TOLERANCE
            ]
            selected_alpha = max(tied)
            for alpha in alphas:
                trace.append(
                    {
                        "outer_fold": outer_fold,
                        "model": model_name,
                        "alpha": float(alpha),
                        "inner_weighted_mse": mse_by_alpha[float(alpha)],
                        "selected": bool(float(alpha) == selected_alpha),
                    }
                )
            predictions[model_name][outer_valid] = _fit_predict_ridge(
                outer_train_frame,
                outer_valid_frame,
                columns=columns,
                train_outcome=outer_train_outcome,
                train_weights=weights[outer_train],
                alpha=selected_alpha,
            )

    for model_name, prediction in predictions.items():
        if not np.isfinite(prediction).all():
            raise AssertionError(f"{model_name} outer OOF prediction coverage is incomplete")
    if not np.isfinite(observed_oof).all():
        raise AssertionError("outer OOF outcome coverage is incomplete")
    context_mse = _weighted_mse(observed_oof, predictions["context"], weights)
    augmented_mse = _weighted_mse(
        observed_oof, predictions["context_plus_morphology"], weights
    )
    oof = pd.DataFrame(
        {
            "outer_fold": outer_folds,
            "report_group_weight": weights,
            "observed": observed_oof,
            "context_oof": predictions["context"],
            "context_plus_morphology_oof": predictions[
                "context_plus_morphology"
            ],
        },
        index=frame.index,
    )
    selections = pd.DataFrame(trace).sort_values(
        ["outer_fold", "model", "alpha"], kind="mergesort"
    ).reset_index(drop=True)
    return NestedRidgeComparison(
        oof=oof,
        selections=selections,
        metrics={
            "rows": int(len(frame)),
            "report_groups": int(pd.Series(groups).nunique()),
            "context_weighted_mse": context_mse,
            "context_plus_morphology_weighted_mse": augmented_mse,
            "morphology_increment": context_mse - augmented_mse,
        },
    )


def nested_group_ridge_oof_comparison(
    frame: pd.DataFrame,
    *,
    outcome_column: str,
    context_columns: Sequence[str],
    morphology_columns: Sequence[str],
    group_column: str = "report_group",
    outer_fold_column: str | None = "outer_fold",
    n_outer_folds: int = OUTER_FOLDS,
    n_inner_folds: int = INNER_FOLDS,
    seed: int = FOLD_SEED,
    alpha_grid: Sequence[float] = RIDGE_ALPHA_GRID,
    inner_fold_columns: Mapping[int, str] | None = None,
    split_feature_provider: Any | None = None,
    split_feature_columns: Sequence[str] = (),
    split_feature_key_column: str = "StudyInstanceUID",
) -> NestedRidgeComparison:
    """Public fixed-outcome wrapper around nested grouped weighted Ridge."""

    return _nested_group_ridge_oof_comparison(
        frame,
        outcome_column=outcome_column,
        context_columns=context_columns,
        morphology_columns=morphology_columns,
        group_column=group_column,
        outer_fold_column=outer_fold_column,
        n_outer_folds=n_outer_folds,
        n_inner_folds=n_inner_folds,
        seed=seed,
        alpha_grid=alpha_grid,
        inner_fold_columns=inner_fold_columns,
        split_feature_provider=split_feature_provider,
        split_feature_columns=split_feature_columns,
        split_feature_key_column=split_feature_key_column,
    )


def nested_group_family_balanced_ridge_oof_comparison(
    frame: pd.DataFrame,
    *,
    ecdf_reference_frame: pd.DataFrame,
    direction: Direction,
    dino_column: str,
    rad_component_columns: Sequence[str],
    target_column: str,
    context_columns: Sequence[str],
    morphology_columns: Sequence[str],
    group_column: str = "report_group",
    outer_fold_column: str | None = "outer_fold",
    n_outer_folds: int = OUTER_FOLDS,
    n_inner_folds: int = INNER_FOLDS,
    seed: int = FOLD_SEED,
    alpha_grid: Sequence[float] = RIDGE_ALPHA_GRID,
    disagreement_gate: float = 0.10,
    minimum_outer_rows: int = 20,
    minimum_outer_report_groups: int = 20,
    inner_fold_columns: Mapping[int, str],
    split_feature_provider: Any | None = None,
    split_feature_columns: Sequence[str] = (),
    split_feature_key_column: str = "StudyInstanceUID",
) -> NestedRidgeComparison:
    """Nested Ridge whose image endpoint is transformed inside every split.

    ``frame`` is one target-by-teacher-polarity model cohort, while
    ``ecdf_reference_frame`` is the complete decisive eligible proxy cohort for
    that target across both teacher polarities.  The fitting reference for every
    outer or inner validation partition contains reference rows from the
    corresponding training folds only.  DINO and the row-median Rad family each
    receive exactly one half of the continuous endpoint.  Image-family
    disagreement and within-Rad dispersion are returned as guardrails and are
    never model predictors.
    """

    if direction not in ("omission", "contradiction"):
        raise ValueError("direction must be omission or contradiction")
    if tuple(float(value) for value in alpha_grid) != RIDGE_ALPHA_GRID:
        raise ValueError("family-balanced analysis requires the frozen alpha grid")
    rad_columns = tuple(str(column) for column in rad_component_columns)
    if len(rad_columns) != 3:
        raise ValueError("rad_component_columns must contain exactly three columns")
    prohibited_predictors = {
        target_column,
        group_column,
        dino_column,
        *rad_columns,
        *DERIVED_IMAGE_EVIDENCE_COLUMNS,
    }
    leaked = sorted(
        prohibited_predictors.intersection(
            {*tuple(context_columns), *tuple(morphology_columns)}
        )
    )
    if leaked:
        raise ValueError(f"image/identity columns cannot be model predictors: {leaked}")
    if outer_fold_column is None:
        raise ValueError(
            "family-balanced nested comparison requires an explicit outer fold column"
        )
    if set(inner_fold_columns) != set(range(int(n_outer_folds))):
        raise ValueError("inner_fold_columns must map every outer fold exactly once")
    resolved_inner = {
        int(outer_fold): str(column)
        for outer_fold, column in inner_fold_columns.items()
    }
    if len(set(resolved_inner.values())) != len(resolved_inner):
        raise ValueError("inner_fold_columns must contain distinct columns")
    required = [
        target_column,
        group_column,
        dino_column,
        *rad_columns,
        str(outer_fold_column),
        *resolved_inner.values(),
    ]
    _required_columns(frame, required)
    _required_columns(ecdf_reference_frame, required)
    if frame.empty or ecdf_reference_frame.empty:
        raise ValueError("model and ECDF reference frames must be non-empty")
    if not frame.index.is_unique or not ecdf_reference_frame.index.is_unique:
        raise ValueError("model and ECDF reference frame indices must be unique")
    target_keys = _nonempty_string_keys(frame[target_column], name=target_column)
    if len(set(target_keys.tolist())) != 1:
        raise ValueError("family-balanced nested comparison requires exactly one target")
    reference_targets = _nonempty_string_keys(
        ecdf_reference_frame[target_column], name=f"reference {target_column}"
    )
    if set(reference_targets.tolist()) != set(target_keys.tolist()):
        raise ValueError("ECDF reference must contain exactly the model target")
    if not frame.index.isin(ecdf_reference_frame.index).all():
        raise ValueError("every model row must be present in the ECDF reference frame")
    shared_reference = ecdf_reference_frame.loc[frame.index]
    for column in [target_column, group_column]:
        left = _nonempty_string_keys(frame[column], name=column)
        right = _nonempty_string_keys(
            shared_reference[column], name=f"reference {column}"
        )
        if not np.array_equal(left, right):
            raise ValueError(f"ECDF reference identity mismatch for {column}")
    for column in (dino_column, *rad_columns):
        _validate_unit_interval(frame[column], name=column)
        _validate_unit_interval(
            ecdf_reference_frame[column], name=f"reference {column}"
        )
        if not np.array_equal(
            _finite_numeric(frame[column], name=column),
            _finite_numeric(shared_reference[column], name=f"reference {column}"),
        ):
            raise ValueError(f"ECDF reference value mismatch for {column}")
    for column in [str(outer_fold_column), *resolved_inner.values()]:
        if not np.array_equal(
            _finite_numeric(frame[column], name=column),
            _finite_numeric(shared_reference[column], name=f"reference {column}"),
        ):
            raise ValueError(f"ECDF reference fold mismatch for {column}")
    if int(minimum_outer_rows) < 1 or int(minimum_outer_report_groups) < 1:
        raise ValueError("outer support minima must be positive integers")
    support_folds = _validated_outer_folds(
        frame,
        group_column=group_column,
        outer_fold_column=outer_fold_column,
        n_splits=n_outer_folds,
        seed=seed,
    )
    reference_outer_folds = _validated_outer_folds(
        ecdf_reference_frame,
        group_column=group_column,
        outer_fold_column=str(outer_fold_column),
        n_splits=n_outer_folds,
        seed=seed,
    )
    reference_inner_folds: dict[int, np.ndarray] = {}
    for outer_fold, inner_column in resolved_inner.items():
        reference_outer_train = reference_outer_folds != outer_fold
        reference_inner_folds[outer_fold] = _validated_outer_folds(
            ecdf_reference_frame.loc[reference_outer_train],
            group_column=group_column,
            outer_fold_column=inner_column,
            n_splits=n_inner_folds,
            seed=int(seed) + outer_fold,
        )
    support_groups = _nonempty_string_keys(frame[group_column], name=group_column)
    for fold in range(int(n_outer_folds)):
        for scope, mask in (
            ("train", support_folds != fold),
            ("validation", support_folds == fold),
        ):
            rows = int(mask.sum())
            groups_count = int(pd.Series(support_groups[mask]).nunique())
            if rows < int(minimum_outer_rows) or groups_count < int(
                minimum_outer_report_groups
            ):
                raise ValueError(
                    f"outer {scope} support gate failed for fold {fold}: "
                    f"rows={rows}, report_groups={groups_count}"
                )

    def provider(
        train_indices: np.ndarray,
        valid_indices: np.ndarray,
        outer_fold: int,
        inner_fold: int | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        reference_mask = reference_outer_folds != outer_fold
        if inner_fold is not None:
            outer_train_positions = np.flatnonzero(reference_mask)
            local_inner = reference_inner_folds[outer_fold]
            if len(local_inner) != len(outer_train_positions):
                raise AssertionError("reference inner-fold map length mismatch")
            reference_mask[outer_train_positions[local_inner == inner_fold]] = False
        reference_frame = ecdf_reference_frame.loc[reference_mask]
        reference = fit_family_balanced_image_reference(
            reference_frame,
            dino_column=dino_column,
            rad_component_columns=rad_columns,
            target_column=target_column,
            disagreement_gate=disagreement_gate,
        )
        train_evidence = reference.transform(frame.iloc[train_indices])
        valid_evidence = reference.transform(frame.iloc[valid_indices])
        if direction == "omission":
            return (
                train_evidence["image_consensus_rank"].to_numpy(float),
                valid_evidence["image_consensus_rank"].to_numpy(float),
            )
        return (
            1.0 - train_evidence["image_consensus_rank"].to_numpy(float),
            1.0 - valid_evidence["image_consensus_rank"].to_numpy(float),
        )

    result = _nested_group_ridge_oof_comparison(
        frame,
        outcome_column=None,
        context_columns=context_columns,
        morphology_columns=morphology_columns,
        group_column=group_column,
        outer_fold_column=outer_fold_column,
        n_outer_folds=n_outer_folds,
        n_inner_folds=n_inner_folds,
        seed=seed,
        alpha_grid=alpha_grid,
        inner_fold_columns=resolved_inner,
        split_outcome_provider=provider,
        split_feature_provider=split_feature_provider,
        split_feature_columns=split_feature_columns,
        split_feature_key_column=split_feature_key_column,
    )
    # Rebuild exact outer-fold evidence from the all-polarity target reference.
    evidence_parts: list[pd.DataFrame] = []
    for outer_fold in range(int(n_outer_folds)):
        valid = support_folds == outer_fold
        reference = fit_family_balanced_image_reference(
            ecdf_reference_frame.loc[reference_outer_folds != outer_fold],
            dino_column=dino_column,
            rad_component_columns=rad_columns,
            target_column=target_column,
            disagreement_gate=disagreement_gate,
        )
        part = reference.transform(frame.loc[valid])
        part["_source_position"] = np.flatnonzero(valid)
        evidence_parts.append(part)
    outer_evidence = pd.concat(evidence_parts).sort_values(
        "_source_position", kind="mergesort"
    )
    outer_evidence = outer_evidence.drop(columns="_source_position")
    outer_evidence = outer_evidence.reindex(frame.index)
    if outer_evidence.isna().any().any():
        raise AssertionError("outer family-evidence coverage is incomplete")
    oof = result.oof.copy()
    for column in outer_evidence.columns:
        oof[column] = outer_evidence[column].to_numpy()
    expected = (
        oof["image_consensus_rank"].to_numpy(float)
        if direction == "omission"
        else 1.0 - oof["image_consensus_rank"].to_numpy(float)
    )
    if not np.allclose(
        oof["observed"].to_numpy(float), expected, rtol=0.0, atol=1e-12
    ):
        raise AssertionError("outer endpoint differs from held-out family evidence")
    return NestedRidgeComparison(
        oof=oof,
        selections=result.selections,
        metrics=result.metrics,
    )


def nested_target_direction_ridge_oof(
    frame: pd.DataFrame,
    *,
    context_columns: Sequence[str],
    morphology_columns: Sequence[str],
    target_column: str = "target",
    teacher_label_column: str = "teacher_hard_label",
    omission_outcome_column: str = "omission_like_continuous",
    contradiction_outcome_column: str = "contradiction_like_continuous",
    group_column: str = "report_group",
    outer_fold_column: str | None = "outer_fold",
    n_outer_folds: int = OUTER_FOLDS,
    n_inner_folds: int = INNER_FOLDS,
    seed: int = FOLD_SEED,
    alpha_grid: Sequence[float] = RIDGE_ALPHA_GRID,
) -> DirectionalNestedRidgeResult:
    """Run the contract model separately for every target and direction.

    Decisive teacher labels that vary inside ``report_group x target`` cause the
    entire group-target block to be excluded and recorded.  Abstentions are not
    modelled.  The function returns only proxy OOF ranks and never consumes Gold.
    """

    _required_columns(
        frame,
        [
            target_column,
            teacher_label_column,
            omission_outcome_column,
            contradiction_outcome_column,
            group_column,
            *(tuple([outer_fold_column]) if outer_fold_column is not None else ()),
        ],
    )
    if frame.empty:
        raise ValueError("target-direction OOF input is empty")
    if not frame.index.is_unique:
        raise ValueError("target-direction OOF frame index must be unique")
    targets = _nonempty_string_keys(frame[target_column].to_numpy(), name=target_column)
    groups = _nonempty_string_keys(frame[group_column].to_numpy(), name=group_column)
    labels = pd.to_numeric(frame[teacher_label_column], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if np.isinf(labels).any() or not np.isin(labels[np.isfinite(labels)], [0.0, 1.0]).all():
        raise ValueError(f"{teacher_label_column} must contain only 0, 1, or null")

    audit = pd.DataFrame(
        {
            "_target": targets,
            "_group": groups,
            "_label": labels,
            "_row": np.arange(len(frame), dtype=np.int64),
        }
    )
    decisive_audit = audit.loc[np.isfinite(audit["_label"])]
    label_counts = decisive_audit.groupby(
        ["_target", "_group"], sort=True, observed=True
    )["_label"].nunique()
    conflicts = set(label_counts.loc[label_counts.gt(1)].index.tolist())
    exclusion_rows = [
        {
            "target": target,
            "report_group": group,
            "reason": "NONHOMOGENEOUS_TEACHER_POLARITY",
        }
        for target, group in sorted(conflicts)
    ]

    oof_parts: list[pd.DataFrame] = []
    selection_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, float | int | str]] = []
    direction_specs: tuple[tuple[Direction, float, str], ...] = (
        ("omission", 0.0, omission_outcome_column),
        ("contradiction", 1.0, contradiction_outcome_column),
    )
    for target in sorted(set(targets.tolist())):
        for direction, required_label, outcome_column in direction_specs:
            keep = (
                (targets == target)
                & (labels == required_label)
                & np.asarray(
                    [(target, group) not in conflicts for group in groups], dtype=bool
                )
            )
            positions = np.flatnonzero(keep)
            if len(positions) == 0:
                raise ValueError(f"no eligible rows for target={target}, direction={direction}")
            subset = frame.iloc[positions].copy()
            result = nested_group_ridge_oof_comparison(
                subset,
                outcome_column=outcome_column,
                context_columns=context_columns,
                morphology_columns=morphology_columns,
                group_column=group_column,
                outer_fold_column=outer_fold_column,
                n_outer_folds=n_outer_folds,
                n_inner_folds=n_inner_folds,
                seed=seed,
                alpha_grid=alpha_grid,
            )
            local_oof = result.oof.copy()
            local_oof.insert(0, "direction", direction)
            local_oof.insert(0, "target", target)
            local_risk = local_oof[
                "context_plus_morphology_oof"
            ].to_numpy(dtype=np.float64)
            local_oof["risk_percentile"] = _apply_sorted_average_tie_reference(
                np.sort(local_risk, kind="mergesort"), local_risk
            )
            local_oof["_source_row"] = positions
            oof_parts.append(local_oof)

            local_selection = result.selections.copy()
            local_selection.insert(0, "direction", direction)
            local_selection.insert(0, "target", target)
            selection_parts.append(local_selection)
            metric_rows.append(
                {"target": target, "direction": direction, **result.metrics}
            )

    oof = pd.concat(oof_parts, axis=0).sort_values(
        "_source_row", kind="mergesort"
    )
    oof = oof.drop(columns="_source_row")
    selections = pd.concat(selection_parts, ignore_index=True).sort_values(
        ["target", "direction", "outer_fold", "model", "alpha"],
        kind="mergesort",
    ).reset_index(drop=True)
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["target", "direction"], kind="mergesort"
    ).reset_index(drop=True)
    exclusions = pd.DataFrame(
        exclusion_rows, columns=["target", "report_group", "reason"]
    )
    return DirectionalNestedRidgeResult(
        oof=oof,
        selections=selections,
        metrics=metrics,
        exclusions=exclusions,
    )


def nested_target_direction_family_balanced_ridge_oof(
    frame: pd.DataFrame,
    *,
    dino_column: str,
    rad_component_columns: Sequence[str],
    context_columns: Sequence[str],
    morphology_columns: Sequence[str],
    target_column: str = "target",
    teacher_label_column: str = "teacher_hard_label",
    group_column: str = "report_group",
    outer_fold_column: str | None = "outer_fold",
    n_outer_folds: int = OUTER_FOLDS,
    n_inner_folds: int = INNER_FOLDS,
    seed: int = FOLD_SEED,
    alpha_grid: Sequence[float] = RIDGE_ALPHA_GRID,
    disagreement_gate: float = 0.10,
    minimum_outer_rows: int = 20,
    minimum_outer_report_groups: int = 20,
    inner_fold_columns: Mapping[int, str] | None = None,
    split_feature_provider: Any | None = None,
    split_feature_columns: Sequence[str] = (),
    split_feature_key_column: str = "StudyInstanceUID",
) -> DirectionalNestedRidgeResult:
    """Run target/direction models with split-local family-balanced endpoints.

    Component ECDFs are always fit on all decisive eligible rows for the target
    across both teacher polarities, restricted to the relevant outer/inner
    training fold.  If the frozen inner-fold columns are not supplied, one
    deterministic global map is materialized once on ``frame`` for synthetic
    or fold-map-construction use; it is never regenerated by direction.
    """

    rad_columns = tuple(str(column) for column in rad_component_columns)
    if len(rad_columns) != 3 or len(set(rad_columns)) != 3:
        raise ValueError("rad_component_columns must contain exactly three columns")
    _required_columns(
        frame,
        [
            target_column,
            teacher_label_column,
            group_column,
            dino_column,
            *rad_columns,
            *(tuple([outer_fold_column]) if outer_fold_column is not None else ()),
        ],
    )
    if frame.empty or not frame.index.is_unique:
        raise ValueError("input must be non-empty with a unique index")
    prohibited = {
        target_column,
        dino_column,
        *rad_columns,
        teacher_label_column,
        group_column,
        *DERIVED_IMAGE_EVIDENCE_COLUMNS,
    }
    leaked = sorted(prohibited.intersection({*context_columns, *morphology_columns}))
    if leaked:
        raise ValueError(f"image/teacher/group columns cannot be model predictors: {leaked}")
    for column in (dino_column, *rad_columns):
        _validate_unit_interval(frame[column], name=column)

    frame, resolved_outer_fold_column, resolved_inner_fold_columns = (
        _materialize_contract_nested_fold_columns(
            frame,
            group_column=group_column,
            outer_fold_column=outer_fold_column,
            inner_fold_columns=inner_fold_columns,
            n_outer_folds=n_outer_folds,
            n_inner_folds=n_inner_folds,
            seed=seed,
        )
    )

    targets = _nonempty_string_keys(frame[target_column], name=target_column)
    groups = _nonempty_string_keys(frame[group_column], name=group_column)
    labels = pd.to_numeric(frame[teacher_label_column], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if np.isinf(labels).any() or not np.isin(labels[np.isfinite(labels)], [0.0, 1.0]).all():
        raise ValueError(f"{teacher_label_column} must contain only 0, 1, or null")
    decisive = pd.DataFrame(
        {"target": targets, "group": groups, "label": labels}
    ).loc[np.isfinite(labels)]
    counts = decisive.groupby(["target", "group"], sort=True)["label"].nunique()
    conflicts = set(counts.loc[counts.gt(1)].index.tolist())
    exclusions = pd.DataFrame(
        [
            {
                "target": target,
                "report_group": group,
                "reason": "NONHOMOGENEOUS_TEACHER_POLARITY",
            }
            for target, group in sorted(conflicts)
        ],
        columns=["target", "report_group", "reason"],
    )

    oof_parts: list[pd.DataFrame] = []
    selection_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, float | int | str]] = []
    for target in sorted(set(targets.tolist())):
        eligible_reference = (
            (targets == target)
            & np.isfinite(labels)
            & np.asarray(
                [(target, group) not in conflicts for group in groups], dtype=bool
            )
        )
        target_reference = frame.iloc[np.flatnonzero(eligible_reference)].copy()
        if target_reference.empty:
            raise ValueError(f"no eligible ECDF reference rows for target={target}")
        for direction, required_label in (
            ("omission", 0.0),
            ("contradiction", 1.0),
        ):
            keep = (
                (targets == target)
                & (labels == required_label)
                & np.asarray(
                    [(target, group) not in conflicts for group in groups], dtype=bool
                )
            )
            positions = np.flatnonzero(keep)
            if len(positions) == 0:
                raise ValueError(f"no eligible rows for target={target}, direction={direction}")
            subset = frame.iloc[positions].copy()
            nested = nested_group_family_balanced_ridge_oof_comparison(
                subset,
                ecdf_reference_frame=target_reference,
                direction=direction,
                dino_column=dino_column,
                rad_component_columns=rad_columns,
                target_column=target_column,
                context_columns=context_columns,
                morphology_columns=morphology_columns,
                group_column=group_column,
                outer_fold_column=resolved_outer_fold_column,
                n_outer_folds=n_outer_folds,
                n_inner_folds=n_inner_folds,
                seed=seed,
                alpha_grid=alpha_grid,
                disagreement_gate=disagreement_gate,
                minimum_outer_rows=minimum_outer_rows,
                minimum_outer_report_groups=minimum_outer_report_groups,
                inner_fold_columns=resolved_inner_fold_columns,
                split_feature_provider=split_feature_provider,
                split_feature_columns=split_feature_columns,
                split_feature_key_column=split_feature_key_column,
            )
            local_oof = nested.oof.copy()
            local_oof.insert(0, "direction", direction)
            local_oof.insert(0, "target", target)
            local_risk = local_oof[
                "context_plus_morphology_oof"
            ].to_numpy(dtype=np.float64)
            local_oof["risk_percentile"] = _apply_sorted_average_tie_reference(
                np.sort(local_risk, kind="mergesort"), local_risk
            )
            guardrail = local_oof["image_high_agreement"].to_numpy(bool)
            local_oof["guardrailed_review_risk_percentile"] = np.where(
                guardrail,
                local_oof["risk_percentile"].to_numpy(float),
                np.nan,
            )
            local_oof["_source_row"] = positions
            oof_parts.append(local_oof)

            selection = nested.selections.copy()
            selection.insert(0, "direction", direction)
            selection.insert(0, "target", target)
            selection_parts.append(selection)
            metric_rows.append(
                {"target": target, "direction": direction, **nested.metrics}
            )

    oof = pd.concat(oof_parts).sort_values("_source_row", kind="mergesort")
    oof = oof.drop(columns="_source_row")
    selections = pd.concat(selection_parts, ignore_index=True).sort_values(
        ["target", "direction", "outer_fold", "model", "alpha"],
        kind="mergesort",
    ).reset_index(drop=True)
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["target", "direction"], kind="mergesort"
    ).reset_index(drop=True)
    return DirectionalNestedRidgeResult(
        oof=oof,
        selections=selections,
        metrics=metrics,
        exclusions=exclusions,
    )


def select_final_directional_ridge_alpha(
    frame: pd.DataFrame,
    *,
    ecdf_reference_frame: pd.DataFrame,
    direction: Direction,
    dino_column: str,
    rad_component_columns: Sequence[str],
    context_columns: Sequence[str],
    morphology_columns: Sequence[str],
    target_column: str = "target",
    teacher_label_column: str = "teacher_hard_label",
    group_column: str = "report_group",
    disagreement_gate: float = 0.10,
    split_feature_provider: Any | None = None,
    split_feature_columns: Sequence[str] = (),
    split_feature_key_column: str = "StudyInstanceUID",
) -> FinalAlphaSelection:
    """Select the final alpha by the contract's fresh proxy-only 3-fold CV.

    The three folds are created from the target-direction report groups with
    seed ``20260930``.  For each held-out fold, the family ECDF excludes its
    report groups while retaining every other decisive eligible target row.
    Any split-feature provider is invoked exactly once per fold and receives
    only disjoint train/validation model frames plus fold/group metadata; it may
    capture a global all-target report reference to fit outcome-blind text
    transforms on the complete training-group corpus.
    """

    if direction not in ("omission", "contradiction"):
        raise ValueError("direction must be omission or contradiction")
    rad_columns = tuple(str(column) for column in rad_component_columns)
    if len(rad_columns) != 3 or len(set(rad_columns)) != 3:
        raise ValueError("rad_component_columns must contain exactly three columns")
    features = (*tuple(context_columns), *tuple(morphology_columns))
    if not features or len(features) != len(set(features)):
        raise ValueError("final-selection feature columns must be non-empty and unique")
    prohibited = {
        target_column,
        teacher_label_column,
        group_column,
        dino_column,
        *rad_columns,
        *DERIVED_IMAGE_EVIDENCE_COLUMNS,
    }
    leaked = sorted(prohibited.intersection(features))
    if leaked:
        raise ValueError(f"image/teacher/group columns cannot be model predictors: {leaked}")
    required = [
        target_column,
        teacher_label_column,
        group_column,
        dino_column,
        *rad_columns,
    ]
    _required_columns(frame, required)
    _required_columns(ecdf_reference_frame, required)
    if frame.empty or ecdf_reference_frame.empty:
        raise ValueError("final-selection frames must be non-empty")
    if not frame.index.is_unique or not ecdf_reference_frame.index.is_unique:
        raise ValueError("final-selection frame indices must be unique")
    if split_feature_provider is not None and not callable(split_feature_provider):
        raise ValueError("split_feature_provider must be callable")
    dynamic_columns = _normalized_split_feature_columns(
        split_feature_columns,
        model_columns=features,
        provider=split_feature_provider,
    )
    if dynamic_columns:
        frame = frame.copy()
        for column in dynamic_columns:
            if column not in frame.columns:
                frame[column] = np.nan
        _required_columns(frame, [split_feature_key_column])
    _validate_feature_columns(frame, features, label="final-selection features")

    targets = _nonempty_string_keys(frame[target_column], name=target_column)
    if len(set(targets.tolist())) != 1:
        raise ValueError("final alpha selection requires exactly one target")
    reference_targets = _nonempty_string_keys(
        ecdf_reference_frame[target_column], name=f"reference {target_column}"
    )
    if set(reference_targets.tolist()) != set(targets.tolist()):
        raise ValueError("final ECDF reference must contain exactly the model target")
    required_label = 0.0 if direction == "omission" else 1.0
    labels = _finite_numeric(frame[teacher_label_column], name=teacher_label_column)
    if not np.all(labels == required_label):
        raise ValueError("final alpha frame contains the wrong teacher polarity")
    for column in (dino_column, *rad_columns):
        _validate_unit_interval(frame[column], name=column)
        _validate_unit_interval(
            ecdf_reference_frame[column], name=f"reference {column}"
        )
    if not frame.index.isin(ecdf_reference_frame.index).all():
        raise ValueError("every final alpha row must occur in the ECDF reference")
    shared_reference = ecdf_reference_frame.loc[frame.index]
    for column in [target_column, group_column, dino_column, *rad_columns]:
        if column in (target_column, group_column):
            left = _nonempty_string_keys(frame[column], name=column)
            right = _nonempty_string_keys(
                shared_reference[column], name=f"reference {column}"
            )
        else:
            left = _finite_numeric(frame[column], name=column)
            right = _finite_numeric(
                shared_reference[column], name=f"reference {column}"
            )
        if not np.array_equal(left, right):
            raise ValueError(f"final ECDF reference mismatch for {column}")

    groups = _nonempty_string_keys(frame[group_column], name=group_column)
    reference_groups = _nonempty_string_keys(
        ecdf_reference_frame[group_column], name=f"reference {group_column}"
    )
    folds = _shuffled_group_fold_ids(
        groups, n_splits=3, seed=FINAL_ALPHA_FOLD_SEED
    )
    weights = inverse_report_group_weights(groups)
    alphas = np.asarray(RIDGE_ALPHA_GRID, dtype=np.float64)
    predictions = {
        float(alpha): np.full(len(frame), np.nan, dtype=np.float64)
        for alpha in alphas
    }
    observed = np.full(len(frame), np.nan, dtype=np.float64)

    for fold in range(3):
        valid = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        valid_groups = set(groups[valid].tolist())
        reference_train = np.asarray(
            [group not in valid_groups for group in reference_groups], dtype=bool
        )
        reference = fit_family_balanced_image_reference(
            ecdf_reference_frame.loc[reference_train],
            dino_column=dino_column,
            rad_component_columns=rad_columns,
            target_column=target_column,
            disagreement_gate=disagreement_gate,
        )
        train_evidence = reference.transform(frame.iloc[train])
        valid_evidence = reference.transform(frame.iloc[valid])
        train_outcome = train_evidence["image_consensus_rank"].to_numpy(float)
        valid_outcome = valid_evidence["image_consensus_rank"].to_numpy(float)
        if direction == "contradiction":
            train_outcome = 1.0 - train_outcome
            valid_outcome = 1.0 - valid_outcome
        observed[valid] = valid_outcome
        train_frame, valid_frame = _materialize_split_feature_frames(
            frame,
            train_indices=train,
            valid_indices=valid,
            provider=split_feature_provider,
            feature_columns=dynamic_columns,
            key_column=split_feature_key_column,
            group_column=group_column,
            outer_fold=fold,
            inner_fold=None,
        )
        for alpha in alphas:
            predictions[float(alpha)][valid] = _fit_predict_ridge(
                train_frame,
                valid_frame,
                columns=features,
                train_outcome=train_outcome,
                train_weights=weights[train],
                alpha=float(alpha),
            )

    if not np.isfinite(observed).all() or any(
        not np.isfinite(values).all() for values in predictions.values()
    ):
        raise AssertionError("final alpha OOF coverage is incomplete")
    mse_by_alpha = {
        alpha: _weighted_mse(observed, values, weights)
        for alpha, values in predictions.items()
    }
    best_mse = min(mse_by_alpha.values())
    selected_alpha = max(
        alpha
        for alpha, mse in mse_by_alpha.items()
        if mse <= best_mse + ALPHA_TIE_TOLERANCE
    )
    trace = pd.DataFrame(
        [
            {
                "alpha": alpha,
                "weighted_validation_mse": mse_by_alpha[alpha],
                "selected": bool(alpha == selected_alpha),
                "folds": 3,
                "seed": FINAL_ALPHA_FOLD_SEED,
            }
            for alpha in sorted(mse_by_alpha)
        ]
    )
    return FinalAlphaSelection(trace=trace, selected_alpha=float(selected_alpha))


def fit_final_directional_ridge_model(
    frame: pd.DataFrame,
    *,
    ecdf_reference_frame: pd.DataFrame,
    target: str,
    direction: Direction,
    alpha: float,
    proxy_oof_risk_scores: Any,
    dino_column: str,
    rad_component_columns: Sequence[str],
    context_columns: Sequence[str],
    morphology_columns: Sequence[str],
    target_column: str = "target",
    teacher_label_column: str = "teacher_hard_label",
    group_column: str = "report_group",
    disagreement_gate: float = 0.10,
    fitted_feature_transformer: Any | None = None,
    feature_transform_reference_frame: pd.DataFrame | None = None,
    split_feature_columns: Sequence[str] = (),
    split_feature_key_column: str = "StudyInstanceUID",
) -> FinalDirectionalRidgeModel:
    """Fit one final model on all eligible proxy rows, never on Gold.

    ``frame`` contains the applicable teacher-polarity rows used to fit this
    target-by-direction model.  ``ecdf_reference_frame`` contains all decisive
    eligible proxy rows for the same target across both teacher polarities.
    ``alpha`` must already be frozen from grouped proxy-only selection.  The
    model stores the full eligible-proxy image ECDF reference and the eligible-
    proxy outer-OOF risk distribution used for every later external percentile.
    External rows therefore cannot alter either reference.
    """

    if direction not in ("omission", "contradiction"):
        raise ValueError("direction must be omission or contradiction")
    alpha_value = float(alpha)
    if not np.isfinite(alpha_value) or not any(
        abs(alpha_value - frozen) <= 1e-12 for frozen in RIDGE_ALPHA_GRID
    ):
        raise ValueError("alpha must be one of the frozen Ridge alpha values")
    rad_columns = tuple(str(column) for column in rad_component_columns)
    if len(rad_columns) != 3 or len(set(rad_columns)) != 3:
        raise ValueError("rad_component_columns must contain exactly three columns")
    features = (*tuple(context_columns), *tuple(morphology_columns))
    if len(features) != len(set(features)) or not features:
        raise ValueError("final feature columns must be non-empty and unique")
    dynamic_columns = tuple(str(column) for column in split_feature_columns)
    if len(set(dynamic_columns)) != len(dynamic_columns):
        raise ValueError("split_feature_columns contains duplicates")
    unknown_dynamic = sorted(set(dynamic_columns).difference(features))
    if unknown_dynamic:
        raise ValueError(
            f"final split features are not declared predictors: {unknown_dynamic}"
        )
    if fitted_feature_transformer is not None and not dynamic_columns:
        raise ValueError("fitted transformer requires split_feature_columns")
    if fitted_feature_transformer is None and feature_transform_reference_frame is not None:
        raise ValueError(
            "feature_transform_reference_frame requires a fitted transformer"
        )
    required = [
        target_column,
        teacher_label_column,
        group_column,
        dino_column,
        *rad_columns,
    ]
    _required_columns(frame, required)
    _required_columns(ecdf_reference_frame, required)
    if frame.empty or ecdf_reference_frame.empty:
        raise ValueError("final fit and ECDF reference frames must be non-empty")
    if not frame.index.is_unique or not ecdf_reference_frame.index.is_unique:
        raise ValueError("final fit and ECDF reference indices must be unique")
    target_keys = _nonempty_string_keys(frame[target_column], name=target_column)
    if set(target_keys.tolist()) != {str(target)}:
        raise ValueError(f"final fit frame must contain only target {target}")
    required_label = 0.0 if direction == "omission" else 1.0
    labels = _finite_numeric(frame[teacher_label_column], name=teacher_label_column)
    if not np.all(labels == required_label):
        raise ValueError("final fit frame contains the wrong teacher polarity")
    reference_targets = _nonempty_string_keys(
        ecdf_reference_frame[target_column], name=f"reference {target_column}"
    )
    if set(reference_targets.tolist()) != {str(target)}:
        raise ValueError("final ECDF reference must contain exactly the fitted target")
    reference_labels = _finite_numeric(
        ecdf_reference_frame[teacher_label_column],
        name=f"reference {teacher_label_column}",
    )
    if not np.isin(reference_labels, [0.0, 1.0]).all():
        raise ValueError("final ECDF reference must contain decisive teacher labels")
    if not frame.index.isin(ecdf_reference_frame.index).all():
        raise ValueError("every final-fit row must be present in the ECDF reference")
    shared_reference = ecdf_reference_frame.loc[frame.index]
    for column in [target_column, group_column]:
        if not np.array_equal(
            _nonempty_string_keys(frame[column], name=column),
            _nonempty_string_keys(
                shared_reference[column], name=f"reference {column}"
            ),
        ):
            raise ValueError(f"final ECDF reference identity mismatch for {column}")
    for column in (dino_column, *rad_columns):
        left = _validate_unit_interval(frame[column], name=column)
        right = _validate_unit_interval(
            ecdf_reference_frame[column], name=f"reference {column}"
        )
        if not np.array_equal(
            left,
            right[ecdf_reference_frame.index.get_indexer(frame.index)],
        ):
            raise ValueError(f"final ECDF reference value mismatch for {column}")
    prohibited = {
        target_column,
        dino_column,
        *rad_columns,
        teacher_label_column,
        group_column,
        *DERIVED_IMAGE_EVIDENCE_COLUMNS,
    }
    leaked = sorted(prohibited.intersection(features))
    if leaked:
        raise ValueError(f"image/teacher/group columns cannot be model predictors: {leaked}")
    if fitted_feature_transformer is not None:
        transform_reference = (
            ecdf_reference_frame
            if feature_transform_reference_frame is None
            else feature_transform_reference_frame
        )
        transformed_reference = _apply_fitted_feature_transformer(
            transform_reference,
            transformer=fitted_feature_transformer,
            feature_columns=dynamic_columns,
            key_column=split_feature_key_column,
            training_reference=True,
        )
        frame = _merge_feature_reference_subset(
            frame,
            transformed_reference,
            feature_columns=dynamic_columns,
            key_column=split_feature_key_column,
        )
    _validate_feature_columns(frame, features, label="final feature columns")
    weights = inverse_report_group_weights(frame[group_column])
    image_reference = fit_family_balanced_image_reference(
        ecdf_reference_frame,
        dino_column=dino_column,
        rad_component_columns=rad_columns,
        target_column=target_column,
        disagreement_gate=disagreement_gate,
    )
    evidence = image_reference.transform(frame)
    outcome = evidence["image_consensus_rank"].to_numpy(float)
    if direction == "contradiction":
        outcome = 1.0 - outcome
    estimator = Pipeline(
        [
            ("preprocess", _preprocessor(frame, features)),
            (
                "ridge",
                Ridge(
                    alpha=alpha_value,
                    fit_intercept=True,
                    solver="lsqr",
                    tol=1e-8,
                    max_iter=10_000,
                ),
            ),
        ]
    )
    estimator.fit(
        frame[list(features)],
        outcome,
        ridge__sample_weight=weights,
    )
    proxy_fit_risk = np.clip(
        np.asarray(estimator.predict(frame[list(features)]), dtype=np.float64),
        0.0,
        1.0,
    )
    if not np.isfinite(proxy_fit_risk).all():
        raise ValueError("final all-proxy fit produced non-finite scores")
    proxy_oof = _validate_unit_interval(
        proxy_oof_risk_scores, name="proxy_oof_risk_scores"
    )
    if len(proxy_oof) != len(frame):
        raise ValueError(
            "proxy_oof_risk_scores must contain exactly one score per final-fit row"
        )
    sorted_risk = np.sort(proxy_oof, kind="mergesort")
    sorted_risk.setflags(write=False)
    return FinalDirectionalRidgeModel(
        target=str(target),
        direction=direction,
        target_column=target_column,
        teacher_label_column=teacher_label_column,
        feature_columns=tuple(str(column) for column in features),
        estimator=estimator,
        image_reference=image_reference,
        proxy_risk_reference=sorted_risk,
        fitted_feature_transformer=fitted_feature_transformer,
        split_feature_columns=dynamic_columns,
        split_feature_key_column=split_feature_key_column,
    )
