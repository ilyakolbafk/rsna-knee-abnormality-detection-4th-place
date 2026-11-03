"""Study-grouped Gold58 cross-fitting for score fusion and label reliability."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from rsna_knee.constants import ID_COLUMN, TARGET_COLUMNS
from rsna_knee.gold58_reliability import score_metrics


KEYS = [ID_COLUMN, "target"]
FUSION_PREDICTION_COLUMNS = [
    "primary_image_raw",
    "crossfit_image_calibration",
    "crossfit_image_plus_report",
]
RELIABILITY_PREDICTION_COLUMN = "crossfit_probability_pseudo_hard_correct"
BOOTSTRAP_SEED = 20260829


def _clip(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)


def _logit(values: pd.Series | np.ndarray) -> np.ndarray:
    probabilities = _clip(values)
    return np.log(probabilities / (1.0 - probabilities))


def _as_bool(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.casefold()
    result = normalized.map({"true": True, "false": False, "1": True, "0": False})
    if result.isna().any():
        raise ValueError(f"{name} contains non-boolean values")
    return result.astype(bool)


def validate_gold58_panel(panel: pd.DataFrame) -> None:
    """Pin the exact 58x12 Gold grid and rederive its pseudo-label outcome."""

    required = {
        *KEYS,
        "gold",
        "report_group",
        "baseline_consensus_score",
        "pseudo_hard_label",
        "pseudo_hard_correct",
        "rubric_report_stance",
        "llm_v4_blend",
        "llm_abs_disagreement",
        "public_image_oof_mean4",
        "dino_oof",
        "rad_family_probability",
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"Gold58 panel missing columns: {missing}")
    if panel[KEYS].isna().any().any() or panel.duplicated(KEYS).any():
        raise ValueError("Gold58 panel keys must be non-null and unique")
    if (
        len(panel) != 58 * len(TARGET_COLUMNS)
        or panel[ID_COLUMN].astype(str).nunique() != 58
        or set(panel["target"].astype(str)) != set(TARGET_COLUMNS)
        or not panel.groupby("target").size().eq(58).all()
        or not panel.groupby(ID_COLUMN)["target"].nunique().eq(len(TARGET_COLUMNS)).all()
    ):
        raise ValueError("Gold58 panel coverage differs from 58 studies x 12 targets")
    if panel["report_group"].isna().any():
        raise ValueError("Gold58 panel contains null report_group values")
    if panel.groupby(ID_COLUMN)["report_group"].nunique().ne(1).any():
        raise ValueError("A Gold58 study maps to more than one report_group")
    gold = pd.to_numeric(panel["gold"], errors="coerce")
    baseline = pd.to_numeric(panel["baseline_consensus_score"], errors="coerce")
    numeric_columns = [
        "llm_v4_blend",
        "llm_abs_disagreement",
        "public_image_oof_mean4",
        "dino_oof",
        "rad_family_probability",
    ]
    numeric = panel[numeric_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if (
        gold.isna().any()
        or not np.isin(gold.to_numpy(float), [0.0, 1.0]).all()
        or baseline.isna().any()
        or not np.isfinite(numeric).all()
        or ((baseline < 0.0) | (baseline > 1.0)).any()
        or ((numeric < 0.0) | (numeric > 1.0)).any()
    ):
        raise ValueError("Gold58 outcomes or probability features are invalid")
    expected_hard = baseline.ge(0.5).astype(int)
    observed_hard = pd.to_numeric(panel["pseudo_hard_label"], errors="coerce")
    if not np.array_equal(observed_hard.to_numpy(float), expected_hard.to_numpy(float)):
        raise ValueError("pseudo_hard_label does not equal baseline_consensus_score >= 0.5")
    expected_correct = expected_hard.eq(gold.astype(int))
    observed_correct = _as_bool(panel["pseudo_hard_correct"], "pseudo_hard_correct")
    if not np.array_equal(observed_correct.to_numpy(bool), expected_correct.to_numpy(bool)):
        raise ValueError("pseudo_hard_correct is inconsistent with baseline and Gold")


def add_crossfit_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Materialize the exact pre-frozen features used by all fitted models."""

    result = panel.copy()
    required = {
        *KEYS,
        "gold",
        "report_group",
        "baseline_consensus_score",
        "llm_v4_blend",
        "llm_abs_disagreement",
        "rubric_report_stance",
        "public_image_oof_mean4",
        "dino_oof",
        "rad_family_probability",
        "pseudo_hard_label",
        "pseudo_hard_correct",
    }
    missing = sorted(required.difference(result.columns))
    if missing:
        raise ValueError(f"Cross-fit panel missing columns: {missing}")

    stance_available = result["rubric_report_stance"].notna()
    stance_signed = result["rubric_report_stance"].fillna(0.5).astype(float) * 2.0 - 1.0
    stance_signed = stance_signed.where(stance_available, 0.0)
    image = result["public_image_oof_mean4"].astype(float)
    pseudo = result["baseline_consensus_score"].astype(float)
    image_signed = image * 2.0 - 1.0
    pseudo_signed = pseudo * 2.0 - 1.0

    result["feature__image_logit"] = _logit(image)
    result["feature__baseline_logit"] = _logit(pseudo)
    result["feature__baseline_margin"] = pseudo_signed.abs()
    result["feature__rubric_stance_signed"] = stance_signed.astype(float)
    result["feature__rubric_available"] = stance_available.astype(float)
    result["feature__llm_v4_blend"] = result["llm_v4_blend"].astype(float)
    result["feature__llm_abs_disagreement"] = result[
        "llm_abs_disagreement"
    ].astype(float)
    result["feature__image_probability"] = image
    result["feature__image_margin"] = image_signed.abs()
    result["feature__dino_rad_gap"] = (
        result["dino_oof"].astype(float)
        - result["rad_family_probability"].astype(float)
    ).abs()
    result["feature__report_image_alignment"] = (
        stance_signed * image_signed
    ).where(stance_available, 0.0)
    result["feature__baseline_image_alignment"] = pseudo_signed * image_signed
    result["feature__report_baseline_alignment"] = (
        stance_signed * pseudo_signed
    ).where(stance_available, 0.0)

    for target in TARGET_COLUMNS:
        result[f"feature__target_image_logit__{target}"] = np.where(
            result["target"].eq(target), result["feature__image_logit"], 0.0
        )
        result[f"feature__target_baseline_logit__{target}"] = np.where(
            result["target"].eq(target), result["feature__baseline_logit"], 0.0
        )
    return result


IMAGE_CALIBRATION_NUMERIC = [
    "feature__image_logit",
    *[f"feature__target_image_logit__{target}" for target in TARGET_COLUMNS],
]
IMAGE_PLUS_REPORT_NUMERIC = [
    *IMAGE_CALIBRATION_NUMERIC,
    "feature__baseline_logit",
    *[f"feature__target_baseline_logit__{target}" for target in TARGET_COLUMNS],
    "feature__rubric_stance_signed",
    "feature__rubric_available",
    "feature__llm_v4_blend",
    "feature__llm_abs_disagreement",
    "feature__report_image_alignment",
]
RELIABILITY_NUMERIC = [
    "feature__baseline_margin",
    "feature__rubric_stance_signed",
    "feature__rubric_available",
    "feature__llm_v4_blend",
    "feature__llm_abs_disagreement",
    "feature__image_probability",
    "feature__image_margin",
    "feature__dino_rad_gap",
    "feature__report_image_alignment",
    "feature__baseline_image_alignment",
    "feature__report_baseline_alignment",
]


def _model(numeric_columns: Sequence[str], seed: int) -> Pipeline:
    transformer = ColumnTransformer(
        [
            ("target", OneHotEncoder(handle_unknown="ignore"), ["target"]),
            ("numeric", StandardScaler(), list(numeric_columns)),
        ],
        remainder="drop",
    )
    classifier = LogisticRegression(
        C=1.0,
        solver="liblinear",
        max_iter=2000,
        random_state=int(seed),
    )
    return Pipeline([("features", transformer), ("model", classifier)])


def _predict_positive(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    """Return P(class=1) without NumPy 2.2's spurious small-matrix matmul warnings."""

    transformed = model.named_steps["features"].transform(frame)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    design = np.asarray(transformed, dtype=float)
    classifier = model.named_steps["model"]
    classes = classifier.classes_.tolist()
    if classes != [0, 1]:
        raise ValueError(f"Expected binary classes [0, 1], received {classes}")
    decision = np.einsum("ij,j->i", design, classifier.coef_[0]) + float(
        classifier.intercept_[0]
    )
    probability = np.empty_like(decision, dtype=float)
    nonnegative = decision >= 0
    probability[nonnegative] = 1.0 / (1.0 + np.exp(-decision[nonnegative]))
    exp_value = np.exp(decision[~nonnegative])
    probability[~nonnegative] = exp_value / (1.0 + exp_value)
    return probability


def validate_folds(panel: pd.DataFrame, folds: pd.DataFrame) -> None:
    """Fail closed on incomplete, duplicated, or group-splitting folds."""

    required = {ID_COLUMN, "report_group", "repeat", "seed", "fold"}
    missing = sorted(required.difference(folds.columns))
    if missing:
        raise ValueError(f"Fold manifest missing columns: {missing}")
    if folds.duplicated([ID_COLUMN, "repeat"]).any():
        raise ValueError("A study occurs more than once inside a repeat")
    repeats = sorted(folds["repeat"].unique().tolist())
    if repeats != [0, 1, 2]:
        raise ValueError(f"Expected repeats [0, 1, 2], received {repeats}")
    expected_studies = set(panel[ID_COLUMN].astype(str))
    expected_groups = (
        panel[[ID_COLUMN, "report_group"]]
        .drop_duplicates()
        .assign(**{ID_COLUMN: lambda value: value[ID_COLUMN].astype(str)})
        .set_index(ID_COLUMN)["report_group"]
        .astype(str)
        .sort_index()
    )
    for repeat, part in folds.groupby("repeat", sort=True):
        if set(part[ID_COLUMN].astype(str)) != expected_studies:
            raise ValueError(f"Repeat {repeat} does not cover the exact Gold58 studies")
        if sorted(part["fold"].unique().tolist()) != [0, 1, 2, 3]:
            raise ValueError(f"Repeat {repeat} does not contain four folds")
        if part.groupby("report_group")["fold"].nunique().gt(1).any():
            raise ValueError(f"Repeat {repeat} splits a report_group")
        observed_groups = (
            part.assign(**{ID_COLUMN: part[ID_COLUMN].astype(str)})
            .set_index(ID_COLUMN)["report_group"]
            .astype(str)
            .sort_index()
        )
        if not observed_groups.equals(expected_groups):
            raise ValueError(f"Repeat {repeat} report_group values differ from the panel")


def crossfit_gold58(panel: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    """Fit three repeated four-fold Gold classifiers and reliability models."""

    validate_gold58_panel(panel)
    features = add_crossfit_features(panel)
    folds = folds.copy()
    folds[ID_COLUMN] = folds[ID_COLUMN].astype(str)
    validate_folds(features, folds)
    base = features.merge(
        folds[[ID_COLUMN, "repeat", "seed", "fold"]],
        on=ID_COLUMN,
        how="left",
        validate="many_to_many",
    )
    if len(base) != len(features) * 3:
        raise ValueError("Cross-fit expansion must have exactly three rows per cell")

    outputs: list[pd.DataFrame] = []
    for (repeat, fold), heldout in base.groupby(["repeat", "fold"], sort=True):
        train = base[(base["repeat"].eq(repeat)) & (~base["fold"].eq(fold))]
        if set(train[ID_COLUMN]) & set(heldout[ID_COLUMN]):
            raise ValueError("Study leakage between train and heldout fold")
        if set(train["report_group"]) & set(heldout["report_group"]):
            raise ValueError("Report-group leakage between train and heldout fold")
        seed = int(heldout["seed"].iloc[0]) + int(fold)

        image_model = _model(IMAGE_CALIBRATION_NUMERIC, seed)
        image_model.fit(train, train["gold"].astype(int))
        image_prediction = _predict_positive(image_model, heldout)

        fusion_model = _model(IMAGE_PLUS_REPORT_NUMERIC, seed)
        fusion_model.fit(train, train["gold"].astype(int))
        fusion_prediction = _predict_positive(fusion_model, heldout)

        reliability_model = _model(RELIABILITY_NUMERIC, seed)
        reliability_model.fit(train, train["pseudo_hard_correct"].astype(int))
        reliability_prediction = _predict_positive(reliability_model, heldout)

        output = heldout[
            [
                *KEYS,
                "report_group",
                "repeat",
                "fold",
                "seed",
                "gold",
                "pseudo_hard_label",
                "pseudo_hard_correct",
                "baseline_consensus_score",
                "feature__baseline_margin",
                "rubric_report_stance",
                "public_image_oof_mean4",
            ]
        ].copy()
        output["primary_image_raw"] = heldout["public_image_oof_mean4"].astype(float)
        output["crossfit_image_calibration"] = image_prediction
        output["crossfit_image_plus_report"] = fusion_prediction
        output[RELIABILITY_PREDICTION_COLUMN] = reliability_prediction
        outputs.append(output)

    result = pd.concat(outputs, ignore_index=True)
    if len(result) != len(panel) * 3 or result.duplicated([*KEYS, "repeat"]).any():
        raise ValueError("Cross-fit prediction coverage is not exact")
    return result.sort_values(["repeat", *KEYS], kind="mergesort").reset_index(drop=True)


def _macro_auc_ap(
    frame: pd.DataFrame, outcome_column: str, score_column: str
) -> dict[str, Any]:
    per_target: dict[str, Any] = {}
    for target, part in frame.groupby("target", sort=True):
        y = part[outcome_column].astype(int).to_numpy()
        score = part[score_column].astype(float).to_numpy()
        if len(np.unique(y)) < 2:
            per_target[str(target)] = {
                "cells": int(len(part)),
                "positives": int(y.sum()),
                "roc_auc": None,
                "average_precision": None,
            }
            continue
        per_target[str(target)] = {
            "cells": int(len(part)),
            "positives": int(y.sum()),
            "roc_auc": float(roc_auc_score(y, score)),
            "average_precision": float(average_precision_score(y, score)),
        }
    valid = [value for value in per_target.values() if value["roc_auc"] is not None]
    return {
        "macro_roc_auc": float(np.mean([value["roc_auc"] for value in valid])),
        "macro_average_precision": float(
            np.mean([value["average_precision"] for value in valid])
        ),
        "pooled_roc_auc": float(
            roc_auc_score(frame[outcome_column].astype(int), frame[score_column])
        ),
        "pooled_average_precision": float(
            average_precision_score(frame[outcome_column].astype(int), frame[score_column])
        ),
        "per_target": per_target,
    }


def _average_repeats(predictions: pd.DataFrame) -> pd.DataFrame:
    invariant = [
        "report_group",
        "gold",
        "pseudo_hard_label",
        "pseudo_hard_correct",
        "baseline_consensus_score",
        "feature__baseline_margin",
        "rubric_report_stance",
        "public_image_oof_mean4",
    ]
    for column in invariant:
        if predictions.groupby(KEYS, sort=False)[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Invariant column changes across repeats: {column}")
    repeated_scores = [
        "crossfit_image_calibration",
        "crossfit_image_plus_report",
        RELIABILITY_PREDICTION_COLUMN,
    ]
    fixed = predictions.groupby(KEYS, as_index=False, sort=True)[invariant].first()
    averaged = predictions.groupby(KEYS, as_index=False, sort=True)[repeated_scores].mean()
    result = fixed.merge(averaged, on=KEYS, validate="one_to_one")
    result["primary_image_raw"] = result["public_image_oof_mean4"].astype(float)
    return result


def selective_risk(
    frame: pd.DataFrame,
    reliability_column: str,
    coverages: Iterable[float] = (0.25, 0.50, 0.70, 0.80, 0.90, 1.00),
) -> dict[str, Any]:
    """Measure hard pseudo-label error after retaining the most reliable cells."""

    output: dict[str, Any] = {}
    for coverage in coverages:
        retained_parts: list[pd.DataFrame] = []
        per_target: dict[str, Any] = {}
        for target, part in frame.groupby("target", sort=True):
            ordered = part.sort_values(
                [reliability_column, ID_COLUMN],
                ascending=[False, True],
                kind="mergesort",
            )
            keep = max(1, int(np.ceil(len(ordered) * float(coverage))))
            retained = ordered.head(keep)
            retained_parts.append(retained)
            per_target[str(target)] = {
                "retained_cells": int(len(retained)),
                "error_rate": float((~retained["pseudo_hard_correct"].astype(bool)).mean()),
            }
        retained_all = pd.concat(retained_parts, ignore_index=True)
        output[f"{coverage:.2f}"] = {
            "retained_cells": int(len(retained_all)),
            "target_balanced_error_rate": float(
                np.mean([value["error_rate"] for value in per_target.values()])
            ),
            "pooled_error_rate": float(
                (~retained_all["pseudo_hard_correct"].astype(bool)).mean()
            ),
            "per_target": per_target,
        }
    return output


def _fusion_metric_values(frame: pd.DataFrame, score_column: str) -> dict[str, float]:
    metrics = score_metrics(frame, score_column)
    return {
        "macro_roc_auc": float(metrics["macro_roc_auc"]),
        "macro_average_precision": float(metrics["macro_average_precision"]),
        "brier": float(metrics["brier"]),
        "log_loss": float(metrics["log_loss"]),
    }


def _bootstrap_macro_auc(
    frame: pd.DataFrame, outcome_column: str, score_column: str
) -> float:
    values: list[float] = []
    for _, part in frame.groupby("target", sort=True):
        y = part[outcome_column].astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            return float("nan")
        values.append(float(roc_auc_score(y, part[score_column].astype(float))))
    if len(values) != len(TARGET_COLUMNS):
        return float("nan")
    return float(np.mean(values))


def study_bootstrap_deltas(
    frame: pd.DataFrame,
    comparisons: Sequence[tuple[str, str, str, str]],
    iterations: int = 5000,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired study bootstrap CIs for predeclared metric differences."""

    studies = np.asarray(sorted(frame[ID_COLUMN].astype(str).unique()))
    by_study = {study: frame[frame[ID_COLUMN].astype(str).eq(study)] for study in studies}
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {name: [] for name, *_ in comparisons}
    accepted = 0
    attempted = 0
    discarded = 0
    maximum_attempts = max(int(iterations) * 20, int(iterations) + 100)
    while accepted < int(iterations):
        attempted += 1
        if attempted > maximum_attempts:
            raise RuntimeError(
                "Unable to obtain the requested number of all-target bootstrap draws"
            )
        drawn = rng.choice(studies, size=len(studies), replace=True)
        bootstrap = pd.concat(
            [by_study[study].assign(_bootstrap_copy=index) for index, study in enumerate(drawn)],
            ignore_index=True,
        )
        deltas: dict[str, float] = {}
        for name, outcome, left, right in comparisons:
            left_value = _bootstrap_macro_auc(bootstrap, outcome, left)
            right_value = _bootstrap_macro_auc(bootstrap, outcome, right)
            deltas[name] = left_value - right_value
        if not all(np.isfinite(value) for value in deltas.values()):
            discarded += 1
            continue
        for name, value in deltas.items():
            samples[name].append(float(value))
        accepted += 1

    output: dict[str, Any] = {}
    for name, outcome, left, right in comparisons:
        observed = _bootstrap_macro_auc(frame, outcome, left) - _bootstrap_macro_auc(
            frame, outcome, right
        )
        values = np.asarray(samples[name], dtype=float)
        output[name] = {
            "outcome": outcome,
            "metric": "target_macro_roc_auc",
            "left": left,
            "right": right,
            "observed_delta": float(observed),
            "ci95_percentile": [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ],
            "valid_iterations": int(len(values)),
        }
    return {
        "requested_valid_iterations": int(iterations),
        "valid_iterations": int(accepted),
        "attempted_draws": int(attempted),
        "discarded_degenerate_draws": int(discarded),
        "degenerate_draw_policy": "discard whole paired draw and redraw until all 12 targets are evaluable",
        "comparisons": output,
    }


def summarize_crossfit(
    predictions: pd.DataFrame, bootstrap_iterations: int = 5000
) -> dict[str, Any]:
    """Summarize repeat-level and repeat-averaged cross-fit predictions."""

    averaged = _average_repeats(predictions)
    averaged["baseline_consensus_reference"] = averaged[
        "baseline_consensus_score"
    ].astype(float)
    averaged["baseline_margin_reference"] = averaged[
        "feature__baseline_margin"
    ].astype(float)

    fusion_columns = ["baseline_consensus_reference", *FUSION_PREDICTION_COLUMNS]
    fusion = {
        column: score_metrics(averaged, column) for column in fusion_columns
    }
    repeat_metrics: dict[str, Any] = {}
    reliability_repeat_metrics: dict[str, Any] = {}
    for repeat, part in predictions.groupby("repeat", sort=True):
        repeat_metrics[str(int(repeat))] = {
            column: _fusion_metric_values(part, column)
            for column in FUSION_PREDICTION_COLUMNS
        }
        reliability_part = part.copy()
        reliability_part["pseudo_hard_error"] = (
            ~reliability_part["pseudo_hard_correct"].astype(bool)
        ).astype(int)
        reliability_part["crossfit_probability_pseudo_hard_error"] = (
            1.0 - reliability_part[RELIABILITY_PREDICTION_COLUMN]
        )
        reliability_repeat_metrics[str(int(repeat))] = {
            "predicting_correct": _macro_auc_ap(
                reliability_part,
                "pseudo_hard_correct",
                RELIABILITY_PREDICTION_COLUMN,
            ),
            "predicting_error": _macro_auc_ap(
                reliability_part,
                "pseudo_hard_error",
                "crossfit_probability_pseudo_hard_error",
            ),
            "selective_risk": selective_risk(
                reliability_part, RELIABILITY_PREDICTION_COLUMN
            ),
        }

    reliability = {
        "crossfit_model_predicting_correct": _macro_auc_ap(
            averaged, "pseudo_hard_correct", RELIABILITY_PREDICTION_COLUMN
        ),
        "baseline_margin_predicting_correct": _macro_auc_ap(
            averaged, "pseudo_hard_correct", "baseline_margin_reference"
        ),
        "crossfit_model_predicting_error": _macro_auc_ap(
            averaged.assign(
                pseudo_hard_error=(~averaged["pseudo_hard_correct"].astype(bool)).astype(int),
                crossfit_probability_pseudo_hard_error=1.0
                - averaged[RELIABILITY_PREDICTION_COLUMN],
            ),
            "pseudo_hard_error",
            "crossfit_probability_pseudo_hard_error",
        ),
        "selective_risk_crossfit": selective_risk(
            averaged, RELIABILITY_PREDICTION_COLUMN
        ),
        "selective_risk_baseline_margin": selective_risk(
            averaged, "baseline_margin_reference"
        ),
        "by_pseudo_polarity": {},
    }
    for polarity, part in averaged.groupby("pseudo_hard_label", sort=True):
        outcome = part["pseudo_hard_correct"].astype(int)
        if outcome.nunique() < 2:
            reliability["by_pseudo_polarity"][str(int(polarity))] = {
                "cells": int(len(part)),
                "errors": int((~part["pseudo_hard_correct"].astype(bool)).sum()),
                "roc_auc": None,
                "average_precision": None,
            }
        else:
            reliability["by_pseudo_polarity"][str(int(polarity))] = {
                "cells": int(len(part)),
                "errors": int((~part["pseudo_hard_correct"].astype(bool)).sum()),
                "roc_auc": float(
                    roc_auc_score(outcome, part[RELIABILITY_PREDICTION_COLUMN])
                ),
                "average_precision": float(
                    average_precision_score(
                        outcome, part[RELIABILITY_PREDICTION_COLUMN]
                    )
                ),
            }

    bootstrap_frame = averaged.copy()
    bootstrap_frame["pseudo_hard_error"] = (
        ~bootstrap_frame["pseudo_hard_correct"].astype(bool)
    ).astype(int)
    bootstrap_frame["crossfit_probability_pseudo_hard_error"] = (
        1.0 - bootstrap_frame[RELIABILITY_PREDICTION_COLUMN]
    )
    bootstrap_frame["baseline_margin_error_reference"] = (
        1.0 - bootstrap_frame["baseline_margin_reference"]
    )
    bootstrap = study_bootstrap_deltas(
        bootstrap_frame,
        comparisons=[
            (
                "image_plus_report_minus_primary_image",
                "gold",
                "crossfit_image_plus_report",
                "primary_image_raw",
            ),
            (
                "image_plus_report_minus_baseline_consensus",
                "gold",
                "crossfit_image_plus_report",
                "baseline_consensus_reference",
            ),
            (
                "image_calibration_minus_primary_image",
                "gold",
                "crossfit_image_calibration",
                "primary_image_raw",
            ),
            (
                "reliability_model_minus_baseline_margin_for_error",
                "pseudo_hard_error",
                "crossfit_probability_pseudo_hard_error",
                "baseline_margin_error_reference",
            ),
        ],
        iterations=bootstrap_iterations,
    )
    return {
        "cells": int(len(averaged)),
        "studies": int(averaged[ID_COLUMN].nunique()),
        "repeats": int(predictions["repeat"].nunique()),
        "folds_per_repeat": int(predictions["fold"].nunique()),
        "pseudo_hard_errors": int((~averaged["pseudo_hard_correct"].astype(bool)).sum()),
        "fusion_repeat_averaged": fusion,
        "fusion_by_repeat": repeat_metrics,
        "reliability": reliability,
        "reliability_by_repeat": reliability_repeat_metrics,
        "study_bootstrap_deltas": bootstrap,
        "bootstrap_iterations": int(bootstrap_iterations),
        "boundaries": {
            "split_unit": "report_group with all 12 study-target cells together",
            "public_image_inputs_are_source_oof_but_not_nested_to_this_outer_crossfit": True,
            "gold58_is_development_not_untouched_validation": True,
            "claim_strength": "descriptive/directional only",
        },
    }
