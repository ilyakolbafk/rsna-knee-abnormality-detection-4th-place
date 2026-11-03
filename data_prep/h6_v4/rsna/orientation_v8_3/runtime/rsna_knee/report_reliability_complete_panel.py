"""Outcome-safe completion helpers for the reliability-v3 cell panel.

The primary proxy model is defined only for decisive teacher states.  The
eligible proxy cohort nevertheless contains a small number of teacher-abstain
cells.  This module gives those cells the same fold-local image-evidence
construction as decisive rows while keeping every report-risk field null.

No Gold outcome is read or accepted here.  The decisive rows are used only as
the training reference for each held-out outer fold, exactly matching the
primary proxy image-ECDF construction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .constants import ID_COLUMN
from .report_reliability_gold_gate import EXTERNAL_REQUIRED_COLUMNS
from .report_reliability_panel import build_proxy_presentation_fields
from .report_reliability_stats import fit_family_balanced_image_reference


DINO_COLUMN = "dino_public20_equal_oof"
RAD_COMPONENT_COLUMNS: tuple[str, str, str] = (
    "radimagenet_oof",
    "radimagenet_v52_oof",
    "radimagenet_e11_public_oof",
)
OUTER_FOLD_COLUMN = "morph_outer_fold"
REPORT_GROUP_COLUMN = "report_group"
IMAGE_EVIDENCE_COLUMNS: tuple[str, ...] = (
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
)


def _require(frame: pd.DataFrame, columns: Sequence[str], *, name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def _assert_no_gold_columns(frame: pd.DataFrame, *, name: str) -> None:
    forbidden = [column for column in frame.columns if "gold" in str(column).casefold()]
    if forbidden:
        raise ValueError(f"{name} contains Gold-bearing columns: {forbidden}")


def _validated_teacher_labels(frame: pd.DataFrame) -> pd.Series:
    original = frame["teacher_hard_label"]
    numeric = pd.to_numeric(original, errors="coerce")
    if (original.notna() & numeric.isna()).any() or not bool(
        numeric.dropna().isin([0.0, 1.0]).all()
    ):
        raise ValueError("teacher_hard_label must be exact 0/1/null")
    return numeric


def crossfit_image_evidence_for_all_proxy_cells(
    cells: pd.DataFrame,
    *,
    fold_column: str = OUTER_FOLD_COLUMN,
    dino_column: str = DINO_COLUMN,
    rad_component_columns: Sequence[str] = RAD_COMPONENT_COLUMNS,
) -> pd.DataFrame:
    """Score every proxy cell from decisive rows outside its outer fold.

    The ECDF reference for one held-out fold contains only teacher-decisive
    rows from the other folds.  Consequently the decisive subset exactly
    reproduces the primary OOF image evidence, while abstain rows never enter a
    fit or reference distribution.
    """

    rad_columns = tuple(str(column) for column in rad_component_columns)
    if len(rad_columns) != 3 or len(set(rad_columns)) != 3:
        raise ValueError("rad_component_columns must contain exactly three columns")
    required = (
        ID_COLUMN,
        REPORT_GROUP_COLUMN,
        "target",
        "teacher_hard_label",
        fold_column,
        dino_column,
        *rad_columns,
    )
    _require(cells, required, name="proxy cells")
    _assert_no_gold_columns(cells, name="proxy cells")
    if cells.duplicated([ID_COLUMN, "target"]).any():
        raise ValueError("proxy cell keys are duplicated")
    if not cells.index.is_unique:
        raise ValueError("proxy cells require a unique row index")

    labels = _validated_teacher_labels(cells)
    folds_numeric = pd.to_numeric(cells[fold_column], errors="coerce")
    if folds_numeric.isna().any() or not np.equal(
        folds_numeric.to_numpy(float), np.floor(folds_numeric.to_numpy(float))
    ).all():
        raise ValueError(f"{fold_column} must contain integer ids")
    folds = folds_numeric.astype(np.int64)
    unique_folds = sorted(folds.unique().tolist())
    if unique_folds != [0, 1, 2, 3, 4]:
        raise ValueError("proxy outer-fold identity must be exactly 0..4")
    group_fold_count = pd.DataFrame(
        {
            REPORT_GROUP_COLUMN: cells[REPORT_GROUP_COLUMN].astype(str),
            "fold": folds,
        }
    ).groupby(REPORT_GROUP_COLUMN, sort=False)["fold"].nunique()
    if group_fold_count.gt(1).any():
        raise ValueError("one report_group is split across proxy outer folds")

    parts: list[pd.DataFrame] = []
    for fold in unique_folds:
        training = labels.notna() & folds.ne(fold)
        heldout = folds.eq(fold)
        reference = fit_family_balanced_image_reference(
            cells.loc[training],
            dino_column=dino_column,
            rad_component_columns=rad_columns,
            target_column="target",
            disagreement_gate=0.10,
        )
        transformed = reference.transform(cells.loc[heldout])
        transformed["__source_position"] = np.flatnonzero(heldout.to_numpy())
        parts.append(transformed)

    result = (
        pd.concat(parts, axis=0)
        .sort_values("__source_position", kind="mergesort")
        .drop(columns="__source_position")
    )
    result = result.reindex(cells.index)
    if not result.index.equals(cells.index) or len(result) != len(cells):
        raise AssertionError("cross-fitted image evidence lost proxy rows")
    if not result["image_oof_valid"].all():
        raise ValueError("eligible proxy image evidence contains invalid rows")
    return result.loc[:, list(IMAGE_EVIDENCE_COLUMNS)]


def assert_decisive_image_evidence_matches_frozen_oof(
    cells: pd.DataFrame,
    evidence: pd.DataFrame,
    frozen_oof: pd.DataFrame,
) -> None:
    """Require exact decisive-row parity with the already frozen OOF panel."""

    _assert_no_gold_columns(cells, name="proxy cells")
    _assert_no_gold_columns(frozen_oof, name="frozen proxy OOF")
    _require(cells, (ID_COLUMN, "target", "teacher_hard_label"), name="proxy cells")
    _require(
        frozen_oof,
        (ID_COLUMN, "target", *IMAGE_EVIDENCE_COLUMNS),
        name="frozen proxy OOF",
    )
    if not evidence.index.equals(cells.index):
        raise ValueError("image evidence index differs from proxy cells")
    decisive = _validated_teacher_labels(cells).notna()
    observed = cells.loc[decisive, [ID_COLUMN, "target"]].copy()
    for column in IMAGE_EVIDENCE_COLUMNS:
        observed[column] = evidence.loc[decisive, column].to_numpy()
    expected = frozen_oof.loc[:, [ID_COLUMN, "target", *IMAGE_EVIDENCE_COLUMNS]].copy()
    observed = observed.sort_values([ID_COLUMN, "target"], kind="mergesort").reset_index(drop=True)
    expected = expected.sort_values([ID_COLUMN, "target"], kind="mergesort").reset_index(drop=True)
    if not observed[[ID_COLUMN, "target"]].equals(expected[[ID_COLUMN, "target"]]):
        raise AssertionError("frozen decisive image-evidence keys differ")
    for column in ("image_oof_valid", "dino_oof_valid", "rad_oof_valid", "image_high_agreement", "image_state"):
        if not observed[column].astype(str).equals(expected[column].astype(str)):
            raise AssertionError(f"frozen decisive image evidence differs in {column}")
    for column in (
        "dino_target_ecdf",
        "rad_probability",
        "rad_target_ecdf",
        "image_consensus_rank",
        "image_family_disagreement",
        "rad_internal_dispersion",
    ):
        left = pd.to_numeric(observed[column], errors="coerce").to_numpy(float)
        right = pd.to_numeric(expected[column], errors="coerce").to_numpy(float)
        if not np.array_equal(np.isnan(left), np.isnan(right)) or not np.allclose(
            left,
            right,
            rtol=0.0,
            atol=np.finfo(np.float64).eps,
            equal_nan=True,
        ):
            raise AssertionError(
                f"frozen decisive image evidence differs beyond float64 CSV round-trip in {column}"
            )


def build_proxy_abstain_presentation(
    cells: pd.DataFrame,
    evidence: pd.DataFrame,
    *,
    model_status_by_direction: Mapping[tuple[str, str], str],
) -> pd.DataFrame:
    """Create the exact canonical rows for teacher-abstain proxy cells."""

    required = (
        ID_COLUMN,
        "target",
        "teacher_state",
        "teacher_hard_label",
        "family_state_summary",
        "other_positive_count",
        "other_negative_count",
        "other_abstain_count",
        "external_audit_disagreement",
        "teacher_component_self_conflict",
        "family_silence_instability",
    )
    _require(cells, required, name="proxy cells")
    _assert_no_gold_columns(cells, name="proxy cells")
    if not evidence.index.equals(cells.index):
        raise ValueError("image evidence index differs from proxy cells")

    abstain = _validated_teacher_labels(cells).isna()
    if not cells.loc[abstain, "teacher_state"].eq("ABSTAIN").all():
        raise ValueError("null teacher labels differ from ABSTAIN state")
    frame = cells.loc[abstain, list(required)].drop(columns="teacher_hard_label").copy()
    for column in IMAGE_EVIDENCE_COLUMNS:
        frame[column] = evidence.loc[abstain, column].to_numpy()
    frame["direction"] = "not_applicable"
    frame["report_risk_percentile_proxy_reference"] = np.nan
    frame["guardrailed_review_risk_percentile"] = np.nan
    result = build_proxy_presentation_fields(
        frame,
        model_status_by_direction=model_status_by_direction,
    ).loc[:, list(EXTERNAL_REQUIRED_COLUMNS)]
    if len(result) != int(abstain.sum()):
        raise AssertionError("proxy abstain presentation row count drift")
    if not result["teacher_state"].eq("ABSTAIN").all():
        raise AssertionError("proxy abstain presentation contains decisive rows")
    if result["label_review_risk_percentile"].notna().any():
        raise AssertionError("teacher abstains received report-risk values")
    if not result["candidate_reason_codes"].str.split("|").map(
        lambda codes: "TEACHER_ABSTAIN" in codes
    ).all():
        raise AssertionError("teacher abstain reason code is missing")
    return result.sort_values([ID_COLUMN, "target"], kind="mergesort").reset_index(drop=True)


__all__ = [
    "DINO_COLUMN",
    "IMAGE_EVIDENCE_COLUMNS",
    "OUTER_FOLD_COLUMN",
    "REPORT_GROUP_COLUMN",
    "RAD_COMPONENT_COLUMNS",
    "assert_decisive_image_evidence_matches_frozen_oof",
    "build_proxy_abstain_presentation",
    "crossfit_image_evidence_for_all_proxy_cells",
]
