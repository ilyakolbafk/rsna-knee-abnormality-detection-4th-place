"""Fail-closed, pre-Gold input assembly for report-reliability analysis.

This module binds the frozen v3.4.0 source identities and deliberately reads
only explicit column allowlists.  Gold *identity* is read from the canonical
manifest solely to construct the non-Gold proxy cohort; Gold values and every
Gold-derived score-panel column are never loaded or returned.

No transform is fit here.  The returned frames are already restricted to the
frozen fold-map allowlist, so downstream fold-fitted transforms cannot
accidentally see the 35 excluded non-Gold studies or Gold58.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .constants import ID_COLUMN, REPORT_COLUMN, TARGET_COLUMNS


REPORT_GROUP_COLUMN = "report_group"

MANIFEST_COLUMNS: tuple[str, ...] = (
    ID_COLUMN,
    REPORT_COLUMN,
    REPORT_GROUP_COLUMN,
    "is_gold",
)

SCORE_COLUMNS: tuple[str, ...] = (
    ID_COLUMN,
    "target",
    "dino_public20_equal_oof",
    "dino_public20_fold",
    "radimagenet_oof",
    "radimagenet_oof_fold",
    "radimagenet_v52_oof",
    "radimagenet_v52_oof_fold",
    "radimagenet_e11_public_oof",
    "radimagenet_e11_public_oof_fold",
    "directional_v45_final_rank",
    "directional_v45_prediction_scope",
)

SCORE_VALUE_COLUMNS: tuple[str, ...] = (
    "dino_public20_equal_oof",
    "radimagenet_oof",
    "radimagenet_v52_oof",
    "radimagenet_e11_public_oof",
    "directional_v45_final_rank",
)

SOURCE_FOLD_COLUMNS: tuple[str, ...] = (
    "dino_public20_fold",
    "radimagenet_oof_fold",
    "radimagenet_v52_oof_fold",
    "radimagenet_e11_public_oof_fold",
)

RAD_FOLD_COLUMNS: tuple[str, ...] = SOURCE_FOLD_COLUMNS[1:]

FOLD_MAP_COLUMNS: tuple[str, ...] = (
    ID_COLUMN,
    REPORT_GROUP_COLUMN,
    "outer_fold",
    "inner_fold_when_outer_0",
    "inner_fold_when_outer_1",
    "inner_fold_when_outer_2",
    "inner_fold_when_outer_3",
    "inner_fold_when_outer_4",
)

MORPH_FOLD_RENAME: Mapping[str, str] = {
    "outer_fold": "morph_outer_fold",
    **{
        f"inner_fold_when_outer_{outer}": f"morph_inner_fold_when_outer_{outer}"
        for outer in range(5)
    },
}

# The complete outcome-blind feature-panel schema.  ``is_gold`` is used only to
# verify canonical identity parity and is removed before returning ``cells``.
FEATURE_PANEL_COLUMNS: tuple[str, ...] = (
    ID_COLUMN,
    "target",
    "teacher_score",
    "teacher_confidence",
    "teacher_count",
    "teacher_agreement",
    "teacher_certainty",
    "teacher_evidence_gate",
    "teacher_state",
    "family_direct_positive_count",
    "family_direct_negative_count",
    "family_uncertain_count",
    "family_unmentioned_count",
    "family_nondirect_or_unknown_count",
    "family_missing_count",
    "family_available_count",
    "family_decisive_direct_count",
    "family_abstain_count",
    "family_direct_conflict",
    "family_silence_instability",
    "family_direct_majority_margin",
    "family_direct_majority_fraction",
    "family_direct_consensus_eligible",
    "family_direct_majority_state",
    "family_direct_consensus_state",
    "family_state_summary",
    "family_unique_nonmissing_state_count",
    "current_canonical_state_diversity",
    "teacher_component_positive_count",
    "teacher_component_negative_count",
    "teacher_component_abstain_count",
    "teacher_component_missing_count",
    "teacher_component_self_conflict",
    "external_audit_consensus_state",
    "external_audit_direct_conflict",
    "external_audit_direct_support_count",
    "teacher_family_support_count",
    "teacher_family_opposition_count",
    "teacher_family_support_fraction_decisive",
    "teacher_consensus_aggregation_conflict",
    "teacher_family_comparison_state",
    "external_audit_disagreement",
    "other_positive_count",
    "other_negative_count",
    "other_abstain_count",
    "other_decisive_count",
    "other_positive_fraction_decisive",
    "other_decisive_zero_denominator",
    *(f"loto_teacher_state__{target}" for target in TARGET_COLUMNS),
    "family_other_positive_count_mean",
    "family_other_positive_count_std",
    "other_canonical_state_diversity_count",
    "other_direct_conflict_count",
    "other_silence_instability_count",
    REPORT_GROUP_COLUMN,
    "is_gold",
    "dominant_script",
    "script_type_count",
    "mixed_script_token_count",
    "mixed_script_token_rate",
    "report_char_count",
    "report_normalized_char_count",
    "report_word_count",
    "report_lexical_token_count",
    "report_line_count",
    "report_nonempty_line_count",
    "report_sentence_count",
    "report_paragraph_count",
    "report_heading_count",
    "report_section_count",
    "report_placeholder_count",
    "report_avg_token_length",
    "report_max_token_length",
    "report_digit_ratio",
    "report_punctuation_ratio",
    "report_uppercase_ratio",
    "report_whitespace_ratio",
    "report_newline_ratio",
    "orth_unicode_replacement_count",
    "orth_mojibake_marker_count",
    "orth_control_count",
    "orth_zero_width_count",
    "orth_surrogate_count",
    "orth_repeated_character_run_count",
    "orth_max_repeated_character_run",
    "orth_repeated_punctuation_run_count",
    "orth_suspicious_long_token_count",
    "orth_suspicious_long_token_rate",
    "orth_alphanumeric_token_count",
    "orth_alphanumeric_token_rate",
    "language_proxy",
    "language_proxy_confidence",
    "any_mixed_script_token",
    "any_encoding_anomaly_proxy",
    "any_repeated_character_proxy",
    "any_repeated_punctuation_proxy",
    "any_suspicious_long_token_proxy",
    "any_alphanumeric_token_proxy",
    "orthographic_noise_proxy_index",
    "series_count",
    "fluid_sensitive_series_count",
    "fat_suppression_series_count",
    "series_plane_axial_count",
    "series_plane_coronal_count",
    "series_plane_sagittal_count",
    "acquisition_manufacturer_mode",
    "acquisition_manufacturer_distinct_count",
    "acquisition_manufacturer_conflict",
    "acquisition_manufacturer_missing",
    "acquisition_manufacturer_series_missing_count",
    "acquisition_model_mode",
    "acquisition_model_distinct_count",
    "acquisition_model_conflict",
    "acquisition_model_missing",
    "acquisition_model_series_missing_count",
    "acquisition_field_strength_mode",
    "acquisition_field_strength_distinct_count",
    "acquisition_field_strength_conflict",
    "acquisition_field_strength_missing",
    "acquisition_field_strength_series_missing_count",
)

MORPHOLOGY_COLUMNS: tuple[str, ...] = (
    ID_COLUMN,
    REPORT_GROUP_COLUMN,
    "report_group_size",
    "dominant_script",
    "script_type_count",
    "mixed_script_token_count",
    "mixed_script_token_rate",
    "report_char_count",
    "report_normalized_char_count",
    "report_word_count",
    "report_lexical_token_count",
    "report_line_count",
    "report_nonempty_line_count",
    "report_sentence_count",
    "report_paragraph_count",
    "report_heading_count",
    "report_section_count",
    "report_placeholder_count",
    "report_avg_token_length",
    "report_max_token_length",
    "report_digit_ratio",
    "report_punctuation_ratio",
    "report_uppercase_ratio",
    "report_whitespace_ratio",
    "report_newline_ratio",
    "orth_unicode_replacement_count",
    "orth_mojibake_marker_count",
    "orth_control_count",
    "orth_zero_width_count",
    "orth_surrogate_count",
    "orth_repeated_character_run_count",
    "orth_max_repeated_character_run",
    "orth_repeated_punctuation_run_count",
    "orth_suspicious_long_token_count",
    "orth_suspicious_long_token_rate",
    "orth_alphanumeric_token_count",
    "orth_alphanumeric_token_rate",
    "delexicalized_template_key",
    "delexicalized_template_group_size",
    "delexicalized_nn_char_similarity",
    "language_proxy",
    "language_proxy_confidence",
    "orth_rare_token_count",
    "orth_rare_token_rate",
    "orth_char_ngram_nonzero_features",
    "orth_char_ngram_outlier_score",
    "style_cluster_id",
    "style_cluster_size",
    "style_cluster_group_size",
    "style_cluster_distance",
    "style_cluster_margin",
    "style_cluster_stable",
    "style_cluster_association_eligible",
    "style_cluster_margin_available",
    "acquisition_profile",
    "style_cluster_segment",
    "exact_report_repeated",
    "delexicalized_template_repeated",
    "any_mixed_script_token",
    "any_encoding_anomaly_proxy",
    "any_repeated_character_proxy",
    "any_repeated_punctuation_proxy",
    "any_suspicious_long_token_proxy",
    "any_alphanumeric_token_proxy",
    "orthographic_noise_proxy_index",
)

FROZEN_DIRECTIONAL_SCOPE = (
    "directional_mixed_oof_and_train_inference_not_oof_not_performance_estimate"
)


@dataclass(frozen=True)
class BoundFile:
    """A path whose bytes must match an exact SHA-256 before being read."""

    path: str
    sha256: str


@dataclass(frozen=True)
class PreGoldAssemblySpec:
    """Frozen identities and coverage gates for one input assembly."""

    contract: BoundFile
    manifest: BoundFile
    feature_panel: BoundFile
    morphology: BoundFile
    score_panel: BoundFile
    fold_map: BoundFile
    contract_version: str
    contract_semantic_hash: str
    targets: tuple[str, ...]
    feature_columns: tuple[str, ...]
    morphology_columns: tuple[str, ...]
    expected_total_studies: int
    expected_gold_identities: int
    expected_eligible_studies: int
    expected_eligible_report_groups: int
    expected_eligible_cells: int
    expected_decisive_cells: int
    expected_abstain_cells: int
    expected_split_groups: int
    expected_split_studies: int
    expected_split_non_gold: int
    expected_split_gold: int
    split_group_sha256: str
    split_uid_sha256: str
    expected_outer_counts: tuple[int, ...]
    directional_scope: str


FROZEN_V34_SPEC = PreGoldAssemblySpec(
    contract=BoundFile(
        ".codex/runs/rsna-knee-report-reliability-v3/ANALYSIS_CONTRACT.yaml",
        "2be69a3454ed7fce1d56c302366733f2ca4aa3e77c366c372f50127c3bc09491",
    ),
    manifest=BoundFile(
        ".codex/runs/rsna-knee-full-cache-pipeline/manifests/manifest_evidence_gated.csv",
        "38f2df1518867a7d6289fcb5be5171e75249f7e7a7d64f1eacc4acb82b485a23",
    ),
    feature_panel=BoundFile(
        ".codex/runs/rsna-knee-report-reliability-v3/report_reliability_feature_panel_v3.csv.gz",
        "40b18680c7a60fc84180ce1416ef6a1499a88e4667c3d82718bdb114bacc1355",
    ),
    morphology=BoundFile(
        ".codex/runs/rsna-knee-morphology-error-risk-v1/report_morphology_features_v1.csv.gz",
        "d4b51f902b119519737d5e991211ca6ac5238d8ea7087d7490c69d23774f4742",
    ),
    score_panel=BoundFile(
        ".codex/runs/rsna-knee-full-train-inference/results/train_model_score_panel_long.csv.gz",
        "109155a7a6e07ae94d7442fe0f7f55e7e49ff9288348cce6d7a4b7aa4389b4d1",
    ),
    fold_map=BoundFile(
        ".codex/runs/rsna-knee-label-risk-research-v2/fold_map_v2.csv",
        "f115e4855e5941caf29c859a4b854eedc69cd5f87d98dfe723785585156cfb26",
    ),
    contract_version="3.4.0",
    contract_semantic_hash=(
        "f00d67756c1d840dfeb5d6738e172d03eee452f2e091c29f4a3366ff4ec30e31"
    ),
    targets=tuple(TARGET_COLUMNS),
    feature_columns=FEATURE_PANEL_COLUMNS,
    morphology_columns=MORPHOLOGY_COLUMNS,
    expected_total_studies=4_407,
    expected_gold_identities=58,
    expected_eligible_studies=4_314,
    expected_eligible_report_groups=4_189,
    expected_eligible_cells=51_768,
    expected_decisive_cells=51_353,
    expected_abstain_cells=415,
    expected_split_groups=11,
    expected_split_studies=36,
    expected_split_non_gold=35,
    expected_split_gold=1,
    split_group_sha256=(
        "0a261ac7089801ffea3e4755e586a2e8ca39f39f65379f60c6c19fff943a0875"
    ),
    split_uid_sha256=(
        "51725b7d75793b15d7e663fdfe67731178536b5eff9d63adb2c7f80bca0e49a9"
    ),
    expected_outer_counts=(847, 862, 855, 901, 849),
    directional_scope=FROZEN_DIRECTIONAL_SCOPE,
)


@dataclass(frozen=True)
class PreGoldInputs:
    """Validated outcome-blind frames ready for fold-fitted analysis."""

    cells: pd.DataFrame
    reports: pd.DataFrame
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class PreGoldExternalInputs:
    """Outcome-free rows reserved for deterministic external proxy scoring.

    The returned frames contain the 35 source-fold-excluded non-Gold studies
    plus all 58 Gold *identities*, but never expose the manifest identity flag
    or any Gold outcome.  DINO values for every source-fold-excluded report
    group are deliberately set to null so downstream image-reference code
    cannot accidentally treat them as valid OOF evidence.  Report morphology
    remains available for the contract's external morphology-only score.
    """

    cells: pd.DataFrame
    reports: pd.DataFrame
    diagnostics: Mapping[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_sha(values: Sequence[object]) -> str:
    return hashlib.sha256(
        "\n".join(str(value) for value in values).encode("utf-8")
    ).hexdigest()


def _resolve(root: Path, bound: BoundFile) -> Path:
    path = Path(bound.path)
    return path if path.is_absolute() else root / path


def _verify_sources(root: Path, spec: PreGoldAssemblySpec) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name in (
        "contract",
        "manifest",
        "feature_panel",
        "morphology",
        "score_panel",
        "fold_map",
    ):
        bound = getattr(spec, name)
        path = _resolve(root, bound)
        if not path.is_file():
            raise FileNotFoundError(f"{name}: missing bound source {path}")
        digest = sha256_file(path)
        if digest != bound.sha256:
            raise ValueError(
                f"{name}: SHA-256 mismatch; expected {bound.sha256}, got {digest}"
            )
        observed[name] = digest
    return observed


def _verify_contract(root: Path, spec: PreGoldAssemblySpec) -> None:
    path = _resolve(root, spec.contract)
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    if contract.get("version") != spec.contract_version:
        raise ValueError("contract version drift")
    if contract.get("semantic_hash") != spec.contract_semantic_hash:
        raise ValueError("contract semantic hash drift")
    payload = contract.get("semantic_hash_payload")
    if not isinstance(payload, str):
        raise ValueError("contract semantic hash payload is missing")
    observed = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if observed != spec.contract_semantic_hash:
        raise ValueError("contract semantic hash payload mismatch")


def _read_csv(path: Path, columns: Sequence[str], *, name: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, usecols=list(columns))
    except ValueError as error:
        raise ValueError(f"{name}: allowlisted schema mismatch: {error}") from error
    if tuple(frame.columns) != tuple(columns):
        # Pandas normally preserves file order rather than usecols order.  An
        # explicit reindex both freezes downstream order and rejects duplicates.
        if len(set(frame.columns)) != len(frame.columns) or set(frame.columns) != set(columns):
            raise ValueError(f"{name}: allowlisted schema mismatch")
        frame = frame.loc[:, list(columns)]
    return frame


def _normalize_key_columns(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    result = frame.copy()
    for column in (ID_COLUMN,):
        if result[column].isna().any():
            raise ValueError(f"{name}: {column} contains nulls")
        result[column] = result[column].astype(str)
        if result[column].str.strip().eq("").any():
            raise ValueError(f"{name}: {column} contains empty strings")
    if REPORT_GROUP_COLUMN in result:
        if result[REPORT_GROUP_COLUMN].isna().any():
            raise ValueError(f"{name}: report_group contains nulls")
        result[REPORT_GROUP_COLUMN] = result[REPORT_GROUP_COLUMN].astype(str)
        if result[REPORT_GROUP_COLUMN].str.strip().eq("").any():
            raise ValueError(f"{name}: report_group contains empty strings")
    if "target" in result:
        if result["target"].isna().any():
            raise ValueError(f"{name}: target contains nulls")
        result["target"] = result["target"].astype(str)
    return result


def _validate_cartesian_cells(
    frame: pd.DataFrame,
    *,
    name: str,
    expected_uids: set[str],
    targets: tuple[str, ...],
) -> None:
    if frame.duplicated([ID_COLUMN, "target"]).any():
        raise ValueError(f"{name}: duplicate StudyInstanceUID x target rows")
    if set(frame[ID_COLUMN]) != expected_uids:
        raise ValueError(f"{name}: UID set differs from canonical manifest")
    if set(frame["target"]) != set(targets):
        raise ValueError(f"{name}: target set drift")
    per_uid = frame.groupby(ID_COLUMN, sort=False)["target"].nunique()
    if not per_uid.eq(len(targets)).all():
        raise ValueError(f"{name}: every study must contain every target")
    expected_rows = len(expected_uids) * len(targets)
    if len(frame) != expected_rows:
        raise ValueError(f"{name}: expected {expected_rows} rows, got {len(frame)}")


def _as_boolean_identity(values: pd.Series) -> pd.Series:
    if values.isna().any():
        raise ValueError("manifest: is_gold contains nulls")
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.casefold()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError("manifest: is_gold must contain only true/false identities")
    return normalized.eq("true")


def _validate_score_values(scores: pd.DataFrame) -> None:
    for column in SCORE_VALUE_COLUMNS:
        numeric = pd.to_numeric(scores[column], errors="coerce").to_numpy(float)
        if not np.isfinite(numeric).all():
            raise ValueError(f"score_panel: {column} contains missing/nonfinite values")
        if np.any((numeric < 0.0) | (numeric > 1.0)):
            raise ValueError(f"score_panel: {column} is outside [0, 1]")
        scores[column] = numeric
    for column in SOURCE_FOLD_COLUMNS:
        numeric = pd.to_numeric(scores[column], errors="coerce").to_numpy(float)
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"score_panel: {column} is not a complete integer fold")
        integer = numeric.astype(np.int8)
        if not np.isin(integer, np.arange(5, dtype=np.int8)).all():
            raise ValueError(f"score_panel: {column} contains a fold outside 0..4")
        scores[column] = integer
        per_uid = scores.groupby(ID_COLUMN, sort=False)[column].nunique(dropna=False)
        if not per_uid.eq(1).all():
            raise ValueError(f"score_panel: {column} varies across targets within a study")


def _split_report_groups(
    scores: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    spec: PreGoldAssemblySpec,
) -> tuple[set[str], dict[str, int], set[str]]:
    joined = scores.merge(
        manifest[[ID_COLUMN, REPORT_GROUP_COLUMN]],
        on=ID_COLUMN,
        validate="many_to_one",
    )
    by_source: dict[str, int] = {}
    union: set[str] = set()
    for column in SOURCE_FOLD_COLUMNS:
        counts = joined.groupby(
            ["target", REPORT_GROUP_COLUMN], sort=False
        )[column].nunique(dropna=False)
        local = {str(index[1]) for index in counts[counts.gt(1)].index}
        by_source[column] = len(local)
        union.update(local)
    if by_source["dino_public20_fold"] != spec.expected_split_groups:
        raise ValueError(f"score_panel: DINO split-group count drift: {by_source}")
    if any(by_source[column] != 0 for column in RAD_FOLD_COLUMNS):
        raise ValueError(f"score_panel: Rad split-group count drift: {by_source}")
    ordered_groups = sorted(union)
    if len(ordered_groups) != spec.expected_split_groups:
        raise ValueError("score_panel: source-fold split union count drift")
    if _line_sha(ordered_groups) != spec.split_group_sha256:
        raise ValueError("score_panel: source-fold split report-group SHA drift")
    split_uids = set(
        manifest.loc[manifest[REPORT_GROUP_COLUMN].isin(union), ID_COLUMN].astype(str)
    )
    if len(split_uids) != spec.expected_split_studies:
        raise ValueError("score_panel: source-fold split UID count drift")
    if _line_sha(sorted(split_uids)) != spec.split_uid_sha256:
        raise ValueError("score_panel: source-fold split UID SHA drift")
    return union, by_source, split_uids


def _validate_fold_map(
    fold_map: pd.DataFrame,
    eligible_manifest: pd.DataFrame,
    *,
    spec: PreGoldAssemblySpec,
) -> pd.DataFrame:
    if fold_map[ID_COLUMN].duplicated().any():
        raise ValueError("fold_map: duplicate StudyInstanceUID rows")
    if set(fold_map[ID_COLUMN]) != set(eligible_manifest[ID_COLUMN]):
        raise ValueError("fold_map: UID allowlist differs from exact eligible cohort")
    if len(fold_map) != spec.expected_eligible_studies:
        raise ValueError("fold_map: eligible study count drift")
    if fold_map[REPORT_GROUP_COLUMN].nunique() != spec.expected_eligible_report_groups:
        raise ValueError("fold_map: eligible report-group count drift")
    group_check = fold_map[[ID_COLUMN, REPORT_GROUP_COLUMN]].merge(
        eligible_manifest[[ID_COLUMN, REPORT_GROUP_COLUMN]],
        on=ID_COLUMN,
        suffixes=("_fold", "_manifest"),
        validate="one_to_one",
    )
    if not group_check[f"{REPORT_GROUP_COLUMN}_fold"].equals(
        group_check[f"{REPORT_GROUP_COLUMN}_manifest"]
    ):
        raise ValueError("fold_map: report_group differs from canonical manifest")

    result = fold_map.copy()
    for column in FOLD_MAP_COLUMNS[2:]:
        numeric = pd.to_numeric(result[column], errors="coerce").to_numpy(float)
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"fold_map: {column} must be a complete integer fold")
        result[column] = numeric.astype(np.int8)
    if set(result["outer_fold"]) != set(range(5)):
        raise ValueError("fold_map: outer_fold must contain exactly 0..4")
    counts = tuple(
        int(result["outer_fold"].value_counts().get(fold, 0)) for fold in range(5)
    )
    if counts != spec.expected_outer_counts:
        raise ValueError(f"fold_map: outer-fold counts drift: {counts}")
    if not result.groupby(REPORT_GROUP_COLUMN)["outer_fold"].nunique().eq(1).all():
        raise ValueError("fold_map: a report_group is split across outer folds")
    for outer in range(5):
        column = f"inner_fold_when_outer_{outer}"
        held_out = result["outer_fold"].eq(outer)
        if not result.loc[held_out, column].eq(-1).all():
            raise ValueError(f"fold_map: {column} must be -1 on outer validation rows")
        if not result.loc[~held_out, column].isin({0, 1, 2}).all():
            raise ValueError(f"fold_map: {column} training rows must be in 0..2")
        training = result.loc[~held_out]
        if set(training[column]) != {0, 1, 2}:
            raise ValueError(f"fold_map: {column} must contain all three inner folds")
        if not training.groupby(REPORT_GROUP_COLUMN)[column].nunique().eq(1).all():
            raise ValueError(f"fold_map: a report_group is split in {column}")
    return result


def _assert_no_gold_columns(frame: pd.DataFrame, *, name: str) -> None:
    forbidden = [column for column in frame.columns if "gold" in str(column).casefold()]
    if forbidden:
        raise ValueError(f"{name}: Gold-derived/identity columns escaped: {forbidden}")


def _sort_cells(frame: pd.DataFrame, targets: tuple[str, ...]) -> pd.DataFrame:
    order = {target: index for index, target in enumerate(targets)}
    result = frame.assign(_target_order=frame["target"].map(order))
    if result["_target_order"].isna().any():
        raise ValueError("cells: unexpected target while sorting")
    return (
        result.sort_values([ID_COLUMN, "_target_order"], kind="mergesort")
        .drop(columns="_target_order")
        .reset_index(drop=True)
    )


def assemble_pre_gold_inputs(
    root: str | Path,
    *,
    spec: PreGoldAssemblySpec = FROZEN_V34_SPEC,
) -> PreGoldInputs:
    """Load and validate the exact pre-Gold cell and report inputs.

    ``cells`` contains one row per eligible StudyInstanceUID x target and
    ``reports`` one row per eligible StudyInstanceUID.  Both are restricted by
    the exact fold-map UID allowlist before they leave this function.  The
    nullable ``teacher_hard_label`` is derived only from ``teacher_state``:
    NEGATIVE -> 0, POSITIVE -> 1, ABSTAIN -> null.
    """

    root_path = Path(root)
    source_hashes = _verify_sources(root_path, spec)
    _verify_contract(root_path, spec)

    manifest = _normalize_key_columns(
        _read_csv(
            _resolve(root_path, spec.manifest), MANIFEST_COLUMNS, name="manifest"
        ),
        name="manifest",
    )
    if manifest[ID_COLUMN].duplicated().any():
        raise ValueError("manifest: duplicate StudyInstanceUID rows")
    if len(manifest) != spec.expected_total_studies:
        raise ValueError("manifest: total study count drift")
    if manifest[REPORT_COLUMN].isna().any() or manifest[REPORT_COLUMN].astype(str).str.strip().eq("").any():
        raise ValueError("manifest: Report contains null/empty text")
    manifest["is_gold"] = _as_boolean_identity(manifest["is_gold"])
    if int(manifest["is_gold"].sum()) != spec.expected_gold_identities:
        raise ValueError("manifest: Gold identity count drift")
    manifest_uids = set(manifest[ID_COLUMN])

    scores = _normalize_key_columns(
        _read_csv(
            _resolve(root_path, spec.score_panel), SCORE_COLUMNS, name="score_panel"
        ),
        name="score_panel",
    )
    _validate_cartesian_cells(
        scores,
        name="score_panel",
        expected_uids=manifest_uids,
        targets=spec.targets,
    )
    _validate_score_values(scores)
    if not scores["directional_v45_prediction_scope"].eq(spec.directional_scope).all():
        raise ValueError("score_panel: directional V4.5 source scope drift")
    if not scores.loc[:, list(RAD_FOLD_COLUMNS)].nunique(axis=1).eq(1).all():
        raise ValueError("score_panel: the three Rad source-fold identities differ")
    split_groups, split_by_source, split_uids = _split_report_groups(
        scores, manifest, spec=spec
    )
    split_identity = manifest[ID_COLUMN].isin(split_uids)
    split_gold = int((split_identity & manifest["is_gold"]).sum())
    split_non_gold = int((split_identity & ~manifest["is_gold"]).sum())
    if (split_non_gold, split_gold) != (
        spec.expected_split_non_gold,
        spec.expected_split_gold,
    ):
        raise ValueError("manifest: source-fold split Gold/non-Gold identity drift")

    eligible_manifest = manifest.loc[
        ~manifest["is_gold"]
        & ~manifest[REPORT_GROUP_COLUMN].isin(split_groups)
    ].copy()
    if len(eligible_manifest) != spec.expected_eligible_studies:
        raise ValueError("manifest: exact eligible study count drift")
    if eligible_manifest[REPORT_GROUP_COLUMN].nunique() != spec.expected_eligible_report_groups:
        raise ValueError("manifest: exact eligible report-group count drift")

    fold_map = _normalize_key_columns(
        _read_csv(
            _resolve(root_path, spec.fold_map), FOLD_MAP_COLUMNS, name="fold_map"
        ),
        name="fold_map",
    )
    fold_map = _validate_fold_map(fold_map, eligible_manifest, spec=spec)
    eligible_uids = set(fold_map[ID_COLUMN])

    # Filtering occurs before any derivation or transform.  The validation above
    # is identity/schema checking only and fits nothing.
    scores = scores.loc[scores[ID_COLUMN].isin(eligible_uids)].copy()
    if len(scores) != spec.expected_eligible_cells:
        raise ValueError("score_panel: eligible cell count drift")
    score_groups = scores.merge(
        eligible_manifest[[ID_COLUMN, REPORT_GROUP_COLUMN]],
        on=ID_COLUMN,
        validate="many_to_one",
    )
    for column in SOURCE_FOLD_COLUMNS:
        maximum = score_groups.groupby(
            [REPORT_GROUP_COLUMN, "target"], sort=False
        )[column].nunique(dropna=False).max()
        if int(maximum) != 1:
            raise ValueError(
                f"score_panel: eligible report_group x target split in {column}"
            )

    features = _normalize_key_columns(
        _read_csv(
            _resolve(root_path, spec.feature_panel),
            spec.feature_columns,
            name="feature_panel",
        ),
        name="feature_panel",
    )
    _validate_cartesian_cells(
        features,
        name="feature_panel",
        expected_uids=manifest_uids,
        targets=spec.targets,
    )
    identity = features[[ID_COLUMN, REPORT_GROUP_COLUMN, "is_gold"]].drop_duplicates()
    if len(identity) != spec.expected_total_studies:
        raise ValueError("feature_panel: report identity varies across targets")
    identity["is_gold"] = _as_boolean_identity(identity["is_gold"])
    identity_check = identity.merge(
        manifest[[ID_COLUMN, REPORT_GROUP_COLUMN, "is_gold"]],
        on=ID_COLUMN,
        suffixes=("_feature", "_manifest"),
        validate="one_to_one",
    )
    if not identity_check[f"{REPORT_GROUP_COLUMN}_feature"].equals(
        identity_check[f"{REPORT_GROUP_COLUMN}_manifest"]
    ) or not identity_check["is_gold_feature"].equals(
        identity_check["is_gold_manifest"]
    ):
        raise ValueError("feature_panel: canonical report/Gold identity mismatch")
    features = features.loc[features[ID_COLUMN].isin(eligible_uids)].copy()
    state_to_label = {"NEGATIVE": 0, "POSITIVE": 1, "ABSTAIN": pd.NA}
    if features["teacher_state"].isna().any() or not features["teacher_state"].isin(state_to_label).all():
        raise ValueError("feature_panel: teacher_state must be POSITIVE/NEGATIVE/ABSTAIN")
    features["teacher_hard_label"] = pd.array(
        features["teacher_state"].map(state_to_label), dtype="Int8"
    )
    decisive = int(features["teacher_hard_label"].notna().sum())
    abstain = int(features["teacher_hard_label"].isna().sum())
    if (decisive, abstain) != (
        spec.expected_decisive_cells,
        spec.expected_abstain_cells,
    ):
        raise ValueError(
            f"feature_panel: decisive/abstain coverage drift: {(decisive, abstain)}"
        )
    homogeneous = (
        features.dropna(subset=["teacher_hard_label"])
        .groupby([REPORT_GROUP_COLUMN, "target"], sort=False)["teacher_hard_label"]
        .nunique()
    )
    if not homogeneous.le(1).all():
        raise ValueError("feature_panel: teacher polarity is not report-group homogeneous")

    morphology = _normalize_key_columns(
        _read_csv(
            _resolve(root_path, spec.morphology),
            spec.morphology_columns,
            name="morphology",
        ),
        name="morphology",
    )
    if morphology[ID_COLUMN].duplicated().any():
        raise ValueError("morphology: duplicate StudyInstanceUID rows")
    if set(morphology[ID_COLUMN]) != manifest_uids:
        raise ValueError("morphology: UID set differs from canonical manifest")
    morphology_group_check = morphology[[ID_COLUMN, REPORT_GROUP_COLUMN]].merge(
        manifest[[ID_COLUMN, REPORT_GROUP_COLUMN]],
        on=ID_COLUMN,
        suffixes=("_morph", "_manifest"),
        validate="one_to_one",
    )
    if not morphology_group_check[f"{REPORT_GROUP_COLUMN}_morph"].equals(
        morphology_group_check[f"{REPORT_GROUP_COLUMN}_manifest"]
    ):
        raise ValueError("morphology: report_group differs from canonical manifest")
    morphology = morphology.loc[morphology[ID_COLUMN].isin(eligible_uids)].copy()

    renamed_folds = fold_map.rename(columns=MORPH_FOLD_RENAME)
    morph_fold_columns = [MORPH_FOLD_RENAME[column] for column in FOLD_MAP_COLUMNS[2:]]
    feature_output_columns = [
        column for column in spec.feature_columns if column != "is_gold"
    ]
    cells = features.loc[:, [*feature_output_columns, "teacher_hard_label"]].merge(
        scores,
        on=[ID_COLUMN, "target"],
        validate="one_to_one",
    )
    cells = cells.merge(
        renamed_folds[[ID_COLUMN, *morph_fold_columns]],
        on=ID_COLUMN,
        validate="many_to_one",
    )
    cells = _sort_cells(cells, spec.targets)

    reports = eligible_manifest[[ID_COLUMN, REPORT_COLUMN, REPORT_GROUP_COLUMN]].merge(
        morphology.drop(columns=REPORT_GROUP_COLUMN),
        on=ID_COLUMN,
        validate="one_to_one",
    )
    reports = reports.merge(
        renamed_folds[[ID_COLUMN, REPORT_GROUP_COLUMN, *morph_fold_columns]],
        on=[ID_COLUMN, REPORT_GROUP_COLUMN],
        validate="one_to_one",
    ).sort_values(ID_COLUMN, kind="mergesort").reset_index(drop=True)

    _assert_no_gold_columns(cells, name="cells")
    _assert_no_gold_columns(reports, name="reports")
    if set(cells[ID_COLUMN]) != eligible_uids or set(reports[ID_COLUMN]) != eligible_uids:
        raise ValueError("assembled frames escaped the exact fold-map UID allowlist")

    diagnostics: dict[str, Any] = {
        "status": "PASS_PRE_GOLD_INPUTS",
        "contract_version": spec.contract_version,
        "contract_semantic_hash": spec.contract_semantic_hash,
        "source_hashes": source_hashes,
        "total_studies": spec.expected_total_studies,
        "eligible_studies": len(reports),
        "eligible_report_groups": reports[REPORT_GROUP_COLUMN].nunique(),
        "eligible_cells": len(cells),
        "decisive_cells": decisive,
        "abstain_cells": abstain,
        "split_groups": len(split_groups),
        "split_studies": len(split_uids),
        "split_by_source": split_by_source,
        "loaded_score_columns": list(SCORE_COLUMNS),
        "output_gold_columns": [],
    }
    return PreGoldInputs(cells=cells, reports=reports, diagnostics=diagnostics)


def assemble_external_pre_gold_inputs(
    root: str | Path,
    *,
    spec: PreGoldAssemblySpec = FROZEN_V34_SPEC,
) -> PreGoldExternalInputs:
    """Load the exact outcome-free external scoring cohort.

    This function intentionally reuses :func:`assemble_pre_gold_inputs` as a
    complete source/hash/fold gate, then performs fresh allowlisted reads for
    only the external identities.  It never reads ``gold58_labels.csv`` and it
    never requests a score-panel column containing ``gold`` or ``*_vs_gold``.
    Gold identity is used internally only to construct the frozen 58-study
    external set and is removed before either frame is returned.
    """

    root_path = Path(root).resolve()
    proxy = assemble_pre_gold_inputs(root_path, spec=spec)
    proxy_uids = set(proxy.reports[ID_COLUMN].astype(str))

    manifest = _normalize_key_columns(
        _read_csv(
            _resolve(root_path, spec.manifest), MANIFEST_COLUMNS, name="manifest"
        ),
        name="manifest",
    )
    manifest["is_gold"] = _as_boolean_identity(manifest["is_gold"])
    if len(manifest) != spec.expected_total_studies:
        raise ValueError("manifest: total study count drift")
    if int(manifest["is_gold"].sum()) != spec.expected_gold_identities:
        raise ValueError("manifest: Gold identity count drift")

    scores = _normalize_key_columns(
        _read_csv(
            _resolve(root_path, spec.score_panel), SCORE_COLUMNS, name="score_panel"
        ),
        name="score_panel",
    )
    manifest_uids = set(manifest[ID_COLUMN])
    _validate_cartesian_cells(
        scores,
        name="score_panel",
        expected_uids=manifest_uids,
        targets=spec.targets,
    )
    _validate_score_values(scores)
    split_groups, split_by_source, split_uids = _split_report_groups(
        scores, manifest, spec=spec
    )

    split_non_gold = set(
        manifest.loc[
            ~manifest["is_gold"]
            & manifest[REPORT_GROUP_COLUMN].isin(split_groups),
            ID_COLUMN,
        ].astype(str)
    )
    identity_only = set(
        manifest.loc[manifest["is_gold"], ID_COLUMN].astype(str)
    )
    if len(split_non_gold) != spec.expected_split_non_gold:
        raise ValueError("external cohort: split non-Gold identity count drift")
    if len(identity_only) != spec.expected_gold_identities:
        raise ValueError("external cohort: Gold identity count drift")
    external_uids = split_non_gold | identity_only
    expected_external = spec.expected_split_non_gold + spec.expected_gold_identities
    if len(external_uids) != expected_external:
        raise ValueError("external cohort: identity union count drift")
    if external_uids.intersection(proxy_uids):
        raise ValueError("external cohort overlaps the eligible proxy cohort")
    if proxy_uids | external_uids != manifest_uids:
        raise ValueError("proxy plus external identity union is incomplete")

    features = _normalize_key_columns(
        _read_csv(
            _resolve(root_path, spec.feature_panel),
            spec.feature_columns,
            name="feature_panel",
        ),
        name="feature_panel",
    )
    _validate_cartesian_cells(
        features,
        name="feature_panel",
        expected_uids=manifest_uids,
        targets=spec.targets,
    )
    identity = features[[ID_COLUMN, REPORT_GROUP_COLUMN, "is_gold"]].drop_duplicates()
    identity["is_gold"] = _as_boolean_identity(identity["is_gold"])
    identity_check = identity.merge(
        manifest[[ID_COLUMN, REPORT_GROUP_COLUMN, "is_gold"]],
        on=ID_COLUMN,
        suffixes=("_feature", "_manifest"),
        validate="one_to_one",
    )
    if not identity_check[f"{REPORT_GROUP_COLUMN}_feature"].equals(
        identity_check[f"{REPORT_GROUP_COLUMN}_manifest"]
    ) or not identity_check["is_gold_feature"].equals(
        identity_check["is_gold_manifest"]
    ):
        raise ValueError("feature_panel: canonical report/Gold identity mismatch")
    features = features.loc[features[ID_COLUMN].isin(external_uids)].copy()
    state_to_label = {"NEGATIVE": 0, "POSITIVE": 1, "ABSTAIN": pd.NA}
    if features["teacher_state"].isna().any() or not features[
        "teacher_state"
    ].isin(state_to_label).all():
        raise ValueError(
            "feature_panel: teacher_state must be POSITIVE/NEGATIVE/ABSTAIN"
        )
    features["teacher_hard_label"] = pd.array(
        features["teacher_state"].map(state_to_label), dtype="Int8"
    )

    morphology = _normalize_key_columns(
        _read_csv(
            _resolve(root_path, spec.morphology),
            spec.morphology_columns,
            name="morphology",
        ),
        name="morphology",
    )
    if morphology[ID_COLUMN].duplicated().any():
        raise ValueError("morphology: duplicate StudyInstanceUID rows")
    if set(morphology[ID_COLUMN]) != manifest_uids:
        raise ValueError("morphology: UID set differs from canonical manifest")
    morphology = morphology.loc[morphology[ID_COLUMN].isin(external_uids)].copy()

    external_manifest = manifest.loc[
        manifest[ID_COLUMN].isin(external_uids),
        [ID_COLUMN, REPORT_COLUMN, REPORT_GROUP_COLUMN],
    ].copy()
    score_external = scores.loc[scores[ID_COLUMN].isin(external_uids)].copy()
    feature_output_columns = [
        column for column in spec.feature_columns if column != "is_gold"
    ]
    cells = features.loc[:, [*feature_output_columns, "teacher_hard_label"]].merge(
        score_external,
        on=[ID_COLUMN, "target"],
        validate="one_to_one",
    )
    split_mask = cells[REPORT_GROUP_COLUMN].isin(split_groups)
    cells["image_oof_source_eligible"] = ~split_mask
    # The contract explicitly forbids DINO/consensus evidence for these source-
    # split rows.  Rad components remain finite and descriptive.
    cells.loc[split_mask, "dino_public20_equal_oof"] = np.nan
    cells = _sort_cells(cells, spec.targets)

    reports = external_manifest.merge(
        morphology.drop(columns=REPORT_GROUP_COLUMN),
        on=ID_COLUMN,
        validate="one_to_one",
    ).sort_values(ID_COLUMN, kind="mergesort").reset_index(drop=True)

    _assert_no_gold_columns(cells, name="external cells")
    _assert_no_gold_columns(reports, name="external reports")
    if len(reports) != expected_external:
        raise ValueError("external cohort: report row count drift")
    if len(cells) != expected_external * len(spec.targets):
        raise ValueError("external cohort: cell row count drift")
    if set(cells[ID_COLUMN]) != external_uids or set(reports[ID_COLUMN]) != external_uids:
        raise ValueError("external cohort: returned identity set drift")
    isolated_uids = set(cells.loc[~cells["image_oof_source_eligible"], ID_COLUMN])
    if isolated_uids != split_uids:
        raise ValueError("external cohort: DINO isolation identity set drift")
    if not cells.loc[~cells["image_oof_source_eligible"], "dino_public20_equal_oof"].isna().all():
        raise AssertionError("external cohort: isolated DINO values remain populated")
    if cells.loc[cells["image_oof_source_eligible"], "dino_public20_equal_oof"].isna().any():
        raise ValueError("external cohort: applicable DINO values are missing")

    sorted_external = sorted(external_uids)
    diagnostics: dict[str, Any] = {
        "status": "PASS_PRE_GOLD_EXTERNAL_IDENTITIES_OUTCOMES_NOT_LOADED",
        "contract_version": spec.contract_version,
        "contract_semantic_hash": spec.contract_semantic_hash,
        "external_studies": len(reports),
        "external_cells": len(cells),
        "identity_only_studies": len(identity_only),
        "split_non_gold_studies": len(split_non_gold),
        "image_isolated_studies": len(isolated_uids),
        "image_isolated_cells": int((~cells["image_oof_source_eligible"]).sum()),
        "external_uid_sequence_sha256": _line_sha(sorted_external),
        "split_uid_sequence_sha256": _line_sha(sorted(split_uids)),
        "split_by_source": split_by_source,
        "loaded_score_columns": list(SCORE_COLUMNS),
        "output_gold_columns": [],
        "gold_outcomes_loaded": False,
    }
    return PreGoldExternalInputs(
        cells=cells.reset_index(drop=True),
        reports=reports,
        diagnostics=diagnostics,
    )


__all__ = [
    "BoundFile",
    "FROZEN_V34_SPEC",
    "PreGoldAssemblySpec",
    "PreGoldExternalInputs",
    "PreGoldInputs",
    "assemble_external_pre_gold_inputs",
    "assemble_pre_gold_inputs",
    "sha256_file",
]
