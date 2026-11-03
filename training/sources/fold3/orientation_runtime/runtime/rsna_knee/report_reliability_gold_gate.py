"""Fail-closed boundary between proxy freeze and Gold58 evaluation.

The verifier in this module never opens the Gold label file.  It validates the
exact proxy-freeze manifest, every bound code/artifact hash, the frozen rules,
and the pre-Gold proxy/external prediction frames.  Only
``evaluate_gold_after_proxy_freeze`` may open Gold, and it does so strictly
after the manifest has passed and its caller-supplied SHA-256 has matched.

The post-freeze panel keeps three evidentiary channels separate:

* morphology/image discordance is a radiologist-*review* proxy, not a confirmed
  report error;
* LLM-family conflict is a parsing/self-consistency signal, not proof of an LLM
  parsing error;
* image-family disagreement is image uncertainty, not report truth.

No label or sample weight is produced or changed by this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .constants import ID_COLUMN, TARGET_COLUMNS
from .report_reliability_inputs import BoundFile, sha256_file


KEYS: tuple[str, str] = (ID_COLUMN, "target")

PROXY_REQUIRED_COLUMNS: tuple[str, ...] = (
    ID_COLUMN,
    "target",
    "report_group",
    "teacher_state",
    "image_consensus_rank",
    "image_family_disagreement",
    "rad_internal_dispersion",
    "image_state",
    "risk_percentile",
    "guardrailed_review_risk_percentile",
)

EXTERNAL_REQUIRED_COLUMNS: tuple[str, ...] = (
    ID_COLUMN,
    "target",
    "teacher_state",
    "family_state_summary",
    "other_positive_count",
    "other_negative_count",
    "other_abstain_count",
    "image_consensus_rank",
    "image_family_disagreement",
    "rad_internal_dispersion",
    "image_state",
    "image_evidence_strength",
    "teacher_component_self_conflict",
    "external_llm_audit_disagreement",
    "llm_silence_instability",
    "omission_like_risk_percentile",
    "contradiction_like_risk_percentile",
    "label_review_risk_percentile",
    "report_risk_percentile_proxy_reference",
    "proxy_model_status",
    "candidate_reason_codes",
    "proxy_evidence_tier",
    "risk_strength",
    "candidate_weight_status",
)

ALLOWED_GOLD_CODES: tuple[str, ...] = (
    "TEACHER_NEGATIVE_GOLD_POSITIVE",
    "TEACHER_POSITIVE_GOLD_NEGATIVE",
    "GOLD_NEGATIVE_IMAGE_CONSENSUS_HIGH",
    "GOLD_POSITIVE_IMAGE_CONSENSUS_LOW",
)

PRE_GOLD_REASON_PRECEDENCE: tuple[str, ...] = (
    "TEACHER_ABSTAIN",
    "IMAGE_OOF_INVALID",
    "IMAGE_FAMILY_DISAGREEMENT",
    "EXTERNAL_LLM_AUDIT_DISAGREEMENT",
    "TEACHER_COMPONENT_SELF_CONFLICT",
    "LLM_SILENCE_INSTABILITY",
    "TEACHER_NEGATIVE_IMAGE_CONSENSUS_HIGH",
    "TEACHER_POSITIVE_IMAGE_CONSENSUS_LOW",
)

CANDIDATE_WEIGHT_STATUS = "UNAPPLIED_REQUIRES_SEPARATE_GROUPED_OOF_RETRAINING"

FREEZE_RULES: Mapping[str, Any] = {
    "gold_outcomes_loaded": False,
    "gold_use_in_selection": "FORBIDDEN",
    "gold_annotation_mode": "APPEND_ONLY_FOUR_DESCRIPTIVE_CODES",
    "gold_allowed_append_codes": list(ALLOWED_GOLD_CODES),
    "proxy_rank_mutable_after_gold": False,
    "feature_or_model_selection_mutable_after_gold": False,
    "alpha_or_threshold_mutable_after_gold": False,
    "shortlist_or_candidate_rank_mutable_after_gold": False,
    "combined_risk_rule": "NO_LEARNED_OR_WEIGHTED_COMBINATION",
    "candidate_weight_status": CANDIDATE_WEIGHT_STATUS,
    "automatic_label_or_weight_change": False,
    "label_review_risk_formula": {
        "teacher_negative": "omission_like_risk_percentile",
        "teacher_positive": "contradiction_like_risk_percentile",
        "teacher_abstain": None,
        "image_disagreement_may_increase_risk": False,
    },
    "risk_tiers": {
        "VERY_HIGH": ">=0.95",
        "HIGH": ">=0.90_and_<0.95",
        "MODERATE": ">=0.80_and_<0.90",
        "LOW": "<0.80",
        "NOT_APPLICABLE": "null",
    },
    "reason_code_precedence": list(PRE_GOLD_REASON_PRECEDENCE),
    "radiologist_review_proxy": {
        "field": "label_review_risk_percentile",
        "meaning": "REVIEW_PRIORITY_NOT_CONFIRMED_REPORT_ERROR",
    },
    "llm_parsing_signal": {
        "fields": [
            "external_llm_audit_disagreement",
            "teacher_component_self_conflict",
            "llm_silence_instability",
        ],
        "meaning": "LLM_STATE_CONFLICT_NOT_PROVEN_PARSING_ERROR",
    },
    "image_uncertainty": {
        "fields": [
            "image_family_disagreement",
            "rad_internal_dispersion",
            "image_evidence_strength",
            "image_state",
        ],
        "meaning": "IMAGE_UNCERTAINTY_NOT_REPORT_TRUTH",
        "may_strengthen_report_risk": False,
    },
    "confirmation_rule": (
        "NO_CONFIRMED_REPORT_ERROR_WITHOUT_BLINDED_MANUAL_REPORT_IMAGE_ADJUDICATION"
    ),
}

REQUIRED_CODE_ROLES: tuple[str, ...] = (
    "driver",
    "input_module",
    "feature_module",
    "stats_module",
    "evaluation_module",
    "panel_module",
    "final_fit_driver",
    "post_oof_module",
    "post_oof_driver",
    "canonical_panel_builder",
    "complete_panel_module",
    "abstain_builder",
    "segment_module",
    "segment_driver",
    "segment_gate_module",
    "segment_verifier_driver",
    "lopo_module",
    "lopo_driver",
    "lopo_verifier_driver",
    "gold_gate_module",
    "postfreeze_gold_module",
    "postfreeze_gold_driver",
    "freeze_manifest_module",
    "freeze_manifest_builder",
)

REQUIRED_ARTIFACT_ROLES: tuple[str, ...] = (
    "proxy_primary_audit",
    "proxy_oof_scores",
    "proxy_direction_metrics",
    "proxy_nested_alpha_trace",
    "proxy_post_oof_audit",
    "proxy_post_oof_determinism_audit",
    "proxy_direction_evaluation",
    "proxy_only_shortlist",
    "proxy_alignment_permutation",
    "proxy_alignment_strata",
    "proxy_component_robustness",
    "proxy_review_capture",
    "final_fit_audit",
    "final_fit_alpha_trace",
    "final_external_scores",
    "final_external_predictions",
    "ecdf_references",
    "selected_alphas",
    "percentile_references",
    "model_parameters",
    "proxy_decisive_presentation",
    "canonical_proxy_presentation_audit",
    "proxy_abstain_presentation",
    "proxy_abstain_presentation_audit",
    "segment_universe",
    "segment_adjustment_cache",
    "segment_adjustment_alpha_trace",
    "segment_adjustment_cache_audit",
    "segment_effects",
    "segment_family_maxt",
    "segment_core_shortlist",
    "segment_effects_audit",
    "segment_effects_independent_audit",
    "segment_lopo_profile_effects",
    "segment_lopo_alpha_trace",
    "segment_lopo_survivor_summary",
    "segment_final_shortlist",
    "segment_lopo_audit",
    "segment_lopo_independent_audit",
)

REQUIRED_CODE_PATHS: Mapping[str, str] = {
    "driver": "scripts/run_report_reliability_proxy_v3.py",
    "input_module": "src/rsna_knee/report_reliability_inputs.py",
    "feature_module": "src/rsna_knee/foldfit_orthographic_features.py",
    "stats_module": "src/rsna_knee/report_reliability_stats.py",
    "evaluation_module": "src/rsna_knee/report_reliability_evaluation.py",
    "panel_module": "src/rsna_knee/report_reliability_panel.py",
    "final_fit_driver": "scripts/freeze_report_reliability_proxy_v3.py",
    "post_oof_module": "src/rsna_knee/report_reliability_post_oof.py",
    "post_oof_driver": "scripts/run_report_reliability_post_oof_v3.py",
    "canonical_panel_builder": "scripts/build_report_reliability_canonical_panels_v3.py",
    "complete_panel_module": "src/rsna_knee/report_reliability_complete_panel.py",
    "abstain_builder": "scripts/build_report_reliability_proxy_abstains_v3.py",
    "segment_module": "src/rsna_knee/report_reliability_segments.py",
    "segment_driver": "scripts/run_report_reliability_segments_v3.py",
    "segment_gate_module": "src/rsna_knee/report_reliability_segment_gate.py",
    "segment_verifier_driver": "scripts/verify_report_reliability_segments_v3.py",
    "lopo_module": "src/rsna_knee/report_reliability_lopo.py",
    "lopo_driver": "scripts/run_report_reliability_lopo_v3.py",
    "lopo_verifier_driver": "scripts/verify_report_reliability_lopo_v3.py",
    "gold_gate_module": "src/rsna_knee/report_reliability_gold_gate.py",
    "postfreeze_gold_module": "src/rsna_knee/report_reliability_postfreeze_gold.py",
    "postfreeze_gold_driver": "scripts/build_report_reliability_postfreeze_v3.py",
    "freeze_manifest_module": "src/rsna_knee/report_reliability_freeze_manifest.py",
    "freeze_manifest_builder": "scripts/build_proxy_freeze_manifest_v3.py",
}

REQUIRED_ARTIFACT_FILENAMES: Mapping[str, str] = {
    "proxy_primary_audit": "PROXY_PRIMARY_AUDIT.json",
    "proxy_oof_scores": "proxy_oof_scores_v3.csv.gz",
    "proxy_direction_metrics": "proxy_direction_metrics_v3.csv",
    "proxy_nested_alpha_trace": "proxy_nested_alpha_trace_v3.csv",
    "proxy_post_oof_audit": "PROXY_POST_OOF_AUDIT.json",
    "proxy_post_oof_determinism_audit": "PROXY_POST_OOF_DETERMINISM_AUDIT.json",
    "proxy_direction_evaluation": "proxy_direction_evaluation_v3.csv",
    "proxy_only_shortlist": "proxy_model_shortlist_v3.csv",
    "proxy_alignment_permutation": "proxy_alignment_permutation_v3.csv",
    "proxy_alignment_strata": "proxy_alignment_strata_v3.csv.gz",
    "proxy_component_robustness": "proxy_component_robustness_v3.csv",
    "proxy_review_capture": "proxy_review_capture_v3.csv",
    "final_fit_audit": "PROXY_FINAL_EXTERNAL_AUDIT.json",
    "final_fit_alpha_trace": "proxy_final_alpha_trace_v3.csv",
    "final_external_scores": "proxy_final_external_scores_v3.csv.gz",
    "final_external_predictions": "final_external_predictions_v3.csv.gz",
    "ecdf_references": "proxy_final_image_ecdf_references_v3.csv.gz",
    "selected_alphas": "proxy_final_selected_alphas_v3.csv",
    "percentile_references": "proxy_final_risk_percentile_references_v3.csv.gz",
    "model_parameters": "proxy_final_model_parameters_v3.json",
    "proxy_decisive_presentation": "proxy_decisive_presentation_v3.csv.gz",
    "canonical_proxy_presentation_audit": "CANONICAL_PROXY_PRESENTATION_AUDIT.json",
    "proxy_abstain_presentation": "proxy_abstain_presentation_v3.csv.gz",
    "proxy_abstain_presentation_audit": "PROXY_ABSTAIN_PRESENTATION_REVALIDATION_AUDIT.json",
    "segment_universe": "SEGMENT_UNIVERSE.json",
    "segment_adjustment_cache": "segment_adjustment_group_cache_v3.csv.gz",
    "segment_adjustment_alpha_trace": "segment_adjustment_alpha_trace_v3.csv.gz",
    "segment_adjustment_cache_audit": "SEGMENT_ADJUSTMENT_CACHE_AUDIT.json",
    "segment_effects": "segment_effects_v3.csv.gz",
    "segment_family_maxt": "segment_family_maxT_v3.csv",
    "segment_core_shortlist": "segment_proxy_shortlist_v3.csv",
    "segment_effects_audit": "SEGMENT_EFFECTS_AUDIT.json",
    "segment_effects_independent_audit": "SEGMENT_EFFECTS_INDEPENDENT_AUDIT.json",
    "segment_lopo_profile_effects": "segment_lopo_profile_effects_v3.csv.gz",
    "segment_lopo_alpha_trace": "segment_lopo_alpha_trace_v3.csv.gz",
    "segment_lopo_survivor_summary": "segment_lopo_survivor_summary_v3.csv",
    "segment_final_shortlist": "segment_proxy_shortlist_lopo_v3.csv",
    "segment_lopo_audit": "SEGMENT_LOPO_AUDIT.json",
    "segment_lopo_independent_audit": "SEGMENT_LOPO_INDEPENDENT_AUDIT.json",
}

REQUIRED_AUDIT_STATUSES: Mapping[str, str] = {
    "proxy_primary_audit": "PASS_PROXY_PRIMARY_OOF_GOLD_NOT_LOADED",
    "proxy_post_oof_audit": "PASS_PROXY_POST_OOF_GOLD_NOT_LOADED",
    "proxy_post_oof_determinism_audit": "PASS_BYTE_IDENTICAL_TWO_RUNS_AND_FINAL",
    "final_fit_audit": "PASS_PROXY_FINAL_EXTERNAL_GOLD_OUTCOMES_NOT_LOADED",
    "canonical_proxy_presentation_audit": "PASS_CANONICAL_PROXY_PRESENTATION_PRE_GOLD",
    "proxy_abstain_presentation_audit": (
        "PASS_PROXY_ABSTAIN_PRESENTATION_REVALIDATED_GOLD_NOT_LOADED"
    ),
    "segment_adjustment_cache_audit": "PASS_SEGMENT_ADJUSTMENT_CACHE_GOLD_NOT_LOADED",
    "segment_effects_audit": "BLOCKED_SEGMENT_CORE_SURVIVORS_REQUIRE_EXACT_LOPO",
    "segment_effects_independent_audit": (
        "PASS_INDEPENDENT_SEGMENT_EFFECTS_EXACT_TWO_RUNS_PENDING_LOPO_GOLD_NOT_LOADED"
    ),
    "segment_lopo_audit": "PASS_LOPO_SURVIVING_ONLY_COMPLETE_GOLD_NOT_LOADED",
    "segment_lopo_independent_audit": (
        "PASS_INDEPENDENT_LOPO_OUTPUTS_GOLD_NOT_LOADED"
    ),
}

EXPECTED_SOURCE_HASHES: Mapping[str, str] = {
    "contract": "2be69a3454ed7fce1d56c302366733f2ca4aa3e77c366c372f50127c3bc09491",
    "manifest": "38f2df1518867a7d6289fcb5be5171e75249f7e7a7d64f1eacc4acb82b485a23",
    "feature_panel": "40b18680c7a60fc84180ce1416ef6a1499a88e4667c3d82718bdb114bacc1355",
    "morphology": "d4b51f902b119519737d5e991211ca6ac5238d8ea7087d7490c69d23774f4742",
    "score_panel": "109155a7a6e07ae94d7442fe0f7f55e7e49ff9288348cce6d7a4b7aa4389b4d1",
    "fold_map": "f115e4855e5941caf29c859a4b854eedc69cd5f87d98dfe723785585156cfb26",
}

EXPECTED_SOURCE_FILES: Mapping[str, BoundFile] = {
    "contract": BoundFile(
        ".codex/runs/rsna-knee-report-reliability-v3/ANALYSIS_CONTRACT.yaml",
        EXPECTED_SOURCE_HASHES["contract"],
    ),
    "manifest": BoundFile(
        ".codex/runs/rsna-knee-full-cache-pipeline/manifests/manifest_evidence_gated.csv",
        EXPECTED_SOURCE_HASHES["manifest"],
    ),
    "feature_panel": BoundFile(
        ".codex/runs/rsna-knee-report-reliability-v3/report_reliability_feature_panel_v3.csv.gz",
        EXPECTED_SOURCE_HASHES["feature_panel"],
    ),
    "morphology": BoundFile(
        ".codex/runs/rsna-knee-morphology-error-risk-v1/report_morphology_features_v1.csv.gz",
        EXPECTED_SOURCE_HASHES["morphology"],
    ),
    "score_panel": BoundFile(
        ".codex/runs/rsna-knee-full-train-inference/results/train_model_score_panel_long.csv.gz",
        EXPECTED_SOURCE_HASHES["score_panel"],
    ),
    "fold_map": BoundFile(
        ".codex/runs/rsna-knee-label-risk-research-v2/fold_map_v2.csv",
        EXPECTED_SOURCE_HASHES["fold_map"],
    ),
}


@dataclass(frozen=True)
class GoldGateSpec:
    schema_version: str
    run_directory: str
    contract: BoundFile
    canonical_identity_manifest: BoundFile
    fold_map: BoundFile
    gold_source: BoundFile
    contract_version: str
    contract_semantic_hash: str
    expected_source_hashes: Mapping[str, str]
    source_files: Mapping[str, BoundFile]
    feature_sets_sha256: str
    targets: tuple[str, ...]
    expected_proxy_rows: int
    expected_proxy_studies: int
    expected_proxy_report_groups: int
    expected_external_rows: int
    expected_external_studies: int
    expected_external_gold_identities: int
    expected_external_non_gold: int
    required_code_roles: tuple[str, ...] = REQUIRED_CODE_ROLES
    required_artifact_roles: tuple[str, ...] = REQUIRED_ARTIFACT_ROLES
    required_code_paths: Mapping[str, str] = field(
        default_factory=lambda: REQUIRED_CODE_PATHS
    )
    required_artifact_filenames: Mapping[str, str] = field(
        default_factory=lambda: REQUIRED_ARTIFACT_FILENAMES
    )
    required_audit_statuses: Mapping[str, str] = field(
        default_factory=lambda: REQUIRED_AUDIT_STATUSES
    )


FROZEN_GOLD_GATE_SPEC = GoldGateSpec(
    schema_version="1.2.0",
    run_directory=".codex/runs/rsna-knee-report-reliability-v3",
    contract=BoundFile(
        ".codex/runs/rsna-knee-report-reliability-v3/ANALYSIS_CONTRACT.yaml",
        EXPECTED_SOURCE_HASHES["contract"],
    ),
    canonical_identity_manifest=BoundFile(
        ".codex/runs/rsna-knee-full-cache-pipeline/manifests/manifest_evidence_gated.csv",
        EXPECTED_SOURCE_HASHES["manifest"],
    ),
    fold_map=BoundFile(
        ".codex/runs/rsna-knee-label-risk-research-v2/fold_map_v2.csv",
        EXPECTED_SOURCE_HASHES["fold_map"],
    ),
    gold_source=BoundFile(
        "data/gold58_mri/gold58_labels.csv",
        "3644ae2217484fcf2b08158005692521c6e3901e0f309873d7e8f0004f558885",
    ),
    contract_version="3.4.0",
    contract_semantic_hash=(
        "f00d67756c1d840dfeb5d6738e172d03eee452f2e091c29f4a3366ff4ec30e31"
    ),
    expected_source_hashes=EXPECTED_SOURCE_HASHES,
    source_files=EXPECTED_SOURCE_FILES,
    feature_sets_sha256=(
        "b356c0c0967410d8b08de45add14610bc753a799caff14596be1b5151f89b388"
    ),
    targets=tuple(TARGET_COLUMNS),
    expected_proxy_rows=51_353,
    expected_proxy_studies=4_314,
    expected_proxy_report_groups=4_189,
    expected_external_rows=93 * len(TARGET_COLUMNS),
    expected_external_studies=93,
    expected_external_gold_identities=58,
    expected_external_non_gold=35,
)


@dataclass(frozen=True)
class VerifiedProxyFreeze:
    manifest: Mapping[str, Any]
    manifest_sha256: str
    proxy_oof: pd.DataFrame
    external_predictions: pd.DataFrame
    artifact_hashes: Mapping[str, str]


@dataclass(frozen=True)
class GoldEvaluationResult:
    panel: pd.DataFrame
    summary: Mapping[str, Any]
    freeze_manifest_sha256: str


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _line_sha(values: Sequence[object]) -> str:
    return hashlib.sha256(
        "\n".join(str(value) for value in values).encode("utf-8")
    ).hexdigest()


def _cell_sha(frame: pd.DataFrame) -> str:
    values = sorted(
        f"{uid}|{target}"
        for uid, target in frame.loc[:, list(KEYS)].itertuples(index=False, name=None)
    )
    return _line_sha(values)


def _columns_sha(columns: Sequence[object]) -> str:
    return _line_sha([str(column) for column in columns])


def _resolve_inside(root: Path, relative: str, *, boundary: Path | None = None) -> Path:
    path = Path(relative)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    workspace = root.resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError(f"path escapes workspace root: {relative}")
    if boundary is not None and not resolved.is_relative_to(boundary.resolve()):
        raise ValueError(f"artifact path escapes frozen run directory: {relative}")
    return resolved


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name}: exact keys required; missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _read_json(path: Path, *, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name}: invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name}: expected a JSON object")
    return value


def _read_csv_allowlist(
    path: Path,
    required: Sequence[str],
    *,
    name: str,
) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = sorted(set(required).difference(header))
    if missing:
        raise ValueError(f"{name}: missing required columns {missing}")
    forbidden = [column for column in header if "gold" in str(column).casefold()]
    if forbidden:
        raise ValueError(f"{name}: pre-Gold artifact contains Gold columns {forbidden}")
    unsafe_weight = [
        column
        for column in header
        if any(
            fragment in str(column).casefold()
            for fragment in ("new_weight", "applied_weight", "label_weight", "training_weight")
        )
    ]
    if unsafe_weight:
        raise ValueError(f"{name}: automatic-weight columns are forbidden {unsafe_weight}")
    return pd.read_csv(path, usecols=header)


def _validate_key_frame(
    frame: pd.DataFrame,
    *,
    name: str,
    targets: tuple[str, ...],
    expected_rows: int,
    expected_studies: int,
) -> pd.DataFrame:
    result = frame.copy()
    for column in KEYS:
        if result[column].isna().any():
            raise ValueError(f"{name}: {column} contains nulls")
        result[column] = result[column].astype(str)
        if result[column].str.strip().eq("").any():
            raise ValueError(f"{name}: {column} contains empty strings")
    if result.duplicated(list(KEYS)).any():
        raise ValueError(f"{name}: duplicate StudyInstanceUID x target rows")
    if len(result) != expected_rows or result[ID_COLUMN].nunique() != expected_studies:
        raise ValueError(f"{name}: frozen row/study coverage drift")
    if set(result["target"]) != set(targets):
        raise ValueError(f"{name}: target set drift")
    return result


def _validate_unit_interval_or_null(frame: pd.DataFrame, columns: Sequence[str], *, name: str) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        nonnull = frame[column].notna()
        if values[nonnull].isna().any() or not np.isfinite(values[nonnull]).all():
            raise ValueError(f"{name}: {column} contains nonnumeric/nonfinite values")
        if ((values[nonnull] < 0.0) | (values[nonnull] > 1.0)).any():
            raise ValueError(f"{name}: {column} is outside [0, 1]")


def _validate_proxy_oof(frame: pd.DataFrame, spec: GoldGateSpec) -> pd.DataFrame:
    result = _validate_key_frame(
        frame,
        name="proxy_oof_scores",
        targets=spec.targets,
        expected_rows=spec.expected_proxy_rows,
        expected_studies=spec.expected_proxy_studies,
    )
    if result["report_group"].isna().any():
        raise ValueError("proxy_oof_scores: report_group contains nulls")
    result["report_group"] = result["report_group"].astype(str)
    if result["report_group"].nunique() != spec.expected_proxy_report_groups:
        raise ValueError("proxy_oof_scores: report-group coverage drift")
    if not result["teacher_state"].isin({"POSITIVE", "NEGATIVE"}).all():
        raise ValueError("proxy_oof_scores: only decisive teacher states are allowed")
    _validate_unit_interval_or_null(
        result,
        (
            "image_consensus_rank",
            "image_family_disagreement",
            "rad_internal_dispersion",
            "risk_percentile",
            "guardrailed_review_risk_percentile",
        ),
        name="proxy_oof_scores",
    )
    return result


def _validate_external_predictions(
    frame: pd.DataFrame, spec: GoldGateSpec
) -> pd.DataFrame:
    result = _validate_key_frame(
        frame,
        name="final_external_predictions",
        targets=spec.targets,
        expected_rows=spec.expected_external_rows,
        expected_studies=spec.expected_external_studies,
    )
    allowed_states = {"POSITIVE", "NEGATIVE", "ABSTAIN"}
    if not result["teacher_state"].isin(allowed_states).all():
        raise ValueError("final_external_predictions: invalid teacher_state")
    _validate_unit_interval_or_null(
        result,
        (
            "image_consensus_rank",
            "image_family_disagreement",
            "rad_internal_dispersion",
            "omission_like_risk_percentile",
            "contradiction_like_risk_percentile",
            "label_review_risk_percentile",
            "report_risk_percentile_proxy_reference",
        ),
        name="final_external_predictions",
    )
    negative = result["teacher_state"].eq("NEGATIVE")
    positive = result["teacher_state"].eq("POSITIVE")
    abstain = result["teacher_state"].eq("ABSTAIN")
    if result.loc[negative, "omission_like_risk_percentile"].isna().any():
        raise ValueError("final_external_predictions: missing negative-direction risk")
    if result.loc[negative, "contradiction_like_risk_percentile"].notna().any():
        raise ValueError("final_external_predictions: contradiction risk on negative teacher")
    if result.loc[positive, "contradiction_like_risk_percentile"].isna().any():
        raise ValueError("final_external_predictions: missing positive-direction risk")
    if result.loc[positive, "omission_like_risk_percentile"].notna().any():
        raise ValueError("final_external_predictions: omission risk on positive teacher")
    if result.loc[abstain, [
        "omission_like_risk_percentile",
        "contradiction_like_risk_percentile",
        "label_review_risk_percentile",
    ]].notna().any().any():
        raise ValueError("final_external_predictions: teacher abstain must have null risks")
    applicable = pd.Series(np.nan, index=result.index, dtype=float)
    applicable.loc[negative] = pd.to_numeric(
        result.loc[negative, "omission_like_risk_percentile"]
    )
    applicable.loc[positive] = pd.to_numeric(
        result.loc[positive, "contradiction_like_risk_percentile"]
    )
    exposed = pd.to_numeric(result["label_review_risk_percentile"], errors="coerce")
    reference = pd.to_numeric(
        result["report_risk_percentile_proxy_reference"], errors="coerce"
    )
    populated = exposed.notna()
    if not np.allclose(
        exposed[populated].to_numpy(float),
        applicable[populated].to_numpy(float),
        rtol=0.0,
        atol=1e-12,
    ) or not np.allclose(
        exposed[populated].to_numpy(float),
        reference[populated].to_numpy(float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "final_external_predictions: disagreement altered/increased frozen risk"
        )
    if not result["candidate_weight_status"].eq(CANDIDATE_WEIGHT_STATUS).all():
        raise ValueError("final_external_predictions: automatic weight status is forbidden")
    reasons = result["candidate_reason_codes"].fillna("").astype(str)
    forbidden_codes = (*ALLOWED_GOLD_CODES, "CONFIRMED_REPORT_ERROR")
    if reasons.str.contains("|".join(forbidden_codes), regex=True).any():
        raise ValueError("final_external_predictions: post-Gold/confirmed code present pre-Gold")
    return result


def _artifact_binding_path(
    root: Path,
    run_dir: Path,
    binding: Mapping[str, Any],
    *,
    role: str,
) -> Path:
    if not isinstance(binding, dict) or not {"path", "sha256"}.issubset(binding):
        raise ValueError(f"artifact {role}: path and sha256 are required")
    path = _resolve_inside(root, str(binding["path"]), boundary=run_dir)
    if "gold" in path.name.casefold():
        raise ValueError(f"artifact {role}: pre-Gold artifact filename contains gold")
    if not path.is_file():
        raise FileNotFoundError(f"artifact {role}: missing {path}")
    digest = sha256_file(path)
    if digest != binding["sha256"]:
        raise ValueError(f"artifact {role}: SHA-256 mismatch")
    return path


def _assert_bound_artifact_pre_gold_safe(path: Path, *, role: str) -> None:
    """Reject hidden Gold fields in every bound pre-Gold tabular/JSON artifact."""

    lower_name = path.name.casefold()
    if lower_name.endswith(".csv") or lower_name.endswith(".csv.gz"):
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        forbidden = [column for column in columns if "gold" in str(column).casefold()]
        if forbidden:
            raise ValueError(f"artifact {role}: Gold columns are forbidden {forbidden}")
        return
    if lower_name.endswith(".json"):
        payload = _read_json(path, name=f"artifact {role}")

        allowed_false_keys = {
            "gold_outcomes_loaded",
            "gold58_labels_path_opened",
        }
        allowed_empty_keys = {
            "external_gold_columns",
            "gold_columns_loaded",
            "gold_files_loaded",
            "gold_outcome_columns_used",
            "is_gold_columns_loaded",
            "output_gold_columns",
        }
        allowed_identity_keys = {
            "gold_identity_used_for_external_routing_only": True,
        }
        allowed_nonnegative_count_keys = {"split_non_gold_studies"}
        allowed_sha256_keys = {"gold_gate_module"}

        def walk(value: Any, location: str) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    normalized = str(key).casefold()
                    if "gold" in normalized:
                        if normalized in allowed_false_keys and child is False:
                            continue
                        if normalized in allowed_empty_keys and child == []:
                            continue
                        if (
                            normalized in allowed_identity_keys
                            and child is allowed_identity_keys[normalized]
                        ):
                            continue
                        if (
                            normalized in allowed_nonnegative_count_keys
                            and isinstance(child, int)
                            and not isinstance(child, bool)
                            and child >= 0
                        ):
                            continue
                        if (
                            normalized in allowed_sha256_keys
                            and isinstance(child, str)
                            and len(child) == 64
                            and all(character in "0123456789abcdef" for character in child)
                        ):
                            continue
                        raise ValueError(
                            f"artifact {role}: Gold-bearing JSON key at {location}.{key}"
                        )
                    walk(child, f"{location}.{key}")
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    walk(child, f"{location}[{index}]")

        walk(payload, role)
        return
    raise ValueError(
        f"artifact {role}: unsupported pre-Gold artifact format {path.name}"
    )


def _validate_required_audits(
    paths: Mapping[str, Path],
    artifact_hashes: Mapping[str, str],
    code_bindings: Mapping[str, Any],
    *,
    spec: GoldGateSpec,
) -> None:
    if not set(spec.required_audit_statuses).issubset(spec.required_artifact_roles):
        raise ValueError("Gold gate required audit statuses reference unknown roles")
    filename_to_role = {
        filename: role for role, filename in spec.required_artifact_filenames.items()
    }
    if len(filename_to_role) != len(spec.required_artifact_filenames):
        raise ValueError("Gold gate required artifact filenames are not unique")

    def validate_hash_map(value: Any, *, audit_role: str, field: str) -> None:
        if not isinstance(value, dict):
            raise ValueError(f"artifact {audit_role}: {field} must be an object")
        for filename, digest in value.items():
            output_role = filename_to_role.get(str(filename))
            if output_role is None:
                raise ValueError(
                    f"artifact {audit_role}: {field} binds un-frozen output {filename}"
                )
            if str(digest) != artifact_hashes[output_role]:
                raise ValueError(
                    f"artifact {audit_role}: {field} hash drift for {filename}"
                )

    for role, expected_status in spec.required_audit_statuses.items():
        payload = _read_json(paths[role], name=f"artifact {role}")
        if payload.get("status") != expected_status:
            raise ValueError(f"artifact {role}: terminal status drift")
        if "contract_version" in payload and payload["contract_version"] != spec.contract_version:
            raise ValueError(f"artifact {role}: contract version drift")
        if (
            "contract_semantic_hash" in payload
            and payload["contract_semantic_hash"] != spec.contract_semantic_hash
        ):
            raise ValueError(f"artifact {role}: contract semantic hash drift")
        if "output_hashes" in payload:
            validate_hash_map(
                payload["output_hashes"], audit_role=role, field="output_hashes"
            )
        if "core_output_hashes" in payload:
            validate_hash_map(
                payload["core_output_hashes"],
                audit_role=role,
                field="core_output_hashes",
            )
        output = payload.get("output")
        if isinstance(output, dict) and {"path", "sha256"}.issubset(output):
            filename = Path(str(output["path"])).name
            output_role = filename_to_role.get(filename)
            if output_role is None or str(output["sha256"]) != artifact_hashes[output_role]:
                raise ValueError(f"artifact {role}: output binding drift")
        rerun = payload.get("deterministic_rerun")
        if isinstance(rerun, dict) and isinstance(rerun.get("hashes"), dict):
            validate_hash_map(
                rerun["hashes"], audit_role=role, field="deterministic_rerun.hashes"
            )

        if expected_status == (
            "PASS_INDEPENDENT_SEGMENT_EFFECTS_EXACT_TWO_RUNS_PENDING_LOPO_GOLD_NOT_LOADED"
        ):
            references = payload.get("reference_directories")
            if (
                not isinstance(references, list)
                or len(references) != 2
                or len(set(map(str, references))) != 2
            ):
                raise ValueError(
                    "independent segment audit must bind exactly two distinct reruns"
                )
            diagnostics = payload.get("segment_gate_diagnostics")
            if not isinstance(diagnostics, dict) or diagnostics.get("status") != (
                "BLOCKED_PRE_GOLD_CORE_SURVIVORS_REQUIRE_EXACT_LOPO"
            ):
                raise ValueError("independent segment audit pre-LOPO boundary drift")
            expected_counts = {
                "rows": 3048,
                "hypotheses": 127,
                "families": 24,
                "permutations_per_family": 999,
                "core_maxT_survivors": 3,
            }
            if any(diagnostics.get(key) != value for key, value in expected_counts.items()):
                raise ValueError("independent segment audit count/parameter drift")
            source_hashes = payload.get("source_hashes")
            expected_code_sources = {
                "segment_gate_module": "segment_gate_module",
                "segments_module": "segment_module",
                "segment_driver": "segment_driver",
                "independent_verifier": "segment_verifier_driver",
            }
            if not isinstance(source_hashes, dict) or any(
                source_hashes.get(source_key)
                != code_bindings[code_role]["sha256"]
                for source_key, code_role in expected_code_sources.items()
            ):
                raise ValueError("independent segment audit code-source binding drift")

        if expected_status == "PASS_INDEPENDENT_LOPO_OUTPUTS_GOLD_NOT_LOADED":
            if payload.get("lopo_audit_sha256") != artifact_hashes["segment_lopo_audit"]:
                raise ValueError("independent LOPO audit canonical-audit binding drift")
            rerun_sha = str(payload.get("deterministic_rerun_audit_sha256", ""))
            if len(rerun_sha) != 64 or any(
                character not in "0123456789abcdef" for character in rerun_sha
            ):
                raise ValueError("independent LOPO rerun-audit SHA-256 is invalid")
            if payload.get("deterministic_rerun", {}).get("status") != (
                "PASS_EXACT_CORE_OUTPUT_HASH_MATCH"
            ):
                raise ValueError("independent LOPO deterministic rerun status drift")
            expected_counts = {
                "core_survivors_input": 3,
                "required_acquisition_profiles": 14,
                "profile_effect_rows": 42,
                "supported_profile_effects": 0,
                "lopo_survivors_retained": 0,
                "lopo_survivors_demoted": 3,
                "lopo_survivors_unresolved": 0,
            }
            counts = payload.get("counts")
            if not isinstance(counts, dict) or any(
                counts.get(key) != value for key, value in expected_counts.items()
            ):
                raise ValueError("independent LOPO terminal count drift")
            source_hashes = payload.get("source_hashes")
            expected_code_sources = {
                "lopo_module": "lopo_module",
                "stats_module": "stats_module",
                "segments_module": "segment_module",
                "typo_module": "feature_module",
                "primary_driver": "driver",
                "segment_driver": "segment_driver",
                "lopo_driver": "lopo_driver",
                "independent_verifier": "lopo_verifier_driver",
            }
            if not isinstance(source_hashes, dict) or any(
                source_hashes.get(source_key)
                != code_bindings[code_role]["sha256"]
                for source_key, code_role in expected_code_sources.items()
            ):
                raise ValueError("independent LOPO audit code-source binding drift")
            if source_hashes.get("contract") != spec.expected_source_hashes["contract"]:
                raise ValueError("independent LOPO audit contract binding drift")
            if source_hashes.get("segment_universe") != artifact_hashes["segment_universe"]:
                raise ValueError("independent LOPO audit segment-universe binding drift")


def verify_proxy_freeze_manifest(
    root: str | Path,
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str,
    spec: GoldGateSpec = FROZEN_GOLD_GATE_SPEC,
) -> VerifiedProxyFreeze:
    """Verify the complete pre-Gold freeze without opening Gold labels."""

    root_path = Path(root).resolve()
    run_dir = _resolve_inside(root_path, spec.run_directory)
    manifest_file = _resolve_inside(
        root_path, str(manifest_path), boundary=run_dir
    )
    if (
        len(expected_manifest_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_manifest_sha256)
    ):
        raise ValueError("expected manifest SHA-256 must be 64 lowercase hex characters")
    observed_manifest_sha = sha256_file(manifest_file)
    if observed_manifest_sha != expected_manifest_sha256:
        raise ValueError("freeze manifest SHA-256 mismatch")
    manifest = _read_json(manifest_file, name="freeze manifest")
    expected_top = {
        "schema_version",
        "status",
        "created_at",
        "contract",
        "source_hashes",
        "code_hashes",
        "feature_sets",
        "feature_sets_sha256",
        "fold_map",
        "identity",
        "artifacts",
        "frozen_rules",
        "gold_source",
        "gold_outcomes_loaded",
    }
    _require_exact_keys(manifest, expected_top, name="freeze manifest")
    if manifest["schema_version"] != spec.schema_version:
        raise ValueError("freeze manifest schema version drift")
    if manifest["status"] != "FROZEN_PRE_GOLD_COMPLETE":
        raise ValueError("freeze manifest is not complete")
    if manifest["gold_outcomes_loaded"] is not False:
        raise ValueError("freeze manifest indicates Gold outcomes were loaded")

    expected_contract = {
        "path": spec.contract.path,
        "sha256": spec.contract.sha256,
        "version": spec.contract_version,
        "semantic_hash": spec.contract_semantic_hash,
    }
    if manifest["contract"] != expected_contract:
        raise ValueError("freeze manifest contract identity drift")
    contract_path = _resolve_inside(root_path, spec.contract.path)
    if sha256_file(contract_path) != spec.contract.sha256:
        raise ValueError("frozen contract bytes drift")
    if manifest["source_hashes"] != dict(spec.expected_source_hashes):
        raise ValueError("freeze manifest source hashes drift")
    if set(spec.source_files) != set(spec.expected_source_hashes):
        raise ValueError("Gold gate source-file bindings are incomplete")
    for role, binding in spec.source_files.items():
        if binding.sha256 != spec.expected_source_hashes[role]:
            raise ValueError(f"Gold gate source-file hash drift for {role}")
        source_path = _resolve_inside(root_path, binding.path)
        if not source_path.is_file() or sha256_file(source_path) != binding.sha256:
            raise ValueError(f"frozen source bytes drift for {role}")
    if manifest["fold_map"] != {
        "path": spec.fold_map.path,
        "sha256": spec.fold_map.sha256,
    }:
        raise ValueError("freeze manifest fold-map binding drift")
    fold_path = _resolve_inside(root_path, spec.fold_map.path)
    if sha256_file(fold_path) != spec.fold_map.sha256:
        raise ValueError("frozen fold-map bytes drift")
    if manifest["gold_source"] != {
        "path": spec.gold_source.path,
        "sha256": spec.gold_source.sha256,
    }:
        raise ValueError("freeze manifest Gold-source binding drift")
    # Deliberately do not resolve, hash, stat, or open gold_source here.

    if manifest["feature_sets_sha256"] != spec.feature_sets_sha256:
        raise ValueError("freeze manifest feature-set SHA drift")
    if _canonical_json_sha256(manifest["feature_sets"]) != spec.feature_sets_sha256:
        raise ValueError("freeze manifest feature-set payload mismatch")
    if manifest["frozen_rules"] != FREEZE_RULES:
        raise ValueError("freeze manifest frozen rules drift")

    code_hashes = manifest["code_hashes"]
    if not isinstance(code_hashes, dict):
        raise ValueError("freeze manifest code_hashes must be an object")
    _require_exact_keys(
        code_hashes, set(spec.required_code_roles), name="freeze manifest code_hashes"
    )
    if set(spec.required_code_paths) != set(spec.required_code_roles):
        raise ValueError("Gold gate required code paths are incomplete")
    for role, binding in code_hashes.items():
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ValueError(f"code {role}: exact path/sha256 binding required")
        if str(binding["path"]) != spec.required_code_paths[role]:
            raise ValueError(f"code {role}: frozen path drift")
        path = _resolve_inside(root_path, str(binding["path"]))
        if not path.is_file() or sha256_file(path) != binding["sha256"]:
            raise ValueError(f"code {role}: missing or SHA-256 mismatch")

    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, dict):
        raise ValueError("freeze manifest artifacts must be an object")
    _require_exact_keys(
        artifacts,
        set(spec.required_artifact_roles),
        name="freeze manifest artifacts",
    )
    if set(spec.required_artifact_filenames) != set(spec.required_artifact_roles):
        raise ValueError("Gold gate required artifact filenames are incomplete")
    paths: dict[str, Path] = {}
    artifact_hashes: dict[str, str] = {}
    seen_paths: set[Path] = set()
    for role, binding in artifacts.items():
        path = _artifact_binding_path(root_path, run_dir, binding, role=role)
        if path.name != spec.required_artifact_filenames[role]:
            raise ValueError(f"artifact {role}: frozen filename drift")
        if path in seen_paths:
            raise ValueError("freeze manifest binds two roles to the same artifact")
        seen_paths.add(path)
        paths[role] = path
        artifact_hashes[role] = str(binding["sha256"])
        _assert_bound_artifact_pre_gold_safe(path, role=role)

    _validate_required_audits(paths, artifact_hashes, code_hashes, spec=spec)

    proxy = _read_csv_allowlist(
        paths["proxy_oof_scores"],
        PROXY_REQUIRED_COLUMNS,
        name="proxy_oof_scores",
    )
    proxy = _validate_proxy_oof(proxy, spec)
    external = _read_csv_allowlist(
        paths["final_external_predictions"],
        EXTERNAL_REQUIRED_COLUMNS,
        name="final_external_predictions",
    )
    external = _validate_external_predictions(external, spec)
    for role, frame in (
        ("proxy_oof_scores", proxy),
        ("final_external_predictions", external),
    ):
        binding = artifacts[role]
        if binding.get("rows") != len(frame):
            raise ValueError(f"artifact {role}: manifest row count drift")
        if binding.get("columns_sha256") != _columns_sha(frame.columns):
            raise ValueError(f"artifact {role}: column-order SHA drift")

    proxy_uids = sorted(proxy[ID_COLUMN].astype(str).unique())
    proxy_groups = sorted(proxy["report_group"].astype(str).unique())
    external_uids = sorted(external[ID_COLUMN].astype(str).unique())
    if set(proxy_uids).intersection(external_uids):
        raise ValueError("proxy and external UID cohorts overlap")

    identity_path = _resolve_inside(root_path, spec.canonical_identity_manifest.path)
    if sha256_file(identity_path) != spec.canonical_identity_manifest.sha256:
        raise ValueError("canonical identity manifest SHA-256 mismatch")
    identity_frame = pd.read_csv(identity_path, usecols=[ID_COLUMN, "is_gold"])
    if identity_frame[ID_COLUMN].duplicated().any():
        raise ValueError("canonical identity manifest has duplicate UIDs")
    identity_frame[ID_COLUMN] = identity_frame[ID_COLUMN].astype(str)
    if identity_frame["is_gold"].isna().any():
        raise ValueError("canonical identity manifest has null is_gold values")
    if pd.api.types.is_bool_dtype(identity_frame["is_gold"]):
        gold_mask = identity_frame["is_gold"].astype(bool)
    else:
        normalized_gold = (
            identity_frame["is_gold"].astype(str).str.strip().str.casefold()
        )
        if not normalized_gold.isin({"true", "false"}).all():
            raise ValueError("canonical identity manifest has invalid is_gold values")
        gold_mask = normalized_gold.eq("true")
    gold_identity = set(identity_frame.loc[gold_mask, ID_COLUMN])
    external_set = set(external_uids)
    if not gold_identity.issubset(external_set):
        raise ValueError("external predictions do not cover all frozen Gold identities")
    external_gold = len(external_set.intersection(gold_identity))
    external_non_gold = len(external_set.difference(gold_identity))
    if (external_gold, external_non_gold) != (
        spec.expected_external_gold_identities,
        spec.expected_external_non_gold,
    ):
        raise ValueError("external Gold/non-Gold identity coverage drift")

    expected_identity = {
        "eligible_studies": spec.expected_proxy_studies,
        "eligible_report_groups": spec.expected_proxy_report_groups,
        "proxy_decisive_cells": spec.expected_proxy_rows,
        "eligible_uid_sha256": _line_sha(proxy_uids),
        "eligible_report_group_sha256": _line_sha(proxy_groups),
        "eligible_cell_sha256": _cell_sha(proxy),
        "external_studies": spec.expected_external_studies,
        "external_cells": spec.expected_external_rows,
        "external_uid_sha256": _line_sha(external_uids),
        "external_cell_sha256": _cell_sha(external),
        "external_gold_identity_studies": external_gold,
        "external_non_gold_studies": external_non_gold,
    }
    if manifest["identity"] != expected_identity:
        raise ValueError("freeze manifest cohort identity hashes/counts drift")

    return VerifiedProxyFreeze(
        manifest=manifest,
        manifest_sha256=observed_manifest_sha,
        proxy_oof=proxy,
        external_predictions=external,
        artifact_hashes=artifact_hashes,
    )


def _load_gold_after_gate(
    root: Path,
    *,
    spec: GoldGateSpec,
    gold_reader: Callable[..., pd.DataFrame],
) -> pd.DataFrame:
    gold_path = _resolve_inside(root, spec.gold_source.path)
    if sha256_file(gold_path) != spec.gold_source.sha256:
        raise ValueError("Gold source SHA-256 mismatch after freeze gate")
    wide = gold_reader(gold_path, usecols=[ID_COLUMN, *spec.targets])
    if wide[ID_COLUMN].isna().any() or wide[ID_COLUMN].duplicated().any():
        raise ValueError("Gold source UID coverage is invalid")
    wide[ID_COLUMN] = wide[ID_COLUMN].astype(str)
    if len(wide) != spec.expected_external_gold_identities:
        raise ValueError("Gold source study count drift")
    long = wide.melt(
        id_vars=ID_COLUMN,
        value_vars=list(spec.targets),
        var_name="target",
        value_name="gold",
    )
    numeric = pd.to_numeric(long["gold"], errors="coerce")
    if numeric.isna().any() or not numeric.isin({0, 1}).all():
        raise ValueError("Gold source must contain exact binary 0/1 targets")
    long["gold"] = numeric.astype(np.int8)
    return long


def _gold_codes(row: pd.Series) -> tuple[str, ...]:
    codes: list[str] = []
    state = str(row["teacher_state"])
    gold = int(row["gold"])
    image_state = str(row["image_state"])
    if state == "NEGATIVE" and gold == 1:
        codes.append("TEACHER_NEGATIVE_GOLD_POSITIVE")
    if state == "POSITIVE" and gold == 0:
        codes.append("TEACHER_POSITIVE_GOLD_NEGATIVE")
    if gold == 0 and image_state == "BOTH_HIGH":
        codes.append("GOLD_NEGATIVE_IMAGE_CONSENSUS_HIGH")
    if gold == 1 and image_state == "BOTH_LOW":
        codes.append("GOLD_POSITIVE_IMAGE_CONSENSUS_LOW")
    return tuple(codes)


def evaluate_gold_after_proxy_freeze(
    root: str | Path,
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str,
    spec: GoldGateSpec = FROZEN_GOLD_GATE_SPEC,
    gold_reader: Callable[..., pd.DataFrame] = pd.read_csv,
) -> GoldEvaluationResult:
    """Open Gold only after an exact proxy freeze and append neutral evidence."""

    root_path = Path(root).resolve()
    verified = verify_proxy_freeze_manifest(
        root_path,
        manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
        spec=spec,
    )
    # This is the first operation in the entire path allowed to touch Gold bytes.
    gold = _load_gold_after_gate(root_path, spec=spec, gold_reader=gold_reader)
    external = verified.external_predictions.copy(deep=True)
    frozen_columns = external.columns.tolist()
    panel = external.merge(gold, on=list(KEYS), how="inner", validate="one_to_one")
    expected_gold_cells = spec.expected_external_gold_identities * len(spec.targets)
    if len(panel) != expected_gold_cells or panel[ID_COLUMN].nunique() != spec.expected_external_gold_identities:
        raise ValueError("frozen external predictions do not exactly cover Gold58")

    codes = panel.apply(_gold_codes, axis=1)
    panel["gold_descriptive_reason_codes"] = ["|".join(value) for value in codes]
    panel["gold_descriptive_evidence_tier"] = np.where(
        pd.Series(codes, index=panel.index).map(bool),
        "GOLD_ANCHORED",
        "UNRESOLVED",
    )
    mismatch = pd.Series(pd.NA, index=panel.index, dtype="boolean")
    mismatch.loc[panel["teacher_state"].eq("NEGATIVE")] = panel.loc[
        panel["teacher_state"].eq("NEGATIVE"), "gold"
    ].eq(1)
    mismatch.loc[panel["teacher_state"].eq("POSITIVE")] = panel.loc[
        panel["teacher_state"].eq("POSITIVE"), "gold"
    ].eq(0)
    panel["teacher_reference_disagreement"] = mismatch
    panel["attribution_status"] = (
        "UNRESOLVED_WITHOUT_BLINDED_MANUAL_REPORT_IMAGE_ADJUDICATION"
    )
    panel["automatic_label_or_weight_action"] = "NONE_FORBIDDEN"

    # The merge selected rows but may not mutate, reorder, or replace any frozen
    # pre-Gold field.  Compare by exact key after Gold annotations are appended.
    expected_frozen = external.loc[
        external[ID_COLUMN].isin(set(gold[ID_COLUMN])), frozen_columns
    ].sort_values(list(KEYS), kind="mergesort").reset_index(drop=True)
    observed_frozen = panel.loc[:, frozen_columns].sort_values(
        list(KEYS), kind="mergesort"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        expected_frozen,
        observed_frozen,
        check_dtype=True,
        check_exact=True,
    )

    assessable = panel["teacher_reference_disagreement"].notna()
    target_support: dict[str, Any] = {}
    for target in spec.targets:
        part = panel.loc[panel["target"].eq(target) & assessable]
        errors = int(part["teacher_reference_disagreement"].sum())
        agreements = int(len(part) - errors)
        target_support[target] = {
            "teacher_reference_disagreements": errors,
            "teacher_reference_agreements": agreements,
            "target_metric_allowed": bool(errors >= 5 and agreements >= 5),
        }
    code_counts = {
        code: int(sum(code in row_codes for row_codes in codes))
        for code in ALLOWED_GOLD_CODES
    }
    summary: dict[str, Any] = {
        "status": "PASS_POST_FREEZE_GOLD_DESCRIPTIVE_ONLY",
        "freeze_manifest_sha256": verified.manifest_sha256,
        "gold_studies": panel[ID_COLUMN].nunique(),
        "gold_cells": len(panel),
        "assessable_teacher_cells": int(assessable.sum()),
        "gold_descriptive_code_counts": code_counts,
        "target_metric_support": target_support,
        "semantic_channels": {
            "radiologist_review_proxy": "REVIEW_PRIORITY_NOT_CONFIRMED_REPORT_ERROR",
            "llm_parsing": "LLM_STATE_CONFLICT_NOT_PROVEN_PARSING_ERROR",
            "image_uncertainty": "IMAGE_UNCERTAINTY_NOT_REPORT_TRUTH",
        },
        "automatic_label_or_weight_change": False,
        "selection_or_rank_change": False,
        "confirmation_boundary": (
            "NO_CONFIRMED_REPORT_ERROR_WITHOUT_BLINDED_MANUAL_REPORT_IMAGE_ADJUDICATION"
        ),
    }
    return GoldEvaluationResult(
        panel=panel.sort_values(list(KEYS), kind="mergesort").reset_index(drop=True),
        summary=summary,
        freeze_manifest_sha256=verified.manifest_sha256,
    )


__all__ = [
    "ALLOWED_GOLD_CODES",
    "CANDIDATE_WEIGHT_STATUS",
    "EXTERNAL_REQUIRED_COLUMNS",
    "FREEZE_RULES",
    "FROZEN_GOLD_GATE_SPEC",
    "GoldEvaluationResult",
    "GoldGateSpec",
    "VerifiedProxyFreeze",
    "evaluate_gold_after_proxy_freeze",
    "verify_proxy_freeze_manifest",
]
