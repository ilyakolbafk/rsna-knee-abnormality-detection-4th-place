"""Outcome-blind report-family reliability features.

The helpers in this module deliberately operate only on frozen report-teacher
states and the frozen final teacher.  They neither load nor accept Gold labels,
image scores, error columns, or report text.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .constants import ID_COLUMN, TARGET_COLUMNS


EFFECTIVE_FAMILIES: tuple[str, ...] = (
    "Laymond Qwen3-8B",
    "Steven v2",
    "Lixin GPT-5.6-Sol",
    "Pilkwang LLM v2",
    "Yunus Gemini Flash",
)

CANONICAL_FAMILY_STATES: tuple[str, ...] = (
    "DIRECT_POSITIVE",
    "DIRECT_NEGATIVE",
    "UNCERTAIN",
    "UNMENTIONED",
    "NONDIRECT_OR_UNKNOWN",
    "MISSING",
)

SOURCE_REQUIRED_COLUMNS: tuple[str, ...] = (
    ID_COLUMN,
    "target",
    "source",
    "source_state",
    "mention_state",
    "direct_binary_vote",
    "directness",
    "raw_score",
)

SOURCE_OUTPUT_COLUMNS: tuple[str, ...] = (
    ID_COLUMN,
    "target",
    "source",
    "source_state",
    "mention_state",
    "binary_vote",
    "direct_binary_vote",
    "raw_score",
    "derived_score",
    "source_confidence",
    "directness",
    "state_quality",
    "semantic_provenance",
    "canonical_family_state",
)

_FORBIDDEN_EXACT = frozenset({"gold", "is_gold", "outcome", "label_error"})
_FORBIDDEN_FRAGMENTS = (
    "_abs_gap",
    "abs_gap_",
    "_error_vs_gold",
    "_hard_error",
    "error_vs_gold",
)

_DIRECT_POSITIVE_SOURCE_STATES = frozenset({"POSITIVE", "SCORE_POSITIVE"})
_DIRECT_NEGATIVE_SOURCE_STATES = frozenset({"NEGATIVE", "SCORE_NEGATIVE"})
_UNCERTAIN_SOURCE_STATES = frozenset({"UNCERTAIN"})
_UNMENTIONED_SOURCE_STATES = frozenset({"UNMENTIONED"})
_MISSING_SOURCE_STATES = frozenset({"MISSING"})

_SOURCE_RECODE_TABLE: dict[
    tuple[str, str, str, str, int | None], str
] = {
    ("LAYMOND QWEN3-8B", "POSITIVE", "MENTIONED", "DIRECT_REPORT_EXTRACTION", 1): "DIRECT_POSITIVE",
    ("LAYMOND QWEN3-8B", "NEGATIVE", "MENTIONED", "DIRECT_REPORT_EXTRACTION", 0): "DIRECT_NEGATIVE",
    ("LAYMOND QWEN3-8B", "UNMENTIONED", "UNMENTIONED", "NO_DECISIVE_DIRECT_LABEL", None): "UNMENTIONED",
    ("LAYMOND QWEN3-8B", "MENTIONED_NONPOSITIVE", "MENTIONED", "NO_DECISIVE_DIRECT_LABEL", None): "NONDIRECT_OR_UNKNOWN",
    ("LAYMOND QWEN3-8B", "MISSING", "MISSING", "DIRECT_REPORT_EXTRACTION", None): "MISSING",
    ("STEVEN V2", "POSITIVE", "MENTIONED", "DIRECT_REPORT_EXTRACTION", 1): "DIRECT_POSITIVE",
    ("STEVEN V2", "NEGATIVE", "MENTIONED", "DIRECT_REPORT_EXTRACTION", 0): "DIRECT_NEGATIVE",
    ("STEVEN V2", "UNMENTIONED", "UNMENTIONED", "NO_DECISIVE_DIRECT_LABEL", None): "UNMENTIONED",
    ("STEVEN V2", "UNMENTIONED", "UNMENTIONED", "DERIVED_FROM_EFFUSION", None): "NONDIRECT_OR_UNKNOWN",
    ("LIXIN GPT-5.6-SOL", "SCORE_POSITIVE", "UNKNOWN", "STATE_UNKNOWN", 1): "NONDIRECT_OR_UNKNOWN",
    ("LIXIN GPT-5.6-SOL", "SCORE_NEGATIVE", "UNKNOWN", "STATE_UNKNOWN", 0): "NONDIRECT_OR_UNKNOWN",
    ("LIXIN GPT-5.6-SOL", "MIDPOINT_UNKNOWN", "UNKNOWN", "STATE_UNKNOWN", None): "NONDIRECT_OR_UNKNOWN",
    ("PILKWANG LLM V2", "POSITIVE", "MENTIONED", "DIRECT_REPORT_EXTRACTION", 1): "DIRECT_POSITIVE",
    ("PILKWANG LLM V2", "NEGATIVE", "MENTIONED", "DIRECT_REPORT_EXTRACTION", 0): "DIRECT_NEGATIVE",
    ("PILKWANG LLM V2", "UNMENTIONED_OR_UNCERTAIN", "UNKNOWN", "NO_DECISIVE_DIRECT_LABEL", None): "NONDIRECT_OR_UNKNOWN",
    ("PILKWANG LLM V2", "MISSING", "MISSING", "MISSING", None): "MISSING",
    ("YUNUS GEMINI FLASH", "POSITIVE", "MENTIONED", "DIRECT_REPORT_EXTRACTION", 1): "DIRECT_POSITIVE",
    ("YUNUS GEMINI FLASH", "NEGATIVE", "MENTIONED", "DIRECT_REPORT_EXTRACTION", 0): "DIRECT_NEGATIVE",
    ("YUNUS GEMINI FLASH", "UNCERTAIN", "MENTIONED", "DIRECT_REPORT_EXTRACTION", None): "UNCERTAIN",
    ("YUNUS GEMINI FLASH", "UNMENTIONED", "UNMENTIONED", "NO_DECISIVE_DIRECT_LABEL", None): "UNMENTIONED",
    ("YUNUS GEMINI FLASH", "STATE_UNKNOWN", "MENTIONED", "STATE_UNKNOWN", None): "NONDIRECT_OR_UNKNOWN",
}

_TEACHER_COMPONENT_FAMILIES = frozenset(
    {"Steven v2", "Lixin GPT-5.6-Sol", "Pilkwang LLM v2"}
)
_EXTERNAL_AUDIT_FAMILIES: tuple[str, str] = (
    "Laymond Qwen3-8B",
    "Yunus Gemini Flash",
)


def _assert_outcome_free(frame: pd.DataFrame, *, name: str) -> None:
    forbidden = []
    for column in frame.columns:
        normalized = str(column).strip().casefold()
        if normalized in _FORBIDDEN_EXACT or any(
            fragment in normalized for fragment in _FORBIDDEN_FRAGMENTS
        ):
            forbidden.append(str(column))
    if forbidden:
        raise ValueError(f"{name}: outcome columns are forbidden: {sorted(forbidden)}")


def _normalized_token(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().upper()


def _normalized_vote(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    vote = float(value)
    if not np.isfinite(vote) or vote not in {0.0, 1.0}:
        raise ValueError(f"direct_binary_vote must be 0, 1, or null; got {value!r}")
    return int(vote)


def canonicalize_source_state(
    source_state: object,
    mention_state: object,
    direct_binary_vote: object,
    *,
    source: object | None = None,
    directness: object | None = None,
) -> str:
    """Map a source-specific state to the frozen six-state taxonomy.

    A non-null ``direct_binary_vote`` is the source registry's explicit
    declaration that the family contributes a direct binary direction.  Silence,
    uncertainty, and source-specific midpoint/partial states remain abstentions.
    """

    state = _normalized_token(source_state)
    mention = _normalized_token(mention_state)
    vote = _normalized_vote(direct_binary_vote)

    if source is not None:
        key = (
            _normalized_token(source),
            state,
            mention,
            _normalized_token(directness),
            vote,
        )
        try:
            return _SOURCE_RECODE_TABLE[key]
        except KeyError as error:
            raise ValueError(f"Unmatched source-state recode combination: {key!r}") from error

    if state in _MISSING_SOURCE_STATES or mention == "MISSING" or not state:
        if vote is not None:
            raise ValueError("A missing source state cannot have a direct vote")
        return "MISSING"

    if vote is not None:
        if mention == "UNMENTIONED":
            raise ValueError("An unmentioned source state cannot have a direct vote")
        if vote == 1 and state in _DIRECT_NEGATIVE_SOURCE_STATES:
            raise ValueError("Positive direct vote conflicts with a negative source state")
        if vote == 0 and state in _DIRECT_POSITIVE_SOURCE_STATES:
            raise ValueError("Negative direct vote conflicts with a positive source state")
        return "DIRECT_POSITIVE" if vote == 1 else "DIRECT_NEGATIVE"

    if state in _UNCERTAIN_SOURCE_STATES:
        return "UNCERTAIN"
    if state in _UNMENTIONED_SOURCE_STATES or mention == "UNMENTIONED":
        return "UNMENTIONED"
    return "NONDIRECT_OR_UNKNOWN"


def _validate_order(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    normalized = tuple(str(value) for value in values)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must be non-empty and unique")
    return normalized


def build_family_state_panel(
    source_states: pd.DataFrame,
    *,
    target_order: Sequence[str] = TARGET_COLUMNS,
    effective_families: Sequence[str] = EFFECTIVE_FAMILIES,
) -> pd.DataFrame:
    """Validate and canonicalize exactly one row for each cell and family."""

    _assert_outcome_free(source_states, name="source_states")
    targets = _validate_order(target_order, name="target_order")
    families = _validate_order(effective_families, name="effective_families")
    if len(families) != 5:
        raise ValueError(f"Exactly five effective families are required; got {len(families)}")

    missing = sorted(set(SOURCE_REQUIRED_COLUMNS).difference(source_states.columns))
    if missing:
        raise ValueError(f"source_states: missing required columns {missing}")

    panel = source_states.copy()
    for column in (ID_COLUMN, "target", "source"):
        if panel[column].isna().any():
            raise ValueError(f"source_states: {column} contains null values")
        panel[column] = panel[column].astype(str)
        if panel[column].str.strip().eq("").any():
            raise ValueError(f"source_states: {column} contains empty values")

    if panel.duplicated([ID_COLUMN, "target", "source"]).any():
        raise ValueError("source_states: duplicate StudyInstanceUID x target x source rows")
    if set(panel["target"]) != set(targets):
        raise ValueError("source_states: target set does not match target_order")
    if set(panel["source"]) != set(families):
        raise ValueError("source_states: source set does not match the five-family registry")

    cell_sizes = panel.groupby([ID_COLUMN, "target"], sort=False)["source"].nunique()
    if not cell_sizes.eq(5).all():
        raise ValueError("source_states: every StudyInstanceUID x target must contain five families")
    study_target_counts = panel.groupby(ID_COLUMN, sort=False)["target"].nunique()
    if not study_target_counts.eq(len(targets)).all():
        raise ValueError("source_states: every study must contain every target")
    expected_rows = panel[ID_COLUMN].nunique() * len(targets) * 5
    if len(panel) != expected_rows:
        raise ValueError(f"source_states: expected {expected_rows} rows, got {len(panel)}")

    panel["canonical_family_state"] = [
        canonicalize_source_state(
            state,
            mention,
            vote,
            source=source,
            directness=directness,
        )
        for source, state, mention, vote, directness in zip(
            panel["source"],
            panel["source_state"],
            panel["mention_state"],
            panel["direct_binary_vote"],
            panel["directness"],
            strict=True,
        )
    ]

    target_rank = {target: index for index, target in enumerate(targets)}
    family_rank = {family: index for index, family in enumerate(families)}
    panel = panel.assign(
        _target_rank=panel["target"].map(target_rank),
        _family_rank=panel["source"].map(family_rank),
    ).sort_values([ID_COLUMN, "_target_rank", "_family_rank"], kind="mergesort")
    panel = panel.drop(columns=["_target_rank", "_family_rank"]).reset_index(drop=True)
    return panel[[column for column in SOURCE_OUTPUT_COLUMNS if column in panel.columns]]


def reshape_final_teacher(
    final_teacher_wide: pd.DataFrame,
    *,
    target_order: Sequence[str] = TARGET_COLUMNS,
) -> pd.DataFrame:
    """Convert the frozen wide teacher table to an outcome-free long table."""

    _assert_outcome_free(final_teacher_wide, name="final_teacher_wide")
    targets = _validate_order(target_order, name="target_order")
    required = {ID_COLUMN, *targets}
    missing = sorted(required.difference(final_teacher_wide.columns))
    if missing:
        raise ValueError(f"final_teacher_wide: missing required columns {missing}")
    if final_teacher_wide[ID_COLUMN].isna().any() or final_teacher_wide[ID_COLUMN].duplicated().any():
        raise ValueError(f"final_teacher_wide: {ID_COLUMN} must be non-null and unique")

    teacher = final_teacher_wide.copy()
    teacher[ID_COLUMN] = teacher[ID_COLUMN].astype(str)
    optional_prefixes = {
        "confidence": "teacher_confidence",
        "teacher_count": "teacher_count",
        "agreement": "teacher_agreement",
        "certainty": "teacher_certainty",
        "evidence_gate": "teacher_evidence_gate",
    }
    present_optional: dict[str, str] = {}
    for prefix, output_name in optional_prefixes.items():
        columns = [f"{prefix}__{target}" for target in targets]
        present = [column in teacher.columns for column in columns]
        if any(present) and not all(present):
            raise ValueError(f"final_teacher_wide: partial {prefix} target columns")
        if all(present):
            present_optional[prefix] = output_name

    rows: list[pd.DataFrame] = []
    for target in targets:
        columns = [ID_COLUMN, target]
        rename = {target: "teacher_score"}
        for prefix, output_name in present_optional.items():
            column = f"{prefix}__{target}"
            columns.append(column)
            rename[column] = output_name
        part = teacher[columns].rename(columns=rename).copy()
        part.insert(1, "target", target)
        rows.append(part)
    long = pd.concat(rows, ignore_index=True)

    numeric_columns = [column for column in long.columns if column not in {ID_COLUMN, "target"}]
    for column in numeric_columns:
        values = pd.to_numeric(long[column], errors="raise").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"final_teacher_wide: {column} must be finite")
        long[column] = values
    if not long["teacher_score"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError("final_teacher_wide: teacher scores must be in [0, 1]")

    scores = long["teacher_score"].to_numpy(float)
    long["teacher_state"] = np.select(
        [scores < 0.5 - 1e-12, scores > 0.5 + 1e-12],
        ["NEGATIVE", "POSITIVE"],
        default="ABSTAIN",
    )
    target_rank = {target: index for index, target in enumerate(targets)}
    long = long.assign(_target_rank=long["target"].map(target_rank)).sort_values(
        [ID_COLUMN, "_target_rank"], kind="mergesort"
    )
    return long.drop(columns="_target_rank").reset_index(drop=True)


def _safe_fraction(numerator: pd.Series, denominator: pd.Series) -> np.ndarray:
    num = numerator.to_numpy(float)
    den = denominator.to_numpy(float)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def build_report_reliability_features(
    source_states: pd.DataFrame,
    final_teacher_wide: pd.DataFrame,
    *,
    target_order: Sequence[str] = TARGET_COLUMNS,
    effective_families: Sequence[str] = EFFECTIVE_FAMILIES,
    minimum_direct_families: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build five-family and leave-current-target-out reliability features.

    Returns ``(family_panel, feature_panel, diagnostics)``.  The feature panel
    has one row per study-target cell and contains no report text, Gold, image
    score, outcome, or error field.
    """

    targets = _validate_order(target_order, name="target_order")
    families = _validate_order(effective_families, name="effective_families")
    if not 1 <= int(minimum_direct_families) <= 5:
        raise ValueError("minimum_direct_families must be between 1 and 5")

    family_panel = build_family_state_panel(
        source_states,
        target_order=targets,
        effective_families=families,
    )
    teacher = reshape_final_teacher(final_teacher_wide, target_order=targets)
    if set(family_panel[ID_COLUMN]) != set(teacher[ID_COLUMN]):
        raise ValueError("Teacher and family state study sets differ")

    keys = [ID_COLUMN, "target"]
    counts = (
        family_panel.groupby(keys, sort=False)["canonical_family_state"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=CANONICAL_FAMILY_STATES, fill_value=0)
        .reset_index()
    )
    counts = counts.rename(
        columns={state: f"family_{state.casefold()}_count" for state in CANONICAL_FAMILY_STATES}
    )
    pos = "family_direct_positive_count"
    neg = "family_direct_negative_count"
    uncertain = "family_uncertain_count"
    unmentioned = "family_unmentioned_count"
    unknown = "family_nondirect_or_unknown_count"
    missing = "family_missing_count"
    counts["family_available_count"] = 5 - counts[missing]
    counts["family_decisive_direct_count"] = counts[pos] + counts[neg]
    counts["family_abstain_count"] = counts[uncertain] + counts[unmentioned] + counts[unknown] + counts[missing]
    counts["family_direct_conflict"] = counts[pos].gt(0) & counts[neg].gt(0)
    counts["family_silence_instability"] = counts[unmentioned].gt(0) & counts["family_decisive_direct_count"].gt(0)
    counts["family_direct_majority_margin"] = (counts[pos] - counts[neg]).abs()
    counts["family_direct_majority_fraction"] = _safe_fraction(
        counts[[pos, neg]].max(axis=1), counts["family_decisive_direct_count"]
    )
    counts["family_direct_consensus_eligible"] = counts["family_decisive_direct_count"].ge(
        int(minimum_direct_families)
    )
    counts["family_direct_majority_state"] = np.select(
        [counts[pos].gt(counts[neg]), counts[neg].gt(counts[pos])],
        ["POSITIVE", "NEGATIVE"],
        default="ABSTAIN",
    )
    counts["family_direct_consensus_state"] = np.where(
        counts["family_direct_consensus_eligible"],
        counts["family_direct_majority_state"],
        "ABSTAIN",
    )
    counts["family_state_summary"] = [
        f"DP{dp}|DN{dn}|U{u}|UM{um}|N{n}|M{m}"
        for dp, dn, u, um, n, m in zip(
            counts[pos],
            counts[neg],
            counts[uncertain],
            counts[unmentioned],
            counts[unknown],
            counts[missing],
            strict=True,
        )
    ]

    nonmissing = family_panel[family_panel["canonical_family_state"].ne("MISSING")]
    unique_states = nonmissing.groupby(keys, sort=False)["canonical_family_state"].nunique()
    counts = counts.merge(
        unique_states.rename("family_unique_nonmissing_state_count").reset_index(),
        on=keys,
        how="left",
        validate="one_to_one",
    )
    counts["family_unique_nonmissing_state_count"] = counts[
        "family_unique_nonmissing_state_count"
    ].fillna(0).astype(int)
    counts["current_canonical_state_diversity"] = counts[
        "family_unique_nonmissing_state_count"
    ].gt(1)

    teacher_components = family_panel[
        family_panel["source"].isin(_TEACHER_COMPONENT_FAMILIES)
    ].copy()
    if set(teacher_components["source"]) != set(_TEACHER_COMPONENT_FAMILIES):
        raise ValueError("Teacher component families are incomplete")
    raw_component = pd.to_numeric(teacher_components["raw_score"], errors="coerce")
    teacher_components["_component_direction"] = np.select(
        [raw_component.lt(0.5), raw_component.gt(0.5), raw_component.eq(0.5)],
        ["NEGATIVE", "POSITIVE", "ABSTAIN"],
        default="MISSING",
    )
    component_counts = (
        teacher_components.groupby(keys, sort=False)["_component_direction"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=["POSITIVE", "NEGATIVE", "ABSTAIN", "MISSING"], fill_value=0)
        .rename(
            columns={
                "POSITIVE": "teacher_component_positive_count",
                "NEGATIVE": "teacher_component_negative_count",
                "ABSTAIN": "teacher_component_abstain_count",
                "MISSING": "teacher_component_missing_count",
            }
        )
        .reset_index()
    )
    component_counts["teacher_component_self_conflict"] = (
        component_counts["teacher_component_positive_count"].gt(0)
        & component_counts["teacher_component_negative_count"].gt(0)
    )
    counts = counts.merge(component_counts, on=keys, how="left", validate="one_to_one")

    external = family_panel[
        family_panel["source"].isin(_EXTERNAL_AUDIT_FAMILIES)
    ].pivot(index=keys, columns="source", values="canonical_family_state")
    external = external.reindex(columns=list(_EXTERNAL_AUDIT_FAMILIES))
    if external.isna().any().any():
        raise ValueError("External audit family states are incomplete")
    laymond = external[_EXTERNAL_AUDIT_FAMILIES[0]]
    yunus = external[_EXTERNAL_AUDIT_FAMILIES[1]]
    both_positive = laymond.eq("DIRECT_POSITIVE") & yunus.eq("DIRECT_POSITIVE")
    both_negative = laymond.eq("DIRECT_NEGATIVE") & yunus.eq("DIRECT_NEGATIVE")
    external["external_audit_consensus_state"] = np.select(
        [both_positive, both_negative], ["POSITIVE", "NEGATIVE"], default="ABSTAIN"
    )
    external["external_audit_direct_conflict"] = (
        (laymond.eq("DIRECT_POSITIVE") & yunus.eq("DIRECT_NEGATIVE"))
        | (laymond.eq("DIRECT_NEGATIVE") & yunus.eq("DIRECT_POSITIVE"))
    )
    external["external_audit_direct_support_count"] = (
        laymond.isin(["DIRECT_POSITIVE", "DIRECT_NEGATIVE"]).astype(int)
        + yunus.isin(["DIRECT_POSITIVE", "DIRECT_NEGATIVE"]).astype(int)
    )
    external = external[
        [
            "external_audit_consensus_state",
            "external_audit_direct_conflict",
            "external_audit_direct_support_count",
        ]
    ].reset_index()
    counts = counts.merge(external, on=keys, how="left", validate="one_to_one")

    features = teacher.merge(counts, on=keys, how="left", validate="one_to_one")
    if features.isna().any().any():
        missing_columns = features.columns[features.isna().any()].tolist()
        raise ValueError(f"Reliability feature merge produced nulls in {missing_columns}")

    features["teacher_family_support_count"] = np.select(
        [features["teacher_state"].eq("POSITIVE"), features["teacher_state"].eq("NEGATIVE")],
        [features[pos], features[neg]],
        default=0,
    ).astype(int)
    features["teacher_family_opposition_count"] = np.select(
        [features["teacher_state"].eq("POSITIVE"), features["teacher_state"].eq("NEGATIVE")],
        [features[neg], features[pos]],
        default=0,
    ).astype(int)
    features["teacher_family_support_fraction_decisive"] = _safe_fraction(
        features["teacher_family_support_count"], features["family_decisive_direct_count"]
    )
    teacher_decisive = features["teacher_state"].isin(["POSITIVE", "NEGATIVE"])
    consensus_decisive = features["family_direct_consensus_state"].isin(["POSITIVE", "NEGATIVE"])
    features["teacher_consensus_aggregation_conflict"] = (
        teacher_decisive
        & consensus_decisive
        & features["teacher_state"].ne(features["family_direct_consensus_state"])
    )
    features["teacher_family_comparison_state"] = np.select(
        [
            ~teacher_decisive,
            ~consensus_decisive,
            features["teacher_consensus_aggregation_conflict"],
        ],
        ["TEACHER_ABSTAIN", "DIRECT_CONSENSUS_INELIGIBLE", "CONFLICT"],
        default="MATCH",
    )
    external_decisive = features["external_audit_consensus_state"].isin(
        ["POSITIVE", "NEGATIVE"]
    )
    features["external_audit_disagreement"] = (
        teacher_decisive
        & external_decisive
        & features["teacher_state"].ne(features["external_audit_consensus_state"])
    )

    teacher_positive = features["teacher_state"].eq("POSITIVE").astype(int)
    teacher_negative = features["teacher_state"].eq("NEGATIVE").astype(int)
    teacher_abstain = features["teacher_state"].eq("ABSTAIN").astype(int)
    features["other_positive_count"] = (
        teacher_positive.groupby(features[ID_COLUMN]).transform("sum") - teacher_positive
    ).astype(int)
    features["other_negative_count"] = (
        teacher_negative.groupby(features[ID_COLUMN]).transform("sum") - teacher_negative
    ).astype(int)
    features["other_abstain_count"] = (
        teacher_abstain.groupby(features[ID_COLUMN]).transform("sum") - teacher_abstain
    ).astype(int)
    features["other_decisive_count"] = features["other_positive_count"] + features["other_negative_count"]
    features["other_positive_fraction_decisive"] = _safe_fraction(
        features["other_positive_count"], features["other_decisive_count"]
    )
    features["other_decisive_zero_denominator"] = features[
        "other_decisive_count"
    ].eq(0)

    teacher_state_matrix = teacher.pivot(
        index=ID_COLUMN, columns="target", values="teacher_state"
    ).reindex(columns=list(targets))
    if teacher_state_matrix.isna().any().any():
        raise ValueError("Teacher state matrix is incomplete")
    feature_uids = features[ID_COLUMN]
    for other_target in targets:
        values = teacher_state_matrix.loc[feature_uids, other_target].to_numpy(object)
        features[f"loto_teacher_state__{other_target}"] = np.where(
            features["target"].eq(other_target), "EXCLUDED_CURRENT", values
        )

    family_positive = family_panel.assign(
        _positive=family_panel["canonical_family_state"].eq("DIRECT_POSITIVE").astype(int)
    )
    family_positive["_positive_all_targets"] = family_positive.groupby(
        [ID_COLUMN, "source"], sort=False
    )["_positive"].transform("sum")
    family_positive["_other_positive"] = (
        family_positive["_positive_all_targets"] - family_positive["_positive"]
    )
    other_family = (
        family_positive.groupby(keys, sort=False)["_other_positive"]
        .agg(family_other_positive_count_mean="mean", family_other_positive_count_std=lambda x: float(np.std(x.to_numpy(float), ddof=0)))
        .reset_index()
    )
    features = features.merge(other_family, on=keys, how="left", validate="one_to_one")

    diversity = features["current_canonical_state_diversity"].astype(int)
    features["other_canonical_state_diversity_count"] = (
        diversity.groupby(features[ID_COLUMN]).transform("sum") - diversity
    ).astype(int)
    direct_conflict = features["family_direct_conflict"].astype(int)
    features["other_direct_conflict_count"] = (
        direct_conflict.groupby(features[ID_COLUMN]).transform("sum") - direct_conflict
    ).astype(int)
    silence_instability = features["family_silence_instability"].astype(int)
    features["other_silence_instability_count"] = (
        silence_instability.groupby(features[ID_COLUMN]).transform("sum")
        - silence_instability
    ).astype(int)

    expected_other = len(targets) - 1
    burden_total = features[["other_positive_count", "other_negative_count", "other_abstain_count"]].sum(axis=1)
    if not burden_total.eq(expected_other).all():
        raise AssertionError("Current target was not excluded correctly from teacher burden")
    for field in (
        "other_canonical_state_diversity_count",
        "other_direct_conflict_count",
        "other_silence_instability_count",
    ):
        if not features[field].between(0, expected_other).all():
            raise AssertionError(f"{field} is outside its deterministic bounds")
    loto_columns = [f"loto_teacher_state__{target}" for target in targets]
    loto = features[loto_columns]
    if not loto.eq("EXCLUDED_CURRENT").sum(axis=1).eq(1).all():
        raise AssertionError("Identity-specific LOTO context must exclude exactly one target")
    if not loto.isin(["POSITIVE", "NEGATIVE", "ABSTAIN", "EXCLUDED_CURRENT"]).all().all():
        raise AssertionError("Identity-specific LOTO context contains an unknown state")

    target_rank = {target: index for index, target in enumerate(targets)}
    features = features.assign(_target_rank=features["target"].map(target_rank)).sort_values(
        [ID_COLUMN, "_target_rank"], kind="mergesort"
    ).drop(columns="_target_rank").reset_index(drop=True)

    numeric = features.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("Reliability feature panel contains non-finite numeric values")

    diagnostics: dict[str, Any] = {
        "status": "PASS",
        "studies": int(features[ID_COLUMN].nunique()),
        "targets": int(features["target"].nunique()),
        "cells": int(len(features)),
        "families": list(families),
        "family_rows": int(len(family_panel)),
        "canonical_state_counts": {
            str(key): int(value)
            for key, value in family_panel["canonical_family_state"].value_counts().sort_index().items()
        },
        "teacher_state_counts": {
            str(key): int(value)
            for key, value in features["teacher_state"].value_counts().sort_index().items()
        },
        "direct_conflict_cells": int(features["family_direct_conflict"].sum()),
        "silence_instability_cells": int(features["family_silence_instability"].sum()),
        "teacher_consensus_conflict_cells": int(
            features["teacher_consensus_aggregation_conflict"].sum()
        ),
        "teacher_component_self_conflict_cells": int(
            features["teacher_component_self_conflict"].sum()
        ),
        "external_audit_disagreement_cells": int(
            features["external_audit_disagreement"].sum()
        ),
        "external_audit_direct_conflict_cells": int(
            features["external_audit_direct_conflict"].sum()
        ),
        "minimum_direct_families": int(minimum_direct_families),
        "teacher_component_families": sorted(_TEACHER_COMPONENT_FAMILIES),
        "external_audit_families": list(_EXTERNAL_AUDIT_FAMILIES),
        "identity_specific_loto_columns": loto_columns,
        "leave_current_target_out_verified": True,
        "numeric_values_finite": True,
        "outcome_columns_used": [],
    }
    return family_panel, features, diagnostics
