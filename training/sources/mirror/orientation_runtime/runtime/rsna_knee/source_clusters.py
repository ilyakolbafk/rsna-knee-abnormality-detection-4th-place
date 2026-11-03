"""Gold-blind report segmentation and anonymous source-proxy clustering."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import Normalizer, StandardScaler

from . import report_rules as rr
from .constants import ID_COLUMN, REPORT_COLUMN, TARGET_COLUMNS
from .label_audit import _SECTION_PATTERNS, _target_clause_match
from .reports import add_report_groups


SEED = 20260829
SEMANTIC_CATEGORICAL = (
    "text_state",
    "criteria_state",
    "evidence_section_proxy",
)
SEMANTIC_RULE_NUMERIC = (
    "rule_score",
    "rule_confidence",
    "positive_mentions",
    "negative_mentions",
    "uncertain_mentions",
)
SEMANTIC_AUGMENTED_NUMERIC = (*SEMANTIC_RULE_NUMERIC, "llm_v4_blend")
SEMANTIC_PANEL_COLUMNS = (
    ID_COLUMN,
    "target",
    "report_group",
    "language_proxy",
    *SEMANTIC_CATEGORICAL,
    *SEMANTIC_AUGMENTED_NUMERIC,
)
FORBIDDEN_CLUSTER_COLUMNS = (
    "gold",
    "is_gold",
    "hard_error",
    "error",
    "audit_priority",
    "recommended_review",
)

_RAW_SPLIT = re.compile(r"(?<=[.;!?])(?:[ \t]+)|\n+")
_PLACEHOLDER = re.compile(r"\[[^\]\n]{1,40}\]")
_NUMBER_MEASUREMENT = re.compile(
    r"\b\d+(?:[.,]\d+)?(?:\s*(?:mm|cm|ml|%|tesla|t))?\b",
    flags=re.IGNORECASE,
)
_WORD = re.compile(r"\w+", flags=re.UNICODE)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(payload: Any, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_deterministic_npz(path: str | Path, **arrays: np.ndarray) -> None:
    """Write an NPZ with stable member order, timestamps, and permissions."""

    target = Path(path)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(arrays[name]), allow_pickle=False)
            member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o600 << 16
            archive.writestr(member, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED)


def _trimmed_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def _base_raw_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in _RAW_SPLIT.finditer(text):
        trimmed = _trimmed_span(text, cursor, match.start())
        if trimmed:
            spans.append(trimmed)
        cursor = match.end()
    trimmed = _trimmed_span(text, cursor, len(text))
    if trimmed:
        spans.append(trimmed)
    return spans


def _comma_children(text: str, start: int, end: int) -> list[tuple[int, int]]:
    span = text[start:end]
    if len(_WORD.findall(span)) <= 25:
        return []
    children: list[tuple[int, int]] = []
    cursor = start
    for match in re.finditer(r",", span):
        child = _trimmed_span(text, cursor, start + match.start())
        if child and len(_WORD.findall(text[child[0] : child[1]])) > 2:
            children.append(child)
        cursor = start + match.end()
    child = _trimmed_span(text, cursor, end)
    if child and len(_WORD.findall(text[child[0] : child[1]])) > 2:
        children.append(child)
    return children if len(children) > 1 else []


def _target_anchor(target: str, clause: str) -> re.Match[str] | None:
    if target in rr.PAIRED:
        pathology = rr._rx(rr.TEAR.pattern, rr.DEGEN.pattern, rr.INJURY.pattern)
        return pathology.search(clause) or rr.ANAT_MATCH[target].search(clause)
    if target in rr.OA_TARGETS:
        return rr.OA_EVIDENCE.search(clause)
    return rr.DIRECT_MATCH[target].search(clause)


def _segment_concepts(clause: str) -> list[str]:
    concepts: list[str] = []
    for target in TARGET_COLUMNS:
        if _target_clause_match(target, clause):
            concepts.append(f"target:{target}")
    for name, pattern in (
        ("negation", rr.NEGATION),
        ("uncertainty", rr.UNCERTAIN),
        ("normality", rr.NORMALITY),
        ("tear", rr.TEAR),
        ("degeneration", rr.DEGEN),
        ("injury", rr.INJURY),
        ("severity_low", rr.SEV_LOW),
        ("severity_high", rr.SEV_HIGH),
        ("trauma", rr.TRAUMA),
    ):
        if pattern.search(clause):
            concepts.append(name)
    return concepts


def _segment_row(
    *,
    study_uid: str,
    report_group: str,
    text: str,
    start: int,
    end: int,
    segment_index: int,
    segment_kind: str,
    parent_segment_index: int | None,
    section: str,
) -> dict[str, Any]:
    raw_span = text[start:end]
    normalized = rr.normalize(raw_span).strip()
    targets = [target for target in TARGET_COLUMNS if _target_clause_match(target, normalized)]
    polarities: list[str] = []
    for target in targets:
        anchor = _target_anchor(target, normalized)
        span = (anchor.start(), anchor.end()) if anchor is not None else None
        polarities.append(rr._polarity(normalized, span))
    polarity = (
        "mixed" if len(set(polarities)) > 1 else polarities[0] if polarities else "not_applicable"
    )
    grade = rr._grade_of(normalized)
    return {
        ID_COLUMN: study_uid,
        "report_group": report_group,
        "segment_index": int(segment_index),
        "segment_kind": segment_kind,
        "parent_segment_index": parent_segment_index,
        "section": section,
        "raw_start": int(start),
        "raw_end": int(end),
        "raw_byte_start": int(len(text[:start].encode("utf-8"))),
        "raw_byte_end": int(len(text[:end].encode("utf-8"))),
        "raw_span": raw_span,
        "raw_span_sha256": hashlib.sha256(raw_span.encode("utf-8")).hexdigest(),
        "clause_normalized": normalized,
        "target_links_json": json.dumps(targets, ensure_ascii=False, separators=(",", ":")),
        "concept_tokens_json": json.dumps(
            _segment_concepts(normalized), ensure_ascii=False, separators=(",", ":")
        ),
        "polarity": polarity,
        "severity": float(rr._severity(normalized)) if targets else np.nan,
        "grade": int(grade) if grade is not None else pd.NA,
    }


def segment_report(study_uid: str, report_group: str, report: object) -> list[dict[str, Any]]:
    text = "" if report is None or pd.isna(report) else str(report)
    rows: list[dict[str, Any]] = []
    active_section = "unknown"
    for start, end in _base_raw_spans(text):
        raw = text[start:end]
        normalized = rr.normalize(raw).strip()
        section_hits = [
            name for name, pattern in _SECTION_PATTERNS if pattern.search(normalized)
        ]
        if section_hits:
            active_section = section_hits[0]
        elif normalized.endswith(":") and len(normalized.split()) <= 14:
            active_section = normalized[:-1].strip() or "unknown"
        parent_index = len(rows)
        rows.append(
            _segment_row(
                study_uid=study_uid,
                report_group=report_group,
                text=text,
                start=start,
                end=end,
                segment_index=parent_index,
                segment_kind="parent",
                parent_segment_index=None,
                section=active_section,
            )
        )
        for child_start, child_end in _comma_children(text, start, end):
            child_index = len(rows)
            rows.append(
                _segment_row(
                    study_uid=study_uid,
                    report_group=report_group,
                    text=text,
                    start=child_start,
                    end=child_end,
                    segment_index=child_index,
                    segment_kind="comma_child",
                    parent_segment_index=parent_index,
                    section=active_section,
                )
            )
    return rows


def build_report_segments(train: pd.DataFrame) -> pd.DataFrame:
    grouped = add_report_groups(train[[ID_COLUMN, REPORT_COLUMN]])
    rows: list[dict[str, Any]] = []
    for record in grouped[[ID_COLUMN, REPORT_COLUMN, "report_group"]].to_dict("records"):
        rows.extend(
            segment_report(
                str(record[ID_COLUMN]),
                str(record["report_group"]),
                record[REPORT_COLUMN],
            )
        )
    output = pd.DataFrame(rows)
    if not output.empty:
        output = output.sort_values(
            [ID_COLUMN, "segment_index"], kind="mergesort"
        ).reset_index(drop=True)
        if output.duplicated([ID_COLUMN, "segment_index"]).any():
            raise AssertionError("duplicate report segment keys")
        source = grouped.set_index(ID_COLUMN)[REPORT_COLUMN].fillna("").astype(str)
        for row in output[
            [
                ID_COLUMN,
                "raw_start",
                "raw_end",
                "raw_byte_start",
                "raw_byte_end",
                "raw_span",
            ]
        ].itertuples(index=False):
            text = source.loc[getattr(row, ID_COLUMN)]
            if text[row.raw_start : row.raw_end] != row.raw_span:
                raise AssertionError("raw report span offset drift")
            raw_bytes = text.encode("utf-8")[row.raw_byte_start : row.raw_byte_end]
            if raw_bytes.decode("utf-8") != row.raw_span:
                raise AssertionError("raw report byte-span offset drift")
    return output


def clinical_mask_patterns() -> list[tuple[str, re.Pattern[str]]]:
    patterns: list[tuple[str, re.Pattern[str]]] = []
    scalar_names = (
        "PRE_NEG",
        "POST_NEG",
        "NORMALITY",
        "NORMAL_PHRASE",
        "UNCERTAIN",
        "TEAR",
        "DEGEN",
        "INJURY",
        "_GRADE_RX",
        "SEV_LOW",
        "SEV_HIGH",
        "GRADE_HIGH",
        "DEGENERATIVE_MARROW",
        "TRAUMA",
        "SYNOVIAL_PROXY",
        "OA_EVIDENCE",
        "TF_SITE",
        "PF_SITE",
        "SIDE_MEDIAL",
        "SIDE_LATERAL",
        "SIDE_ANTERIOR",
        "GLOBAL_OA",
        "PLURAL_MENISCI",
        "STEM_MENISCUS",
        "STEM_CRUCIATE",
        "STEM_COLLATERAL",
        "STEM_FRACTURE",
        "POSTERIOR_ONLY",
        "LATERAL_COLL_ONLY",
    )
    for name in scalar_names:
        patterns.append((name, getattr(rr, name)))
    for group_name in ("ANAT", "DIRECT", "DECOY"):
        for name, pattern in sorted(getattr(rr, group_name).items()):
            patterns.append((f"{group_name}.{name}", pattern))
    return patterns


def _folded_text_with_source_map(text: str) -> tuple[str, list[int]]:
    folded: list[str] = []
    source_indices: list[int] = []
    for index, character in enumerate(text):
        translated = character.translate(rr._PRE).lower()
        decomposed = __import__("unicodedata").normalize("NFKD", translated)
        for output in decomposed:
            if __import__("unicodedata").combining(output):
                continue
            folded.append(" " if output in "_-/\\" else output)
            source_indices.append(index)
    return "".join(folded), source_indices


def _merged_mask_spans(text: str) -> list[tuple[int, int]]:
    folded, source_map = _folded_text_with_source_map(text)
    spans: list[tuple[int, int]] = []
    for _, pattern in clinical_mask_patterns():
        for match in pattern.finditer(folded):
            if match.start() < match.end() and source_map:
                spans.append((source_map[match.start()], source_map[match.end() - 1] + 1))
    for pattern in (_PLACEHOLDER, _NUMBER_MEASUREMENT):
        spans.extend((match.start(), match.end()) for match in pattern.finditer(text))
    if not spans:
        return []
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def mask_clinical_text(value: object) -> tuple[str, float, int]:
    text = "" if value is None or pd.isna(value) else str(value)
    spans = _merged_mask_spans(text)
    if not spans:
        return text, 0.0, 0
    output: list[str] = []
    cursor = 0
    for start, end in spans:
        output.extend((text[cursor:start], "[CLIN]"))
        cursor = end
    output.append(text[cursor:])
    words = list(_WORD.finditer(text))
    masked_words = sum(
        any(word.start() < end and word.end() > start for start, end in spans)
        for word in words
    )
    fraction = float(masked_words / len(words)) if words else 0.0
    return "".join(output), fraction, len(spans)


def delexicalize_style(value: object) -> str:
    text = "" if value is None or pd.isna(value) else str(value)
    tokens: list[str] = []
    cursor = 0
    for match in _WORD.finditer(text):
        for character in text[cursor : match.start()]:
            if character == "\n":
                tokens.append("NL")
            elif character == "\t":
                tokens.append("TAB")
            elif not character.isspace():
                tokens.append("P_" + format(ord(character), "x"))
        word = match.group()
        length = len(word)
        bucket = str(length) if length <= 5 else "6_8" if length <= 8 else "9_12" if length <= 12 else "13P"
        if word.isupper():
            case = "U"
        elif word.istitle():
            case = "T"
        elif word.islower():
            case = "L"
        elif word.isnumeric():
            case = "N"
        else:
            case = "M"
        tokens.append(f"W_{bucket}_{case}")
        cursor = match.end()
    for character in text[cursor:]:
        if character == "\n":
            tokens.append("NL")
        elif character == "\t":
            tokens.append("TAB")
        elif not character.isspace():
            tokens.append("P_" + format(ord(character), "x"))
    return " ".join(tokens)


def structural_style_features(value: object) -> tuple[list[float], list[str]]:
    text = "" if value is None or pd.isna(value) else str(value)
    words = _WORD.findall(text)
    letters = [character for character in text if character.isalpha()]
    total = max(len(text), 1)
    word_lengths = np.asarray([len(word) for word in words], dtype=float)
    lines = text.splitlines() if text else []
    names = [
        "log_chars",
        "log_words",
        "log_lines",
        "log_heading_lines",
        "log_placeholder_count",
        "uppercase_ratio",
        "digit_ratio",
        "mean_word_length",
        "std_word_length",
        "blank_line_ratio",
        "indented_line_ratio",
        "period_ratio",
        "comma_ratio",
        "semicolon_ratio",
        "colon_ratio",
        "exclamation_ratio",
        "question_ratio",
        "parenthesis_ratio",
        "dash_ratio",
        "slash_ratio",
    ]
    heading_lines = sum(bool(line.strip()) and line.strip().endswith(":") for line in lines)
    values = [
        math.log1p(len(text)),
        math.log1p(len(words)),
        math.log1p(len(lines)),
        math.log1p(heading_lines),
        math.log1p(len(_PLACEHOLDER.findall(text))),
        sum(character.isupper() for character in letters) / max(len(letters), 1),
        sum(character.isdigit() for character in text) / total,
        float(word_lengths.mean()) if len(word_lengths) else 0.0,
        float(word_lengths.std()) if len(word_lengths) else 0.0,
        sum(not line.strip() for line in lines) / max(len(lines), 1),
        sum(bool(line[:1].isspace()) for line in lines) / max(len(lines), 1),
        text.count(".") / total,
        text.count(",") / total,
        text.count(";") / total,
        text.count(":") / total,
        text.count("!") / total,
        text.count("?") / total,
        (text.count("(") + text.count(")")) / total,
        (text.count("-") + text.count("–") + text.count("—")) / total,
        (text.count("/") + text.count("\\")) / total,
    ]
    return values, names


def load_semantic_panel(path: str | Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    forbidden_present = sorted(set(header.columns).intersection(FORBIDDEN_CLUSTER_COLUMNS))
    # The source panel may contain Gold audit fields; strict usecols ensures they never enter.
    missing = sorted(set(SEMANTIC_PANEL_COLUMNS).difference(header.columns))
    if missing:
        raise ValueError(f"semantic panel missing columns: {missing}")
    panel = pd.read_csv(
        path,
        usecols=list(SEMANTIC_PANEL_COLUMNS),
        dtype={ID_COLUMN: str, "target": str, "report_group": str},
    )
    panel.attrs["forbidden_source_columns_ignored"] = forbidden_present
    if panel.duplicated([ID_COLUMN, "target"]).any():
        raise ValueError("semantic panel has duplicate study-target rows")
    if len(panel) != 4407 * len(TARGET_COLUMNS):
        raise ValueError(f"semantic panel expected 52884 rows, got {len(panel)}")
    target_counts = panel.groupby(ID_COLUMN)["target"].agg(lambda x: set(x))
    expected_targets = set(TARGET_COLUMNS)
    if not target_counts.map(lambda value: value == expected_targets).all():
        raise ValueError("semantic panel target coverage drift")
    return panel


def _deterministic_mode(values: pd.Series) -> Any:
    kept = values.dropna()
    if kept.empty:
        return "__missing__"
    counts = kept.astype(str).value_counts()
    candidates = counts[counts.eq(counts.max())].index.tolist()
    return sorted(candidates, key=str.casefold)[0]


def _validate_report_group_invariance(panel: pd.DataFrame) -> None:
    rule_fields = (*SEMANTIC_CATEGORICAL, *SEMANTIC_RULE_NUMERIC, "language_proxy")
    grouped = panel.groupby(["report_group", "target"], sort=False)
    drift: dict[str, int] = {}
    for column in rule_fields:
        counts = grouped[column].nunique(dropna=False)
        bad = int(counts.gt(1).sum())
        if bad:
            drift[column] = bad
    if drift:
        raise ValueError(f"rule-derived duplicate report_group drift: {drift}")


def build_semantic_signatures(
    train: pd.DataFrame,
    panel: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], pd.DataFrame]:
    reports = add_report_groups(train[[ID_COLUMN, REPORT_COLUMN]])
    if reports[ID_COLUMN].isna().any() or reports[ID_COLUMN].duplicated().any():
        raise ValueError("train reports must be unique by StudyInstanceUID")
    study_panel = panel.drop_duplicates(ID_COLUMN).set_index(ID_COLUMN)
    expected = reports.set_index(ID_COLUMN)["report_group"]
    observed = study_panel.reindex(expected.index)["report_group"]
    if observed.isna().any() or not observed.eq(expected).all():
        raise ValueError("panel report_group does not match frozen report normalization")
    _validate_report_group_invariance(panel)

    groups = np.asarray(sorted(reports["report_group"].unique()), dtype=str)
    grouped_panel = panel.groupby(["report_group", "target"], sort=True)
    aggregate_rows: list[dict[str, Any]] = []
    for (report_group, target), part in grouped_panel:
        row: dict[str, Any] = {"report_group": str(report_group), "target": str(target)}
        for column in SEMANTIC_CATEGORICAL:
            row[column] = _deterministic_mode(part[column])
        for column in SEMANTIC_AUGMENTED_NUMERIC:
            values = pd.to_numeric(part[column], errors="coerce")
            row[column] = float(values.median())
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)
    if aggregate.duplicated(["report_group", "target"]).any():
        raise AssertionError("semantic group-target aggregation drift")

    feature_blocks: list[np.ndarray] = []
    augmented_extra: list[np.ndarray] = []
    feature_names: list[str] = []
    augmented_feature_names: list[str] = []
    category_schema: dict[str, list[str]] = {}
    numeric_schema: dict[str, dict[str, float]] = {}
    index = aggregate.set_index(["report_group", "target"])
    for target in TARGET_COLUMNS:
        target_rows = index.xs(target, level="target").reindex(groups)
        if target_rows.isna().all(axis=1).any():
            raise ValueError(f"missing semantic rows for target {target}")
        for column in SEMANTIC_CATEGORICAL:
            values = target_rows[column].fillna("__missing__").astype(str).to_numpy()
            categories = sorted(set(values.tolist()), key=str.casefold)
            key = f"{target}|{column}"
            category_schema[key] = categories
            block = np.column_stack([values == category for category in categories]).astype(
                np.float32
            )
            feature_blocks.append(block)
            feature_names.extend(f"{key}|{category}" for category in categories)
        for column in SEMANTIC_RULE_NUMERIC:
            values = pd.to_numeric(target_rows[column], errors="coerce").to_numpy(float)
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite semantic field {target}/{column}")
            mean = float(values.mean())
            scale = float(values.std()) or 1.0
            feature_blocks.append(((values - mean) / scale).astype(np.float32)[:, None])
            name = f"{target}|{column}"
            feature_names.append(name)
            numeric_schema[name] = {"mean": mean, "scale": scale, "aggregation": "median"}
        llm_values = pd.to_numeric(target_rows["llm_v4_blend"], errors="coerce").to_numpy(float)
        if not np.isfinite(llm_values).all():
            raise ValueError(f"non-finite llm_v4_blend for {target}")
        llm_mean = float(llm_values.mean())
        llm_scale = float(llm_values.std()) or 1.0
        augmented_extra.append(
            ((llm_values - llm_mean) / llm_scale).astype(np.float32)[:, None]
        )
        llm_name = f"{target}|llm_v4_blend"
        augmented_feature_names.append(llm_name)
        numeric_schema[llm_name] = {
            "mean": llm_mean,
            "scale": llm_scale,
            "aggregation": "report_group_target_median",
        }

    rule_only = np.concatenate(feature_blocks, axis=1).astype(np.float32)
    augmented_parts: list[np.ndarray] = []
    rule_cursor = 0
    extra_cursor = 0
    augmented_names: list[str] = []
    # Preserve the canonical per-target field order, adding LLM after five rule numerics.
    for target in TARGET_COLUMNS:
        category_count = sum(
            len(category_schema[f"{target}|{column}"])
            for column in SEMANTIC_CATEGORICAL
        )
        width = category_count + len(SEMANTIC_RULE_NUMERIC)
        augmented_parts.append(rule_only[:, rule_cursor : rule_cursor + width])
        augmented_names.extend(feature_names[rule_cursor : rule_cursor + width])
        augmented_parts.append(augmented_extra[extra_cursor])
        augmented_names.append(augmented_feature_names[extra_cursor])
        rule_cursor += width
        extra_cursor += 1
    augmented = np.concatenate(augmented_parts, axis=1).astype(np.float32)
    if rule_only.shape != (len(groups), 276):
        raise AssertionError(f"rule-only semantic signature expected (*,276), got {rule_only.shape}")
    if augmented.shape != (len(groups), 288):
        raise AssertionError(f"augmented semantic signature expected (*,288), got {augmented.shape}")
    if not np.isfinite(rule_only).all() or not np.isfinite(augmented).all():
        raise AssertionError("semantic signatures contain non-finite values")

    schema = {
        "target_order": TARGET_COLUMNS,
        "categorical_fields": list(SEMANTIC_CATEGORICAL),
        "rule_numeric_fields": list(SEMANTIC_RULE_NUMERIC),
        "augmented_numeric_fields": list(SEMANTIC_AUGMENTED_NUMERIC),
        "category_vocabularies": category_schema,
        "numeric_scalers": numeric_schema,
        "rule_only_feature_names": feature_names,
        "augmented_feature_names": augmented_names,
        "rule_only_dimensions": int(rule_only.shape[1]),
        "augmented_dimensions": int(augmented.shape[1]),
        "group_count": int(len(groups)),
        "llm_augmented_restriction": (
            "sensitivity only; not eligible as primary grouping for LLM disagreement"
        ),
    }
    study_map = reports[[ID_COLUMN, "report_group"]].merge(
        panel.drop_duplicates(ID_COLUMN)[[ID_COLUMN, "language_proxy"]],
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    if study_map["language_proxy"].isna().any():
        raise ValueError("missing language proxy")
    return groups, rule_only, augmented, schema, study_map


def semantic_embedding(
    raw_features: np.ndarray,
    *,
    fit_indices: np.ndarray | None = None,
    seed: int = SEED,
    components: int = 48,
) -> np.ndarray:
    fit_indices = (
        np.arange(len(raw_features), dtype=int)
        if fit_indices is None
        else np.asarray(fit_indices, dtype=int)
    )
    n_components = max(
        2,
        min(components, len(fit_indices) - 1, raw_features.shape[1] - 1),
    )
    features = np.asarray(raw_features, dtype=np.float64)
    if not np.isfinite(features).all() or np.max(np.abs(features)) > 1e6:
        raise ValueError("semantic features fail finite/range gate")
    reducer = TruncatedSVD(
        n_components=n_components,
        algorithm="arpack",
        random_state=seed,
        tol=1e-6,
    )
    # Apple Accelerate can leave floating-point status flags set after finite matmul;
    # suppress those spurious warnings locally and enforce explicit output gates below.
    with np.errstate(all="ignore"):
        reducer.fit(features[fit_indices])
        embedding = Normalizer(copy=False).fit_transform(reducer.transform(features))
    if not np.isfinite(embedding).all():
        raise ValueError("semantic embedding contains non-finite values")
    return np.asarray(embedding, dtype=np.float64)


def _alignment(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, list[float]]:
    ref_values = np.unique(reference)
    cand_values = np.unique(candidate)
    contingency = np.zeros((len(ref_values), len(cand_values)), dtype=int)
    for row, ref_label in enumerate(ref_values):
        for column, cand_label in enumerate(cand_values):
            contingency[row, column] = int(
                np.sum((reference == ref_label) & (candidate == cand_label))
            )
    rows, columns = linear_sum_assignment(-contingency)
    mapping = {cand_values[column]: ref_values[row] for row, column in zip(rows, columns)}
    aligned = np.asarray([mapping.get(value, value) for value in candidate])
    jaccards: list[float] = []
    for ref_label in ref_values:
        ref_mask = reference == ref_label
        candidate_mask = aligned == ref_label
        union = int(np.sum(ref_mask | candidate_mask))
        jaccards.append(float(np.sum(ref_mask & candidate_mask) / union) if union else 1.0)
    return aligned, jaccards


def _p10(values: Sequence[float]) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), 0.1)) if values else 1.0


def _canonical_cluster_labels(
    labels: np.ndarray,
    group_ids: np.ndarray,
    *,
    prefix: str,
) -> tuple[np.ndarray, dict[int, str], dict[int, str]]:
    fingerprints: dict[int, str] = {}
    for label in np.unique(labels):
        members = sorted(group_ids[labels == label].astype(str).tolist())
        fingerprints[int(label)] = hashlib.sha256(("\n".join(members) + "\n").encode()).hexdigest()
    ordered = sorted(fingerprints, key=lambda label: fingerprints[label])
    names = {
        label: f"{prefix}_{rank:02d}_{fingerprints[label][:12]}"
        for rank, label in enumerate(ordered)
    }
    return np.asarray([names[int(label)] for label in labels]), names, fingerprints


def _candidate_stability(
    *,
    reference_embedding: np.ndarray,
    alternative_embeddings: Sequence[np.ndarray],
    bootstrap_embeddings: Sequence[tuple[np.ndarray, np.ndarray]],
    group_ids: np.ndarray,
    study_weights: np.ndarray,
    k: int,
    seed: int,
) -> tuple[dict[str, Any], KMeans, np.ndarray, np.ndarray]:
    if not np.isfinite(reference_embedding).all():
        raise ValueError("reference cluster embedding contains non-finite values")
    model = KMeans(n_clusters=k, random_state=seed, n_init=50, algorithm="lloyd")
    with np.errstate(all="ignore"):
        reference_labels = model.fit_predict(reference_embedding)
    alternatives: list[np.ndarray] = []
    seed_aris: list[float] = []
    bootstrap_aris: list[float] = []
    seed_jaccards: list[float] = []
    bootstrap_jaccards: list[float] = []
    for offset, embedding in enumerate(alternative_embeddings, start=1):
        if not np.isfinite(embedding).all():
            raise ValueError("alternate cluster embedding contains non-finite values")
        with np.errstate(all="ignore"):
            candidate = KMeans(
                n_clusters=k,
                random_state=seed + offset,
                n_init=50,
                algorithm="lloyd",
            ).fit_predict(embedding)
        seed_aris.append(float(adjusted_rand_score(reference_labels, candidate)))
        aligned, jaccards = _alignment(reference_labels, candidate)
        alternatives.append(aligned)
        seed_jaccards.extend(jaccards)
    for offset, (embedding, fit_indices) in enumerate(bootstrap_embeddings, start=1):
        boot_model = KMeans(
            n_clusters=k,
            random_state=seed + 100 + offset,
            n_init=50,
            algorithm="lloyd",
        )
        if not np.isfinite(embedding).all():
            raise ValueError("bootstrap cluster embedding contains non-finite values")
        with np.errstate(all="ignore"):
            boot_model.fit(embedding[fit_indices])
            candidate = boot_model.predict(embedding)
        bootstrap_aris.append(float(adjusted_rand_score(reference_labels, candidate)))
        aligned, jaccards = _alignment(reference_labels, candidate)
        alternatives.append(aligned)
        bootstrap_jaccards.extend(jaccards)
    if alternatives:
        consensus = np.mean(
            np.column_stack([labels == reference_labels for labels in alternatives]), axis=1
        )
    else:
        consensus = np.ones(len(reference_labels), dtype=float)
    group_sizes = Counter(reference_labels.tolist())
    study_sizes = {
        int(label): int(study_weights[reference_labels == label].sum())
        for label in np.unique(reference_labels)
    }
    metrics = {
        "k": int(k),
        "group_cluster_sizes": {str(key): int(value) for key, value in group_sizes.items()},
        "study_cluster_sizes": {str(key): int(value) for key, value in study_sizes.items()},
        "minimum_study_cluster_size": int(min(study_sizes.values())),
        "seed_ari": seed_aris,
        "seed_ari_p10": _p10(seed_aris),
        "bootstrap_ari": bootstrap_aris,
        "bootstrap_ari_p10": _p10(bootstrap_aris),
        "seed_aligned_jaccard_p10": _p10(seed_jaccards),
        "bootstrap_aligned_jaccard_p10": _p10(bootstrap_jaccards),
        "assignment_consensus_p10": _p10(consensus.tolist()),
        "assignment_consensus_mean": float(consensus.mean()),
    }
    metrics["passes"] = bool(
        metrics["minimum_study_cluster_size"] >= 80
        and metrics["seed_ari_p10"] >= 0.8
        and metrics["bootstrap_ari_p10"] >= 0.8
        and min(
            metrics["seed_aligned_jaccard_p10"],
            metrics["bootstrap_aligned_jaccard_p10"],
        )
        >= 0.6
        and metrics["assignment_consensus_p10"] >= 0.8
    )
    return metrics, model, reference_labels, consensus


def choose_stable_clusters(
    *,
    raw_features: np.ndarray,
    group_ids: np.ndarray,
    study_weights: np.ndarray,
    candidate_ks: Sequence[int],
    seed: int = SEED,
    bootstrap_runs: int = 7,
    alternate_seed_runs: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    all_indices = np.arange(len(group_ids), dtype=int)
    reference_embedding = semantic_embedding(raw_features, seed=seed)
    alternatives = [
        semantic_embedding(raw_features, seed=seed + offset)
        for offset in range(1, alternate_seed_runs + 1)
    ]
    rng = np.random.default_rng(seed)
    bootstraps: list[tuple[np.ndarray, np.ndarray]] = []
    for offset in range(bootstrap_runs):
        fit_indices = np.sort(
            rng.choice(
                all_indices,
                size=max(2, int(round(0.8 * len(all_indices)))),
                replace=False,
            )
        )
        bootstraps.append(
            (
                semantic_embedding(
                    raw_features, fit_indices=fit_indices, seed=seed + 100 + offset
                ),
                fit_indices,
            )
        )
    candidate_results: list[dict[str, Any]] = []
    selected: tuple[KMeans, np.ndarray, np.ndarray, dict[str, Any]] | None = None
    for k in candidate_ks:
        metrics, model, labels, consensus = _candidate_stability(
            reference_embedding=reference_embedding,
            alternative_embeddings=alternatives,
            bootstrap_embeddings=bootstraps,
            group_ids=group_ids,
            study_weights=study_weights,
            k=int(k),
            seed=seed,
        )
        candidate_results.append(metrics)
        if selected is None and metrics["passes"]:
            selected = model, labels, consensus, metrics
    if selected is None:
        raise RuntimeError(f"No semantic cluster candidate passed: {candidate_results}")
    model, labels, consensus, selected_metrics = selected
    with np.errstate(all="ignore"):
        distances = model.transform(reference_embedding)
    if not np.isfinite(distances).all():
        raise ValueError("semantic cluster distances contain non-finite values")
    ordered = np.sort(distances, axis=1)
    margin = ordered[:, 1] - ordered[:, 0]
    diagnostics = {
        "chosen_k": int(selected_metrics["k"]),
        "candidates": candidate_results,
        "bootstrap_scope": "80% report-group refit of SVD and KMeans",
    }
    return labels, distances.min(axis=1), margin, consensus, reference_embedding, diagnostics


def _safe_svd_block(
    matrix: Any,
    *,
    fit_indices: np.ndarray,
    components: int,
    seed: int,
) -> np.ndarray:
    n_components = min(components, len(fit_indices) - 1, matrix.shape[1] - 1)
    if n_components < 2:
        dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
        return Normalizer(copy=False).fit_transform(dense.astype(np.float32))
    reducer = TruncatedSVD(
        n_components=n_components,
        algorithm="arpack",
        random_state=seed,
        tol=1e-6,
    )
    with np.errstate(all="ignore"):
        reducer.fit(matrix[fit_indices])
        block = Normalizer(copy=False).fit_transform(reducer.transform(matrix))
    if not np.isfinite(block).all():
        raise ValueError("style SVD block contains non-finite values")
    return np.asarray(block, dtype=np.float64)


def build_style_embedding(
    reports: Sequence[str],
    masked_reports: Sequence[str],
    semantic_features: np.ndarray,
    *,
    fit_indices: np.ndarray | None = None,
    seed: int = SEED,
) -> tuple[np.ndarray, dict[str, Any]]:
    all_indices = np.arange(len(reports), dtype=int)
    fit_indices = all_indices if fit_indices is None else np.asarray(fit_indices, dtype=int)
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_features=12_000,
        sublinear_tf=True,
        dtype=np.float64,
    )
    char_vectorizer.fit([masked_reports[index] for index in fit_indices])
    char_matrix = char_vectorizer.transform(masked_reports)
    char_block = _safe_svd_block(
        char_matrix, fit_indices=fit_indices, components=48, seed=seed
    )

    delexicalized = [delexicalize_style(text) for text in masked_reports]
    delex_vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 5),
        min_df=3,
        max_features=8_000,
        sublinear_tf=True,
        dtype=np.float64,
    )
    delex_vectorizer.fit([delexicalized[index] for index in fit_indices])
    delex_matrix = delex_vectorizer.transform(delexicalized)
    delex_block = _safe_svd_block(
        delex_matrix, fit_indices=fit_indices, components=24, seed=seed + 1
    )

    structural_rows: list[list[float]] = []
    structural_names: list[str] | None = None
    for text in reports:
        values, names = structural_style_features(text)
        structural_rows.append(values)
        structural_names = names
    structural_raw = np.asarray(structural_rows, dtype=np.float64)
    structural_scaler = StandardScaler().fit(structural_raw[fit_indices])
    structural_block = Normalizer(copy=False).fit_transform(
        structural_scaler.transform(structural_raw)
    ).astype(np.float64)

    energy = math.sqrt(3.0)
    combined = np.concatenate(
        [char_block / energy, delex_block / energy, structural_block / energy], axis=1
    ).astype(np.float64)
    if not np.isfinite(combined).all() or np.max(np.abs(combined)) > 1e6:
        raise ValueError("combined style representation fails finite/range gate")
    predictions = np.zeros_like(combined)
    if len(fit_indices) == len(all_indices) and np.array_equal(fit_indices, all_indices):
        splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
        for train_index, validation_index in splitter.split(all_indices):
            model = Ridge(alpha=1.0, solver="lsqr")
            with np.errstate(all="ignore"):
                model.fit(semantic_features[train_index], combined[train_index])
                predictions[validation_index] = model.predict(
                    semantic_features[validation_index]
                )
    else:
        model = Ridge(alpha=1.0, solver="lsqr")
        with np.errstate(all="ignore"):
            model.fit(semantic_features[fit_indices], combined[fit_indices])
            predictions = model.predict(semantic_features).astype(np.float64)
    if not np.isfinite(predictions).all():
        raise ValueError("semantic style projection contains non-finite values")
    residual = combined - predictions
    scaler = StandardScaler().fit(residual[fit_indices])
    residual = Normalizer(copy=False).fit_transform(scaler.transform(residual)).astype(
        np.float64
    )
    if not np.isfinite(residual).all():
        raise ValueError("style residual contains non-finite values")
    diagnostics = {
        "rows": int(len(reports)),
        "fit_rows": int(len(fit_indices)),
        "masked_char_vocabulary": int(len(char_vectorizer.vocabulary_)),
        "masked_char_components": int(char_block.shape[1]),
        "delex_vocabulary": int(len(delex_vectorizer.vocabulary_)),
        "delex_components": int(delex_block.shape[1]),
        "structural_features": structural_names or [],
        "residual_dimensions": int(residual.shape[1]),
        "semantic_projection": (
            "5-fold cross-fit" if len(fit_indices) == len(all_indices) else "bootstrap-fit"
        ),
    }
    return residual, diagnostics


def build_group_style_corpus(
    train: pd.DataFrame,
    study_map: pd.DataFrame,
) -> tuple[np.ndarray, list[str], list[str], np.ndarray, np.ndarray, dict[str, Any]]:
    reports = add_report_groups(train[[ID_COLUMN, REPORT_COLUMN]])
    merged = reports.merge(
        study_map[[ID_COLUMN, "language_proxy"]],
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    if merged.groupby("report_group")["language_proxy"].nunique(dropna=False).max() > 1:
        raise ValueError("duplicate report_group has multiple language proxies")
    records: list[dict[str, Any]] = []
    for report_group, part in merged.groupby("report_group", sort=True):
        raw_reports = part[REPORT_COLUMN].fillna("").astype(str).tolist()
        representative = sorted(raw_reports)[0]
        records.append(
            {
                "report_group": str(report_group),
                "report": representative,
                "language_proxy": str(part["language_proxy"].iloc[0]),
                "study_count": int(len(part)),
            }
        )
    groups = np.asarray([row["report_group"] for row in records], dtype=str)
    raw = [row["report"] for row in records]
    language = np.asarray([row["language_proxy"] for row in records], dtype=str)
    weights = np.asarray([row["study_count"] for row in records], dtype=int)
    masked: list[str] = []
    fractions: list[float] = []
    span_counts: list[int] = []
    for text in raw:
        masked_text, fraction, span_count = mask_clinical_text(text)
        masked.append(masked_text)
        fractions.append(fraction)
        span_counts.append(span_count)
    rules_path = Path(rr.__file__).resolve()
    manifest = {
        "report_rules_path": str(rules_path),
        "report_rules_sha256": sha256_file(rules_path),
        "pattern_names": [name for name, _ in clinical_mask_patterns()],
        "additional_patterns": ["placeholder", "number_or_measurement"],
        "groups": int(len(groups)),
        "studies": int(weights.sum()),
        "masked_token_fraction": {
            "mean": float(np.mean(fractions)),
            "median": float(np.median(fractions)),
            "p10": float(np.quantile(fractions, 0.1)),
            "p90": float(np.quantile(fractions, 0.9)),
        },
        "mask_span_count": {
            "total": int(sum(span_counts)),
            "median_per_group": float(np.median(span_counts)),
        },
    }
    return groups, raw, masked, language, weights, manifest


def choose_language_style_clusters(
    *,
    group_ids: np.ndarray,
    language: np.ndarray,
    study_weights: np.ndarray,
    reference_embedding: np.ndarray,
    alternative_embeddings: Sequence[np.ndarray],
    bootstrap_embeddings: Sequence[tuple[np.ndarray, np.ndarray]],
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    candidate_map: dict[str, list[int]] = {
        "english_proxy": [6, 5, 4, 3, 2],
        "spanish_proxy": [2],
        "turkish_proxy": [2],
        "south_slavic_latin_proxy": [2],
    }
    cluster_ids = np.empty(len(group_ids), dtype=object)
    distance = np.full(len(group_ids), np.nan, dtype=float)
    margin = np.full(len(group_ids), np.nan, dtype=float)
    consensus = np.ones(len(group_ids), dtype=float)
    diagnostics: dict[str, Any] = {"strata": {}}
    for stratum in sorted(set(language.tolist())):
        global_indices = np.flatnonzero(language == stratum)
        local_groups = group_ids[global_indices]
        local_weights = study_weights[global_indices]
        candidates = candidate_map.get(stratum, [1])
        if candidates == [1]:
            fingerprint = hashlib.sha256(
                ("\n".join(sorted(local_groups.tolist())) + "\n").encode()
            ).hexdigest()
            cluster_ids[global_indices] = f"style_{stratum}_00_{fingerprint[:12]}"
            distance[global_indices] = 0.0
            diagnostics["strata"][stratum] = {
                "chosen_k": 1,
                "passes": True,
                "association_eligible": False,
                "study_rows": int(local_weights.sum()),
                "group_rows": int(len(local_groups)),
            }
            continue
        local_alternatives = [embedding[global_indices] for embedding in alternative_embeddings]
        local_bootstraps: list[tuple[np.ndarray, np.ndarray]] = []
        for embedding, fit_indices in bootstrap_embeddings:
            fit_set = set(np.asarray(fit_indices, dtype=int).tolist())
            local_fit = np.asarray(
                [position for position, index in enumerate(global_indices) if index in fit_set],
                dtype=int,
            )
            local_bootstraps.append((embedding[global_indices], local_fit))
        candidate_results: list[dict[str, Any]] = []
        selected: tuple[KMeans, np.ndarray, np.ndarray, dict[str, Any]] | None = None
        for k in candidates:
            metrics, model, labels, local_consensus = _candidate_stability(
                reference_embedding=reference_embedding[global_indices],
                alternative_embeddings=local_alternatives,
                bootstrap_embeddings=local_bootstraps,
                group_ids=local_groups,
                study_weights=local_weights,
                k=k,
                seed=seed,
            )
            candidate_results.append(metrics)
            if selected is None and metrics["passes"]:
                selected = model, labels, local_consensus, metrics
        if selected is None:
            fingerprint = hashlib.sha256(
                ("\n".join(sorted(local_groups.tolist())) + "\n").encode()
            ).hexdigest()
            cluster_ids[global_indices] = f"style_{stratum}_00_{fingerprint[:12]}"
            distance[global_indices] = 0.0
            diagnostics["strata"][stratum] = {
                "chosen_k": 1,
                "passes": True,
                "association_eligible": False,
                "fallback_reason": "no_multi_cluster_candidate_passed",
                "candidates": candidate_results,
                "study_rows": int(local_weights.sum()),
                "group_rows": int(len(local_groups)),
            }
            continue
        model, labels, local_consensus, metrics = selected
        canonical, _, _ = _canonical_cluster_labels(
            labels, local_groups, prefix=f"style_{stratum}"
        )
        with np.errstate(all="ignore"):
            distances = model.transform(reference_embedding[global_indices])
        if not np.isfinite(distances).all():
            raise ValueError(f"style cluster distances are non-finite for {stratum}")
        ordered = np.sort(distances, axis=1)
        cluster_ids[global_indices] = canonical
        distance[global_indices] = distances.min(axis=1)
        margin[global_indices] = ordered[:, 1] - ordered[:, 0]
        consensus[global_indices] = local_consensus
        diagnostics["strata"][stratum] = {
            "chosen_k": int(metrics["k"]),
            "passes": True,
            "association_eligible": True,
            "candidates": candidate_results,
            "study_rows": int(local_weights.sum()),
            "group_rows": int(len(local_groups)),
        }
    return cluster_ids.astype(str), distance, margin, consensus, diagnostics
