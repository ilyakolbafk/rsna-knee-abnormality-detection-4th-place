"""Deterministic, text-free report morphology features.

This module deliberately treats spelling and language as *morphology proxies*.
It does not infer a country, site, radiologist, or whether a clinical statement
is correct.  Corpus-relative rarity and character-ngram scores are fitted
within the frozen ``language_proxy`` strata so non-English vocabulary is not
penalised merely for being non-English.

Only hashes, categorical proxy names, and numeric summaries leave the module.
Raw, normalized, tokenized, or delexicalized report text is never returned.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import re
import unicodedata
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .constants import ID_COLUMN, REPORT_COLUMN
from .reports import dominant_script, normalize_report, report_hash
from .source_clusters import delexicalize_style


DEFAULT_SEED = 20260830
STYLE_CLUSTER_VIEW = "style_residual_within_language"
RARE_DOCUMENT_FREQUENCY_MAX = 2

_WORDISH_RE = re.compile(r"[^\W_]+(?:['\u2019][^\W_]+)?", re.UNICODE)
_REPEATED_ALNUM_RE = re.compile(r"([^\W_])\1{3,}", re.UNICODE)
_REPEATED_PUNCTUATION_RE = re.compile(r"([^\w\s])\1{2,}", re.UNICODE)
_MOJIBAKE_RE = re.compile(
    r"(?:[\u00c3\u00c2\u00d0\u00d1][\u0080-\u00bf]|"
    r"\u00e2(?:\u20ac|\u2122|\u0153|\u201d|\u201c|\u2013|\u2014)|\u00f0\u0178)"
)
_ZERO_WIDTH = frozenset({"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"})
_PLACEHOLDER_RE = re.compile(r"\[[^\]\n]{1,32}\]")
_HEADING_RE = re.compile(
    r"(?:^|[\n.!?]\s+)[^\W\d_][^:\n]{0,39}:", re.UNICODE
)
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?]+(?:\s+|$)|\n+")
_PARAGRAPH_BOUNDARY_RE = re.compile(r"(?:\r?\n\s*){2,}")


def _sha256_lines(values: Sequence[object]) -> str:
    payload = "\n".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_columns(frame: pd.DataFrame, columns: set[str], *, name: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def _clean_uid_column(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if ID_COLUMN not in frame:
        raise ValueError(f"{name} missing column {ID_COLUMN!r}")
    if frame[ID_COLUMN].isna().any():
        raise ValueError(f"{name} has missing study identifiers")
    result = frame.copy()
    result[ID_COLUMN] = result[ID_COLUMN].astype(str)
    if result[ID_COLUMN].str.strip().eq("").any():
        raise ValueError(f"{name} has empty study identifiers")
    return result


def _require_exact_uid_set(
    frame: pd.DataFrame, expected: set[str], *, name: str
) -> None:
    observed = set(frame[ID_COLUMN].astype(str))
    if observed != expected:
        missing = len(expected.difference(observed))
        extra = len(observed.difference(expected))
        raise ValueError(f"{name} UID coverage mismatch: missing={missing}, extra={extra}")


def _coerce_boolean(series: pd.Series, *, name: str) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"{name} has missing boolean values")
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(bool)
    if pd.api.types.is_numeric_dtype(series.dtype):
        numeric = pd.to_numeric(series, errors="raise")
        if not numeric.isin([0, 1]).all():
            raise ValueError(f"{name} has non-boolean numeric values")
        return numeric.astype(bool)
    normalized = series.astype(str).str.strip().str.casefold()
    if not normalized.isin(["true", "false"]).all():
        raise ValueError(f"{name} has non-boolean string values")
    return normalized.eq("true")


def _script_of_character(character: str) -> str | None:
    if not character.isalpha():
        return None
    name = unicodedata.name(character, "")
    for script in ("LATIN", "GREEK", "CYRILLIC", "ARABIC", "HEBREW"):
        if script in name:
            return script.casefold()
    return "other"


def _word_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return [match.group() for match in _WORDISH_RE.finditer(normalized)]


def _lexical_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for token in _word_tokens(text):
        letters = sum(character.isalpha() for character in token)
        if letters >= 2 and all(
            character.isalpha() or character in {"'", "\u2019"}
            for character in token
        ):
            tokens.append(token)
    return tokens


def delexicalized_template_key(value: object) -> str:
    """Return a stable SHA-256 key for word-identity-free report structure.

    ``delexicalize_style`` retains layout, punctuation code points, word-length
    buckets, and case classes.  Hashing its result prevents even that internal
    representation from entering downstream CSV artifacts.
    """

    delexicalized = delexicalize_style(value)
    return hashlib.sha256(delexicalized.encode("utf-8")).hexdigest()


def _report_metrics(text: str) -> dict[str, int | float | str]:
    normalized = normalize_report(text)
    tokens = _word_tokens(text)
    lexical_tokens = _lexical_tokens(text)
    letters = [character for character in text if character.isalpha()]
    characters = max(len(text), 1)
    lines = text.splitlines() or [text]
    nonempty_lines = [line for line in lines if line.strip()]
    sentence_parts = [
        part.strip() for part in _SENTENCE_BOUNDARY_RE.split(text) if part.strip()
    ]
    paragraph_parts = [
        part.strip() for part in _PARAGRAPH_BOUNDARY_RE.split(text.strip()) if part.strip()
    ]
    heading_count = len(_HEADING_RE.findall(text))

    mixed_script_token_count = 0
    script_types: set[str] = set()
    for token in tokens:
        token_scripts = {
            script
            for character in token
            if (script := _script_of_character(character)) is not None
        }
        script_types.update(token_scripts)
        mixed_script_token_count += len(token_scripts) > 1

    repeated_alnum = list(_REPEATED_ALNUM_RE.finditer(text.casefold()))
    repeated_punctuation = list(_REPEATED_PUNCTUATION_RE.finditer(text))
    long_tokens = [token for token in tokens if len(token) >= 20]
    alphanumeric_tokens = [
        token
        for token in tokens
        if any(character.isalpha() for character in token)
        and any(character.isdigit() for character in token)
    ]
    token_lengths = [len(token) for token in tokens]

    return {
        "dominant_script": dominant_script(text),
        "script_type_count": len(script_types),
        "mixed_script_token_count": mixed_script_token_count,
        "mixed_script_token_rate": mixed_script_token_count / max(len(tokens), 1),
        "report_char_count": len(text),
        "report_normalized_char_count": len(normalized),
        "report_word_count": len(tokens),
        "report_lexical_token_count": len(lexical_tokens),
        "report_line_count": len(lines),
        "report_nonempty_line_count": len(nonempty_lines),
        "report_sentence_count": max(len(sentence_parts), 1),
        "report_paragraph_count": max(len(paragraph_parts), 1),
        "report_heading_count": heading_count,
        "report_section_count": max(heading_count, len(paragraph_parts), 1),
        "report_placeholder_count": len(_PLACEHOLDER_RE.findall(text)),
        "report_avg_token_length": float(np.mean(token_lengths)) if token_lengths else 0.0,
        "report_max_token_length": max(token_lengths, default=0),
        "report_digit_ratio": sum(character.isdigit() for character in text) / characters,
        "report_punctuation_ratio": sum(
            unicodedata.category(character).startswith("P") for character in text
        )
        / characters,
        "report_uppercase_ratio": sum(character.isupper() for character in letters)
        / max(len(letters), 1),
        "report_whitespace_ratio": sum(character.isspace() for character in text)
        / characters,
        "report_newline_ratio": (text.count("\n") + text.count("\r")) / characters,
        "orth_unicode_replacement_count": text.count("\ufffd"),
        "orth_mojibake_marker_count": len(_MOJIBAKE_RE.findall(text)),
        "orth_control_count": sum(
            unicodedata.category(character) == "Cc" and character not in "\n\r\t"
            for character in text
        ),
        "orth_zero_width_count": sum(character in _ZERO_WIDTH for character in text),
        "orth_surrogate_count": sum(
            unicodedata.category(character) == "Cs" for character in text
        ),
        "orth_repeated_character_run_count": len(repeated_alnum),
        "orth_max_repeated_character_run": max(
            (len(match.group()) for match in repeated_alnum), default=0
        ),
        "orth_repeated_punctuation_run_count": len(repeated_punctuation),
        "orth_suspicious_long_token_count": len(long_tokens),
        "orth_suspicious_long_token_rate": len(long_tokens) / max(len(tokens), 1),
        "orth_alphanumeric_token_count": len(alphanumeric_tokens),
        "orth_alphanumeric_token_rate": len(alphanumeric_tokens) / max(len(tokens), 1),
    }


def _within_language_rare_token_features(
    reports: Sequence[str], languages: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, int]]]:
    rare_count = np.zeros(len(reports), dtype=np.int64)
    rare_rate = np.zeros(len(reports), dtype=np.float64)
    diagnostics: dict[str, dict[str, int]] = {}
    tokenized = [_lexical_tokens(text) for text in reports]
    for language in sorted(set(languages.tolist())):
        indices = np.flatnonzero(languages == language)
        document_frequency: Counter[str] = Counter()
        for index in indices:
            document_frequency.update(set(tokenized[index]))
        for index in indices:
            count = sum(
                document_frequency[token] <= RARE_DOCUMENT_FREQUENCY_MAX
                for token in tokenized[index]
            )
            rare_count[index] = count
            rare_rate[index] = count / max(len(tokenized[index]), 1)
        diagnostics[str(language)] = {
            "studies": int(len(indices)),
            "vocabulary_size": int(len(document_frequency)),
            "rare_vocabulary_size": int(
                sum(
                    frequency <= RARE_DOCUMENT_FREQUENCY_MAX
                    for frequency in document_frequency.values()
                )
            ),
        }
    return rare_count, rare_rate, diagnostics


def _within_language_char_ngram_features(
    reports: Sequence[str], languages: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, int | float]]]:
    """Compute leave-one-out centroid distance within each language proxy."""

    outlier = np.zeros(len(reports), dtype=np.float64)
    nonzero = np.zeros(len(reports), dtype=np.int64)
    diagnostics: dict[str, dict[str, int | float]] = {}
    normalized = [normalize_report(text) for text in reports]
    for language in sorted(set(languages.tolist())):
        indices = np.flatnonzero(languages == language)
        corpus = [normalized[index] for index in indices]
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1 if len(indices) == 1 else 2,
            max_features=20_000,
            lowercase=False,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float64,
        )
        matrix = vectorizer.fit_transform(corpus).tocsr()
        nonzero[indices] = matrix.getnnz(axis=1)

        if len(indices) > 1 and matrix.shape[1] > 0:
            centroid_sum = np.asarray(matrix.sum(axis=0), dtype=np.float64).ravel()
            row_dot_sum = np.asarray(matrix @ centroid_sum, dtype=np.float64).ravel()
            row_norm_sq = np.asarray(
                matrix.multiply(matrix).sum(axis=1), dtype=np.float64
            ).ravel()
            dot_other = row_dot_sum - row_norm_sq
            other_norm_sq = np.maximum(
                float(np.dot(centroid_sum, centroid_sum))
                - 2.0 * row_dot_sum
                + row_norm_sq,
                0.0,
            )
            denominator = np.sqrt(np.maximum(row_norm_sq * other_norm_sq, 0.0))
            similarity = np.ones(len(indices), dtype=np.float64)
            usable = denominator > np.finfo(np.float64).eps
            similarity[usable] = dot_other[usable] / denominator[usable]
            outlier[indices] = 1.0 - np.clip(similarity, 0.0, 1.0)

        diagnostics[str(language)] = {
            "studies": int(len(indices)),
            "vocabulary_size": int(matrix.shape[1]),
            "mean_nonzero_features": float(np.mean(matrix.getnnz(axis=1))),
            "mean_outlier_score": float(np.mean(outlier[indices])),
        }
    return outlier, nonzero, diagnostics


def _nearest_delexicalized_similarity(
    delexicalized: Sequence[str],
) -> tuple[np.ndarray, dict[str, int | float]]:
    if len(delexicalized) == 1:
        return np.zeros(1, dtype=np.float64), {
            "vocabulary_size": 0,
            "mean_similarity": 0.0,
        }
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=20_000,
        lowercase=False,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float64,
    )
    matrix = vectorizer.fit_transform(delexicalized).tocsr()
    neighbours = NearestNeighbors(
        n_neighbors=2, algorithm="brute", metric="cosine", n_jobs=1
    )
    neighbours.fit(matrix)
    distances, _ = neighbours.kneighbors(matrix, return_distance=True)
    similarity = np.clip(1.0 - distances[:, 1], 0.0, 1.0).astype(np.float64)
    return similarity, {
        "vocabulary_size": int(matrix.shape[1]),
        "mean_similarity": float(np.mean(similarity)),
    }


def _collapse_language_map(language_map: pd.DataFrame) -> pd.DataFrame:
    columns = (ID_COLUMN, "language_proxy", "language_proxy_confidence")
    _require_columns(language_map, set(columns), name="language_map")
    frame = _clean_uid_column(language_map[list(columns)], name="language_map")
    if frame[["language_proxy", "language_proxy_confidence"]].isna().any().any():
        raise ValueError("language_map has missing proxy values")
    grouped = frame.groupby(ID_COLUMN, sort=False)
    drift = {
        column: int(grouped[column].nunique(dropna=False).gt(1).sum())
        for column in ("language_proxy", "language_proxy_confidence")
    }
    if any(drift.values()):
        raise ValueError(f"language_map has within-study drift: {drift}")
    result = frame.sort_values(ID_COLUMN, kind="mergesort").drop_duplicates(ID_COLUMN)
    result["language_proxy"] = result["language_proxy"].astype(str)
    confidence = pd.to_numeric(result["language_proxy_confidence"], errors="raise")
    if not np.isfinite(confidence).all() or not confidence.between(0.0, 1.0).all():
        raise ValueError("language proxy confidence fails finite/range gate")
    result["language_proxy_confidence"] = confidence.astype(np.float64)
    return result


def _prepare_prior_features(prior_features: pd.DataFrame) -> pd.DataFrame:
    columns = (
        ID_COLUMN,
        "report_group",
        "report_group_size",
        "dominant_script",
        "report_style_cluster_32",
        "report_style_cluster_size",
    )
    _require_columns(prior_features, set(columns), name="prior_features")
    result = _clean_uid_column(prior_features[list(columns)], name="prior_features")
    if result.duplicated(ID_COLUMN).any():
        raise ValueError("prior_features has duplicate study identifiers")
    if result[list(columns[1:])].isna().any().any():
        raise ValueError("prior_features has missing required values")
    return result.rename(
        columns={
            "report_style_cluster_32": "prior_style_cluster_32",
            "report_style_cluster_size": "prior_style_cluster_size",
        }
    )


def _prepare_style_clusters(cluster_assignments: pd.DataFrame) -> pd.DataFrame:
    columns = (
        ID_COLUMN,
        "cluster_view",
        "language_proxy",
        "cluster_id",
        "cluster_size",
        "cluster_group_size",
        "distance",
        "margin",
        "stable",
        "association_eligible",
    )
    _require_columns(cluster_assignments, set(columns), name="cluster_assignments")
    frame = _clean_uid_column(
        cluster_assignments[list(columns)], name="cluster_assignments"
    )
    frame = frame.loc[frame["cluster_view"].eq(STYLE_CLUSTER_VIEW)].copy()
    if frame.empty:
        raise ValueError(f"cluster_assignments lacks view {STYLE_CLUSTER_VIEW!r}")
    if frame.duplicated(ID_COLUMN).any():
        raise ValueError("style cluster assignments have duplicate study identifiers")
    required_without_margin = [column for column in columns if column != "margin"]
    if frame[required_without_margin].isna().any().any():
        raise ValueError("style cluster assignments have missing required values")
    # A margin is structurally undefined when a language stratum has only one
    # candidate cluster.  Preserve that state explicitly while keeping the
    # numeric feature matrix finite.
    frame["style_cluster_margin_available"] = frame["margin"].notna()
    frame["margin"] = frame["margin"].fillna(0.0)

    for column in ("cluster_size", "cluster_group_size", "distance", "margin"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column]).all():
            raise ValueError(f"style cluster {column} fails finite gate")
    if (frame[["cluster_size", "cluster_group_size"]] < 1).any().any():
        raise ValueError("style cluster sizes must be positive")
    if (frame[["distance", "margin"]] < 0).any().any():
        raise ValueError("style cluster distance/margin must be nonnegative")
    return frame.rename(
        columns={
            "cluster_id": "style_cluster_id",
            "cluster_size": "style_cluster_size",
            "cluster_group_size": "style_cluster_group_size",
            "distance": "style_cluster_distance",
            "margin": "style_cluster_margin",
            "stable": "style_cluster_stable",
            "association_eligible": "style_cluster_association_eligible",
            "language_proxy": "style_cluster_language_proxy",
        }
    ).drop(columns="cluster_view")


def _validate_feature_output(features: pd.DataFrame) -> None:
    forbidden = {
        REPORT_COLUMN,
        "report_normalized",
        "delexicalized_report",
        "report_text",
        "raw_report",
    }
    present = sorted(forbidden.intersection(features.columns))
    if present:
        raise ValueError(f"raw/derived report text columns escaped: {present}")
    if features.isna().any().any():
        bad = features.columns[features.isna().any()].tolist()
        raise ValueError(f"morphology features contain missing values: {bad}")
    numeric = features.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("morphology numeric features fail finite gate")
    bounded_suffixes = ("_ratio", "_rate", "_similarity", "_outlier_score")
    bounded = [column for column in numeric if column.endswith(bounded_suffixes)]
    if bounded and not (
        (numeric[bounded] >= -1e-12) & (numeric[bounded] <= 1.0 + 1e-12)
    ).all().all():
        raise ValueError("morphology ratio/similarity features fail range gate")


def build_report_morphology_features(
    manifest_df: pd.DataFrame,
    language_map: pd.DataFrame,
    prior_features: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one deterministic, text-free feature row per study.

    Inputs may arrive in any row order.  The output is sorted by
    ``StudyInstanceUID``.  ``language_map`` may be a long study-target panel,
    but every repeated study value must be invariant.  Exact UID coverage and
    the frozen report/style identities are fail-closed.
    """

    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    _require_columns(
        manifest_df, {ID_COLUMN, REPORT_COLUMN, "report_group"}, name="manifest_df"
    )
    manifest = _clean_uid_column(
        manifest_df[[ID_COLUMN, REPORT_COLUMN, "report_group"]], name="manifest_df"
    )
    if manifest.duplicated(ID_COLUMN).any():
        raise ValueError("manifest_df has duplicate study identifiers")
    if manifest[REPORT_COLUMN].isna().any():
        raise ValueError("manifest_df has missing reports")
    manifest[REPORT_COLUMN] = manifest[REPORT_COLUMN].astype(str)
    if manifest[REPORT_COLUMN].map(normalize_report).eq("").any():
        raise ValueError("manifest_df has empty normalized reports")
    manifest = manifest.sort_values(ID_COLUMN, kind="mergesort").reset_index(drop=True)
    expected_uids = set(manifest[ID_COLUMN])

    language = _collapse_language_map(language_map)
    prior = _prepare_prior_features(prior_features)
    style = _prepare_style_clusters(cluster_assignments)
    for frame, name in (
        (language, "language_map"),
        (prior, "prior_features"),
        (style, "cluster_assignments/style view"),
    ):
        _require_exact_uid_set(frame, expected_uids, name=name)

    reports = manifest[REPORT_COLUMN].tolist()
    computed_report_groups = [report_hash(text) for text in reports]
    if computed_report_groups != manifest["report_group"].astype(str).tolist():
        raise ValueError("manifest report_group does not match normalized report SHA-256")
    report_group_sizes = pd.Series(computed_report_groups).map(
        Counter(computed_report_groups)
    )

    rows = [_report_metrics(text) for text in reports]
    features = pd.DataFrame(rows)
    features.insert(0, ID_COLUMN, manifest[ID_COLUMN].to_numpy())
    features.insert(1, "report_group", computed_report_groups)
    features.insert(2, "report_group_size", report_group_sizes.to_numpy(dtype=np.int64))

    delexicalized = [delexicalize_style(text) for text in reports]
    template_keys = [
        hashlib.sha256(value.encode("utf-8")).hexdigest() for value in delexicalized
    ]
    template_counts = Counter(template_keys)
    features["delexicalized_template_key"] = template_keys
    features["delexicalized_template_group_size"] = [
        template_counts[key] for key in template_keys
    ]
    nearest_similarity, nearest_diagnostics = _nearest_delexicalized_similarity(
        delexicalized
    )
    features["delexicalized_nn_char_similarity"] = nearest_similarity

    features = features.merge(language, on=ID_COLUMN, how="left", validate="one_to_one")
    languages = features["language_proxy"].astype(str).to_numpy()
    rare_count, rare_rate, rare_diagnostics = _within_language_rare_token_features(
        reports, languages
    )
    char_outlier, char_nonzero, char_diagnostics = (
        _within_language_char_ngram_features(delexicalized, languages)
    )
    features["orth_rare_token_count"] = rare_count
    features["orth_rare_token_rate"] = rare_rate
    features["orth_char_ngram_nonzero_features"] = char_nonzero
    features["orth_char_ngram_outlier_score"] = char_outlier

    features = features.merge(prior, on=ID_COLUMN, how="left", validate="one_to_one")
    if not features["report_group_x"].eq(features["report_group_y"]).all():
        raise ValueError("prior_features report_group identity drift")
    if not features["report_group_size_x"].eq(features["report_group_size_y"]).all():
        raise ValueError("prior_features report_group_size drift")
    if not features["dominant_script_x"].eq(features["dominant_script_y"]).all():
        raise ValueError("prior_features dominant_script drift")
    features = features.rename(
        columns={
            "report_group_x": "report_group",
            "report_group_size_x": "report_group_size",
            "dominant_script_x": "dominant_script",
        }
    ).drop(columns=["report_group_y", "report_group_size_y", "dominant_script_y"])

    features = features.merge(style, on=ID_COLUMN, how="left", validate="one_to_one")
    if not features["language_proxy"].eq(features["style_cluster_language_proxy"]).all():
        raise ValueError("style cluster language proxy drift")
    features = features.drop(columns="style_cluster_language_proxy")

    boolean_columns = [
        "style_cluster_stable",
        "style_cluster_association_eligible",
        "style_cluster_margin_available",
    ]
    for column in boolean_columns:
        features[column] = _coerce_boolean(features[column], name=column)
    integer_columns = [
        column
        for column in features.columns
        if column.endswith("_count")
        or column.endswith("_size")
        or column in {
            "script_type_count",
            "orth_max_repeated_character_run",
            "orth_char_ngram_nonzero_features",
        }
    ]
    for column in integer_columns:
        features[column] = pd.to_numeric(features[column], errors="raise").astype(np.int64)

    features = features.sort_values(ID_COLUMN, kind="mergesort").reset_index(drop=True)
    _validate_feature_output(features)

    numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
    diagnostics: dict[str, Any] = {
        "schema_version": "report-morphology-features-v1",
        "seed": seed,
        "study_count": int(len(features)),
        "uid_sequence_sha256": _sha256_lines(features[ID_COLUMN]),
        "report_group_sequence_sha256": _sha256_lines(features["report_group"]),
        "feature_column_count": int(len(features.columns) - 1),
        "numeric_feature_count": int(len(numeric_columns)),
        "finite_numeric_gate": "PASS",
        "raw_report_text_emitted": False,
        "output_columns": features.columns.tolist(),
        "language_proxy_counts": {
            str(key): int(value)
            for key, value in features["language_proxy"].value_counts(sort=False).sort_index().items()
        },
        "exact_report_group_count": int(features["report_group"].nunique()),
        "delexicalized_template_group_count": int(
            features["delexicalized_template_key"].nunique()
        ),
        "rare_token_document_frequency_max": RARE_DOCUMENT_FREQUENCY_MAX,
        "within_language_rare_token": rare_diagnostics,
        "within_language_char_ngram": char_diagnostics,
        "delexicalized_nearest_neighbour": nearest_diagnostics,
        "style_cluster_view": STYLE_CLUSTER_VIEW,
    }
    return features, diagnostics


__all__ = [
    "DEFAULT_SEED",
    "STYLE_CLUSTER_VIEW",
    "build_report_morphology_features",
    "delexicalized_template_key",
]
