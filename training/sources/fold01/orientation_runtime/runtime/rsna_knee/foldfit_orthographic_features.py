"""Fold-fitted, outcome-blind orthographic anomaly features.

The transformer in this module is intentionally narrower than the global
report-morphology builder.  It fits a token document-frequency reference on
*training report groups only* and applies that immutable reference to held-out
groups.  Gold labels, teacher outputs, and image-model outputs are never read.

The two primary features are morphology proxies, not verified spelling errors:

``foldfit_edit1_rare_token_count``
    Number of eligible token occurrences whose training document frequency is
    at most two and that are exactly one insertion, deletion, substitution, or
    adjacent transposition from a common training token in the same language
    partition.

``foldfit_edit1_rare_token_rate``
    The count divided by all NFKC/casefolded alphabetic token occurrences with
    lengths in the inclusive range 4..30.  A zero denominator maps to 0.0 and
    is recorded by the auxiliary ``foldfit_edit1_zero_denominator`` flag.

Raw or normalized report text and raw token strings are never returned by the
public diagnostics.  The fitted object retains raw tokens privately because
exact edit-neighbour lookup cannot be performed from cryptographic hashes.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import re
import unicodedata
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .constants import ID_COLUMN, REPORT_COLUMN
from .reports import normalize_report


REPORT_GROUP_COLUMN = "report_group"
LANGUAGE_COLUMN = "language_proxy"

TOKEN_LENGTH_MINIMUM = 4
TOKEN_LENGTH_MAXIMUM = 30
COMMON_DOCUMENT_FREQUENCY_MINIMUM = 20
RARE_DOCUMENT_FREQUENCY_MAXIMUM = 2

COUNT_COLUMN = "foldfit_edit1_rare_token_count"
RATE_COLUMN = "foldfit_edit1_rare_token_rate"
ZERO_DENOMINATOR_COLUMN = "foldfit_edit1_zero_denominator"
FEATURE_COLUMNS: tuple[str, ...] = (COUNT_COLUMN, RATE_COLUMN)
AUXILIARY_COLUMNS: tuple[str, ...] = (ZERO_DENOMINATOR_COLUMN,)

_ALPHABETIC_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_partition(value: object) -> str:
    if value is None or pd.isna(value):
        raise ValueError(f"{LANGUAGE_COLUMN} has missing values")
    normalized = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    if not normalized:
        raise ValueError(f"{LANGUAGE_COLUMN} has empty values")
    return normalized


def _normalize_identifier(value: object, *, column: str) -> str:
    if value is None or pd.isna(value):
        raise ValueError(f"{column} has missing values")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{column} has empty values")
    return normalized


def _eligible_alphabetic_tokens(value: object) -> list[str]:
    if value is None or pd.isna(value):
        raise ValueError(f"{REPORT_COLUMN} has missing values")
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return [
        match.group()
        for match in _ALPHABETIC_TOKEN_RE.finditer(normalized)
        if TOKEN_LENGTH_MINIMUM <= len(match.group()) <= TOKEN_LENGTH_MAXIMUM
    ]


def _is_exactly_one_edit_neighbour(left: str, right: str) -> bool:
    """Return whether two tokens differ by one frozen orthographic edit.

    The accepted operations are insertion, deletion, substitution, and an
    adjacent transposition.  Equality and every distance greater than one are
    rejected.
    """

    left_length = len(left)
    right_length = len(right)
    if abs(left_length - right_length) > 1 or left == right:
        return False

    if left_length == right_length:
        differing = [
            index
            for index, (left_character, right_character) in enumerate(
                zip(left, right, strict=True)
            )
            if left_character != right_character
        ]
        if len(differing) == 1:
            return True
        if len(differing) != 2:
            return False
        first, second = differing
        return (
            second == first + 1
            and left[first] == right[second]
            and left[second] == right[first]
        )

    shorter, longer = (left, right) if left_length < right_length else (right, left)
    shorter_index = 0
    longer_index = 0
    skipped = False
    while shorter_index < len(shorter) and longer_index < len(longer):
        if shorter[shorter_index] == longer[longer_index]:
            shorter_index += 1
            longer_index += 1
            continue
        if skipped:
            return False
        skipped = True
        longer_index += 1
    # Length differs by exactly one, so either the mismatch above or a trailing
    # character is the single insertion/deletion.
    return True


@dataclass(frozen=True)
class _LanguageReference:
    document_frequency: Mapping[str, int]
    common_tokens: frozenset[str]
    deletion_index: Mapping[str, tuple[str, ...]]
    report_group_count: int

    def has_common_edit1_neighbour(self, token: str) -> bool:
        candidates: set[str] = set(self.deletion_index.get(token, ()))
        for index in range(len(token)):
            deletion = token[:index] + token[index + 1 :]
            if deletion in self.common_tokens:
                candidates.add(deletion)
            candidates.update(self.deletion_index.get(deletion, ()))
        return any(
            _is_exactly_one_edit_neighbour(token, candidate)
            for candidate in sorted(candidates)
        )


def _build_language_reference(
    documents: Iterable[tuple[str, list[str]]],
) -> _LanguageReference:
    document_frequency: Counter[str] = Counter()
    group_count = 0
    for _group, tokens in documents:
        group_count += 1
        document_frequency.update(set(tokens))

    common_tokens = frozenset(
        token
        for token, frequency in document_frequency.items()
        if frequency >= COMMON_DOCUMENT_FREQUENCY_MINIMUM
    )
    deletion_index_mutable: dict[str, set[str]] = defaultdict(set)
    for token in sorted(common_tokens):
        for index in range(len(token)):
            deletion_index_mutable[token[:index] + token[index + 1 :]].add(token)
    deletion_index = {
        deletion: tuple(sorted(tokens))
        for deletion, tokens in sorted(deletion_index_mutable.items())
    }
    return _LanguageReference(
        document_frequency=dict(sorted(document_frequency.items())),
        common_tokens=common_tokens,
        deletion_index=deletion_index,
        report_group_count=group_count,
    )


class FoldFittedOrthographicAnomalyTransformer:
    """Fit the frozen edit-one rarity reference and transform pandas frames.

    ``fit`` reads only ``StudyInstanceUID``, ``Report``, ``report_group``, and
    ``language_proxy``.  Extra columns are deliberately ignored, so changing a
    Gold, teacher, or image outcome column cannot change fitted state or output.

    Ordinary ``transform`` is reserved for held-out report groups and fails on
    any overlap with the fit groups.  ``fit_transform`` and
    ``transform_training`` provide training-fold features, but the latter
    requires the exact same order-insensitive source identity as the fit frame.
    """

    def __init__(self) -> None:
        self._fitted = False

    @staticmethod
    def _validated_source(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame")
        required = {
            ID_COLUMN,
            REPORT_COLUMN,
            REPORT_GROUP_COLUMN,
            LANGUAGE_COLUMN,
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")

        # This exact projection is the outcome-blind boundary.  No value from
        # any extra input column is inspected or serialized.
        source = frame[
            [ID_COLUMN, REPORT_COLUMN, REPORT_GROUP_COLUMN, LANGUAGE_COLUMN]
        ].copy()
        source[ID_COLUMN] = [
            _normalize_identifier(value, column=ID_COLUMN)
            for value in source[ID_COLUMN]
        ]
        source[REPORT_GROUP_COLUMN] = [
            _normalize_identifier(value, column=REPORT_GROUP_COLUMN)
            for value in source[REPORT_GROUP_COLUMN]
        ]
        source[LANGUAGE_COLUMN] = [
            _normalize_partition(value) for value in source[LANGUAGE_COLUMN]
        ]
        if source[ID_COLUMN].duplicated().any():
            duplicates = int(source[ID_COLUMN].duplicated(keep=False).sum())
            raise ValueError(f"{name} has duplicate study identifiers: {duplicates}")
        if source[REPORT_COLUMN].isna().any():
            raise ValueError(f"{name} has missing {REPORT_COLUMN} values")
        source[REPORT_COLUMN] = source[REPORT_COLUMN].astype(str)

        # A report group is the leakage/document unit.  It must not combine
        # different normalized reports or language partitions.
        group_drift = (
            source.assign(
                _normalized_report=source[REPORT_COLUMN].map(normalize_report)
            )
            .groupby(REPORT_GROUP_COLUMN, sort=True, observed=True)[
                ["_normalized_report", LANGUAGE_COLUMN]
            ]
            .nunique(dropna=False)
        )
        if (group_drift > 1).any().any():
            bad_groups = int((group_drift > 1).any(axis=1).sum())
            raise ValueError(
                f"{name} has report/language drift within {bad_groups} report groups"
            )
        return source.sort_values(ID_COLUMN, kind="mergesort").reset_index(drop=True)

    @staticmethod
    def _source_identity_sha256(source: pd.DataFrame) -> str:
        records = [
            {
                ID_COLUMN: row[ID_COLUMN],
                REPORT_COLUMN: row[REPORT_COLUMN],
                REPORT_GROUP_COLUMN: row[REPORT_GROUP_COLUMN],
                LANGUAGE_COLUMN: row[LANGUAGE_COLUMN],
            }
            for row in source.to_dict(orient="records")
        ]
        return _canonical_json_sha256(records)

    def fit(self, train_frame: pd.DataFrame) -> "FoldFittedOrthographicAnomalyTransformer":
        # A failed re-fit must not leave the previously fitted reference usable.
        self._fitted = False
        source = self._validated_source(train_frame, name="train_frame")
        if source.empty:
            raise ValueError("train_frame must contain at least one study")

        group_documents: list[dict[str, Any]] = []
        for group, rows in source.groupby(REPORT_GROUP_COLUMN, sort=True, observed=True):
            representative = rows.iloc[0]
            group_documents.append(
                {
                    REPORT_GROUP_COLUMN: str(group),
                    LANGUAGE_COLUMN: str(representative[LANGUAGE_COLUMN]),
                    "tokens": _eligible_alphabetic_tokens(
                        representative[REPORT_COLUMN]
                    ),
                }
            )

        references: dict[str, _LanguageReference] = {}
        by_language: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
        for document in group_documents:
            by_language[document[LANGUAGE_COLUMN]].append(
                (document[REPORT_GROUP_COLUMN], document["tokens"])
            )
        for language in sorted(by_language):
            references[language] = _build_language_reference(by_language[language])

        token_hashes = {
            language: tuple(
                sorted(_text_sha256(token) for token in reference.document_frequency)
            )
            for language, reference in sorted(references.items())
        }
        reference_payload = {
            "normalization": "Unicode NFKC plus casefold; alphabetic tokens only",
            "token_length_inclusive": [
                TOKEN_LENGTH_MINIMUM,
                TOKEN_LENGTH_MAXIMUM,
            ],
            "common_document_frequency_minimum": (
                COMMON_DOCUMENT_FREQUENCY_MINIMUM
            ),
            "rare_document_frequency_maximum": RARE_DOCUMENT_FREQUENCY_MAXIMUM,
            "document_unit": REPORT_GROUP_COLUMN,
            "fit_source_sha256": self._source_identity_sha256(source),
            "fit_group_hashes": sorted(
                _text_sha256(group) for group in source[REPORT_GROUP_COLUMN].unique()
            ),
            "partitions": {
                language: [
                    [_text_sha256(token), int(frequency)]
                    for token, frequency in sorted(
                        reference.document_frequency.items(),
                        key=lambda item: _text_sha256(item[0]),
                    )
                ]
                for language, reference in sorted(references.items())
            },
        }

        self._references = references
        self._fit_groups = frozenset(source[REPORT_GROUP_COLUMN])
        self._fit_source_sha256 = self._source_identity_sha256(source)
        self._training_token_hashes = token_hashes
        self.reference_sha256_ = _canonical_json_sha256(reference_payload)
        self.fit_diagnostics_ = {
            "status": "PASS",
            "fit_rows": int(len(source)),
            "fit_report_groups": int(len(self._fit_groups)),
            "language_partitions": {
                language: {
                    "report_groups": int(reference.report_group_count),
                    "vocabulary_size": int(len(reference.document_frequency)),
                    "common_vocabulary_size": int(len(reference.common_tokens)),
                    "rare_vocabulary_size": int(
                        sum(
                            frequency <= RARE_DOCUMENT_FREQUENCY_MAXIMUM
                            for frequency in reference.document_frequency.values()
                        )
                    ),
                }
                for language, reference in sorted(references.items())
            },
            "reference_sha256": self.reference_sha256_,
            "raw_report_text_emitted": False,
            "raw_token_text_emitted": False,
            "extra_columns_read": False,
            "outcome_data_used": False,
        }
        self._fitted = True
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return bool(self._fitted)

    @property
    def training_token_hashes_(self) -> dict[str, tuple[str, ...]]:
        self._require_fitted()
        return deepcopy(self._training_token_hashes)

    def get_feature_names_out(
        self, input_features: object | None = None
    ) -> np.ndarray:
        del input_features
        return np.asarray(FEATURE_COLUMNS + AUXILIARY_COLUMNS, dtype=object)

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Transformer must be fitted before transform")

    def _transform_source(
        self,
        source: pd.DataFrame,
        *,
        allow_training_identity: bool,
    ) -> pd.DataFrame:
        self._require_fitted()
        overlap = self._fit_groups.intersection(source[REPORT_GROUP_COLUMN])
        if overlap and not allow_training_identity:
            raise ValueError(
                "apply_frame overlaps fitted report groups; use "
                "transform_training only for the exact fitted source"
            )
        if allow_training_identity:
            identity = self._source_identity_sha256(source)
            if identity != self._fit_source_sha256:
                raise ValueError(
                    "training transform source identity does not match the fitted source"
                )

        counts: list[int] = []
        rates: list[float] = []
        zero_denominators: list[int] = []
        unseen_language_rows = 0
        for row in source.itertuples(index=False, name=None):
            # Column order is frozen by _validated_source.
            _uid, report, _group, language = row
            tokens = _eligible_alphabetic_tokens(report)
            denominator = len(tokens)
            reference = self._references.get(language)
            if reference is None:
                count = 0
                unseen_language_rows += 1
            else:
                count = sum(
                    reference.document_frequency.get(token, 0)
                    <= RARE_DOCUMENT_FREQUENCY_MAXIMUM
                    and reference.has_common_edit1_neighbour(token)
                    for token in tokens
                )
            rate = float(count / denominator) if denominator else 0.0
            counts.append(int(count))
            rates.append(rate)
            zero_denominators.append(int(denominator == 0))

        output = pd.DataFrame(
            {
                ID_COLUMN: source[ID_COLUMN].to_numpy(copy=True),
                COUNT_COLUMN: np.asarray(counts, dtype=np.int64),
                RATE_COLUMN: np.asarray(rates, dtype=np.float64),
                ZERO_DENOMINATOR_COLUMN: np.asarray(
                    zero_denominators, dtype=np.int8
                ),
            }
        )
        numeric = output[list(FEATURE_COLUMNS + AUXILIARY_COLUMNS)].to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(numeric).all():
            raise ValueError("orthographic feature finite-value gate failed")
        if (output[COUNT_COLUMN] < 0).any() or not output[RATE_COLUMN].between(
            0.0, 1.0, inclusive="both"
        ).all():
            raise ValueError("orthographic feature range gate failed")

        self.last_transform_diagnostics_ = {
            "status": "PASS",
            "rows": int(len(output)),
            "report_groups": int(source[REPORT_GROUP_COLUMN].nunique()),
            "unseen_language_rows": int(unseen_language_rows),
            "zero_denominator_rows": int(sum(zero_denominators)),
            "rare_token_occurrences": int(sum(counts)),
            "finite_value_gate": "PASS",
            "raw_report_text_emitted": False,
            "raw_token_text_emitted": False,
            "reference_sha256": self.reference_sha256_,
        }
        return output

    def transform(self, apply_frame: pd.DataFrame) -> pd.DataFrame:
        source = self._validated_source(apply_frame, name="apply_frame")
        return self._transform_source(source, allow_training_identity=False)

    def transform_training(self, train_frame: pd.DataFrame) -> pd.DataFrame:
        source = self._validated_source(train_frame, name="train_frame")
        return self._transform_source(source, allow_training_identity=True)

    def fit_transform(self, train_frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_frame).transform_training(train_frame)


def fit_fold_fitted_orthographic_anomaly(
    train_frame: pd.DataFrame,
) -> FoldFittedOrthographicAnomalyTransformer:
    """Fit and return the frozen orthographic anomaly transformer."""

    return FoldFittedOrthographicAnomalyTransformer().fit(train_frame)


__all__ = [
    "AUXILIARY_COLUMNS",
    "COMMON_DOCUMENT_FREQUENCY_MINIMUM",
    "COUNT_COLUMN",
    "FEATURE_COLUMNS",
    "FoldFittedOrthographicAnomalyTransformer",
    "LANGUAGE_COLUMN",
    "RATE_COLUMN",
    "RARE_DOCUMENT_FREQUENCY_MAXIMUM",
    "REPORT_GROUP_COLUMN",
    "TOKEN_LENGTH_MAXIMUM",
    "TOKEN_LENGTH_MINIMUM",
    "ZERO_DENOMINATOR_COLUMN",
    "fit_fold_fitted_orthographic_anomaly",
]
