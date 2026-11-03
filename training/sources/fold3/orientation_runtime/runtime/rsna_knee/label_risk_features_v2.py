"""Leakage-bounded report features for the v2 label-risk audit.

The edit-neighbour feature is intentionally named a proxy: it detects rare
tokens one edit away from a common token in the *fit* corpus.  It neither
corrects text nor asserts that the token is a typographical error.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


TOKEN_PATTERN = re.compile(r"[^\W\d_]+", flags=re.UNICODE)


def normalized_alpha_tokens(text: object, *, minimum: int = 4, maximum: int = 30) -> list[str]:
    """Return NFKC/casefolded alphabetic tokens with a frozen length range."""

    if not isinstance(minimum, int) or not isinstance(maximum, int):
        raise TypeError("token length bounds must be integers")
    if minimum < 1 or maximum < minimum:
        raise ValueError("invalid token length bounds")
    normalized = unicodedata.normalize("NFKC", "" if text is None else str(text)).casefold()
    return [
        token
        for token in TOKEN_PATTERN.findall(normalized)
        if minimum <= len(token) <= maximum
    ]


@dataclass(frozen=True)
class _LanguageLexicon:
    frequency: Mapping[str, int]
    common: frozenset[str]
    substitution_patterns: frozenset[str]
    common_deletions: frozenset[str]


@dataclass(frozen=True)
class EditNeighbourProxy:
    """Vocabulary fitted only on the declared training reports."""

    lexicons: Mapping[str, _LanguageLexicon]
    common_frequency_minimum: int
    rare_frequency_maximum: int
    minimum_token_length: int
    maximum_token_length: int

    def transform_one(self, report: object, language: object) -> tuple[int, float]:
        language_key = str(language)
        lexicon = self.lexicons.get(language_key)
        tokens = normalized_alpha_tokens(
            report,
            minimum=self.minimum_token_length,
            maximum=self.maximum_token_length,
        )
        if not tokens or lexicon is None:
            return 0, 0.0
        candidate_count = 0
        for token in tokens:
            if int(lexicon.frequency.get(token, 0)) > self.rare_frequency_maximum:
                continue
            if token in lexicon.common:
                continue
            if _has_edit_one_common_neighbour(token, lexicon):
                candidate_count += 1
        return candidate_count, float(candidate_count / len(tokens))

    def transform(
        self,
        reports: Sequence[object] | Iterable[object],
        languages: Sequence[object] | Iterable[object],
    ) -> tuple[np.ndarray, np.ndarray]:
        report_values = list(reports)
        language_values = list(languages)
        if len(report_values) != len(language_values):
            raise ValueError("reports and languages have different lengths")
        counts = np.zeros(len(report_values), dtype=np.int32)
        rates = np.zeros(len(report_values), dtype=np.float64)
        for index, (report, language) in enumerate(
            zip(report_values, language_values, strict=True)
        ):
            counts[index], rates[index] = self.transform_one(report, language)
        if not np.isfinite(rates).all() or np.any((rates < 0.0) | (rates > 1.0)):
            raise AssertionError("edit-neighbour rates are outside [0, 1]")
        return counts, rates


def _substitution_pattern(token: str, position: int) -> str:
    return f"{token[:position]}*{token[position + 1:]}"


def _has_edit_one_common_neighbour(token: str, lexicon: _LanguageLexicon) -> bool:
    # Candidate contains one extra character.
    if any(token[:index] + token[index + 1 :] in lexicon.common for index in range(len(token))):
        return True
    # One substitution.
    if any(
        _substitution_pattern(token, index) in lexicon.substitution_patterns
        for index in range(len(token))
    ):
        return True
    # A common token contains one extra character.
    if token in lexicon.common_deletions:
        return True
    # One adjacent transposition.
    for index in range(len(token) - 1):
        if token[index] == token[index + 1]:
            continue
        swapped = (
            token[:index]
            + token[index + 1]
            + token[index]
            + token[index + 2 :]
        )
        if swapped in lexicon.common:
            return True
    return False


def fit_edit_neighbour_proxy(
    reports: Sequence[object] | Iterable[object],
    languages: Sequence[object] | Iterable[object],
    *,
    common_frequency_minimum: int = 20,
    rare_frequency_maximum: int = 2,
    minimum_token_length: int = 4,
    maximum_token_length: int = 30,
) -> EditNeighbourProxy:
    """Fit language-partitioned common-token dictionaries on fit reports only."""

    if common_frequency_minimum <= rare_frequency_maximum:
        raise ValueError("common frequency must exceed rare frequency")
    if rare_frequency_maximum < 0:
        raise ValueError("rare frequency maximum must be non-negative")
    report_values = list(reports)
    language_values = list(languages)
    if len(report_values) != len(language_values):
        raise ValueError("reports and languages have different lengths")
    if not report_values:
        raise ValueError("fit corpus is empty")
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    for report, language in zip(report_values, language_values, strict=True):
        if language is None or str(language) == "":
            raise ValueError("language key is empty")
        # Frozen definition is document frequency, so repeated mentions inside
        # one report contribute only once to vocabulary fitting.
        counters[str(language)].update(
            set(normalized_alpha_tokens(
                report,
                minimum=minimum_token_length,
                maximum=maximum_token_length,
            ))
        )
    lexicons: dict[str, _LanguageLexicon] = {}
    for language, frequency in sorted(counters.items()):
        common = frozenset(
            token
            for token, count in frequency.items()
            if count >= int(common_frequency_minimum)
        )
        substitutions = frozenset(
            _substitution_pattern(token, index)
            for token in common
            for index in range(len(token))
        )
        deletions = frozenset(
            token[:index] + token[index + 1 :]
            for token in common
            for index in range(len(token))
        )
        lexicons[language] = _LanguageLexicon(
            frequency=dict(frequency),
            common=common,
            substitution_patterns=substitutions,
            common_deletions=deletions,
        )
    return EditNeighbourProxy(
        lexicons=lexicons,
        common_frequency_minimum=int(common_frequency_minimum),
        rare_frequency_maximum=int(rare_frequency_maximum),
        minimum_token_length=int(minimum_token_length),
        maximum_token_length=int(maximum_token_length),
    )


__all__ = ["EditNeighbourProxy", "fit_edit_neighbour_proxy", "normalized_alpha_tokens"]
