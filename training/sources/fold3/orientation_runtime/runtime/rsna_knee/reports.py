"""Report normalization, leakage grouping, and language/script diagnostics."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from typing import Any

import pandas as pd

from .constants import ID_COLUMN, REPORT_COLUMN

_SPACE_RE = re.compile(r"\s+")


def normalize_report(value: object) -> str:
    """Normalize text conservatively for exact duplicate/template grouping."""

    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return _SPACE_RE.sub(" ", text).strip()


def report_hash(value: object, *, missing_key: str | None = None) -> str:
    normalized = normalize_report(value)
    if not normalized and missing_key is not None:
        normalized = f"__missing__:{missing_key}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def dominant_script(value: object) -> str:
    counts: Counter[str] = Counter()
    for character in normalize_report(value):
        codepoint = ord(character)
        if "a" <= character <= "z" or 0x00C0 <= codepoint <= 0x024F:
            counts["latin"] += 1
        elif 0x0370 <= codepoint <= 0x03FF:
            counts["greek"] += 1
        elif 0x0400 <= codepoint <= 0x052F:
            counts["cyrillic"] += 1
        elif 0x0600 <= codepoint <= 0x06FF:
            counts["arabic"] += 1
        elif character.isalpha():
            counts["other"] += 1
    return counts.most_common(1)[0][0] if counts else "unknown"


def add_report_groups(frame: pd.DataFrame) -> pd.DataFrame:
    if ID_COLUMN not in frame or REPORT_COLUMN not in frame:
        raise ValueError(f"Expected columns {ID_COLUMN!r} and {REPORT_COLUMN!r}")
    result = frame.copy()
    result["report_normalized"] = result[REPORT_COLUMN].map(normalize_report)
    result["report_group"] = [
        report_hash(report, missing_key=str(uid))
        for uid, report in zip(result[ID_COLUMN], result[REPORT_COLUMN], strict=True)
    ]
    return result


def audit_reports(frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    grouped = add_report_groups(frame)
    group_sizes = grouped.groupby("report_group", sort=False).size().rename("group_size")
    duplicate_sizes = group_sizes[group_sizes > 1]
    normalized_lengths = grouped["report_normalized"].str.len()
    scripts = grouped[REPORT_COLUMN].map(dominant_script).value_counts(dropna=False)

    summary: dict[str, Any] = {
        "studies": int(len(grouped)),
        "missing_or_empty_reports": int((normalized_lengths == 0).sum()),
        "unique_normalized_reports": int(grouped["report_group"].nunique()),
        "duplicate_report_groups": int(len(duplicate_sizes)),
        "studies_in_duplicate_groups": int(duplicate_sizes.sum()),
        "largest_duplicate_group": int(duplicate_sizes.max()) if len(duplicate_sizes) else 1,
        "normalized_length": {
            "min": int(normalized_lengths.min()) if len(grouped) else 0,
            "median": float(normalized_lengths.median()) if len(grouped) else 0.0,
            "p95": float(normalized_lengths.quantile(0.95)) if len(grouped) else 0.0,
            "max": int(normalized_lengths.max()) if len(grouped) else 0,
        },
        "dominant_script_counts": {str(key): int(value) for key, value in scripts.items()},
    }

    duplicate_rows = grouped[grouped["report_group"].isin(duplicate_sizes.index)].copy()
    duplicate_rows["group_size"] = duplicate_rows["report_group"].map(group_sizes)
    duplicate_rows = duplicate_rows[
        ["report_group", "group_size", ID_COLUMN, REPORT_COLUMN]
    ].sort_values(["group_size", "report_group", ID_COLUMN], ascending=[False, True, True])
    return summary, duplicate_rows

