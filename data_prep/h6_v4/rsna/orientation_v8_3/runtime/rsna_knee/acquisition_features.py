"""Deterministic, outcome-blind study-level acquisition features.

The builder in this module joins a series-header table to the official series
manifest only after proving exact ``(StudyInstanceUID, SeriesInstanceUID)``
coverage.  It exposes a deliberately small allowlist of anonymous acquisition
proxies; patient identifiers and sex are never copied to the output.

String modes are computed after Unicode NFKC normalization, surrounding-space
removal, and case folding.  Ties are resolved by the lexicographically smallest
normalized value, making the result invariant to input row order.
"""

from __future__ import annotations

import hashlib
import math
import unicodedata
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .constants import ID_COLUMN, SERIES_COLUMN, SERIES_REQUIRED_COLUMNS


PATIENT_COLUMNS: frozenset[str] = frozenset({"patient_id", "patient_sex"})

_HEADER_COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "manufacturer": ("manufacturer",),
    "model": ("model_name", "manufacturer_model_name"),
    "field_strength": ("field_strength", "magnetic_field_strength"),
}

_PLANE_OUTPUT_COLUMNS: dict[str, str] = {
    "axial": "series_plane_axial_count",
    "coronal": "series_plane_coronal_count",
    "sagittal": "series_plane_sagittal_count",
}


def normalize_acquisition_value(value: object) -> str | None:
    """Return a normalized scalar category or ``None`` for a missing value.

    Normalization is intentionally narrow and auditable: Unicode NFKC,
    ``strip()``, then ``casefold()``.  Internal whitespace and punctuation are
    preserved.  Numeric values are accepted as scalar categories, while
    non-finite numeric values fail closed.
    """

    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return None
        if not math.isfinite(float(value)):
            raise ValueError("acquisition value is non-finite")
    else:
        try:
            missing = pd.isna(value)
        except (TypeError, ValueError):
            missing = False
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return None
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    elif isinstance(value, (str, int, float, bool)):
        text = str(value)
    else:
        raise ValueError(
            "acquisition values must be scalar strings or numbers, "
            f"got {type(value).__name__}"
        )
    normalized = unicodedata.normalize("NFKC", text).strip().casefold()
    return normalized or None


def _require_columns(frame: pd.DataFrame, columns: set[str], *, name: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def _clean_pair_frame(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    _require_columns(frame, {ID_COLUMN, SERIES_COLUMN}, name=name)
    if frame[[ID_COLUMN, SERIES_COLUMN]].isna().any().any():
        raise ValueError(f"{name} has missing series-pair identifiers")
    result = frame.copy()
    for column in (ID_COLUMN, SERIES_COLUMN):
        result[column] = result[column].astype(str)
        if result[column].str.strip().eq("").any():
            raise ValueError(f"{name} has empty {column} values")
    if result.duplicated([ID_COLUMN, SERIES_COLUMN]).any():
        raise ValueError(f"{name} has duplicate study-series pairs")
    if result[SERIES_COLUMN].duplicated().any():
        raise ValueError(f"{name} maps a SeriesInstanceUID to multiple studies")
    return result


def _pair_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(
        map(
            tuple,
            frame[[ID_COLUMN, SERIES_COLUMN]].astype(str).itertuples(
                index=False, name=None
            ),
        )
    )


def _pair_set_sha256(pairs: set[tuple[str, str]]) -> str:
    payload = "".join(f"{study}\t{series}\n" for study, series in sorted(pairs))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve_header_column(
    headers: pd.DataFrame,
    *,
    role: str,
    explicit: str | None,
) -> str:
    if explicit is not None:
        if explicit not in headers:
            raise ValueError(f"series_headers missing requested {role} column: {explicit}")
        return explicit
    for candidate in _HEADER_COLUMN_CANDIDATES[role]:
        if candidate in headers:
            return candidate
    raise ValueError(
        f"series_headers missing {role} column; expected one of "
        f"{list(_HEADER_COLUMN_CANDIDATES[role])}"
    )


def _coerce_binary(series: pd.Series, *, name: str) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"{name} has missing values")
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(np.int64)
    if pd.api.types.is_numeric_dtype(series.dtype):
        numeric = pd.to_numeric(series, errors="raise")
        values = numeric.to_numpy(dtype=float)
        if not np.isfinite(values).all() or not numeric.isin([0, 1]).all():
            raise ValueError(f"{name} must contain only finite 0/1 values")
        return numeric.astype(np.int64)
    normalized = series.map(normalize_acquisition_value)
    mapping = {
        "0": 0,
        "false": 0,
        "no": 0,
        "1": 1,
        "true": 1,
        "yes": 1,
    }
    if normalized.isna().any() or not normalized.isin(mapping).all():
        raise ValueError(f"{name} must contain only boolean or 0/1 values")
    return normalized.map(mapping).astype(np.int64)


def _normalized_mode(values: pd.Series) -> tuple[str | None, int, int]:
    normalized = values.map(normalize_acquisition_value)
    kept = normalized.dropna()
    missing_count = int(normalized.isna().sum())
    if kept.empty:
        return None, 0, missing_count
    counts = kept.value_counts(sort=False)
    maximum = int(counts.max())
    candidates = sorted(str(value) for value in counts[counts.eq(maximum)].index)
    return candidates[0], int(kept.nunique()), missing_count


def _prepare_official_series(train_series: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        train_series,
        set(SERIES_REQUIRED_COLUMNS),
        name="train_series",
    )
    official = _clean_pair_frame(train_series, name="train_series")
    official["_fluid"] = _coerce_binary(
        official["Fluid_Sensitive"], name="train_series.Fluid_Sensitive"
    )
    official["_fat"] = _coerce_binary(
        official["Fat_Suppression"], name="train_series.Fat_Suppression"
    )
    plane = official["Anatomical_Plane"].map(normalize_acquisition_value)
    expected_planes = set(_PLANE_OUTPUT_COLUMNS)
    if plane.isna().any() or not set(plane).issubset(expected_planes):
        unexpected = sorted(set(plane.dropna()).difference(expected_planes))
        raise ValueError(
            "train_series.Anatomical_Plane contains missing or unexpected "
            f"values: {unexpected}"
        )
    official["_plane"] = plane
    return official.sort_values(
        [ID_COLUMN, SERIES_COLUMN], kind="mergesort"
    ).reset_index(drop=True)


def build_study_acquisition_features(
    series_headers: pd.DataFrame,
    train_series: pd.DataFrame,
    *,
    manufacturer_column: str | None = None,
    model_column: str | None = None,
    field_strength_column: str | None = None,
    split_column: str | None = "split",
    train_split_value: str = "train",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate exact-pair series metadata into deterministic study features.

    If ``split_column`` exists in ``series_headers``, only rows whose normalized
    value equals ``train_split_value`` enter the exact-pair check.  This supports
    source tables that contain a small test partition without silently accepting
    arbitrary extra train series.
    """

    official = _prepare_official_series(train_series)
    headers_source = series_headers.copy()
    excluded_non_train_rows = 0
    if split_column is not None and split_column in headers_source:
        split = headers_source[split_column].map(normalize_acquisition_value)
        if split.isna().any():
            raise ValueError(f"series_headers.{split_column} has missing values")
        expected_split = normalize_acquisition_value(train_split_value)
        if expected_split is None:
            raise ValueError("train_split_value must be non-empty")
        keep = split.eq(expected_split)
        excluded_non_train_rows = int((~keep).sum())
        headers_source = headers_source.loc[keep].copy()

    headers = _clean_pair_frame(headers_source, name="series_headers")
    header_columns = {
        "manufacturer": _resolve_header_column(
            headers, role="manufacturer", explicit=manufacturer_column
        ),
        "model": _resolve_header_column(headers, role="model", explicit=model_column),
        "field_strength": _resolve_header_column(
            headers, role="field_strength", explicit=field_strength_column
        ),
    }

    official_pairs = _pair_set(official)
    header_pairs = _pair_set(headers)
    if header_pairs != official_pairs:
        missing = len(official_pairs.difference(header_pairs))
        extra = len(header_pairs.difference(official_pairs))
        raise ValueError(
            "series pair set mismatch: "
            f"missing={missing}, extra={extra}"
        )

    selected_columns = [
        ID_COLUMN,
        SERIES_COLUMN,
        *header_columns.values(),
    ]
    aligned = official.merge(
        headers[selected_columns],
        on=[ID_COLUMN, SERIES_COLUMN],
        how="left",
        validate="one_to_one",
        sort=False,
    )

    rows: list[dict[str, Any]] = []
    for study_uid, part in aligned.groupby(ID_COLUMN, sort=True):
        row: dict[str, Any] = {
            ID_COLUMN: str(study_uid),
            "series_count": int(len(part)),
            "fluid_sensitive_series_count": int(part["_fluid"].sum()),
            "fat_suppression_series_count": int(part["_fat"].sum()),
        }
        for plane, output_column in _PLANE_OUTPUT_COLUMNS.items():
            row[output_column] = int(part["_plane"].eq(plane).sum())
        for role, source_column in header_columns.items():
            mode, distinct, missing_count = _normalized_mode(part[source_column])
            prefix = f"acquisition_{role}"
            row[f"{prefix}_mode"] = mode
            row[f"{prefix}_distinct_count"] = distinct
            row[f"{prefix}_conflict"] = bool(distinct > 1)
            row[f"{prefix}_missing"] = bool(distinct == 0)
            row[f"{prefix}_series_missing_count"] = missing_count
        rows.append(row)

    features = pd.DataFrame(rows).sort_values(ID_COLUMN, kind="mergesort").reset_index(
        drop=True
    )
    if len(features) != official[ID_COLUMN].nunique() or not features[ID_COLUMN].is_unique:
        raise AssertionError("study acquisition output coverage drift")
    plane_counts = features[list(_PLANE_OUTPUT_COLUMNS.values())].sum(axis=1)
    if not plane_counts.eq(features["series_count"]).all():
        raise AssertionError("anatomical-plane counts do not sum to series_count")
    for column in ("fluid_sensitive_series_count", "fat_suppression_series_count"):
        if not features[column].between(0, features["series_count"]).all():
            raise AssertionError(f"{column} lies outside [0, series_count]")
    numeric = features.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("study acquisition numeric finite gate failed")
    if PATIENT_COLUMNS.intersection(features.columns):
        raise AssertionError("patient columns escaped the acquisition allowlist")

    diagnostics: dict[str, Any] = {
        "status": "PASS",
        "study_count": int(len(features)),
        "official_series_count": int(len(official)),
        "header_series_count": int(len(headers)),
        "excluded_non_train_header_rows": excluded_non_train_rows,
        "series_pair_set_exact": True,
        "series_pair_set_sha256": _pair_set_sha256(official_pairs),
        "normalization": "unicode_nfkc_strip_casefold",
        "mode_tie_break": "lexicographically_smallest_normalized_value",
        "header_columns": header_columns,
        "excluded_patient_columns_present": sorted(
            PATIENT_COLUMNS.intersection(series_headers.columns)
        ),
        "patient_columns_emitted": False,
        "outcome_columns_emitted": False,
        "numeric_finite_gate": "PASS",
        "missing_study_counts": {
            role: int(features[f"acquisition_{role}_missing"].sum())
            for role in header_columns
        },
        "conflict_study_counts": {
            role: int(features[f"acquisition_{role}_conflict"].sum())
            for role in header_columns
        },
    }
    return features, diagnostics


__all__: Sequence[str] = (
    "PATIENT_COLUMNS",
    "build_study_acquisition_features",
    "normalize_acquisition_value",
)
