"""Competition metadata validation and descriptive audit."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .constants import (
    ID_COLUMN,
    REPORT_COLUMN,
    SERIES_COLUMN,
    SERIES_REQUIRED_COLUMNS,
    TARGET_COLUMNS,
)
from .reports import audit_reports


def _describe_integer_series(values: pd.Series) -> dict[str, float | int]:
    if values.empty:
        return {"min": 0, "median": 0.0, "p95": 0.0, "max": 0}
    return {
        "min": int(values.min()),
        "median": float(values.median()),
        "p95": float(values.quantile(0.95)),
        "max": int(values.max()),
    }


def validate_metadata(train: pd.DataFrame, series: pd.DataFrame) -> None:
    required_train = {ID_COLUMN, REPORT_COLUMN, *TARGET_COLUMNS}
    missing_train = sorted(required_train.difference(train.columns))
    missing_series = sorted(set(SERIES_REQUIRED_COLUMNS).difference(series.columns))
    if missing_train:
        raise ValueError(f"train.csv is missing columns: {missing_train}")
    if missing_series:
        raise ValueError(f"series CSV is missing columns: {missing_series}")
    if train[ID_COLUMN].duplicated().any():
        raise ValueError("train.csv must contain one row per StudyInstanceUID")
    if series[[ID_COLUMN, SERIES_COLUMN]].duplicated().any():
        raise ValueError("series CSV contains duplicate study/series keys")
    unknown_studies = set(series[ID_COLUMN]).difference(train[ID_COLUMN])
    if unknown_studies:
        raise ValueError(f"Series metadata references {len(unknown_studies)} unknown studies")


def audit_metadata(train: pd.DataFrame, series: pd.DataFrame) -> dict[str, Any]:
    validate_metadata(train, series)
    labels = train[TARGET_COLUMNS].apply(pd.to_numeric, errors="coerce")
    invalid_values = int((~labels.isin([0, 1]) & labels.notna()).to_numpy().sum())
    known_per_study = labels.notna().sum(axis=1)
    report_summary, _ = audit_reports(train[[ID_COLUMN, REPORT_COLUMN]])

    series_per_study = series.groupby(ID_COLUMN, sort=False).size()
    missing_series_studies = int((~train[ID_COLUMN].isin(series_per_study.index)).sum())
    plane_counts = series["Anatomical_Plane"].fillna("<missing>").value_counts()
    slot_counts = (
        series.assign(
            Fluid_Sensitive=pd.to_numeric(series["Fluid_Sensitive"], errors="coerce"),
            Fat_Suppression=pd.to_numeric(series["Fat_Suppression"], errors="coerce"),
        )
        .groupby(["Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"], dropna=False)
        .size()
    )

    target_summary: dict[str, Any] = {}
    for target in TARGET_COLUMNS:
        known = labels[target].dropna()
        target_summary[target] = {
            "known": int(len(known)),
            "positive": int((known == 1).sum()),
            "negative": int((known == 0).sum()),
            "prevalence": float(known.mean()) if len(known) else None,
        }

    return {
        "studies": int(len(train)),
        "series": int(len(series)),
        "unique_series": int(series[SERIES_COLUMN].nunique()),
        "studies_with_all_12_labels": int((known_per_study == len(TARGET_COLUMNS)).sum()),
        "studies_with_partial_labels": int(
            ((known_per_study > 0) & (known_per_study < len(TARGET_COLUMNS))).sum()
        ),
        "studies_without_labels": int((known_per_study == 0).sum()),
        "invalid_non_binary_labels": invalid_values,
        "studies_without_series_metadata": missing_series_studies,
        "series_per_study": _describe_integer_series(series_per_study),
        "plane_counts": {str(key): int(value) for key, value in plane_counts.items()},
        "series_attribute_counts": {
            f"{plane}|fluid={fluid}|fat_suppression={fat}": int(count)
            for (plane, fluid, fat), count in slot_counts.items()
        },
        "targets": target_summary,
        "reports": report_summary,
    }

