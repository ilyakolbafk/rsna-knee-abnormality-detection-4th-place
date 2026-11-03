"""Fail-closed validation for downloaded header-only extraction artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .constants import ID_COLUMN, SERIES_COLUMN
from .dicom_headers import sha256_file, sha256_sorted_values


HEADER_FILENAME = "dicom_series_headers_v1.csv.gz"
PROFILE_FILENAME = "study_scanner_profiles_v1.csv.gz"
DIAGNOSTICS_FILENAME = "header_extraction_diagnostics_v1.json"


def _as_bool(series: pd.Series, *, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    values = series.astype(str).str.strip().str.casefold().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if values.isna().any():
        raise ValueError(f"{name} contains non-boolean values")
    return values.astype(bool)


def validate_header_readback(
    *,
    artifact_dir: str | Path,
    train_csv: str | Path,
    series_csv: str | Path,
    contract_semantic_hash: str,
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate full extracted-header identity, coverage, and recorded hashes.

    The function reads only the downloaded tables and official metadata. It never
    opens a DICOM file and it raises on every failed gate.
    """

    root = Path(artifact_dir)
    train_path = Path(train_csv)
    series_path = Path(series_csv)
    header_path = root / HEADER_FILENAME
    profile_path = root / PROFILE_FILENAME
    diagnostics_path = root / DIAGNOSTICS_FILENAME
    required_paths = [
        train_path,
        series_path,
        header_path,
        profile_path,
        diagnostics_path,
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Header readback files are missing: {missing}")

    train = pd.read_csv(train_path, usecols=[ID_COLUMN], dtype={ID_COLUMN: str})
    series = pd.read_csv(
        series_path,
        usecols=[ID_COLUMN, SERIES_COLUMN],
        dtype={ID_COLUMN: str, SERIES_COLUMN: str},
    )
    headers = pd.read_csv(
        header_path,
        dtype={
            ID_COLUMN: str,
            SERIES_COLUMN: str,
            "dicom_study_uid": str,
            "dicom_series_uid": str,
            "sop_instance_uid": str,
        },
    )
    profiles = pd.read_csv(profile_path, dtype={ID_COLUMN: str})
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))

    required_header_columns = {
        ID_COLUMN,
        SERIES_COLUMN,
        "header_status",
        "dicom_study_uid",
        "dicom_series_uid",
        "sop_instance_uid",
        "path_uid_match",
        "pixel_data_present",
    }
    required_profile_columns = {
        ID_COLUMN,
        "expected_series_count",
        "readable_series_count",
        "scanner_header_coverage",
    }
    missing_header_columns = sorted(required_header_columns.difference(headers.columns))
    missing_profile_columns = sorted(required_profile_columns.difference(profiles.columns))
    if missing_header_columns or missing_profile_columns:
        raise ValueError(
            "Header readback schema drift: "
            f"headers={missing_header_columns}, profiles={missing_profile_columns}"
        )

    expected_studies = int(source_identity["expected_studies"])
    expected_series = int(source_identity["expected_series"])
    expected_study_hash = str(source_identity["study_uid_set_sha256"])
    expected_series_hash = str(source_identity["series_uid_set_sha256"])
    actual_hashes = {
        HEADER_FILENAME: sha256_file(header_path),
        PROFILE_FILENAME: sha256_file(profile_path),
        DIAGNOSTICS_FILENAME: sha256_file(diagnostics_path),
    }
    header_studies = set(headers[ID_COLUMN].astype(str))
    profile_studies = set(profiles[ID_COLUMN].astype(str))
    train_studies = set(train[ID_COLUMN].astype(str))
    header_series = set(headers[SERIES_COLUMN].astype(str))
    metadata_series = set(series[SERIES_COLUMN].astype(str))
    path_match = _as_bool(headers["path_uid_match"], name="path_uid_match")
    pixel_present = _as_bool(
        headers["pixel_data_present"], name="pixel_data_present"
    )
    recorded_output_hashes = diagnostics.get("output_sha256", {})

    gates = {
        "train_csv_hash": sha256_file(train_path)
        == str(source_identity["train_csv_sha256"]),
        "series_csv_hash": sha256_file(series_path)
        == str(source_identity["train_series_csv_sha256"]),
        "metadata_study_count": len(train_studies) == expected_studies,
        "metadata_series_count": len(series) == expected_series,
        "metadata_study_uid_hash": sha256_sorted_values(train_studies)
        == expected_study_hash,
        "metadata_series_uid_hash": sha256_sorted_values(metadata_series)
        == expected_series_hash,
        "header_rows": len(headers) == expected_series,
        "header_series_unique": not headers[SERIES_COLUMN].duplicated().any(),
        "header_series_set_exact": header_series == metadata_series,
        "header_study_set_exact": header_studies == train_studies,
        "header_status_ok": headers["header_status"].eq("ok").all(),
        "path_uid_match": path_match.all(),
        "dicom_study_uid_match": (
            headers["dicom_study_uid"].astype(str) == headers[ID_COLUMN].astype(str)
        ).all(),
        "dicom_series_uid_match": (
            headers["dicom_series_uid"].astype(str)
            == headers[SERIES_COLUMN].astype(str)
        ).all(),
        "sop_uid_present": headers["sop_instance_uid"].notna().all(),
        "pixel_data_absent": not pixel_present.any(),
        "profile_rows": len(profiles) == expected_studies,
        "profile_study_unique": not profiles[ID_COLUMN].duplicated().any(),
        "profile_study_set_exact": profile_studies == train_studies,
        "profile_complete_coverage": profiles["scanner_header_coverage"].eq(1.0).all(),
        "profile_series_totals": int(profiles["expected_series_count"].sum())
        == expected_series
        and int(profiles["readable_series_count"].sum()) == expected_series,
        "diagnostics_success": diagnostics.get("success") is True,
        "diagnostics_contract_hash": diagnostics.get("contract_semantic_hash")
        == contract_semantic_hash,
        "diagnostics_rows": diagnostics.get("series_rows") == expected_series
        and diagnostics.get("studies") == expected_studies,
        "diagnostics_uid_set_match": diagnostics.get("exact_series_uid_set_match")
        is True,
        "diagnostics_output_hashes": recorded_output_hashes.get(HEADER_FILENAME)
        == actual_hashes[HEADER_FILENAME]
        and recorded_output_hashes.get(PROFILE_FILENAME)
        == actual_hashes[PROFILE_FILENAME],
    }
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        raise RuntimeError(f"Header readback acceptance failed: {failed}")

    return {
        "format": "rsna-knee-header-readback-audit-v1",
        "status": "PASS",
        "contract_semantic_hash": contract_semantic_hash,
        "studies": expected_studies,
        "series": expected_series,
        "source_identity": dict(source_identity),
        "artifact_sha256": actual_hashes,
        "gates": gates,
    }
