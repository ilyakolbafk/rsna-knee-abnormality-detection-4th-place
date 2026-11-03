from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .constants import ID_COLUMN, SERIES_COLUMN


HEADER_TAGS: tuple[str, ...] = (
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "Manufacturer",
    "ManufacturerModelName",
    "MagneticFieldStrength",
    "SoftwareVersions",
    "StationName",
    "DeviceSerialNumber",
    "InstitutionName",
    "InstitutionalDepartmentName",
    "ProtocolName",
    "SeriesDescription",
    "SequenceName",
    "ScanningSequence",
    "SequenceVariant",
    "ScanOptions",
    "MRAcquisitionType",
    "RepetitionTime",
    "EchoTime",
    "FlipAngle",
    "SliceThickness",
    "SpacingBetweenSlices",
    "PixelSpacing",
    "Rows",
    "Columns",
)

TEXT_TAG_COLUMNS: dict[str, str] = {
    "Manufacturer": "manufacturer",
    "ManufacturerModelName": "manufacturer_model_name",
    "SoftwareVersions": "software_versions",
    "StationName": "station_name",
    "DeviceSerialNumber": "device_serial_number",
    "InstitutionName": "institution_name",
    "InstitutionalDepartmentName": "institutional_department_name",
    "ProtocolName": "protocol_name",
    "SeriesDescription": "series_description",
    "SequenceName": "sequence_name",
    "ScanningSequence": "scanning_sequence",
    "SequenceVariant": "sequence_variant",
    "ScanOptions": "scan_options",
    "MRAcquisitionType": "mr_acquisition_type",
}

NUMERIC_TAG_COLUMNS: dict[str, str] = {
    "MagneticFieldStrength": "magnetic_field_strength",
    "RepetitionTime": "repetition_time",
    "EchoTime": "echo_time",
    "FlipAngle": "flip_angle",
    "SliceThickness": "slice_thickness",
    "SpacingBetweenSlices": "spacing_between_slices",
    "Rows": "rows",
    "Columns": "columns",
}

PROFILE_COLUMNS: tuple[str, ...] = (
    "manufacturer",
    "manufacturer_model_name",
    "magnetic_field_strength",
    "software_versions",
    "station_name",
    "device_serial_number",
    "institution_name",
    "institutional_department_name",
)

ACQUISITION_PROFILE_COLUMNS: tuple[str, ...] = (
    "protocol_name",
    "series_description",
    "sequence_name",
    "scanning_sequence",
    "sequence_variant",
    "scan_options",
    "mr_acquisition_type",
    "repetition_time",
    "echo_time",
    "flip_angle",
    "slice_thickness",
    "spacing_between_slices",
    "pixel_spacing_row",
    "pixel_spacing_col",
    "rows",
    "columns",
)


def _clean_scalar(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, (str, bytes, Iterable)) and pd.isna(value):
        return None
    if isinstance(value, (str, bytes)):
        text = value.decode(errors="replace") if isinstance(value, bytes) else value
        text = " ".join(text.strip().split())
        return text or None
    if isinstance(value, Iterable):
        parts = [_clean_scalar(item) for item in value]
        kept = [part for part in parts if part]
        return (
            json.dumps(kept, ensure_ascii=False, separators=(",", ":"))
            if kept
            else None
        )
    text = " ".join(str(value).strip().split())
    return text or None


def _clean_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _pixel_spacing(value: Any) -> tuple[float | None, float | None]:
    if value is None:
        return None, None
    try:
        values = list(value)
    except TypeError:
        return None, None
    if len(values) < 2:
        return None, None
    return _clean_float(values[0]), _clean_float(values[1])


def _normalise_category(value: Any) -> str | None:
    text = _clean_scalar(value)
    return text.casefold() if text else None


def _mode(values: pd.Series) -> Any:
    kept = values.dropna()
    if kept.empty:
        return np.nan
    counts = kept.value_counts(dropna=True)
    maximum = int(counts.max())
    candidates = list(counts[counts.eq(maximum)].index)
    return sorted(candidates, key=lambda value: str(value).casefold())[0]


def _profile_values(values: pd.Series, *, numeric: bool) -> pd.Series:
    if numeric:
        return pd.to_numeric(values, errors="coerce").round(6)
    return values.map(_normalise_category)


def _canonical_values_json(values: pd.Series) -> str | None:
    kept: dict[str, Any] = {}
    for value in values.dropna().tolist():
        native = value.item() if isinstance(value, np.generic) else value
        key = json.dumps(native, ensure_ascii=False, separators=(",", ":"))
        kept[key] = native
    if not kept:
        return None
    ordered = [kept[key] for key in sorted(kept, key=str.casefold)]
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class SeriesHeaderRequest:
    study_uid: str
    series_uid: str
    series_dir: Path


def read_representative_series_header(request: SeriesHeaderRequest) -> dict[str, Any]:
    """Read one representative series header without accessing PixelData."""

    try:
        import pydicom
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("DICOM header extraction requires pydicom") from error

    base: dict[str, Any] = {
        ID_COLUMN: request.study_uid,
        SERIES_COLUMN: request.series_uid,
        "source_file": None,
        "source_file_count": 0,
        "read_attempt_count": 0,
        "header_status": "missing_series_directory",
        "header_error_type": "missing_series_directory",
        "header_error": None,
        "dicom_study_uid": None,
        "dicom_series_uid": None,
        "sop_instance_uid": None,
        "path_uid_match": None,
        "pixel_data_present": False,
    }
    for column in (*TEXT_TAG_COLUMNS.values(), *NUMERIC_TAG_COLUMNS.values()):
        base[column] = None
    base["pixel_spacing_row"] = None
    base["pixel_spacing_col"] = None

    if not request.series_dir.is_dir():
        return base
    base["header_error_type"] = None
    files = sorted(path for path in request.series_dir.iterdir() if path.is_file())
    base["source_file_count"] = len(files)
    if not files:
        base["header_status"] = "empty_series_directory"
        base["header_error_type"] = "empty_series_directory"
        return base

    errors: list[str] = []
    for attempt, path in enumerate(files, start=1):
        base["read_attempt_count"] = attempt
        try:
            dataset = pydicom.dcmread(
                path,
                stop_before_pixels=True,
                specific_tags=list(HEADER_TAGS),
                force=True,
            )
        except Exception as error:  # pragma: no cover - exercised via corrupted test file
            errors.append(f"{path.name}:{type(error).__name__}")
            continue
        base["source_file"] = path.name
        base["dicom_study_uid"] = _clean_scalar(
            getattr(dataset, "StudyInstanceUID", None)
        )
        base["dicom_series_uid"] = _clean_scalar(
            getattr(dataset, "SeriesInstanceUID", None)
        )
        base["sop_instance_uid"] = _clean_scalar(
            getattr(dataset, "SOPInstanceUID", None)
        )
        base["pixel_data_present"] = bool("PixelData" in dataset)
        if base["dicom_study_uid"] and base["dicom_series_uid"]:
            base["path_uid_match"] = bool(
                base["dicom_study_uid"] == request.study_uid
                and base["dicom_series_uid"] == request.series_uid
            )
        for tag, column in TEXT_TAG_COLUMNS.items():
            base[column] = _clean_scalar(getattr(dataset, tag, None))
        for tag, column in NUMERIC_TAG_COLUMNS.items():
            base[column] = _clean_float(getattr(dataset, tag, None))
        row, column = _pixel_spacing(getattr(dataset, "PixelSpacing", None))
        base["pixel_spacing_row"] = row
        base["pixel_spacing_col"] = column
        if base["pixel_data_present"]:
            base["header_status"] = "pixel_data_present"
            base["header_error_type"] = "pixel_data_present"
            base["header_error"] = "stop_before_pixels contract violated"
        elif (
            not base["dicom_study_uid"]
            or not base["dicom_series_uid"]
            or not base["sop_instance_uid"]
        ):
            base["header_status"] = "missing_uid_identity"
            base["header_error_type"] = "missing_uid_identity"
            base["header_error"] = "StudyInstanceUID, SeriesInstanceUID, or SOPInstanceUID absent"
        elif not base["path_uid_match"]:
            base["header_status"] = "uid_identity_mismatch"
            base["header_error_type"] = "uid_identity_mismatch"
            base["header_error"] = (
                f"manifest={request.study_uid}/{request.series_uid};"
                f"dicom={base['dicom_study_uid']}/{base['dicom_series_uid']}"
            )
        else:
            base["header_status"] = "ok"
            base["header_error_type"] = None
        break

    if base["source_file"] is None:
        base["header_status"] = "all_files_unreadable"
        base["header_error_type"] = "all_files_unreadable"
        base["header_error"] = ";".join(errors[:5]) or "unknown"
    return base


def build_series_header_table(
    series: pd.DataFrame,
    images_root: str | Path,
    *,
    workers: int = 8,
) -> pd.DataFrame:
    required = {ID_COLUMN, SERIES_COLUMN}
    missing = sorted(required.difference(series.columns))
    if missing:
        raise ValueError(f"series metadata missing columns: {missing}")
    metadata = series[[ID_COLUMN, SERIES_COLUMN]].copy()
    metadata[ID_COLUMN] = metadata[ID_COLUMN].astype(str)
    metadata[SERIES_COLUMN] = metadata[SERIES_COLUMN].astype(str)
    if metadata.isna().any().any() or metadata.duplicated(SERIES_COLUMN).any():
        raise ValueError("series metadata must be non-null and unique by SeriesInstanceUID")
    root = Path(images_root)
    requests = [
        SeriesHeaderRequest(
            study_uid=row[ID_COLUMN],
            series_uid=row[SERIES_COLUMN],
            series_dir=root / row[ID_COLUMN] / row[SERIES_COLUMN],
        )
        for row in metadata.to_dict("records")
    ]
    if workers <= 1:
        rows = [read_representative_series_header(request) for request in requests]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(read_representative_series_header, requests))
    output = pd.DataFrame(rows).sort_values(
        [ID_COLUMN, SERIES_COLUMN], kind="mergesort"
    ).reset_index(drop=True)
    if len(output) != len(metadata) or output.duplicated(SERIES_COLUMN).any():
        raise AssertionError("series header output coverage drift")
    return output


def build_study_scanner_profiles(
    headers: pd.DataFrame,
    series: pd.DataFrame,
) -> pd.DataFrame:
    expected = series[[ID_COLUMN, SERIES_COLUMN]].copy()
    expected[ID_COLUMN] = expected[ID_COLUMN].astype(str)
    expected[SERIES_COLUMN] = expected[SERIES_COLUMN].astype(str)
    if headers.duplicated(SERIES_COLUMN).any():
        raise ValueError("headers contain duplicate SeriesInstanceUID rows")
    merged = expected.merge(
        headers,
        on=[ID_COLUMN, SERIES_COLUMN],
        how="left",
        validate="one_to_one",
    )
    rows: list[dict[str, Any]] = []
    for study_uid, part in merged.groupby(ID_COLUMN, sort=True):
        accepted = part.loc[part["header_status"].eq("ok")]
        row: dict[str, Any] = {
            ID_COLUMN: study_uid,
            "expected_series_count": int(len(part)),
            "readable_series_count": int(part["header_status"].eq("ok").sum()),
        }
        for column in PROFILE_COLUMNS:
            normalised = _profile_values(
                accepted[column], numeric=column == "magnetic_field_strength"
            )
            row[f"scanner_{column}"] = _mode(normalised)
            row[f"scanner_{column}_distinct"] = int(normalised.dropna().nunique())
            row[f"scanner_{column}_conflict"] = bool(normalised.dropna().nunique() > 1)
        for column in ACQUISITION_PROFILE_COLUMNS:
            normalised = _profile_values(
                accepted[column], numeric=column in {
                    *NUMERIC_TAG_COLUMNS.values(),
                    "pixel_spacing_row",
                    "pixel_spacing_col",
                }
            )
            row[f"acquisition_{column}_values_json"] = _canonical_values_json(normalised)
            row[f"acquisition_{column}_distinct"] = int(normalised.dropna().nunique())
        rows.append(row)
    output = pd.DataFrame(rows).sort_values(ID_COLUMN, kind="mergesort").reset_index(drop=True)
    output["scanner_header_coverage"] = (
        output["readable_series_count"] / output["expected_series_count"].clip(lower=1)
    )
    manufacturer = output["scanner_manufacturer"].astype("string")
    model = output["scanner_manufacturer_model_name"].astype("string")
    field = output["scanner_magnetic_field_strength"].astype("string")
    output["scanner_proxy"] = (
        manufacturer.fillna("") + "|" + model.fillna("") + "|" + field.fillna("")
    ).mask(manufacturer.isna() & model.isna() & field.isna(), pd.NA)
    return output


def header_diagnostics(
    headers: pd.DataFrame,
    profiles: pd.DataFrame,
    *,
    contract_semantic_hash: str,
) -> dict[str, Any]:
    non_null = {
        column: int(headers[column].notna().sum())
        for column in (*TEXT_TAG_COLUMNS.values(), *NUMERIC_TAG_COLUMNS.values())
    }
    return {
        "contract_semantic_hash": contract_semantic_hash,
        "series_rows": int(len(headers)),
        "unique_series": int(headers[SERIES_COLUMN].nunique()),
        "studies": int(profiles[ID_COLUMN].nunique()),
        "header_status_counts": {
            str(key): int(value)
            for key, value in headers["header_status"].value_counts(dropna=False).items()
        },
        "pixel_data_present_rows": int(headers["pixel_data_present"].sum()),
        "uid_identity_match_rows": int(headers["path_uid_match"].fillna(False).sum()),
        "sop_instance_uid_present_rows": int(headers["sop_instance_uid"].notna().sum()),
        "duplicate_series_rows": int(headers.duplicated(SERIES_COLUMN).sum()),
        "non_null_series_tags": non_null,
        "complete_study_header_coverage": int(profiles["scanner_header_coverage"].eq(1.0).sum()),
        "study_conflict_counts": {
            column: int(profiles[column].sum())
            for column in profiles.columns
            if column.endswith("_conflict")
        },
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_sorted_values(values: Sequence[Any]) -> str:
    canonical = "\n".join(sorted(str(value) for value in values)) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
