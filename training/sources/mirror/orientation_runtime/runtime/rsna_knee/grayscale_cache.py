"""Deterministic, resumable all-study grayscale DICOM cache.

The legacy pilot cache stores model-ready 2.5D RGB triplets per selected series.
This cache deliberately stores each sampled grayscale plane once.  It is therefore
independent of the downstream 2.5D channel assembly and can feed both 224 px and
336 px encoders from one 336 px source artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cache import resolve_series_dir
from .constants import ID_COLUMN, SERIES_COLUMN, SERIES_REQUIRED_COLUMNS
from .dicom import (
    center_crop_mm,
    dicom_pixels,
    dicom_sort_key,
    discover_dicom_files,
    physical_slice_position,
    pixel_spacing,
    resize_image,
    robust_scale,
)


CACHE_FORMAT = "rsna-knee-grayscale-u8-v1"
REQUIRED_SHARD_FILES = {
    "images.npy",
    "valid.npy",
    "positions.npy",
    "source_indices.npy",
    "study_uids.npy",
    "series_uids.npy",
    "provenance.csv",
}
GEOMETRY_TAGS = [
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "SliceLocation",
    "InstanceNumber",
]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_partition(study_uid: str, num_parts: int) -> int:
    """Map a UID to a stable part without depending on row order."""

    if num_parts <= 0:
        raise ValueError("num_parts must be positive")
    value = int.from_bytes(
        hashlib.sha256(str(study_uid).encode("utf-8")).digest()[:8], "big"
    )
    return value % num_parts


def partition_study_ids(
    study_ids: Iterable[str], *, part_index: int, num_parts: int
) -> list[str]:
    if not 0 <= part_index < num_parts:
        raise ValueError("part_index must lie in [0, num_parts)")
    unique = sorted({str(uid) for uid in study_ids})
    return [uid for uid in unique if stable_partition(uid, num_parts) == part_index]


def _normalise_slots(slots: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    names: set[str] = set()
    for slot in slots:
        name = str(slot.get("name", ""))
        if not name or name in names:
            raise ValueError("Every cache slot needs a unique non-empty name")
        names.add(name)
        count = int(slot.get("slices", slot.get("slices_per_series", 0)))
        if count <= 0:
            raise ValueError(f"Cache slot {name!r} needs a positive slices value")
        window = slot.get("window")
        if window is None:
            plane = str(slot["plane"]).casefold()
            window = (0.20, 0.80) if plane == "coronal" else (0.12, 0.88)
        if len(window) != 2:
            raise ValueError(f"Cache slot {name!r} window must have two values")
        low, high = float(window[0]), float(window[1])
        if not 0.0 <= low < high <= 1.0:
            raise ValueError(f"Cache slot {name!r} has invalid window {window}")
        normalised = {
            "name": name,
            "plane": str(slot["plane"]),
            "fluid_sensitive": int(slot["fluid_sensitive"]),
            "slices": count,
            "window": [low, high],
        }
        if "fat_suppression" in slot:
            normalised["fat_suppression"] = int(slot["fat_suppression"])
        result.append(normalised)
    return result


def cache_spec(data_config: dict[str, Any]) -> dict[str, Any]:
    slots = _normalise_slots(data_config["slots"])
    size = int(data_config.get("source_image_size", data_config["image_size"]))
    if size <= 0:
        raise ValueError("source_image_size must be positive")
    percentile_low = float(data_config.get("percentile_low", 0.5))
    percentile_high = float(data_config.get("percentile_high", 99.5))
    if not 0 <= percentile_low < percentile_high <= 100:
        raise ValueError("Expected 0 <= percentile_low < percentile_high <= 100")
    spec = {
        "format": CACHE_FORMAT,
        "source_image_size": size,
        "crop_mm": float(data_config.get("crop_mm", 130.0)),
        "percentile_low": percentile_low,
        "percentile_high": percentile_high,
        "dtype": "uint8",
        "channel_policy": str(
            data_config.get("channel_policy", "adjacent_sampled_planes")
        ),
        "slots": slots,
        "token_count": int(sum(slot["slices"] for slot in slots)),
    }
    if "centering_mode" in data_config:
        spec["centering_mode"] = str(data_config["centering_mode"])
    if "crop_profile" in data_config:
        spec["crop_profile"] = str(data_config["crop_profile"])
    return spec


def cache_spec_hash(spec: dict[str, Any]) -> str:
    encoded = json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def slot_offsets(slots: Sequence[dict[str, Any]]) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    start = 0
    for slot in slots:
        end = start + int(slot["slices"])
        offsets.append((start, end))
        start = end
    return offsets


def _candidate_file_count(path: Path) -> int:
    try:
        return sum(1 for item in path.iterdir() if item.is_file())
    except OSError:
        return 0


def select_series_for_cache(
    study_series: pd.DataFrame,
    images_root: str | Path,
    slots: Sequence[dict[str, Any]],
) -> list[dict[str, Any] | None]:
    """Choose the longest matching series, with UID as a deterministic tie-break."""

    missing = sorted(set(SERIES_REQUIRED_COLUMNS).difference(study_series.columns))
    if missing:
        raise ValueError(f"Series metadata is missing columns: {missing}")
    selected: list[dict[str, Any] | None] = []
    for slot in slots:
        candidates = study_series[
            study_series["Anatomical_Plane"].astype(str).str.casefold()
            == str(slot["plane"]).casefold()
        ].copy()
        candidates = candidates[
            pd.to_numeric(candidates["Fluid_Sensitive"], errors="coerce")
            == int(slot["fluid_sensitive"])
        ]
        if "fat_suppression" in slot:
            candidates = candidates[
                pd.to_numeric(candidates["Fat_Suppression"], errors="coerce")
                == int(slot["fat_suppression"])
            ]
        if candidates.empty:
            selected.append(None)
            continue
        ranked: list[dict[str, Any]] = []
        for _, row in candidates.iterrows():
            series_uid = str(row[SERIES_COLUMN])
            directory = resolve_series_dir(images_root, str(row[ID_COLUMN]), series_uid)
            ranked.append(
                {
                    "series_uid": series_uid,
                    "directory": directory,
                    "file_count": _candidate_file_count(directory),
                }
            )
        ranked.sort(key=lambda item: (-int(item["file_count"]), str(item["series_uid"])))
        selected.append(ranked[0])
    return selected


def sample_window_indices(
    n_slices: int, count: int, low_fraction: float, high_fraction: float
) -> np.ndarray:
    if n_slices <= 0 or count <= 0:
        raise ValueError("n_slices and count must be positive")
    low = int(round((n_slices - 1) * low_fraction))
    high = int(round((n_slices - 1) * high_fraction))
    high = max(low, high)
    if count == 1:
        return np.asarray([(low + high) // 2], dtype=np.int32)
    return np.rint(np.linspace(low, high, count)).astype(np.int32)


def _ordered_dicom_entries(series_dir: str | Path) -> tuple[list[dict[str, Any]], int]:
    try:
        import pydicom
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("DICOM cache building requires pydicom") from error

    paths = discover_dicom_files(series_dir)
    entries: list[dict[str, Any]] = []
    header_failures = 0
    for path in paths:
        try:
            header = pydicom.dcmread(
                path,
                stop_before_pixels=True,
                specific_tags=GEOMETRY_TAGS,
                force=True,
            )
            sort_key = dicom_sort_key(header, path.name)
            physical = physical_slice_position(header)
        except Exception:
            header_failures += 1
            sort_key = (3, 0.0, 0.0, path.name)
            physical = None
        entries.append(
            {
                "path": path,
                "sort_key": sort_key,
                "physical_position": physical,
            }
        )
    entries.sort(key=lambda item: item["sort_key"])
    return entries, header_failures


def preprocess_grayscale_slot(
    series_dir: str | Path,
    *,
    count: int,
    image_size: int,
    crop_mm: float,
    percentile_low: float,
    percentile_high: float,
    window: Sequence[float],
    centering_mode: str = "geometric",
    crop_profile: str = "local_rectangular_mm",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, int]]:
    """Return sampled `[K,H,W] uint8` planes plus exact source provenance."""

    try:
        import pydicom
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("DICOM cache building requires pydicom") from error

    entries, header_failures = _ordered_dicom_entries(series_dir)
    if centering_mode not in {
        "geometric",
        "joint_line_v1",
        "joint_line_full_v1",
    }:
        raise ValueError(f"Unknown centering_mode: {centering_mode}")
    centering_decision = None
    if centering_mode != "geometric":
        from .study_centering import locate_series_center

        locator_datasets = [
            pydicom.dcmread(entry["path"], force=True) for entry in entries
        ]
        centering_decision = locate_series_center(
            locator_datasets,
            coverage_policy=(
                "full_non_axial"
                if centering_mode == "joint_line_full_v1"
                else "safe"
            ),
        )
    indices = sample_window_indices(len(entries), count, float(window[0]), float(window[1]))
    images = np.zeros((count, image_size, image_size), dtype=np.float32)
    valid = np.zeros(count, dtype=bool)
    positions = np.full(count, np.nan, dtype=np.float32)
    source_indices = indices.astype(np.int32, copy=True)
    source_files = [str(entries[int(index)]["path"].name) for index in indices]
    pixel_failures = 0

    decoded: dict[int, np.ndarray] = {}
    for source_index in sorted(set(indices.tolist())):
        entry = entries[int(source_index)]
        try:
            dataset = pydicom.dcmread(entry["path"], force=True)
            image = dicom_pixels(dataset)
            if centering_decision is None and crop_profile == "local_rectangular_mm":
                image = center_crop_mm(image, pixel_spacing(dataset), crop_mm)
            else:
                from .study_centering import SeriesCenteringDecision, crop_image_for_decision

                decision = centering_decision or SeriesCenteringDecision(
                    "global_fallback",
                    "geometric",
                    None,
                    0.0,
                    None,
                    None,
                    (),
                    (),
                    (),
                    None,
                )
                image = crop_image_for_decision(
                    image,
                    dataset,
                    decision,
                    crop_mm=crop_mm,
                    profile=crop_profile,
                )
            decoded[int(source_index)] = resize_image(image, image_size)
        except Exception:
            pixel_failures += 1

    for output_index, source_index in enumerate(indices):
        source_index = int(source_index)
        if source_index not in decoded:
            continue
        images[output_index] = decoded[source_index]
        valid[output_index] = True
        physical = entries[source_index]["physical_position"]
        if physical is not None and np.isfinite(physical):
            positions[output_index] = float(physical)

    if valid.any():
        images[valid] = robust_scale(
            images[valid], percentile_low=percentile_low, percentile_high=percentile_high
        )
    images_u8 = np.rint(images * 255.0).clip(0, 255).astype(np.uint8)
    images_u8[~valid] = 0
    stats = {
        "source_slices": int(len(entries)),
        "sampled_slices": int(count),
        "valid_slices": int(valid.sum()),
        "header_failures": int(header_failures),
        "pixel_failures": int(pixel_failures),
    }
    return images_u8, valid, positions, source_indices, source_files, stats


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporary, path)


def _sha256_json_records(records: Iterable[Sequence[Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        encoded = json.dumps(
            [None if pd.isna(value) else str(value) for value in record],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest.update(encoded.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _build_input_contract(
    *,
    manifest: pd.DataFrame,
    series: pd.DataFrame,
    ids: Sequence[str],
    groups: dict[str, pd.DataFrame],
    empty_series: pd.DataFrame,
    images_root: str | Path,
    slots: Sequence[dict[str, Any]],
    spec_hash: str,
    part_index: int,
    num_parts: int,
    shard_studies: int,
    limit: int | None,
    source_version: str,
    strict_dicom: bool,
) -> dict[str, Any]:
    missing = sorted(set(SERIES_REQUIRED_COLUMNS).difference(series.columns))
    if missing:
        raise ValueError(f"Series metadata is missing columns: {missing}")
    all_manifest_ids = sorted(manifest[ID_COLUMN].astype(str).tolist())
    relevant_series = series.loc[
        series[ID_COLUMN].astype(str).isin(set(ids)), SERIES_REQUIRED_COLUMNS
    ].copy()
    relevant_series[ID_COLUMN] = relevant_series[ID_COLUMN].astype(str)
    relevant_series[SERIES_COLUMN] = relevant_series[SERIES_COLUMN].astype(str)
    relevant_series = relevant_series.sort_values(SERIES_REQUIRED_COLUMNS).reset_index(
        drop=True
    )
    inventory: list[list[Any]] = []
    for study_uid in ids:
        selections = select_series_for_cache(
            groups.get(study_uid, empty_series), images_root, slots
        )
        for slot, selection in zip(slots, selections, strict=True):
            inventory.append(
                [
                    study_uid,
                    slot["name"],
                    None if selection is None else selection["series_uid"],
                    0 if selection is None else int(selection["file_count"]),
                ]
            )
    return {
        "format": CACHE_FORMAT,
        "cache_spec_sha256": spec_hash,
        "source_version": str(source_version),
        "strict_dicom": bool(strict_dicom),
        "part_index": int(part_index),
        "num_parts": int(num_parts),
        "shard_studies": int(shard_studies),
        "limit": None if limit is None else int(limit),
        "all_manifest_studies": int(len(all_manifest_ids)),
        "assigned_studies": int(len(ids)),
        "all_manifest_uids_sha256": _sha256_json_records(
            ([uid] for uid in all_manifest_ids)
        ),
        "assigned_uids_sha256": _sha256_json_records(([uid] for uid in ids)),
        "series_metadata_sha256": _sha256_json_records(
            relevant_series.itertuples(index=False, name=None)
        ),
        "selected_series_inventory_sha256": _sha256_json_records(inventory),
    }


def _verify_completed_shard(
    shard_dir: Path, expected_spec_hash: str, *, full_hash: bool
) -> dict[str, Any] | None:
    marker = shard_dir / "COMPLETED.json"
    if not marker.is_file():
        return None
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("cache_spec_sha256") != expected_spec_hash:
        return None
    if payload.get("format") != CACHE_FORMAT:
        return None
    if set(payload.get("files", {})) != REQUIRED_SHARD_FILES:
        return None
    for name, expected in payload.get("files", {}).items():
        path = shard_dir / name
        if not path.is_file() or path.stat().st_size != int(expected["bytes"]):
            return None
        if full_hash and sha256_file(path) != expected["sha256"]:
            return None
    return payload


def _validate_shard_layout(
    shard_dir: Path,
    marker: dict[str, Any],
    spec: dict[str, Any],
    expected_index: pd.DataFrame,
) -> None:
    """Validate UID-to-row identity and array schemas without loading pixels."""

    expected_rows = int(marker["studies"])
    if len(expected_index) != expected_rows:
        raise ValueError(f"Shard index row count mismatch: {shard_dir}")
    rows = pd.to_numeric(expected_index["row"], errors="raise").astype(int)
    if sorted(rows.tolist()) != list(range(expected_rows)):
        raise ValueError(f"Shard index rows are not a permutation of 0..N-1: {shard_dir}")
    expected_by_row = expected_index.assign(_row=rows).sort_values("_row")
    expected_uids = expected_by_row[ID_COLUMN].astype(str).tolist()

    study_uids = np.load(shard_dir / "study_uids.npy", mmap_mode="r")
    series_uids = np.load(shard_dir / "series_uids.npy", mmap_mode="r")
    images = np.load(shard_dir / "images.npy", mmap_mode="r")
    valid = np.load(shard_dir / "valid.npy", mmap_mode="r")
    positions = np.load(shard_dir / "positions.npy", mmap_mode="r")
    source_indices = np.load(shard_dir / "source_indices.npy", mmap_mode="r")
    token_count = int(spec["token_count"])
    size = int(spec["source_image_size"])
    slots = len(spec["slots"])
    expected_shapes = {
        "images": (expected_rows, token_count, size, size),
        "valid": (expected_rows, token_count),
        "positions": (expected_rows, token_count),
        "source_indices": (expected_rows, token_count),
        "study_uids": (expected_rows,),
        "series_uids": (expected_rows, slots),
    }
    arrays = {
        "images": images,
        "valid": valid,
        "positions": positions,
        "source_indices": source_indices,
        "study_uids": study_uids,
        "series_uids": series_uids,
    }
    for name, expected_shape in expected_shapes.items():
        if arrays[name].shape != expected_shape:
            raise ValueError(
                f"Unexpected {name} shape in {shard_dir}: "
                f"{arrays[name].shape} != {expected_shape}"
            )
    expected_dtypes = {
        "images": np.dtype(np.uint8),
        "valid": np.dtype(np.bool_),
        "positions": np.dtype(np.float32),
        "source_indices": np.dtype(np.int32),
    }
    for name, expected_dtype in expected_dtypes.items():
        if arrays[name].dtype != expected_dtype:
            raise ValueError(
                f"Unexpected {name} dtype in {shard_dir}: "
                f"{arrays[name].dtype} != {expected_dtype}"
            )
    if study_uids.astype(str).tolist() != expected_uids:
        raise ValueError(f"Shard UID-to-row mapping mismatch: {shard_dir}")
    provenance = pd.read_csv(
        shard_dir / "provenance.csv",
        dtype={ID_COLUMN: str},
        usecols=[ID_COLUMN, "row", "token_index"],
    )
    if len(provenance) != expected_rows * token_count:
        raise ValueError(f"Provenance row count mismatch: {shard_dir}")
    for row, uid in enumerate(expected_uids):
        part = provenance.loc[pd.to_numeric(provenance["row"]) == row]
        if len(part) != token_count or set(part[ID_COLUMN].astype(str)) != {uid}:
            raise ValueError(f"Provenance UID-to-row mismatch: {shard_dir}, row={row}")
        if sorted(pd.to_numeric(part["token_index"]).astype(int).tolist()) != list(
            range(token_count)
        ):
            raise ValueError(f"Provenance token coverage mismatch: {shard_dir}, row={row}")


def _write_shard(
    *,
    shard_dir: Path,
    study_ids: Sequence[str],
    series_groups: dict[str, pd.DataFrame],
    empty_series: pd.DataFrame,
    images_root: str | Path,
    spec: dict[str, Any],
    spec_hash: str,
    part_index: int,
    shard_index: int,
    progress: Callable[[dict[str, Any]], None] | None,
    completed_studies_before: int,
    total_part_studies: int,
    strict_dicom: bool,
) -> dict[str, Any]:
    temporary = shard_dir.parent / f".{shard_dir.name}.{os.getpid()}.tmp"
    if temporary.exists():
        raise RuntimeError(f"Stale temporary shard requires inspection: {temporary}")
    temporary.mkdir(parents=True)

    n_studies = len(study_ids)
    image_size = int(spec["source_image_size"])
    token_count = int(spec["token_count"])
    slots = spec["slots"]
    offsets = slot_offsets(slots)
    images = np.lib.format.open_memmap(
        temporary / "images.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(n_studies, token_count, image_size, image_size),
    )
    valid = np.lib.format.open_memmap(
        temporary / "valid.npy", mode="w+", dtype=np.bool_, shape=(n_studies, token_count)
    )
    positions = np.lib.format.open_memmap(
        temporary / "positions.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n_studies, token_count),
    )
    source_indices = np.lib.format.open_memmap(
        temporary / "source_indices.npy",
        mode="w+",
        dtype=np.int32,
        shape=(n_studies, token_count),
    )
    images[:] = 0
    valid[:] = False
    positions[:] = np.nan
    source_indices[:] = -1

    maximum_uid_length = max(max(map(len, study_ids), default=1), 1)
    study_uid_array = np.asarray(study_ids, dtype=f"U{maximum_uid_length}")
    series_uids: list[list[str]] = []
    provenance: list[dict[str, Any]] = []
    totals = {
        "selected_series": 0,
        "missing_slots": 0,
        "valid_slices": 0,
        "header_failures": 0,
        "pixel_failures": 0,
    }
    started = time.monotonic()

    for row_index, study_uid in enumerate(study_ids):
        selections = select_series_for_cache(
            series_groups.get(study_uid, empty_series), images_root, slots
        )
        row_series: list[str] = []
        for slot_index, (slot, selection, (start, end)) in enumerate(
            zip(slots, selections, offsets, strict=True)
        ):
            count = end - start
            if selection is None:
                totals["missing_slots"] += 1
                row_series.append("")
                for local_index in range(count):
                    provenance.append(
                        {
                            ID_COLUMN: study_uid,
                            "part": part_index,
                            "shard": shard_index,
                            "row": row_index,
                            "token_index": start + local_index,
                            "slot_index": slot_index,
                            "slot_name": slot["name"],
                            SERIES_COLUMN: "",
                            "source_file": "",
                            "source_index": -1,
                            "physical_position": "",
                            "valid": False,
                            "source_slice_count": 0,
                        }
                    )
                continue

            series_uid = str(selection["series_uid"])
            row_series.append(series_uid)
            totals["selected_series"] += 1
            try:
                slot_images, slot_valid, slot_positions, slot_source_indices, files, stats = (
                    preprocess_grayscale_slot(
                        selection["directory"],
                        count=count,
                        image_size=image_size,
                        crop_mm=float(spec["crop_mm"]),
                        percentile_low=float(spec["percentile_low"]),
                        percentile_high=float(spec["percentile_high"]),
                        window=slot["window"],
                        centering_mode=str(spec.get("centering_mode", "geometric")),
                        crop_profile=str(spec.get("crop_profile", "local_rectangular_mm")),
                    )
                )
            except Exception as error:
                if strict_dicom:
                    failure = {
                        "status": "FAILED",
                        ID_COLUMN: study_uid,
                        SERIES_COLUMN: series_uid,
                        "slot_name": slot["name"],
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                    _atomic_json(failure, temporary / "FAILED.json")
                    raise RuntimeError(
                        f"Selected series preprocessing failed for study={study_uid}, "
                        f"slot={slot['name']}, series={series_uid}: "
                        f"{type(error).__name__}: {error}"
                    ) from error
                slot_images = np.zeros((count, image_size, image_size), dtype=np.uint8)
                slot_valid = np.zeros(count, dtype=bool)
                slot_positions = np.full(count, np.nan, dtype=np.float32)
                slot_source_indices = np.full(count, -1, dtype=np.int32)
                files = [""] * count
                stats = {
                    "source_slices": int(selection["file_count"]),
                    "sampled_slices": count,
                    "valid_slices": 0,
                    "header_failures": 0,
                    "pixel_failures": count,
                }
            if strict_dicom and (
                int(stats["header_failures"]) > 0
                or int(stats["pixel_failures"]) > 0
                or int(stats["valid_slices"]) != count
            ):
                failure = {
                    "status": "FAILED",
                    ID_COLUMN: study_uid,
                    SERIES_COLUMN: series_uid,
                    "slot_name": slot["name"],
                    "stats": stats,
                }
                _atomic_json(failure, temporary / "FAILED.json")
                raise RuntimeError(
                    f"Selected series did not decode cleanly for study={study_uid}, "
                    f"slot={slot['name']}, series={series_uid}: {stats}"
                )
            images[row_index, start:end] = slot_images
            valid[row_index, start:end] = slot_valid
            positions[row_index, start:end] = slot_positions
            source_indices[row_index, start:end] = slot_source_indices
            totals["valid_slices"] += int(stats["valid_slices"])
            totals["header_failures"] += int(stats["header_failures"])
            totals["pixel_failures"] += int(stats["pixel_failures"])
            for local_index in range(count):
                physical = float(slot_positions[local_index])
                provenance.append(
                    {
                        ID_COLUMN: study_uid,
                        "part": part_index,
                        "shard": shard_index,
                        "row": row_index,
                        "token_index": start + local_index,
                        "slot_index": slot_index,
                        "slot_name": slot["name"],
                        SERIES_COLUMN: series_uid,
                        "source_file": files[local_index],
                        "source_index": int(slot_source_indices[local_index]),
                        "physical_position": physical if np.isfinite(physical) else "",
                        "valid": bool(slot_valid[local_index]),
                        "source_slice_count": int(stats["source_slices"]),
                    }
                )
        series_uids.append(row_series)
        if progress is not None:
            progress(
                {
                    "part": part_index,
                    "shard": shard_index,
                    "study_position": completed_studies_before + row_index + 1,
                    "study_total": total_part_studies,
                    "study_uid": study_uid,
                    "elapsed_seconds": time.monotonic() - started,
                    **totals,
                }
            )

    images.flush()
    valid.flush()
    positions.flush()
    source_indices.flush()
    del images, valid, positions, source_indices
    np.save(temporary / "study_uids.npy", study_uid_array, allow_pickle=False)
    maximum_series_length = max(
        (len(uid) for row in series_uids for uid in row), default=1
    )
    np.save(
        temporary / "series_uids.npy",
        np.asarray(series_uids, dtype=f"U{max(1, maximum_series_length)}"),
        allow_pickle=False,
    )
    pd.DataFrame(provenance).to_csv(temporary / "provenance.csv", index=False)

    file_records: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in temporary.iterdir() if item.is_file()):
        file_records[path.name] = {
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    marker = {
        "format": CACHE_FORMAT,
        "cache_spec_sha256": spec_hash,
        "part": int(part_index),
        "shard": int(shard_index),
        "studies": int(n_studies),
        "first_study_uid": study_ids[0] if study_ids else None,
        "last_study_uid": study_ids[-1] if study_ids else None,
        "quality": {name: int(value) for name, value in totals.items()},
        "strict_dicom": bool(strict_dicom),
        "files": file_records,
    }
    _atomic_json(marker, temporary / "COMPLETED.json")
    os.replace(temporary, shard_dir)
    return marker


def build_grayscale_cache_part(
    manifest: pd.DataFrame,
    series: pd.DataFrame,
    data_config: dict[str, Any],
    *,
    output_root: str | Path,
    part_index: int,
    num_parts: int = 4,
    shard_studies: int = 64,
    limit: int | None = None,
    verify_existing_hashes: bool = False,
    progress: Callable[[dict[str, Any]], None] | None = None,
    strict_dicom: bool = True,
) -> dict[str, Any]:
    if ID_COLUMN not in manifest or manifest[ID_COLUMN].duplicated().any():
        raise ValueError(f"Manifest must be unique by {ID_COLUMN}")
    if shard_studies <= 0:
        raise ValueError("shard_studies must be positive")
    spec = cache_spec(data_config)
    spec_hash = cache_spec_hash(spec)
    ids = partition_study_ids(
        manifest[ID_COLUMN].astype(str), part_index=part_index, num_parts=num_parts
    )
    if limit is not None:
        ids = ids[:limit]
    groups = {
        str(uid): part.copy() for uid, part in series.groupby(ID_COLUMN, sort=False)
    }
    empty_series = series.iloc[0:0]
    input_contract = _build_input_contract(
        manifest=manifest,
        series=series,
        ids=ids,
        groups=groups,
        empty_series=empty_series,
        images_root=data_config["images_root"],
        slots=spec["slots"],
        spec_hash=spec_hash,
        part_index=part_index,
        num_parts=num_parts,
        shard_studies=shard_studies,
        limit=limit,
        source_version=str(
            data_config.get("source_version", "rsna-knee-competition-v1")
        ),
        strict_dicom=strict_dicom,
    )
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    spec_path = output_root / "cache_spec.json"
    if spec_path.is_file():
        existing = json.loads(spec_path.read_text(encoding="utf-8"))
        if cache_spec_hash(existing) != spec_hash:
            raise ValueError("Existing output_root uses a different cache specification")
    else:
        _atomic_json(spec, spec_path)
    input_contract_path = output_root / "input_contract.json"
    if input_contract_path.is_file():
        existing_contract = json.loads(input_contract_path.read_text(encoding="utf-8"))
        if existing_contract != input_contract:
            raise ValueError(
                "Existing output_root was built from different input metadata, "
                "partition parameters, or selected-series inventory"
            )
    else:
        if any(output_root.glob("part-*-shard-*")):
            raise ValueError(
                "Existing shards have no input_contract.json and cannot be resumed safely"
            )
        _atomic_json(input_contract, input_contract_path)
    completed_studies = 0
    shard_records: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for shard_index, start in enumerate(range(0, len(ids), shard_studies)):
        shard_ids = ids[start : start + shard_studies]
        shard_name = f"part-{part_index:02d}-shard-{shard_index:04d}"
        shard_dir = output_root / shard_name
        marker = _verify_completed_shard(
            shard_dir, spec_hash, full_hash=verify_existing_hashes
        )
        if marker is None:
            if shard_dir.exists():
                raise RuntimeError(
                    f"Incomplete or incompatible shard requires inspection: {shard_dir}"
                )
            marker = _write_shard(
                shard_dir=shard_dir,
                study_ids=shard_ids,
                series_groups=groups,
                empty_series=empty_series,
                images_root=data_config["images_root"],
                spec=spec,
                spec_hash=spec_hash,
                part_index=part_index,
                shard_index=shard_index,
                progress=progress,
                completed_studies_before=completed_studies,
                total_part_studies=len(ids),
                strict_dicom=strict_dicom,
            )
        stored_uids = np.load(shard_dir / "study_uids.npy", allow_pickle=False).astype(str)
        if stored_uids.tolist() != shard_ids:
            raise ValueError(f"Shard UID order mismatch: {shard_dir}")
        shard_records.append({"directory": shard_name, **marker})
        for row, uid in enumerate(shard_ids):
            index_rows.append(
                {
                    ID_COLUMN: uid,
                    "part": part_index,
                    "shard": shard_index,
                    "shard_directory": shard_name,
                    "row": row,
                }
            )
        completed_studies += len(shard_ids)

    index_frame = pd.DataFrame(index_rows)
    index_path = output_root / "shard_index.csv"
    temporary_index = index_path.with_name(index_path.name + f".{os.getpid()}.tmp")
    index_frame.to_csv(temporary_index, index=False)
    os.replace(temporary_index, index_path)
    total_bytes = sum(
        int(record["bytes"])
        for shard in shard_records
        for record in shard["files"].values()
    )
    quality = {
        name: int(
            sum(
                int(shard.get("quality", {}).get(name, 0))
                for shard in shard_records
            )
        )
        for name in (
            "selected_series",
            "missing_slots",
            "valid_slices",
            "header_failures",
            "pixel_failures",
        )
    }
    summary = {
        "format": CACHE_FORMAT,
        "cache_spec_sha256": spec_hash,
        "input_contract_sha256": sha256_file(input_contract_path),
        "part_index": int(part_index),
        "num_parts": int(num_parts),
        "studies": int(len(ids)),
        "shards": int(len(shard_records)),
        "token_count": int(spec["token_count"]),
        "source_image_size": int(spec["source_image_size"]),
        "uncompressed_shard_bytes": int(total_bytes),
        "quality": quality,
        "elapsed_seconds": float(time.monotonic() - started),
        "shard_records": shard_records,
        "shard_index_sha256": sha256_file(index_path),
    }
    _atomic_json(summary, output_root / "part_manifest.json")
    return summary


class GrayscaleCacheIndex:
    """Lazy mmap reader over one or more independently produced cache parts."""

    def __init__(self, roots: Sequence[str | Path]) -> None:
        if not roots:
            raise ValueError("At least one grayscale cache root is required")
        self.roots = [Path(root) for root in roots]
        self.entries: dict[str, tuple[Path, int]] = {}
        self._arrays: dict[Path, dict[str, np.ndarray]] = {}
        self.spec: dict[str, Any] | None = None
        self.spec_hash: str | None = None
        for root in self.roots:
            spec_path = root / "cache_spec.json"
            index_path = root / "shard_index.csv"
            manifest_path = root / "part_manifest.json"
            input_contract_path = root / "input_contract.json"
            if not all(
                path.is_file()
                for path in (
                    spec_path,
                    index_path,
                    manifest_path,
                    input_contract_path,
                )
            ):
                raise FileNotFoundError(f"Incomplete grayscale cache root: {root}")
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            current_hash = cache_spec_hash(spec)
            if self.spec_hash is None:
                self.spec = spec
                self.spec_hash = current_hash
            elif current_hash != self.spec_hash:
                raise ValueError("Grayscale cache roots use incompatible specifications")
            part_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if part_manifest.get("format") != CACHE_FORMAT:
                raise ValueError(f"Unexpected part manifest format: {manifest_path}")
            if part_manifest.get("cache_spec_sha256") != current_hash:
                raise ValueError(f"Part manifest spec hash mismatch: {manifest_path}")
            if part_manifest.get("input_contract_sha256") != sha256_file(
                input_contract_path
            ):
                raise ValueError(f"Input contract hash mismatch: {root}")
            if part_manifest.get("shard_index_sha256") != sha256_file(index_path):
                raise ValueError(f"Shard index hash mismatch: {root}")
            frame = pd.read_csv(index_path, dtype={ID_COLUMN: str})
            required_index_columns = {
                ID_COLUMN,
                "part",
                "shard",
                "shard_directory",
                "row",
            }
            if not required_index_columns.issubset(frame.columns):
                raise ValueError(f"Shard index schema mismatch: {index_path}")
            if frame[ID_COLUMN].duplicated().any():
                raise ValueError(f"Duplicate UIDs inside cache root: {root}")
            if int(part_manifest.get("studies", -1)) != len(frame):
                raise ValueError(f"Part manifest study count mismatch: {root}")
            manifest_shards = {
                str(record["directory"]): record
                for record in part_manifest.get("shard_records", [])
            }
            indexed_shards = set(frame["shard_directory"].astype(str))
            if set(manifest_shards) != indexed_shards:
                raise ValueError(f"Part manifest shard set mismatch: {root}")
            for directory, shard_index in frame.groupby(
                "shard_directory", sort=True
            ):
                shard_dir = root / str(directory)
                marker = _verify_completed_shard(
                    shard_dir, current_hash, full_hash=False
                )
                if marker is None:
                    raise ValueError(f"Uncommitted grayscale cache shard: {shard_dir}")
                expected_record = dict(manifest_shards[str(directory)])
                expected_record.pop("directory", None)
                if marker != expected_record:
                    raise ValueError(f"Part/shard manifest disagreement: {shard_dir}")
                parts = pd.to_numeric(shard_index["part"], errors="raise").astype(int)
                shards = pd.to_numeric(shard_index["shard"], errors="raise").astype(int)
                if set(parts) != {int(marker["part"])} or set(shards) != {
                    int(marker["shard"])
                }:
                    raise ValueError(f"Shard coordinates disagree with index: {shard_dir}")
                _validate_shard_layout(shard_dir, marker, spec, shard_index)
                for row in shard_index.itertuples(index=False):
                    uid = str(getattr(row, ID_COLUMN))
                    if uid in self.entries:
                        raise ValueError(f"Study appears in multiple cache roots: {uid}")
                    self.entries[uid] = (shard_dir, int(row.row))
        assert self.spec is not None
        self.slots = self.spec["slots"]
        self.offsets = slot_offsets(self.slots)
        self.token_count = int(self.spec["token_count"])
        self.source_image_size = int(self.spec["source_image_size"])

    def __contains__(self, study_uid: str) -> bool:
        return str(study_uid) in self.entries

    def _load_shard(self, shard_dir: Path) -> dict[str, np.ndarray]:
        if shard_dir not in self._arrays:
            self._arrays[shard_dir] = {
                "images": np.load(shard_dir / "images.npy", mmap_mode="r"),
                "valid": np.load(shard_dir / "valid.npy", mmap_mode="r"),
                "positions": np.load(shard_dir / "positions.npy", mmap_mode="r"),
                "source_indices": np.load(
                    shard_dir / "source_indices.npy", mmap_mode="r"
                ),
                "series_uids": np.load(shard_dir / "series_uids.npy", mmap_mode="r"),
            }
        return self._arrays[shard_dir]

    def load_study(self, study_uid: str) -> dict[str, np.ndarray]:
        uid = str(study_uid)
        if uid not in self.entries:
            raise KeyError(f"Study is absent from grayscale cache: {uid}")
        shard_dir, row = self.entries[uid]
        arrays = self._load_shard(shard_dir)
        return {name: np.asarray(value[row]) for name, value in arrays.items()}


def verify_grayscale_cache(
    roots: Sequence[str | Path],
    *,
    expected_study_ids: Iterable[str] | None = None,
    full_hash: bool = False,
) -> dict[str, Any]:
    """Fail closed on incomplete shards, checksum drift, duplicates, or coverage gaps."""

    index = GrayscaleCacheIndex(roots)
    checked_shards = 0
    checked_files = 0
    total_bytes = 0
    for root in index.roots:
        frame = pd.read_csv(root / "shard_index.csv", dtype={ID_COLUMN: str})
        for directory in sorted(frame["shard_directory"].astype(str).unique()):
            shard_dir = root / directory
            marker = _verify_completed_shard(
                shard_dir, str(index.spec_hash), full_hash=full_hash
            )
            if marker is None:
                raise ValueError(f"Cache shard verification failed: {shard_dir}")
            checked_shards += 1
            checked_files += len(marker["files"])
            total_bytes += sum(
                int(record["bytes"]) for record in marker["files"].values()
            )
    cached_ids = set(index.entries)
    coverage: dict[str, Any] | None = None
    if expected_study_ids is not None:
        expected = {str(uid) for uid in expected_study_ids}
        missing = sorted(expected - cached_ids)
        unexpected = sorted(cached_ids - expected)
        if missing or unexpected:
            raise ValueError(
                f"Cache coverage mismatch: {len(missing)} missing, "
                f"{len(unexpected)} unexpected"
            )
        coverage = {
            "expected": len(expected),
            "missing": 0,
            "unexpected": 0,
        }
    return {
        "format": CACHE_FORMAT,
        "cache_spec_sha256": index.spec_hash,
        "roots": len(index.roots),
        "studies": len(index.entries),
        "shards": checked_shards,
        "files": checked_files,
        "bytes": total_bytes,
        "full_hash": bool(full_hash),
        "coverage": coverage,
    }
