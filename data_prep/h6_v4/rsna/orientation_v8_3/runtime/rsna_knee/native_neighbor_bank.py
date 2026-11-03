"""Small, provenance-bound native-neighbor banks independent of the trainer.

The original grayscale cache remains the authority for series, centers, validity
and uint8 center pixels. Raw DICOMs supply native neighbors and geometry. The
builder refuses to call a bank comparable unless raw center reproduction is exact.
Candidate joint XYZ is an unvalidated image-locator estimate, never a crop change.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .dicom import (center_crop_mm, dicom_pixels, dicom_sort_key,
                    discover_dicom_files, pixel_spacing, resize_image)
from .grayscale_cache import GrayscaleCacheIndex, sample_window_indices, sha256_file

FORMAT = "rsna-knee-native-neighbor-pilot-v1"
STATUS = {"missing_center": 0, "native_neighbor": 1, "center": 2,
          "series_end_replicated": 3, "read_error_replicated": 4,
          "geometry_unverified_neighbor": 5}
HEADER_TAGS = ["StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID",
               "ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing",
               "Rows", "Columns", "SliceThickness", "FrameOfReferenceUID",
               "SliceLocation", "InstanceNumber"]
GEOMETRY_COLUMNS = ["ipp_x", "ipp_y", "ipp_z", "iop_col_x", "iop_col_y", "iop_col_z",
                    "iop_row_x", "iop_row_y", "iop_row_z", "spacing_row", "spacing_col",
                    "source_rows", "source_cols", "normal_x", "normal_y", "normal_z",
                    "normal_projection", "slice_thickness", "previous_gap_mm", "next_gap_mm",
                    "crop_row0", "crop_col0", "crop_rows", "crop_cols", "output_size"]


def semantic_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _vector(ds: Any, name: str, count: int) -> np.ndarray | None:
    try:
        values = np.asarray(getattr(ds, name), dtype=np.float64)
    except (AttributeError, TypeError, ValueError):
        return None
    return values if values.shape == (count,) and np.isfinite(values).all() else None


def plane_geometry(ds: Any) -> dict[str, Any]:
    """Checked geometry; IOP[:3] advances columns and IOP[3:] advances rows."""
    ipp, iop, spacing = (_vector(ds, key, n) for key, n in
                          [("ImagePositionPatient", 3), ("ImageOrientationPatient", 6), ("PixelSpacing", 2)])
    reasons = []
    if ipp is None or iop is None or spacing is None:
        return {"valid": False, "reasons": ["missing_or_nonfinite_geometry"]}
    if (spacing <= 0).any():
        reasons.append("nonpositive_pixel_spacing")
    if not np.allclose(np.linalg.norm(iop.reshape(2, 3), axis=1), 1, atol=1e-3, rtol=0):
        reasons.append("nonunit_iop")
    if abs(float(iop[:3] @ iop[3:])) > 1e-3:
        reasons.append("nonorthogonal_iop")
    normal = np.cross(iop[:3], iop[3:])
    length = np.linalg.norm(normal)
    if length <= 1e-8:
        return {"valid": False, "reasons": reasons + ["degenerate_normal"]}
    normal /= length
    return {"valid": not reasons, "reasons": reasons, "ipp": ipp, "iop": iop,
            "spacing": spacing, "normal": normal, "projection": float(ipp @ normal)}


def crop_transform(rows: int, cols: int, spacing: Sequence[float] | None,
                   crop_mm: float, output_size: int) -> dict[str, Any]:
    """Legacy rectangular center crop and pixel-center model(x,y)->source(x,y)."""
    if min(rows, cols, output_size) <= 0:
        raise ValueError("Image dimensions must be positive")
    cr, cc = rows, cols
    if spacing is not None and crop_mm > 0:
        if len(spacing) != 2 or not np.isfinite(spacing).all() or min(spacing) <= 0:
            raise ValueError("Pixel spacing must be two positive finite values")
        cr = min(rows, max(1, int(round(crop_mm / spacing[0]))))
        cc = min(cols, max(1, int(round(crop_mm / spacing[1]))))
    r0, c0 = max(0, (rows - cr) // 2), max(0, (cols - cc) // 2)
    sx, sy = cc / output_size, cr / output_size
    affine = np.array([[sx, 0, c0 + (sx - 1) / 2],
                       [0, sy, r0 + (sy - 1) / 2], [0, 0, 1]], dtype=np.float64)
    return {"crop_row0": r0, "crop_col0": c0, "crop_rows": cr, "crop_cols": cc,
            "output_size": output_size, "model_to_source": affine,
            "padding": [0, 0, 0, 0], "pixel_convention": "integer_pixel_centers_xy"}


def model_lps_transform(geometry: dict, crop: dict) -> np.ndarray:
    if not geometry["valid"]:
        return np.full((3, 3), np.nan)
    source_lps = np.column_stack((geometry["iop"][:3] * geometry["spacing"][1],
                                  geometry["iop"][3:] * geometry["spacing"][0], geometry["ipp"]))
    return source_lps @ crop["model_to_source"]


def lps_to_model(point: Sequence[float], model_to_lps: np.ndarray) -> tuple[np.ndarray, float]:
    """Project to plane using a Gram inverse; retain signed off-plane distance."""
    axes, origin = model_to_lps[:, :2], model_to_lps[:, 2]
    delta = np.asarray(point, dtype=float) - origin
    xy = np.linalg.solve(axes.T @ axes, axes.T @ delta)
    normal = np.cross(axes[:, 0], axes[:, 1])
    normal /= np.linalg.norm(normal)
    return xy, float(normal @ delta)


def normalization_bounds(centers: np.ndarray, valid: np.ndarray,
                         percentiles: tuple[float, float]) -> tuple[np.float64, np.float64]:
    """Use every old valid center occurrence, including duplicated indices."""
    if centers.ndim != 3 or valid.shape != centers.shape[:1]:
        raise ValueError("Expected selected centers [T,H,W] and validity [T]")
    values = centers[valid]
    finite = values[np.isfinite(values)]
    if not len(finite):
        return np.float64(0), np.float64(0)
    low, high = np.percentile(finite, percentiles)
    return np.float64(low), np.float64(high)


def quantize_with_bounds(planes: np.ndarray, low: float, high: float) -> np.ndarray:
    # np.float64 scalars deliberately match legacy robust_scale under NumPy 2.
    low, high = np.float64(low), np.float64(high)
    if not np.isfinite(high - low) or high <= low:
        return np.zeros(planes.shape, dtype=np.uint8)
    scaled = np.clip((planes.astype(np.float32) - low) / (high - low), 0, 1).astype(np.float32)
    return np.rint(scaled * 255).clip(0, 255).astype(np.uint8)


def neighbor_indices(center: int, source_count: int) -> tuple[list[int], list[bool]]:
    if not 0 <= center < source_count:
        raise ValueError("Center is outside original source series")
    wanted = [center - 1, center, center + 1]
    native = [0 <= value < source_count for value in wanted]
    return [value if ok else center for value, ok in zip(wanted, native)], native


def assemble_native_triplets(images: np.ndarray, references: np.ndarray,
                             channel_valid: np.ndarray) -> np.ndarray:
    """Gather explicitly; -1 never uses NumPy's implicit last-image indexing.

    End/read-error replicas have a nonnegative center reference and channel_valid
    false. They retain center pixels. Missing tokens have only -1 references.
    """
    if references.ndim != 2 or references.shape[1] != 3 or channel_valid.shape != references.shape:
        raise ValueError("Expected channel references and masks [T,3]")
    if references.dtype.kind not in "iu":
        raise ValueError("Bank references must be integers")
    if ((references < -1) | (references >= len(images))).any():
        raise ValueError("Out-of-range bank reference")
    if (channel_valid & (references < 0)).any():
        raise ValueError("A valid channel must reference an image")
    out = np.zeros((*references.shape, *images.shape[1:]), dtype=images.dtype)
    available = references >= 0
    out[available] = images[references[available]]
    return out


def read_ordered_headers(directory: Path) -> tuple[list[dict], dict]:
    import pydicom
    entries = []
    for path in discover_dicom_files(directory):
        try:
            header = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=HEADER_TAGS, force=True)
            geometry = plane_geometry(header)
            key, error = dicom_sort_key(header, path.name), None
        except Exception as exc:
            header, geometry = None, {"valid": False, "reasons": ["header_read_error"]}
            key, error = (3, 0.0, 0.0, path.name), type(exc).__name__
        entries.append({"path": path, "header": header, "geometry": geometry, "sort_key": key,
                        "header_error": error})
    entries.sort(key=lambda entry: entry["sort_key"])
    reasons = []
    if not all(entry["geometry"]["valid"] for entry in entries):
        reasons.append("invalid_source_geometry")
    positions = np.array([entry["geometry"].get("projection", np.nan) for entry in entries])
    gaps = np.diff(positions)
    if np.isfinite(gaps).all() and len(gaps):
        if (gaps <= 1e-4).any():
            reasons.append("duplicate_or_nonincreasing_positions")
        positive = gaps[gaps > 1e-4]
        if len(positive) and (gaps > 1.5 * np.median(positive) + 1e-4).any():
            reasons.append("large_slice_interval")
    if all(entry["geometry"]["valid"] for entry in entries):
        first = entries[0]["geometry"]
        if any(not np.allclose(entry["geometry"]["iop"], first["iop"], atol=1e-4, rtol=0)
               for entry in entries):
            reasons.append("mixed_orientation")
        if any(not np.allclose(entry["geometry"]["spacing"], first["spacing"], atol=1e-4, rtol=0)
               for entry in entries):
            reasons.append("mixed_pixel_spacing")
    shapes = {(getattr(entry["header"], "Rows", None), getattr(entry["header"], "Columns", None)) for entry in entries}
    if len(shapes) != 1 or any(value is None for value in next(iter(shapes))):
        reasons.append("mixed_or_missing_source_shape")
    frames = {str(getattr(entry["header"], "FrameOfReferenceUID", "")) for entry in entries}
    if len(frames) != 1 or "" in frames:
        reasons.append("mixed_or_missing_frame_of_reference")
    return entries, {"source_count": len(entries), "within_series_valid": not reasons,
                     "reasons": reasons, "geometry_status": "within_series_valid" if not reasons else "invalid_or_missing",
                     "frame_uids": sorted(frames), "median_interval_mm": float(np.median(gaps)) if np.isfinite(gaps).all() and len(gaps) else None}


def build_native_neighbor_pilot(*, cache_roots: Sequence[Path], raw_root: Path,
                                study_uids: Sequence[str], joint_manifest: Path,
                                output: Path, max_bytes: int = 5_000_000_000) -> dict:
    """Create one small packed bank; refuse source drift or old pixel changes."""
    import pydicom
    import PIL
    started = time.monotonic()
    uids = list(map(str, study_uids))
    if not uids or len(set(uids)) != len(uids):
        raise ValueError("Choose a nonempty unique study UID list")
    cache = GrayscaleCacheIndex(cache_roots)
    spec = cache.spec
    if spec.get("centering_mode", "geometric") != "geometric" or spec.get("crop_profile", "local_rectangular_mm") != "local_rectangular_mm":
        raise ValueError("Pilot supports only the frozen geometric rectangular crop")
    size = cache.source_image_size
    upper_bound = len(uids) * cache.token_count * 3 * size * size + 64_000_000
    if max_bytes > 5_000_000_000 or upper_bound > max_bytes:
        raise ValueError("Requested pilot exceeds its <=5GB bound")
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite pilot output: {output}")
    output.mkdir(parents=True)
    manifest = json.loads(joint_manifest.read_text())
    images, bank_rows, token_rows, series_rows, normalization = [], [], [], [], []
    geometries, geometry_valid, affines, lps_affines = [], [], [], []
    refs, token_valid, channel_valid, statuses, slot_ids, token_sources, duplicate_of = [], [], [], [], [], [], []
    study_offsets, token_offsets = [0], [0]
    bank_keys, token_keys, series_keys, components = {}, {}, {}, []
    headers_cache, input_files, source_headers = {}, {}, []
    joint_points, joint_centers, parity, decode_errors, z_checks = [], [], [], [], []
    counters = Counter()

    def bind(path: Path) -> str:
        key = str(path.resolve())
        if key not in input_files:
            input_files[key] = sha256_file(path)
        return input_files[key]

    def headers(sid, series):
        key = (sid, series)
        if key not in headers_cache:
            entries, qc = read_ordered_headers(raw_root / sid / series)
            sop_files = {}
            for idx, entry in enumerate(entries):
                ds, geo = entry["header"], entry["geometry"]
                if ds is not None:
                    if str(getattr(ds, "StudyInstanceUID", "")) != sid or str(getattr(ds, "SeriesInstanceUID", "")) != series:
                        raise ValueError(f"DICOM UID/path identity mismatch: {entry['path']}")
                    sop = str(getattr(ds, "SOPInstanceUID", ""))
                    if sop and sop in sop_files:
                        raise ValueError(f"Duplicate SOPInstanceUID in source series: {sop}")
                    sop_files[sop] = idx
                values = {}
                for tag in HEADER_TAGS:
                    value = getattr(ds, tag, None)
                    if value is not None and not isinstance(value, (str, int, float)):
                        value = list(value)
                    values[tag] = value
                source_headers.append({"StudyInstanceUID": sid, "SeriesInstanceUID": series,
                    "source_index": idx, "source_file": str(entry["path"].resolve()),
                    "geometry_tags": values, "geometry_sha256": semantic_hash(values)})
            headers_cache[key] = (entries, qc)
        return headers_cache[key]

    bind(joint_manifest)
    bind(Path(__file__))
    bind(Path(__file__).with_name("dicom.py"))
    bind(Path(__file__).with_name("grayscale_cache.py"))
    for root in cache_roots:
        for name in ("cache_spec.json", "shard_index.csv", "part_manifest.json", "input_contract.json"):
            bind(root / name)
    for sid in uids:
        old = cache.load_study(sid)
        shard, row_index = cache.entries[sid]
        for name in ("provenance.csv", "COMPLETED.json"):
            bind(shard / name)
        old_pixel_sha = hashlib.sha256(old["images"].tobytes()).hexdigest()
        provenance = pd.read_csv(shard / "provenance.csv", keep_default_na=False)
        provenance = provenance[provenance.StudyInstanceUID == sid].sort_values("token_index")
        if len(provenance) != cache.token_count or provenance.token_index.duplicated().any():
            raise ValueError("Old center provenance is incomplete")
        entries_manifest = manifest.get("studies", {}).get(sid, {}).get("series", {})
        for entry in entries_manifest.values():
            component = entry.get("component_id")
            if component and component not in components:
                components.append(component)
        # Restore candidate XYZ only within each source series/frame. The component
        # representative is selected from one series; differing frames are not averaged.
        component_candidates, source_heights = {}, {}
        for series, saved in sorted(entries_manifest.items()):
            decision = saved["decision"]
            if saved["source"] != "native_coronal_locator" or not saved.get("component_id"):
                continue
            ordered, qc = headers(sid, series)
            accepted = set(decision["accepted_indices"])
            points = []
            for candidate in decision["candidates"]:
                idx = candidate["index"]
                if idx not in accepted or not 0 <= idx < len(ordered):
                    continue
                entry = ordered[idx]
                geo = entry["geometry"]
                if not geo["valid"]:
                    continue
                point = geo["ipp"] + geo["iop"][:3] * geo["spacing"][1] * candidate["native_column"] + geo["iop"][3:] * geo["spacing"][0] * candidate["native_row"]
                points.append(point)
                joint_points.append({"StudyInstanceUID": sid, "SeriesInstanceUID": series,
                    "knee_component_id": saved["component_id"], "source_index": idx,
                    "source_file": entry["path"].name, "SOPInstanceUID": str(getattr(entry["header"], "SOPInstanceUID", "")),
                    "native_row": candidate["native_row"], "native_column": candidate["native_column"],
                    "x": float(point[0]), "y": float(point[1]), "z": float(point[2]),
                    "locator_confidence": candidate["confidence"], "geometry_status": "cross_series_raw_lps",
                    "within_source_geometry_valid": qc["within_series_valid"]})
            if points and qc["within_series_valid"]:
                points = np.array(points)
                center = np.median(points, axis=0)
                source_heights[series] = float(np.median(points[:, 2]))
                record = {"StudyInstanceUID": sid, "knee_component_id": saved["component_id"],
                    "center_lps": center.tolist(), "center_kind": "joint_candidate",
                    "center_confidence": float(decision["confidence"]), "center_source_series": [series],
                    "source_frame_uid": qc["frame_uids"][0], "geometry_status": "cross_series_raw_lps",
                    "candidate_count": len(points), "uncertainty_mad_xyz_mm": np.median(abs(points - center), axis=0).tolist(),
                    "candidate_range_xyz_mm": np.ptp(points, axis=0).tolist(),
                    "limitation": "Coronal locator candidate median; column derives from FOV center. AP/ML anatomical center not independently estimated or validated."}
                component_candidates.setdefault(saved["component_id"], []).append(record)
        for saved in entries_manifest.values():
            expected_z = saved.get("coronal_consensus_z_mm")
            sources = saved.get("coronal_source_series", [])
            if expected_z is not None and sources and all(series in source_heights for series in sources):
                restored_z = float(np.median([source_heights[series] for series in sources]))
                z_checks.append({"StudyInstanceUID": sid, "knee_component_id": saved["component_id"],
                                 "expected_z_mm": expected_z, "restored_z_mm": restored_z,
                                 "absolute_error_mm": abs(expected_z - restored_z)})
        study_components = sorted(set(e.get("component_id") for e in entries_manifest.values() if e.get("component_id")))
        if not study_components:
            joint_centers.append({"StudyInstanceUID": sid, "knee_component_id": None,
                "center_lps": None, "center_kind": "unavailable", "center_confidence": 0,
                "geometry_status": "invalid_or_missing", "fallback": "preserve_original_crop",
                "reason": "no_saved_study_locator" if not entries_manifest else "no_saved_knee_component"})
        for cid in study_components:
            options = component_candidates.get(cid, [])
            if options:
                options.sort(key=lambda rec: (-rec["center_confidence"], rec["center_source_series"][0]))
                joint_centers.append({**options[0], "selection": "highest_saved_locator_confidence_then_series_uid",
                                      "alternative_source_candidates": options[1:], "cross_series_registration_validated": False})
            else:
                joint_centers.append({"StudyInstanceUID": sid, "knee_component_id": cid,
                    "center_lps": None, "center_kind": "unavailable", "center_confidence": 0,
                    "geometry_status": "invalid_or_missing", "fallback": "preserve_original_crop"})

        for slot_index, (start, end) in enumerate(cache.offsets):
            slot = cache.slots[slot_index]
            pp = provenance[(provenance.token_index >= start) & (provenance.token_index < end)]
            expected_valid = np.asarray(old["valid"][start:end], bool)
            series = str(old["series_uids"][slot_index])
            if not expected_valid.any():
                counters["missing_slots"] += 1
                for local in range(end - start):
                    refs.append([-1] * 3); channel_valid.append([False] * 3)
                    statuses.append([STATUS["missing_center"]] * 3); token_valid.append(False)
                    slot_ids.append(slot_index); token_sources.append(-1); duplicate_of.append(-1)
                    token_rows.append({"StudyInstanceUID": sid, "token_index": start + local,
                                       "slot_index": slot_index, "slot_name": slot["name"], "series_index": -1,
                                       "source_index": -1, "valid": False})
                continue
            ordered, qc = headers(sid, series)
            count = len(ordered)
            source_indices = np.asarray(old["source_indices"][start:end], int)
            sampled = sample_window_indices(count, end - start, *slot["window"])
            if not np.array_equal(sampled, source_indices):
                raise ValueError(f"Source center/order drift: {sid} / {series}")
            if pp.source_file.tolist() != [ordered[idx]["path"].name for idx in source_indices]:
                raise ValueError(f"Source filename/order drift: {sid} / {series}")
            series_index = series_keys.setdefault((sid, series), len(series_keys))
            component = entries_manifest.get(series, {}).get("component_id")
            component_index = components.index(component) if component in components else -1
            series_rows.append({"StudyInstanceUID": sid, "SeriesInstanceUID": series,
                "series_index": series_index, "slot_index": slot_index, "slot_name": slot["name"],
                "knee_component_id": component, "knee_component_index": component_index, **qc})
            wanted = sorted({idx for center in source_indices[expected_valid]
                             for idx in neighbor_indices(int(center), count)[0]})
            decoded, errors, crop_records = {}, {}, {}
            for idx in wanted:
                path = ordered[idx]["path"]
                bind(path)
                try:
                    ds = pydicom.dcmread(path, force=True)
                    image = dicom_pixels(ds)
                    crop = crop_transform(*image.shape, pixel_spacing(ds), float(spec["crop_mm"]), size)
                    decoded[idx] = resize_image(center_crop_mm(image, pixel_spacing(ds), float(spec["crop_mm"])), size)
                    crop_records[idx] = crop
                except Exception as exc:
                    errors[idx] = type(exc).__name__
                    decode_errors.append({"StudyInstanceUID": sid, "SeriesInstanceUID": series,
                        "source_index": idx, "source_file": str(path.resolve()), "error_type": type(exc).__name__,
                        "fallback": "replicate_original_center_with_channel_valid_false"})
            current_valid = np.array([idx in decoded for idx in source_indices], bool)
            if not np.array_equal(current_valid, expected_valid):
                raise ValueError(f"Original center decode validity drift: {sid} / {series}")
            centers = np.zeros((end - start, size, size), np.float32)
            for local, idx in enumerate(source_indices):
                if expected_valid[local]:
                    centers[local] = decoded[idx]
            low, high = normalization_bounds(centers, expected_valid,
                                             (float(spec["percentile_low"]), float(spec["percentile_high"])))
            reproduced = quantize_with_bounds(centers, low, high)
            reproduced[~expected_valid] = 0
            differing = int(np.count_nonzero(reproduced != old["images"][start:end]))
            parity.append({"StudyInstanceUID": sid, "slot_index": slot_index, "slot_name": slot["name"],
                           "selected_centers": end - start, "differing_pixels": differing,
                           "old_center_pixel_sha256": hashlib.sha256(old["images"][start:end].tobytes()).hexdigest(),
                           "reproduced_center_pixel_sha256": hashlib.sha256(reproduced.tobytes()).hexdigest()})
            if differing:
                _json(output / "CENTER_PARITY_FAILURE.json", parity[-1])
                raise ValueError(f"Center pixels changed: {sid} / {series}: {differing}")
            norm_id = len(normalization)
            normalization.append({"StudyInstanceUID": sid, "SeriesInstanceUID": series, "slot_index": slot_index,
                "normalization_id": norm_id, "low": float(low), "high": float(high),
                "percentiles": [spec["percentile_low"], spec["percentile_high"]],
                "source_indices_with_multiplicity": source_indices[expected_valid].tolist(),
                "population": "legacy selected center crop/resize float32 pixels; duplicates retained",
                "quantization": "legacy robust_scale float32 output; rint(x*255).clip(0,255).uint8"})
            local_bank = {}
            for idx in wanted:
                if idx not in decoded:
                    continue
                entry, crop = ordered[idx], crop_records[idx]
                ds, geo = entry["header"], entry["geometry"]
                sop = str(getattr(ds, "SOPInstanceUID", ""))
                if not sop:
                    raise ValueError("A readable bank image requires SOPInstanceUID")
                transform_identity = {"crop": crop["model_to_source"].tolist(), "low": float(low),
                                      "high": float(high), "output_size": size}
                variant = semantic_hash(transform_identity)
                key = (sid, series, sop, variant)
                if key in bank_keys:
                    local_bank[idx] = bank_keys[key]
                    continue
                bank_index = len(images)
                bank_keys[key], local_bank[idx] = bank_index, bank_index
                image_u8 = quantize_with_bounds(decoded[idx], low, high)
                images.append(image_u8)
                affine = crop["model_to_source"]
                lps = model_lps_transform(geo, crop)
                affines.append(affine); lps_affines.append(lps)
                good = geo["valid"] and qc["within_series_valid"]
                geometry_valid.append(good)
                pos = geo.get("projection", np.nan)
                previous = ordered[idx - 1]["geometry"].get("projection", np.nan) if idx > 0 else np.nan
                following = ordered[idx + 1]["geometry"].get("projection", np.nan) if idx + 1 < count else np.nan
                thickness = float(getattr(ds, "SliceThickness", np.nan))
                values = [*geo.get("ipp", [np.nan]*3), *geo.get("iop", [np.nan]*6),
                          *geo.get("spacing", [np.nan]*2), float(getattr(ds, "Rows", np.nan)),
                          float(getattr(ds, "Columns", np.nan)), *geo.get("normal", [np.nan]*3), pos,
                          thickness, pos - previous, following - pos,
                          crop["crop_row0"], crop["crop_col0"], crop["crop_rows"], crop["crop_cols"], size]
                geometries.append(values)
                bank_rows.append({"bank_index": bank_index, "StudyInstanceUID": sid, "SeriesInstanceUID": series,
                    "SOPInstanceUID": sop, "source_file": entry["path"].name, "source_index": idx,
                    "source_slice_count": count, "series_index": series_index, "knee_component_index": component_index,
                    "frame_uid": str(getattr(ds, "FrameOfReferenceUID", "")), "normalization_id": norm_id,
                    "transform_sha256": variant, "source_file_sha256": input_files[str(entry["path"].resolve())],
                    "pixel_sha256": hashlib.sha256(image_u8.tobytes()).hexdigest(),
                    "is_old_center": bool(idx in source_indices[expected_valid]),
                    "outside_old_center_window": bool(idx < sampled.min() or idx > sampled.max()),
                    "geometry_status": qc["geometry_status"]})
            for local, center in enumerate(source_indices):
                token_index = len(refs)
                if not expected_valid[local]:
                    channel_refs, channel_ok, channel_status = [-1]*3, [False]*3, [0]*3
                    duplicate = -1
                else:
                    neighbors, native = neighbor_indices(int(center), count)
                    channel_refs, channel_ok, channel_status = [], [], []
                    for channel, (idx, exists) in enumerate(zip(neighbors, native)):
                        if not exists:
                            ref, ok, status = local_bank[int(center)], False, STATUS["series_end_replicated"]
                        elif idx in errors:
                            ref, ok, status = local_bank[int(center)], False, STATUS["read_error_replicated"]
                        else:
                            ref, ok = local_bank[idx], True
                            status = STATUS["center"] if channel == 1 else STATUS["native_neighbor"] if qc["within_series_valid"] else STATUS["geometry_unverified_neighbor"]
                        channel_refs.append(ref); channel_ok.append(ok); channel_status.append(status)
                    token_key = (sid, slot_index, int(center))
                    duplicate = token_keys.get(token_key, -1)
                    token_keys.setdefault(token_key, token_index)
                refs.append(channel_refs); channel_valid.append(channel_ok); statuses.append(channel_status)
                token_valid.append(bool(expected_valid[local])); slot_ids.append(slot_index)
                token_sources.append(int(center)); duplicate_of.append(duplicate)
                token_rows.append({"StudyInstanceUID": sid, "token_index": start + local,
                    "slot_index": slot_index, "slot_name": slot["name"], "series_index": series_index,
                    "source_index": int(center), "valid": bool(expected_valid[local]),
                    "duplicate_of_global_token": duplicate})
            counters["neighbor_decode_errors"] += len(errors)
        study_offsets.append(len(images)); token_offsets.append(len(refs))
        # Verify the gathered central channels, including missing slots, directly.
        gathered = np.zeros_like(old["images"])
        for local, ref in enumerate(refs[token_offsets[-2]:token_offsets[-1]]):
            if ref[1] >= 0:
                gathered[local] = images[ref[1]]
        if not np.array_equal(gathered, old["images"]):
            raise ValueError("Packed center gathering changed original pixels")
        counters["center_pixels_checked"] += gathered.size
        counters["studies_exact_center_parity"] += 1
        if hashlib.sha256(old["images"].tobytes()).hexdigest() != old_pixel_sha:
            raise ValueError("Source cache changed during pilot")

    arrays = {"images": np.stack(images), "study_offsets": np.array(study_offsets, np.int64),
        "token_offsets": np.array(token_offsets, np.int64), "token_channels": np.array(refs, np.int64),
        "token_valid": np.array(token_valid, bool), "channel_valid": np.array(channel_valid, bool),
        "slot_ids": np.array(slot_ids, np.int16), "neighbor_status": np.array(statuses, np.uint8),
        "token_source_index": np.array(token_sources, np.int32), "token_duplicate_of": np.array(duplicate_of, np.int64),
        "geometry": np.array(geometries, np.float64), "geometry_valid": np.array(geometry_valid, bool),
        "model_to_source": np.array(affines, np.float64), "model_to_lps": np.array(lps_affines, np.float64)}
    for name in ("source_index", "source_slice_count", "series_index", "knee_component_index"):
        arrays[name] = np.array([row[name] for row in bank_rows], np.int32)
    candidate_projection_errors, plane_residuals = [], []
    candidate_by_source = {}
    for point in joint_points:
        key = (point["StudyInstanceUID"], point["SeriesInstanceUID"], point["source_index"])
        candidate_by_source.setdefault(key, []).append(point)
    for row in bank_rows:
        index = row["bank_index"]
        if not geometry_valid[index]:
            continue
        key = (row["StudyInstanceUID"], row["SeriesInstanceUID"], row["source_index"])
        for point in candidate_by_source.get(key, []):
            expected_xy = np.linalg.solve(affines[index], [point["native_column"], point["native_row"], 1])[:2]
            actual_xy, residual = lps_to_model([point["x"], point["y"], point["z"]], lps_affines[index])
            candidate_projection_errors.append(float(np.linalg.norm(actual_xy - expected_xy)))
            plane_residuals.append(abs(residual))
    estimated = sum(array.nbytes for array in arrays.values())
    if estimated + 64_000_000 > max_bytes:
        raise ValueError("Materialized arrays exceed disk bound")
    for name, array in arrays.items():
        np.save(output / f"{name}.npy", array, allow_pickle=False)
    pd.DataFrame(bank_rows).to_csv(output / "provenance.csv", index=False)
    pd.DataFrame(token_rows).to_csv(output / "token_provenance.csv", index=False)
    point_columns = ["StudyInstanceUID", "SeriesInstanceUID", "knee_component_id", "source_index",
        "source_file", "SOPInstanceUID", "native_row", "native_column", "x", "y", "z",
        "locator_confidence", "geometry_status", "within_source_geometry_valid"]
    pd.DataFrame(joint_points, columns=point_columns).to_csv(output / "joint_candidates.csv", index=False)
    _json(output / "series.json", series_rows)
    _json(output / "joint_centers.json", joint_centers)
    _json(output / "consensus_z_reconstruction.json", z_checks)
    _json(output / "normalization.json", normalization)
    _json(output / "decode_errors.json", decode_errors)
    _json(output / "geometry_columns.json", GEOMETRY_COLUMNS)
    _json(output / "dictionaries.json", {"study_uids": uids, "knee_components": components,
                                        "neighbor_status": STATUS, "slots": cache.slots})
    with (output / "source_headers.jsonl").open("w") as stream:
        for record in source_headers:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    for path, expected in input_files.items():
        if sha256_file(path) != expected:
            raise ValueError(f"Input changed during pilot: {path}")
    _json(output / "INPUT_SHA256.json", input_files)
    output_spec = {"format": FORMAT, "source_cache_spec_sha256": cache.spec_hash,
        "study_uids": uids, "center_policy": "exact_old_provenance_with_duplicates",
        "bank_policy": "native +/-1 original geometrically sorted indices including outside center window",
        "missing_reference": -1, "channel_valid": "false for absent neighbor; explicit replica may still reference center",
        "crop_and_intensity_policy": "unchanged_rectangular_130mm_selected_center_percentiles",
        "geometry_status": "within_series_QC_separate_from_unvalidated_cross_series_raw_LPS",
        "large_interval_qc": ">1.5*median_positive_step+1e-4mm", "input_size": size,
        "normalization_scalar_dtype": "float64", "resized_plane_dtype": "float32", "pixel_dtype": "uint8",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                    "pydicom": pydicom.__version__, "Pillow": PIL.__version__}}
    output_spec["semantic_sha256"] = semantic_hash(output_spec)
    _json(output / "cache_spec.json", output_spec)
    report = {"status": "PASS", "format": FORMAT, "studies": len(uids), "study_uids": uids,
        "bank_images": len(images), "tokens": len(refs), "valid_tokens": sum(token_valid),
        "center_parity": parity, "center_pixel_differences": 0,
        "halo_planes_outside_center_window": sum(row["outside_old_center_window"] for row in bank_rows),
        "duplicate_center_tokens": sum(value >= 0 for value in duplicate_of),
        "neighbor_status_counts": {name: int((arrays["neighbor_status"] == code).sum()) for name, code in STATUS.items()},
        "geometry_valid_bank_images": sum(geometry_valid), "geometry_invalid_series": [r for r in series_rows if not r["within_series_valid"]],
        "joint_candidate_points": len(joint_points),
        "joint_knee_components": sum(record["knee_component_id"] is not None for record in joint_centers),
        "joint_center_unavailable_studies": sorted({record["StudyInstanceUID"] for record in joint_centers if record["center_kind"] == "unavailable"}),
        "consensus_z_reconstruction_checked_records": len(z_checks),
        "consensus_z_reconstruction_max_abs_error_mm": max((row["absolute_error_mm"] for row in z_checks), default=None),
        "own_source_candidate_projection_checks": len(candidate_projection_errors),
        "own_source_candidate_projection_max_error_px": max(candidate_projection_errors, default=None),
        "own_source_candidate_plane_residual_max_mm": max(plane_residuals, default=None),
        "source_dicom_full_hash_count": sum(str(raw_root.resolve()) in path for path in input_files),
        "source_header_records": len(source_headers), "array_bytes": estimated, "disk_limit_bytes": max_bytes,
        "elapsed_seconds": time.monotonic() - started, "counters": dict(counters),
        "limitations": ["Bounded metadata-selected microshard; no all4407 cache or GPU/inference integration.",
            "Joint XYZ remains coronal locator candidate; independent ML/AP anatomical center estimation and landmark validation are pending.",
            "Different source frames are not averaged; representative uses one best-confidence coronal source per knee component.",
            "Same FrameOfReference and in-FOV projections do not establish anatomical registration.",
            "No crop is moved. No gain in AUC or clinical localization is measured."]}
    _json(output / "CPU_PILOT_REPORT.json", report)
    files = {path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
             for path in output.iterdir() if path.is_file()}
    _json(output / "COMPLETED.json", {"format": FORMAT, "cache_spec_sha256": output_spec["semantic_sha256"],
                                     "studies": len(uids), "images": len(images), "tokens": len(refs), "files": files})
    validate_native_bank(output, full_hash=True)
    return report


def validate_native_bank(directory: Path, *, full_hash: bool = True) -> dict:
    marker = json.loads((directory / "COMPLETED.json").read_text())
    if marker["format"] != FORMAT:
        raise ValueError("Unsupported native bank format")
    for name, record in marker["files"].items():
        path = directory / name
        if path.stat().st_size != record["bytes"] or (full_hash and sha256_file(path) != record["sha256"]):
            raise ValueError(f"Native bank hash/size mismatch: {name}")
    spec = json.loads((directory / "cache_spec.json").read_text())
    claimed = spec.pop("semantic_sha256")
    if semantic_hash(spec) != claimed or marker["cache_spec_sha256"] != claimed:
        raise ValueError("Native bank semantic hash mismatch")
    arrays = {path.stem: np.load(path, mmap_mode="r", allow_pickle=False) for path in directory.glob("*.npy")}
    p, t = len(arrays["images"]), len(arrays["token_channels"])
    if (p != marker["images"] or t != marker["tokens"] or
        len(arrays["study_offsets"]) != marker["studies"] + 1 or
        len(arrays["token_offsets"]) != marker["studies"] + 1):
        raise ValueError("Bank completion counts do not match arrays")
    if arrays["images"].dtype != np.uint8 or arrays["token_channels"].shape != (t, 3):
        raise ValueError("Invalid bank image/reference layout")
    for name, end in (("study_offsets", p), ("token_offsets", t)):
        offsets = arrays[name]
        if offsets.dtype != np.int64 or offsets[0] != 0 or offsets[-1] != end or (np.diff(offsets) < 0).any():
            raise ValueError(f"Invalid {name}")
    refs, masks = arrays["token_channels"], arrays["channel_valid"]
    if refs.dtype != np.int64 or masks.dtype != bool or refs.shape != masks.shape:
        raise ValueError("Invalid reference/mask types")
    if ((refs < -1) | (refs >= p)).any() or (masks & (refs < 0)).any():
        raise ValueError("Unsafe bank references")
    if not np.array_equal(arrays["token_valid"], masks[:, 1]):
        raise ValueError("Center mask drift")
    for name in ("geometry", "geometry_valid", "model_to_source", "model_to_lps", "source_index",
                 "source_slice_count", "series_index", "knee_component_index"):
        if len(arrays[name]) != p:
            raise ValueError(f"Per-plane geometry/index count mismatch: {name}")
    for name in ("token_valid", "slot_ids", "neighbor_status", "token_source_index", "token_duplicate_of"):
        if len(arrays[name]) != t:
            raise ValueError(f"Per-token count mismatch: {name}")
    for a, b, ta, tb in zip(arrays["study_offsets"][:-1], arrays["study_offsets"][1:],
                             arrays["token_offsets"][:-1], arrays["token_offsets"][1:]):
        study_refs = refs[ta:tb]
        if ((study_refs >= 0) & ((study_refs < a) | (study_refs >= b))).any():
            raise ValueError("Cross-study bank reference")
    return {"status": "PASS", "studies": len(arrays["study_offsets"]) - 1, "images": p, "tokens": t}


class NativeNeighborBank:
    """Read-only mmap consumer; optional selection changes centers, never adjacency."""

    def __init__(self, directory: str | Path, *, full_hash: bool = True):
        self.directory = Path(directory)
        validate_native_bank(self.directory, full_hash=full_hash)
        self.arrays = {path.stem: np.load(path, mmap_mode="r", allow_pickle=False)
                       for path in self.directory.glob("*.npy")}
        self.dictionary = json.loads((self.directory / "dictionaries.json").read_text())
        self.study_rows = {uid: index for index, uid in enumerate(self.dictionary["study_uids"])}

    def load_study(self, uid: str, center_indices: Sequence[int] | None = None) -> dict:
        row = self.study_rows[uid]
        start, end = self.arrays["token_offsets"][row:row+2]
        if center_indices is None:
            selected = np.arange(start, end, dtype=np.int64)
        else:
            local = np.asarray(center_indices)
            if local.ndim != 1 or local.dtype.kind not in "iu" or ((local < 0) | (local >= end-start)).any():
                raise ValueError("Selected centers must be valid local integer token indices")
            selected = local.astype(np.int64) + start
        refs = np.asarray(self.arrays["token_channels"][selected])
        channel_valid = np.asarray(self.arrays["channel_valid"][selected])
        return {"uid": uid, "tokens_u8": assemble_native_triplets(self.arrays["images"], refs, channel_valid),
                "token_mask": np.asarray(self.arrays["token_valid"][selected]),
                "channel_valid": channel_valid, "bank_refs": refs,
                "slot_ids": np.asarray(self.arrays["slot_ids"][selected]),
                "neighbor_status": np.asarray(self.arrays["neighbor_status"][selected])}
