"""DICOM ordering and deterministic 2.5D preprocessing."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


MANIFEST_CENTERING_OFFSETS_MM = {
    "coronal_first_v1": 0.0,
    "coronal_first_superior16p4_v1": 16.4,
    "coronal_first_superior20_v1": 20.0,
}


def _as_float_array(value: Any, expected: int) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    return array if len(array) >= expected else None


def physical_slice_position(dataset: Any) -> float | None:
    """Project ImagePositionPatient onto the slice normal from IOP."""

    orientation = _as_float_array(getattr(dataset, "ImageOrientationPatient", None), 6)
    position = _as_float_array(getattr(dataset, "ImagePositionPatient", None), 3)
    if orientation is None or position is None:
        return None
    normal = np.cross(orientation[:3], orientation[3:6])
    norm = float(np.linalg.norm(normal))
    if norm < 1e-8:
        return None
    return float(np.dot(position[:3], normal / norm))


def dicom_sort_key(dataset: Any, fallback_name: str = "") -> tuple[int, float, float, str]:
    physical = physical_slice_position(dataset)
    if physical is not None and math.isfinite(physical):
        return (0, physical, 0.0, fallback_name)
    try:
        location = float(getattr(dataset, "SliceLocation"))
        if math.isfinite(location):
            return (1, location, 0.0, fallback_name)
    except (AttributeError, TypeError, ValueError):
        pass
    try:
        instance = float(getattr(dataset, "InstanceNumber"))
        if math.isfinite(instance):
            return (2, instance, 0.0, fallback_name)
    except (AttributeError, TypeError, ValueError):
        pass
    return (3, 0.0, 0.0, fallback_name)


def sample_slice_indices(n_slices: int, count: int) -> np.ndarray:
    if n_slices <= 0:
        raise ValueError("n_slices must be positive")
    if count <= 0:
        raise ValueError("count must be positive")
    if count == 1:
        return np.asarray([(n_slices - 1) // 2], dtype=int)
    return np.rint(np.linspace(0, n_slices - 1, num=count)).astype(int)


def triplet_indices(centres: Sequence[int], n_slices: int) -> np.ndarray:
    centres_array = np.asarray(centres, dtype=int)
    offsets = np.asarray([-1, 0, 1], dtype=int)
    return np.clip(centres_array[:, None] + offsets[None, :], 0, n_slices - 1)


def pixel_spacing(dataset: Any) -> tuple[float, float] | None:
    spacing = _as_float_array(getattr(dataset, "PixelSpacing", None), 2)
    if spacing is None or not np.isfinite(spacing[:2]).all() or np.any(spacing[:2] <= 0):
        return None
    return float(spacing[0]), float(spacing[1])


def dicom_pixels(dataset: Any) -> np.ndarray:
    pixels = np.asarray(dataset.pixel_array, dtype=np.float32)
    if pixels.ndim != 2:
        raise ValueError(f"Expected a 2D MR slice, got shape {pixels.shape}")
    slope = float(getattr(dataset, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0) or 0.0)
    pixels = pixels * slope + intercept
    if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        pixels = float(pixels.max() + pixels.min()) - pixels
    return pixels


def center_crop_mm(
    image: np.ndarray,
    spacing: tuple[float, float] | None,
    crop_mm: float | None,
) -> np.ndarray:
    if crop_mm is None or crop_mm <= 0 or spacing is None:
        return image
    rows, cols = image.shape
    # Never invent anatomy by padding when the acquired FOV is smaller than
    # the requested crop. In that dimension, preserve the complete image.
    target_rows = min(rows, max(1, int(round(crop_mm / spacing[0]))))
    target_cols = min(cols, max(1, int(round(crop_mm / spacing[1]))))
    start_row = max(0, (rows - target_rows) // 2)
    start_col = max(0, (cols - target_cols) // 2)
    return image[start_row : start_row + target_rows, start_col : start_col + target_cols]


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    if image.shape == (size, size):
        return image.astype(np.float32, copy=False)
    try:
        from PIL import Image

        resized = Image.fromarray(image.astype(np.float32), mode="F").resize(
            (size, size), resample=Image.Resampling.BILINEAR
        )
        return np.asarray(resized, dtype=np.float32)
    except ImportError:
        # Dependency-free nearest-neighbour fallback, mainly for tests/audits.
        row_index = np.rint(np.linspace(0, image.shape[0] - 1, size)).astype(int)
        col_index = np.rint(np.linspace(0, image.shape[1] - 1, size)).astype(int)
        return image[np.ix_(row_index, col_index)].astype(np.float32, copy=False)


def robust_scale(
    images: np.ndarray,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> np.ndarray:
    if not 0 <= percentile_low < percentile_high <= 100:
        raise ValueError("Expected 0 <= percentile_low < percentile_high <= 100")
    finite = images[np.isfinite(images)]
    if finite.size == 0:
        return np.zeros_like(images, dtype=np.float32)
    low, high = np.percentile(finite, [percentile_low, percentile_high])
    if not np.isfinite(high - low) or high <= low:
        return np.zeros_like(images, dtype=np.float32)
    result = np.clip((images.astype(np.float32) - low) / (high - low), 0.0, 1.0)
    return result.astype(np.float32, copy=False)


def discover_dicom_files(series_dir: str | Path) -> list[Path]:
    series_dir = Path(series_dir)
    if not series_dir.is_dir():
        raise FileNotFoundError(f"DICOM series directory does not exist: {series_dir}")
    files = sorted(path for path in series_dir.rglob("*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No files found in DICOM series directory: {series_dir}")
    return files


def preprocess_dicom_series(
    series_dir: str | Path,
    *,
    image_size: int,
    slices_per_series: int,
    crop_mm: float | None = 180.0,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
    centering_mode: str = "geometric",
    centering_decision: Any | None = None,
) -> np.ndarray:
    """Return fixed `[K, 3, H, W]` float32 2.5D tokens in [0, 1].

    ``centering_mode="geometric"`` preserves the original preprocessing.
    ``"joint_line_v1"`` moves only confident series; ``"joint_line_full_v1"``
    uses a fixed inferior prior after locator abstention.  The study-aware
    The study-aware ``coronal_first`` modes require an explicit frozen manifest
    decision; this function never tries to infer a cross-plane decision from
    one series.  Their immutable mode names map to fixed patient-superior
    offsets so distinct offsets cannot share a cache identity.
    """

    try:
        import pydicom
    except ImportError as error:
        raise RuntimeError(
            "DICOM preprocessing requires pydicom; install the 'train' extra"
        ) from error

    paths = discover_dicom_files(series_dir)
    datasets: list[tuple[Path, Any]] = []
    failures: list[str] = []
    for path in paths:
        try:
            datasets.append((path, pydicom.dcmread(path)))
        except Exception as error:  # pydicom raises several format/decode exceptions.
            failures.append(f"{path.name}: {error}")
    if not datasets:
        preview = "; ".join(failures[:3])
        raise ValueError(f"No readable DICOM slices in {series_dir}. {preview}")

    datasets.sort(key=lambda pair: dicom_sort_key(pair[1], pair[0].name))
    if centering_mode not in {
        "geometric",
        "joint_line_v1",
        "joint_line_full_v1",
        *MANIFEST_CENTERING_OFFSETS_MM,
    }:
        raise ValueError(f"Unknown centering_mode: {centering_mode}")
    if centering_mode in {"joint_line_v1", "joint_line_full_v1"}:
        from .study_centering import locate_series_center

        centering_decision = locate_series_center(
            [dataset for _, dataset in datasets],
            coverage_policy=(
                "full_non_axial"
                if centering_mode == "joint_line_full_v1"
                else "safe"
            ),
        )
    elif centering_mode in MANIFEST_CENTERING_OFFSETS_MM:
        if centering_decision is None:
            raise ValueError(
                f"{centering_mode} requires an explicit frozen centering decision"
            )
    else:
        centering_decision = None
    centres = sample_slice_indices(len(datasets), slices_per_series)
    indices = triplet_indices(centres, len(datasets))
    unique_indices = sorted(set(indices.reshape(-1).tolist()))

    prepared: dict[int, np.ndarray] = {}
    for index in unique_indices:
        dataset = datasets[index][1]
        image = dicom_pixels(dataset)
        if centering_decision is None:
            image = center_crop_mm(image, pixel_spacing(dataset), crop_mm)
        else:
            from .study_centering import crop_image_for_decision

            image = crop_image_for_decision(
                image,
                dataset,
                centering_decision,
                crop_mm=crop_mm,
                profile="local_rectangular_mm",
                superior_to_joint_mm=MANIFEST_CENTERING_OFFSETS_MM.get(
                    centering_mode, 0.0
                ),
            )
        prepared[index] = resize_image(image, image_size)

    stacked_unique = np.stack([prepared[index] for index in unique_indices], axis=0)
    scaled_unique = robust_scale(stacked_unique, percentile_low, percentile_high)
    scaled = {index: scaled_unique[position] for position, index in enumerate(unique_indices)}
    tokens = np.stack(
        [np.stack([scaled[int(index)] for index in triplet], axis=0) for triplet in indices],
        axis=0,
    )
    return tokens.astype(np.float32, copy=False)
