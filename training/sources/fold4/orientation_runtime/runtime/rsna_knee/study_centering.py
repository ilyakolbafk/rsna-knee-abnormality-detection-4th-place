"""Fail-closed, series-level crop centering for knee MRI.

The module never rewrites a DICOM.  It estimates one physical displacement for
an entire series and converts that displacement back to native pixel
coordinates at crop time.  Unsupported planes and uncertain estimates retain
the existing geometric-centre crop.

The coronal locator is deliberately modest: on several central slices it looks
for a strong, bilaterally supported horizontal interface in the lower half of
the image.  This is an engineering landmark for crop alignment, not a meniscus
segmentation or a clinical measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
from scipy import ndimage, signal

from .anatomy_roi import (
    anatomical_plane,
    image_center_patient,
    infer_knee_laterality,
    pixel_patient_position,
)
from .dicom import dicom_pixels, pixel_spacing, robust_scale


CenteringMode = Literal["joint_line", "fixed_inferior_prior", "global_fallback"]
CoveragePolicy = Literal["safe", "full_non_axial"]
CropProfile = Literal[
    "local_rectangular_mm",
    "dino_v2_square",
    "dino_v3_clipped_square",
    "coatnet_square",
]


@dataclass(frozen=True)
class CoronalSliceCandidate:
    """One image-native joint-line proposal on a coronal slice."""

    index: int
    native_row: float
    native_column: float
    inferior_shift_mm: float
    peak_strength: float
    prominence: float
    bilateral_delta_mm: float
    confidence: float


@dataclass(frozen=True)
class SeriesCenteringDecision:
    """One auditable physical crop displacement for a DICOM series."""

    mode: CenteringMode
    reason: str
    plane: str | None
    confidence: float
    inferior_shift_mm: float | None
    patient_offset_mm: tuple[float, float, float] | None
    selected_indices: tuple[int, ...]
    accepted_indices: tuple[int, ...]
    candidates: tuple[CoronalSliceCandidate, ...]
    median_absolute_deviation_mm: float | None


@dataclass(frozen=True)
class CropBox:
    """Integer crop bounds; ``bottom`` and ``right`` are exclusive."""

    top: int
    left: int
    bottom: int
    right: int
    requested_center_row: float
    requested_center_column: float
    actual_center_row: float
    actual_center_column: float
    was_clamped: bool
    profile: CropProfile


@dataclass(frozen=True)
class CoronalJointHeight:
    """A coronal image-supported joint height in patient LPS coordinates."""

    status: Literal["candidate", "fallback"]
    reason: str
    z_mm: float | None
    confidence: float
    source_indices: tuple[int, ...]
    mad_z_mm: float | None


@dataclass(frozen=True)
class _SeriesPatientVolume:
    origin: np.ndarray
    column_direction: np.ndarray
    row_direction: np.ndarray
    normal: np.ndarray
    width_mm: float
    height_mm: float
    slice_min_mm: float
    slice_max_mm: float
    center: np.ndarray
    bounds_low: np.ndarray
    bounds_high: np.ndarray


def _fallback(
    reason: str,
    *,
    plane: str | None = None,
    confidence: float = 0.0,
    selected_indices: Sequence[int] = (),
    candidates: Sequence[CoronalSliceCandidate] = (),
    mad_mm: float | None = None,
) -> SeriesCenteringDecision:
    return SeriesCenteringDecision(
        mode="global_fallback",
        reason=reason,
        plane=plane,
        confidence=float(confidence),
        inferior_shift_mm=None,
        patient_offset_mm=None,
        selected_indices=tuple(int(value) for value in selected_indices),
        accepted_indices=(),
        candidates=tuple(candidates),
        median_absolute_deviation_mm=mad_mm,
    )


def _height_fallback(reason: str, *, confidence: float = 0.0) -> CoronalJointHeight:
    return CoronalJointHeight(
        status="fallback",
        reason=reason,
        z_mm=None,
        confidence=float(confidence),
        source_indices=(),
        mad_z_mm=None,
    )


def _single_series_text_tag(datasets: Sequence[Any], name: str) -> str | None:
    values = {
        str(getattr(dataset, name, "") or "").strip()
        for dataset in datasets
    }
    values.discard("")
    return next(iter(values)) if len(values) == 1 else None


def series_patient_bounds(
    datasets: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return an axis-aligned LPS box around every pixel centre in a series."""

    points: list[np.ndarray] = []
    for dataset in datasets:
        try:
            rows, columns = int(dataset.Rows), int(dataset.Columns)
        except (AttributeError, TypeError, ValueError):
            return None
        if rows <= 0 or columns <= 0:
            return None
        for row, column in (
            (0.0, 0.0),
            (0.0, columns - 1.0),
            (rows - 1.0, 0.0),
            (rows - 1.0, columns - 1.0),
        ):
            point = pixel_patient_position(dataset, row, column)
            if point is None or not np.isfinite(point).all():
                return None
            points.append(point)
    if not points:
        return None
    values = np.asarray(points, dtype=float)
    return values.min(axis=0), values.max(axis=0)


def _series_patient_volume(datasets: Sequence[Any]) -> _SeriesPatientVolume | None:
    if not datasets:
        return None
    reference = datasets[0]
    orientation = _orientation(reference)
    spacing = pixel_spacing(reference)
    bounds = series_patient_bounds(datasets)
    try:
        origin = np.asarray(reference.ImagePositionPatient, dtype=float).reshape(3)
        rows, columns = int(reference.Rows), int(reference.Columns)
    except (AttributeError, TypeError, ValueError):
        return None
    if orientation is None or spacing is None or bounds is None:
        return None
    column_direction, row_direction, normal = orientation
    positions: list[float] = []
    centers: list[np.ndarray] = []
    for dataset in datasets:
        current_orientation = _orientation(dataset)
        current_spacing = pixel_spacing(dataset)
        center = image_center_patient(dataset)
        try:
            position = np.asarray(dataset.ImagePositionPatient, dtype=float).reshape(3)
            current_rows, current_columns = int(dataset.Rows), int(dataset.Columns)
        except (AttributeError, TypeError, ValueError):
            return None
        if (
            current_orientation is None
            or current_spacing is None
            or center is None
            or current_rows != rows
            or current_columns != columns
            or not np.allclose(current_spacing, spacing, atol=1e-4)
            or any(
                float(np.dot(first, second)) < 0.995
                for first, second in zip(orientation, current_orientation)
            )
        ):
            return None
        positions.append(float(np.dot(position - origin, normal)))
        centers.append(center)
    ordered = np.unique(np.round(np.sort(np.asarray(positions, dtype=float)), decimals=5))
    if len(ordered) > 1:
        half_slice = 0.5 * float(np.median(np.diff(ordered)))
    else:
        try:
            half_slice = 0.5 * float(getattr(reference, "SliceThickness"))
        except (AttributeError, TypeError, ValueError):
            half_slice = 0.0
        if not np.isfinite(half_slice) or half_slice < 0:
            half_slice = 0.0
    low, high = bounds
    return _SeriesPatientVolume(
        origin=origin,
        column_direction=column_direction,
        row_direction=row_direction,
        normal=normal,
        width_mm=float((columns - 1.0) * spacing[1]),
        height_mm=float((rows - 1.0) * spacing[0]),
        slice_min_mm=float(np.min(positions) - half_slice),
        slice_max_mm=float(np.max(positions) + half_slice),
        center=np.mean(np.asarray(centers, dtype=float), axis=0),
        bounds_low=low,
        bounds_high=high,
    )


def _point_volume_margin(point: np.ndarray, volume: _SeriesPatientVolume) -> float:
    offset = np.asarray(point, dtype=float) - volume.origin
    column_mm = float(np.dot(offset, volume.column_direction))
    row_mm = float(np.dot(offset, volume.row_direction))
    slice_mm = float(np.dot(offset, volume.normal))
    return min(
        column_mm,
        volume.width_mm - column_mm,
        row_mm,
        volume.height_mm - row_mm,
        slice_mm - volume.slice_min_mm,
        volume.slice_max_mm - slice_mm,
    )


def same_knee_fov_overlap(
    first: Sequence[Any],
    second: Sequence[Any],
    *,
    tolerance_mm: float = 0.0,
) -> bool:
    """Require reciprocal centre containment and LPS AABB overlap."""

    if not np.isfinite(tolerance_mm) or tolerance_mm < 0:
        raise ValueError("tolerance_mm must be finite and non-negative")
    first_volume = _series_patient_volume(first)
    second_volume = _series_patient_volume(second)
    if first_volume is None or second_volume is None:
        return False
    overlap = np.minimum(first_volume.bounds_high, second_volume.bounds_high) - np.maximum(
        first_volume.bounds_low, second_volume.bounds_low
    )
    reciprocal_margin = min(
        _point_volume_margin(first_volume.center, second_volume),
        _point_volume_margin(second_volume.center, first_volume),
    )
    return bool(
        np.all(overlap >= -float(tolerance_mm))
        and reciprocal_margin >= -float(tolerance_mm)
    )


def _orientation(dataset: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        values = np.asarray(dataset.ImageOrientationPatient, dtype=float).reshape(6)
    except (AttributeError, TypeError, ValueError):
        return None
    if not np.isfinite(values).all():
        return None
    column_direction = values[:3]
    row_direction = values[3:]
    if (
        not np.isclose(np.linalg.norm(column_direction), 1.0, atol=1e-3)
        or not np.isclose(np.linalg.norm(row_direction), 1.0, atol=1e-3)
        or not np.isclose(float(np.dot(column_direction, row_direction)), 0.0, atol=1e-3)
    ):
        return None
    normal = np.cross(column_direction, row_direction)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-8:
        return None
    return column_direction, row_direction, normal / norm


def in_plane_inferior_direction(
    dataset: Any, *, minimum_projection_norm: float = 0.90
) -> np.ndarray:
    """Return a unit vector pointing patient-inferior within the image plane."""

    orientation = _orientation(dataset)
    if orientation is None:
        raise ValueError("invalid ImageOrientationPatient")
    _, _, normal = orientation
    inferior = np.asarray([0.0, 0.0, -1.0])
    projected = inferior - float(np.dot(inferior, normal)) * normal
    norm = float(np.linalg.norm(projected))
    if norm < minimum_projection_norm:
        # For axial acquisitions, inferior is mostly through-plane.  Moving an
        # in-plane crop cannot implement the desired anatomical displacement.
        raise ValueError("patient-inferior direction is not sufficiently in-plane")
    return projected / norm


def inferior_pixel_shift(dataset: Any, distance_mm: float) -> tuple[float, float]:
    """Convert a signed in-plane inferior distance to native row/column pixels."""

    orientation = _orientation(dataset)
    spacing = pixel_spacing(dataset)
    if orientation is None or spacing is None or not np.isfinite(distance_mm):
        raise ValueError("invalid DICOM geometry")
    column_direction, row_direction, _ = orientation
    inferior = in_plane_inferior_direction(dataset)
    delta_row = float(distance_mm) * float(np.dot(inferior, row_direction)) / spacing[0]
    delta_column = (
        float(distance_mm) * float(np.dot(inferior, column_direction)) / spacing[1]
    )
    return delta_row, delta_column


def patient_offset_pixel_shift(
    dataset: Any, offset_mm: Sequence[float]
) -> tuple[float, float]:
    """Project a patient-coordinate displacement into a native image plane."""

    orientation = _orientation(dataset)
    spacing = pixel_spacing(dataset)
    offset = np.asarray(offset_mm, dtype=float).reshape(-1)
    if (
        orientation is None
        or spacing is None
        or len(offset) < 3
        or not np.isfinite(offset[:3]).all()
    ):
        raise ValueError("invalid DICOM geometry or patient offset")
    column_direction, row_direction, _ = orientation
    delta_row = float(np.dot(offset[:3], row_direction)) / spacing[0]
    delta_column = float(np.dot(offset[:3], column_direction)) / spacing[1]
    return delta_row, delta_column


def _canonical_coronal(
    dataset: Any, *, minimum_inferior_axis_alignment: float = 0.80
) -> tuple[np.ndarray, float, bool, bool]:
    """Return image with patient-inferior down and its native-axis mapping.

    The booleans are ``vertical_is_native_row`` and ``vertical_was_flipped``.
    The operation is a transpose/flip only, so no interpolation is introduced.
    """

    if anatomical_plane(dataset) != "Coronal":
        raise ValueError("slice is not sufficiently coronal")
    orientation = _orientation(dataset)
    spacing = pixel_spacing(dataset)
    if orientation is None or spacing is None:
        raise ValueError("missing coronal geometry")
    column_direction, row_direction, _ = orientation
    vertical_is_native_row = abs(float(row_direction[2])) >= abs(
        float(column_direction[2])
    )
    axis = row_direction if vertical_is_native_row else column_direction
    if abs(float(axis[2])) < minimum_inferior_axis_alignment:
        raise ValueError("no stable native patient-inferior image axis")
    image = dicom_pixels(dataset)
    canonical = image if vertical_is_native_row else image.T
    vertical_was_flipped = bool(axis[2] > 0.0)
    if vertical_was_flipped:
        canonical = canonical[::-1]
    vertical_spacing = spacing[0] if vertical_is_native_row else spacing[1]
    return (
        canonical.astype(np.float32, copy=False),
        float(vertical_spacing),
        vertical_is_native_row,
        vertical_was_flipped,
    )


def _row_profile(gradient: np.ndarray, start: int, end: int) -> np.ndarray:
    band = gradient[:, start:end]
    if band.shape[1] < 4:
        raise ValueError("coronal side band is too narrow")
    return np.quantile(band, 0.85, axis=1) + 0.40 * np.mean(band, axis=1)


def locate_coronal_joint_line_on_slice(
    dataset: Any,
    *,
    index: int,
) -> CoronalSliceCandidate:
    """Locate a crop-alignment landmark on one coronal slice."""

    canonical, vertical_spacing, vertical_is_row, flipped = _canonical_coronal(dataset)
    height, width = canonical.shape
    if height < 96 or width < 96:
        raise ValueError("coronal field of view is too small")
    scaled = robust_scale(canonical, 1.0, 99.0)
    if float(np.std(scaled)) < 1e-3:
        raise ValueError("flat coronal image")
    smooth = ndimage.gaussian_filter(scaled, sigma=1.4)
    gradient = np.abs(ndimage.sobel(smooth, axis=0, mode="nearest")) / 8.0

    left_start, left_end = int(round(0.15 * width)), int(round(0.44 * width))
    right_start, right_end = int(round(0.56 * width)), int(round(0.85 * width))
    left_profile = _row_profile(gradient, left_start, left_end)
    right_profile = _row_profile(gradient, right_start, right_end)
    combined = left_profile + right_profile
    search_start = int(round(0.48 * height))
    search_end = int(round(0.75 * height))
    search = combined[search_start:search_end]
    if len(search) < 24 or not np.isfinite(search).all():
        raise ValueError("invalid coronal search profile")
    peaks, _ = signal.find_peaks(search, distance=max(2, int(round(0.025 * height))))
    if len(peaks) == 0:
        peak_local = int(np.argmax(search))
        prominence = 0.0
    else:
        peak_local = int(peaks[int(np.argmax(search[peaks]))])
        prominence = float(signal.peak_prominences(search, [peak_local])[0][0])
    peak_row = search_start + peak_local
    peak_strength = float(search[peak_local])

    # Each compartment should support approximately the same row.  Measure the
    # strongest side-specific response near the shared peak rather than letting
    # either side jump to an unrelated distant interface.
    radius = max(3, int(round(0.06 * height)))
    local_start = max(search_start, peak_row - radius)
    local_end = min(search_end, peak_row + radius + 1)
    left_row = local_start + int(np.argmax(left_profile[local_start:local_end]))
    right_row = local_start + int(np.argmax(right_profile[local_start:local_end]))
    bilateral_delta_mm = abs(left_row - right_row) * vertical_spacing

    if vertical_is_row:
        native_row = float(height - 1 - peak_row if flipped else peak_row)
        native_column = (float(dataset.Columns) - 1.0) / 2.0
    else:
        native_column = float(height - 1 - peak_row if flipped else peak_row)
        native_row = (float(dataset.Rows) - 1.0) / 2.0

    centre_row = (float(dataset.Rows) - 1.0) / 2.0
    centre_column = (float(dataset.Columns) - 1.0) / 2.0
    orientation = _orientation(dataset)
    spacing = pixel_spacing(dataset)
    if orientation is None or spacing is None:
        raise ValueError("missing coronal geometry")
    column_direction, row_direction, _ = orientation
    displacement = (
        row_direction * spacing[0] * (native_row - centre_row)
        + column_direction * spacing[1] * (native_column - centre_column)
    )
    inferior = in_plane_inferior_direction(dataset)
    inferior_shift_mm = float(np.dot(displacement, inferior))

    strength_score = float(np.clip(peak_strength / 0.25, 0.0, 1.0))
    prominence_score = float(np.clip(prominence / 0.10, 0.0, 1.0))
    bilateral_score = float(np.clip(1.0 - bilateral_delta_mm / 12.0, 0.0, 1.0))
    confidence = 0.45 * strength_score + 0.35 * prominence_score + 0.20 * bilateral_score
    return CoronalSliceCandidate(
        index=int(index),
        native_row=native_row,
        native_column=native_column,
        inferior_shift_mm=inferior_shift_mm,
        peak_strength=peak_strength,
        prominence=prominence,
        bilateral_delta_mm=float(bilateral_delta_mm),
        confidence=float(confidence),
    )


def _central_locator_indices(n_slices: int, maximum_slices: int = 5) -> np.ndarray:
    if n_slices <= 0:
        raise ValueError("n_slices must be positive")
    count = min(maximum_slices, n_slices)
    if count == 1:
        return np.asarray([(n_slices - 1) // 2], dtype=int)
    low = int(round(0.35 * (n_slices - 1)))
    high = int(round(0.65 * (n_slices - 1)))
    return np.unique(np.rint(np.linspace(low, high, count)).astype(int))


def _locate_coronal_series(
    datasets: Sequence[Any],
    *,
    maximum_shift_mm: float,
    minimum_slice_confidence: float,
    minimum_series_confidence: float,
) -> SeriesCenteringDecision:
    selected = _central_locator_indices(len(datasets))
    candidates: list[CoronalSliceCandidate] = []
    for index in selected:
        try:
            candidate = locate_coronal_joint_line_on_slice(
                datasets[int(index)], index=int(index)
            )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            continue
        candidates.append(candidate)
    if not candidates:
        return _fallback(
            "no_valid_coronal_candidates",
            plane="Coronal",
            selected_indices=selected,
        )

    usable = [
        candidate
        for candidate in candidates
        if candidate.confidence >= minimum_slice_confidence
        and 0.0 <= candidate.inferior_shift_mm <= maximum_shift_mm
    ]
    minimum_usable = 1 if len(selected) == 1 else min(3, len(selected))
    if len(usable) < minimum_usable:
        confidence = float(np.median([value.confidence for value in candidates]))
        return _fallback(
            "too_few_confident_coronal_candidates",
            plane="Coronal",
            confidence=confidence,
            selected_indices=selected,
            candidates=candidates,
        )

    shifts = np.asarray([value.inferior_shift_mm for value in usable], dtype=float)
    centre = float(np.median(shifts))
    deviations = np.abs(shifts - centre)
    stable_mask = deviations <= 7.0
    stable = [value for value, keep in zip(usable, stable_mask) if bool(keep)]
    minimum_stable = 1 if len(selected) == 1 else min(3, len(selected))
    if len(stable) < minimum_stable:
        return _fallback(
            "unstable_coronal_candidates",
            plane="Coronal",
            confidence=float(np.median([value.confidence for value in usable])),
            selected_indices=selected,
            candidates=candidates,
            mad_mm=float(np.median(deviations)),
        )

    stable_shifts = np.asarray([value.inferior_shift_mm for value in stable], dtype=float)
    shift = float(np.median(stable_shifts))
    mad = float(np.median(np.abs(stable_shifts - shift)))
    consensus = len(stable) / max(len(selected), 1)
    slice_confidence = float(np.median([value.confidence for value in stable]))
    stability = float(np.clip(1.0 - mad / 7.0, 0.0, 1.0))
    confidence = 0.55 * slice_confidence + 0.25 * consensus + 0.20 * stability
    if confidence < minimum_series_confidence:
        return _fallback(
            "below_coronal_series_confidence",
            plane="Coronal",
            confidence=confidence,
            selected_indices=selected,
            candidates=candidates,
            mad_mm=mad,
        )
    if not 0.0 <= shift <= maximum_shift_mm:
        return _fallback(
            "coronal_shift_out_of_bounds",
            plane="Coronal",
            confidence=confidence,
            selected_indices=selected,
            candidates=candidates,
            mad_mm=mad,
        )
    return SeriesCenteringDecision(
        mode="joint_line",
        reason="accepted_coronal_joint_line",
        plane="Coronal",
        confidence=float(confidence),
        inferior_shift_mm=shift,
        patient_offset_mm=None,
        selected_indices=tuple(int(value) for value in selected),
        accepted_indices=tuple(value.index for value in stable),
        candidates=tuple(candidates),
        median_absolute_deviation_mm=mad,
    )


def _locate_sagittal_series(
    datasets: Sequence[Any], *, maximum_shift_mm: float
) -> SeriesCenteringDecision:
    # Imported lazily to keep the coronal-only path light and to make the
    # sagittal locator's stricter geometry contract explicit.
    from .joint_line_locator import locate_joint_line_multislice

    laterality = infer_knee_laterality(datasets)
    if laterality.status != "resolved" or laterality.side is None:
        return _fallback("sagittal_laterality_unresolved", plane="Sagittal")
    decision = locate_joint_line_multislice(datasets, laterality.side)
    if decision.status != "candidate" or decision.y_mm is None or decision.z_mm is None:
        return _fallback(
            f"sagittal_locator_{decision.reason}",
            plane="Sagittal",
            confidence=decision.confidence,
            selected_indices=decision.selected_indices,
        )
    reference = datasets[(len(datasets) - 1) // 2]
    centre = image_center_patient(reference)
    if centre is None:
        return _fallback("invalid_sagittal_reference_geometry", plane="Sagittal")
    offset = np.asarray([0.0, decision.y_mm - centre[1], decision.z_mm - centre[2]])
    magnitude = float(np.linalg.norm(offset))
    if not np.isfinite(magnitude) or magnitude > maximum_shift_mm:
        return _fallback(
            "sagittal_shift_out_of_bounds",
            plane="Sagittal",
            confidence=decision.confidence,
            selected_indices=decision.selected_indices,
        )
    return SeriesCenteringDecision(
        mode="joint_line",
        reason="accepted_sagittal_joint_line",
        plane="Sagittal",
        confidence=float(decision.confidence),
        inferior_shift_mm=None,
        patient_offset_mm=tuple(float(value) for value in offset),
        selected_indices=decision.selected_indices,
        accepted_indices=decision.cluster_indices,
        candidates=(),
        median_absolute_deviation_mm=decision.mad_z_mm,
    )


def locate_series_center(
    datasets: Sequence[Any],
    *,
    maximum_shift_mm: float = 35.0,
    minimum_coronal_slice_confidence: float = 0.45,
    minimum_coronal_series_confidence: float = 0.55,
    coverage_policy: CoveragePolicy = "safe",
    fixed_inferior_prior_mm: float = 19.1,
) -> SeriesCenteringDecision:
    """Estimate one deterministic crop displacement for train or test series.

    ``safe`` moves only image-supported proposals.  ``full_non_axial`` uses a
    fixed patient-inferior layout prior when the image locator abstains, giving
    every geometrically valid coronal/sagittal series a shift.  The latter is an
    experimental coverage policy, not equivalent evidence to a detected joint.
    """

    if not datasets:
        return _fallback("empty_series")
    if maximum_shift_mm <= 0:
        raise ValueError("maximum_shift_mm must be positive")
    if coverage_policy not in {"safe", "full_non_axial"}:
        raise ValueError(f"Unknown coverage_policy: {coverage_policy}")
    if not np.isfinite(fixed_inferior_prior_mm) or not (
        0.0 <= fixed_inferior_prior_mm <= maximum_shift_mm
    ):
        raise ValueError("fixed_inferior_prior_mm must be within the shift bound")
    planes = {anatomical_plane(dataset) for dataset in datasets}
    if len(planes) != 1 or None in planes:
        return _fallback("mixed_or_invalid_plane")
    plane = next(iter(planes))
    frame_uids = {
        str(getattr(dataset, "FrameOfReferenceUID", "") or "") for dataset in datasets
    }
    if len(frame_uids) != 1 or "" in frame_uids:
        return _fallback("missing_or_mixed_frame_of_reference", plane=plane)
    orientations = [_orientation(dataset) for dataset in datasets]
    if any(value is None for value in orientations):
        return _fallback("invalid_series_orientation", plane=plane)
    reference = orientations[0]
    assert reference is not None
    reference_column, reference_row, reference_normal = reference
    for orientation in orientations[1:]:
        assert orientation is not None
        column, row, normal = orientation
        if (
            float(np.dot(reference_column, column)) < 0.995
            or float(np.dot(reference_row, row)) < 0.995
            or float(np.dot(reference_normal, normal)) < 0.995
        ):
            return _fallback("inconsistent_series_orientation", plane=plane)
    if plane == "Coronal":
        decision = _locate_coronal_series(
            datasets,
            maximum_shift_mm=maximum_shift_mm,
            minimum_slice_confidence=minimum_coronal_slice_confidence,
            minimum_series_confidence=minimum_coronal_series_confidence,
        )
    elif plane == "Sagittal":
        decision = _locate_sagittal_series(datasets, maximum_shift_mm=maximum_shift_mm)
    else:
        return _fallback("axial_inferior_axis_is_through_plane", plane="Axial")
    if coverage_policy == "safe":
        return decision
    return expand_with_fixed_inferior_prior(
        datasets,
        decision,
        fixed_inferior_prior_mm=fixed_inferior_prior_mm,
        maximum_shift_mm=maximum_shift_mm,
    )


def coronal_joint_height(
    datasets: Sequence[Any],
    decision: SeriesCenteringDecision | None = None,
) -> CoronalJointHeight:
    """Convert an accepted coronal landmark to an absolute patient-Z height."""

    if not datasets:
        return _height_fallback("empty_coronal_series")
    resolved = locate_series_center(datasets) if decision is None else decision
    if (
        resolved.mode != "joint_line"
        or resolved.plane != "Coronal"
        or resolved.reason != "accepted_coronal_joint_line"
    ):
        return _height_fallback(
            f"coronal_locator_{resolved.reason}", confidence=resolved.confidence
        )
    accepted = set(resolved.accepted_indices)
    heights: list[tuple[int, float]] = []
    for candidate in resolved.candidates:
        if candidate.index not in accepted or not 0 <= candidate.index < len(datasets):
            continue
        point = pixel_patient_position(
            datasets[candidate.index], candidate.native_row, candidate.native_column
        )
        if point is None or not np.isfinite(point).all():
            continue
        heights.append((candidate.index, float(point[2])))
    if not heights:
        return _height_fallback(
            "no_physical_coronal_candidates", confidence=resolved.confidence
        )
    values = np.asarray([value for _, value in heights], dtype=float)
    height = float(np.median(values))
    mad = float(np.median(np.abs(values - height)))
    return CoronalJointHeight(
        status="candidate",
        reason="accepted_coronal_absolute_z",
        z_mm=height,
        confidence=float(resolved.confidence),
        source_indices=tuple(index for index, _ in heights),
        mad_z_mm=mad,
    )


def sagittal_center_from_absolute_z(
    sagittal_datasets: Sequence[Any],
    joint_z_mm: float,
    *,
    confidence: float,
    reason: str,
    maximum_shift_mm: float = 35.0,
    source_indices: Sequence[int] = (),
    candidates: Sequence[CoronalSliceCandidate] = (),
    mad_z_mm: float | None = None,
) -> SeriesCenteringDecision:
    """Map an already validated absolute joint height into a sagittal series."""

    if not sagittal_datasets or not np.isfinite(joint_z_mm):
        return _fallback("cross_plane_invalid_absolute_z", plane="Sagittal")
    if {anatomical_plane(dataset) for dataset in sagittal_datasets} != {"Sagittal"}:
        return _fallback("cross_plane_invalid_sagittal_plane", plane="Sagittal")
    target_orientations = [_orientation(dataset) for dataset in sagittal_datasets]
    if any(value is None for value in target_orientations):
        return _fallback("cross_plane_invalid_sagittal_orientation", plane="Sagittal")
    reference_orientation = target_orientations[0]
    assert reference_orientation is not None
    for orientation in target_orientations[1:]:
        assert orientation is not None
        if any(
            float(np.dot(first, second)) < 0.995
            for first, second in zip(reference_orientation, orientation)
        ):
            return _fallback("cross_plane_inconsistent_sagittal_orientation", plane="Sagittal")

    reference = sagittal_datasets[(len(sagittal_datasets) - 1) // 2]
    centre = image_center_patient(reference)
    if centre is None:
        return _fallback("cross_plane_invalid_sagittal_center", plane="Sagittal")
    try:
        inferior = in_plane_inferior_direction(reference)
    except ValueError:
        return _fallback("cross_plane_inferior_not_in_sagittal_plane", plane="Sagittal")
    if abs(float(inferior[2])) < 0.80:
        return _fallback("cross_plane_unstable_sagittal_z_projection", plane="Sagittal")

    distance_mm = float((joint_z_mm - centre[2]) / inferior[2])
    if not np.isfinite(distance_mm) or abs(distance_mm) > maximum_shift_mm:
        return _fallback(
            "cross_plane_shift_out_of_bounds",
            plane="Sagittal",
            confidence=confidence,
        )
    offset = inferior * distance_mm
    try:
        delta_row, delta_column = patient_offset_pixel_shift(reference, offset)
    except ValueError:
        return _fallback("cross_plane_pixel_projection_failed", plane="Sagittal")
    requested_row = (int(reference.Rows) - 1.0) / 2.0 + delta_row
    requested_column = (int(reference.Columns) - 1.0) / 2.0 + delta_column
    if not (
        0.0 <= requested_row <= int(reference.Rows) - 1.0
        and 0.0 <= requested_column <= int(reference.Columns) - 1.0
    ):
        return _fallback(
            "cross_plane_requested_center_outside_fov",
            plane="Sagittal",
            confidence=confidence,
        )
    return SeriesCenteringDecision(
        mode="joint_line",
        reason=reason,
        plane="Sagittal",
        confidence=float(confidence),
        inferior_shift_mm=None,
        patient_offset_mm=tuple(float(value) for value in offset),
        selected_indices=tuple(int(value) for value in source_indices),
        accepted_indices=tuple(int(value) for value in source_indices),
        candidates=tuple(candidates),
        median_absolute_deviation_mm=mad_z_mm,
    )


def transfer_coronal_height_to_sagittal(
    coronal_datasets: Sequence[Any],
    sagittal_datasets: Sequence[Any],
    *,
    coronal_decision: SeriesCenteringDecision | None = None,
    maximum_shift_mm: float = 35.0,
    fov_tolerance_mm: float = 0.0,
    allow_mismatched_frame_of_reference: bool = False,
) -> SeriesCenteringDecision:
    """Map a coronal absolute joint height into one sagittal series.

    A mismatched ``FrameOfReferenceUID`` has no general DICOM transfer
    guarantee.  Callers must opt into the corpus-specific raw-LPS experiment;
    same-study, laterality, geometry, FOV, and shift guards still fail closed.
    """

    if not coronal_datasets or not sagittal_datasets:
        return _fallback("cross_plane_empty_series", plane="Sagittal")
    if not np.isfinite(maximum_shift_mm) or maximum_shift_mm <= 0:
        raise ValueError("maximum_shift_mm must be finite and positive")

    coronal_planes = {anatomical_plane(dataset) for dataset in coronal_datasets}
    sagittal_planes = {anatomical_plane(dataset) for dataset in sagittal_datasets}
    if coronal_planes != {"Coronal"} or sagittal_planes != {"Sagittal"}:
        return _fallback("cross_plane_invalid_planes", plane="Sagittal")

    coronal_study = _single_series_text_tag(coronal_datasets, "StudyInstanceUID")
    sagittal_study = _single_series_text_tag(sagittal_datasets, "StudyInstanceUID")
    if coronal_study is None or sagittal_study is None or coronal_study != sagittal_study:
        return _fallback("cross_plane_study_mismatch", plane="Sagittal")

    coronal_frame = _single_series_text_tag(coronal_datasets, "FrameOfReferenceUID")
    sagittal_frame = _single_series_text_tag(sagittal_datasets, "FrameOfReferenceUID")
    if coronal_frame is None or sagittal_frame is None:
        return _fallback("cross_plane_missing_frame_of_reference", plane="Sagittal")
    shared_frame = coronal_frame == sagittal_frame
    if not shared_frame and not allow_mismatched_frame_of_reference:
        return _fallback("cross_plane_frame_mismatch", plane="Sagittal")

    coronal_side = infer_knee_laterality(coronal_datasets)
    sagittal_side = infer_knee_laterality(sagittal_datasets)
    if coronal_side.status == "conflict" or sagittal_side.status == "conflict":
        return _fallback("cross_plane_laterality_conflict", plane="Sagittal")
    if (
        coronal_side.status == "resolved"
        and sagittal_side.status == "resolved"
        and coronal_side.side != sagittal_side.side
    ):
        return _fallback("cross_plane_opposite_laterality", plane="Sagittal")
    if not same_knee_fov_overlap(
        coronal_datasets, sagittal_datasets, tolerance_mm=fov_tolerance_mm
    ):
        return _fallback("cross_plane_no_same_knee_fov_overlap", plane="Sagittal")

    height = coronal_joint_height(coronal_datasets, coronal_decision)
    if height.status != "candidate" or height.z_mm is None:
        return _fallback(
            f"cross_plane_{height.reason}",
            plane="Sagittal",
            confidence=height.confidence,
        )

    confidence = float(height.confidence if shared_frame else 0.90 * height.confidence)
    reason = (
        "accepted_coronal_height_transfer_shared_frame"
        if shared_frame
        else "accepted_coronal_height_transfer_raw_lps"
    )
    return sagittal_center_from_absolute_z(
        sagittal_datasets,
        height.z_mm,
        confidence=confidence,
        reason=reason,
        maximum_shift_mm=maximum_shift_mm,
        source_indices=height.source_indices,
        candidates=(() if coronal_decision is None else coronal_decision.candidates),
        mad_z_mm=height.mad_z_mm,
    )


def expand_with_fixed_inferior_prior(
    datasets: Sequence[Any],
    decision: SeriesCenteringDecision,
    *,
    fixed_inferior_prior_mm: float = 19.1,
    maximum_shift_mm: float = 35.0,
) -> SeriesCenteringDecision:
    """Fill a safe-policy abstention for valid non-axial series.

    Keeping this transformation separate allows audits to compare the safe and
    full-coverage policies without running the image locator twice.
    """

    if decision.mode == "joint_line" or decision.plane not in {"Coronal", "Sagittal"}:
        return decision
    if not np.isfinite(fixed_inferior_prior_mm) or not (
        0.0 <= fixed_inferior_prior_mm <= maximum_shift_mm
    ):
        raise ValueError("fixed_inferior_prior_mm must be within the shift bound")
    try:
        for dataset in datasets:
            in_plane_inferior_direction(dataset)
    except ValueError:
        return decision
    return SeriesCenteringDecision(
        mode="fixed_inferior_prior",
        reason=f"fixed_inferior_after_{decision.reason}",
        plane=decision.plane,
        confidence=0.25,
        inferior_shift_mm=float(fixed_inferior_prior_mm),
        patient_offset_mm=None,
        selected_indices=decision.selected_indices,
        accepted_indices=(),
        candidates=decision.candidates,
        median_absolute_deviation_mm=decision.median_absolute_deviation_mm,
    )


def requested_center_pixel(
    dataset: Any, decision: SeriesCenteringDecision
) -> tuple[float, float]:
    """Convert a series decision to the requested native image centre."""

    rows, columns = int(dataset.Rows), int(dataset.Columns)
    centre_row = (rows - 1.0) / 2.0
    centre_column = (columns - 1.0) / 2.0
    if decision.mode == "global_fallback":
        return centre_row, centre_column
    if decision.inferior_shift_mm is not None:
        delta_row, delta_column = inferior_pixel_shift(
            dataset, decision.inferior_shift_mm
        )
    elif decision.patient_offset_mm is not None:
        delta_row, delta_column = patient_offset_pixel_shift(
            dataset, decision.patient_offset_mm
        )
    else:
        return centre_row, centre_column
    return centre_row + delta_row, centre_column + delta_column


def superior_to_joint_center_pixel(
    dataset: Any,
    decision: SeriesCenteringDecision,
    *,
    superior_mm: float,
) -> tuple[float, float]:
    """Place an accepted joint centre superiorly without moving anatomy horizontally.

    Patient ``+Z`` points superiorly.  For coronal images the preserved
    anatomical horizontal coordinate is patient ``X``; for sagittal images it
    is patient ``Y``.  Solving the two-coordinate DICOM affine is important for
    oblique acquisitions, where a pure patient-superior displacement may use
    both native image axes.

    Fallback and axial decisions are deliberately returned unchanged.
    """

    if not np.isfinite(superior_mm) or superior_mm < 0.0:
        raise ValueError("superior_mm must be finite and non-negative")
    joint_row, joint_column = requested_center_pixel(dataset, decision)
    if superior_mm == 0.0 or decision.mode != "joint_line":
        return joint_row, joint_column
    if decision.plane == "Coronal":
        horizontal_axis = 0
    elif decision.plane == "Sagittal":
        horizontal_axis = 1
    else:
        return joint_row, joint_column

    orientation = _orientation(dataset)
    spacing = pixel_spacing(dataset)
    joint_point = pixel_patient_position(dataset, joint_row, joint_column)
    if orientation is None or spacing is None or joint_point is None:
        raise ValueError("invalid DICOM geometry for superior joint offset")
    column_direction, row_direction, _ = orientation
    row_spacing, column_spacing = spacing
    affine = np.asarray(
        [
            [
                row_direction[horizontal_axis] * row_spacing,
                column_direction[horizontal_axis] * column_spacing,
            ],
            [
                row_direction[2] * row_spacing,
                column_direction[2] * column_spacing,
            ],
        ],
        dtype=float,
    )
    determinant = float(np.linalg.det(affine))
    if not np.isfinite(determinant) or abs(determinant) < 1e-8:
        raise ValueError("superior joint offset affine is singular")
    delta_row, delta_column = np.linalg.solve(
        affine, np.asarray([0.0, float(superior_mm)], dtype=float)
    )
    target_row = float(joint_row + delta_row)
    target_column = float(joint_column + delta_column)
    if not np.isfinite([target_row, target_column]).all():
        raise ValueError("superior joint offset produced a non-finite centre")

    target_point = pixel_patient_position(dataset, target_row, target_column)
    if target_point is None:
        raise ValueError("cannot map superior joint target back to patient coordinates")
    if not np.isclose(
        float(target_point[horizontal_axis] - joint_point[horizontal_axis]),
        0.0,
        atol=1e-5,
    ) or not np.isclose(
        float(target_point[2] - joint_point[2]), float(superior_mm), atol=1e-5
    ):
        raise ValueError("superior joint target failed the patient-coordinate check")
    return target_row, target_column


def _clamped_fixed_size_start(center: float, size: int, extent: int) -> int:
    start = int(np.floor(center - (size - 1.0) / 2.0))
    return int(np.clip(start, 0, max(0, extent - size)))


def crop_box_for_center(
    dataset: Any,
    *,
    center_row: float,
    center_column: float,
    crop_mm: float | None,
    profile: CropProfile = "local_rectangular_mm",
) -> CropBox:
    """Build a bounded crop with explicit zero-shift preprocessing semantics."""

    rows, columns = int(dataset.Rows), int(dataset.Columns)
    spacing = pixel_spacing(dataset)
    geometric_row = (rows - 1.0) / 2.0
    geometric_column = (columns - 1.0) / 2.0
    if crop_mm is None or crop_mm <= 0 or spacing is None:
        return CropBox(
            0,
            0,
            rows,
            columns,
            center_row,
            center_column,
            geometric_row,
            geometric_column,
            False,
            profile,
        )

    row_spacing, column_spacing = spacing
    if profile == "local_rectangular_mm":
        height = min(rows, max(1, int(round(crop_mm / row_spacing))))
        width = min(columns, max(1, int(round(crop_mm / column_spacing))))
        top = _clamped_fixed_size_start(center_row, height, rows)
        left = _clamped_fixed_size_start(center_column, width, columns)
        bottom, right = top + height, left + width
    elif profile == "dino_v2_square":
        want = int(round(crop_mm / row_spacing))
        if not (16 < want < min(rows, columns)):
            top, left, bottom, right = 0, 0, rows, columns
        else:
            size = 2 * (want // 2)
            top = _clamped_fixed_size_start(center_row, size, rows)
            left = _clamped_fixed_size_start(center_column, size, columns)
            bottom, right = top + size, left + size
    elif profile == "dino_v3_clipped_square":
        half = int(round(crop_mm / row_spacing / 2.0))
        centre_row_int = int(np.floor(center_row + 0.5))
        centre_column_int = int(np.floor(center_column + 0.5))
        top = max(0, centre_row_int - half)
        bottom = min(rows, centre_row_int + half)
        left = max(0, centre_column_int - half)
        right = min(columns, centre_column_int + half)
    elif profile == "coatnet_square":
        size = min(max(1, int(round(crop_mm / max(row_spacing, 1e-3)))), rows, columns)
        top = _clamped_fixed_size_start(center_row, size, rows)
        left = _clamped_fixed_size_start(center_column, size, columns)
        bottom, right = top + size, left + size
    else:  # pragma: no cover - protected by Literal in typed callers.
        raise ValueError(f"Unknown crop profile: {profile}")

    actual_row = (top + bottom - 1.0) / 2.0
    actual_column = (left + right - 1.0) / 2.0
    was_clamped = not (
        np.isclose(actual_row, center_row, atol=1.0)
        and np.isclose(actual_column, center_column, atol=1.0)
    )
    return CropBox(
        top=int(top),
        left=int(left),
        bottom=int(bottom),
        right=int(right),
        requested_center_row=float(center_row),
        requested_center_column=float(center_column),
        actual_center_row=float(actual_row),
        actual_center_column=float(actual_column),
        was_clamped=bool(was_clamped),
        profile=profile,
    )


def crop_box_for_decision(
    dataset: Any,
    decision: SeriesCenteringDecision,
    *,
    crop_mm: float | None,
    profile: CropProfile = "local_rectangular_mm",
    superior_to_joint_mm: float = 0.0,
) -> CropBox:
    center_row, center_column = superior_to_joint_center_pixel(
        dataset,
        decision,
        superior_mm=superior_to_joint_mm,
    )
    return crop_box_for_center(
        dataset,
        center_row=center_row,
        center_column=center_column,
        crop_mm=crop_mm,
        profile=profile,
    )


def crop_image_for_decision(
    image: np.ndarray,
    dataset: Any,
    decision: SeriesCenteringDecision,
    *,
    crop_mm: float | None,
    profile: CropProfile = "local_rectangular_mm",
    superior_to_joint_mm: float = 0.0,
) -> np.ndarray:
    box = crop_box_for_decision(
        dataset,
        decision,
        crop_mm=crop_mm,
        profile=profile,
        superior_to_joint_mm=superior_to_joint_mm,
    )
    return image[box.top : box.bottom, box.left : box.right]
