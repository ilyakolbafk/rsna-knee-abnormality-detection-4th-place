"""Fail-closed patient-coordinate helpers for experimental anatomy ROIs.

These helpers deliberately stop at geometry.  A study-level diagnosis is not a
spatial annotation, and DICOM geometry alone cannot identify a meniscus contour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence

import numpy as np


KneeSide = Literal["L", "R"]
Plane = Literal["Sagittal", "Coronal", "Axial"]


def _float_vector(value: Any, expected: int) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if len(vector) < expected or not np.isfinite(vector[:expected]).all():
        return None
    return vector[:expected]


def _orientation_vectors(dataset: Any) -> tuple[np.ndarray, np.ndarray] | None:
    orientation = _float_vector(
        getattr(dataset, "ImageOrientationPatient", None), 6
    )
    if orientation is None:
        return None
    first = orientation[:3]
    second = orientation[3:6]
    if not np.isclose(np.linalg.norm(first), 1.0, atol=1e-3):
        return None
    if not np.isclose(np.linalg.norm(second), 1.0, atol=1e-3):
        return None
    if not np.isclose(float(np.dot(first, second)), 0.0, atol=1e-3):
        return None
    return first, second


def _side_value(value: Any) -> KneeSide | None:
    text = str(value or "").strip().upper()
    if text.startswith("L"):
        return "L"
    if text.startswith("R"):
        return "R"
    return None


def tagged_knee_side(dataset: Any) -> KneeSide | None:
    """Return a consistent DICOM laterality tag for one dataset, if present."""

    values = {
        side
        for side in (
            _side_value(getattr(dataset, "Laterality", None)),
            _side_value(getattr(dataset, "ImageLaterality", None)),
        )
        if side is not None
    }
    return next(iter(values)) if len(values) == 1 else None


def pixel_patient_position(dataset: Any, row: float, column: float) -> np.ndarray | None:
    """Map a DICOM pixel coordinate to the patient LPS coordinate system.

    In DICOM patient coordinates, +X points to the patient's left, +Y to the
    posterior, and +Z to the head.  The first IOP triplet advances with image
    columns; the second advances with image rows.
    """

    directions = _orientation_vectors(dataset)
    origin = _float_vector(getattr(dataset, "ImagePositionPatient", None), 3)
    spacing = _float_vector(getattr(dataset, "PixelSpacing", None), 2)
    if directions is None or origin is None or spacing is None:
        return None
    if np.any(spacing <= 0):
        return None
    first, second = directions
    return (
        origin
        + first * spacing[1] * float(column)
        + second * spacing[0] * float(row)
    )


def image_center_patient(dataset: Any) -> np.ndarray | None:
    """Return the physical centre of a DICOM image in patient coordinates."""

    try:
        rows = float(getattr(dataset, "Rows"))
        columns = float(getattr(dataset, "Columns"))
    except (AttributeError, TypeError, ValueError):
        return None
    if rows <= 0 or columns <= 0:
        return None
    # Pixel indices run from 0 to N - 1, so the physical centre lies halfway
    # between the first and last pixel centres.
    return pixel_patient_position(dataset, (rows - 1.0) / 2.0, (columns - 1.0) / 2.0)


def anatomical_plane(dataset: Any, *, minimum_alignment: float = 0.80) -> Plane | None:
    """Infer the closest orthogonal plane from DICOM orientation.

    Oblique series that do not align sufficiently with one patient axis are
    rejected instead of being silently assigned to a plane.
    """

    directions = _orientation_vectors(dataset)
    if directions is None:
        return None
    normal = np.cross(*directions)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-8:
        return None
    aligned = np.abs(normal / norm)
    axis = int(np.argmax(aligned))
    if float(aligned[axis]) < minimum_alignment:
        return None
    return ("Sagittal", "Coronal", "Axial")[axis]


@dataclass(frozen=True)
class LateralityDecision:
    side: KneeSide | None
    status: Literal["resolved", "unresolved", "conflict"]
    source: Literal["tag+geometry", "tag", "geometry", "none", "conflict"]
    tag_side: KneeSide | None
    geometry_side: KneeSide | None
    median_center_x_mm: float | None


def infer_knee_laterality(
    datasets: Iterable[Any],
    *,
    minimum_center_offset_mm: float = 20.0,
) -> LateralityDecision:
    """Resolve knee side from tags and image centres, failing on disagreement."""

    tag_values: set[KneeSide] = set()
    center_x: list[float] = []
    for dataset in datasets:
        # Keep every valid tag value.  Collapsing an intra-instance
        # Laterality/ImageLaterality disagreement to None would allow geometry
        # to override a real metadata conflict, which is not fail-closed.
        tag_values.update(
            side
            for side in (
                _side_value(getattr(dataset, "Laterality", None)),
                _side_value(getattr(dataset, "ImageLaterality", None)),
            )
            if side is not None
        )
        center = image_center_patient(dataset)
        if center is not None:
            center_x.append(float(center[0]))

    tag_side: KneeSide | None = (
        next(iter(tag_values)) if len(tag_values) == 1 else None
    )
    median_x = float(np.median(center_x)) if center_x else None
    geometry_side: KneeSide | None = None
    if median_x is not None and abs(median_x) >= minimum_center_offset_mm:
        geometry_side = "L" if median_x > 0 else "R"

    if len(tag_values) > 1:
        return LateralityDecision(
            side=None,
            status="conflict",
            source="conflict",
            tag_side=None,
            geometry_side=geometry_side,
            median_center_x_mm=median_x,
        )
    if tag_side is not None and geometry_side is not None:
        if tag_side != geometry_side:
            return LateralityDecision(
                side=None,
                status="conflict",
                source="conflict",
                tag_side=tag_side,
                geometry_side=geometry_side,
                median_center_x_mm=median_x,
            )
        return LateralityDecision(
            side=tag_side,
            status="resolved",
            source="tag+geometry",
            tag_side=tag_side,
            geometry_side=geometry_side,
            median_center_x_mm=median_x,
        )
    if tag_side is not None:
        return LateralityDecision(
            side=tag_side,
            status="resolved",
            source="tag",
            tag_side=tag_side,
            geometry_side=None,
            median_center_x_mm=median_x,
        )
    if geometry_side is not None:
        # The sign of patient X is a useful corpus-specific hint, but DICOM does
        # not define X=0 as the patient's anatomical midline.  Keep the hint for
        # QA and ordering after external confirmation; do not accept it as
        # laterality on its own.
        return LateralityDecision(
            side=None,
            status="unresolved",
            source="geometry",
            tag_side=None,
            geometry_side=geometry_side,
            median_center_x_mm=median_x,
        )
    return LateralityDecision(
        side=None,
        status="unresolved",
        source="none",
        tag_side=None,
        geometry_side=None,
        median_center_x_mm=median_x,
    )


def lateral_coordinate(dataset: Any, side: KneeSide) -> float | None:
    """Return a coordinate that increases toward the knee's lateral side."""

    center = image_center_patient(dataset)
    if center is None:
        return None
    return float(center[0] if side == "L" else -center[0])


def normalised_lateral_positions(
    datasets: Sequence[Any], side: KneeSide
) -> np.ndarray:
    """Map sagittal slice centres to [0, 1], medial-to-lateral.

    All inputs must have valid patient geometry.  A degenerate or single-position
    series is rejected; the caller should then use the global fallback.
    """

    coordinates = [lateral_coordinate(dataset, side) for dataset in datasets]
    if any(value is None for value in coordinates):
        raise ValueError("Every slice requires valid DICOM patient geometry")
    values = np.asarray(coordinates, dtype=float)
    span = float(values.max() - values.min())
    if not np.isfinite(span) or span < 1e-6:
        raise ValueError("Sagittal series has no usable lateral coordinate span")
    return (values - values.min()) / span


@dataclass(frozen=True)
class CropRoutingDecision:
    mode: Literal["lateral_roi", "global_fallback"]
    reason: Literal[
        "accepted_locator",
        "laterality_not_resolved",
        "locator_missing",
        "locator_below_threshold",
    ]


def route_lateral_crop(
    laterality: LateralityDecision,
    *,
    locator_confidence: float | None,
    minimum_locator_confidence: float = 0.80,
) -> CropRoutingDecision:
    """Fail closed to the existing global crop unless both gates pass.

    Confidence is a placeholder interface for an independently validated anatomy
    locator; this helper does not manufacture confidence from DICOM geometry.
    """

    if laterality.side is None or laterality.status != "resolved":
        return CropRoutingDecision("global_fallback", "laterality_not_resolved")
    if locator_confidence is None or not np.isfinite(locator_confidence):
        return CropRoutingDecision("global_fallback", "locator_missing")
    if float(locator_confidence) < minimum_locator_confidence:
        return CropRoutingDecision("global_fallback", "locator_below_threshold")
    return CropRoutingDecision("lateral_roi", "accepted_locator")
