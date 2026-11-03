"""Conservative, competition-data-only tibiofemoral joint-line locator.

The locator does not segment a meniscus.  It searches for paired, predominantly
horizontal bone/cartilage interfaces on several sagittal slices and only returns
a candidate when those slices agree in patient coordinates.  Callers must keep
the existing global crop as the fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from scipy import ndimage

from .anatomy_roi import KneeSide, normalised_lateral_positions
from .dicom import dicom_pixels, dicom_sort_key


@dataclass(frozen=True)
class CanonicalSagittal:
    image: np.ndarray
    valid: np.ndarray
    y_mm: np.ndarray
    z_mm: np.ndarray
    spacing_mm: float


@dataclass(frozen=True)
class SliceJointLineCandidate:
    index: int
    lateral_position: float
    y_mm: float
    z_mm: float
    pair_separation_mm: float
    support_mm: float
    score: float
    prominence: float
    physical_support: float
    boundary_distance_mm: float


@dataclass(frozen=True)
class JointLineDecision:
    status: Literal["candidate", "fallback"]
    reason: Literal[
        "accepted_multislice_candidate",
        "invalid_geometry",
        "too_few_lateral_slices",
        "too_few_valid_candidates",
        "no_dominant_cluster",
        "insufficient_lateral_span",
        "unstable_z",
        "unstable_y",
        "below_confidence_threshold",
    ]
    confidence: float
    y_mm: float | None
    z_mm: float | None
    selected_indices: tuple[int, ...]
    cluster_indices: tuple[int, ...]
    candidates: tuple[SliceJointLineCandidate, ...]
    mad_y_mm: float | None
    mad_z_mm: float | None
    lateral_span_mm: float | None


def _orientation(dataset: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        values = np.asarray(dataset.ImageOrientationPatient, dtype=float).reshape(6)
    except (AttributeError, TypeError, ValueError):
        return None
    column_direction = values[:3]
    row_direction = values[3:]
    normal = np.cross(column_direction, row_direction)
    if not np.isfinite(values).all() or np.linalg.norm(normal) < 1e-8:
        return None
    return column_direction, row_direction, normal / np.linalg.norm(normal)


def canonicalise_sagittal(
    dataset: Any,
    *,
    spacing_mm: float = 1.0,
    minimum_yz_determinant: float = 0.60,
) -> CanonicalSagittal:
    """Resample one sagittal slice onto a patient-Y/Z grid.

    Columns increase toward patient posterior (+Y); rows increase toward patient
    inferior (-Z).  This removes scanner display flips before edge detection.
    """

    orientation = _orientation(dataset)
    if orientation is None or spacing_mm <= 0:
        raise ValueError("invalid sagittal orientation")
    column_direction, row_direction, normal = orientation
    if abs(float(normal[0])) < 0.80:
        raise ValueError("slice is not sufficiently sagittal")
    try:
        origin = np.asarray(dataset.ImagePositionPatient, dtype=float).reshape(3)
        pixel_spacing = np.asarray(dataset.PixelSpacing, dtype=float).reshape(2)
        rows = int(dataset.Rows)
        columns = int(dataset.Columns)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("missing sagittal geometry") from error
    if (
        rows <= 0
        or columns <= 0
        or np.any(pixel_spacing <= 0)
        or not np.isfinite(origin).all()
    ):
        raise ValueError("invalid sagittal geometry")

    # Maps [native column, native row] offsets to patient [Y, Z] offsets.
    yz_matrix = np.asarray(
        [
            [column_direction[1] * pixel_spacing[1], row_direction[1] * pixel_spacing[0]],
            [column_direction[2] * pixel_spacing[1], row_direction[2] * pixel_spacing[0]],
        ],
        dtype=float,
    )
    determinant = float(np.linalg.det(yz_matrix))
    scale = float(np.prod(pixel_spacing))
    if not np.isfinite(determinant) or abs(determinant) < minimum_yz_determinant * scale:
        raise ValueError("sagittal plane cannot be mapped stably to patient Y/Z")
    inverse = np.linalg.inv(yz_matrix)

    corners = np.asarray(
        [
            [0.0, 0.0],
            [columns - 1.0, 0.0],
            [0.0, rows - 1.0],
            [columns - 1.0, rows - 1.0],
        ]
    )
    corner_yz = origin[[1, 2]][None, :] + corners @ yz_matrix.T
    y_values = np.arange(
        float(corner_yz[:, 0].min()),
        float(corner_yz[:, 0].max()) + spacing_mm * 0.5,
        spacing_mm,
    )
    z_values = np.arange(
        float(corner_yz[:, 1].max()),
        float(corner_yz[:, 1].min()) - spacing_mm * 0.5,
        -spacing_mm,
    )
    if len(y_values) < 32 or len(z_values) < 32:
        raise ValueError("canonical sagittal field of view is too small")

    grid_z, grid_y = np.meshgrid(z_values, y_values, indexing="ij")
    yz_offsets = np.stack(
        [grid_y - origin[1], grid_z - origin[2]], axis=-1
    )
    native = yz_offsets @ inverse.T
    native_columns = native[..., 0]
    native_rows = native[..., 1]
    valid = (
        (native_rows >= 0)
        & (native_rows <= rows - 1)
        & (native_columns >= 0)
        & (native_columns <= columns - 1)
    )
    pixels = dicom_pixels(dataset)
    sampled = ndimage.map_coordinates(
        pixels,
        [native_rows, native_columns],
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    ).astype(np.float32)
    return CanonicalSagittal(
        image=sampled,
        valid=valid & np.isfinite(sampled),
        y_mm=y_values,
        z_mm=z_values,
        spacing_mm=float(spacing_mm),
    )


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    threshold = 0.5 * float(sorted_weights.sum())
    return float(sorted_values[np.searchsorted(np.cumsum(sorted_weights), threshold)])


def _robust_scale(image: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = image[valid]
    if values.size < 100:
        raise ValueError("too few valid sagittal pixels")
    low, high = np.percentile(values, [1.0, 99.0])
    if not np.isfinite(high - low) or high <= low:
        raise ValueError("flat sagittal image")
    return np.clip((image - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _largest_tissue_component(scaled: np.ndarray, valid: np.ndarray) -> np.ndarray:
    # MR backgrounds are usually constant black.  Keeping the largest component
    # avoids treating annotations and isolated wrap-around artefacts as anatomy.
    mask = valid & np.isfinite(scaled) & (scaled > 0.025)
    mask = ndimage.binary_closing(mask, iterations=3)
    mask = ndimage.binary_fill_holes(mask)
    labels, count = ndimage.label(mask)
    if count == 0:
        raise ValueError("no tissue component")
    sizes = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
    label = int(np.argmax(sizes)) + 1
    component = labels == label
    if int(component.sum()) < 0.05 * int(valid.sum()):
        raise ValueError("tissue component is too small")
    return component


def locate_joint_line_on_slice(
    dataset: Any,
    *,
    index: int,
    lateral_position: float,
    canonical_spacing_mm: float = 1.0,
) -> SliceJointLineCandidate:
    """Find a paired horizontal-interface candidate on one sagittal slice."""

    canonical = canonicalise_sagittal(dataset, spacing_mm=canonical_spacing_mm)
    scaled = _robust_scale(canonical.image, canonical.valid)
    tissue = _largest_tissue_component(scaled, canonical.valid)
    tissue_rows, tissue_columns = np.where(tissue)
    row_min, row_max = int(tissue_rows.min()), int(tissue_rows.max())
    col_min, col_max = int(tissue_columns.min()), int(tissue_columns.max())
    tissue_height = row_max - row_min + 1
    tissue_width = col_max - col_min + 1
    if tissue_height < 60 or tissue_width < 60:
        raise ValueError("tissue field of view is too small")

    valid_float = canonical.valid.astype(np.float32)
    filled = np.where(canonical.valid, scaled, 0.0)
    numerator = ndimage.gaussian_filter(filled, sigma=1.0 / canonical.spacing_mm)
    denominator = ndimage.gaussian_filter(valid_float, sigma=1.0 / canonical.spacing_mm)
    smooth = numerator / np.maximum(denominator, 1e-4)
    dz, dy = np.gradient(smooth, canonical.spacing_mm, canonical.spacing_mm)
    horizontal = np.abs(dz) > 0.4 * np.abs(dy)
    gradient_scale = float(np.percentile(np.abs(dz)[tissue], 95.0))
    if not np.isfinite(gradient_scale) or gradient_scale <= 1e-6:
        raise ValueError("insufficient sagittal edge energy")
    h_positive = np.where(horizontal & tissue, np.maximum(dz, 0.0), 0.0) / gradient_scale
    h_negative = np.where(horizontal & tissue, np.maximum(-dz, 0.0), 0.0) / gradient_scale
    h_positive = np.clip(h_positive, 0.0, 3.0)
    h_negative = np.clip(h_negative, 0.0, 3.0)

    search_row_min = row_min + int(round(0.30 * tissue_height))
    search_row_max = row_min + int(round(0.75 * tissue_height))
    search_col_min = col_min + int(round(0.10 * tissue_width))
    search_col_max = col_min + int(round(0.90 * tissue_width))
    if search_row_max - search_row_min < 24 or search_col_max - search_col_min < 30:
        raise ValueError("joint-line search window is too small")

    separations = range(
        max(3, int(round(3.0 / canonical.spacing_mm))),
        max(4, int(round(16.0 / canonical.spacing_mm))) + 1,
    )
    support_widths = [
        max(5, int(round(width_mm / canonical.spacing_mm)))
        for width_mm in (20.0, 30.0, 40.0, 50.0, 60.0, 70.0)
    ]
    row_scores = np.full(search_row_max - search_row_min + 1, np.nan, dtype=float)
    best: tuple[float, int, int, int, np.ndarray] | None = None
    for row_offset, centre_row in enumerate(range(search_row_min, search_row_max + 1)):
        best_row_score = -np.inf
        for separation in separations:
            upper_offset = separation // 2
            lower_offset = separation - upper_offset
            upper = centre_row - upper_offset
            lower = centre_row + lower_offset
            if upper < 0 or lower >= smooth.shape[0]:
                continue
            pair_a = np.sqrt(h_positive[upper] * h_negative[lower])
            pair_b = np.sqrt(h_negative[upper] * h_positive[lower])
            pair = np.maximum(pair_a, pair_b)
            pair = ndimage.gaussian_filter1d(pair, sigma=2.0 / canonical.spacing_mm)
            pair[:search_col_min] = 0.0
            pair[search_col_max + 1 :] = 0.0
            for support_width in support_widths:
                if support_width > search_col_max - search_col_min + 1:
                    continue
                window_score = ndimage.uniform_filter1d(
                    pair, size=support_width, mode="constant"
                )
                centre_column = int(
                    np.argmax(window_score[search_col_min : search_col_max + 1])
                    + search_col_min
                )
                score = float(window_score[centre_column])
                if score > best_row_score:
                    best_row_score = score
                if best is None or score > best[0]:
                    best = (score, centre_row, centre_column, separation, pair.copy())
        row_scores[row_offset] = best_row_score
    if best is None or not np.isfinite(best[0]):
        raise ValueError("no paired-edge candidate")

    score, centre_row, centre_column, separation, pair = best
    finite_scores = row_scores[np.isfinite(row_scores)]
    best_offset = centre_row - search_row_min
    competing_scores = row_scores.copy()
    exclusion = max(1, int(round(8.0 / canonical.spacing_mm)))
    competing_scores[
        max(0, best_offset - exclusion) : min(len(competing_scores), best_offset + exclusion + 1)
    ] = np.nan
    second_peak = float(np.nanmax(competing_scores)) if np.isfinite(competing_scores).any() else 0.0
    peak_ratio = score / max(second_peak, 1e-6)
    prominence = float(np.clip((peak_ratio - 1.15) / 0.85, 0.0, 1.0))

    local_threshold = max(float(np.percentile(pair[search_col_min : search_col_max + 1], 75.0)), 0.05)
    support_mask = pair >= local_threshold
    support_labels, support_count = ndimage.label(support_mask)
    support_pixels = 0
    if support_count:
        chosen_label = int(support_labels[centre_column])
        if chosen_label:
            support_pixels = int(np.sum(support_labels == chosen_label))
        else:
            sizes = ndimage.sum(
                support_mask, support_labels, index=np.arange(1, support_count + 1)
            )
            support_pixels = int(np.max(sizes))
    support_mm = float(support_pixels * canonical.spacing_mm)
    physical_support = float(np.clip((support_mm - 10.0) / 30.0, 0.0, 1.0))
    boundary_distance_mm = float(
        min(
            centre_row - search_row_min,
            search_row_max - centre_row,
        )
        * canonical.spacing_mm
    )
    return SliceJointLineCandidate(
        index=int(index),
        lateral_position=float(lateral_position),
        y_mm=float(canonical.y_mm[centre_column]),
        z_mm=float(canonical.z_mm[centre_row]),
        pair_separation_mm=float(separation * canonical.spacing_mm),
        support_mm=support_mm,
        score=float(score),
        prominence=prominence,
        physical_support=physical_support,
        boundary_distance_mm=boundary_distance_mm,
    )


def _slice_plane_coordinate(dataset: Any) -> float:
    orientation = _orientation(dataset)
    if orientation is None:
        raise ValueError("invalid slice orientation")
    normal = orientation[2]
    origin = np.asarray(dataset.ImagePositionPatient, dtype=float).reshape(3)
    return float(np.dot(origin, normal))


def _fallback(
    reason: JointLineDecision.__annotations__["reason"],
    *,
    candidates: Sequence[SliceJointLineCandidate] = (),
    selected_indices: Sequence[int] = (),
    confidence: float = 0.0,
    mad_y_mm: float | None = None,
    mad_z_mm: float | None = None,
    lateral_span_mm: float | None = None,
) -> JointLineDecision:
    return JointLineDecision(
        status="fallback",
        reason=reason,
        confidence=float(confidence),
        y_mm=None,
        z_mm=None,
        selected_indices=tuple(int(value) for value in selected_indices),
        cluster_indices=(),
        candidates=tuple(candidates),
        mad_y_mm=mad_y_mm,
        mad_z_mm=mad_z_mm,
        lateral_span_mm=lateral_span_mm,
    )


def locate_joint_line_multislice(
    datasets: Sequence[Any],
    side: KneeSide,
    *,
    lateral_low: float = 0.50,
    lateral_high: float = 0.85,
    maximum_slices: int = 9,
    minimum_confidence: float = 0.80,
) -> JointLineDecision:
    """Locate a joint-line centre using only one sagittal DICOM series."""

    if len(datasets) < 3:
        return _fallback("too_few_lateral_slices")
    frame_uids = {
        str(getattr(dataset, "FrameOfReferenceUID", "") or "")
        for dataset in datasets
    }
    if len(frame_uids) != 1 or "" in frame_uids:
        return _fallback("invalid_geometry")
    try:
        positions = normalised_lateral_positions(datasets, side)
        plane_coordinates = np.asarray([_slice_plane_coordinate(ds) for ds in datasets])
    except (TypeError, ValueError):
        return _fallback("invalid_geometry")
    eligible = np.flatnonzero((positions >= lateral_low) & (positions <= lateral_high))
    if len(eligible) < 3:
        return _fallback("too_few_lateral_slices")
    if len(eligible) > maximum_slices:
        chosen = np.rint(np.linspace(0, len(eligible) - 1, maximum_slices)).astype(int)
        eligible = eligible[chosen]
    selected_indices = tuple(int(value) for value in eligible)

    candidates: list[SliceJointLineCandidate] = []
    for index in eligible:
        try:
            candidate = locate_joint_line_on_slice(
                datasets[int(index)],
                index=int(index),
                lateral_position=float(positions[int(index)]),
            )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            continue
        if (
            candidate.boundary_distance_mm >= 10.0
            and 3.0 <= candidate.pair_separation_mm <= 16.0
            and candidate.support_mm >= 20.0
        ):
            candidates.append(candidate)
    if len(candidates) < 3:
        return _fallback(
            "too_few_valid_candidates",
            candidates=candidates,
            selected_indices=selected_indices,
        )

    y = np.asarray([candidate.y_mm for candidate in candidates], dtype=float)
    z = np.asarray([candidate.z_mm for candidate in candidates], dtype=float)
    weights = np.asarray(
        [max(candidate.score, 1e-6) for candidate in candidates], dtype=float
    )
    neighbourhoods = []
    for index in range(len(candidates)):
        members = np.flatnonzero((np.abs(y - y[index]) <= 12.0) & (np.abs(z - z[index]) <= 8.0))
        neighbourhoods.append(members)
    cluster = max(
        neighbourhoods,
        key=lambda members: (len(members), float(weights[members].sum())),
    )
    if len(cluster) < 3 or len(cluster) / len(candidates) < 0.50:
        return _fallback(
            "no_dominant_cluster",
            candidates=candidates,
            selected_indices=selected_indices,
        )

    cluster_y = y[cluster]
    cluster_z = z[cluster]
    cluster_weights = weights[cluster]
    centre_y = _weighted_median(cluster_y, cluster_weights)
    centre_z = _weighted_median(cluster_z, cluster_weights)
    mad_y = float(np.median(np.abs(cluster_y - centre_y)))
    mad_z = float(np.median(np.abs(cluster_z - centre_z)))
    cluster_plane = plane_coordinates[[candidates[int(index)].index for index in cluster]]
    lateral_span = float(cluster_plane.max() - cluster_plane.min())
    if lateral_span < 6.0:
        return _fallback(
            "insufficient_lateral_span",
            candidates=candidates,
            selected_indices=selected_indices,
            mad_y_mm=mad_y,
            mad_z_mm=mad_z,
            lateral_span_mm=lateral_span,
        )
    if mad_z > 4.0:
        return _fallback(
            "unstable_z",
            candidates=candidates,
            selected_indices=selected_indices,
            mad_y_mm=mad_y,
            mad_z_mm=mad_z,
            lateral_span_mm=lateral_span,
        )
    if mad_y > 12.0:
        return _fallback(
            "unstable_y",
            candidates=candidates,
            selected_indices=selected_indices,
            mad_y_mm=mad_y,
            mad_z_mm=mad_z,
            lateral_span_mm=lateral_span,
        )

    cluster_candidates = [candidates[int(index)] for index in cluster]
    peak_prominence = float(np.median([value.prominence for value in cluster_candidates]))
    slice_consensus = float(
        np.clip(len(cluster) / max(len(candidates), 1), 0.0, 1.0)
        * np.clip(len(cluster) / 4.0, 0.0, 1.0)
    )
    z_stability = float(np.clip(1.0 - mad_z / 4.0, 0.0, 1.0))
    y_stability = float(np.clip(1.0 - mad_y / 12.0, 0.0, 1.0))
    support = float(np.median([value.physical_support for value in cluster_candidates]))
    confidence = (
        0.30 * peak_prominence
        + 0.25 * slice_consensus
        + 0.20 * z_stability
        + 0.10 * y_stability
        + 0.15 * support
    )
    if confidence < minimum_confidence:
        return _fallback(
            "below_confidence_threshold",
            candidates=candidates,
            selected_indices=selected_indices,
            confidence=confidence,
            mad_y_mm=mad_y,
            mad_z_mm=mad_z,
            lateral_span_mm=lateral_span,
        )
    return JointLineDecision(
        status="candidate",
        reason="accepted_multislice_candidate",
        confidence=float(confidence),
        y_mm=centre_y,
        z_mm=centre_z,
        selected_indices=selected_indices,
        cluster_indices=tuple(candidates[int(index)].index for index in cluster),
        candidates=tuple(candidates),
        mad_y_mm=mad_y,
        mad_z_mm=mad_z,
        lateral_span_mm=lateral_span,
    )


def load_sorted_dicom_series(series_dir: str | Path) -> tuple[list[Path], list[Any]]:
    """Read and sort a local DICOM series for the experimental locator."""

    import pydicom

    directory = Path(series_dir)
    rows: list[tuple[Path, Any]] = []
    for path in sorted(value for value in directory.iterdir() if value.is_file()):
        try:
            rows.append((path, pydicom.dcmread(path, force=True)))
        except Exception:
            continue
    rows.sort(key=lambda pair: dicom_sort_key(pair[1], pair[0].name))
    if not rows:
        raise ValueError(f"No readable DICOMs in {directory}")
    return [path for path, _ in rows], [dataset for _, dataset in rows]


def patient_yz_to_pixel(dataset: Any, y_mm: float, z_mm: float) -> tuple[float, float]:
    """Project patient Y/Z onto a sagittal image, returning native row/column."""

    orientation = _orientation(dataset)
    if orientation is None:
        raise ValueError("invalid slice orientation")
    column_direction, row_direction, _ = orientation
    origin = np.asarray(dataset.ImagePositionPatient, dtype=float).reshape(3)
    spacing = np.asarray(dataset.PixelSpacing, dtype=float).reshape(2)
    matrix = np.asarray(
        [
            [column_direction[1] * spacing[1], row_direction[1] * spacing[0]],
            [column_direction[2] * spacing[1], row_direction[2] * spacing[0]],
        ]
    )
    column, row = np.linalg.solve(matrix, np.asarray([y_mm, z_mm]) - origin[[1, 2]])
    return float(row), float(column)


def centred_crop_box_mm(
    dataset: Any,
    *,
    centre_row: float,
    centre_column: float,
    crop_mm: float,
) -> tuple[float, float, float, float]:
    """Return a native-pixel square crop box as left, top, right, bottom."""

    spacing = np.asarray(dataset.PixelSpacing, dtype=float).reshape(2)
    half_rows = crop_mm / spacing[0] / 2.0
    half_columns = crop_mm / spacing[1] / 2.0
    return (
        float(centre_column - half_columns),
        float(centre_row - half_rows),
        float(centre_column + half_columns),
        float(centre_row + half_rows),
    )
