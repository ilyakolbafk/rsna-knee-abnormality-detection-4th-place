"""Deterministic study-level coronal-first centering manifests.

The manifest is deliberately separate from the pixel cache.  It records the
per-series decision that must be reused for every crop in a study, lets cache
identity include the exact decision bytes, and prevents a missing centered
cache entry from silently falling back to a different policy.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .anatomy_roi import anatomical_plane, infer_knee_laterality
from .constants import ID_COLUMN, SERIES_COLUMN
from .dicom import dicom_sort_key, discover_dicom_files
from .study_centering import (
    CoronalJointHeight,
    CoronalSliceCandidate,
    SeriesCenteringDecision,
    coronal_joint_height,
    locate_series_center,
    sagittal_center_from_absolute_z,
    same_knee_fov_overlap,
)


CENTERING_MANIFEST_FORMAT = "rsna-knee-coronal-first-centering-v1"
CORONAL_FIRST_POLICY = {
    "name": "coronal_first_v1",
    "coronal": "safe_series_locator",
    "sagittal_priority": [
        "stable_same_knee_coronal_absolute_z",
        "safe_native_sagittal_locator",
        "geometric_fallback",
    ],
    "axial": "geometric_fallback",
    "fixed_prior_mm": None,
    "coronal_consensus_mad_max_mm": 4.0,
    "coronal_consensus_range_max_mm": 10.0,
    "maximum_shift_mm": 35.0,
    "fov_tolerance_mm": 0.0,
    "allow_raw_lps_after_geometry_guards": True,
}


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def policy_sha256() -> str:
    return _canonical_sha256(CORONAL_FIRST_POLICY)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def decision_to_dict(decision: SeriesCenteringDecision) -> dict[str, Any]:
    return asdict(decision)


def decision_from_dict(payload: Mapping[str, Any]) -> SeriesCenteringDecision:
    candidates = tuple(
        CoronalSliceCandidate(
            index=int(value["index"]),
            native_row=float(value["native_row"]),
            native_column=float(value["native_column"]),
            inferior_shift_mm=float(value["inferior_shift_mm"]),
            peak_strength=float(value["peak_strength"]),
            prominence=float(value["prominence"]),
            bilateral_delta_mm=float(value["bilateral_delta_mm"]),
            confidence=float(value["confidence"]),
        )
        for value in payload.get("candidates", ())
    )
    offset = payload.get("patient_offset_mm")
    return SeriesCenteringDecision(
        mode=str(payload["mode"]),
        reason=str(payload["reason"]),
        plane=(None if payload.get("plane") is None else str(payload["plane"])),
        confidence=float(payload.get("confidence", 0.0)),
        inferior_shift_mm=(
            None
            if payload.get("inferior_shift_mm") is None
            else float(payload["inferior_shift_mm"])
        ),
        patient_offset_mm=(
            None if offset is None else tuple(float(value) for value in offset)
        ),
        selected_indices=tuple(int(value) for value in payload.get("selected_indices", ())),
        accepted_indices=tuple(int(value) for value in payload.get("accepted_indices", ())),
        candidates=candidates,
        median_absolute_deviation_mm=(
            None
            if payload.get("median_absolute_deviation_mm") is None
            else float(payload["median_absolute_deviation_mm"])
        ),
    )


@dataclass(frozen=True)
class _SeriesInfo:
    series_uid: str
    datasets: tuple[Any, ...]
    plane: str | None
    native_decision: SeriesCenteringDecision
    coronal_height: CoronalJointHeight | None


def _single_study_uid(datasets: Sequence[Any]) -> str | None:
    values = {
        str(getattr(dataset, "StudyInstanceUID", "") or "").strip()
        for dataset in datasets
    }
    values.discard("")
    return next(iter(values)) if len(values) == 1 else None


def _same_knee_pair(coronal: _SeriesInfo, sagittal: _SeriesInfo) -> bool:
    coronal_side = infer_knee_laterality(coronal.datasets)
    sagittal_side = infer_knee_laterality(sagittal.datasets)
    if coronal_side.status == "conflict" or sagittal_side.status == "conflict":
        return False
    if (
        coronal_side.status == "resolved"
        and sagittal_side.status == "resolved"
        and coronal_side.side != sagittal_side.side
    ):
        return False
    return same_knee_fov_overlap(
        coronal.datasets,
        sagittal.datasets,
        tolerance_mm=float(CORONAL_FIRST_POLICY["fov_tolerance_mm"]),
    )


def _fov_components(
    coronals: Sequence[_SeriesInfo], sagittals: Sequence[_SeriesInfo]
) -> list[tuple[list[int], list[int]]]:
    adjacency: dict[tuple[str, int], set[tuple[str, int]]] = {}
    for coronal_index, coronal in enumerate(coronals):
        for sagittal_index, sagittal in enumerate(sagittals):
            if not _same_knee_pair(coronal, sagittal):
                continue
            coronal_node = ("c", coronal_index)
            sagittal_node = ("s", sagittal_index)
            adjacency.setdefault(coronal_node, set()).add(sagittal_node)
            adjacency.setdefault(sagittal_node, set()).add(coronal_node)

    components: list[tuple[list[int], list[int]]] = []
    seen: set[tuple[str, int]] = set()
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack = [start]
        coronal_indices: set[int] = set()
        sagittal_indices: set[int] = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            kind, index = node
            (coronal_indices if kind == "c" else sagittal_indices).add(index)
            stack.extend(adjacency.get(node, ()))
        if coronal_indices and sagittal_indices:
            components.append((sorted(coronal_indices), sorted(sagittal_indices)))
    return components


def _fallback_source(decision: SeriesCenteringDecision) -> str:
    if decision.mode == "joint_line" and decision.plane == "Sagittal":
        return "native_sagittal_locator"
    if decision.mode == "joint_line" and decision.plane == "Coronal":
        return "native_coronal_locator"
    if decision.plane == "Axial":
        return "axial_geometric"
    return "geometric_fallback"


def plan_coronal_first_study(
    study_uid: str,
    series_datasets: Mapping[str, Sequence[Any]],
) -> dict[str, dict[str, Any]]:
    """Return one no-constant centering record for every supplied series."""

    infos: list[_SeriesInfo] = []
    for series_uid in sorted(series_datasets):
        datasets = tuple(series_datasets[series_uid])
        if not datasets:
            raise ValueError(f"Empty DICOM series: {study_uid}/{series_uid}")
        observed_study = _single_study_uid(datasets)
        if observed_study != str(study_uid):
            raise ValueError(
                f"Study UID mismatch for {series_uid}: {observed_study!r} != {study_uid!r}"
            )
        plane = anatomical_plane(datasets[(len(datasets) - 1) // 2])
        decision = locate_series_center(
            datasets,
            maximum_shift_mm=float(CORONAL_FIRST_POLICY["maximum_shift_mm"]),
            coverage_policy="safe",
        )
        height = (
            coronal_joint_height(datasets, decision) if plane == "Coronal" else None
        )
        infos.append(
            _SeriesInfo(
                series_uid=str(series_uid),
                datasets=datasets,
                plane=plane,
                native_decision=decision,
                coronal_height=height,
            )
        )

    records: dict[str, dict[str, Any]] = {
        info.series_uid: {
            "decision": decision_to_dict(info.native_decision),
            "source": _fallback_source(info.native_decision),
            "component_id": None,
            "coronal_consensus_z_mm": None,
            "coronal_consensus_mad_mm": None,
            "coronal_consensus_range_mm": None,
            "coronal_source_series": [],
        }
        for info in infos
    }
    coronals = [info for info in infos if info.plane == "Coronal"]
    sagittals = [info for info in infos if info.plane == "Sagittal"]
    covered_sagittals: set[str] = set()
    for component_number, (coronal_indices, sagittal_indices) in enumerate(
        _fov_components(coronals, sagittals), start=1
    ):
        component_id = f"{study_uid}::knee{component_number}"
        component_coronals = [coronals[index] for index in coronal_indices]
        component_sagittals = [sagittals[index] for index in sagittal_indices]
        accepted_coronals = [
            info
            for info in component_coronals
            if info.coronal_height is not None
            and info.coronal_height.status == "candidate"
            and info.coronal_height.z_mm is not None
        ]
        heights = np.asarray(
            [float(info.coronal_height.z_mm) for info in accepted_coronals],
            dtype=float,
        )
        consensus_z: float | None = None
        consensus_mad: float | None = None
        consensus_range: float | None = None
        if heights.size:
            consensus_z = float(np.median(heights))
            consensus_mad = float(np.median(np.abs(heights - consensus_z)))
            consensus_range = float(np.ptp(heights))
        consensus_ok = bool(
            consensus_z is not None
            and consensus_mad is not None
            and consensus_mad
            <= float(CORONAL_FIRST_POLICY["coronal_consensus_mad_max_mm"])
            and consensus_range is not None
            and consensus_range
            <= float(CORONAL_FIRST_POLICY["coronal_consensus_range_max_mm"])
        )
        source_uids = [info.series_uid for info in accepted_coronals]
        confidence = (
            0.0
            if not accepted_coronals
            else 0.90
            * float(
                np.median(
                    [info.native_decision.confidence for info in accepted_coronals]
                )
            )
        )
        for info in component_coronals:
            records[info.series_uid]["component_id"] = component_id
        for sagittal in component_sagittals:
            covered_sagittals.add(sagittal.series_uid)
            chosen = sagittal.native_decision
            source = _fallback_source(chosen)
            if consensus_ok and consensus_z is not None:
                transfer = sagittal_center_from_absolute_z(
                    sagittal.datasets,
                    consensus_z,
                    confidence=confidence,
                    reason="accepted_coronal_consensus_height_transfer_raw_lps",
                    maximum_shift_mm=float(CORONAL_FIRST_POLICY["maximum_shift_mm"]),
                    mad_z_mm=consensus_mad,
                )
                if transfer.mode == "joint_line":
                    chosen = transfer
                    source = "coronal_consensus_transfer"
            records[sagittal.series_uid] = {
                "decision": decision_to_dict(chosen),
                "source": source,
                "component_id": component_id,
                "coronal_consensus_z_mm": consensus_z,
                "coronal_consensus_mad_mm": consensus_mad,
                "coronal_consensus_range_mm": consensus_range,
                "coronal_source_series": source_uids,
            }

    for sagittal in sagittals:
        if sagittal.series_uid not in covered_sagittals:
            records[sagittal.series_uid]["component_id"] = None
    return records


def _resolve_series_dir(images_root: str | Path, study_uid: str, series_uid: str) -> Path:
    root = Path(images_root)
    candidates = (root / study_uid / series_uid, root / series_uid)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def _load_sorted_series(directory: str | Path) -> tuple[Any, ...]:
    try:
        import pydicom
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("Centering manifests require pydicom") from error

    rows: list[tuple[Path, Any]] = []
    failures: list[str] = []
    for path in discover_dicom_files(directory):
        try:
            rows.append((path, pydicom.dcmread(path, force=True)))
        except Exception as error:
            failures.append(f"{path.name}: {error}")
    if not rows:
        raise ValueError(
            f"No readable DICOMs in {directory}; first failures={failures[:3]}"
        )
    rows.sort(key=lambda pair: dicom_sort_key(pair[1], pair[0].name))
    return tuple(dataset for _, dataset in rows)


def _uid_set_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(set(str(item) for item in values)):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _atomic_json(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(temporary, destination)


def build_coronal_first_manifest(
    manifest: pd.DataFrame,
    series: pd.DataFrame,
    *,
    images_root: str | Path,
    output_path: str | Path,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Materialize all study-specific decisions before any centered crop."""

    if ID_COLUMN not in manifest or manifest[ID_COLUMN].duplicated().any():
        raise ValueError(f"Manifest must be unique by {ID_COLUMN}")
    required = {ID_COLUMN, SERIES_COLUMN}
    missing = sorted(required.difference(series.columns))
    if missing:
        raise ValueError(f"Series metadata is missing columns: {missing}")
    study_ids = manifest[ID_COLUMN].astype(str).tolist()
    groups = {
        str(uid): part.copy() for uid, part in series.groupby(ID_COLUMN, sort=False)
    }
    studies: dict[str, Any] = {}
    inventory: list[str] = []
    source_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    started = time.monotonic()
    for position, study_uid in enumerate(study_ids, start=1):
        study_series = groups.get(study_uid)
        if study_series is None or study_series.empty:
            raise ValueError(f"No series metadata for study {study_uid}")
        loaded: dict[str, tuple[Any, ...]] = {}
        for series_uid in sorted(study_series[SERIES_COLUMN].astype(str).unique()):
            directory = _resolve_series_dir(images_root, study_uid, series_uid)
            datasets = _load_sorted_series(directory)
            loaded[series_uid] = datasets
            inventory.append(f"{study_uid}|{series_uid}|{len(datasets)}")
        records = plan_coronal_first_study(study_uid, loaded)
        for record in records.values():
            source_counts[str(record["source"])] += 1
            reason_counts[str(record["decision"]["reason"])] += 1
        studies[study_uid] = {"series": records}
        if progress is not None:
            progress(
                {
                    "position": position,
                    "total": len(study_ids),
                    "study_uid": study_uid,
                    "series": len(records),
                    "elapsed_seconds": time.monotonic() - started,
                }
            )

    payload = {
        "format": CENTERING_MANIFEST_FORMAT,
        "policy": CORONAL_FIRST_POLICY,
        "policy_sha256": policy_sha256(),
        "study_uid_set_sha256": _uid_set_sha256(study_ids),
        "series_inventory_sha256": _canonical_sha256(sorted(inventory)),
        "studies": studies,
        "summary": {
            "studies": len(study_ids),
            "series": int(sum(len(value["series"]) for value in studies.values())),
            "source_counts": dict(sorted(source_counts.items())),
            "reason_counts": dict(sorted(reason_counts.items())),
            "fixed_prior_decisions": 0,
        },
    }
    _atomic_json(payload, output_path)
    return {
        **payload["summary"],
        "format": CENTERING_MANIFEST_FORMAT,
        "policy_sha256": payload["policy_sha256"],
        "study_uid_set_sha256": payload["study_uid_set_sha256"],
        "series_inventory_sha256": payload["series_inventory_sha256"],
        "output_sha256": sha256_file(output_path),
        "elapsed_seconds": float(time.monotonic() - started),
    }


def load_centering_manifest(
    path: str | Path, *, expected_sha256: str | None = None
) -> dict[str, Any]:
    source = Path(path)
    actual_sha256 = sha256_file(source)
    if expected_sha256 is not None and actual_sha256 != str(expected_sha256):
        raise ValueError(
            f"Centering manifest SHA-256 mismatch: {actual_sha256} != {expected_sha256}"
        )
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("format") != CENTERING_MANIFEST_FORMAT:
        raise ValueError(f"Unknown centering manifest format: {payload.get('format')!r}")
    if payload.get("policy_sha256") != policy_sha256():
        raise ValueError("Centering manifest policy does not match coronal_first_v1")
    if payload.get("policy") != CORONAL_FIRST_POLICY:
        raise ValueError("Centering manifest contains a non-canonical policy payload")
    if not isinstance(payload.get("studies"), dict):
        raise ValueError("Centering manifest lacks studies")
    return payload


def manifest_decision(
    payload: Mapping[str, Any], study_uid: str, series_uid: str
) -> SeriesCenteringDecision:
    try:
        record = payload["studies"][str(study_uid)]["series"][str(series_uid)]
    except (KeyError, TypeError) as error:
        raise KeyError(
            f"Centering manifest lacks {study_uid}/{series_uid}"
        ) from error
    return decision_from_dict(record["decision"])
