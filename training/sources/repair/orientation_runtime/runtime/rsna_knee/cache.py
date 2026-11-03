"""Series slot selection and bounded persistent preprocessing cache."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from collections.abc import Callable
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .constants import ID_COLUMN, SERIES_COLUMN, SERIES_REQUIRED_COLUMNS
from .dicom import preprocess_dicom_series


def select_series_for_slots(
    study_series: pd.DataFrame,
    slots: list[dict[str, Any]],
) -> list[str | None]:
    missing = sorted(set(SERIES_REQUIRED_COLUMNS).difference(study_series.columns))
    if missing:
        raise ValueError(f"Series metadata is missing columns: {missing}")
    selected: list[str | None] = []
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
        desired_fat = int(slot.get("fat_suppression", slot["fluid_sensitive"]))
        candidates["_fat_match"] = (
            pd.to_numeric(candidates["Fat_Suppression"], errors="coerce") == desired_fat
        ).astype(int)
        candidates = candidates.sort_values(
            ["_fat_match", SERIES_COLUMN], ascending=[False, True]
        )
        selected.append(str(candidates.iloc[0][SERIES_COLUMN]))
    return selected


def resolve_series_dir(images_root: str | Path, study_uid: str, series_uid: str) -> Path:
    root = Path(images_root)
    candidates = [root / study_uid / series_uid, root / series_uid]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def cache_path(
    cache_root: str | Path,
    study_uid: str,
    series_uid: str,
    *,
    preprocessing_variant: str | None = None,
) -> Path:
    root = Path(cache_root)
    if preprocessing_variant:
        root = root / "preprocessing" / preprocessing_variant
    return root / study_uid / f"{series_uid}.npy"


def _preprocessing_variant(data_config: dict[str, Any]) -> str | None:
    mode = str(data_config.get("centering_mode", "geometric"))
    if mode == "geometric":
        return None
    if mode in {
        "coronal_first_v1",
        "coronal_first_superior16p4_v1",
        "coronal_first_superior20_v1",
    }:
        expected_sha256 = str(data_config.get("centering_manifest_sha256", ""))
        if len(expected_sha256) != 64:
            raise ValueError(
                f"{mode} requires data.centering_manifest_sha256"
            )
        return f"{mode}-{expected_sha256[:16]}"
    return mode


@lru_cache(maxsize=8)
def _load_centering_manifest_cached(
    path: str, expected_sha256: str
) -> dict[str, Any]:
    from .centering_manifest import load_centering_manifest

    return load_centering_manifest(path, expected_sha256=expected_sha256)


def _manifest_centering_decision(
    data_config: dict[str, Any], study_uid: str, series_uid: str
) -> Any:
    from .centering_manifest import manifest_decision

    path = str(data_config.get("centering_manifest", ""))
    expected_sha256 = str(data_config.get("centering_manifest_sha256", ""))
    if not path or len(expected_sha256) != 64:
        raise ValueError(
            "Manifest-aware centering requires data.centering_manifest and its SHA-256"
        )
    payload = _load_centering_manifest_cached(path, expected_sha256)
    return manifest_decision(payload, study_uid, series_uid)


def load_or_create_series_cache(
    *,
    images_root: str | Path,
    cache_root: str | Path,
    study_uid: str,
    series_uid: str,
    data_config: dict[str, Any],
) -> np.ndarray:
    centering_mode = str(data_config.get("centering_mode", "geometric"))
    centering_decision = (
        _manifest_centering_decision(data_config, study_uid, series_uid)
        if centering_mode
        in {
            "coronal_first_v1",
            "coronal_first_superior16p4_v1",
            "coronal_first_superior20_v1",
        }
        else None
    )
    destination = cache_path(
        cache_root,
        study_uid,
        series_uid,
        preprocessing_variant=_preprocessing_variant(data_config),
    )
    if destination.is_file():
        return np.load(destination, mmap_mode="r")

    source = resolve_series_dir(images_root, study_uid, series_uid)
    tokens = preprocess_dicom_series(
        source,
        image_size=int(data_config["image_size"]),
        slices_per_series=int(data_config["slices_per_series"]),
        crop_mm=data_config.get("crop_mm"),
        percentile_low=float(data_config.get("percentile_low", 1.0)),
        percentile_high=float(data_config.get("percentile_high", 99.0)),
        centering_mode=centering_mode,
        centering_decision=centering_decision,
    )
    cache_dtype = str(data_config.get("cache_dtype", "uint8"))
    if cache_dtype == "uint8":
        cached = np.rint(tokens * 255.0).clip(0, 255).astype(np.uint8)
    elif cache_dtype == "float16":
        cached = tokens.astype(np.float16)
    else:
        raise ValueError("data.cache_dtype must be 'uint8' or 'float16'")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f".{os.getpid()}.tmp.npy")
    np.save(temporary, cached, allow_pickle=False)
    try:
        os.replace(temporary, destination)
    except OSError:
        if temporary.exists():
            temporary.unlink()
        if not destination.exists():
            raise
    return np.load(destination, mmap_mode="r")


def build_cache(
    manifest: pd.DataFrame,
    series: pd.DataFrame,
    data_config: dict[str, Any],
    *,
    limit: int | None = None,
    progress: Callable[[dict[str, int]], None] | None = None,
) -> dict[str, int]:
    if ID_COLUMN not in manifest:
        raise ValueError(f"Manifest is missing {ID_COLUMN}")
    groups = {str(uid): part for uid, part in series.groupby(ID_COLUMN, sort=False)}
    slots = data_config["slots"]
    images_root = data_config["images_root"]
    cache_root = data_config["cache_root"]
    summary = {"studies_seen": 0, "series_selected": 0, "series_cached": 0, "missing_slots": 0}

    studies = manifest[ID_COLUMN].astype(str).tolist()
    if limit is not None:
        studies = studies[:limit]
    empty = series.iloc[0:0]
    for study_uid in studies:
        summary["studies_seen"] += 1
        selections = select_series_for_slots(groups.get(study_uid, empty), slots)
        for series_uid in selections:
            if series_uid is None:
                summary["missing_slots"] += 1
                continue
            summary["series_selected"] += 1
            destination = cache_path(
                cache_root,
                study_uid,
                series_uid,
                preprocessing_variant=_preprocessing_variant(data_config),
            )
            was_cached = destination.is_file()
            load_or_create_series_cache(
                images_root=images_root,
                cache_root=cache_root,
                study_uid=study_uid,
                series_uid=series_uid,
                data_config=data_config,
            )
            if not was_cached:
                summary["series_cached"] += 1
        if progress is not None:
            progress(dict(summary))
    return summary
