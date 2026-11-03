"""Local-head diagnostics, independent of training and prediction-based selection.

Attention is a model diagnostic, not a clinical localization annotation. Perturbation
replays remove post-projection features, not raw pixels or physical anatomy.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import ID_COLUMN, TARGET_COLUMNS


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path: str | Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def deterministic_selection(eligible: pd.DataFrame, *, per_fold: int = 10,
                            folds: tuple[int, ...] = (0, 3), seed: int = 20260905) -> pd.DataFrame:
    """Round-robin metadata strata within each fold; never consumes predictions/labels.

    This intentionally covers metadata diversity, rather than estimating population
    prevalence. Input ordering cannot affect selection; each stratum and UID is hash-ranked.
    """
    columns = [ID_COLUMN, "fold", "manufacturer", "thickness_band", "valid_slots",
               "valid_tokens", "geometry_status"]
    if per_fold < 1 or eligible[ID_COLUMN].duplicated().any():
        raise ValueError("Selection needs positive size and unique study UIDs")
    frame = eligible[columns].copy()
    frame[ID_COLUMN] = frame[ID_COLUMN].astype(str)
    frame["stratum"] = (frame["manufacturer"].astype(str) + "|" +
                        frame["thickness_band"].astype(str) + "|slots=" +
                        frame["valid_slots"].astype(str))
    def rank(text: str) -> str:
        return hashlib.sha256(f"{seed}|{text}".encode()).hexdigest()
    selected = []
    for fold in folds:
        part = frame.loc[frame["fold"] == fold].copy()
        if len(part) < per_fold:
            raise ValueError(f"Only {len(part)} eligible studies in fold {fold}; require {per_fold}")
        queues = {}
        for stratum, group in part.groupby("stratum", sort=True):
            queues[stratum] = sorted(group.to_dict("records"), key=lambda row: rank(row[ID_COLUMN]))
        keys = sorted(queues, key=lambda key: rank(f"fold={fold}|stratum={key}"))
        chosen = []
        while len(chosen) < per_fold:
            for key in keys:
                if queues[key]:
                    chosen.append(queues[key].pop(0))
                if len(chosen) == per_fold:
                    break
        selected.extend(chosen)
    result = pd.DataFrame(selected)
    result.insert(0, "selection_index", range(len(result)))
    return result


def read_geometry(path: Path) -> dict[str, Any]:
    """Read only spatial tags, avoiding patient/demographic metadata and pixel decoding."""
    import pydicom
    names = ["Rows", "Columns", "PixelSpacing", "ImagePositionPatient", "ImageOrientationPatient",
             "FrameOfReferenceUID", "SOPInstanceUID", "SliceThickness", "SpacingBetweenSlices"]
    dataset = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=names)
    result = {}
    for name in names:
        value = getattr(dataset, name, None)
        if value is None:
            continue
        if name in {"FrameOfReferenceUID", "SOPInstanceUID"}:
            result[name] = str(value)
        elif name in {"Rows", "Columns"}:
            result[name] = int(value)
        elif name in {"PixelSpacing", "ImagePositionPatient", "ImageOrientationPatient"}:
            result[name] = [float(item) for item in value]
        else:
            result[name] = float(value)
    return result


def geometry_affines(tags: dict[str, Any], *, crop_mm: float = 130.0, image_size: int = 224,
                     centering_mode: str = "geometric",
                     crop_profile: str = "local_rectangular_mm") -> dict[str, Any]:
    """Nominal pixel-center transform for the documented geometric cache crop.

    The affine describes sampling coordinates, not interpolation support or verified
    cross-series registration. It cannot recover a missing/unknown cache crop policy.
    """
    try:
        rows, cols = int(tags["Rows"]), int(tags["Columns"])
        spacing = np.asarray(tags["PixelSpacing"], dtype=np.float64)
        ipp = np.asarray(tags["ImagePositionPatient"], dtype=np.float64)
        iop = np.asarray(tags["ImageOrientationPatient"], dtype=np.float64)
        if (rows < 1 or cols < 1 or image_size < 1 or spacing.shape != (2,) or
                ipp.shape != (3,) or iop.shape != (6,) or
                not np.isfinite(np.r_[spacing, ipp, iop, crop_mm]).all() or np.any(spacing <= 0)):
            raise ValueError("Missing/invalid spatial tags")
        if not (np.allclose(np.linalg.norm(iop.reshape(2, 3), axis=1), 1, atol=.01) and
                abs(float(iop[:3] @ iop[3:])) < .01):
            raise ValueError("Invalid orientation basis")
    except (KeyError, TypeError, ValueError) as error:
        return {"status": "unknown", "reason": str(error), "raw_tags": tags}
    raw_to_lps = np.column_stack([iop[:3] * spacing[1], iop[3:] * spacing[0], ipp])
    result = {"raw_tags": tags, "raw_xy1_to_lps": raw_to_lps.tolist(),
              "registration_status": "raw_DICOM_LPS_only_cross_series_registration_unverified"}
    if centering_mode != "geometric" or crop_profile != "local_rectangular_mm":
        return dict(result, status="unknown_crop", reason="Cache crop transform is not recorded for this policy")
    h = min(rows, max(1, int(round(crop_mm / spacing[0])))) if crop_mm > 0 else rows
    w = min(cols, max(1, int(round(crop_mm / spacing[1])))) if crop_mm > 0 else cols
    top, left = max(0, (rows - h) // 2), max(0, (cols - w) // 2)
    model_to_raw = np.array([[w / image_size, 0, left + .5 * w / image_size - .5],
                             [0, h / image_size, top + .5 * h / image_size - .5], [0, 0, 1.]])
    return dict(result, status="nominal_reconstructed", crop_xywh=[left, top, w, h],
                crop_policy="geometric/local_rectangular_mm", model_xy1_to_raw_xy1=model_to_raw.tolist(),
                model_xy1_to_lps=(raw_to_lps @ model_to_raw).tolist(),
                coordinate_note="x=column,y=row,zero-based pixel centers; PIL bilinear cache then align_corners=False resize; interpolation support not represented")


def find_raw_source(raw_roots: list[Path], uid: str, series_uid: str, filename: str) -> Path | None:
    # Provenance names are never interpreted as arbitrary paths.
    if any(Path(item).name != item or item in {"", ".", ".."} for item in [uid, series_uid, filename]):
        raise ValueError("Invalid source identifiers in cache provenance")
    for root in raw_roots:
        path = root / uid / series_uid / filename
        if path.is_file():
            return path
    return None


def study_provenance(cache: Any, uid: str, memo: dict[Path, pd.DataFrame]) -> pd.DataFrame:
    shard, row_index = cache.entries[uid]
    path = Path(shard) / "provenance.csv"
    if path not in memo:
        memo[path] = pd.read_csv(path, dtype={ID_COLUMN: str, "SeriesInstanceUID": str,
                                            "source_file": str}, keep_default_na=False)
    frame = memo[path]
    part = frame.loc[frame[ID_COLUMN] == uid].sort_values("token_index").copy()
    if (len(part) != cache.token_count or part["token_index"].tolist() != list(range(cache.token_count))
            or set(part["row"].astype(int)) != {int(row_index)}):
        raise ValueError(f"Incomplete/mismatched provenance for {uid}")
    return part


def eligible_metadata(manifest: pd.DataFrame, cache: Any, raw_roots: list[Path],
                      profiles: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Eligibility is raw-source presence plus valid per-series spatial headers."""
    if manifest[ID_COLUMN].duplicated().any() or profiles[ID_COLUMN].duplicated().any():
        raise ValueError("Manifest and scanner profiles must be unique by study")
    profile_map = profiles.set_index(ID_COLUMN).to_dict("index")
    memo: dict[Path, pd.DataFrame] = {}
    records, excluded = [], {"not_cached_or_no_raw_study": 0, "missing_source": 0, "invalid_geometry": 0}
    candidates = manifest.loc[manifest["fold"].isin([0, 3]), [ID_COLUMN, "fold"]]
    for uid, fold in candidates.itertuples(index=False, name=None):
        uid = str(uid)
        if uid not in cache.entries or not any((root / uid).is_dir() for root in raw_roots):
            excluded["not_cached_or_no_raw_study"] += 1
            continue
        provenance = study_provenance(cache, uid, memo)
        valid = provenance.loc[provenance["valid"].astype(str).str.lower().isin(["true", "1"])]
        paths = [find_raw_source(raw_roots, uid, str(row.SeriesInstanceUID), str(row.source_file))
                 for row in valid.itertuples(index=False)]
        if not len(valid) or any(path is None for path in paths):
            excluded["missing_source"] += 1
            continue
        geometry_ok = True
        seen = set()
        for row, path in zip(valid.itertuples(index=False), paths, strict=True):
            if row.SeriesInstanceUID in seen:
                continue
            seen.add(row.SeriesInstanceUID)
            try:
                geometry_ok &= geometry_affines(read_geometry(path))["status"] == "nominal_reconstructed"
            except Exception:
                geometry_ok = False
        if not geometry_ok:
            excluded["invalid_geometry"] += 1
            continue
        profile = profile_map.get(uid, {})
        vendor = str(profile.get("scanner_manufacturer", "unknown"))
        if vendor in {"", "nan", "None"}:
            vendor = "unknown"
        try:
            thickness = [float(x) for x in json.loads(profile.get("acquisition_slice_thickness_values_json", "[]"))]
            thickness = [x for x in thickness if math.isfinite(x) and x > 0]
        except (ValueError, TypeError):
            thickness = []
        band = "unknown" if not thickness else ("max_le_3mm" if max(thickness) <= 3 else "max_gt_3mm")
        records.append({ID_COLUMN: uid, "fold": int(fold), "manufacturer": vendor,
                        "thickness_band": band, "valid_slots": int(valid.slot_index.nunique()),
                        "valid_tokens": len(valid), "geometry_status": "all_sources_present_one_header_per_series_valid"})
    return pd.DataFrame(records), excluded


def validate_diagnostics(result: dict[str, Any], *, atol: float = 1e-5) -> dict[str, Any]:
    import torch
    alpha = result["alpha"]
    mask = result["cell_mask"].unsqueeze(-1)
    contribution = result["signed_contributions"]
    expected_shape = (*mask.shape[:-1], len(TARGET_COLUMNS))
    if tuple(alpha.shape) != expected_shape or contribution.shape != alpha.shape:
        raise ValueError("Expected attention and contributions for all 12 labels")
    for key in ["alpha", "signed_contributions", "global_logits", "local_logits", "logits",
                "gated_local_logits", "local_bias", "gated_local_bias", "local_features"]:
        if not torch.isfinite(result[key]).all():
            raise ValueError(f"Nonfinite diagnostic tensor: {key}")
    if (alpha < 0).any() or (alpha.masked_select(~mask.expand_as(alpha)) != 0).any():
        raise ValueError("Attention is negative or reaches masked cells")
    expected_mass = mask.any(dim=(1, 2, 3)).float().expand(-1, len(TARGET_COLUMNS))
    if not torch.allclose(alpha.sum((1, 2, 3)), expected_mass, atol=atol, rtol=0):
        raise ValueError("Raw attention does not sum to one over valid study cells")
    rebuilt = contribution.sum((1, 2, 3)) + result["gated_local_bias"]
    residual_error = float((rebuilt - result["gated_local_logits"]).abs().max())
    total_error = float((result["global_logits"] + rebuilt - result["logits"]).abs().max())
    if max(residual_error, total_error) > atol:
        raise ValueError(f"Signed contribution decomposition failed: {residual_error}, {total_error}")
    return {"all_finite": True, "decomposition_max_abs_error": max(residual_error, total_error)}


def removal_masks(alpha: np.ndarray, valid: np.ndarray, slots: np.ndarray, *, fraction: float = .1,
                  repeats: int = 10, seed: int = 20260905) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return keep masks [T,H,W,C]; random removals match top counts per semantic slot/label."""
    if alpha.ndim != 4 or valid.shape != alpha.shape[:3] or slots.shape != alpha.shape[:1]:
        raise ValueError("Expected alpha [T,H,W,C], valid [T,H,W], slots [T]")
    if not 0 < fraction < 1 or repeats != 10 or not np.isfinite(alpha).all():
        raise ValueError("Use a finite attention tensor and exactly ten random repeats")
    flat_valid = np.flatnonzero(valid.ravel())
    if len(flat_valid) < 2:
        raise ValueError("Need at least two valid cells for perturbation")
    count = min(len(flat_valid) - 1, max(1, int(math.ceil(fraction * len(flat_valid)))))
    labels = alpha.shape[-1]
    keep = np.broadcast_to(valid[..., None], alpha.shape).copy()
    flat_keep = keep.reshape(-1, labels)
    flat_alpha = alpha.reshape(-1, labels)
    flat_slots = np.broadcast_to(slots[:, None, None], valid.shape).ravel()
    random_keep = [np.broadcast_to(valid[..., None], alpha.shape).copy() for _ in range(repeats)]
    for label in range(labels):
        order = np.argsort(-flat_alpha[flat_valid, label], kind="stable")
        removed = flat_valid[order[:count]]
        flat_keep[removed, label] = False
        for repeat, mask in enumerate(random_keep):
            # Independent repeat/label seed; UID-derived seed supplied by caller.
            rng = np.random.default_rng(np.random.SeedSequence([seed, repeat, label]))
            for slot in np.unique(flat_slots[removed]):
                slot_count = int(np.sum(flat_slots[removed] == slot))
                candidates = flat_valid[flat_slots[flat_valid] == slot]
                random_removed = rng.choice(candidates, slot_count, replace=False)
                mask.reshape(-1, labels)[random_removed, label] = False
    return keep, random_keep


def feature_perturbations(model: Any, result: dict[str, Any], *, uid: str,
                          fraction: float = .1, seed: int = 20260905) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay one study in FP32 with fixed global logits and identical masked softmax."""
    import torch
    if fraction != .1:
        raise ValueError("The frozen perturbation protocol uses exactly 10%")
    if model.training or result["alpha"].shape[0] != 1:
        raise ValueError("Perturbation export requires eval mode and one study")
    features, global_logits = result["local_features"], result["global_logits"]
    if features.dtype != torch.float32 or global_logits.dtype != torch.float32:
        raise ValueError("Use FP32 original forward and replay without autocast")
    baseline = result["logits"].detach()
    uid_seed = int(hashlib.sha256(f"{seed}|{uid}".encode()).hexdigest()[:8], 16)
    top, random_masks = removal_masks(result["alpha"][0].cpu().numpy(),
                                     result["cell_mask"][0].cpu().numpy(),
                                     result["slot_ids"][0].cpu().numpy(), fraction=fraction, seed=uid_seed)
    rows = []
    def record(condition: str, repeat: int, logits: Any, mask: Any = None) -> None:
        for index, label in enumerate(TARGET_COLUMNS):
            changed = float(logits[0, index])
            original = float(baseline[0, index])
            removed = result["cell_mask"] if mask is None else result["cell_mask"] & ~mask[..., index]
            removed_by_slot = {str(int(slot)): int(removed[0, result["slot_ids"][0] == slot].sum())
                               for slot in result["slot_ids"][0].unique()}
            rows.append({ID_COLUMN: uid, "label": label, "condition": condition, "repeat": repeat,
                         "seed": uid_seed, "baseline_logit": original, "perturbed_logit": changed,
                         "delta_logit": changed - original,
                         "baseline_probability": float(baseline.sigmoid()[0, index]),
                         "perturbed_probability": float(logits.sigmoid()[0, index]),
                         "delta_probability": float((logits.sigmoid() - baseline.sigmoid())[0, index]),
                         "removed_cells": int(result["cell_mask"].sum()) if mask is None else
                            int(result["cell_mask"].sum()) - int(mask[..., index].sum()),
                         "removed_by_slot_json": json.dumps(removed_by_slot, sort_keys=True),
                         "removed_mask_sha256": hashlib.sha256(np.packbits(removed.cpu().numpy()).tobytes()).hexdigest(),
                         "global_logits_fixed": True, "feature_scope": "post_projection_slot_position_LN_features",
                         "selection_basis": "original_raw_alpha", "renormalization": "original_masked_softmax"})
    with torch.inference_mode():
        replay = model.forward_local_from_features(features, result["cell_mask"], global_logits, diagnostics=True)
        replay_error = float((replay["logits"] - baseline).abs().max())
        if replay_error > 1e-5:
            raise ValueError(f"Unmodified feature replay differs: {replay_error}")
        record("local_off", -1, global_logits)
        for condition, repeat, mask in [("top_alpha_10pct", -1, top)] + [
                ("slot_matched_random_10pct", i, mask) for i, mask in enumerate(random_masks)]:
            tensor_mask = torch.as_tensor(mask[None], device=features.device)
            if (tensor_mask & ~result["cell_mask"][..., None]).any():
                raise ValueError("Perturbation may not unmask an originally invalid cell")
            altered = model.forward_local_from_features(features, tensor_mask, global_logits, diagnostics=True)
            if not torch.equal(altered["global_logits"], global_logits) or not torch.isfinite(altered["logits"]).all():
                raise ValueError("Replay changed global logits or generated nonfinite values")
            record(condition, repeat, altered["logits"], tensor_mask)
    return rows, {"replay_max_abs_error": replay_error, "random_repeats": 10,
                  "removal_fraction": fraction, "top_removed_per_label": int((~top[..., 0] & result["cell_mask"][0].cpu().numpy()).sum())}


def slot_masses(result: dict[str, Any], slot_names: list[str]) -> list[dict[str, Any]]:
    import torch
    records = []
    for index, name in enumerate(slot_names):
        selected = (result["slot_ids"][0] == index) & result["token_mask"][0]
        alpha = result["alpha"][0, selected]
        contribution = result["signed_contributions"][0, selected]
        records.append({"slot_index": index, "slot_name": name, "valid_tokens": int(selected.sum()),
                        "missing": not bool(selected.any()), "alpha_mass": alpha.sum((0, 1, 2)).tolist(),
                        "signed_contribution": contribution.sum((0, 1, 2)).tolist(),
                        "positive_contribution": torch.clamp(contribution, min=0).sum((0, 1, 2)).tolist(),
                        "negative_contribution": torch.clamp(contribution, max=0).sum((0, 1, 2)).tolist()})
    return records


def embedded_json(value: Any) -> str:
    """Safe inline JSON, including adversarial study/label text containing </script>."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c").replace("&", "\\u0026").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def verify_diagnostics_bundle(folder: str | Path) -> dict[str, Any]:
    """Fail closed on corruption, wrong selection, missing labels or incomplete replays.

    Checkpoint/run binding to the *external queue contract* remains the caller's job:
    compare returned training_identity_sha256/checkpoint_sha256/run_id to its frozen run.
    """
    folder = Path(folder)
    marker = folder / "DIAGNOSTICS_COMPLETE.json"
    complete = json.loads(marker.read_text())
    alias = folder / "DIAGNOSTICS_MANIFEST.json"
    if alias.exists() and alias.read_bytes() != marker.read_bytes():
        raise ValueError("Diagnostic completion aliases differ")
    required = {"diagnostic_studies.csv", "selection_contract.json", "diagnostic_geometry.json",
                "attention_maps.npz", "signed_contributions.npz", "local_head_perturbations.csv",
                "diagnostics.json", "index.html", "README.md"}
    if complete.get("version") != 1 or complete.get("status") != "COMPLETE" or not required.issubset(complete.get("files", {})):
        raise ValueError("Incomplete diagnostics manifest")
    for relative, expected in complete["files"].items():
        path = folder / relative
        if (Path(relative).is_absolute() or ".." in Path(relative).parts or path.is_symlink()
                or not path.is_file() or path.stat().st_size != expected["bytes"]
                or sha256_file(path) != expected["sha256"]):
            raise ValueError(f"Diagnostic payload checksum/size/path mismatch: {relative}")
    for key in ["checkpoint_sha256", "training_identity_sha256", "selection_sha256", "selection_contract_sha256"]:
        value = complete.get(key, "")
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"Missing diagnostic binding: {key}")
    if (sha256_file(folder / "diagnostic_studies.csv") != complete["selection_sha256"] or
            sha256_file(folder / "selection_contract.json") != complete["selection_contract_sha256"] or
            complete["labels"] != list(TARGET_COLUMNS)):
        raise ValueError("Selection/target binding mismatch")
    selection = pd.read_csv(folder / "diagnostic_studies.csv")
    contract = json.loads((folder / "selection_contract.json").read_text())
    if (contract["status"] != "FROZEN" or contract["selection_sha256"] != complete["selection_sha256"]
            or contract["predictions_read_for_selection"] is not False
            or contract["checkpoint_read_for_selection"] is not False
            or contract["study_uids"] != selection[ID_COLUMN].tolist()
            or len(selection) != contract["study_count"] or selection[ID_COLUMN].duplicated().any()
            or sha256_file(folder / "diagnostic_geometry.json") != contract["geometry_sidecar_sha256"]):
        raise ValueError("Frozen selection contract mismatch")
    expected_uids = selection.loc[selection.fold == complete["fold"], ID_COLUMN].tolist()
    if (complete["study_uids"] != expected_uids or complete["study_count"] != len(expected_uids)
            or len(expected_uids) < 1):
        raise ValueError("Exported studies differ from selected holdout fold")
    checks = complete["checks"]
    if (checks.get("all_finite") is not True or checks.get("precision") != "float32_no_autocast"
            or checks.get("random_repeats") != 10):
        raise ValueError("Missing numerical/precision/replay checks")
    for key in ["decomposition_max_abs_error", "replay_max_abs_error"]:
        value = checks.get(key, float("nan"))
        if not math.isfinite(value) or value < 0 or value > 1e-5:
            raise ValueError(f"Failed numerical check: {key}")
    with np.load(folder / "attention_maps.npz", allow_pickle=False) as attention, np.load(
            folder / "signed_contributions.npz", allow_pickle=False) as contributions:
        alpha, signed = attention["alpha"], contributions["signed_contributions"]
        expected_shape = (len(expected_uids), 46, 14, 14, len(TARGET_COLUMNS))
        if alpha.shape != expected_shape or signed.shape != expected_shape:
            raise ValueError("Expected every cell of 46 tokens and all 12 labels")
        for archive in [attention, contributions]:
            if archive["study_uids"].tolist() != expected_uids or archive["labels"].tolist() != list(TARGET_COLUMNS):
                raise ValueError("Map UID/target order differs")
        mask = attention["token_mask"].astype(bool)
        if mask.shape != expected_shape[:2] or not np.array_equal(mask, contributions["token_mask"]):
            raise ValueError("Map validity mismatch")
        if (alpha.dtype != np.float32 or signed.dtype != np.float32 or
                not np.isfinite(alpha).all() or not np.isfinite(signed).all() or (alpha < 0).any()
                or (alpha[~mask] != 0).any() or (signed[~mask] != 0).any()
                or not np.allclose(alpha.sum((1, 2, 3)), 1, atol=1e-5, rtol=0)):
            raise ValueError("Raw map values or normalization invalid")
        rebuilt = signed.sum((1, 2, 3)) + contributions["gated_local_bias"] + contributions["global_logits"]
        if not np.isfinite(rebuilt).all() or not np.allclose(rebuilt, contributions["logits"], atol=1e-5, rtol=0):
            raise ValueError("Saved contribution maps do not reconstruct logits")
    frame = pd.read_csv(folder / "local_head_perturbations.csv")
    if len(frame) != len(expected_uids) * 12 * 12 or len(frame) != checks["perturbation_rows"]:
        raise ValueError("Incomplete perturbation table")
    expected_conditions = {("local_off", -1), ("top_alpha_10pct", -1)} | {
        ("slot_matched_random_10pct", repeat) for repeat in range(10)}
    if set(frame[ID_COLUMN]) != set(expected_uids) or set(frame["label"]) != set(TARGET_COLUMNS):
        raise ValueError("Perturbation studies or labels differ")
    for (uid, label), part in frame.groupby([ID_COLUMN, "label"]):
        if len(part) != 12 or set(zip(part.condition, part.repeat)) != expected_conditions:
            raise ValueError(f"Missing/duplicate perturbation conditions: {uid}/{label}")
        top = part.loc[part.condition == "top_alpha_10pct"].iloc[0]
        top_counts = json.loads(top.removed_by_slot_json)
        if sum(top_counts.values()) != int(top.removed_cells):
            raise ValueError("Per-slot removed cell counts do not add up")
        for row in part.loc[part.condition == "slot_matched_random_10pct"].itertuples(index=False):
            if json.loads(row.removed_by_slot_json) != top_counts or row.removed_cells != top.removed_cells:
                raise ValueError("Random perturbation is not matched per slot and label")
    if not frame.global_logits_fixed.astype(str).str.lower().eq("true").all():
        raise ValueError("Perturbation global branch was not fixed")
    numeric = frame[["baseline_logit", "perturbed_logit", "delta_logit", "baseline_probability",
                     "perturbed_probability", "delta_probability"]].to_numpy(dtype=float)
    if (not np.isfinite(numeric).all() or
            not np.allclose(numeric[:, 1] - numeric[:, 0], numeric[:, 2], atol=1e-8, rtol=0)
            or not np.allclose(numeric[:, 4] - numeric[:, 3], numeric[:, 5], atol=1e-6, rtol=0)):
        raise ValueError("Invalid perturbation arithmetic")
    return complete
