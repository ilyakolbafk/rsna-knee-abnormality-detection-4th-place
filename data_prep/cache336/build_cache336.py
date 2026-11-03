#!/usr/bin/env python3
"""Build one immutable v8.3 AUX60 model-ready pixel-cache component.

The same builder is used for pilot, production train shards, and the separate
Gold58 validation shard. It never mixes Gold58 into the training namespace.
"""
from __future__ import annotations

import ast
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def array_hashes(arrays: dict) -> dict:
    return {name: sha256_array(arrays[name]) for name in ("pixels", "mask", "meta")}


def load_compact(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        if str(z["format"]) != "native64-token-bank-v1":
            raise RuntimeError("Unexpected compact cache format")
        bank = z["bank"]
        refs = z["refs"]
        if bank.dtype != np.uint8 or refs.shape != (64, 3) or refs.min() < 0 or refs.max() >= len(bank):
            raise RuntimeError("Invalid compact bank/refs")
        return {"pixels": bank[refs], "mask": z["mask"], "meta": z["meta"]}


def dump_compact(path: Path, arrays: dict, identity: dict) -> dict:
    pixels = arrays["pixels"]
    lookup = {}
    bank = []
    refs = []
    for plane in pixels.reshape(-1, 336, 336):
        key = hashlib.sha256(plane.tobytes()).digest()
        if key not in lookup:
            lookup[key] = len(bank)
            bank.append(plane)
        elif not np.array_equal(plane, bank[lookup[key]]):
            raise RuntimeError("SHA collision in compact bank")
        refs.append(lookup[key])
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("wb") as f:
        np.savez_compressed(
            f,
            format=np.asarray("native64-token-bank-v1"),
            bank=np.stack(bank),
            refs=np.asarray(refs, np.int16).reshape(64, 3),
            mask=arrays["mask"],
            meta=arrays["meta"],
            **{name: np.asarray(value) for name, value in identity.items()},
        )
    restored = load_compact(tmp)
    hashes = array_hashes(arrays)
    if hashes != array_hashes(restored):
        raise RuntimeError("Lossless compact roundtrip failed")
    tmp.replace(path)
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "unique_frames": len(bank),
        "array_sha256": hashes,
        "lossless_roundtrip": True,
    }


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def find_asset_root() -> Path:
    override = os.environ.get("V83_ASSET_ROOT")
    if override:
        p = Path(override)
        if (p / "ASSET_MANIFEST.json").is_file():
            return p
        raise FileNotFoundError(p / "ASSET_MANIFEST.json")
    root = Path("/kaggle/input")
    candidates = [
        root / "rsna-v83-cache-assets-20260915" / "v83_cache_assets_20260915",
        root / "rsna-v83-cache-assets-20260915",
    ]
    for depth in range(1, 5):
        pattern = "/".join(["*"] * depth + ["ASSET_MANIFEST.json"])
        candidates.extend(path.parent for path in root.glob(pattern))
    found = sorted({p.resolve() for p in candidates if (p / "ASSET_MANIFEST.json").is_file()})
    if len(found) != 1:
        raise RuntimeError(f"Expected one asset root, found {found}")
    return found[0]


def find_competition_root() -> Path:
    override = os.environ.get("V83_COMPETITION_ROOT")
    candidates = [Path(override)] if override else [
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
    ]
    found = [p for p in candidates if p and (p / "train_series.csv").is_file() and (p / "train_series").is_dir()]
    if len(found) != 1:
        raise RuntimeError(f"Expected one competition root, found {found}")
    return found[0]


def parse_pose(value: str):
    value = str(value).strip()
    if not value or value.lower() == "nan":
        return None
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"Invalid pose: {value}")
    return parsed


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes"}


def env_uids(name: str):
    raw = os.environ.get(name, "").strip()
    return [x.strip() for x in raw.split(",") if x.strip()] if raw else None


started = time.monotonic()
asset = find_asset_root()
competition = find_competition_root()
images = competition / "train_series"
output = Path(os.environ.get("V83_OUTPUT_ROOT", "/kaggle/working/global_aux60_v83_cache"))
(output / "train").mkdir(parents=True, exist_ok=True)
(output / "gold58").mkdir(parents=True, exist_ok=True)

# Verify the immutable asset packet before importing its runtime.
asset_manifest = json.loads((asset / "ASSET_MANIFEST.json").read_text())
for rec in asset_manifest["files"]:
    p = asset / rec["path"]
    if not p.is_file() or p.stat().st_size != rec["bytes"] or sha256_file(p) != rec["sha256"]:
        raise RuntimeError(f"Asset mismatch: {rec['path']}")

runtime = asset / "orientation_runtime/runtime"
sys.path[:0] = [str(runtime), str(runtime / "rsna_knee")]
from raw_inputs import DecodedStudy, SLOTS  # noqa: E402
from resolver import LateralityResolver  # noqa: E402
from rsna_knee.grayscale_cache import select_series_for_cache  # noqa: E402

series = pd.read_csv(competition / "train_series.csv", dtype=str).fillna("")
training = pd.read_csv(asset / "training_overlay_soft_v2/training_manifest4352.csv", dtype=str).fillna("")
operations = pd.read_csv(asset / "training_overlay_soft_v2/series_operations.csv", dtype=str).fillna("")
smoke = pd.read_csv(asset / "manifests/smoke_train_parents200.csv", dtype=str).fillna("")
parent_shards = pd.read_csv(asset / "manifests/parent_shards16.csv", dtype=str).fillna("")
gold_labels = pd.read_csv(asset / "gold58/gold58_labels.csv", dtype=str).fillna("")
gold_manifest = pd.read_csv(asset / "gold58/gold58_validation_manifest.csv", dtype=str).fillna("")

requested_train = env_uids("V83_TRAIN_UIDS")
shard_raw = os.environ.get("V83_PARENT_SHARD_ID", "").strip()
train_mode = os.environ.get("V83_TRAIN_MODE", "pilot").strip().lower()
if requested_train:
    train_uids = requested_train
elif shard_raw:
    shard_id = int(shard_raw)
    if shard_id not in range(16):
        raise ValueError(f"V83_PARENT_SHARD_ID must be 0..15, got {shard_id}")
    train_uids = parent_shards.loc[
        parent_shards["shard"].astype(int).eq(shard_id), "ParentStudyInstanceUID"
    ].drop_duplicates().tolist()
elif train_mode == "all":
    train_uids = parent_shards["ParentStudyInstanceUID"].drop_duplicates().tolist()
elif train_mode == "none":
    train_uids = []
elif train_mode == "pilot":
    train_uids = smoke["ParentStudyInstanceUID"].drop_duplicates().tolist()
else:
    raise ValueError(f"Unknown V83_TRAIN_MODE={train_mode}")

include_gold = parse_bool(os.environ.get("V83_INCLUDE_GOLD", "1"))
gold_uids = (env_uids("V83_GOLD_UIDS") or gold_manifest["StudyInstanceUID"].drop_duplicates().tolist()) if include_gold else []
train_limit = int(os.environ.get("V83_LIMIT_TRAIN", str(len(train_uids))))
gold_limit = int(os.environ.get("V83_LIMIT_GOLD", str(len(gold_uids))))
train_uids = train_uids[:train_limit]
gold_uids = gold_uids[:gold_limit]

if len(set(train_uids)) != len(train_uids) or len(set(gold_uids)) != len(gold_uids):
    raise RuntimeError("Duplicate parent UID")
if set(train_uids) & set(gold_uids):
    raise RuntimeError("Train/Gold overlap")
if not set(train_uids).issubset(set(training["ParentStudyInstanceUID"])):
    raise RuntimeError("Train UID missing from v8.3 manifest")
if not set(gold_uids).issubset(set(gold_manifest["StudyInstanceUID"])):
    raise RuntimeError("Gold UID missing from Gold58 manifest")
if gold_limit and (gold_manifest["train_mask"].astype(int) != 0).any():
    raise RuntimeError("Gold train_mask is not zero")

series_groups = {uid: g.copy() for uid, g in series.groupby("StudyInstanceUID", sort=False)}
operation_groups = {uid: g.copy() for uid, g in operations.groupby("uid", sort=False)}
training_groups = {uid: g.copy() for uid, g in training.groupby("ParentStudyInstanceUID", sort=False)}


def train_decision(uid: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    manifest_rows = training_groups[uid].copy()
    op_rows = operation_groups[uid].copy()
    routes = manifest_rows["side"].tolist()
    if not routes or any(route not in ("L", "R") for route in routes):
        raise RuntimeError(f"Invalid v8.3 routes for {uid}: {routes}")
    if len(set(routes)) != len(routes):
        raise RuntimeError(f"Duplicate route for {uid}")
    records = []
    for row in op_rows.to_dict("records"):
        records.append({
            "sid": row["sid"],
            "plane": row["plane"],
            "assigned_side": row["assigned_side"],
            "appearance_side": row["appearance_side"],
            "apply_pose": parse_pose(row["apply_pose"]),
            "reverse_depth_extra": parse_bool(row["reverse_depth_extra"]),
        })
    decision = {
        "side": routes[0],
        "bilateral": len(routes) > 1,
        "routes": routes,
        "confidence_tier": manifest_rows["side_tier"].iloc[0],
        "series": records,
    }
    return decision, manifest_rows, op_rows


def selected_rows(uid: str, decision: dict, rows: pd.DataFrame) -> pd.DataFrame:
    selected = set()
    for route in decision["routes"]:
        allowed = {
            rec["sid"] for rec in decision["series"]
            if not decision["bilateral"] or rec["assigned_side"] == route
        }
        candidates = rows[rows["SeriesInstanceUID"].isin(allowed)]
        choices = select_series_for_cache(candidates, images, SLOTS)
        selected.update(choice["series_uid"] for choice in choices if choice is not None)
    result = rows[rows["SeriesInstanceUID"].isin(selected)].copy()
    if result.empty:
        raise RuntimeError(f"No selected H1 series for {uid}")
    return result


def save_sample(namespace: str, uid: str, sample_id: str, side: str, inputs, audit: dict) -> dict:
    pixels_f, mask_t, meta_t = inputs
    valid_pixels = np.rint(pixels_f.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    valid_mask = mask_t.detach().cpu().numpy().astype(bool)
    valid_meta = meta_t.detach().cpu().numpy().astype(np.float32)
    if valid_pixels.ndim != 4 or valid_pixels.shape[1:] != (3, 336, 336):
        raise RuntimeError(f"Unexpected pixel shape {valid_pixels.shape}")
    if len(valid_pixels) > 64 or valid_mask.shape != (len(valid_pixels),) or not valid_mask.all() or valid_meta.shape != (len(valid_pixels), 6):
        raise RuntimeError("Unexpected mask/meta shape")
    # h1_adapter intentionally removes windows rejected for a bilateral route.
    # Restore the fixed training-cache width with masked zero padding.
    pixels = np.zeros((64, 3, 336, 336), np.uint8)
    mask = np.zeros(64, bool)
    meta = np.zeros((64, 6), np.float32)
    pixels[: len(valid_pixels)] = valid_pixels
    mask[: len(valid_pixels)] = True
    meta[: len(valid_pixels)] = valid_meta
    if not np.isfinite(meta).all() or not pixels.any():
        raise RuntimeError("Invalid model-ready arrays")
    key = hashlib.sha256(sample_id.encode()).hexdigest()[:24]
    destination = output / namespace / f"{key}.npz"
    arrays = {"pixels": pixels, "mask": mask, "meta": meta}
    packed = dump_compact(destination, arrays, {
        "study_uid": uid,
        "sample_id": sample_id,
        "side": side,
        "split": "gold58_validation" if namespace == "gold58" else "train",
    })
    return {
        "namespace": namespace,
        "StudyInstanceUID": uid,
        "SampleID": sample_id,
        "side": side,
        "cache_file": str(destination.relative_to(output)),
        "tokens": int(mask.sum()),
        "padded_tokens": int((~mask).sum()),
        "bytes": packed["bytes"],
        "sha256": packed["sha256"],
        "pixels_sha256": packed["array_sha256"]["pixels"],
        "mask_sha256": packed["array_sha256"]["mask"],
        "meta_sha256": packed["array_sha256"]["meta"],
        "unique_frames": packed["unique_frames"],
        "side_tier": audit.get("tier", ""),
        "decoded_source_frames": int(audit.get("decoded_source_frames", 0)),
        "source_errors": len(audit.get("source_errors", [])),
    }


rows_out = []
failures = []
max_output_bytes = int(float(os.environ.get("V83_OUTPUT_HARD_CAP_GB", "18")) * 10**9)
expected_train_samples = int(training[training["ParentStudyInstanceUID"].isin(train_uids)].shape[0])


def progress(phase: str, done: int, total: int) -> None:
    current = sum(r["bytes"] for r in rows_out)
    if current >= max_output_bytes:
        raise RuntimeError(f"Output hard cap reached: {current}")
    live = {
        "status": "RUNNING",
        "phase": phase,
        "done_parents": done,
        "total_parents": total,
        "samples_written": len(rows_out),
        "output_bytes": current,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(output / "LIVE.json", live)
    print(json.dumps(live), flush=True)


try:
    for index, uid in enumerate(train_uids, 1):
        if uid not in series_groups or uid not in operation_groups or uid not in training_groups:
            raise RuntimeError(f"Missing train metadata for {uid}")
        decision, manifest_rows, _ = train_decision(uid)
        rows = selected_rows(uid, decision, series_groups[uid])
        study = DecodedStudy(uid, rows.to_dict("records"), images)
        for route in decision["routes"]:
            match = manifest_rows[manifest_rows["side"].eq(route)]
            if len(match) != 1:
                raise RuntimeError(f"Manifest route mismatch for {uid}/{route}")
            sample_id = match["SampleID"].iloc[0]
            inputs, audit = study.h1_inputs(decision, route)
            rows_out.append(save_sample("train", uid, sample_id, route, inputs, audit))
        del study
        gc.collect()
        progress("train", index, len(train_uids))

    resolver = None
    if gold_uids:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        resolver = LateralityResolver(runtime, device=device)
        print(json.dumps({"gold_laterality_device": device}), flush=True)
    for index, uid in enumerate(gold_uids, 1):
        if uid not in series_groups:
            raise RuntimeError(f"Missing Gold metadata for {uid}")
        study = DecodedStudy(uid, series_groups[uid].to_dict("records"), images)
        decision = study.side_inputs(resolver)
        if not decision.get("routes") or any(route not in ("L", "R") for route in decision["routes"]):
            raise RuntimeError(f"Gold side unresolved for {uid}: {decision}")
        for route in decision["routes"]:
            sample_id = uid if len(decision["routes"]) == 1 else f"{uid}::{route}"
            inputs, audit = study.h1_inputs(decision, route)
            rows_out.append(save_sample("gold58", uid, sample_id, route, inputs, audit))
        del study
        gc.collect()
        progress("gold58", index, len(gold_uids))

    manifest = pd.DataFrame(rows_out)
    manifest.to_csv(output / "CACHE_MANIFEST.csv", index=False)
    training[training["ParentStudyInstanceUID"].isin(train_uids)].to_csv(output / "TRAIN_LABELS.csv", index=False)
    gold_labels[gold_labels["StudyInstanceUID"].isin(gold_uids)].to_csv(output / "GOLD58_LABELS.csv", index=False)
    if manifest[manifest.namespace.eq("train")].SampleID.nunique() != expected_train_samples:
        raise RuntimeError("Train sample cardinality mismatch")
    if manifest[manifest.namespace.eq("gold58")].StudyInstanceUID.nunique() != len(gold_uids):
        raise RuntimeError("Gold parent cardinality mismatch")
    if set(manifest.loc[manifest.namespace.eq("train"), "StudyInstanceUID"]) & set(manifest.loc[manifest.namespace.eq("gold58"), "StudyInstanceUID"]):
        raise RuntimeError("Train/Gold output overlap")
    train_cache_rows = manifest[manifest.namespace.eq("train")]
    projected_bytes = (
        float(train_cache_rows.bytes.mean() * 4352 + manifest[manifest.namespace.eq("gold58")].bytes.sum())
        if len(train_cache_rows) else None
    )
    run_kind = "production_train_shard" if shard_raw else ("gold58_shard" if train_mode == "none" else "pixel_pilot")
    complete = {
        "status": "PASS_PIXEL_CACHE",
        "run_kind": run_kind,
        "parent_shard_id": int(shard_raw) if shard_raw else None,
        "family": "global_aux60_v83",
        "format": "native64-token-bank-v1-npz-deflate",
        "orientation_policy": "knee-laterality-v8.3",
        "orientation_state": "canonical_L_v8_3",
        "same_builder_for_train_and_gold": True,
        "train_parents": len(train_uids),
        "train_samples": int((manifest.namespace == "train").sum()),
        "gold_studies": len(gold_uids),
        "gold_samples": int((manifest.namespace == "gold58").sum()),
        "gold_train_mask": 0,
        "train_gold_overlap": 0,
        "cache_bytes": int(manifest.bytes.sum()),
        "mean_bytes_per_sample": float(manifest.bytes.mean()),
        "projected_4352_train_plus_gold_bytes": projected_bytes,
        "elapsed_seconds": time.monotonic() - started,
        "builder_sha256": (
            os.environ.get("V83_BUILDER_SHA256")
            or sha256_file(Path(globals()["__file__"]))
        ),
        "asset_manifest_sha256": sha256_file(asset / "ASSET_MANIFEST.json"),
        "train_series_sha256": sha256_file(competition / "train_series.csv"),
        "cache_manifest_sha256": sha256_file(output / "CACHE_MANIFEST.csv"),
        "train_labels_sha256": sha256_file(output / "TRAIN_LABELS.csv"),
        "gold_labels_sha256": sha256_file(output / "GOLD58_LABELS.csv"),
        "output_hard_cap_bytes": max_output_bytes,
        "validation_only_gold": True,
        "not_training": True,
    }
    atomic_json(output / "CACHE_COMPLETE.json", complete)
    atomic_json(output / "LIVE.json", {**complete, "phase": "complete"})
    print(json.dumps(complete, indent=2, sort_keys=True), flush=True)
except BaseException as exc:
    atomic_json(output / "FATAL.json", {
        "status": "FAIL",
        "type": type(exc).__name__,
        "message": str(exc),
        "samples_preserved": len(rows_out),
        "elapsed_seconds": time.monotonic() - started,
    })
    raise
