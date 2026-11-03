#!/usr/bin/env python3
"""Per-token semantic-slot (series) ids for the frozen global AUX60 v8.3 token bank, without touching the cache.

The v8.3 builder (`build_global_pixel_cache.py`, sha 935625ae...) lays the 64 tokens out slot by slot
(SLOTS order: sagittal_fluid, sagittal_structural, coronal_fluid, coronal_structural, axial_fluid), and the
number of tokens per slot depends only on the selected series lengths through
`central_sampling.center_window` + `allocate_centers` (budget 64, equal weights, trims [.1,.1,.1,.1,0]).
Series lengths are known offline (`artifacts/series_presence_matrix_v1/all_series_geom.csv`), so the slot
id of every token is reproducible on CPU and verified per sample against the cache manifest:
    sum(counts) == manifest.tokens   and   sum(series lengths) == manifest.decoded_source_frames.
A sample that fails either check, has decode errors, or is bilateral with an ambiguous route is marked
`unmapped` and the training loader treats it as a single series (parameters shared over the whole study,
no series dropout). Nothing here reads pixels. Written by Claude 2026-09-18 (core v2, arm A).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

SLOT_NAMES = ["sagittal_fluid", "sagittal_structural", "coronal_fluid", "coronal_structural", "axial_fluid"]
SLOT_TYPES = ["Sagittal_fluid", "Sagittal_structural", "Coronal_fluid", "Coronal_structural", "Axial_fluid"]
TRIMS = [0.1, 0.1, 0.1, 0.1, 0.0]
BUDGET = 64
MAP_FORMAT = "core-v2-a-token-slot-map-v1"


def center_window(source_count: int, trim_each_side: float) -> tuple[int, int]:
    """Verbatim `rsna_knee.central_sampling.center_window`."""
    if not math.isfinite(trim_each_side) or not 0 <= trim_each_side < 0.5 or source_count < 1:
        raise ValueError("Trim must be finite in [0,0.5); source count must be positive")
    cut = int(math.floor(source_count * trim_each_side))
    return cut, source_count - cut


def allocate_centers(capacities, budget: int, weights=None) -> np.ndarray:
    """Verbatim `rsna_knee.central_sampling.allocate_centers` (capped weighted fairness)."""
    cap = np.asarray(capacities, dtype=int)
    w = np.ones(len(cap)) if weights is None else np.asarray(weights, dtype=float)
    if cap.ndim != 1 or np.any(cap < 0) or w.shape != cap.shape or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("Invalid slot capacities/weights")
    if budget < int((cap > 0).sum()):
        raise ValueError("Token budget must cover every available semantic slot")
    result = (cap > 0).astype(int)
    while result.sum() < min(budget, cap.sum()):
        cost = (result + 1) / w
        cost[result >= cap] = np.inf
        result[int(np.argmin(cost))] += 1
    return result


def slot_counts(series_lengths) -> np.ndarray:
    """Tokens per slot for five series lengths (0 = slot absent). Decode-valid assumed for every slice."""
    capacities = []
    for slot, count in enumerate(series_lengths):
        if count <= 0:
            capacities.append(0)
            continue
        lo, hi = center_window(int(count), TRIMS[slot])
        capacities.append(hi - lo)
    return allocate_centers(capacities, BUDGET, [1.0] * 5)


def slot_ids_from_counts(counts) -> np.ndarray:
    """int64[64]: slot id of each token in cache order, -1 for the zero-padded tail."""
    counts = np.asarray(counts, dtype=int)
    ids = np.full(BUDGET, -1, np.int64)
    at = 0
    for slot, count in enumerate(counts):
        ids[at : at + count] = slot
        at += count
    if at > BUDGET:
        raise ValueError("Slot counts exceed the 64-token budget")
    return ids


def select_series(rows: pd.DataFrame) -> list[tuple[str, int] | None]:
    """Longest series per slot type, UID as the deterministic tie-break (`select_series_for_cache`)."""
    chosen = []
    for slot_type in SLOT_TYPES:
        candidates = rows.loc[rows["type"].eq(slot_type), ["sid", "cnt"]]
        if candidates.empty:
            chosen.append(None)
            continue
        ranked = sorted(((int(c), str(s)) for s, c in zip(candidates.sid, candidates.cnt)), key=lambda v: (-v[0], v[1]))
        chosen.append((ranked[0][1], ranked[0][0]))
    return chosen


def build_map(geom: pd.DataFrame, operations: pd.DataFrame, manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    geom = geom.copy()
    geom["sid"] = geom["sid"].astype(str)
    geom["uid"] = geom["uid"].astype(str)
    side_of = {(str(u), str(s)): str(a) for u, s, a in zip(operations.uid, operations.sid, operations.assigned_side)}
    by_uid = {uid: g for uid, g in geom.groupby("uid", sort=False)}
    bilateral_parents = set(manifest.loc[manifest.StudyInstanceUID.duplicated(keep=False), "StudyInstanceUID"].astype(str))
    records = []
    for row in manifest.itertuples(index=False):
        uid, sample_id, side = str(row.StudyInstanceUID), str(row.SampleID), str(row.side)
        record = {"SampleID": sample_id, "StudyInstanceUID": uid, "side": side, "tokens": int(row.tokens),
                  "decoded_source_frames": int(row.decoded_source_frames), "source_errors": int(row.source_errors),
                  "bilateral": uid in bilateral_parents, "status": "mapped", "reason": ""}
        rows = by_uid.get(uid)
        if rows is None:
            record.update(status="unmapped", reason="no_series_geometry")
            records.append({**record, **{f"count_{i}": 0 for i in range(5)}, **{f"sid_{i}": "" for i in range(5)}})
            continue
        if record["bilateral"]:
            keep = [side_of.get((uid, sid)) == side for sid in rows.sid]
            rows = rows.loc[keep]
        chosen = select_series(rows)
        lengths = [c[1] if c else 0 for c in chosen]
        counts = slot_counts(lengths)
        for i in range(5):
            record[f"count_{i}"] = int(counts[i])
            record[f"sid_{i}"] = chosen[i][0] if chosen[i] else ""
        if not record["bilateral"]:
            listed = rows.loc[rows.slot.notna()].set_index("slot").sid.to_dict()
            for name, choice in zip(SLOT_NAMES, chosen):
                if listed.get(name, "") != (choice[0] if choice else ""):
                    record.update(status="unmapped", reason="slot_selection_disagrees_with_geometry_table")
        if record["source_errors"] > 0:
            record.update(status="unmapped", reason="decode_errors")
        elif int(counts.sum()) != record["tokens"]:
            record.update(status="unmapped", reason="token_count_mismatch")
        elif not record["bilateral"] and sum(lengths) != record["decoded_source_frames"]:
            record.update(status="unmapped", reason="decoded_frame_count_mismatch")
        records.append(record)
    table = pd.DataFrame(records)
    summary = {
        "format": MAP_FORMAT,
        "samples": int(len(table)),
        "mapped": int(table.status.eq("mapped").sum()),
        "unmapped": int(table.status.eq("unmapped").sum()),
        "unmapped_reasons": table.loc[table.status.eq("unmapped"), "reason"].value_counts().to_dict(),
        "bilateral_samples": int(table.bilateral.sum()),
        "tokens_per_slot_mean": {SLOT_NAMES[i]: float(table[f"count_{i}"].mean()) for i in range(5)},
        "studies_with_all_five_slots": int((table[[f"count_{i}" for i in range(5)]] > 0).all(axis=1).sum()),
    }
    return table, summary


class TokenSlotMap:
    """Runtime lookup used by the training loader."""

    def __init__(self, path: str | Path):
        table = pd.read_csv(path, dtype={"SampleID": str})
        self.counts = {}
        self.unmapped = set()
        for row in table.itertuples(index=False):
            if row.status == "mapped":
                self.counts[str(row.SampleID)] = np.asarray([getattr(row, f"count_{i}") for i in range(5)], np.int64)
            else:
                self.unmapped.add(str(row.SampleID))

    def slot_ids(self, sample_id: str, valid_tokens: int) -> np.ndarray:
        """Slot id per token for the first `valid_tokens` tokens; all zeros (one shared series) when unmapped."""
        counts = self.counts.get(sample_id)
        if counts is None:
            if sample_id not in self.unmapped:
                raise KeyError(f"Sample missing from the token slot map: {sample_id}")
            return np.zeros(valid_tokens, np.int64)
        if int(counts.sum()) != valid_tokens:
            raise ValueError(f"Token slot map disagrees with cache mask for {sample_id}: {counts.sum()} != {valid_tokens}")
        return slot_ids_from_counts(counts)[:valid_tokens]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", required=True, help="artifacts/series_presence_matrix_v1/all_series_geom.csv")
    parser.add_argument("--operations", required=True, help="training_overlay_soft_v2/series_operations.csv")
    parser.add_argument("--manifest", required=True, help="v8.3 TRAIN_MANIFEST.csv (4,352 rows)")
    parser.add_argument("--output", required=True, help="TOKEN_SLOT_MAP.csv; a RECEIPT json is written beside it")
    parser.add_argument("--max-unmapped-fraction", type=float, default=0.02)
    args = parser.parse_args()
    geom = pd.read_csv(args.geometry, dtype={"uid": str, "sid": str})
    operations = pd.read_csv(args.operations, dtype=str)
    manifest = pd.read_csv(args.manifest, dtype={"SampleID": str, "StudyInstanceUID": str})
    table, summary = build_map(geom, operations, manifest)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    fraction = summary["unmapped"] / max(summary["samples"], 1)
    summary["unmapped_fraction"] = fraction
    summary["status"] = "PASS_TOKEN_SLOT_MAP" if fraction <= args.max_unmapped_fraction else "FAIL_TOKEN_SLOT_MAP"
    receipt = out.with_name(out.stem + "_RECEIPT.json")
    receipt.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["status"].startswith("PASS") else 1)


if __name__ == "__main__":
    main()
