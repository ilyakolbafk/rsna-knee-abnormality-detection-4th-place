"""Isolated K24 base-preserving adaptive extension sampler and metadata8 loader."""
from __future__ import annotations

import csv
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import h1_base as h
from allseries_inputs import metadata8
from aug336 import IDENTITY, series_params, token_keep_mask
from matched_core import ConfidenceBags
from v83_data import _rng


class ExtensionIndex:
    """Every intended missing extension is explicit; an absent file never falls back."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.records = {}
        with (self.root / "TRAIN_EXT_SAMPLE_MAP.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 4352 or len({r["SampleID"] for r in rows}) != 4352:
            raise ValueError("Extension sample map must cover all4352 training SampleIDs")
        excluded = []
        for row in rows:
            sid = row["SampleID"]
            if row["extension_status"] == "excluded_bilateral":
                if row["cache_file"] or "::" not in sid or int(row["windows"]) != 0:
                    raise ValueError("Malformed explicitly excluded bilateral sample")
                self.records[sid] = None
                excluded.append(sid)
            elif row["extension_status"] == "verified_cache":
                self.records[sid] = self._path(row["cache_file"])
            else:
                raise ValueError("Unknown extension status")
        if len(excluded) != 14:
            raise ValueError("Exactly14 bilateral rows have no extension")
        with (self.root / "gold58/EXT_MANIFEST.csv").open() as stream:
            gold = list(csv.DictReader(stream))
        if len(gold) != 58 or len({r["SampleID"] for r in gold}) != 58:
            raise ValueError("Gold extension manifest must cover58 unique studies")
        for row in gold:
            sid = row["SampleID"]
            if sid in self.records:
                raise ValueError("Gold/training extension overlap")
            self.records[sid] = self._path("gold58/" + row["cache_file"])

    def _path(self, relative):
        p = Path(relative)
        if p.is_absolute() or ".." in p.parts or not p.parts:
            raise ValueError("Extension path must be a bounded relative path")
        return self.root / p

    def path(self, sid):
        if sid not in self.records:
            raise ValueError("Sample missing from the exact extension roster: " + sid)
        return self.records[sid]


def checked_extension(path, sample_id):
    """Read compact arrays and validate before gathering any pixels."""
    with np.load(Path(path), allow_pickle=False) as z:
        if str(z["format"]) != "k24-allseries-ext-v1" or str(z["sample_id"]) != sample_id:
            raise ValueError("Extension format/sample identity mismatch")
        if (str(z["orientation_policy"]), str(z["orientation_state"])) != (
            "knee-laterality-v8.3", "canonical_L_v8_3"
        ):
            raise ValueError("Extension orientation mismatch")
        bank, refs = z["bank"], z["refs"]
        mask, meta = z["mask"], z["meta6"]
        series, oblique = z["series_uid"], z["is_oblique"]
        orient, native = z["orient"], z["native_orientation"]
        n = len(mask)
        if bank.dtype != np.uint8 or bank.ndim != 3 or bank.shape[1:] != (336, 336):
            raise ValueError("Invalid extension pixel bank")
        if refs.shape != (n, 3) or not np.issubdtype(refs.dtype, np.integer):
            raise ValueError("Invalid extension references")
        if meta.shape != (n, 6) or not np.isfinite(meta).all():
            raise ValueError("Invalid extension metadata6")
        if mask.shape != (n,) or not np.isin(mask, [False, True]).all():
            raise ValueError("Invalid extension mask")
        if any(v.shape != (n,) for v in [series, oblique, orient, native]):
            raise ValueError("Extension aligned arrays differ")
        if native.any() or not np.isin(orient, ["legacy", "v3"]).all():
            raise ValueError("Unresolved extension orientation")
        if not np.isin(oblique, [0, 1]).all():
            raise ValueError("Invalid is_oblique flag")
        valid = np.flatnonzero(mask)
        if len(valid) > 108:
            raise ValueError("Extension exceeds the frozen108-window eval bound")
        if len(valid):
            _, counts = np.unique(series[valid], return_counts=True)
            if counts.max() > 12 or np.any(series[valid].astype(str) == ""):
                raise ValueError("Extension exceeds12 windows per added series or has no series identity")
        if len(valid) and (refs[valid].min() < 0 or refs[valid].max() >= len(bank)):
            raise ValueError("Extension reference out of bounds")
        return bank, refs, valid, meta.astype(np.float32), series.astype(str), oblique


class AdaptiveBags(ConfidenceBags):
    def __init__(self, *args, ext_index, eval_variant="N0", **kwargs):
        super().__init__(*args, **kwargs)
        if eval_variant not in ("N0", "N1", "N2", "Z"):
            raise ValueError("Unknown frozen evaluation variant")
        if self.train and eval_variant != "N0":
            raise ValueError("Only N0 is a training input")
        self.ext_index, self.eval_variant = ext_index, eval_variant

    def __getitem__(self, index):
        # This getter owns every base draw. Extra RNG must never enter this path.
        x, meta, y, weight, idx, gain, params = super().__getitem__(index)
        base_meta = metadata8(meta.numpy(), False, False)
        sid = str(self.frame.iloc[index].SampleID)
        path = self.ext_index.path(sid)
        if path is not None and self.eval_variant != "N1":
            bank, refs, valid, xm, series, oblique = checked_extension(path, sid)
            chosen = valid
            if self.train and len(chosen):
                present = np.unique(series[chosen])
                rng = _rng(self.seed, self.epoch, sid, "x_series_drop")
                kept = [s for s in present if rng.random() >= self.config.series_drop]
                chosen = chosen[np.isin(series[chosen], kept)]
                count = min(24, round(.375 * len(chosen)))
                chosen = h.random_windows(chosen, count, _rng(self.seed, self.epoch, sid, "x_windows")) if count else np.empty(0, np.int64)
                if len(chosen) and self.config.token_drop > 0:
                    # Same token-drop rule as base: >=1 if E>0. Series-drop may empty extra.
                    chosen = chosen[token_keep_mask(len(chosen), _rng(self.seed, self.epoch, sid, "x_token_drop"), self.config.token_drop)]
            if len(chosen):
                xp = np.tile(IDENTITY, (len(chosen), 1)).astype(np.float32)
                if self.train and self.config.params:
                    per_series = {s: series_params(_rng(self.seed, self.epoch, sid, "x_params_" + s)) for s in np.unique(series[chosen])}
                    for j, token in enumerate(chosen):
                        xp[j] = per_series[series[token]]
                pixels = torch.from_numpy(np.ascontiguousarray(bank[refs[chosen]])).float().div_(255)
                x = torch.cat([x, pixels])
                base_meta = np.concatenate([base_meta, metadata8(xm[chosen], True, oblique[chosen])])
                params = torch.cat([params, torch.from_numpy(xp)])
        if self.eval_variant == "N2":
            base_meta[:, 6:] = 0
        if self.eval_variant == "Z":
            base_meta = base_meta[:, :6].copy()
        return x, torch.from_numpy(base_meta), y, weight, idx, gain, params


def collate(rows):
    batch, tokens = len(rows), max(len(r[0]) for r in rows)
    width = rows[0][1].shape[1]
    if width not in (6, 8) or any(r[1].shape[1] != width for r in rows):
        raise ValueError("Mixed metadata widths")
    x = torch.zeros((batch, tokens, 3, 336, 336))
    meta = torch.zeros((batch, tokens, width))
    mask = torch.zeros((batch, tokens), dtype=torch.bool)
    params = torch.from_numpy(np.tile(IDENTITY, (batch, tokens, 1)))
    for i, (pixels, token_meta, _y, _w, _idx, _gain, token_params) in enumerate(rows):
        n = len(pixels)
        if not n or token_meta.shape != (n, width) or token_params.shape != (n, len(IDENTITY)):
            raise ValueError("Invalid bag tensor shape")
        x[i, :n], meta[i, :n], mask[i, :n], params[i, :n] = pixels, token_meta, True, token_params
    return (x, mask, meta, torch.stack([r[2] for r in rows]), torch.stack([r[3] for r in rows]),
            torch.tensor([r[4] for r in rows]), torch.tensor([r[5] for r in rows]), params)


def loader(dataset, batch, workers, seed, shuffle):
    options = dict(batch_size=batch, shuffle=shuffle, num_workers=workers, collate_fn=collate,
                   pin_memory=True, generator=torch.Generator().manual_seed(seed))
    if workers:
        options.update(prefetch_factor=1, persistent_workers=False)
    return DataLoader(dataset, **options)


def dataset_factory(ext_index, variant="N0"):
    return partial(AdaptiveBags, ext_index=ext_index, eval_variant=variant)
