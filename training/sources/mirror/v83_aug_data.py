"""Core v2 arm A loader: frozen v8.3 token bank + E1-style train-time perturbations, eval path untouched.

Train-time order per sample (all draws seeded by (seed, epoch, sample_id, suffix) exactly like v83_data._rng):
  1. slot ids per token from TOKEN_SLOT_MAP.csv (unmapped studies: one shared series);
  2. series dropout p=0.10 over the present slots, at least one slot kept ("a_series_drop");
  3. the v8.3 draw of 12 windows with repetition from the surviving pool, using the *same* rng stream as
     V83Bags (`_rng(seed, epoch, sample_id)`), so with series dropout off the windows are identical;
  4. position jitter -1/0/+1 slice per drawn window where the bank holds the needed frames ("a_jitter");
  5. token dropout p=0.15 over the drawn windows, at least one kept ("a_token_drop");
  6. one augmentation parameter vector per slot ("a_params_<slot>"), broadcast to its tokens.
With `AugConfig.identity()` every step is a no-op and __getitem__ reproduces V83Bags bit for bit (tested).
Eval (train=False) delegates to V83Bags and appends IDENTITY parameters. Written by Claude 2026-09-18.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import h1_base as h
from aug336 import IDENTITY, jitter_refs, series_drop_mask, series_params, token_keep_mask
from token_slot_map import TokenSlotMap
from v83_data import V83Bags, _rng
from mirror_tokens import MirrorPlaneMap, mirror_draw, reflect_tokens

FORMAT = "native64-token-bank-v1"
WINDOWS = 24


@dataclass(frozen=True)
class AugConfig:
    series_drop: float = 0.10
    token_drop: float = 0.15
    jitter: int = 1
    params: bool = True

    @classmethod
    def identity(cls) -> "AugConfig":
        return cls(series_drop=0.0, token_drop=0.0, jitter=0, params=False)

    def as_dict(self) -> dict:
        return {"series_drop": self.series_drop, "token_drop": self.token_drop, "position_jitter_slices": self.jitter,
                "series_parameters": self.params}


class V83AugBags(V83Bags):
    def __init__(self, frame, aux_names, *, train=False, seed=20260908, epoch=0, slot_map: TokenSlotMap | None = None,
                 config: AugConfig | None = None):
        super().__init__(frame, aux_names, train=train, seed=seed, epoch=epoch)
        self.slot_map = slot_map
        self.config = config or AugConfig()
        self.mirror_map = MirrorPlaneMap(Path(__file__).parent / "assets/FULLFIT_TOKEN_SLOT_MAP.csv") if self.train else None
        self.force_mirror = None
        if self.train and self.slot_map is None:
            raise ValueError("Training needs the token slot map (TOKEN_SLOT_MAP.csv)")

    def __getitem__(self, index):
        if not self.train:
            x, meta, y, weight, idx, gain = super().__getitem__(index)
            return x, meta, y, weight, idx, gain, torch.from_numpy(np.tile(IDENTITY, (len(x), 1)))
        cfg = self.config
        row = self.frame.iloc[index]
        sample_id = str(row.SampleID)
        path = Path(row.cache_path)
        with np.load(path, allow_pickle=False) as z:
            if str(z["format"]) != FORMAT:
                raise ValueError(f"Unexpected cache format: {path}")
            if str(z["sample_id"]) != sample_id:
                raise ValueError(f"Sample identity mismatch: {path}")
            bank = z["bank"]
            refs_all = z["refs"]
            mask = z["mask"].astype(bool, copy=False)
            valid = np.flatnonzero(mask)
            if not len(valid):
                raise ValueError(f"No valid tokens: {sample_id}")
            slots = np.full(len(mask), -1, np.int64)
            slots[valid] = self.slot_map.slot_ids(sample_id, len(valid))
            pool = valid
            if cfg.series_drop > 0:
                present = np.unique(slots[valid])
                keep = series_drop_mask(present, _rng(self.seed, self.epoch, sample_id, "a_series_drop"), cfg.series_drop)
                kept_slots = set(present[keep].tolist())
                pool = valid[np.isin(slots[valid], list(kept_slots))]
            chosen = h.random_windows(pool, WINDOWS, _rng(self.seed, self.epoch, sample_id))
            if cfg.jitter > 0:
                zero_frame = ~bank.reshape(len(bank), -1).any(axis=1)
                refs, _applied = jitter_refs(refs_all, mask, zero_frame, chosen, _rng(self.seed, self.epoch, sample_id, "a_jitter"), cfg.jitter)
            else:
                refs = refs_all[chosen].astype(np.int64)
            if cfg.token_drop > 0:
                keep = token_keep_mask(len(chosen), _rng(self.seed, self.epoch, sample_id, "a_token_drop"), cfg.token_drop)
                chosen, refs = chosen[keep], refs[keep]
            if int(refs.max()) >= len(bank) or int(refs.min()) < 0:
                raise ValueError(f"Out-of-bounds frame reference: {sample_id}")
            pixels = bank[refs]
            meta = z["meta"][chosen]
            all_planes = np.full(len(mask), -1, np.int64)
            all_planes[valid] = self.mirror_map.plane_ids(sample_id, len(valid))
            planes = all_planes[chosen]
            do_mirror = mirror_draw(self.seed, self.epoch, sample_id) if self.force_mirror is None else bool(self.force_mirror)
            if do_mirror:
                pixels, meta = reflect_tokens(pixels, meta, planes)
            params = np.tile(IDENTITY, (len(chosen), 1)).astype(np.float32)
            if cfg.params:
                per_slot = {int(s): series_params(_rng(self.seed, self.epoch, sample_id, f"a_params_{int(s)}")) for s in np.unique(slots[chosen])}
                for i, token in enumerate(chosen):
                    params[i] = per_slot[int(slots[token])]
        x = torch.from_numpy(np.ascontiguousarray(pixels)).float().div_(255)
        gain = 1 + (_rng(self.seed, self.epoch, sample_id, "gain").random() - 0.5) * 0.2
        return (x, torch.from_numpy(np.asarray(meta, np.float32)), torch.from_numpy(self.y[index]),
                torch.from_numpy(self.label_weight[index]), index, gain, torch.from_numpy(params))


def collate(rows):
    """h1_base.collate plus a padded [B,K,13] parameter tensor (IDENTITY on padding)."""
    batch = len(rows)
    tokens = max(len(row[0]) for row in rows)
    x = torch.zeros((batch, tokens, 3, 336, 336))
    meta = torch.zeros((batch, tokens, 6))
    mask = torch.zeros((batch, tokens), dtype=torch.bool)
    params = torch.from_numpy(np.tile(IDENTITY, (batch, tokens, 1)))
    for i, (pixels, token_meta, _y, _w, _idx, _gain, token_params) in enumerate(rows):
        n = len(pixels)
        x[i, :n] = pixels
        meta[i, :n] = token_meta
        mask[i, :n] = True
        params[i, :n] = token_params
    return (x, mask, meta, torch.stack([row[2] for row in rows]), torch.stack([row[3] for row in rows]),
            torch.tensor([row[4] for row in rows]), torch.tensor([row[5] for row in rows]), params)


def loader(dataset, batch, workers, seed, shuffle):
    from torch.utils.data import DataLoader
    options = dict(batch_size=batch, shuffle=shuffle, num_workers=workers, collate_fn=collate, pin_memory=True,
                   generator=torch.Generator().manual_seed(seed))
    if workers:
        options.update(prefetch_factor=1, persistent_workers=False)
    return DataLoader(dataset, **options)
