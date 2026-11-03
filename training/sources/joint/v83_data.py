"""Load the verified knee-laterality-v8.3 token bank and evaluate at fixed grains."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset

import h1_base as h


def _rng(seed: int, epoch: int, sample_id: str, suffix: str = "") -> random.Random:
    payload = [seed, epoch, sample_id]
    if suffix:
        payload.append(suffix)
    return random.Random(int(h.semantic(payload)[:16], 16))


def _main_arrays(frame: pd.DataFrame):
    y = frame[h.TARGETS].to_numpy(np.float32)
    mask = frame[["label_mask__" + t for t in h.TARGETS]].to_numpy(bool, copy=True)
    mask &= np.isfinite(y) & (y >= 0) & (y <= 1)
    return np.nan_to_num(y), mask


class V83Bags(Dataset):
    """One row per v8.3 SampleID, including the two rows of a bilateral parent."""

    def __init__(self, frame, aux_names, *, train=False, seed=20260908, epoch=0):
        self.frame = frame.reset_index(drop=True).copy()
        self.aux_names = list(aux_names)
        self.train = bool(train)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.y, main_mask = _main_arrays(self.frame)
        if self.aux_names:
            ay = self.frame[["aux__" + n for n in self.aux_names]].to_numpy(np.float32)
            conf = self.frame[["confidence__aux__" + n for n in self.aux_names]].to_numpy(np.float32)
            mask = self.frame[["mask__aux__" + n for n in self.aux_names]].to_numpy(np.float32)
            valid = np.isfinite(ay) & (ay >= 0) & (ay <= 1) & (conf > 0) & (mask > 0)
            self.y = np.concatenate([self.y, np.where(valid, ay, 0)], axis=1)
            self.label_weight = np.concatenate(
                [main_mask.astype(np.float32), np.where(valid, conf * mask, 0)], axis=1
            )
        else:
            self.label_weight = main_mask.astype(np.float32)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        sample_id = str(row.SampleID)
        path = Path(row.cache_path)
        with np.load(path, allow_pickle=False) as z:
            if str(z["format"]) != "native64-token-bank-v1":
                raise ValueError(f"Unexpected cache format: {path}")
            if str(z["sample_id"]) != sample_id:
                raise ValueError(f"Sample identity mismatch: {path}")
            bank = z["bank"]
            refs_all = z["refs"]
            valid = np.flatnonzero(z["mask"])
            if not len(valid):
                raise ValueError(f"No valid tokens: {sample_id}")
            chosen = h.random_windows(valid, 12, _rng(self.seed, self.epoch, sample_id)) if self.train else valid
            refs = refs_all[chosen]
            pixels = np.zeros((*refs.shape, 336, 336), np.uint8)
            ok = refs >= 0
            if ok.any():
                if int(refs[ok].max()) >= len(bank):
                    raise ValueError(f"Out-of-bounds frame reference: {sample_id}")
                pixels[ok] = bank[refs[ok]]
            meta = z["meta"][chosen]
        x = torch.from_numpy(np.ascontiguousarray(pixels)).float().div_(255)
        gain = 1 + (_rng(self.seed, self.epoch, sample_id, "gain").random() - .5) * .2 if self.train else 1.
        return (
            x,
            torch.from_numpy(np.asarray(meta, np.float32)),
            torch.from_numpy(self.y[index]),
            torch.from_numpy(self.label_weight[index]),
            index,
            gain,
        )


def attach_cache_paths(labels: pd.DataFrame, manifest: pd.DataFrame, root: Path) -> pd.DataFrame:
    manifest = manifest.copy()
    manifest["cache_path"] = manifest.cache_file.map(lambda p: str(root / p))
    cols = ["SampleID", "cache_path", "sha256", "tokens", "padded_tokens", "side_tier"]
    out = labels.merge(manifest[cols], on="SampleID", how="inner", validate="one_to_one")
    if len(out) != len(labels):
        missing = sorted(set(labels.SampleID) - set(out.SampleID))[:5]
        raise ValueError(f"Missing cache samples: {missing}")
    if not ((out.tokens + out.padded_tokens) == 64).all():
        raise ValueError("Every v8.3 sample must contain exactly 64 token slots")
    return out


def _auc(labels, predictions):
    labels = np.asarray(labels, int)
    predictions = np.asarray(predictions, float)
    if len(labels) == 0 or len(np.unique(labels)) != 2:
        return None
    return float(roc_auc_score(labels, predictions))


def pseudo_auc(frame: pd.DataFrame, predictions: np.ndarray, group_key: str | None):
    """Main12 pseudo AUC; grouped modes max predictions and positive labels."""
    y, mask = _main_arrays(frame)
    confidence = frame[["confidence__" + t for t in h.TARGETS]].to_numpy(float)
    usable = mask & (confidence >= .5) & (np.abs(y - .5) > 0)
    scores, counts = {}, {}
    for j, target in enumerate(h.TARGETS):
        if group_key is None:
            use = usable[:, j]
            labels = (y[use, j] >= .5).astype(int)
            pred = predictions[use, j]
        else:
            rows = pd.DataFrame(
                {
                    "group": frame[group_key].astype(str),
                    "prediction": predictions[:, j],
                    "label": np.where(usable[:, j], (y[:, j] >= .5).astype(float), np.nan),
                }
            )
            grouped = rows.groupby("group", sort=False).agg(prediction=("prediction", "max"), label=("label", "max"))
            grouped = grouped[grouped.label.notna()]
            labels = grouped.label.to_numpy(int)
            pred = grouped.prediction.to_numpy(float)
        scores[target] = _auc(labels, pred)
        counts[target] = {"n": int(len(labels)), "positive": int(labels.sum())}
    if any(v is None for v in scores.values()):
        raise ValueError(f"All 12 pseudo targets must be evaluable at {group_key or 'sample'} grain")
    return {"macro_auc": float(np.mean(list(scores.values()))), "per_target": scores, "counts": counts}


def gold_auc(gold_labels: pd.DataFrame, predictions: np.ndarray):
    scores, counts = {}, {}
    for j, target in enumerate(h.TARGETS):
        labels = gold_labels[target].to_numpy(int)
        scores[target] = _auc(labels, predictions[:, j])
        counts[target] = {"n": int(len(labels)), "positive": int(labels.sum())}
    if any(v is None for v in scores.values()):
        raise ValueError("All 12 Gold58 targets must be evaluable")
    return {"macro_auc": float(np.mean(list(scores.values()))), "per_target": scores, "counts": counts}


def parent_key(frame: pd.DataFrame) -> str:
    return "ParentStudyInstanceUID" if "ParentStudyInstanceUID" in frame.columns else "StudyInstanceUID"
