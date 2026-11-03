"""Study dataset backed by the all-study grayscale shard cache."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .constants import ID_COLUMN, TARGET_COLUMNS
from .grayscale_cache import GrayscaleCacheIndex

try:
    import torch
    import torch.nn.functional as torch_functional
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - optional training dependency
    torch = None
    torch_functional = None
    Dataset = object


def assemble_cached_triplets(
    images: np.ndarray,
    valid: np.ndarray,
    offsets: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble previous/centre/next channels within each semantic slot."""

    if images.ndim != 3 or valid.shape != images.shape[:1]:
        raise ValueError("Expected images [T,H,W] and valid [T]")
    tokens = np.zeros((len(images), 3, *images.shape[1:]), dtype=images.dtype)
    token_mask = np.zeros(len(images), dtype=bool)
    slot_ids = np.zeros(len(images), dtype=np.int64)
    for slot_index, (start, end) in enumerate(offsets):
        slot_ids[start:end] = slot_index
        valid_positions = np.flatnonzero(valid[start:end]) + start
        if not len(valid_positions):
            continue
        for rank, centre in enumerate(valid_positions):
            previous = valid_positions[max(0, rank - 1)]
            following = valid_positions[min(len(valid_positions) - 1, rank + 1)]
            tokens[centre, 0] = images[previous]
            tokens[centre, 1] = images[centre]
            tokens[centre, 2] = images[following]
            token_mask[centre] = True
    return tokens, token_mask, slot_ids


class ShardedGrayscaleStudyDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        config: dict[str, Any],
        *,
        training: bool,
        include_labels: bool = True,
    ) -> None:
        if torch is None or torch_functional is None:
            raise RuntimeError("ShardedGrayscaleStudyDataset requires PyTorch")
        self.manifest = manifest.reset_index(drop=True).copy()
        if ID_COLUMN not in self.manifest or self.manifest[ID_COLUMN].duplicated().any():
            raise ValueError(f"Manifest must be unique by {ID_COLUMN}")
        self.config = config
        self.data_config = config["data"]
        self.train_config = config.get("train", {})
        self.training = training
        self.include_labels = include_labels
        roots = self.data_config.get("cache_roots")
        if isinstance(roots, (str, bytes)):
            roots = [roots]
        self.cache = GrayscaleCacheIndex(roots or [])
        missing = [
            uid
            for uid in self.manifest[ID_COLUMN].astype(str)
            if uid not in self.cache
        ]
        if missing:
            raise ValueError(
                f"Grayscale cache is missing {len(missing)} manifest studies; "
                f"first={missing[0]}"
            )
        configured_slots = self.data_config["slots"]
        configured_signature = [
            (str(slot["name"]), int(slot.get("slices", 0)))
            for slot in configured_slots
        ]
        cached_signature = [
            (str(slot["name"]), int(slot["slices"])) for slot in self.cache.slots
        ]
        if configured_signature != cached_signature:
            raise ValueError(
                "Configured slots do not match the frozen grayscale cache: "
                f"{configured_signature} != {cached_signature}"
            )
        self.token_count = self.cache.token_count
        self.image_size = int(self.data_config["image_size"])
        if self.image_size > self.cache.source_image_size:
            raise ValueError(
                "A grayscale cache may be downsampled but not upsampled beyond its "
                f"source size {self.cache.source_image_size}"
            )

    def __len__(self) -> int:
        return len(self.manifest)

    def _tokens(self, study_uid: str):
        cached = self.cache.load_study(study_uid)
        # Each grayscale plane participates in up to three neighbouring tokens.
        # Resize it once before assembling channels; interpolation is independent
        # per plane/channel. float() owns its storage, leaving the mmap unchanged.
        planes = torch.from_numpy(cached["images"].copy()).float().div_(255.0)
        if self.image_size != self.cache.source_image_size:
            planes = torch_functional.interpolate(
                planes[:, None],
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )[:, 0]
        triplets, token_mask, slot_ids = assemble_cached_triplets(
            planes.numpy(), cached["valid"].astype(bool), self.cache.offsets
        )
        tokens = torch.from_numpy(triplets)
        return tokens, token_mask, slot_ids

    def _subsample_tokens(self, token_mask: np.ndarray, slot_ids: np.ndarray) -> np.ndarray:
        """Randomly keep at most ``train.max_train_tokens`` valid tokens.

        Sampling is stratified per slot (proportional to each slot's valid
        count, at least one token per non-empty slot) so a study never loses a
        whole plane.  Acts as a strong, cheap augmentation and cuts the encoder
        cost per step; evaluation always uses every valid token.
        """

        max_tokens = int(self.train_config.get("max_train_tokens", 0))
        valid_positions = np.flatnonzero(token_mask)
        if max_tokens <= 0 or len(valid_positions) <= max_tokens:
            return token_mask
        keep = np.zeros_like(token_mask)
        slots = np.unique(slot_ids[valid_positions])
        counts = {int(s): int((slot_ids[valid_positions] == s).sum()) for s in slots}
        total = sum(counts.values())
        budget = {s: max(1, int(round(max_tokens * c / total))) for s, c in counts.items()}
        # Fix rounding so the total equals max_tokens exactly.
        while sum(budget.values()) > max_tokens:
            largest = max(budget, key=lambda s: budget[s])
            if budget[largest] <= 1:
                break
            budget[largest] -= 1
        while sum(budget.values()) < max_tokens:
            room = {s: counts[s] - budget[s] for s in budget if counts[s] > budget[s]}
            if not room:
                break
            budget[max(room, key=room.get)] += 1
        for s, n in budget.items():
            candidates = valid_positions[slot_ids[valid_positions] == s]
            chosen = torch.randperm(len(candidates))[:n].numpy()
            keep[candidates[chosen]] = True
        return keep

    def _random_resized_crop(self, tokens):
        """Study-consistent random zoom-crop (same window for every token)."""

        scale_min = float(self.train_config.get("random_crop_scale_min", 1.0))
        if scale_min >= 1.0:
            return tokens
        if not 0.0 < scale_min < 1.0:
            raise ValueError("train.random_crop_scale_min must lie in (0, 1]")
        size = tokens.shape[-1]
        scale = scale_min + float(torch.rand(())) * (1.0 - scale_min)
        crop = max(8, int(round(size * scale)))
        top = int(torch.randint(0, size - crop + 1, ()))
        left = int(torch.randint(0, size - crop + 1, ()))
        window = tokens[:, :, top : top + crop, left : left + crop]
        return torch_functional.interpolate(
            window, size=(size, size), mode="bilinear", align_corners=False, antialias=False
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.manifest.iloc[index]
        uid = str(row[ID_COLUMN])
        inputs = self._tokens(uid)
        tokens, token_mask, slot_ids = inputs[:3]
        if self.training:
            token_mask = self._subsample_tokens(token_mask, slot_ids)
            flip_probability = float(
                self.train_config.get("horizontal_flip_probability", 0.0)
            )
            if torch.rand(()) < flip_probability:
                tokens = tokens.flip(-1)
            tokens = self._random_resized_crop(tokens)
            jitter = float(self.train_config.get("intensity_jitter", 0.0))
            if jitter > 0:
                scale = 1.0 + (torch.rand(()) * 2.0 - 1.0) * jitter
                shift = (torch.rand(()) * 2.0 - 1.0) * jitter
                tokens = (tokens * scale + shift).clamp(0.0, 1.0)

        item: dict[str, Any] = {
            "uid": uid,
            "tokens": tokens,
            "token_mask": torch.from_numpy(token_mask),
            "slot_ids": torch.from_numpy(slot_ids),
        }
        if len(inputs) == 4:
            item['token_features'] = inputs[3]
        if not self.include_labels:
            return item

        labels = pd.to_numeric(row[TARGET_COLUMNS], errors="coerce").to_numpy(float)
        label_mask = np.isfinite(labels)
        for target_index, target in enumerate(TARGET_COLUMNS):
            explicit_mask = row.get(f"label_mask__{target}", True)
            label_mask[target_index] &= bool(explicit_mask) if pd.notna(explicit_mask) else False
        labels = np.nan_to_num(labels, nan=0.0).astype(np.float32)
        is_gold = bool(row.get("is_gold", False))
        if is_gold:
            label_weight = np.full(
                len(TARGET_COLUMNS),
                float(self.train_config.get("gold_weight", 1.0)),
                dtype=np.float32,
            )
        else:
            pseudo_weight = float(self.train_config.get("pseudo_weight", 1.0))
            minimum_confidence = float(
                self.train_config.get("pseudo_min_confidence", 0.0)
            )
            fallback_confidence = pd.to_numeric(
                row.get("pseudo_confidence", 1.0), errors="coerce"
            )
            if not np.isfinite(fallback_confidence):
                fallback_confidence = 0.0
            label_weight = np.asarray(
                [
                    pd.to_numeric(
                        row.get(f"confidence__{target}", fallback_confidence),
                        errors="coerce",
                    )
                    for target in TARGET_COLUMNS
                ],
                dtype=np.float32,
            )
            label_weight[~np.isfinite(label_weight)] = 0.0
            label_weight[label_weight < minimum_confidence] = 0.0
            label_weight = np.clip(label_weight, 0.0, 1.0) * pseudo_weight
            label_mask &= label_weight > 0
        item.update(
            {
                "labels": torch.from_numpy(labels),
                "label_mask": torch.from_numpy(label_mask),
                "label_weight": torch.from_numpy(label_weight),
                "is_gold": torch.tensor(is_gold, dtype=torch.bool),
            }
        )
        return item
