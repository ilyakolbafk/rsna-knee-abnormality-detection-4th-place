"""Study-level fixed-token dataset backed by the series cache."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .cache import load_or_create_series_cache, select_series_for_slots
from .constants import ID_COLUMN, TARGET_COLUMNS

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # Audits and fold generation deliberately work without torch.
    torch = None
    Dataset = object


class StudyDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        series: pd.DataFrame,
        config: dict[str, Any],
        *,
        training: bool,
        include_labels: bool = True,
    ) -> None:
        if torch is None:
            raise RuntimeError("StudyDataset requires PyTorch; install the 'train' extra")
        self.manifest = manifest.reset_index(drop=True).copy()
        self.config = config
        self.data_config = config["data"]
        self.train_config = config.get("train", {})
        self.training = training
        self.include_labels = include_labels
        self.slots = self.data_config["slots"]
        self.series_groups = {
            str(uid): part.copy() for uid, part in series.groupby(ID_COLUMN, sort=False)
        }
        self.empty_series = series.iloc[0:0]
        self.token_count = len(self.slots) * int(
            self.data_config["slices_per_series"]
        )

    def __len__(self) -> int:
        return len(self.manifest)

    def _load_tokens(self, row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        study_uid = str(row[ID_COLUMN])
        selections = select_series_for_slots(
            self.series_groups.get(study_uid, self.empty_series), self.slots
        )
        count = int(self.data_config["slices_per_series"])
        size = int(self.data_config["image_size"])
        all_tokens: list[np.ndarray] = []
        all_masks: list[np.ndarray] = []
        all_slots: list[np.ndarray] = []
        for slot_index, series_uid in enumerate(selections):
            if series_uid is None:
                tokens = np.zeros((count, 3, size, size), dtype=np.float32)
                mask = np.zeros(count, dtype=bool)
            else:
                cached = load_or_create_series_cache(
                    images_root=self.data_config["images_root"],
                    cache_root=self.data_config["cache_root"],
                    study_uid=study_uid,
                    series_uid=series_uid,
                    data_config=self.data_config,
                )
                tokens = np.asarray(cached, dtype=np.float32)
                if cached.dtype == np.uint8:
                    tokens = tokens / 255.0
                if tokens.shape != (count, 3, size, size):
                    raise ValueError(
                        f"Unexpected cached shape for {series_uid}: {tokens.shape}; "
                        f"expected {(count, 3, size, size)}"
                    )
                mask = np.ones(count, dtype=bool)
            all_tokens.append(tokens)
            all_masks.append(mask)
            all_slots.append(np.full(count, slot_index, dtype=np.int64))
        return (
            np.concatenate(all_tokens, axis=0),
            np.concatenate(all_masks, axis=0),
            np.concatenate(all_slots, axis=0),
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.manifest.iloc[index]
        tokens, token_mask, slot_ids = self._load_tokens(row)
        token_tensor = torch.from_numpy(tokens.copy())
        if self.training:
            flip_probability = float(
                self.train_config.get("horizontal_flip_probability", 0.0)
            )
            if torch.rand(()) < flip_probability:
                token_tensor = token_tensor.flip(-1)
            jitter = float(self.train_config.get("intensity_jitter", 0.0))
            if jitter > 0:
                scale = 1.0 + (torch.rand(()) * 2.0 - 1.0) * jitter
                shift = (torch.rand(()) * 2.0 - 1.0) * jitter
                token_tensor = (token_tensor * scale + shift).clamp(0.0, 1.0)

        item: dict[str, Any] = {
            "uid": str(row[ID_COLUMN]),
            "tokens": token_tensor,
            "token_mask": torch.from_numpy(token_mask),
            "slot_ids": torch.from_numpy(slot_ids),
        }
        if self.include_labels:
            labels = pd.to_numeric(row[TARGET_COLUMNS], errors="coerce").to_numpy(float)
            label_mask = np.isfinite(labels)
            for target_index, target in enumerate(TARGET_COLUMNS):
                explicit_mask = row.get(f"label_mask__{target}", True)
                label_mask[target_index] &= (
                    bool(explicit_mask) if pd.notna(explicit_mask) else False
                )
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
                # A low-confidence pseudo-label is an abstention, not a weak
                # negative.  Excluding it from the mask also keeps downstream
                # diagnostics from calling it an observed label.
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


def create_study_dataset(
    manifest: pd.DataFrame,
    series: pd.DataFrame | None,
    config: dict[str, Any],
    *,
    training: bool,
    include_labels: bool = True,
):
    """Construct the legacy or sharded-cache dataset from one config switch."""

    cache_format = str(config["data"].get("cache_format", "legacy_triplets"))
    if cache_format == "full_source_bank_v1":
        from .full_source_dataset import FullSourceStudyDataset

        return FullSourceStudyDataset(manifest, config, training=training,
                                      include_labels=include_labels)
    if cache_format == "grayscale_shards_v1":
        from .sharded_dataset import ShardedGrayscaleStudyDataset

        return ShardedGrayscaleStudyDataset(
            manifest,
            config,
            training=training,
            include_labels=include_labels,
        )
    if cache_format != "legacy_triplets":
        raise ValueError(f"Unknown data.cache_format: {cache_format}")
    if series is None:
        raise ValueError("Legacy triplet datasets require series metadata")
    return StudyDataset(
        manifest,
        series,
        config,
        training=training,
        include_labels=include_labels,
    )
