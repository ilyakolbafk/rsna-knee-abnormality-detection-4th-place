"""YAML configuration loading and lightweight validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import re

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    for section in ("data", "model", "train"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Missing mapping section: {section}")
    data = config["data"]
    if int(data.get("image_size", 0)) <= 0:
        raise ValueError("data.image_size must be positive")
    cache_format = str(data.get("cache_format", "legacy_triplets"))
    if cache_format not in {"legacy_triplets", "grayscale_shards_v1", "full_source_bank_v1"}:
        raise ValueError(
            "Unknown data.cache_format"
        )
    if cache_format == "full_source_bank_v1":
        from .central_sampling import center_window

        if data.get('centering_mode', 'geometric') != 'geometric' or int(data.get('source_image_size', 0)) != 336:
            raise ValueError('Full-source pixels are already geometric crops at source_image_size336')
        if not data.get('cache_index') or data.get('channel_policy') != 'native_source_neighbors':
            raise ValueError('Full-source inputs require cache_index and native_source_neighbors')
        if not 0 < int(data['image_size']) <= 336:
            raise ValueError('Full-source image_size must be at most 336')
        budget = int(data.get('token_budget', 0))
        if budget < len(data.get('slots', [])) or budget < 1:
            raise ValueError('Full-source budget must cover all configured slots')
        if int(config['train'].get('max_train_tokens', budget)) != budget:
            raise ValueError('Full-source token_budget and max_train_tokens must agree')
        trims = data.get('trim_each_side', .1)
        if isinstance(trims, list):
            if len(trims) != len(data.get('slots', [])):
                raise ValueError('One trim fraction per slot required')
        else:
            trims = [trims]
        for trim in trims:
            center_window(30, float(trim))
        if data.get('center_policy', 'central_uniform') not in {'central_uniform', 'legacy_centers'}:
            raise ValueError('Unknown full-source center policy')
    if cache_format == "legacy_triplets" and int(
        data.get("slices_per_series", 0)
    ) <= 0:
        raise ValueError("data.slices_per_series must be positive")
    if cache_format == "grayscale_shards_v1":
        roots = data.get("cache_roots")
        if isinstance(roots, str):
            roots = [roots]
        if not isinstance(roots, list) or not roots:
            raise ValueError(
                "data.cache_roots must list at least one grayscale shard root"
            )
        source_size = int(data.get("source_image_size", 0))
        if source_size <= 0:
            raise ValueError("data.source_image_size must be positive")
        if int(data["image_size"]) > source_size:
            raise ValueError(
                "data.image_size cannot exceed data.source_image_size for a "
                "grayscale shard cache"
            )
        if str(data.get("channel_policy", "adjacent_sampled_planes")) != (
            "adjacent_sampled_planes"
        ):
            raise ValueError(
                "The current grayscale shard reader supports only "
                "data.channel_policy='adjacent_sampled_planes'"
            )
    centering_mode = str(data.get("centering_mode", "geometric"))
    if centering_mode not in {
        "geometric",
        "joint_line_v1",
        "joint_line_full_v1",
        "coronal_first_v1",
        "coronal_first_superior16p4_v1",
        "coronal_first_superior20_v1",
    }:
        raise ValueError(f"Unknown data.centering_mode: {centering_mode}")
    if centering_mode in {
        "coronal_first_v1",
        "coronal_first_superior16p4_v1",
        "coronal_first_superior20_v1",
    }:
        if cache_format != "legacy_triplets":
            raise ValueError(
                f"{centering_mode} currently requires the manifest-aware "
                "legacy_triplets cache builder"
            )
        if not str(data.get("centering_manifest", "")):
            raise ValueError(
                f"{centering_mode} requires data.centering_manifest"
            )
        manifest_sha256 = str(data.get("centering_manifest_sha256", ""))
        if re.fullmatch(r"[0-9a-f]{64}", manifest_sha256) is None:
            raise ValueError(
                f"{centering_mode} requires a lowercase 64-hex "
                "data.centering_manifest_sha256"
            )
        if data.get("fixed_inferior_prior_mm") is not None:
            raise ValueError(f"{centering_mode} forbids a fixed inferior prior")
    elif "centering_manifest" in data or "centering_manifest_sha256" in data:
        raise ValueError(
            "Centering manifest fields are valid only for manifest-aware centering"
        )
    slots = data.get("slots")
    if not isinstance(slots, list) or not slots:
        raise ValueError("data.slots must be a non-empty list")
    names = [slot.get("name") for slot in slots]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("Every data slot needs a unique name")
    if cache_format == "grayscale_shards_v1" and any(
        int(slot.get("slices", 0)) <= 0 for slot in slots
    ):
        raise ValueError("Every grayscale cache slot needs a positive slices value")

    model = config["model"]
    backbone = str(model.get("backbone", "tiny"))
    finetune_mode = str(model.get("finetune_mode", "full"))
    if finetune_mode not in {"head_only", "last_blocks", "full"}:
        raise ValueError(
            "model.finetune_mode must be 'head_only', 'last_blocks', or 'full'"
        )
    if backbone == "tiny" and finetune_mode != "full":
        raise ValueError("The tiny backbone supports only model.finetune_mode='full'")
    last_blocks = int(model.get("unfreeze_last_blocks", 0))
    if finetune_mode == "last_blocks" and last_blocks <= 0:
        raise ValueError(
            "model.unfreeze_last_blocks must be positive for last_blocks fine-tuning"
        )
    input_normalization = str(model.get("input_normalization", "none"))
    if input_normalization not in {"none", "pretrained", "minus_one_one"}:
        raise ValueError(
            "model.input_normalization must be 'none', 'pretrained', or "
            "'minus_one_one'"
        )
    if bool(model.get("pretrained", False)) and model.get("encoder_checkpoint"):
        raise ValueError(
            "Use either model.pretrained=true or model.encoder_checkpoint, not both"
        )
    if "patch14_dinov2" in backbone and int(data["image_size"]) % 14:
        raise ValueError(
            "DINOv2 patch14 requires data.image_size divisible by 14 to avoid "
            "dropping or padding border pixels"
        )

    train = config["train"]
    center_sampling = train.get("center_sampling", "fixed")
    if center_sampling not in {"fixed", "central_stratified_v1"}:
        raise ValueError("Unknown train.center_sampling")
    if center_sampling == "central_stratified_v1":
        if cache_format != "full_source_bank_v1":
            raise ValueError("Dynamic centers require full_source_bank_v1")
        if data.get("center_policy", "central_uniform") != "central_uniform":
            raise ValueError("Dynamic centers require central_uniform evaluation")
        if train.get("data_rng_mode") != "study_epoch_v1":
            raise ValueError("Dynamic centers require study_epoch_v1")
    if "save_epoch_predictions" in train and not isinstance(train["save_epoch_predictions"], bool):
        raise ValueError("train.save_epoch_predictions must be boolean")
    if float(train.get("learning_rate", 0.0)) <= 0:
        raise ValueError("train.learning_rate must be positive")
    if "encoder_learning_rate" in train and float(train["encoder_learning_rate"]) <= 0:
        raise ValueError("train.encoder_learning_rate must be positive")


def with_path_overrides(
    config: dict[str, Any],
    *,
    images_root: str | None = None,
    cache_root: str | None = None,
    cache_roots: list[str] | None = None,
) -> dict[str, Any]:
    result = deepcopy(config)
    if images_root is not None:
        result["data"]["images_root"] = images_root
    if cache_root is not None:
        result["data"]["cache_root"] = cache_root
    if cache_roots is not None:
        result["data"]["cache_roots"] = list(cache_roots)
    return result
