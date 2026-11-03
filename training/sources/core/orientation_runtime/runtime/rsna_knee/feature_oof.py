"""One-time frozen image features and fixed-step repeated OOF heads."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .constants import ID_COLUMN, TARGET_COLUMNS
from .dataset import create_study_dataset
from .grayscale_cache import cache_spec, cache_spec_hash


FEATURE_CACHE_FORMAT = "rsna-knee-frozen-features-v1"


def _torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("Frozen feature experiments require PyTorch") from error
    return torch


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _uid_set_sha256(uids: Any) -> str:
    digest = hashlib.sha256()
    for uid in sorted({str(value) for value in uids}):
        digest.update(uid.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _data_identity_hash(data_config: dict[str, Any]) -> str:
    if data_config.get('cache_format') == 'full_source_bank_v1':
        raise ValueError('Full-source central inputs require their explicit training/index path; legacy frozen-feature export is not configured')
    if str(data_config.get("cache_format", "legacy_triplets")) == (
        "grayscale_shards_v1"
    ):
        return cache_spec_hash(cache_spec(data_config))
    identity = {
        "cache_format": "legacy_triplets",
        "image_size": int(data_config["image_size"]),
        "slices_per_series": int(data_config["slices_per_series"]),
        "crop_mm": data_config.get("crop_mm"),
        "percentile_low": data_config.get("percentile_low", 1.0),
        "percentile_high": data_config.get("percentile_high", 99.0),
        "slots": data_config["slots"],
    }
    if "centering_mode" in data_config:
        identity["centering_mode"] = str(data_config["centering_mode"])
    if "centering_manifest_sha256" in data_config:
        identity["centering_manifest_sha256"] = str(
            data_config["centering_manifest_sha256"]
        )
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def resolve_pretrained_weight_file(config: dict[str, Any]) -> Path:
    local_checkpoint = config["model"].get("encoder_checkpoint")
    if local_checkpoint:
        path = Path(local_checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Encoder checkpoint does not exist: {path}")
        return path
    model_name = str(config["model"]["backbone"])
    if not model_name.startswith("timm:"):
        raise ValueError("Frozen pilot requires a timm backbone")
    try:
        import timm
        from huggingface_hub import hf_hub_download
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("timm and huggingface_hub are required") from error
    short_name = model_name.split(":", maxsplit=1)[1]
    model = timm.create_model(
        short_name,
        pretrained=False,
        num_classes=0,
        img_size=int(config["data"]["image_size"]),
    )
    repository = model.pretrained_cfg.get("hf_hub_id")
    if not repository:
        raise RuntimeError(f"Backbone {short_name!r} has no pinned Hugging Face source")
    return Path(hf_hub_download(repository, "model.safetensors"))


def extract_frozen_features(
    config: dict[str, Any],
    manifest: pd.DataFrame,
    series: pd.DataFrame | None,
    output_path: str | Path,
    *,
    device_name: str | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Encode every valid cached token exactly once and save an atomic NPZ."""

    torch = _torch()
    from .model import KneeMILModel

    if manifest[ID_COLUMN].duplicated().any():
        raise ValueError("Feature manifest must be unique by StudyInstanceUID")
    if device_name is None:
        if torch.cuda.is_available():
            device_name = "cuda"
        elif torch.backends.mps.is_available():
            device_name = "mps"
        else:
            device_name = "cpu"
    device = torch.device(device_name)
    weight_file = resolve_pretrained_weight_file(config)
    expected_hash = config["model"].get("weights_sha256")
    actual_hash = _sha256(weight_file)
    if expected_hash and actual_hash != expected_hash:
        raise RuntimeError(
            f"Pretrained weight hash mismatch: {actual_hash}, expected {expected_hash}"
        )

    dataset = create_study_dataset(
        manifest,
        series,
        config,
        training=False,
        include_labels=False,
    )
    model = KneeMILModel(config).to(device).eval()
    encoder = model.encoder
    feature_dim = int(model.feature_dim)
    token_count = int(dataset.token_count)
    features = np.zeros((len(dataset), token_count, feature_dim), dtype=np.float16)
    masks = np.zeros((len(dataset), token_count), dtype=bool)
    slots = np.zeros((len(dataset), token_count), dtype=np.int16)
    uids: list[str] = []
    chunk_size = int(config["model"].get("encoder_chunk_size", 8))
    started = time.monotonic()
    valid_token_count = 0
    with torch.inference_mode():
        for index in range(len(dataset)):
            item = dataset[index]
            tokens = item["tokens"]
            token_mask = item["token_mask"].numpy().astype(bool)
            slot_ids = item["slot_ids"].numpy().astype(np.int16)
            valid_positions = np.flatnonzero(token_mask)
            encoded_parts = []
            for start in range(0, len(valid_positions), chunk_size):
                positions = valid_positions[start : start + chunk_size]
                batch = tokens[positions].to(device, non_blocking=False)
                use_amp = device.type in {"mps", "cuda"}
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    encoded = encoder(batch)
                encoded_parts.append(encoded.float().cpu().numpy())
            if encoded_parts:
                encoded_array = np.concatenate(encoded_parts, axis=0).astype(np.float16)
                features[index, valid_positions] = encoded_array
            masks[index] = token_mask
            slots[index] = slot_ids
            uids.append(str(item["uid"]))
            valid_token_count += int(token_mask.sum())
            if progress is not None:
                progress(
                    {
                        "position": index + 1,
                        "total": len(dataset),
                        "elapsed_seconds": time.monotonic() - started,
                        "uid": item["uid"],
                        "valid_tokens": valid_token_count,
                    }
                )

    maximum_uid_length = max(len(uid) for uid in uids)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp.npz")
    np.savez(
        temporary,
        uids=np.asarray(uids, dtype=f"U{maximum_uid_length}"),
        features=features,
        token_mask=masks,
        slot_ids=slots,
        feature_cache_format=np.asarray(FEATURE_CACHE_FORMAT),
        cache_spec_sha256=np.asarray(_data_identity_hash(config["data"])),
        image_size=np.asarray(int(config["data"]["image_size"]), dtype=np.int32),
        source_image_size=np.asarray(
            int(config["data"].get("source_image_size", config["data"]["image_size"])),
            dtype=np.int32,
        ),
        encoder=np.asarray(str(config["model"]["backbone"])),
        weights_sha256=np.asarray(actual_hash),
        feature_dim=np.asarray(feature_dim, dtype=np.int32),
        uid_set_sha256=np.asarray(_uid_set_sha256(uids)),
    )
    os.replace(temporary, output)
    summary = {
        "studies": len(uids),
        "token_count": token_count,
        "feature_dim": feature_dim,
        "valid_tokens": int(valid_token_count),
        "missing_tokens": int(len(uids) * token_count - valid_token_count),
        "feature_dtype": "float16",
        "device": device.type,
        "encoder": config["model"]["backbone"],
        "weights_file_sha256": actual_hash,
        "cache_spec_sha256": _data_identity_hash(config["data"]),
        "uid_set_sha256": _uid_set_sha256(uids),
        "output_sha256": _sha256(output),
        "elapsed_seconds": float(time.monotonic() - started),
    }
    return summary


def load_feature_cache(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = {"uids", "features", "token_mask", "slot_ids"}
        if not required.issubset(payload.files):
            raise ValueError(f"Feature cache lacks keys: {sorted(required - set(payload.files))}")
        result = {key: payload[key].copy() for key in payload.files}
    if len(set(result["uids"].astype(str))) != len(result["uids"]):
        raise ValueError("Feature cache contains duplicate StudyInstanceUID values")
    if result["features"].shape[:2] != result["token_mask"].shape:
        raise ValueError("Feature and token-mask shapes disagree")
    if result["slot_ids"].shape != result["token_mask"].shape:
        raise ValueError("Slot and token-mask shapes disagree")
    return result


def _scalar_text(value: np.ndarray, name: str) -> str:
    if value.shape != ():
        raise ValueError(f"Feature metadata {name} must be a scalar")
    return str(value.item())


def _scalar_int(value: np.ndarray, name: str) -> int:
    if value.shape != ():
        raise ValueError(f"Feature metadata {name} must be a scalar")
    return int(value.item())


def validate_feature_cache_identity(
    feature_cache: dict[str, np.ndarray], config: dict[str, Any]
) -> None:
    required = {
        "feature_cache_format",
        "cache_spec_sha256",
        "image_size",
        "source_image_size",
        "encoder",
        "weights_sha256",
        "feature_dim",
        "uid_set_sha256",
    }
    missing = sorted(required.difference(feature_cache))
    if missing:
        raise ValueError(f"Feature cache lacks identity metadata: {missing}")
    observed = {
        "feature_cache_format": _scalar_text(
            feature_cache["feature_cache_format"], "feature_cache_format"
        ),
        "cache_spec_sha256": _scalar_text(
            feature_cache["cache_spec_sha256"], "cache_spec_sha256"
        ),
        "image_size": _scalar_int(feature_cache["image_size"], "image_size"),
        "source_image_size": _scalar_int(
            feature_cache["source_image_size"], "source_image_size"
        ),
        "encoder": _scalar_text(feature_cache["encoder"], "encoder"),
        "weights_sha256": _scalar_text(
            feature_cache["weights_sha256"], "weights_sha256"
        ),
        "feature_dim": _scalar_int(feature_cache["feature_dim"], "feature_dim"),
        "uid_set_sha256": _scalar_text(
            feature_cache["uid_set_sha256"], "uid_set_sha256"
        ),
    }
    expected = {
        "feature_cache_format": FEATURE_CACHE_FORMAT,
        "cache_spec_sha256": _data_identity_hash(config["data"]),
        "image_size": int(config["data"]["image_size"]),
        "source_image_size": int(
            config["data"].get("source_image_size", config["data"]["image_size"])
        ),
        "encoder": str(config["model"]["backbone"]),
        "weights_sha256": str(config["model"].get("weights_sha256", "")),
        "feature_dim": int(config["model"]["feature_dim"]),
        "uid_set_sha256": _uid_set_sha256(feature_cache["uids"].astype(str)),
    }
    if not expected["weights_sha256"]:
        expected["weights_sha256"] = observed["weights_sha256"]
    mismatches = {
        name: {"observed": observed[name], "expected": expected[name]}
        for name in expected
        if observed[name] != expected[name]
    }
    if mismatches:
        raise ValueError(f"Feature cache identity mismatch: {mismatches}")


def _seed_everything(seed: int) -> None:
    torch = _torch()
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _feature_head(feature_dim: int, num_slots: int, dropout: float):
    torch = _torch()
    nn = torch.nn

    class FrozenAttentionHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.slot_embedding = nn.Embedding(num_slots, feature_dim)
            self.normalization = nn.LayerNorm(feature_dim)
            self.dropout = nn.Dropout(dropout)
            self.label_queries = nn.Parameter(
                torch.empty(len(TARGET_COLUMNS), feature_dim)
            )
            self.head_weights = nn.Parameter(
                torch.empty(len(TARGET_COLUMNS), feature_dim)
            )
            self.head_bias = nn.Parameter(torch.zeros(len(TARGET_COLUMNS)))
            nn.init.normal_(self.label_queries, std=0.02)
            nn.init.normal_(self.head_weights, std=0.02)

        def forward(self, features, token_mask, slot_ids):
            encoded = self.normalization(features + self.slot_embedding(slot_ids))
            encoded = self.dropout(encoded)
            scores = torch.einsum("btd,cd->btc", encoded, self.label_queries)
            scores = scores / math.sqrt(feature_dim)
            scores = scores.masked_fill(~token_mask.unsqueeze(-1), -1e4)
            attention = scores.softmax(dim=1)
            pooled = torch.einsum("btc,btd->bcd", attention, encoded)
            return (
                pooled * self.head_weights.unsqueeze(0)
            ).sum(dim=-1) + self.head_bias

    return FrozenAttentionHead()


def _normalized_target_loss(logits, labels, mask, weight):
    torch = _torch()
    raw = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    effective = mask.float() * weight
    denominator = effective.sum(dim=0)
    valid = denominator > 0
    if not bool(valid.any()):
        return logits.sum() * 0.0
    per_target = (raw * effective).sum(dim=0)[valid] / denominator[valid]
    return per_target.mean()


def _manifest_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    labels = frame[TARGET_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(
        np.float32
    )
    masks = np.isfinite(labels)
    for index, target in enumerate(TARGET_COLUMNS):
        column = f"label_mask__{target}"
        if column in frame:
            masks[:, index] &= frame[column].astype(bool).to_numpy()
    confidence = np.ones_like(labels, dtype=np.float32)
    for index, target in enumerate(TARGET_COLUMNS):
        column = f"confidence__{target}"
        if column in frame:
            confidence[:, index] = pd.to_numeric(
                frame[column], errors="coerce"
            ).fillna(0.0).to_numpy(np.float32)
    return {
        "labels": np.nan_to_num(labels, nan=0.0),
        "mask": masks,
        "confidence": np.clip(confidence, 0.0, 1.0),
    }


def train_fixed_feature_head(
    feature_cache: dict[str, np.ndarray],
    train_manifest: pd.DataFrame,
    validation_manifest: pd.DataFrame,
    config: dict[str, Any],
    *,
    seed: int,
    device_name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Train a head for fixed steps; outer validation is touched only once at the end."""

    torch = _torch()
    _seed_everything(seed)
    uid_to_index = {
        str(uid): index for index, uid in enumerate(feature_cache["uids"].astype(str))
    }

    def feature_indices(frame: pd.DataFrame) -> np.ndarray:
        missing = [uid for uid in frame[ID_COLUMN].astype(str) if uid not in uid_to_index]
        if missing:
            raise ValueError(f"Feature cache is missing {len(missing)} studies")
        return np.asarray(
            [uid_to_index[uid] for uid in frame[ID_COLUMN].astype(str)], dtype=int
        )

    train_indices = feature_indices(train_manifest)
    validation_indices = feature_indices(validation_manifest)
    is_gold = train_manifest["is_gold"].astype(bool).to_numpy()
    gold_rows = np.flatnonzero(is_gold)
    pseudo_rows = np.flatnonzero(~is_gold)
    if len(gold_rows) == 0:
        raise ValueError("Every OOF training run requires gold training rows")

    arrays = _manifest_arrays(train_manifest)
    features_np = feature_cache["features"]
    mask_np = feature_cache["token_mask"]
    slot_np = feature_cache["slot_ids"]
    device = torch.device(device_name)
    head = _feature_head(
        feature_dim=features_np.shape[-1],
        num_slots=len(config["data"]["slots"]),
        dropout=float(config["model"].get("dropout", 0.2)),
    ).to(device)
    train_config = config["train"]
    steps = int(train_config["fixed_steps"])
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    generator = np.random.default_rng(seed)
    gold_batch_size = int(train_config["gold_batch_size"])
    pseudo_batch_size = int(train_config["pseudo_batch_size"])
    minimum_confidence = float(train_config["pseudo_min_confidence"])
    pseudo_coefficient = float(train_config["pseudo_loss_coefficient"])
    loss_history: list[float] = []

    train_features_tensor = torch.from_numpy(
        features_np[train_indices].astype(np.float32)
    ).to(device)
    train_token_mask_tensor = torch.from_numpy(mask_np[train_indices]).to(device)
    train_slot_tensor = torch.from_numpy(
        slot_np[train_indices].astype(np.int64)
    ).to(device)
    train_labels_tensor = torch.from_numpy(arrays["labels"]).to(device)
    train_label_mask_tensor = torch.from_numpy(arrays["mask"]).to(device)
    train_confidence_tensor = torch.from_numpy(arrays["confidence"]).to(device)

    def batch_tensors(rows: np.ndarray):
        tensor_rows = torch.as_tensor(rows, dtype=torch.long, device=device)
        return (
            train_features_tensor[tensor_rows],
            train_token_mask_tensor[tensor_rows],
            train_slot_tensor[tensor_rows],
            train_labels_tensor[tensor_rows],
            train_label_mask_tensor[tensor_rows],
            train_confidence_tensor[tensor_rows],
        )

    head.train()
    for _ in range(steps):
        chosen_gold = generator.choice(
            gold_rows, size=gold_batch_size, replace=len(gold_rows) < gold_batch_size
        )
        gold_features, gold_tokens, gold_slots, gold_labels, gold_mask, _ = batch_tensors(
            chosen_gold
        )
        gold_logits = head(gold_features, gold_tokens, gold_slots)
        gold_loss = _normalized_target_loss(
            gold_logits,
            gold_labels,
            gold_mask,
            torch.ones_like(gold_labels),
        )
        loss = gold_loss
        if len(pseudo_rows):
            chosen_pseudo = generator.choice(
                pseudo_rows,
                size=pseudo_batch_size,
                replace=len(pseudo_rows) < pseudo_batch_size,
            )
            pseudo_features, pseudo_tokens, pseudo_slots, pseudo_labels, pseudo_mask, confidence = batch_tensors(
                chosen_pseudo
            )
            pseudo_mask &= confidence >= minimum_confidence
            pseudo_weight = 0.2 + 0.8 * confidence
            pseudo_logits = head(pseudo_features, pseudo_tokens, pseudo_slots)
            pseudo_loss = _normalized_target_loss(
                pseudo_logits,
                pseudo_labels,
                pseudo_mask,
                pseudo_weight,
            )
            loss = loss + pseudo_coefficient * pseudo_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            head.parameters(), float(train_config["gradient_clip"])
        )
        optimizer.step()
        scheduler.step()
        loss_history.append(float(loss.detach().cpu()))

    head.eval()
    with torch.inference_mode():
        validation_features = torch.from_numpy(
            features_np[validation_indices].astype(np.float32)
        ).to(device)
        validation_mask = torch.from_numpy(mask_np[validation_indices]).to(device)
        validation_slots = torch.from_numpy(
            slot_np[validation_indices].astype(np.int64)
        ).to(device)
        probabilities = head(
            validation_features, validation_mask, validation_slots
        ).sigmoid().cpu().numpy()
    diagnostics = {
        "seed": int(seed),
        "fixed_steps": steps,
        "gold_train_rows": int(len(gold_rows)),
        "pseudo_train_rows": int(len(pseudo_rows)),
        "validation_rows": int(len(validation_manifest)),
        "initial_loss_mean": float(np.mean(loss_history[: min(10, steps)])),
        "final_loss_mean": float(np.mean(loss_history[-min(10, steps) :])),
        "minimum_loss": float(np.min(loss_history)),
    }
    return probabilities, diagnostics


def run_repeated_feature_oof(
    config: dict[str, Any],
    feature_cache_path: str | Path,
    condition_manifests: dict[str, pd.DataFrame],
    repeated_folds: pd.DataFrame,
    *,
    device_name: str,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cache = load_feature_cache(feature_cache_path)
    validate_feature_cache_identity(cache, config)
    predictions: list[pd.DataFrame] = []
    run_diagnostics: dict[str, Any] = {}
    total_runs = len(condition_manifests) * repeated_folds[["repeat", "fold"]].drop_duplicates().shape[0]
    run_position = 0
    for condition, manifest in condition_manifests.items():
        condition_diagnostics: dict[str, Any] = {}
        for repeat, repeat_folds in repeated_folds.groupby("repeat", sort=True):
            for fold, validation_rows in repeat_folds.groupby("fold", sort=True):
                validation_ids = set(validation_rows[ID_COLUMN].astype(str))
                validation_groups = set(validation_rows["report_group"].astype(str))
                validation = manifest.loc[
                    manifest[ID_COLUMN].astype(str).isin(validation_ids)
                ].copy()
                if len(validation) != len(validation_ids):
                    raise ValueError(
                        f"Condition {condition} lacks held-out gold rows for repeat {repeat}, fold {fold}"
                    )
                train = manifest.loc[
                    ~manifest["report_group"].astype(str).isin(validation_groups)
                ].copy()
                if set(train["report_group"].astype(str)).intersection(validation_groups):
                    raise AssertionError("Report-group leakage before feature-head training")
                seed = int(validation_rows["seed"].iloc[0]) + int(fold)
                probabilities, diagnostics = train_fixed_feature_head(
                    cache,
                    train,
                    validation,
                    config,
                    seed=seed,
                    device_name=device_name,
                )
                frame = validation[[ID_COLUMN, *TARGET_COLUMNS]].copy()
                frame.insert(0, "condition", condition)
                frame.insert(1, "repeat", int(repeat))
                frame.insert(2, "fold", int(fold))
                frame = frame.rename(
                    columns={target: f"gold__{target}" for target in TARGET_COLUMNS}
                )
                for index, target in enumerate(TARGET_COLUMNS):
                    frame[f"prediction__{target}"] = probabilities[:, index]
                predictions.append(frame)
                condition_diagnostics[f"repeat_{repeat}_fold_{fold}"] = diagnostics
                run_position += 1
                if progress is not None:
                    progress(
                        {
                            "position": run_position,
                            "total": total_runs,
                            "condition": condition,
                            "repeat": int(repeat),
                            "fold": int(fold),
                            **diagnostics,
                        }
                    )
        run_diagnostics[condition] = condition_diagnostics
    result = pd.concat(predictions, ignore_index=True)
    expected = repeated_folds[ID_COLUMN].nunique()
    counts = result.groupby(["condition", "repeat"])[ID_COLUMN].nunique()
    if not counts.eq(expected).all():
        raise RuntimeError("OOF predictions do not cover every gold study per repeat")
    return result, run_diagnostics


def _metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    per_target_auc = {
        target: float(roc_auc_score(y[:, index], probability[:, index]))
        for index, target in enumerate(TARGET_COLUMNS)
    }
    per_target_ap = {
        target: float(average_precision_score(y[:, index], probability[:, index]))
        for index, target in enumerate(TARGET_COLUMNS)
    }
    clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
    return {
        "macro_roc_auc": float(np.mean(list(per_target_auc.values()))),
        "macro_average_precision": float(np.mean(list(per_target_ap.values()))),
        "brier": float(np.mean((probability - y) ** 2)),
        "log_loss": float(
            -np.mean(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))
        ),
        "per_target_roc_auc": per_target_auc,
        "per_target_average_precision": per_target_ap,
    }


def evaluate_repeated_oof(
    predictions: pd.DataFrame,
    *,
    bootstrap_iterations: int = 5000,
    seed: int = 20260829,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Average repeat predictions, then run a paired study bootstrap."""

    conditions = list(dict.fromkeys(predictions["condition"].astype(str)))
    prediction_columns = [f"prediction__{target}" for target in TARGET_COLUMNS]
    gold_columns = [f"gold__{target}" for target in TARGET_COLUMNS]
    averaged_parts = []
    metrics: dict[str, Any] = {}
    matrices: dict[str, np.ndarray] = {}
    reference_ids: list[str] | None = None
    reference_gold: np.ndarray | None = None
    for condition in conditions:
        part = predictions.loc[predictions["condition"] == condition].copy()
        gold = part.groupby(ID_COLUMN, sort=True)[gold_columns].first()
        averaged = part.groupby(ID_COLUMN, sort=True)[prediction_columns].mean()
        if part.groupby(ID_COLUMN)[gold_columns].nunique().max().max() != 1:
            raise ValueError("Gold labels change across repeated predictions")
        ids = averaged.index.astype(str).tolist()
        y = gold.to_numpy(float)
        probability = averaged.to_numpy(float)
        if reference_ids is None:
            reference_ids, reference_gold = ids, y
        elif ids != reference_ids or not np.array_equal(y, reference_gold):
            raise ValueError("Conditions are not aligned on identical gold rows")
        metrics[condition] = _metrics(y, probability)
        matrices[condition] = probability
        output = averaged.reset_index()
        output.insert(0, "condition", condition)
        for index, target in enumerate(TARGET_COLUMNS):
            output[f"gold__{target}"] = y[:, index]
        averaged_parts.append(output)
    assert reference_gold is not None
    rng = np.random.default_rng(seed)
    bootstrap_scores = {condition: [] for condition in conditions}
    for _ in range(bootstrap_iterations):
        sample = rng.integers(0, len(reference_gold), size=len(reference_gold))
        y = reference_gold[sample]
        for condition in conditions:
            probability = matrices[condition][sample]
            values = []
            for target_index in range(len(TARGET_COLUMNS)):
                if np.unique(y[:, target_index]).size == 2:
                    values.append(
                        roc_auc_score(y[:, target_index], probability[:, target_index])
                    )
            bootstrap_scores[condition].append(float(np.mean(values)))
    for condition in conditions:
        distribution = np.asarray(bootstrap_scores[condition])
        metrics[condition]["macro_roc_auc_bootstrap_95"] = [
            float(np.quantile(distribution, 0.025)),
            float(np.quantile(distribution, 0.975)),
        ]
    comparisons: dict[str, Any] = {}
    pairs = []
    for condition in conditions:
        if condition != "gold_only" and "gold_only" in conditions:
            pairs.append((condition, "gold_only"))
    for baseline in ("rule_only", "public_v52"):
        if "evidence_gated" in conditions and baseline in conditions:
            pairs.append(("evidence_gated", baseline))
    for left, right in pairs:
        delta = np.asarray(bootstrap_scores[left]) - np.asarray(bootstrap_scores[right])
        comparisons[f"{left}_minus_{right}"] = {
            "observed_delta_macro_roc_auc": float(
                metrics[left]["macro_roc_auc"] - metrics[right]["macro_roc_auc"]
            ),
            "paired_bootstrap_95": [
                float(np.quantile(delta, 0.025)),
                float(np.quantile(delta, 0.975)),
            ],
            "bootstrap_probability_positive": float(np.mean(delta > 0)),
        }
    summary = {
        "gold_studies": int(len(reference_gold)),
        "targets": len(TARGET_COLUMNS),
        "repeats_averaged": int(predictions["repeat"].nunique()),
        "bootstrap_iterations": int(bootstrap_iterations),
        "conditions": metrics,
        "comparisons": comparisons,
        "claim_strength": "descriptive/directional only",
    }
    return pd.concat(averaged_parts, ignore_index=True), summary
