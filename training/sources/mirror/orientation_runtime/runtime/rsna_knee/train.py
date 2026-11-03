"""Fold training and OOF prediction for the 2.5D model."""

from __future__ import annotations

from .batch_inputs import model_inputs
from .epoch_stop import plateau_stop

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .constants import ID_COLUMN, TARGET_COLUMNS
from .dataset import create_study_dataset
from .metrics import macro_roc_auc
from .model import create_model, parameter_summary
from .pseudo_oof import evaluate_oof_frame
from .training_state import (FORMAT_VERSION, atomic_write, capture_rng, export_best_artifacts,
    file_sha256, load_training_state, make_training_identity, restore_rng, save_json,
    save_torch, validate_best_artifacts)

VALIDATION_SCOPES = ("gold_only", "all_holdout", "full_train")


def build_fold_partitions(
    manifest: pd.DataFrame,
    *,
    fold: int,
    n_splits: int,
    validation_scope: str = "gold_only",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split one manifest into ``(train_frame, valid_frame)`` for ``fold``.

    ``gold_only`` is the legacy protocol: the outer validation partition holds
    only gold studies of the fold, and pseudo rows may carry the sentinel fold
    ``-1`` to stay in every training set.

    ``all_holdout`` is the v2 protocol: every row carries a real fold, the whole
    fold (gold and pseudo) is held out and scored with
    :mod:`rsna_knee.pseudo_oof`, and no training row may share a report group
    with a held-out row.
    """

    if validation_scope not in VALIDATION_SCOPES:
        raise ValueError(
            f"train.validation_scope must be one of {VALIDATION_SCOPES}, got {validation_scope!r}"
        )
    if validation_scope == 'full_train':
        if fold != -1:
            raise ValueError('Full-data training requires fold=-1; it produces no OOF')
        if manifest.empty or manifest[ID_COLUMN].isna().any() or manifest[ID_COLUMN].astype(str).duplicated().any():
            raise ValueError('Full-data manifest must contain unique, non-null study IDs')
        if not manifest[TARGET_COLUMNS].apply(pd.to_numeric, errors='coerce').notna().any(axis=1).all():
            raise ValueError('Full-data manifest contains a study without any usable target')
        return manifest.reset_index(drop=True), manifest.iloc[:0].copy()
    if "fold" not in manifest:
        raise ValueError("Training manifest must contain a fold column")
    if n_splits <= 1:
        raise ValueError("config.folds.n_splits must be greater than one")
    numeric_folds = pd.to_numeric(manifest["fold"], errors="raise").astype(int)
    if fold not in range(n_splits):
        raise ValueError(f"Requested fold {fold} is outside [0, {n_splits})")
    nonnegative_folds = set(numeric_folds[numeric_folds >= 0].tolist())
    if nonnegative_folds != set(range(n_splits)):
        raise ValueError(
            f"Manifest fold set {sorted(nonnegative_folds)} does not match "
            f"config.folds.n_splits={n_splits}"
        )
    if (numeric_folds < -1).any():
        raise ValueError("Only pseudo training rows may use the sentinel fold -1")
    manifest = manifest.assign(fold=numeric_folds)
    numeric_labels = manifest[TARGET_COLUMNS].apply(pd.to_numeric, errors="coerce")
    usable = numeric_labels.notna().any(axis=1)
    manifest = manifest.loc[usable].reset_index(drop=True)
    is_gold = manifest.get("is_gold", pd.Series(True, index=manifest.index)).astype(bool)

    if validation_scope == "gold_only":
        # Non-gold rows may carry a non-negative fold solely to mark a shared
        # validation report group.  They are excluded from that fold's training
        # set but must never enter the scored outer validation partition.
        train_frame = manifest[manifest["fold"] != fold].reset_index(drop=True)
        valid_frame = manifest[(manifest["fold"] == fold) & is_gold].reset_index(drop=True)
        if "is_gold" in valid_frame and not valid_frame["is_gold"].astype(bool).all():
            raise ValueError("Outer validation must contain gold studies only")
    else:
        if (manifest["fold"] < 0).any():
            raise ValueError(
                "validation_scope='all_holdout' requires a real fold for every row; "
                "sentinel fold -1 is not allowed"
            )
        valid_frame = manifest[manifest["fold"] == fold].reset_index(drop=True)
        train_frame = manifest[manifest["fold"] != fold].reset_index(drop=True)
        if "report_group" in manifest:
            held_groups = set(valid_frame["report_group"].astype(str))
            train_frame = train_frame[
                ~train_frame["report_group"].astype(str).isin(held_groups)
            ].reset_index(drop=True)
    if train_frame.empty or valid_frame.empty:
        raise ValueError(f"Fold {fold} has an empty train or validation partition")
    if "report_group" in manifest:
        training_groups = set(train_frame["report_group"].astype(str))
        validation_groups = set(valid_frame["report_group"].astype(str))
        overlap = sorted(training_groups.intersection(validation_groups))
        if overlap:
            raise ValueError(
                "Outer validation report groups remain in training; "
                f"first={overlap[0]}"
            )
    return train_frame, valid_frame


class ExponentialMovingAverage:
    """Parameter EMA evaluated in place of the raw weights when enabled."""

    def __init__(self, model: Any, decay: float) -> None:
        torch = _require_torch()
        if not 0.0 < decay < 1.0:
            raise ValueError("train.ema_decay must lie in (0, 1)")
        self.decay = float(decay)
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.state_dict().items()
            if torch.is_floating_point(parameter)
        }
        self._backup: dict[str, Any] | None = None

    def update(self, model: Any) -> None:
        torch = _require_torch()
        with torch.no_grad():
            for name, value in model.state_dict().items():
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    def apply(self, model: Any) -> None:
        state = model.state_dict()
        self._backup = {name: state[name].detach().clone() for name in self.shadow}
        model.load_state_dict({**state, **self.shadow}, strict=True)

    def restore(self, model: Any) -> None:
        if self._backup is None:
            return
        state = model.state_dict()
        model.load_state_dict({**state, **self._backup}, strict=True)
        self._backup = None


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Training requires PyTorch; install the 'train' extra") from error
    return torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch = _require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def weighted_bce_loss_per_sample(
    logits: Any,
    labels: Any,
    label_mask: Any,
    label_weight: Any,
    *,
    normalization: str = "active_count",
) -> tuple[Any, Any]:
    """Return one masked BCE value per study plus its validity mask.

    ``active_count`` is the corrected production geometry: it normalizes by the
    number of active target cells, so multiplying every active weight in a study
    also multiplies that study's gradient.  ``weight_sum`` is retained only for
    deterministic comparison with the legacy behavior, where a uniform study
    weight cancels out.
    """

    torch = _require_torch()
    if logits.shape != labels.shape:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} does not match labels "
            f"shape {tuple(labels.shape)}"
        )
    if label_weight.shape != labels.shape:
        raise ValueError(
            f"label_weight shape {tuple(label_weight.shape)} does not match labels "
            f"shape {tuple(labels.shape)}"
        )
    if label_mask.shape != labels.shape:
        raise ValueError(
            f"label_mask shape {tuple(label_mask.shape)} does not match labels "
            f"shape {tuple(labels.shape)}"
        )
    if not bool(torch.isfinite(label_weight).all()):
        raise ValueError("label_weight must contain only finite values")
    if bool((label_weight < 0).any()):
        raise ValueError("label_weight must be non-negative")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must contain only finite values")
    element_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        torch.where(
            label_mask.bool() & (label_weight > 0),
            labels,
            torch.zeros_like(labels),
        ),
        reduction="none",
    )
    active = label_mask.bool() & (label_weight > 0)
    if not bool(torch.isfinite(labels[active]).all()):
        raise ValueError("active labels must contain only finite values")
    weights = active.float() * label_weight
    if normalization == "active_count":
        denominator = active.float().sum(dim=-1)
    elif normalization == "weight_sum":
        denominator = weights.sum(dim=-1)
    else:
        raise ValueError(f"Unknown loss normalization: {normalization}")
    numerator = (element_loss * weights).sum(dim=-1)
    valid = denominator > 0
    safe_denominator = torch.where(valid, denominator, torch.ones_like(denominator))
    losses = torch.where(
        valid,
        numerator / safe_denominator,
        torch.zeros_like(numerator),
    )
    return losses, valid


def weighted_bce_loss(
    logits: Any,
    labels: Any,
    label_mask: Any,
    label_weight: Any,
    *,
    normalization: str = "active_count",
) -> Any | None:
    """Average valid study losses while preserving absolute study weights."""

    losses, valid = weighted_bce_loss_per_sample(
        logits,
        labels,
        label_mask,
        label_weight,
        normalization=normalization,
    )
    if not bool(valid.any()):
        return None
    return losses[valid].mean()


def build_optimizer(model: Any, train_config: dict[str, Any]) -> Any:
    """Build AdamW with a low-LR backbone group and a regular-LR task head."""

    torch = _require_torch()
    head_learning_rate = float(train_config.get("learning_rate", 3e-4))
    encoder_learning_rate = float(
        train_config.get("encoder_learning_rate", head_learning_rate)
    )
    weight_decay = float(train_config.get("weight_decay", 1e-4))
    layer_decay = float(train_config.get('layer_decay', 1.0))
    if not 0.0 < layer_decay <= 1.0:
        raise ValueError('train.layer_decay must lie in (0, 1]')

    encoder = getattr(model, "encoder", None)
    backbone = getattr(encoder, "backbone", encoder)
    backbone_parameter_ids = (
        {id(parameter) for parameter in backbone.parameters()}
        if backbone is not None
        else set()
    )
    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) in backbone_parameter_ids
    ]
    head_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_parameter_ids
    ]
    parameter_groups: list[dict[str, Any]] = []
    if head_parameters:
        parameter_groups.append(
            {
                "params": head_parameters,
                "lr": head_learning_rate,
                "group_name": "head",
            }
        )
    if backbone_parameters and layer_decay < 1.0:
        stages = getattr(backbone, 'stages', None)
        if stages is None or len(stages) != 4 or not all(hasattr(s, 'blocks') for s in stages):
            raise ValueError('Layer decay currently requires native four-stage ConvNeXt')
        depths = [len(s.blocks) for s in stages]
        max_layer = sum(depths) + 1
        layers: dict[int, list[Any]] = {}
        for name, parameter in backbone.named_parameters():
            if not parameter.requires_grad:
                continue
            parts = name.split('.')
            if name.startswith('stem.'):
                layer = 0
            elif name.startswith('stages.'):
                stage = int(parts[1])
                layer = 1 + sum(depths[:stage])
                if len(parts) > 3 and parts[2] == 'blocks':
                    layer += int(parts[3])
            else:
                layer = max_layer
            layers.setdefault(layer, []).append(parameter)
        for layer, parameters in sorted(layers.items()):
            parameter_groups.append(dict(params=parameters,
                lr=encoder_learning_rate * layer_decay ** (max_layer - layer),
                group_name=f'backbone_layer_{layer}'))
    elif backbone_parameters:
        parameter_groups.append(
            {
                "params": backbone_parameters,
                "lr": encoder_learning_rate,
                "group_name": "backbone",
            }
        )
    if not parameter_groups:
        raise ValueError("Model has no trainable parameters")
    return torch.optim.AdamW(parameter_groups, weight_decay=weight_decay)


def optimizer_summary(optimizer: Any) -> list[dict[str, Any]]:
    return [
        {
            "name": str(group.get("group_name", f"group_{index}")),
            "learning_rate": float(group["lr"]),
            "parameters": int(
                sum(parameter.numel() for parameter in group["params"])
            ),
        }
        for index, group in enumerate(optimizer.param_groups)
    ]


def _predict_loader(
    model: Any,
    loader: Any,
    device: Any,
    *,
    valid_frame: pd.DataFrame | None = None,
    fold: int | None = None,
    pseudo_eval_min_confidence: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    torch = _require_torch()
    model.eval()
    rows: list[dict[str, Any]] = []
    labels_all: list[np.ndarray] = []
    predictions_all: list[np.ndarray] = []
    masks_all: list[np.ndarray] = []
    gold_all: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            logits = model(*model_inputs(batch, device))
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("Validation logits must be finite")
            predictions = logits.sigmoid().float().cpu().numpy()
            labels = batch["labels"].numpy()
            masks = batch["label_mask"].numpy().astype(bool)
            is_gold = batch["is_gold"].numpy().astype(bool)
            labels_all.append(labels)
            predictions_all.append(predictions)
            masks_all.append(masks)
            gold_all.append(is_gold)
            for batch_index, uid in enumerate(batch["uid"]):
                row: dict[str, Any] = {ID_COLUMN: uid, "is_gold": bool(is_gold[batch_index])}
                if fold is not None:
                    row["fold"] = int(fold)
                for target_index, target in enumerate(TARGET_COLUMNS):
                    row[target] = (
                        float(labels[batch_index, target_index])
                        if masks[batch_index, target_index]
                        else np.nan
                    )
                    row[f"pred__{target}"] = float(predictions[batch_index, target_index])
                rows.append(row)

    if not predictions_all:
        raise ValueError("Validation loader produced no batches")
    labels_array = np.concatenate(labels_all)
    predictions_array = np.concatenate(predictions_all)
    mask_array = np.concatenate(masks_all)
    gold_array = np.concatenate(gold_all)
    frame = pd.DataFrame(rows)
    if valid_frame is not None:
        # Carry teacher confidence so the frame is a complete OOF record for
        # rsna_knee.pseudo_oof; missing confidence columns default to 1.0.
        confidence_columns = [f"confidence__{target}" for target in TARGET_COLUMNS]
        source = valid_frame.set_index(ID_COLUMN)
        for column in confidence_columns:
            if column in source:
                frame[column] = (
                    frame[ID_COLUMN].map(pd.to_numeric(source[column], errors="coerce")).to_numpy(float)
                )
            else:
                frame[column] = 1.0
        if "report_group" in source:
            frame["report_group"] = frame[ID_COLUMN].map(source["report_group"]).to_numpy()
    metrics: dict[str, Any] = {
        "all_observed": macro_roc_auc(labels_array, predictions_array, mask_array),
        "gold": macro_roc_auc(
            labels_array,
            predictions_array,
            mask_array & gold_array[:, None],
        ),
    }
    scorecard = evaluate_oof_frame(frame, min_confidence=pseudo_eval_min_confidence)
    metrics["pseudo"] = scorecard["pseudo"]
    metrics["calibration"] = scorecard["calibration"]
    # Keep legacy selectors unchanged; the explicit alias prevents implying that
    # all 881 studies have authoritative binary outcomes.
    metrics["binary_observed"] = metrics["all_observed"]
    metrics["validation_counts"] = {
        "studies": int(len(frame)), "gold_studies": int(gold_array.sum()),
        "observed_cells": int(mask_array.sum()),
        "binary_gold_cells": int((mask_array & gold_array[:, None]
                                  & ((labels_array == 0) | (labels_array == 1))).sum()),
        "binary_non_gold_cells": int((mask_array & ~gold_array[:, None]
                                      & ((labels_array == 0) | (labels_array == 1))).sum()),
    }
    metrics["protocol_notes"] = {
        "all_observed": "compatibility alias of binary_observed; not a whole-fold pseudo metric",
        "pseudo": "gold plus confidence-filtered, thresholded report-derived labels; teacher agreement",
        "calibration": "against observed soft/binary labels; not independently verified clinical truth",
    }
    return frame, metrics


def _optimizer_step(model: Any, optimizer: Any, scaler: Any,
                    ema: ExponentialMovingAverage | None, telemetry: list | None = None) -> bool:
    previous_scale = scaler.get_scale()
    local_before = {}
    entry = None
    if telemetry is not None and len(telemetry) < 3:
        entry = {'scale': previous_scale, 'parameters': {}}
        for name, parameter in model.local_head.named_parameters():
            local_before[name] = parameter.detach().clone()
            grad = parameter.grad
            entry['parameters'][name] = {
                'grad_abs_max': float(grad.detach().abs().max()) / previous_scale if grad is not None else None,
                'grad_nonzero_fraction': float((grad.detach() != 0).float().mean()) if grad is not None else 0.0,
                'grad_finite': bool(grad.isfinite().all()) if grad is not None else True,
            }
    scaler.step(optimizer)
    scaler.update()
    succeeded = scaler.get_scale() >= previous_scale
    if entry is not None:
        entry['succeeded'] = succeeded
        for name, parameter in model.local_head.named_parameters():
            entry['parameters'][name]['update_abs_max'] = float((parameter.detach() - local_before[name]).abs().max())
        telemetry.append(entry)
    optimizer.zero_grad(set_to_none=True)
    # Preserve the historical recipe: EMA advances per attempted update.
    # Successful/skipped counters now make this policy explicit.
    if ema is not None:
        ema.update(model)
    return succeeded


def build_training_scheduler(optimizer: Any, train_config: dict[str, Any],
                             updates_per_epoch: int) -> tuple[Any, str]:
    """Keep legacy epoch cosine or use a resumable successful-update warmup clock."""
    torch = _require_torch()
    training_epochs = int(train_config.get('epochs', 12))
    epochs = int(train_config.get('scheduler_epochs', training_epochs))
    if epochs < training_epochs:
        raise ValueError('Scheduler horizon cannot be shorter than training')
    warmup = float(train_config.get('warmup_epochs', 0))
    if not 0 <= warmup < epochs or updates_per_epoch < 1:
        raise ValueError('warmup_epochs must be in [0, epochs); updates_per_epoch positive')
    if warmup == 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs), 'epoch'
    start_factor = float(train_config.get('warmup_start_factor', 0.1))
    if not 0 < start_factor <= 1:
        raise ValueError('warmup_start_factor must be in (0, 1]')
    warmup_updates = max(1, round(warmup * updates_per_epoch))
    total_updates = epochs * updates_per_epoch
    if warmup_updates >= total_updates:
        raise ValueError('Warmup must leave at least one cosine update')
    def multiplier(completed_updates: int) -> float:
        if completed_updates < warmup_updates:
            return start_factor + (1 - start_factor) * completed_updates / warmup_updates
        progress = min(1.0, (completed_updates - warmup_updates) / (total_updates - warmup_updates))
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier), 'successful_optimizer_update'


def normalize_accumulated_gradients(model: Any, nominal_studies: int, valid_studies: int) -> None:
    """Convert sum/nominal scaled gradients to the mean over actual valid studies."""
    if nominal_studies < 1 or valid_studies < 1:
        raise ValueError('Gradient normalization requires positive study counts')
    factor = nominal_studies / valid_studies
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(factor)


def train_fold(
    config: dict[str, Any], manifest: pd.DataFrame, series: pd.DataFrame | None,
    *, fold: int, output_dir: str | Path, resume: str | Path | None = None,
    max_epochs_this_run: int | None = None, device_name: str | None = None,
) -> dict[str, Any]:
    """Train with an exclusive output lock and epoch-boundary resume."""
    import fcntl
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / '.training.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another trainer owns this output directory') from exc
        return _train_fold(config, manifest, series, fold=fold, output_dir=output_dir,
                           resume=resume, max_epochs_this_run=max_epochs_this_run,
                           device_name=device_name)


def _train_fold(
    config: dict[str, Any], manifest: pd.DataFrame, series: pd.DataFrame | None,
    *, fold: int, output_dir: Path, resume: str | Path | None,
    max_epochs_this_run: int | None, device_name: str | None,
) -> dict[str, Any]:
    torch = _require_torch()
    from torch.utils.data import DataLoader
    if max_epochs_this_run is not None and max_epochs_this_run < 1:
        raise ValueError('max_epochs_this_run must be positive')
    occupied = any((output_dir / name).exists() for name in
                   ['last_training.pt', 'best.pt', 'history.json', 'summary.json', 'resolved_config.yaml', 'COMPLETED.json'])
    if occupied and resume is None:
        raise ValueError('Output directory already contains a run; explicit --resume is required')
    if resume is not None and Path(resume).resolve().parent != output_dir.resolve():
        raise ValueError('Resume checkpoint must belong to output_dir; move the whole run together')
    train_config = config['train']
    epochs = int(train_config.get('epochs', 12))
    if epochs < 1:
        raise ValueError('train.epochs must be positive')
    weight_epochs = train_config.get('save_weight_epochs', [])
    if (not isinstance(weight_epochs, list) or
            any(type(e) is not int or not 1 <= e <= epochs for e in weight_epochs) or
            len(set(weight_epochs)) != len(weight_epochs)):
        raise ValueError('save_weight_epochs must contain unique one-based epochs within the run')
    validation_scope = str(train_config.get('validation_scope', 'gold_only'))
    full_train = validation_scope == 'full_train'
    if full_train and (train_config.get('selection_metric') != 'last_epoch'
                       or train_config.get('plateau_stop')
                       or train_config.get('report_raw_at_completion', False)
                       or train_config.get('save_epoch_predictions', False)):
        raise ValueError('Full-data training requires fixed last_epoch selection and no validation artifacts or early stopping')
    pseudo_eval_min_confidence = float(train_config.get('pseudo_eval_min_confidence', 0.5))
    train_frame, valid_frame = build_fold_partitions(
        manifest, fold=fold, n_splits=int(config.get('folds', {}).get('n_splits', 0)),
        validation_scope=validation_scope)
    seed_offset = 0 if full_train else int(fold)
    seed_everything(int(config.get('seed', 20260828)) + seed_offset)
    if device_name is not None:
        device = torch.device(device_name)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    identity = make_training_identity(config, train_frame, valid_frame, series, fold=fold, device=device)
    restored = load_training_state(resume, identity) if resume is not None else None
    if restored is not None:
        for record in restored['history']:
            if record['epoch'] + 1 in weight_epochs:
                receipt = record.get('weight_snapshot', {})
                name = f"weights_epoch_{record['epoch']:03d}.pt"
                path = output_dir / name
                if (receipt.get('name') != name or not path.is_file() or path.is_symlink()
                        or path.stat().st_size != receipt.get('bytes')
                        or file_sha256(path) != receipt.get('sha256')):
                    raise ValueError('Committed weight snapshot missing or changed before resume')
    if restored is not None:
        validate_best_artifacts(output_dir, restored['best_artifacts'])
        if not 0 <= restored['next_epoch'] <= epochs:
            raise ValueError('Invalid checkpoint next_epoch')
    # A completion marker may refer to a later/unfinished export generation.
    # It cannot survive any resumed writes; publish a fresh marker only at end.
    (output_dir / 'COMPLETED.json').unlink(missing_ok=True)
    train_dataset = create_study_dataset(train_frame, series, config, training=True, include_labels=True)
    valid_dataset = None if full_train else create_study_dataset(valid_frame, series, config, training=False, include_labels=True)
    if train_config.get('preverify_archives', False):
        if config['data'].get('cache_format') != 'full_source_bank_v1':
            raise ValueError('Archive preverification requires a full-source bank')
        for dataset, frame in ((train_dataset, train_frame), (valid_dataset, valid_frame)):
            for uid in frame[ID_COLUMN].astype(str):
                dataset.cache.verify_archive(uid)
    loader_arguments = dict(batch_size=int(train_config.get('batch_size', 2)),
                            num_workers=int(train_config.get('num_workers', 2)),
                            pin_memory=device.type == 'cuda')
    if loader_arguments['num_workers'] > 0:
        loader_arguments['prefetch_factor'] = int(train_config.get('prefetch_factor', 2))
    data_rng_mode = train_config.get('data_rng_mode', 'legacy_global')
    train_generator = valid_generator = None
    if data_rng_mode == 'study_epoch_v1':
        from .data_rng import EpochSeededDataset, stream_seed
        data_seed = int(config.get('seed', 20260828)) + seed_offset
        train_dataset = EpochSeededDataset(train_dataset, train_frame[ID_COLUMN], data_seed)
        train_generator = torch.Generator()
        valid_generator = torch.Generator().manual_seed(stream_seed(data_seed, 0, 'validation'))
    elif data_rng_mode != 'legacy_global':
        raise ValueError(f'Unknown train.data_rng_mode: {data_rng_mode}')
    train_loader = DataLoader(train_dataset, shuffle=True, generator=train_generator, **loader_arguments)
    valid_loader = None if full_train else DataLoader(valid_dataset, shuffle=False, generator=valid_generator, **loader_arguments)
    model = create_model(config, initialize_backbone=restored is None).to(device)
    optimizer = build_optimizer(model, train_config)
    initial_optimizer_groups = optimizer_summary(optimizer)
    accumulate_steps = max(1, int(train_config.get('accumulate_steps', 1)))
    accumulation_mode = str(train_config.get('accumulation_normalization', 'legacy_microbatch'))
    if accumulation_mode not in ('legacy_microbatch', 'valid_study_mean'):
        raise ValueError(f'Unknown accumulation_normalization: {accumulation_mode}')
    nominal_studies = loader_arguments['batch_size'] * accumulate_steps
    scheduler, scheduler_clock = build_training_scheduler(
        optimizer, train_config, math.ceil(len(train_loader) / accumulate_steps))
    use_amp = bool(train_config.get('amp', True)) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    selection_metric = str(train_config.get('selection_metric', 'gold_macro_auc'))
    if selection_metric not in ('gold_macro_auc', 'all_observed_macro_auc', 'pseudo_macro_auc', 'last_epoch'):
        raise ValueError(f'Unknown train.selection_metric: {selection_metric}')
    ema_decay = float(train_config.get('ema_decay', 0.0))
    ema = ExponentialMovingAverage(model, ema_decay) if ema_decay > 0 else None
    history: list[dict[str, Any]] = []
    best_score, best_epoch, best_metrics = -float('inf'), -1, None
    best_artifacts: dict[str, Any] = {}
    attempted_steps = successful_steps = skipped_steps = start_epoch = 0
    if restored is not None:
        model.load_state_dict(restored['raw_model_state'])
        optimizer.load_state_dict(restored['optimizer_state'])
        scheduler.load_state_dict(restored['scheduler_state'])
        scaler.load_state_dict(restored['scaler_state'])
        if ema is not None:
            ema.shadow = {k: v.to(device) for k, v in restored['ema_state'].items()}
        history = restored['history']
        best_score = restored['best_score'] if restored['best_score'] is not None else -float('inf')
        best_epoch, best_metrics = restored['best_epoch'], restored['best_metrics']
        best_artifacts = restored['best_artifacts']
        attempted_steps, successful_steps, skipped_steps = [restored[k] for k in
            ['attempted_steps', 'successful_steps', 'skipped_steps']]
        start_epoch = restored['next_epoch']
        if best_artifacts:
            export_best_artifacts(output_dir, best_artifacts)
        # Initialization and validation iterator creation consume RNG: restore last.
        restore_rng(restored['rng_state'], device)
    def commit_training_state(next_epoch: int) -> None:
        state = dict(format_version=FORMAT_VERSION, kind='epoch_training_state', identity=identity,
            raw_model_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            ema_state={k: v.detach().cpu().clone() for k, v in ema.shadow.items()} if ema else None,
            optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(),
            scaler_state=scaler.state_dict(), rng_state=capture_rng(device), next_epoch=next_epoch,
            history=history, best_score=best_score if np.isfinite(best_score) else None,
            best_epoch=best_epoch, best_metrics=best_metrics, best_artifacts=best_artifacts,
            attempted_steps=attempted_steps, successful_steps=successful_steps, skipped_steps=skipped_steps)
        save_torch(output_dir / 'last_training.pt', state)
        if train_config.get('epoch_backup_manifest', False):
            checkpoint_path = output_dir / 'last_training.pt'
            save_json(output_dir / 'EPOCH_COMMITTED.json', dict(
                training_identity_sha256=identity['sha256'], fold=fold,
                next_epoch=next_epoch, best_artifacts=best_artifacts,
                weight_snapshots=[r['weight_snapshot'] for r in history if 'weight_snapshot' in r],
                checkpoint=dict(name='last_training.pt', bytes=checkpoint_path.stat().st_size,
                                sha256=file_sha256(checkpoint_path))))
    if restored is None:
        # Commit epoch0 before any other run artifacts: even a first-batch crash
        # can restart from identical weights and RNG without an overwrite bypass.
        commit_training_state(0)
    atomic_write(output_dir / 'resolved_config.yaml',
                 lambda f: f.write(yaml.safe_dump(config, sort_keys=False, allow_unicode=True).encode()))
    save_json(output_dir / 'training_identity.json', identity)
    if full_train:
        atomic_write(output_dir / 'training_studies.csv',
                     lambda f: f.write(train_frame[[ID_COLUMN]].to_csv(index=False).encode()))
    stop_epoch = min(epochs, start_epoch + max_epochs_this_run) if max_epochs_this_run else epochs
    stop_policy = train_config.get('plateau_stop')
    early_stopped = plateau_stop(history, best_epoch, stop_policy, epochs)
    if early_stopped:
        stop_epoch = start_epoch
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, stop_epoch):
        if data_rng_mode == 'study_epoch_v1':
            train_dataset.set_epoch(epoch)
            train_generator.manual_seed(stream_seed(data_seed, epoch, 'sampler'))
            valid_generator.manual_seed(stream_seed(data_seed, epoch, 'validation'))
        model.train()
        local_telemetry = [] if train_config.get('local_head_telemetry', False) else None
        local_probe = None
        if local_telemetry is not None and model.local_head is None:
            raise ValueError('local_head_telemetry requires a local head')
        losses: list[Any] = []
        epoch_valid_studies = 0
        cycle_valid_studies = 0
        pending_steps = 0
        epoch_initial_learning_rates = [float(g['lr']) for g in optimizer.param_groups]
        successful_before_epoch = successful_steps
        train_started = time.perf_counter()
        for batch in train_loader:
            with torch.autocast('cuda', enabled=use_amp):
                inputs = model_inputs(batch, device)
                if local_telemetry is not None and local_probe is None:
                    probe = model.forward_diagnostics(*inputs)
                    logits = probe['logits']
                    local_probe = dict(residual_abs_max=float(probe['gated_local_logits'].detach().abs().max()),
                        residual_abs_mean=float(probe['gated_local_logits'].detach().abs().mean()),
                        global_abs_mean=float(probe['global_logits'].detach().abs().mean()),
                        feature_dtype=str(probe['local_features'].dtype))
                    del probe
                else:
                    logits = model(*inputs)
                sample_losses, sample_valid = weighted_bce_loss_per_sample(logits, *(batch[k].to(device, non_blocking=True)
                            for k in ('labels', 'label_mask', 'label_weight')),
                            normalization=str(train_config.get('loss_normalization', 'active_count')))
            valid_studies = int(sample_valid.sum().item())
            if not valid_studies:
                continue
            loss_sum = sample_losses[sample_valid].sum()
            loss = loss_sum / valid_studies
            backward_loss = loss_sum / nominal_studies if accumulation_mode == 'valid_study_mean' else loss / accumulate_steps
            scaler.scale(backward_loss).backward()
            pending_steps += 1
            cycle_valid_studies += valid_studies
            epoch_valid_studies += valid_studies
            if pending_steps % accumulate_steps == 0:
                if accumulation_mode == 'valid_study_mean':
                    normalize_accumulated_gradients(model, nominal_studies, cycle_valid_studies)
                succeeded = (_optimizer_step(model, optimizer, scaler, ema, local_telemetry)
                             if local_telemetry is not None else _optimizer_step(model, optimizer, scaler, ema))
                cycle_valid_studies = 0
                if succeeded and scheduler_clock == 'successful_optimizer_update':
                    scheduler.step()
                attempted_steps += 1
                successful_steps += int(succeeded)
                skipped_steps += int(not succeeded)
            losses.append((loss_sum if accumulation_mode == 'valid_study_mean' else loss).detach())
        if pending_steps % accumulate_steps:
            if accumulation_mode == 'valid_study_mean':
                normalize_accumulated_gradients(model, nominal_studies, cycle_valid_studies)
            succeeded = (_optimizer_step(model, optimizer, scaler, ema, local_telemetry)
                             if local_telemetry is not None else _optimizer_step(model, optimizer, scaler, ema))
            if succeeded and scheduler_clock == 'successful_optimizer_update':
                scheduler.step()
            attempted_steps += 1
            successful_steps += int(succeeded)
            skipped_steps += int(not succeeded)
        if not losses:
            raise ValueError('No usable training batches')
        if successful_steps == successful_before_epoch:
            raise FloatingPointError('No successful optimizer updates in this epoch')
        if full_train and epoch_valid_studies != len(train_frame):
            raise ValueError('Full-data epoch did not consume every training study with usable supervision')
        loss_values = torch.stack(losses).cpu()
        train_loss = (float(loss_values.sum()) / epoch_valid_studies if accumulation_mode == 'valid_study_mean'
                      else float(loss_values.mean()))
        train_seconds = time.perf_counter() - train_started
        if scheduler_clock == 'epoch':
            scheduler.step()
        validation_started = time.perf_counter()
        if ema is not None:
            ema.apply(model)
        try:
            if full_train:
                validation_frame, metrics = None, {}
            else:
                validation_frame, metrics = _predict_loader(model, valid_loader, device,
                    valid_frame=valid_frame, fold=fold, pseudo_eval_min_confidence=pseudo_eval_min_confidence)
            evaluated_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        finally:
            if ema is not None:
                ema.restore(model)
        validation_seconds = time.perf_counter() - validation_started
        gold_score, all_score, pseudo_score = [metrics.get(k, {}).get('macro_auc') for k in ('gold', 'all_observed', 'pseudo')]
        if full_train:
            selected_score = float(epoch)
        elif selection_metric == 'gold_macro_auc':
            selected_score = gold_score if gold_score is not None else all_score
        else:
            selected_score = {'all_observed_macro_auc': all_score, 'pseudo_macro_auc': pseudo_score,
                              'last_epoch': float(epoch)}[selection_metric]
        score = -float('inf') if selected_score is None else float(selected_score)
        epoch_predictions = None
        if train_config.get('save_epoch_predictions', False):
            epoch_path = output_dir / f'epoch_{epoch:03d}_predictions.csv'
            atomic_write(epoch_path, lambda f: f.write(validation_frame.to_csv(index=False).encode()))
            epoch_predictions = {'name': epoch_path.name, 'sha256': file_sha256(epoch_path)}
        epoch_record = dict(epoch=epoch, train_loss=train_loss,
            learning_rate=float(optimizer.param_groups[0]['lr']), optimizer_steps=attempted_steps,
            learning_rates_at_epoch_start=epoch_initial_learning_rates,
            scheduler_clock=scheduler_clock, accumulation_normalization=accumulation_mode,
            valid_training_studies=epoch_valid_studies,
            attempted_optimizer_steps=attempted_steps, successful_optimizer_steps=successful_steps,
            skipped_optimizer_steps=skipped_steps, ema=ema is not None,
            ema_update_policy='legacy_attempt', metrics=metrics,
            train_seconds=train_seconds, validation_seconds=validation_seconds)
        if epoch_predictions is not None:
            epoch_record['validation_predictions'] = epoch_predictions
        if epoch + 1 in weight_epochs:
            weight_path = output_dir / f'weights_epoch_{epoch:03d}.pt'
            save_torch(weight_path, {
                'model_state': evaluated_state,
                'raw_model_state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                'selected_weights': 'ema' if ema else 'raw', 'config': config,
                'fold': fold, 'epoch': epoch, 'validation_scope': validation_scope,
                'selection_metric': 'fixed_epoch_snapshot', 'metrics': metrics,
                'training_identity': identity})
            epoch_record['weight_snapshot'] = dict(name=weight_path.name,
                sha256=file_sha256(weight_path), bytes=weight_path.stat().st_size)
        if local_telemetry is not None:
            epoch_record['local_head_mechanics'] = dict(first_batch=local_probe, first_updates=local_telemetry,
                raw_readout_abs_max=float(model.local_head.head_weights.detach().abs().max()),
                ema_readout_abs_max=float(ema.shadow['local_head.head_weights'].abs().max()) if ema else None)
        history.append(epoch_record)
        improved = best_epoch < 0 or score > best_score
        if improved:
            best_score, best_epoch, best_metrics = score, epoch, metrics
            checkpoint_path = output_dir / f'best_epoch_{epoch:03d}.pt'
            predictions_path = output_dir / f'best_epoch_{epoch:03d}_predictions.csv'
            save_torch(checkpoint_path, {
                'model_state': evaluated_state,
                'raw_model_state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                'selected_weights': 'ema' if ema else 'raw', 'config': config,
                'fold': fold, 'epoch': epoch, 'validation_scope': validation_scope,
                'selection_metric': selection_metric, 'metrics': metrics, 'training_identity': identity})
            if not full_train:
                atomic_write(predictions_path, lambda f: f.write(validation_frame.to_csv(index=False).encode()))
            selected_artifacts = [('checkpoint', checkpoint_path)]
            if not full_train:
                selected_artifacts.append(('predictions', predictions_path))
            best_artifacts = {key: {'name': path.name, 'sha256': file_sha256(path)}
                              for key, path in selected_artifacts}
        # The last_training commit binds immutable best weights+OOF to the same history.
        commit_training_state(epoch + 1)
        save_json(output_dir / 'history.json', history)
        if improved:
            export_best_artifacts(output_dir, best_artifacts)
        print(json.dumps(epoch_record, ensure_ascii=False), flush=True)
        del evaluated_state
        if plateau_stop(history, best_epoch, stop_policy, epochs):
            early_stopped = True
            break
    complete = len(history) == epochs or early_stopped
    raw_metrics = None
    if complete and ema is not None and bool(train_config.get('report_raw_at_completion', False)):
        selected = torch.load(output_dir / 'best.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(selected['raw_model_state'])
        raw_frame, raw_metrics = _predict_loader(model, valid_loader, device,
            valid_frame=valid_frame, fold=fold, pseudo_eval_min_confidence=pseudo_eval_min_confidence)
        atomic_write(output_dir / 'raw_validation_predictions.csv', lambda f: f.write(raw_frame.to_csv(index=False).encode()))
    summary = dict(fold=fold, status='complete' if complete else 'paused', completed_epochs=len(history),
        planned_epochs=epochs, validation_scope=validation_scope, selection_metric=selection_metric,
        selected_weights='ema' if ema else 'raw', ema_decay=ema_decay, ema_update_policy='legacy_attempt',
        train_studies=len(train_frame), validation_studies=len(valid_frame),
        validation_gold_studies=int(valid_frame.get('is_gold', pd.Series(False, index=valid_frame.index)).astype(bool).sum()),
        best_epoch=best_epoch, best_selection_score=best_score if np.isfinite(best_score) else None,
        best_metrics=best_metrics, raw_metrics_at_selected_epoch=raw_metrics, device=str(device),
        parameters=parameter_summary(model), optimizer_groups=initial_optimizer_groups,
        attempted_optimizer_steps=attempted_steps, successful_optimizer_steps=successful_steps,
        skipped_optimizer_steps=skipped_steps, training_identity_sha256=identity['sha256'],
        scheduler_clock=scheduler_clock, accumulation_normalization=accumulation_mode,
        best_artifacts=best_artifacts)
    if early_stopped:
        summary['termination_reason'] = 'predeclared_plateau_stop'
        summary['plateau_stop'] = stop_policy
    save_json(output_dir / 'history.json', history)
    save_json(output_dir / 'summary.json', summary)
    if complete:
        names = ['best.pt', 'summary.json', 'history.json', 'training_identity.json', 'resolved_config.yaml']
        names += ['training_studies.csv'] if full_train else ['oof_predictions.csv', 'best_validation_predictions.csv']
        if raw_metrics is not None:
            names.append('raw_validation_predictions.csv')
        if train_config.get('save_epoch_predictions', False):
            for record in history:
                receipt = record['validation_predictions']
                if file_sha256(output_dir / receipt['name']) != receipt['sha256']:
                    raise ValueError('Epoch prediction receipt differs before completion')
                names.append(receipt['name'])
        for record in history:
            if record['epoch'] + 1 in weight_epochs:
                receipt = record['weight_snapshot']
                if file_sha256(output_dir / receipt['name']) != receipt['sha256']:
                    raise ValueError('Weight snapshot receipt differs before completion')
                names.append(receipt['name'])
        save_json(output_dir / 'COMPLETED.json', {'fold': fold, 'completed_epochs': len(history),
            'training_identity_sha256': identity['sha256'], 'best_epoch': best_epoch,
            'files': {name: {'sha256': file_sha256(output_dir / name), 'bytes': (output_dir / name).stat().st_size}
                      for name in names}})
    return summary
