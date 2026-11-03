"""Small, deterministic overfit diagnostic for the complete weak-label path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .constants import ID_COLUMN, TARGET_COLUMNS
from .dataset import StudyDataset
from .model import create_model, parameter_summary
from .train import (
    _require_torch,
    build_optimizer,
    optimizer_summary,
    seed_everything,
    weighted_bce_loss,
)


def _resolve_device(torch: Any, requested: str | None) -> Any:
    if requested:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _evaluate_subset(
    model: Any,
    loader: Any,
    device: Any,
) -> tuple[float, float, pd.DataFrame]:
    torch = _require_torch()
    model.eval()
    numerator = 0.0
    denominator = 0.0
    prediction_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            labels = batch["labels"].to(device)
            mask = batch["label_mask"].to(device)
            label_weight = batch["label_weight"].to(device)
            logits = model(
                batch["tokens"].to(device),
                batch["token_mask"].to(device),
                batch["slot_ids"].to(device),
            )
            element_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels, reduction="none"
            )
            weights = mask.float() * label_weight
            numerator += float((element_loss * weights).sum().cpu())
            denominator += float(weights.sum().cpu())
            probabilities = logits.sigmoid().cpu().numpy()
            labels_cpu = labels.cpu().numpy()
            weights_cpu = weights.cpu().numpy()
            for row_index, uid in enumerate(batch["uid"]):
                row: dict[str, Any] = {ID_COLUMN: str(uid)}
                for target_index, target in enumerate(TARGET_COLUMNS):
                    row[f"label__{target}"] = float(labels_cpu[row_index, target_index])
                    row[f"weight__{target}"] = float(weights_cpu[row_index, target_index])
                    row[f"pred__{target}"] = float(probabilities[row_index, target_index])
                prediction_rows.append(row)
    if denominator <= 0:
        raise ValueError("The smoke subset has no non-abstained training targets")
    return numerator / denominator, denominator, pd.DataFrame(prediction_rows)


def run_overfit_smoke(
    config: dict[str, Any],
    manifest: pd.DataFrame,
    series: pd.DataFrame,
    *,
    output_dir: str | Path,
    max_studies: int = 2,
    steps: int = 20,
    evaluate_every: int = 2,
    learning_rate: float = 3e-3,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Fit and evaluate on the same tiny subset.

    A decreasing loss proves only that gradients connect report-derived targets to
    real DICOM tensors.  It is deliberately not a validation or leaderboard metric.
    """

    if max_studies <= 0 or steps <= 0 or evaluate_every <= 0:
        raise ValueError("max_studies, steps, and evaluate_every must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    missing = sorted({ID_COLUMN, *TARGET_COLUMNS}.difference(manifest.columns))
    if missing:
        raise ValueError(f"Training manifest is missing columns: {missing}")

    numeric_labels = manifest[TARGET_COLUMNS].apply(pd.to_numeric, errors="coerce")
    subset = manifest.loc[numeric_labels.notna().any(axis=1)].head(max_studies).copy()
    if subset.empty:
        raise ValueError("The smoke manifest has no usable studies")

    torch = _require_torch()
    from torch.utils.data import DataLoader

    seed = int(config.get("seed", 20260828))
    seed_everything(seed)
    device = _resolve_device(torch, device_name)
    dataset = StudyDataset(subset, series, config, training=False, include_labels=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = create_model(config).to(device)
    smoke_train_config = dict(config["train"])
    smoke_train_config["learning_rate"] = float(learning_rate)
    optimizer = build_optimizer(model, smoke_train_config)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    initial_loss, total_weight, initial_predictions = _evaluate_subset(
        model, loader, device
    )
    initial_predictions.to_csv(output_dir / "predictions_step_000.csv", index=False)
    history: list[dict[str, Any]] = [
        {
            "step": 0,
            "subset_weighted_bce": float(initial_loss),
            "train_batch_loss": None,
        }
    ]
    iterator = iter(loader)
    final_predictions = initial_predictions
    last_batch_loss: float | None = None
    optimizer_steps = 0
    skipped_all_abstain_batches = 0

    while optimizer_steps < steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            batch["tokens"].to(device),
            batch["token_mask"].to(device),
            batch["slot_ids"].to(device),
        )
        loss = weighted_bce_loss(
            logits,
            batch["labels"].to(device),
            batch["label_mask"].to(device),
            batch["label_weight"].to(device),
        )
        if loss is None:
            skipped_all_abstain_batches += 1
            continue
        loss.backward()
        optimizer.step()
        optimizer_steps += 1
        step = optimizer_steps
        last_batch_loss = float(loss.detach().cpu())

        if step % evaluate_every == 0 or step == steps:
            subset_loss, _, final_predictions = _evaluate_subset(model, loader, device)
            history.append(
                {
                    "step": int(step),
                    "subset_weighted_bce": float(subset_loss),
                    "train_batch_loss": last_batch_loss,
                }
            )

    final_loss = float(history[-1]["subset_weighted_bce"])
    final_predictions.to_csv(output_dir / "predictions_final.csv", index=False)
    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, ensure_ascii=False, indent=2, allow_nan=False)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "steps": int(steps),
            "study_uids": subset[ID_COLUMN].astype(str).tolist(),
            "diagnostic_only": True,
        },
        output_dir / "overfit_smoke.pt",
    )

    reduction = float(initial_loss - final_loss)
    summary = {
        "status": "PASS" if reduction > 0 else "NO_LOSS_DECREASE",
        "studies": int(len(subset)),
        "study_uids": subset[ID_COLUMN].astype(str).tolist(),
        "optimizer_steps": int(steps),
        "skipped_all_abstain_batches": int(skipped_all_abstain_batches),
        "learning_rate": float(learning_rate),
        "device": str(device),
        "model_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "parameters": parameter_summary(model),
        "optimizer_groups": optimizer_summary(optimizer),
        "active_target_weight": float(total_weight),
        "initial_subset_weighted_bce": float(initial_loss),
        "final_subset_weighted_bce": final_loss,
        "absolute_loss_reduction": reduction,
        "relative_loss_reduction": float(reduction / initial_loss),
        "interpretation_boundary": (
            "Same-study overfit diagnostic only: proves end-to-end trainability, not "
            "generalization, pseudo-label validity, clinical performance, or Kaggle AUC."
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return summary
