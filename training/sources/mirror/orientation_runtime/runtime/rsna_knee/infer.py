"""Checkpoint ensembling and Kaggle submission creation."""

from __future__ import annotations

from .batch_inputs import model_inputs
from .input_contract import require_input_contract

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .constants import ID_COLUMN, TARGET_COLUMNS
from .dataset import create_study_dataset
from .model import create_model


def predict_submission(
    config: dict[str, Any],
    test: pd.DataFrame,
    series: pd.DataFrame | None,
    checkpoints: Sequence[str | Path],
    *,
    output_path: str | Path,
) -> pd.DataFrame:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as error:
        raise RuntimeError("Inference requires PyTorch; install the 'train' extra") from error
    if not checkpoints:
        raise ValueError("At least one checkpoint is required")
    if ID_COLUMN not in test or test[ID_COLUMN].duplicated().any():
        raise ValueError("test.csv must be unique by StudyInstanceUID")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    dataset = create_study_dataset(
        test, series, config, training=False, include_labels=False
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["train"].get("batch_size", 2)),
        shuffle=False,
        num_workers=int(config["train"].get("num_workers", 2)),
        pin_memory=device.type == "cuda",
    )
    ensemble: list[np.ndarray] = []
    reference_uids: list[str] | None = None
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        require_input_contract(config, checkpoint)
        # The full task checkpoint already contains the encoder weights.  Avoid
        # downloading or reloading the pretrained backbone for every ensemble
        # member before the strict checkpoint readback below.
        model = create_model(config, initialize_backbone=False)
        state = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        predictions: list[np.ndarray] = []
        uids: list[str] = []
        with torch.inference_mode():
            for batch in loader:
                logits = model(*model_inputs(batch, device))
                predictions.append(logits.sigmoid().cpu().numpy())
                uids.extend(batch["uid"])
        if reference_uids is None:
            reference_uids = uids
        elif reference_uids != uids:
            raise AssertionError("Checkpoint inference changed study ordering")
        ensemble.append(np.concatenate(predictions, axis=0))

    probabilities = np.mean(np.stack(ensemble, axis=0), axis=0)
    if probabilities.shape != (len(test), len(TARGET_COLUMNS)):
        raise AssertionError(f"Unexpected prediction shape: {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise ValueError("Predictions contain NaN or infinite values")
    submission = pd.DataFrame(probabilities, columns=TARGET_COLUMNS)
    submission.insert(0, ID_COLUMN, reference_uids)
    if list(submission.columns) != [ID_COLUMN, *TARGET_COLUMNS]:
        raise AssertionError("Submission schema mismatch")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission
