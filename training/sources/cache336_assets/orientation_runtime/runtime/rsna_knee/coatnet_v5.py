"""Exact CoAtNet V5 model contract and window layouts.

The baseline functions mirror the frozen DINOsaur V14 CoAtNet block.  New
window layouts are explicit opt-in modes so importing this module cannot
silently change the 62-window leaderboard recipe.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np

from .constants import TARGET_COLUMNS


CHECKPOINT_FILENAME = "raptor_ft_coatnet_v5_full_swa.pt"
CHECKPOINT_SHA256 = (
    "20d45e0732a0e9120025d0172a4d2dd0170900bc250b06742dfb7ae08a63786a"
)
FROZEN_PREPROCESSING_BUNDLE_SHA256 = (
    "07f04d3387782a26a14d1a3a68df923e700ed3b5e2e3aebcb270d030db9c3470"
)
FROZEN_MODEL_DEFINITION_BUNDLE_SHA256 = (
    "5117cf8122185f78106e9e2d0a05472e22e9057df4ef58259f3027dd97b7ffb0"
)
BASELINE_CACHE_ARM = "coatnet_v5_maxspan62_336to384_ordinary_v1"
ARCHITECTURE = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"
SOURCE_IMAGE_SIZE = 336
MODEL_RESOLUTION = 384
CROP_MM = 140.0
FEATURE_DIM = 1024
TARGET_COUNT = 12
TARGETS = tuple(TARGET_COLUMNS)
BASELINE_WINDOW_COUNT = 62
PRIMARY_SPAN = (0.02, 0.98)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class SlotSpec:
    plane: str
    fluid_sensitive: int
    slice_count: int


SLOT_SPECS = (
    SlotSpec("Sagittal", 1, 18),
    SlotSpec("Sagittal", 0, 14),
    SlotSpec("Coronal", 1, 12),
    SlotSpec("Coronal", 0, 8),
    SlotSpec("Axial", -1, 12),
)
MAX_SLICES = sum(slot.slice_count for slot in SLOT_SPECS)


@dataclass(frozen=True)
class WindowMetadata:
    """Geometry and validity of one three-slice CoAtNet input."""

    window_index: int
    center_index: int
    source_indices: tuple[int, int, int]
    source_slot_indices: tuple[int, int, int]
    center_slot_index: int
    local_center_index: int
    normalized_slot_position: float
    crosses_slot_boundary: bool
    all_slices_valid: bool
    is_duplicate_center: bool
    slot_safe: bool
    layout: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


WindowLayout = Literal["baseline", "slot_safe"]


def slot_ranges(
    slots: Sequence[SlotSpec] = SLOT_SPECS,
) -> tuple[tuple[int, int], ...]:
    """Return half-open slice ranges for the fixed slot sequence."""

    ranges: list[tuple[int, int]] = []
    start = 0
    for slot in slots:
        if slot.slice_count < 3:
            raise ValueError("Every slot must contain at least three slice positions")
        stop = start + int(slot.slice_count)
        ranges.append((start, stop))
        start = stop
    return tuple(ranges)


def slice_slot_indices(
    slots: Sequence[SlotSpec] = SLOT_SPECS,
) -> np.ndarray:
    """Map every fixed volume position to its slot index."""

    values: list[int] = []
    for slot_index, slot in enumerate(slots):
        values.extend([slot_index] * int(slot.slice_count))
    return np.asarray(values, dtype=np.int16)


def frozen_eval_centers(mask: np.ndarray, depth: int, count: int) -> list[int]:
    """Mirror the exact frozen V5 ``_eval_centers`` implementation."""

    if depth < 3:
        raise ValueError("depth must be at least three")
    if count <= 0:
        raise ValueError("count must be positive")
    values = np.asarray(mask).reshape(-1)
    if len(values) != depth:
        raise ValueError(f"mask length {len(values)} does not match depth {depth}")
    valid = np.where(values > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, depth))
    low, high = int(valid.min()), int(valid.max())
    centers = [
        center
        for center in range(low + 1, high)
        if center - 1 >= low and center + 1 <= high
    ]
    if not centers:
        centers = [max(1, min((low + high) // 2, depth - 2))]
    indices = np.linspace(0, len(centers) - 1, count).round().astype(int)
    return [centers[index] for index in indices]


def _window_metadata(
    centers: Iterable[int],
    mask: np.ndarray,
    *,
    layout: WindowLayout,
    slots: Sequence[SlotSpec],
) -> list[WindowMetadata]:
    mask_values = np.asarray(mask).reshape(-1)
    slot_by_slice = slice_slot_indices(slots)
    if len(mask_values) != len(slot_by_slice):
        raise ValueError(
            f"mask length {len(mask_values)} does not match slot depth "
            f"{len(slot_by_slice)}"
        )
    ranges = slot_ranges(slots)
    records: list[WindowMetadata] = []
    seen_centers: set[int] = set()
    for window_index, center in enumerate(centers):
        if center <= 0 or center >= len(mask_values) - 1:
            raise ValueError(f"center {center} cannot form a three-slice window")
        source = (center - 1, center, center + 1)
        source_slots = tuple(int(slot_by_slice[index]) for index in source)
        center_slot = int(slot_by_slice[center])
        start, stop = ranges[center_slot]
        denominator = max(1, stop - start - 1)
        crosses_slot_boundary = len(set(source_slots)) != 1
        all_slices_valid = bool(np.all(mask_values[list(source)] > 0))
        is_duplicate_center = center in seen_centers
        seen_centers.add(center)
        records.append(
            WindowMetadata(
                window_index=window_index,
                center_index=int(center),
                source_indices=source,
                source_slot_indices=source_slots,
                center_slot_index=center_slot,
                local_center_index=int(center - start),
                normalized_slot_position=float((center - start) / denominator),
                crosses_slot_boundary=crosses_slot_boundary,
                all_slices_valid=all_slices_valid,
                is_duplicate_center=is_duplicate_center,
                slot_safe=bool(
                    all_slices_valid
                    and not crosses_slot_boundary
                    and not is_duplicate_center
                ),
                layout=layout,
            )
        )
    return records


def baseline_window_metadata(
    mask: np.ndarray,
    *,
    count: int = BASELINE_WINDOW_COUNT,
    slots: Sequence[SlotSpec] = SLOT_SPECS,
) -> list[WindowMetadata]:
    """Describe the unchanged global V5 layout, including mixed windows."""

    depth = sum(slot.slice_count for slot in slots)
    centers = frozen_eval_centers(mask, depth, count)
    return _window_metadata(centers, mask, layout="baseline", slots=slots)


def slot_safe_window_metadata(
    mask: np.ndarray,
    *,
    slots: Sequence[SlotSpec] = SLOT_SPECS,
) -> list[WindowMetadata]:
    """Create unique, valid windows wholly contained inside one slot.

    Missing slots and internal zero-filled slices reduce the number of windows;
    the function never pads by duplication or crosses into a neighbouring slot.
    A complete five-slot study produces 54 windows.
    """

    mask_values = np.asarray(mask).reshape(-1)
    expected_depth = sum(slot.slice_count for slot in slots)
    if len(mask_values) != expected_depth:
        raise ValueError(
            f"mask length {len(mask_values)} does not match slot depth {expected_depth}"
        )
    centers: list[int] = []
    for start, stop in slot_ranges(slots):
        for center in range(start + 1, stop - 1):
            if np.all(mask_values[center - 1 : center + 2] > 0):
                centers.append(center)
    return _window_metadata(centers, mask_values, layout="slot_safe", slots=slots)


def baseline_to_slot_safe_indices(
    mask: np.ndarray,
    *,
    count: int = BASELINE_WINDOW_COUNT,
    slots: Sequence[SlotSpec] = SLOT_SPECS,
) -> np.ndarray:
    """Map slot-safe windows to rows of the baseline feature tensor.

    The frozen global layout may duplicate centers in incomplete studies.  The
    first matching baseline row is used, so missing data is never replaced by
    artificial repeated safe windows.
    """

    baseline = baseline_window_metadata(mask, count=count, slots=slots)
    safe = slot_safe_window_metadata(mask, slots=slots)
    first_by_center: dict[int, int] = {}
    for record in baseline:
        first_by_center.setdefault(record.center_index, record.window_index)
    missing = [record.center_index for record in safe if record.center_index not in first_by_center]
    if missing:
        raise RuntimeError(f"safe centers absent from baseline layout: {missing}")
    return np.asarray(
        [first_by_center[record.center_index] for record in safe], dtype=np.int16
    )


def metadata_arrays(records: Sequence[WindowMetadata]) -> dict[str, np.ndarray]:
    """Materialize stable arrays suitable for a compressed feature cache."""

    return {
        "window_index": np.asarray([record.window_index for record in records], dtype=np.int16),
        "center_index": np.asarray([record.center_index for record in records], dtype=np.int16),
        "source_indices": np.asarray(
            [record.source_indices for record in records], dtype=np.int16
        ),
        "source_slot_indices": np.asarray(
            [record.source_slot_indices for record in records], dtype=np.int8
        ),
        "center_slot_index": np.asarray(
            [record.center_slot_index for record in records], dtype=np.int8
        ),
        "local_center_index": np.asarray(
            [record.local_center_index for record in records], dtype=np.int16
        ),
        "normalized_slot_position": np.asarray(
            [record.normalized_slot_position for record in records], dtype=np.float32
        ),
        "crosses_slot_boundary": np.asarray(
            [record.crosses_slot_boundary for record in records], dtype=np.bool_
        ),
        "all_slices_valid": np.asarray(
            [record.all_slices_valid for record in records], dtype=np.bool_
        ),
        "is_duplicate_center": np.asarray(
            [record.is_duplicate_center for record in records], dtype=np.bool_
        ),
        "slot_safe": np.asarray(
            [record.slot_safe for record in records], dtype=np.bool_
        ),
        "layout": np.asarray([record.layout for record in records], dtype="U16"),
    }


def repeated_source_mask(
    records: Sequence[WindowMetadata],
    slice_source_positions: np.ndarray,
) -> np.ndarray:
    """Flag within-slot triplets that reuse one sampled source DICOM.

    Geometric slot safety only proves that three volume positions belong to the
    same selected series. Short series can still make the frozen linspace
    sampler repeat a source position; callers must persist those positions to
    distinguish that case from three unique neighbouring acquisitions.
    """

    positions = np.asarray(slice_source_positions).reshape(-1)
    if len(positions) != MAX_SLICES:
        raise ValueError(
            f"source-position length {len(positions)} does not match {MAX_SLICES}"
        )
    result = []
    for record in records:
        values = [int(positions[index]) for index in record.source_indices]
        valid = all(value >= 0 for value in values)
        within_slot = not record.crosses_slot_boundary
        result.append(bool(valid and within_slot and len(set(values)) != len(values)))
    return np.asarray(result, dtype=np.bool_)


def eval_windows(
    volume: np.ndarray,
    mask: np.ndarray,
    *,
    resolution: int = MODEL_RESOLUTION,
    normalization: Literal["imagenet", "none"] = "imagenet",
    layout: WindowLayout = "baseline",
    baseline_count: int = BASELINE_WINDOW_COUNT,
) -> tuple[Any, list[WindowMetadata]]:
    """Build normalized `[K,3,H,W]` windows and return their metadata."""

    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as error:
        raise RuntimeError("CoAtNet window tensors require torch") from error

    values = np.asarray(volume)
    if values.ndim != 3 or values.shape[0] != MAX_SLICES:
        raise ValueError(f"expected volume [{MAX_SLICES},H,W], got {values.shape}")
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if layout == "baseline":
        records = baseline_window_metadata(mask, count=baseline_count)
    elif layout == "slot_safe":
        records = slot_safe_window_metadata(mask)
    else:
        raise ValueError(f"unknown window layout: {layout}")
    if not records:
        raise ValueError(f"window layout {layout!r} produced no valid windows")

    windows = np.empty((len(records), 3, resolution, resolution), np.float32)
    for output_index, record in enumerate(records):
        triplet = values[list(record.source_indices)].astype(np.float32) / 255.0
        tensor = torch.from_numpy(triplet)
        if tensor.shape[-2:] != (resolution, resolution):
            tensor = functional.interpolate(
                tensor[None],
                size=(resolution, resolution),
                mode="bilinear",
                align_corners=False,
            )[0]
        windows[output_index] = tensor.numpy()
    result = torch.from_numpy(windows)
    if normalization == "imagenet":
        mean = torch.tensor(IMAGENET_MEAN, dtype=result.dtype).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=result.dtype).view(3, 1, 1)
        result = (result - mean) / std
    elif normalization != "none":
        raise ValueError(f"unknown normalization: {normalization}")
    return result, records


try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only without train extras
    torch = None
    nn = None


if nn is not None:

    class RaptorClassifier(nn.Module):
        """Exact V5 target-wise attention and classifier head."""

        def __init__(self, backbone: nn.Module, feature_dim: int, targets: int = 12, drop: float = 0.2):
            super().__init__()
            self.backbone = backbone
            self.norm = nn.LayerNorm(feature_dim)
            self.att = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.Tanh(),
                nn.Dropout(drop),
                nn.Linear(256, targets),
            )
            self.clsW = nn.Parameter(torch.zeros(targets, feature_dim))
            self.clsb = nn.Parameter(torch.zeros(targets))
            nn.init.trunc_normal_(self.clsW, std=0.02)
            self.n = targets

        def encode(self, images: Any) -> Any:
            batch, windows = images.shape[:2]
            features = self.backbone(images.flatten(0, 1))
            return features.view(batch, windows, -1)

        def head(self, features: Any) -> Any:
            hidden = self.norm(features)
            attention = torch.softmax(self.att(hidden), dim=1)
            pooled = torch.einsum("bkn,bkf->bnf", attention, hidden)
            return (pooled * self.clsW).sum(-1) + self.clsb

        def forward(self, images: Any) -> Any:
            return self.head(self.encode(images))


def build_backbone(architecture: str = ARCHITECTURE, *, pretrained: bool = False) -> Any:
    """Build the backbone with the same timm routing as the frozen source."""

    try:
        import timm
    except ImportError as error:
        raise RuntimeError("CoAtNet V5 requires timm") from error
    hybrid = architecture.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = (not hybrid) and any(
        token in architecture for token in ("vit", "deit", "dinov2", "eva", "beit")
    )
    arguments: dict[str, Any] = {
        "pretrained": pretrained,
        "num_classes": 0,
        "in_chans": 3,
    }
    if is_vit:
        arguments.update(global_pool="token", dynamic_img_size=True)
    else:
        arguments.update(global_pool="avg")
    return timm.create_model(architecture, **arguments)


def load_model(
    checkpoint_path: str | Path,
    *,
    device: str | Any = "cpu",
    verify_identity: bool = True,
) -> tuple[Any, int]:
    """Strictly load the exact V5 checkpoint and return model and resolution."""

    if torch is None or nn is None:
        raise RuntimeError("CoAtNet V5 model loading requires torch")
    checkpoint_path = Path(checkpoint_path)
    if verify_identity:
        digest = hashlib.sha256()
        with checkpoint_path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 << 20), b""):
                digest.update(block)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != CHECKPOINT_SHA256:
            raise RuntimeError(
                f"checkpoint SHA drift: {actual_sha256} != {CHECKPOINT_SHA256}"
            )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    architecture = str(checkpoint.get("arch", ARCHITECTURE))
    resolution = int(checkpoint.get("res", MODEL_RESOLUTION))
    labels = tuple(str(value) for value in checkpoint.get("lab", ()))
    if architecture != ARCHITECTURE:
        raise RuntimeError(f"checkpoint architecture drift: {architecture}")
    if resolution != MODEL_RESOLUTION:
        raise RuntimeError(f"checkpoint resolution drift: {resolution}")
    if labels != TARGETS:
        raise RuntimeError(f"checkpoint target order drift: {labels}")
    backbone = build_backbone(architecture, pretrained=False)
    feature_dim = int(backbone.num_features)
    if feature_dim != FEATURE_DIM:
        raise RuntimeError(f"feature width drift: {feature_dim} != {FEATURE_DIM}")
    model = RaptorClassifier(backbone, feature_dim=feature_dim, targets=TARGET_COUNT)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().to(device)
    return model, resolution


def encode_features(
    model: Any,
    windows: Any,
    *,
    device: str | Any,
    chunk_size: int | None = None,
) -> Any:
    """Encode windows, optionally in memory-bounded chunks."""

    if torch is None:
        raise RuntimeError("CoAtNet V5 feature extraction requires torch")
    if windows.ndim != 4:
        raise ValueError(f"expected [K,3,H,W] windows, got {tuple(windows.shape)}")
    count = int(windows.shape[0])
    step = count if chunk_size is None else int(chunk_size)
    if step <= 0:
        raise ValueError("chunk_size must be positive")
    features = []
    model.eval()
    with torch.no_grad():
        for start in range(0, count, step):
            batch = windows[start : start + step].to(device)
            features.append(model.backbone(batch).float().cpu())
    return torch.cat(features, dim=0)


def predict_from_features(model: Any, features: Any, *, device: str | Any) -> np.ndarray:
    """Apply the exact saved attention head to one study's feature matrix."""

    if torch is None:
        raise RuntimeError("CoAtNet V5 prediction requires torch")
    if features.ndim != 2 or int(features.shape[1]) != FEATURE_DIM:
        raise ValueError(
            f"expected [K,{FEATURE_DIM}] features, got {tuple(features.shape)}"
        )
    model.eval()
    with torch.no_grad():
        logits = model.head(features[None].to(device)).float()
        result = torch.sigmoid(logits)[0].cpu().numpy().astype(np.float32)
    if result.shape != (TARGET_COUNT,) or not np.isfinite(result).all():
        raise RuntimeError("non-finite or malformed CoAtNet prediction")
    return result
