"""Single-model 2.5D label-query MIL network."""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any

from .constants import TARGET_COLUMNS

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


if nn is not None:

    class TinyEncoder(nn.Module):
        def __init__(self, output_dim: int) -> None:
            super().__init__()
            channels = [3, 32, 64, 128, 256]
            blocks = []
            for input_channels, output_channels in zip(channels[:-1], channels[1:], strict=True):
                blocks.extend(
                    [
                        nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(output_channels),
                        nn.SiLU(inplace=True),
                    ]
                )
            self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1))
            self.projection = nn.Linear(channels[-1], output_dim)

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return self.projection(self.features(images).flatten(1))


    class TimmEncoder(nn.Module):
        def __init__(
            self,
            name: str,
            output_dim: int,
            *,
            pretrained: bool,
            checkpoint: str | None,
            image_size: int,
            global_pool: str,
            input_normalization: str,
            finetune_mode: str,
            unfreeze_last_blocks: int,
            initialize_backbone: bool,
            drop_path_rate: float | None = None,
        ) -> None:
            super().__init__()
            try:
                import timm
            except ImportError as error:
                raise RuntimeError("A timm backbone requires the 'timm' package") from error
            create_kwargs: dict[str, Any] = {
                "pretrained": bool(
                    initialize_backbone and (pretrained or checkpoint is not None)
                ),
                "num_classes": 0,
                "global_pool": global_pool,
            }
            if drop_path_rate is not None:
                if not math.isfinite(drop_path_rate) or not 0.0 <= drop_path_rate < 1.0:
                    raise ValueError("model.drop_path_rate must lie in [0, 1)")
                create_kwargs["drop_path_rate"] = drop_path_rate
            if checkpoint and initialize_backbone:
                # Route local files through timm's model-specific checkpoint
                # conversion and positional-embedding filter instead of a loose
                # strict=False state-dict load.
                create_kwargs["pretrained_cfg_overlay"] = {"file": checkpoint}
            # DINOv2 uses 14 px patches and pretrained positional embeddings from
            # 518 px inputs.  Supplying the experiment size lets timm resample the
            # positional grid once instead of silently keeping an incompatible one.
            if "dinov2" in name:
                create_kwargs["img_size"] = int(image_size)
            self.backbone = timm.create_model(name, **create_kwargs)
            feature_count = int(self.backbone.num_features)
            self.projection = (
                nn.Identity() if feature_count == output_dim else nn.Linear(feature_count, output_dim)
            )
            self.finetune_mode = finetune_mode
            self._trainable_backbone_modules: list[nn.Module] = []

            if input_normalization == "pretrained":
                pretrained_config = getattr(self.backbone, "pretrained_cfg", {})
                mean = tuple(float(value) for value in pretrained_config.get("mean", ()))
                std = tuple(float(value) for value in pretrained_config.get("std", ()))
                if len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
                    raise ValueError(
                        f"Backbone {name!r} has no valid three-channel pretrained mean/std"
                    )
            else:
                mean = (0.0, 0.0, 0.0)
                std = (1.0, 1.0, 1.0)
            self.register_buffer(
                "input_mean",
                torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                "input_std",
                torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1),
                persistent=False,
            )
            self._configure_finetuning(finetune_mode, unfreeze_last_blocks)

        @staticmethod
        def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
            for parameter in module.parameters():
                parameter.requires_grad = enabled

        def _configure_finetuning(self, mode: str, last_blocks: int) -> None:
            if mode == "full":
                self._set_requires_grad(self.backbone, True)
                return

            self._set_requires_grad(self.backbone, False)
            if mode == "head_only":
                return
            if mode != "last_blocks":
                raise ValueError(f"Unknown model.finetune_mode: {mode}")

            blocks = getattr(self.backbone, "blocks", None)
            if blocks is None:
                raise ValueError(
                    "model.finetune_mode='last_blocks' requires a timm backbone "
                    "with a .blocks collection"
                )
            block_count = len(blocks)
            if last_blocks <= 0 or last_blocks > block_count:
                raise ValueError(
                    f"model.unfreeze_last_blocks must lie in [1, {block_count}], "
                    f"got {last_blocks}"
                )
            for block in list(blocks)[-last_blocks:]:
                self._set_requires_grad(block, True)
                self._trainable_backbone_modules.append(block)
            for attribute in ("norm", "fc_norm"):
                module = getattr(self.backbone, attribute, None)
                if isinstance(module, nn.Module):
                    self._set_requires_grad(module, True)
                    self._trainable_backbone_modules.append(module)

        @property
        def has_trainable_backbone(self) -> bool:
            return any(parameter.requires_grad for parameter in self.backbone.parameters())

        def train(self, mode: bool = True) -> TimmEncoder:
            super().train(mode)
            if mode and self.finetune_mode != "full":
                # Keep the frozen prefix deterministic while allowing the selected
                # final transformer blocks and normalization layers to train.
                self.backbone.eval()
                for module in self._trainable_backbone_modules:
                    module.train(True)
            return self

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            images = (images - self.input_mean) / self.input_std
            if self.finetune_mode == "head_only":
                with torch.no_grad():
                    features = self.backbone(images)
            else:
                features = self.backbone(images)
            return self.projection(features)

        def forward_with_local(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """One native ConvNeXt pass: pooled stage 4 and NCHW stage 3."""
            images = (images - self.input_mean) / self.input_std
            context = torch.no_grad() if self.finetune_mode == "head_only" else nullcontext()
            with context:
                final_features, intermediates = self.backbone.forward_intermediates(
                    images, indices=[2], stop_early=False, output_fmt="NCHW"
                )
                features = self.backbone.forward_head(final_features)
                local_map = intermediates[0]
            if local_map.ndim != 4 or tuple(local_map.shape[1:]) != (384, 14, 14):
                raise ValueError("L16 requires native NCHW stage[2] maps [N, 384, 14, 14]")
            return self.projection(features), local_map


    class RadImageNetResNet50Encoder(nn.Module):
        """Public RadImageNet ResNet-50 checkpoint used by strong RSNA recipes."""

        def __init__(
            self,
            output_dim: int,
            *,
            checkpoint: str | None,
            finetune_mode: str,
            initialize_backbone: bool,
        ) -> None:
            super().__init__()
            try:
                from torchvision.models import resnet50
            except ImportError as error:
                raise RuntimeError("RadImageNet ResNet-50 requires torchvision") from error
            # The public ResNet50.pt state uses keys backbone.0 ... backbone.7
            # and was created from children()[:-2]. Spatial global averaging is
            # intentionally kept in forward to preserve that exact state schema.
            self.backbone = nn.Sequential(*list(resnet50(weights=None).children())[:-2])
            if initialize_backbone:
                if not checkpoint:
                    raise ValueError(
                        "radimagenet:resnet50 requires model.encoder_checkpoint"
                    )
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                self.load_state_dict(state, strict=True)
            self.projection = (
                nn.Identity() if output_dim == 2048 else nn.Linear(2048, output_dim)
            )
            self.finetune_mode = finetune_mode
            if finetune_mode == "head_only":
                for parameter in self.backbone.parameters():
                    parameter.requires_grad = False
            elif finetune_mode != "full":
                raise ValueError(
                    "radimagenet:resnet50 supports head_only or full fine-tuning"
                )

        def train(self, mode: bool = True):
            super().train(mode)
            if mode and self.finetune_mode == "head_only":
                self.backbone.eval()
            return self

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            # This is the normalization used by the public RSNA RadImageNet
            # inference notebooks: uint8 / 127.5 - 1.
            images = images.mul(2.0).sub(1.0)
            if self.finetune_mode == "head_only":
                with torch.no_grad():
                    features = self.backbone(images)
            else:
                features = self.backbone(images)
            return self.projection(features.mean(dim=(2, 3)))


    class LocalSpatialHead(nn.Module):
        """Linear label readout over native stride-16 cells, with a scalar residual gate.

        Coordinates are learned row/column indices in the model's 14x14 grid,
        not physical coordinates. No resampling of the feature map is performed.
        """

        grid_size = 14
        feature_dim = 512

        def __init__(self, num_slots: int, num_targets: int, dropout: float, gate_init: float,
                     residual_mode: str = "gated", precision: str = "inherit"):
            super().__init__()
            if residual_mode not in ("gated", "zero_readout"):
                raise ValueError("model.local_residual_mode must be gated or zero_readout")
            if precision not in ("inherit", "fp32"):
                raise ValueError("model.local_precision must be inherit or fp32")
            if residual_mode == "zero_readout" and (gate_init != 0 or precision != "fp32"):
                raise ValueError("zero_readout requires local_precision=fp32 and no local_gate_init override")
            self.residual_mode = residual_mode
            self.precision = precision
            if not math.isfinite(gate_init):
                raise ValueError("model.local_gate_init must be finite")
            self.projection = nn.Linear(384, self.feature_dim)
            self.slot_embedding = nn.Embedding(num_slots, self.feature_dim)
            self.row_embedding = nn.Embedding(self.grid_size, self.feature_dim)
            self.column_embedding = nn.Embedding(self.grid_size, self.feature_dim)
            self.normalization = nn.LayerNorm(self.feature_dim)
            self.dropout = nn.Dropout(dropout)
            self.label_queries = nn.Parameter(torch.empty(num_targets, self.feature_dim))
            self.head_weights = nn.Parameter(torch.empty(num_targets, self.feature_dim))
            self.head_bias = nn.Parameter(torch.zeros(num_targets))
            if residual_mode == "gated":
                self.gate = nn.Parameter(torch.tensor(float(gate_init)))
            else:
                self.register_parameter("gate", None)
            nn.init.normal_(self.label_queries, std=0.02)
            nn.init.normal_(self.head_weights, std=0.02)
            nn.init.normal_(self.row_embedding.weight, std=0.02)
            nn.init.normal_(self.column_embedding.weight, std=0.02)
            if residual_mode == "zero_readout":
                # The initial residual is zero, but its per-label readout learns
                # immediately. Upstream local parameters wake after that update.
                nn.init.zeros_(self.head_weights)

        def _precision_context(self, tensor: torch.Tensor):
            from contextlib import nullcontext
            return (torch.autocast(tensor.device.type, enabled=False)
                    if self.precision == "fp32" else nullcontext())

        def prepare_features(self, maps: torch.Tensor, slot_ids: torch.Tensor) -> torch.Tensor:
            """[B,T,384,14,14] -> projected, embedded, normalized [B,T,14,14,512]."""
            with self._precision_context(maps):
                return self._prepare_features(maps.float() if self.precision == "fp32" else maps, slot_ids)

        def _prepare_features(self, maps: torch.Tensor, slot_ids: torch.Tensor) -> torch.Tensor:
            features = self.projection(maps.permute(0, 1, 3, 4, 2))
            positions = self.row_embedding.weight[:, None] + self.column_embedding.weight[None, :]
            features = features + self.slot_embedding(slot_ids)[:, :, None, None] + positions
            return self.dropout(self.normalization(features))

        def readout(
            self,
            features: torch.Tensor,
            cell_mask: torch.Tensor,
            *,
            diagnostics: bool = False,
        ) -> torch.Tensor | dict[str, torch.Tensor]:
            with self._precision_context(features):
                return self._readout(features.float() if self.precision == "fp32" else features,
                                     cell_mask, diagnostics=diagnostics)

        def _readout(self, features, cell_mask, *, diagnostics=False):
            """Replay post-dropout features; masks may be shared or label-specific.

            A [B,T,14,14] mask applies to all labels; [B,T,14,14,C] supports
            per-label perturbations. Every call recomputes attention and applies
            the same masked softmax. Fully masked labels have zero residual,
            including zero bias. No projection, dropout or CNN runs here.
            """
            if features.ndim != 5 or tuple(features.shape[2:]) != (14, 14, 512):
                raise ValueError("local_features must have shape [B,T,14,14,512]")
            if cell_mask.dtype != torch.bool:
                raise ValueError("local cell masks must be boolean")
            shape = features.shape[:-1]
            num_targets = self.label_queries.shape[0]
            if tuple(cell_mask.shape) == tuple(shape):
                label_mask = cell_mask.unsqueeze(-1).expand(*shape, num_targets)
            elif tuple(cell_mask.shape) == (*shape, num_targets):
                label_mask = cell_mask
            else:
                raise ValueError("local cell mask must match [B,T,14,14] or [B,T,14,14,C]")
            flat = features.reshape(features.shape[0], -1, self.feature_dim)
            mask = label_mask.reshape(features.shape[0], -1, num_targets)
            scores = torch.einsum("bnd,cd->bnc", flat, self.label_queries) / math.sqrt(self.feature_dim)
            # Do the softmax in fp32 even when CNN/readout matmuls use AMP.
            scores = scores.float().masked_fill(~mask, -torch.inf)
            has_cells = mask.any(dim=1)
            scores = torch.where(has_cells[:, None], scores, torch.zeros_like(scores))
            alpha = scores.softmax(dim=1).masked_fill(~mask, 0.0)
            cell_logits = torch.einsum("bnd,cd->bnc", flat, self.head_weights).float()
            bias = self.head_bias.float().unsqueeze(0) * has_cells
            contributions = alpha * cell_logits
            local_logits = contributions.sum(dim=1) + bias
            gain = self.gate if self.gate is not None else local_logits.new_ones(())
            gated_logits = gain * local_logits
            if not diagnostics:
                return gated_logits
            return {
                "local_logits": local_logits,
                "gated_local_logits": gated_logits,
                "gate": gain,
                "alpha": alpha.reshape(*shape, num_targets),
                "signed_contributions": (gain * contributions).reshape(*shape, num_targets),
                "local_bias": bias,
                "gated_local_bias": gain * bias,
                "cell_mask": cell_mask,
                "local_features": features,
            }


    class KneeMILModel(nn.Module):
        def __init__(
            self,
            config: dict[str, Any],
            *,
            initialize_backbone: bool = True,
        ) -> None:
            super().__init__()
            model_config = config["model"]
            local_mode = model_config.get("local_head")
            if local_mode not in (None, "none", "l16"):
                raise ValueError("model.local_head must be absent, 'none', or 'l16'")
            self.local_head_mode = local_mode
            feature_dim = int(model_config.get("feature_dim", 256))
            self.feature_dim = feature_dim
            self.encoder_chunk_size = max(
                1, int(model_config.get("encoder_chunk_size", 16))
            )
            self.gradient_checkpointing = bool(
                model_config.get("gradient_checkpointing", False)
            )
            backbone = str(model_config.get("backbone", "tiny"))
            if local_mode == "l16" and (
                backbone.split(".", maxsplit=1)[0] != "timm:convnext_tiny"
                or int(config["data"]["image_size"]) != 224
                or feature_dim != 512
                or model_config.get("global_pool", "avg") != "avg"
            ):
                raise ValueError("L16 requires ConvNeXt-Tiny, image_size=224, feature_dim=512, global_pool='avg'")
            if backbone == "tiny":
                self.encoder = TinyEncoder(feature_dim)
            elif backbone.startswith("timm:"):
                self.encoder = TimmEncoder(
                    backbone.split(":", maxsplit=1)[1],
                    feature_dim,
                    pretrained=bool(model_config.get("pretrained", False)),
                    checkpoint=model_config.get("encoder_checkpoint"),
                    image_size=int(config["data"]["image_size"]),
                    global_pool=str(model_config.get("global_pool", "avg")),
                    input_normalization=str(
                        model_config.get("input_normalization", "none")
                    ),
                    finetune_mode=str(model_config.get("finetune_mode", "full")),
                    unfreeze_last_blocks=int(
                        model_config.get("unfreeze_last_blocks", 0)
                    ),
                    initialize_backbone=initialize_backbone,
                    drop_path_rate=(
                        float(model_config["drop_path_rate"])
                        if model_config.get("drop_path_rate") is not None else None
                    ),
                )
            elif backbone == "radimagenet:resnet50":
                self.encoder = RadImageNetResNet50Encoder(
                    feature_dim,
                    checkpoint=model_config.get("encoder_checkpoint"),
                    finetune_mode=str(model_config.get("finetune_mode", "head_only")),
                    initialize_backbone=initialize_backbone,
                )
            else:
                raise ValueError(
                    "model.backbone must be 'tiny', start with 'timm:', or equal "
                    "'radimagenet:resnet50'"
                )

            num_slots = len(config["data"]["slots"])
            num_targets = len(TARGET_COLUMNS)
            self.slot_embedding = nn.Embedding(num_slots, feature_dim)
            self.normalization = nn.LayerNorm(feature_dim)
            self.dropout = nn.Dropout(float(model_config.get("dropout", 0.2)))
            self.label_queries = nn.Parameter(torch.empty(num_targets, feature_dim))
            self.head_weights = nn.Parameter(torch.empty(num_targets, feature_dim))
            self.head_bias = nn.Parameter(torch.zeros(num_targets))
            nn.init.normal_(self.label_queries, std=0.02)
            nn.init.normal_(self.head_weights, std=0.02)

            # Keep every global parameter and the subsequent CPU RNG identical
            # for paired baseline/L16 construction from the same seed.
            self.local_head = None
            if local_mode == "l16":
                with torch.random.fork_rng(devices=[]):
                    self.local_head = LocalSpatialHead(
                        num_slots, num_targets,
                        float(model_config.get("dropout", 0.2)),
                        float(model_config.get("local_gate_init", 0.0)),
                        residual_mode=str(model_config.get("local_residual_mode", "gated")),
                        precision=str(model_config.get("local_precision", "inherit")),
                    )
            self.token_metadata = None
            if model_config.get('token_metadata', False):
                from .canonical_inputs import FEATURE_NAMES
                with torch.random.fork_rng(devices=[]):
                    self.token_metadata = nn.Linear(len(FEATURE_NAMES), feature_dim, bias=False)
                    nn.init.zeros_(self.token_metadata.weight)

        def _encode_valid_tokens(self, images: torch.Tensor, *, return_local: bool = False):
            chunks = []
            local_chunks = []
            encode = self.encoder.forward_with_local if return_local else self.encoder
            for start in range(0, len(images), self.encoder_chunk_size):
                chunk = images[start : start + self.encoder_chunk_size]
                if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
                    from torch.utils.checkpoint import checkpoint

                    encoded = checkpoint(encode, chunk, use_reentrant=False)
                else:
                    encoded = encode(chunk)
                if return_local:
                    encoded, local_map = encoded
                    local_chunks.append(local_map)
                chunks.append(encoded)
            global_features = torch.cat(chunks, dim=0)
            return (global_features, torch.cat(local_chunks, dim=0)) if return_local else global_features

        def forward(
            self,
            tokens: torch.Tensor,
            token_mask: torch.Tensor,
            slot_ids: torch.Tensor,
            token_features: torch.Tensor | None = None,
        ) -> torch.Tensor:
            return self._forward(tokens, token_mask, slot_ids, token_features, diagnostics=False)

        def forward_diagnostics(
            self, tokens: torch.Tensor, token_mask: torch.Tensor, slot_ids: torch.Tensor,
            token_features: torch.Tensor | None = None,
        ) -> dict[str, torch.Tensor]:
            """Return raw probabilities of attention and an exact linear decomposition.

            ``local_features`` are post-projection/position/LN/dropout features,
            not raw CNN maps. Replay them with ``forward_local_from_features``.
            Inference/export callers should use eval() and inference_mode().
            """
            return self._forward(tokens, token_mask, slot_ids, token_features, diagnostics=True)

        def forward_local_from_features(
            self,
            local_features: torch.Tensor,
            cell_mask: torch.Tensor,
            global_logits: torch.Tensor,
            *,
            diagnostics: bool = False,
        ):
            """Re-evaluate local masks while retaining the exact supplied global logits.

            Use exported ``local_features`` and a subset of exported ``cell_mask``;
            label-specific [B,T,14,14,12] masks are supported. Attention is
            renormalized over remaining cells. The global branch never runs.
            """
            if self.local_head is None:
                raise ValueError("Local feature replay requires model.local_head='l16'")
            result = self.local_head.readout(local_features, cell_mask, diagnostics=diagnostics)
            if not diagnostics:
                return global_logits + result
            result["global_logits"] = global_logits
            result["logits"] = global_logits + result["gated_local_logits"]
            return result

        def _forward(self, tokens, token_mask, slot_ids, token_features=None, *, diagnostics):
            batch, token_count, channels, height, width = tokens.shape
            if self.local_head is not None and (height, width) != (224, 224):
                raise ValueError("L16 currently requires 224x224 input; position interpolation is not enabled")
            flat_tokens = tokens.reshape(batch * token_count, channels, height, width)
            flat_mask = token_mask.reshape(batch * token_count)
            encoded_flat = flat_tokens.new_zeros((batch * token_count, self.feature_dim))
            local_flat = None
            if self.local_head is not None:
                local_flat = flat_tokens.new_zeros((batch * token_count, 384, 14, 14))
            if flat_mask.any():
                # Under CUDA autocast the encoder returns fp16 while the zero buffer is
                # fp32; index_put requires matching dtypes.
                features = self._encode_valid_tokens(
                    flat_tokens[flat_mask], return_local=self.local_head is not None
                )
                if self.local_head is not None:
                    features, maps = features
                    local_flat[flat_mask] = maps.to(local_flat.dtype)
                encoded_flat[flat_mask] = features.to(encoded_flat.dtype)
            encoded = encoded_flat.reshape(batch, token_count, self.feature_dim)
            encoded = encoded + self.slot_embedding(slot_ids)
            if self.token_metadata is not None:
                if token_features is None or token_features.shape != (*token_mask.shape, self.token_metadata.in_features):
                    raise ValueError('Configured token metadata must be provided for every token')
                if not torch.isfinite(token_features).all():
                    raise ValueError('Nonfinite token metadata')
                encoded = encoded + self.token_metadata(token_features.to(encoded.dtype))
            elif token_features is not None:
                raise ValueError('Unexpected token metadata for a baseline model')
            encoded = self.normalization(encoded)
            encoded = self.dropout(encoded)
            scores = torch.einsum("btd,cd->btc", encoded, self.label_queries)
            scores = scores / math.sqrt(encoded.shape[-1])
            scores = scores.masked_fill(~token_mask.unsqueeze(-1), -1e4)
            attention = scores.softmax(dim=1)
            pooled = torch.einsum("btc,btd->bcd", attention, encoded)
            global_logits = (pooled * self.head_weights.unsqueeze(0)).sum(dim=-1) + self.head_bias
            if self.local_head is None:
                if diagnostics:
                    return {"logits": global_logits, "global_logits": global_logits,
                            "global_alpha": attention, "token_mask": token_mask, "slot_ids": slot_ids}
                return global_logits
            local_features = self.local_head.prepare_features(
                local_flat.reshape(batch, token_count, 384, 14, 14), slot_ids
            )
            cell_mask = token_mask[:, :, None, None].expand(batch, token_count, 14, 14)
            result = self.forward_local_from_features(
                local_features, cell_mask, global_logits, diagnostics=diagnostics
            )
            if diagnostics:
                result.update(global_alpha=attention, token_mask=token_mask, slot_ids=slot_ids)
            return result


else:

    class KneeMILModel:  # type: ignore[no-redef]
        def __init__(
            self,
            config: dict[str, Any],
            *,
            initialize_backbone: bool = True,
        ) -> None:
            raise RuntimeError("KneeMILModel requires PyTorch; install the 'train' extra")


def parameter_summary(model: Any) -> dict[str, int]:
    """Return total/trainable parameter counts for the model and image backbone."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    encoder = getattr(model, "encoder", None)
    backbone = getattr(encoder, "backbone", encoder)
    if backbone is None:
        backbone_total = 0
        backbone_trainable = 0
    else:
        backbone_total = sum(parameter.numel() for parameter in backbone.parameters())
        backbone_trainable = sum(
            parameter.numel()
            for parameter in backbone.parameters()
            if parameter.requires_grad
        )
    return {
        "total": int(total),
        "trainable": int(trainable),
        "backbone_total": int(backbone_total),
        "backbone_trainable": int(backbone_trainable),
    }


def create_model(
    config: dict[str, Any],
    *,
    initialize_backbone: bool = True,
) -> KneeMILModel:
    return KneeMILModel(config, initialize_backbone=initialize_backbone)
