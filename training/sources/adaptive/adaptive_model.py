"""Same H1AuxAug recipe with memory-transparent outer feature checkpoints."""
from contextlib import nullcontext
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import h1_base as h
from core_v2_a_model import H1AuxAug, make as ordinary_make


def widen_metadata(model):
    old = model.meta_projection
    if old.in_features != 6 or old.bias is not None:
        raise ValueError("Expected unchanged ordinary metadata6 projection")
    with torch.random.fork_rng(devices=[]):
        new = nn.Linear(8, old.out_features, bias=False, device=old.weight.device, dtype=old.weight.dtype)
        nn.init.zeros_(new.weight)
    model.meta_projection = new
    return model


class MemoryH1AuxAug(H1AuxAug):
    def features_aug(self, x, gain, params):
        original = super().features_aug
        if self.training and torch.is_grad_enabled() and self.checkpointing:
            # Preserve the existing 96-token chunk and augmentation call. Inner block
            # checkpoints remain active. Recomputation must not update BN a second time.
            return checkpoint(original, x, gain, params, use_reentrant=False,
                              preserve_rng_state=True,
                              context_fn=lambda: (nullcontext(), h.preserve_bn(self.backbone)))
        return original(x, gain, params)


def make(assets, chunk=96):
    return widen_metadata(MemoryH1AuxAug(assets / 'coatnet_imagenet.safetensors',
                                       chunk=chunk, checkpointing=True))
