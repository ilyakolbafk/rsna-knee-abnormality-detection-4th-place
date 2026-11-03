"""Core v2 arm A network: the exact AUX60 CoAtNet-rmlp-2-384 H1Aux with GPU augmentation in front of the
untouched v8.3 feature path. Same parameters, same state_dict keys, same eval forward as models.H1Aux.
Written by Claude 2026-09-18.
"""
from __future__ import annotations

import torch

from aug336 import augment336
from models import H1Aux


class H1AuxAug(H1Aux):
    def features_aug(self, x, gain, params):
        # Augment on the 336 px source grid, then run the old path verbatim (336->384 bilinear, gain, clamp,
        # (x-0.5)/0.5, per-block checkpointing). augment336 is a no-op on an all-identity batch.
        return self.features(augment336(x, params), gain)

    def forward(self, x, mask, meta, gain=None, params=None):
        if not self.training or params is None:
            return super().forward(x, mask, meta, gain)
        if not mask.any(1).all():
            raise ValueError("Empty study")
        b, k = mask.shape
        idx = mask.flatten().nonzero().flatten()
        selected = x.flatten(0, 1).index_select(0, idx)
        if gain is None:
            gain = x.new_ones(b)
        gains = gain[:, None].expand(b, k).reshape(-1).index_select(0, idx)
        token_params = params.flatten(0, 1).index_select(0, idx)
        values = []
        for xx, gg, pp in zip(selected.split(self.chunk), gains.split(self.chunk), token_params.split(self.chunk)):
            values.append(self.features_aug(xx, gg, pp))
        encoded = torch.cat(values).float()
        padded = encoded.new_zeros(b * k, encoded.shape[-1]).index_copy(0, idx, encoded).view(b, k, -1)
        with torch.autocast(x.device.type, enabled=False):
            hidden = self.norm(padded + self.meta_projection(meta.float()))
            alpha = self.att(hidden).masked_fill(~mask[:, :, None], -torch.inf).softmax(1)
            pooled = torch.einsum("bkn,bkf->bnf", alpha, hidden)
            main = (pooled * self.clsW).sum(-1) + self.clsb
            alpha_aux = self.aux_att(hidden).masked_fill(~mask[:, :, None], -torch.inf).softmax(1)
            pooled_aux = torch.einsum("bkn,bkf->bnf", alpha_aux, hidden)
            return torch.cat([main, (pooled_aux * self.auxW).sum(-1) + self.auxb], dim=1)


def make(assets, chunk: int = 96):
    """Same construction as models.make_model('AUX60', assets, chunk=96)."""
    return H1AuxAug(assets / "coatnet_imagenet.safetensors", chunk=chunk, checkpointing=True)
