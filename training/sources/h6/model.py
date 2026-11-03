"""H5/H6 spatial architecture with explicit, independently trainable VM domain heads."""
import copy
import torch
from torch import nn
from public_math import Net


class Readout(nn.Module):
    spatial_pool = Net.spatial_pool
    head = Net.head
    _decompose_syn = Net._decompose_syn

    def __init__(self, public):
        super().__init__()
        self.spatial = True
        self.extent = False
        self.has_state = False
        self.syn_decomp = True
        self._syn_spec = None
        self._ext_area = None
        for name in ['proj', 'slot', 'view', 'geom', 'drop', 'win_proj', 'out_norm',
                     'sp_key', 'sp_pos', 'sp_head_norm']:
            setattr(self, name, getattr(public, name))
        for name in ['q_win', 'cls_w', 'bias', 'syn_beta', 'sp_q']:
            setattr(self, name, getattr(public, name))

    def forward(self, fmap, meta, mask):
        return self.head(self.spatial_pool(fmap), meta, mask)


class Model(nn.Module):
    def __init__(self, checkpoint_path, arm, chunk=32, *, public=None):
        super().__init__()
        assert arm in ['core6', 'vm']
        self.arm, self.chunk = arm, int(chunk)
        if public is None:
            public = Net('convnext_small.fb_in22k_ft_in1k_384', None,
                         hidden=384, drop=.1, crop=352, spatial=True,
                         syn_decomp=True, fm_aux=11)
            if checkpoint_path is None:raise ValueError('IMAGENET_PRETRAIN_REQUIRED')
            # The retained file is the original Facebook-format checkpoint.
            # Public Net's generic strict=False reader does not remap it.
            # Use timm's official tensor-preserving key adapter, then require
            # EVERY feature weight and the moved pooled LayerNorm to load.
            from timm.models.convnext import checkpoint_filter_fn
            state=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
            mapped=checkpoint_filter_fn(state,public.enc)
            norm={k[len('head.norm.'):]:v for k,v in mapped.items() if k.startswith('head.norm.')}
            if public.sp_head_norm is None or set(norm)!={'weight','bias'}:raise ValueError('PRETRAIN_POOLED_NORM_REQUIRED')
            public.sp_head_norm.load_state_dict(norm,strict=True)
            feature={k:v for k,v in mapped.items() if not k.startswith('head.norm.') and k not in ['head.fc.weight','head.fc.bias']}
            public.enc.load_state_dict(feature,strict=True)
            self.pretrain_coverage={'feature_tensors':len(feature),'pooled_norm_tensors':len(norm),'ignored_classification_head_only':True}
            public.enc.set_grad_checkpointing(True)
        assert public.spatial and public.syn_decomp and not public.has_state and not public.extent
        self.encoder = public.enc
        self.main = Readout(public)
        # Copy the whole readout, including pooled LN, exactly once. Never mutate
        # enc.head.norm a second time. No shared disease-head parameter storage.
        self.oai_vm = copy.deepcopy(self.main) if arm == 'vm' else None
        self.register_buffer('mean', public.mean.detach().clone())
        self.register_buffer('std', public.std.detach().clone())
        for name in ['aux_ln_w', 'aux_ln_b', 'aux_w', 'aux_b']:
            setattr(self, name, getattr(public, name))
        self.has_bn = bool(public.has_bn)
        assert not self.has_bn, 'THIS_RECIPE_IS_CONVNEXT_ONLY'

    def encode(self, images):
        images = images.float().div(255)
        return self.encoder(((images-self.mean)/self.std).contiguous(memory_format=torch.channels_last))

    def forward(self, images, meta, mask, *, head='main'):
        assert head in ['main', 'oai_vm']
        readout = self.main if head == 'main' else self.oai_vm
        assert readout is not None
        assert mask.ndim == 2 and bool(mask.any(1).all()), 'EMPTY_STUDY'
        selected = images[mask]
        real_count = len(selected)
        if real_count % 32 and (not self.training or not self.has_bn):
            selected = torch.cat([selected, selected.new_zeros((32-real_count % 32,) + tuple(selected.shape[1:]))])
        tokens = []
        for start in range(0, len(selected), self.chunk):
            tokens.append(readout.spatial_pool(self.encode(selected[start:start+self.chunk])))
        return readout.head(torch.cat(tokens)[:real_count], meta, mask)

    def gold(self, images, meta, mask):
        # Gold is a separate diagnostic, always the RSNA output head.
        return self.forward(images, meta, mask, head='main')

    def fm_dense(self, images):
        fmap = self.encode(images)
        with torch.autocast('cuda', enabled=False):
            f = fmap.permute(0, 2, 3, 1).float()
            f = torch.nn.functional.layer_norm(f, (f.shape[-1],), self.aux_ln_w, self.aux_ln_b, 1e-6)
            return (f @ self.aux_w.t() + self.aux_b).permute(0, 3, 1, 2)

    def parameter_groups(self):
        return [{'params': list(self.encoder.parameters()), 'lr': 1e-4},
                {'params': [p for n, p in self.named_parameters() if not n.startswith('encoder.')], 'lr': 1e-3}]

    @torch.no_grad()
    def initialize_priors(self,main_base,oai_base=None):
        self.main.bias.copy_(torch.logit(torch.tensor(main_base,device=self.main.bias.device).clamp(.001,.999)))
        if self.oai_vm is not None:
            assert oai_base is not None
            self.oai_vm.bias.copy_(torch.logit(torch.tensor(oai_base,device=self.oai_vm.bias.device).clamp(.001,.999)))


def clinical_loss(model, batch):
    """One pooled batch; domain losses each keep denominator TOTAL batch×12."""
    from public_math import lr_loss
    size = len(batch['source_id'])
    result = None
    for head, weights, ratios in [('main', 'main_weight', 'main_lr'),
                                  ('oai_vm', 'oai_vm_weight', 'oai_vm_lr')]:
        selected = batch[weights].gt(0).any(1)
        if not bool(selected.any()):
            continue
        assert head == 'main' or model.arm == 'vm'
        if model.arm == 'vm':
            expected_source = 0 if head == 'main' else 1
            assert bool((batch['source_id'][selected] == expected_source).all()), 'DOMAIN_HEAD_ROUTING'
        logits = model(batch['x'][selected], batch['meta'][selected], batch['mask'][selected], head=head)
        value = lr_loss(logits.float(), batch[ratios][selected], batch[weights][selected]) * selected.sum()/size
        result = value if result is None else result + value
    # A no-label draw still has FM supervision; do not manufacture a disease loss.
    return result
