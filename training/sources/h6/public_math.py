"""Exact extracted public v9.8 primitives; CLI is deliberately absent. ConvNeXt only."""
import math
import torch
from torch import nn
from torch.nn import functional as F
LABELS=['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
C=12
OFL_BACKBONE="ofl_l"
FM_BIAS0=-4.0

class Net(nn.Module):
    def __init__(self, backbone, init, hidden=384, drop=0.1, freeze_blocks=16, crop=256, state_spec=None, in_chans=3, spatial=False,
                 extent=False, syn_decomp=False, extent_classes=None, fm_aux=0):
        super().__init__()
        import timm
        self.backbone = backbone
        self.crop = int(crop)
        self.in_chans = int(in_chans)
        self.spatial = bool(spatial)
        self.extent = bool(extent)
        self._ext_area = None
        self.syn_decomp = bool(syn_decomp)      # v9.2 (10-10, mechanisms_r4d_20261010.md 2.4): z_syn = z_spec + beta * sg(z_eff)
        self._syn_spec = None
        if self.syn_decomp:                     # created only when asked: state_dict keys unchanged otherwise
            self.syn_beta = nn.Parameter(torch.zeros(()))   # init 0 -> the baseline network at step 0
        if self.spatial and backbone == OFL_BACKBONE:
            raise SystemExit("--spatial-attn needs a timm backbone (the OFL encoder returns pooled features)")
        if backbone == OFL_BACKBONE:
            assert self.in_chans == 3, "the OFL patch embedding is a public 3-channel weight"
            self.enc = OFLEncoder(init, freeze=freeze_blocks, img_size=crop)
        else:
            self.enc = self._timm_encoder(backbone, init, self.in_chans)
        self.has_bn = any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in self.enc.modules())
        self._build_head(hidden, drop)
        self.state_spec = state_spec
        self.has_state = state_spec is not None
        if self.has_state:
            self._build_state_head(hidden, state_spec)
        if self.spatial:                        # created last: every baseline parameter keeps the baseline's initial values
            d = self.enc.num_features
            self.sp_key = nn.Linear(d, 128)     # keys from per-position normalised features
            self.sp_pos = nn.Linear(16, 128, bias=False)
            nn.init.zeros_(self.sp_pos.weight)
            self.sp_q = nn.Parameter(torch.zeros(C, 128))   # zero query = uniform weights = the baseline average
        if self.extent:                         # OA defect extent (after every other parameter: same init as the baseline)
            self.ext_u = nn.Parameter(torch.zeros(C, hidden))
            self.ext_b = nn.Parameter(torch.full((C,), -4.0))   # sigmoid(-4) = 0.018: area starts ~4 mm^2, not ~3,000 (audit 3)
            self.ext_gamma = nn.Parameter(torch.zeros(C))    # zero = the baseline logit at init
            self.ext_view = nn.Parameter(torch.zeros(C, 6))  # per (class, view) weight inside a series; 0 = uniform (audit 4)
            em = torch.zeros(C)
            # v9.4: --extent-classes picks the OA classes whose logit receives the area term (None = the three = v9.3); the area
            # itself is still computed and supervised for all three
            for l in (("Medial OA", "Lateral OA", "PF OA") if extent_classes is None else extent_classes):
                em[LABELS.index(l)] = 1.0
            self.register_buffer("ext_mask", em)
        self.fm_aux = int(fm_aux)
        if self.fm_aux:                         # v9.8 --fm-aux: the auxiliary head, created LAST (every v9.7 parameter keeps its initial
            # value and its position in parameters() / state_dict()) from a private generator (the global stream is not consumed);
            # per-position LayerNorm(d) -> linear d -> A on the encoder's map; never part of a competition output, stripped from the
            # fold model by deploy_state_dict
            d = self.enc.num_features
            g_aux = torch.Generator().manual_seed(9800)
            self.aux_ln_w = nn.Parameter(torch.ones(d))
            self.aux_ln_b = nn.Parameter(torch.zeros(d))
            self.aux_w = nn.Parameter(torch.randn(self.fm_aux, d, generator=g_aux) * .02)
            self.aux_b = nn.Parameter(torch.full((self.fm_aux,), FM_BIAS0))

    def aux_dense(self, fmap):
        """v9.8: encoder map [M, d, G, G] (the live map, NOT detached: the auxiliary gradient reaches every encoder parameter) -> dense
        logits [M, A, G, G] float32: LayerNorm over d at every position, then the linear map d -> A at every position (no pooling first).
        Run in float32 with autocast off."""
        assert fmap.ndim == 4, "v9.8 --fm-aux needs the encoder's spatial map [M, d, G, G]"
        with torch.autocast("cuda", enabled=False):
            f = fmap.permute(0, 2, 3, 1).float()
            f = F.layer_norm(f, (f.shape[-1],), self.aux_ln_w, self.aux_ln_b, 1e-6)
            z = torch.matmul(f, self.aux_w.t()) + self.aux_b
        return z.permute(0, 3, 1, 2)

    def _build_state_head(self, hidden, spec):
        """Report-state heads: for class c a K_c-way read-out of the class token (which report state the images show),
        OA-sign / acuity binaries on the parent class token, and the frozen prior vector per state (mean report prior of the state's category over the pool, not fitted on gold).
        The class logit gains  gate_c * (logit(sum_s p(s|x) prior_c(s)) - logit(base_c)): the 'align to the official
        label' mapping applied to predicted states; gate starts at 0 so the direct path is untouched at init."""
        cats = spec["categories"]
        self.state_k = [len(cats[l]) for l in LABELS]
        kmax = max(self.state_k)
        self.state_w = nn.Parameter(torch.randn(C, kmax, hidden) * .02)
        self.state_b = nn.Parameter(torch.zeros(C, kmax))
        prior = torch.full((C, kmax), float("nan"))
        valid = torch.zeros((C, kmax), dtype=torch.bool)
        for i, l in enumerate(LABELS):
            for j, pv in enumerate(spec["prior"][l]):
                prior[i, j] = float(pv); valid[i, j] = True
        self.register_buffer("state_prior", torch.nan_to_num(prior, nan=0.5).clamp(1e-3, 1 - 1e-3))
        self.register_buffer("state_valid", valid)
        self.register_buffer("state_base_logit", torch.logit(torch.tensor([float(spec["base"][l]) for l in LABELS]).clamp(1e-3, 1 - 1e-3)))
        self.state_gate = nn.Parameter(torch.zeros(C))
        self.bin_names = [(b, l) for b, labs in spec["binaries"].items() for l in labs]
        self.bin_parent = [LABELS.index(l) for _, l in self.bin_names]
        self.bin_w = nn.Parameter(torch.randn(len(self.bin_names), hidden) * .02)
        self.bin_b = nn.Parameter(torch.zeros(len(self.bin_names)))

    def state_logits(self, study):
        """study: [B, C, hidden] class tokens -> state logits [B, C, Kmax] (invalid slots -inf), binary logits [B, nbin]."""
        sl = torch.einsum("bch,ckh->bck", study, self.state_w).float() + self.state_b
        sl = sl.masked_fill(~self.state_valid[None], float("-inf"))
        bl = torch.einsum("bjh,jh->bj", study[:, self.bin_parent], self.bin_w).float() + self.bin_b
        return sl, bl

    def state_evidence(self, sl):
        """logit of the official probability implied by the predicted state distribution, relative to the class base rate."""
        p = sl.softmax(-1)                                          # -inf slots -> 0
        e = (p * self.state_prior[None]).sum(-1).clamp(1e-3, 1 - 1e-3)
        return torch.logit(e) - self.state_base_logit[None]

    def deploy_state_dict(self):
        """Everything except the frozen public OFL weights (identical to state_dict() for other backbones).  v9.8: and except the
        auxiliary head aux_* (--fm-aux), so a fold model has v9.7's key set (none of its keys starts with aux_)."""
        frozen = {"enc." + k for k in getattr(self.enc, "frozen_keys", set())}
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k not in frozen and not k.startswith("aux_")}

    def load_deploy_state_dict(self, sd):
        missing, unexpected = self.load_state_dict(sd, strict=False)
        frozen = {"enc." + k for k in getattr(self.enc, "frozen_keys", set())}
        if unexpected or set(missing) - frozen:
            raise ValueError("deploy state does not match: missing %s unexpected %s" % (sorted(set(missing) - frozen)[:5], unexpected[:5]))

    def _timm_encoder(self, backbone, init, in_chans=3):
        import timm
        # init == "timm": public ImageNet weights from the local hub cache (HF_HOME); a path: a state dict file
        is_vit = backbone.startswith("vit_")
        kw = dict(pretrained=(init == "timm"), num_classes=0, in_chans=in_chans,
                  drop_path_rate=0.1 if ("convnext" in backbone or is_vit) else 0.0)
        if is_vit:
            kw["dynamic_img_size"] = True       # keeps the pretrained grid (DINOv2 518 / DINOv3 256); position embeddings or RoPE
        else:                                   # are interpolated to the crop at forward; pooling = the model's own (token / avg)
            kw["global_pool"] = "avg"
        if backbone.startswith(("coatnet", "maxvit", "maxxvit")):
            kw["img_size"] = self.crop          # relative-position tables are built for this input size (CoAtNet / MaxViT)
        if self.spatial:                        # keep the spatial grid: CNN -> normed map [N, d, h, w], ViT -> all tokens [N, T, d]
            kw["global_pool"] = ""
        enc = timm.create_model(backbone, **kw)
        if init and init != "timm":
            if str(init).endswith(".safetensors"):
                from safetensors.torch import load as load_bytes
                with open(str(init), "rb") as fh:
                    sd = load_bytes(fh.read())
            else:
                sd = torch.load(init, map_location="cpu")
                sd = sd.get("state_dict", sd.get("model", sd)) if isinstance(sd, dict) else sd
            if in_chans != 3:                       # inflate the pretrained RGB stem: mean over RGB, replicated K times, scaled 3/K
                for k, w in list(sd.items()):
                    if w.ndim == 4 and w.shape[1] == 3 and k in enc.state_dict() and enc.state_dict()[k].shape[1] == in_chans:
                        sd[k] = w.mean(1, keepdim=True).repeat(1, in_chans, 1, 1) * (3.0 / in_chans)
                        print("inflated %s %s -> %s" % (k, tuple(w.shape), tuple(sd[k].shape)), flush=True)
            missing, unexpected = enc.load_state_dict(sd, strict=False)
            print("init %s: missing %d unexpected %d" % (init, len(missing), len(unexpected)), flush=True)
        self.sp_head_norm = None
        if self.spatial and isinstance(getattr(getattr(enc, "head", None), "norm", None), nn.Module) \
                and not isinstance(enc.head.norm, nn.Identity):
            # CNN heads pool THEN normalise: keep that order per class (pool the raw map, then this norm on each class's
            # pooled vector), so a zero spatial query reproduces the baseline feature LN(mean_p x_p) exactly
            self.sp_head_norm = enc.head.norm
            enc.head.norm = nn.Identity()
        return enc

    def _build_head(self, hidden, drop):
        d = self.enc.num_features
        self.proj = nn.Sequential(nn.Linear(d, hidden), nn.LayerNorm(hidden))
        self.slot = nn.Embedding(10, hidden)
        self.view = nn.Embedding(6, hidden)
        self.geom = nn.Linear(4, hidden, bias=False)
        for m in (self.slot, self.view):
            nn.init.zeros_(m.weight)
        nn.init.zeros_(self.geom.weight)
        self.drop = nn.Dropout(drop)
        self.win_proj = nn.Linear(hidden, hidden)
        self.q_win = nn.Parameter(torch.randn(C, hidden) * .02)
        self.out_norm = nn.LayerNorm(hidden)
        self.cls_w = nn.Parameter(torch.randn(C, hidden) * .02)
        self.bias = nn.Parameter(torch.zeros(C))
        if self.in_chans == 3:
            self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
        else:                                       # grey slices: one ImageNet-mean/std per channel
            self.register_buffer("mean", torch.full((1, self.in_chans, 1, 1), .449))
            self.register_buffer("std", torch.full((1, self.in_chans, 1, 1), .226))

    def spatial_pool(self, x):
        """Encoder output with its spatial grid ([N, d, h, w] CNN map or [N, T, d] ViT tokens) -> per-class window features
        [N, C, d]: class c weighs the window's positions with its own query (sp_q[c]; zero at init = the plain average)
        over keys tanh(W f(p) + W_pos pe(p)), so a class can read its own structure inside a window shared with others."""
        if x.ndim == 4:
            h, w = x.shape[-2:]
            f = x.flatten(2).transpose(1, 2)                                      # [N, h*w, d]
        else:
            f = x[:, getattr(self.enc, "num_prefix_tokens", 0):]
            h = w = int(round(f.shape[1] ** 0.5))
            assert h * w == f.shape[1], "non-square ViT token grid"
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=f.device), torch.linspace(-1, 1, w, device=f.device), indexing="ij")
        fr = torch.tensor([1., 2., 4., 8.], device=f.device) * (math.pi / 2)
        yy, xx = yy.reshape(-1, 1) * fr, xx.reshape(-1, 1) * fr
        pe = torch.cat([yy.sin(), yy.cos(), xx.sin(), xx.cos()], 1)              # [h*w, 16] fixed 2-D position features
        kin = F.layer_norm(f.float(), (f.shape[-1],)).to(f.dtype)                # keys: per-position normalised features
        k = torch.tanh(self.sp_key(kin) + self.sp_pos(pe.to(f.dtype))[None])    # [N, P, 128]
        att = torch.einsum("npk,ck->npc", k, self.sp_q.to(k.dtype)).float().softmax(1).to(f.dtype)
        g = torch.einsum("npc,npd->ncd", att, f)                                 # [N, C, d]
        if self.sp_head_norm is not None:                                        # CNN: the head norm after pooling, per class
            n_ = self.sp_head_norm
            g = F.layer_norm(g.float(), (g.shape[-1],), n_.weight, n_.bias, n_.eps).to(f.dtype)
        return g

    def extent_area(self, e, meta, mask):
        """e [B, W, C] window defect logits -> A [B, C] (mm^2, an area proxy): inside each series the mean sigmoid(e) of
        each view's windows, combined over the views present by a learned per-(class, view) softmax (uniform at init: a
        compartment sits inside only some of the locals), times the series' slab thickness (slices x spacing) = mm of
        slab with defect evidence; per plane the max over series; then sagittal x coronal (tibiofemoral OA) or axial x
        sagittal (PF OA); other classes 0."""
        V = 6
        sid = meta[..., 6].long()
        # one slot per series, never merged: 16 covers every training study (at most 14 series); a deployed study with more
        # series gets more slots (lossless -- an empty slot has mean 0 and the per-plane max runs over values >= 0)
        S_MAX = max(16, int(sid.max()) + 1)
        vid = meta[..., 1].long().clamp(0, V - 1)
        q = torch.sigmoid(e.float()) * mask[..., None]
        ohv = F.one_hot(sid * V + vid, S_MAX * V).float() * mask[..., None]     # [B, W, S*V]
        cntv = ohv.sum(1)                                                          # [B, S*V]
        mean_v = (torch.einsum("bwk,bwc->bkc", ohv, q) / cntv.clamp_min(1)[..., None]).view(len(e), S_MAX, V, -1)
        present = (cntv > 0).view(len(e), S_MAX, V, 1)
        wv = self.ext_view.float().t()[None, None].masked_fill(~present, -1e4).softmax(2) * present   # [B, S, V, C]
        mean = (wv * mean_v).sum(2)                                                # [B, S, C]; 0 for a series without windows
        oh = F.one_hot(sid, S_MAX).float() * mask[..., None]                     # [B, W, S]
        cnt = oh.sum(1)                                                            # [B, S]
        span = torch.expm1(meta[..., 3].float()).clamp_min(0) * meta[..., 8].float()   # slab thickness (mm) of each window's series
        span_s = torch.einsum("bws,bw->bs", oh, span) / cnt.clamp_min(1)
        plane_s = (torch.einsum("bws,bw->bs", oh, meta[..., 7].float()) / cnt.clamp_min(1)).round()
        E = mean * span_s[..., None]                                               # [B, S, C] mm of slab with defect evidence
        per_plane = []
        for pl in (0, 1, 2):
            ok = ((plane_s == pl) & (cnt > 0))[..., None]
            per_plane.append(E.masked_fill(~ok, 0.0).amax(1))                     # [B, C]; 0 when the plane is absent
        sag, cor, ax = per_plane
        area = torch.zeros_like(sag)
        for l in ("Medial OA", "Lateral OA"):
            c = LABELS.index(l)
            area[:, c] = sag[:, c] * cor[:, c]
        c = LABELS.index("PF OA")
        area[:, c] = ax[:, c] * sag[:, c]
        return area

    def head(self, tok, meta, mask, return_aux=False):
        """Encoder output of the real windows [N, d] (or [N, C, d] with --spatial-attn) -> class logits [B, C] (and the
        state/binary logits when asked)."""
        B, W = mask.shape
        f = self.proj(tok.float())
        m = meta
        if self.spatial:                                                          # class-specific window tokens [B, W, C, H]
            emb = self.slot(m[..., 0].long().clamp(0, 9)) + self.view(m[..., 1].long().clamp(0, 5)) + self.geom(m[..., 2:6])
            t = f.new_zeros((B, W) + tuple(f.shape[1:]))
            t[mask] = f
            t = self.drop(t + emb[:, :, None, :])
            s2 = torch.einsum("bwch,ch->bwc", torch.tanh(self.win_proj(t)), self.q_win).float()
            s2 = s2.masked_fill(~mask[:, :, None], float("-inf"))
            a2 = s2.softmax(1).to(t.dtype)
            study = self.out_norm(torch.einsum("bwc,bwch->bch", a2, t))
            if self.extent:
                e_win = (t.float() * self.ext_u[None, None]).sum(-1) + self.ext_b
        else:
            t = f.new_zeros((B, W, f.shape[-1]))
            t[mask] = f
            t = t + self.slot(m[..., 0].long().clamp(0, 9)) + self.view(m[..., 1].long().clamp(0, 5)) + self.geom(m[..., 2:6])
            t = self.drop(t)
            s2 = torch.einsum("bwh,ch->bwc", torch.tanh(self.win_proj(t)), self.q_win).float()
            s2 = s2.masked_fill(~mask[:, :, None], float("-inf"))
            a2 = s2.softmax(1).to(t.dtype)
            study = self.out_norm(torch.einsum("bwc,bwh->bch", a2, t))
            if self.extent:
                e_win = torch.einsum("bwh,ch->bwc", t.float(), self.ext_u) + self.ext_b
        z = (study * self.cls_w).sum(-1).float() + self.bias
        if self.extent:
            area = self.extent_area(e_win, meta, mask)
            self._ext_area = area
            z = z + self.ext_gamma * self.ext_mask * torch.log1p(area / 100.0)
        if not self.has_state:
            z = self._decompose_syn(z)
            return (z, None, None) if return_aux else z
        sl, bl = self.state_logits(study)
        z = z + self.state_gate[None] * self.state_evidence(sl)
        z = self._decompose_syn(z)
        return (z, sl, bl) if return_aux else z

    def _decompose_syn(self, z):
        """v9.2: the synovitis logit = a synovium-specific part + beta * the effusion logit taken as a fixed covariate
        (detached: the synovitis loss must not reshape the effusion logit).  Off: z unchanged.  The specific part is kept
        for the out-of-fold read-out (oof_synspec) and, later, as the teacher evidence for silent synovitis cells."""
        if not self.syn_decomp:
            return z
        SYN, EFF = LABELS.index("Synovitis"), LABELS.index("Effusion")
        self._syn_spec = z[:, SYN].detach().float()
        ze = z[:, EFF].detach().clone()                            # own storage: no shared version counter with z
        onehot = torch.zeros(z.shape[1], device=z.device, dtype=z.dtype)
        onehot[SYN] = 1
        return z + (self.syn_beta * ze)[:, None] * onehot[None, :]  # out of place: only the synovitis column changes

    def forward(self, img, meta, mask, chunk=0, return_aux=False):
        x = img[mask]                                               # only real windows go through the encoder
        n = len(x)
        if n % 32 and (not self.training or not self.has_bn):     # pad the encoder batch to a multiple of 32: a handful of shapes
            x = torch.cat([x, x.new_zeros((32 - n % 32,) + tuple(x.shape[1:]))])   # instead of one per step (allocator, cuDNN autotune);
        feats = []
        step = chunk or len(x)
        for s in range(0, len(x), step):
            xs = x[s:s + step].float().div_(255.)
            xs = ((xs - self.mean) / self.std).contiguous(memory_format=torch.channels_last)
            e = self.enc(xs)
            feats.append(self.spatial_pool(e) if self.spatial else e)
        return self.head(torch.cat(feats)[:n], meta, mask, return_aux=return_aux)

def lr_loss(z, lr, weight=None):
    """Likelihood-ratio loss on soft targets.  v8: optional per-cell weight [B, C]; the mean stays over cells, so a cell
    with weight 0.5 pushes half as hard (weight None = the v7 loss, bit for bit)."""
    per = -torch.logaddexp(F.logsigmoid(z) + torch.log(lr), F.logsigmoid(-z))
    if weight is None:
        return per.mean(0).mean()
    return (per * weight).mean(0).mean()
