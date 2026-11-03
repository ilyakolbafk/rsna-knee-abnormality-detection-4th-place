"""Compact fine-tuned image model on the per-slice pixel cache.  The user authorised fine-tuning a compact model on 2026-10-05 evening
# v9.8 (10-11 01:00, spec card plans/spec_trainer_v98_fmaux_20261011.md md5 fc451b3e, sections B and E): --fm-aux DIR = the fastMRI+
#   lesion-box auxiliary task on the shared encoder.  Every optimiser step, after the main path's loss.backward() (v9.7's batch, windows,
#   augmentation, random streams, untouched), one auxiliary batch of M = --fm-aux-windows (96) fastMRI 3-slice windows from the fm384_v1
#   store (format: FmStore) -- 4 volumes per step (75% drawn from volumes with boxes, 25% from reviewed volumes without findings), per boxed
#   volume half the windows centred on slices with a box (any class) and half on slices without, per window a view uniform in {0, 2, 3},
#   neighbours at phys step 1 clamped as window_rows -- is cropped to the run's --crop (offset uniform on 0..384-crop, augment()'s
#   distribution, own stream), translated by an integer (dy, dx) uniform on -N..N when --shift-px N > 0 (zero fill, v9.5's convention),
#   flipped left-right with probability --fm-aux-flip and gamma / contrast jittered as augment() does (per window), and goes through the
#   SAME model.enc (train mode, bf16 autocast, the compiled module under --compile): map [M, 768, G, G], G = crop / 32 (asserted equal to the
#   main batch's spatial size / 32 and to the map's grid).  An independent head aux_* (per-position LayerNorm(768) -> linear 768 -> 11,
#   float32) gives dense logits [M, 11, G, G]; presence = per class log-mean-exp over the G^2 positions.  L_aux = BCE(dense, per-cell
#   coverage fraction of the class's boxes after crop / shift / flip; pos_weight 8) + BCE(presence, a box of the class keeping >= 25% of its
#   original area in the window, any of the 3 slices); means over windows x positions x classes / windows x classes.  lambda * L_aux
#   (--fm-aux-weight 0.3) is back-propagated into the same .grad (two backward passes; = the gradient of L_main + lambda * L_aux), then ONE
#   clip over every parameter (fm_clip: v9.7's norm over v9.7's parameters combined with the head's), one AdamW step, one EMA update.  The
#   auxiliary step (fetch, forward, backward) runs inside torch.random.fork_rng with the seed fm_fork_seed(seed, fold, epoch, step): the
#   global CPU / CUDA random states after it are the ones before it, bit for bit.  Sampling and augmentation draws: two private
#   torch.Generators seeded per epoch from 9800 + 17*seed + fold + 1000*epoch (augmentation: + 2**32), so a resume at an epoch boundary
#   repeats the uninterrupted run's auxiliary batches; 2 loader workers of their own (--fm-aux-workers).  The aux_* head is created last
#   from a private generator (the global stream is untouched), trains in the head group, joins the EMA and the checkpoint, and is
#   stripped from deploy_state_dict (fold models keep v9.7's key set).  Logs: every --fm-aux-log-every (100) steps |lambda grad L_aux| /
#   |grad L_main| over the encoder (before the clip); per epoch L_dense, L_presence, the sample digest and the per-class presence AUC on
#   the store's aux_holdout volumes (window level; volume level as information), at the fold's end with the shipped (EMA) weights, plus the
#   B5 gate (meniscus_tear >= 0.75, effusion >= 0.80, cart_full >= 0.75) -> fm_aux_<arm>_s<seed>_f<k>.json.  Refused: a non-ConvNeXt
#   backbone (BatchNorm backbones: the aux batch would move the statistics; ViT / OFL: no map), no --spatial-attn, --reeval-run, a crop that
#   is not a multiple of 32 up to 384, M not a multiple of 8, M < 0.25 x the main windows per step, a store failing FmStore's checks.
#   warm_load also accepts a source without the aux_* keys (missing keys must be exactly that set).  E (v9.7 audit notes): E1 --reeval-run
#   with a state spec whose md5 differs from the one in the source run_config is refused at start-up (both paths and md5s in the message);
#   E2 an --out that already holds a done marker ((partial_)preds_<arm>_s<seed>.json, the state read-out's preds_state_* excepted) is
#   refused before any file is written, and run_config.json / pool_uids.json are written only after every start-up check (now including
#   the arms' targets, the reeval pool and E1) has passed.  Off (no --fm-aux, the default) = v9.7 bit for bit in every output and printed
#   line, except E1 / E2's refusals (base: compact_train_v97.py md5 0b7bc5d8).
# v9.7 (10-10 21:00, spec card spec_trainer_v97_state_oof_20261010.md): --save-state-oof = the state head's read-out written next to the
#   class OOF (R6 state-level fusion in the label builder).  The forward that gives a study's class logits in predict() (now called
#   with return_aux) also returns the state head's logits; for every held-out study a fold model predicts (its own fold; with
#   --holdout-fold also the holdout fold's studies) and for every gold study it predicts: per class the probability vector over the
#   class's categories = the softmax over the class's valid slots, the very p of Net.state_evidence, and per binary head
#   sigmoid(logit).  Files (the partial_ prefix with --only-folds as for the class files; also under --reeval-run):
#   oof_state_<arm>_s<seed>.npy  float32 [pool, D], rows = pool_uids.json, columns = per class (LABELS order) its categories in the
#     spec's order, then the binaries in the spec's order; NaN = a row this process did not predict (another fold's study);
#     holdout-fold rows = sum over the fold models of p / (K - 1) (the class OOF's convention);
#   oof_state_<arm>_s<seed>.json  the column layout (class, category names, column range; binary, class, column) + provenance;
#   preds_state_<arm>_s<seed>.json  gold_uids + per fold model and per class the gold studies' probability lists (+ their mean
#     over the fold models once the file holds every fold model).
#   Written before the preds json (the run's done marker); a done run without them is refused, not skipped.  Needs the state head
#   (--state-targets).  Merge fold-parallel runs with merge_partial_v2.py.  Training, losses, sampling, augmentation, EMA and the
#   class outputs are untouched; off (default) = v9.6 bit for bit (base: compact_train_v96.py md5 d0d362c3).
# v9.6 (10-10 18:00, spec card spec_r5_repair_training_20261010.md): --warm-start DIR = repair training.  Fold k starts from the source
#   run's fold-k model DIR/models/<arm>_s<seed>_f<k>.pt: the WHOLE Net state dict (encoder and every head, parameters and buffers) loaded
#   with strict=True after the initialisation and the bias0 copy, before the optimizer and the EMA are built -- so the EMA starts from the
#   loaded weights.  The optimizer and the LR schedule start fresh over --epochs (same warm-up + cosine shape).  --warm-lr-scale s
#   (default 1.0) multiplies every parameter group's LR (encoder, the layer-wise groups, head) before the schedule records it.  Refused at
#   start-up unless DIR's run_config equals this run in K (--folds), the holdout fold, backbone, crop, slices, spatial attention, extent
#   (and its classes) and syn-decomp, the source's state spec has this run's categories and binaries, pool_uids.json equals the pool,
#   every fold model to load exists and --out is not the source; at load time the configuration-derived buffers of the fresh Net must
#   equal the file's.  The state spec's prior / base values MAY differ (coordinator correction, 10-10: R5 trains with a re-calibrated
#   spec): after the strict load, state_prior / state_valid / state_base_logit are re-set from this run's spec (per-class change logged);
#   every learned parameter keeps the loaded value.  A fold-k model never saw fold k's studies and the hash folds are the same, so the
#   continued model's out-of-fold predictions stay out-of-fold.  ofl_l is refused (its fold files omit the frozen public weights).
#   --aug-grid-fp32 (coordinator addition, 10-10): under --aug v2 the rotation / scale grid
#   (F.affine_grid, whose matmul autocast runs in bf16 inside the training autocast region: ~0.7 px grid error) and its bilinear
#   resampling (F.grid_sample) run in float32 with autocast off for just those two calls; same draws, same order, same gamma / contrast.
#   --draw-seed (coordinator, SCAN_A D1): the training samplers draw with replacement, but Studies seeded each study's window choice by
#   (seed, epoch, index), so a study drawn k times in an epoch got the same windows k times; with the flag every training draw carries a
#   nonce (its position in the epoch) and seeds [seed, epoch, index, nonce]; the drawn index sequence is unchanged; int indices
#   (evaluation, BN recalibration) keep the old seeding.  --rare-boost-names (SCAN_A D4): the rare-boost terms by state name -- MCL +2 no
#   longer for chronic_or_healed_change (v9.5's "MCL >= 2"); the other four terms asserted to select the same studies as before.  Each
#   of the three switches is recorded in the checkpoint (a resumed fold cannot switch mode).
#   Without any v9.6 flag (the defaults) = v9.5 bit for bit (base: compact_train_v95.py md5 64358dcb).
# v9.5 (10-10 12:00, spec card spec_trainer_v95_shift_20261010.md): --shift-px N = batch translation jitter, one draw per TRAINING
#   batch: inside augment(), after the crop and before the --aug v2 affine, every window of the batch (every study, window, plane and
#   slice channel) is translated by the same integer (dy, dx), each uniform on -N..N, with zero fill -- the batch is padded with N
#   zeros on both pixel axes and cropped back at offset (N + dy, N + dx) (never torch.roll: nothing wraps around).  The draws come
#   from their own torch.Generator (9500 + 17*seed + fold, checkpointed as shift_gen), so the crop offsets, rotation / scale,
#   resolution, gamma / contrast, mirror, sampler and dropout draws are v9.4's with the flag on.  Purpose: --crop 384 gets the
#   +-16 px translation that --crop 352 takes from its random crop offset (the R4 field-of-view screen changed both at once).
#   Never in predict / reeval / bn_recalibrate (centre crop).  --shift-px 0 (default) = v9.4 bit for bit: no draw, no tensor touched
#   (base: compact_train_v94.py md5 450b5fc6).
# v9.4 (10-10 10:30, spec card spec_trainer_v94_mirror_20261010.md): --mirror p = anatomical mirror + output-permutation augmentation,
#   one draw per TRAINING batch (both studies of a batch, so both studies of a matched pair, share it): coronal / axial windows are
#   flipped on the width axis (the patient's left-right axis), sagittal windows are NOT flipped in-plane (in-plane left-right is
#   anterior-posterior) but reversed along the slice axis -- the K slice channels [s-r .. s+r] -> [s+r .. s-r] and rel_pos -> -rel_pos
#   (the cache stores sagittal slices in a side-canonical order) -- and the coronal bilateral locals change sides (view id 2 <-> 3).
#   The model keeps the canonical class order; for a mirrored batch its outputs are re-indexed by P = swap(Medial, Lateral Meniscus)
#   x swap(Medial OA, Lateral OA) before EVERY loss (12 logits, state logits and binaries per class, extent area; the syn-decomp
#   columns are fixed points), targets / weights / state targets / pair signatures untouched.  MCL is not trained on a mirrored
#   batch (main, state and pair weights 0: the mirrored MCL position has no truth).  Never in predict / reeval / bn_recalibrate.
#   --extent-classes = the OA classes whose logit receives the extent head's area term (default the three OA classes; the area,
#   its supervision and the area families are unchanged).  Both flags at their defaults = v9.3 bit for bit (base:
#   compact_train_v93.py md5 2c523804).
# v9.3 (10-10 02:30, spec card spec_trainer_v93_20261010.md; designs mechanisms_r4d_20261010.md 1.4 and 3.5): the P1 families are
#   indexed by FAMILY (role / sig / pw are [N, F]; a class may own several families; a family may rank the class logit or the
#   extent head's area; optional per-study masks) -- with the v9 list this is the v9.2 computation column for column.  Four
#   families behind four flags: --oa-area-rank (high_grade_large > high_grade_small on log1p(area/100), + the extent head's
#   soft target 0.2 for high_grade_small; needs the opusv11c spec), --oa-sign-decoy (high grade with no bone sign written >
#   low grade with a sign written), --acl-acuity-decoy (chronic / calm-unstated ACL tear > ACL normal with acute co-findings;
#   age from --lig-age <csv>), --mcl-boundary (high-grade partial / complete, not chronic > grade 2).  All off = v9.2b bit for bit
#   (base: compact_train_v92.py md5 29e681e7, the audited v9.2b).
# v9.2b (10-10 01:50, audit D1 fixed: beta logged per epoch inside the fold loop): --syn-decomp = synovitis logit decomposed into a specific part + beta * stop-grad(effusion logit) (design mechanisms_r4d_20261010.md 2.4); without it identical to v9.1.
# v9.1 (10-09 23:50): --grad-ckpt-blocks N = partial gradient checkpointing (memory only, same maths); without it identical to v9.
(the mainline OrthoFoundation training hold on the 5090 is untouched).  v1.1: fold models are saved.
v9 (10-09 evening, spec card plans/spec_trainer_v9_grade_pairs_20261009.md): optional --pair-extra adds four within-class grade
families to the P1 matched-pair ranking -- Effusion moderate/large over none/small, Baker's moderate/large over none/small,
Contusion contusion/traumatic oedema over degenerative/reactive oedema, Fracture acute/impaction/avulsion over trabecular
microfracture/old fracture -- the grade boundaries where both R2 students err together (shared-error map, 10-09).  These
families match no co-finding (signature 0).  Without --pair-extra the code path is v8's.

Same contract as probe_train.py (so probe_eval.py and fuse_eval.py read its outputs unchanged):
  - never reads official labels; gold studies are never trained on; targets come from the same CSVs
  - K-fold over the non-gold pool with the same hash folds; writes gold predictions (mean of K models)
    and out-of-fold predictions for every pool study
  - arms: S_strict, P_presence, E_prior, E_lrprior, or name=bce:<column> / name=lr:<column> from --extra

Model: 2.5D windows (three physically adjacent slices as channels) -> CNN encoder -> per-class attention
over the windows of a study -> 12 logits.  Training samples a fixed number of windows per study; evaluation
uses every third slice as a centre (the mainline's cover sampling) in both views.
"""
import argparse, collections, contextlib, csv, hashlib, json, math, os, re, sys, time
sys.dont_write_bytecode = True
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
C = len(LABELS)


PREDICT_CHUNK = int(os.environ.get("CT_PREDICT_CHUNK", "512"))   # v7: evaluation windows per encoder call (512 = v6)


def covering_centers(n, phase=1, stride=3):
    if stride <= 1:
        return list(range(n))
    c = list(range(min(phase, n - 1), n, stride))
    if c[0] > 1:
        c.insert(0, 0)
    if c[-1] < n - 2:
        c.append(n - 1)
    return c


def phys_step(spacing_mm, unit_mm):
    """slices per ~unit_mm of anatomy: 1 for an ordinary 3-4 mm series, ~9 for a 0.4 mm 3-D stack (unit_mm <= 0: always 1)."""
    if unit_mm <= 0:
        return 1
    return max(1, int(round(unit_mm / max(float(spacing_mm), 0.1))))


def window_rows(blocks, meta_of, mode, slices, eval_stride=3, cover_stride=3, phys_mm=0.0, rng=None, slice_jitter=0.0,
                train_sampling="cover"):
    """The window list of one study.  blocks: [{'n_slices': n, 'views': [(file, view_id, start), ...]}]; meta_of(file, row)
    -> the px meta row of that window (column 5 = log1p(slice spacing in mm)).  Returns tuples
    (file, centre row, row of slice -r, ..., row of slice +r).  'train' covers every (series, view) at stride
    cover_stride x phys_step with a random phase (and takes the K neighbours two steps apart with probability slice_jitter);
    'eval' covers at stride eval_stride x phys_step.  phys_mm = 0 reproduces the old per-slice behaviour."""
    cand = []
    r_ = slices // 2
    for b in blocks:
        n = b["n_slices"]
        for f, view_id, start in b["views"]:
            ps = phys_step(math.expm1(float(meta_of(f, start)[5])), phys_mm)
            if mode == "train":
                if train_sampling == "cover":
                    st = cover_stride * ps
                    ks = covering_centers(n, int(rng.integers(0, st)), st)
                else:
                    ks = range(n)
            else:
                ks = covering_centers(n, 1, eval_stride * ps)
            for k in ks:
                step_ = ps * (2 if (mode == "train" and slice_jitter > 0 and rng.random() < slice_jitter) else 1)
                cand.append((f, start + k) + tuple(start + min(n - 1, max(0, k + d * step_)) for d in range(-r_, r_ + 1)))
    return cand


class Studies(torch.utils.data.Dataset):
    """Returns uint8 windows [W,3,S,S] and meta [W,6] for one study.  mode 'train' samples, 'eval' covers."""

    def __init__(self, px_dir, uids, n_win, mode, seed, eval_stride=3, train_sampling="uniform", cover_stride=3, px_extra=None, slices=3,
                 slice_jitter=0.0, phys_mm=0.0, px_override=None, extent_meta=False):
        self.dir, self.uids, self.n_win, self.mode, self.seed = px_dir, uids, n_win, mode, seed
        self.extent_meta = bool(extent_meta)
        self.eval_stride, self.train_sampling, self.cover_stride = eval_stride, train_sampling, cover_stride
        self.slices = int(slices)
        self.slice_jitter = float(slice_jitter)
        self.phys_mm = float(phys_mm)
        assert self.slices % 2 == 1 and self.slices >= 1
        index = json.load(open(os.path.join(px_dir, "index.json"), encoding="utf-8"))
        # blocks carry (file, view_id, start): file 0 = the main cache, file 1 = the add-on cache (e.g. the v2 patellofemoral
        # view built with px_cache --views v2add); a study missing from the add-on simply has no add-on windows
        self.blocks = {u: [dict(b, views=[(0, v, st) for v, st in b["views"]]) for b in index["blocks"][u]] for u in uids}
        self.dirs = [px_dir]
        if px_extra:
            extra = json.load(open(os.path.join(px_extra, "index.json"), encoding="utf-8"))
            self.dirs.append(px_extra)
            n_add = 0
            for u in uids:
                eb = extra["blocks"].get(u)
                if not eb:
                    continue
                main = self.blocks[u]
                if [b["series"] for b in eb] != [b["series"] for b in main]:
                    raise SystemExit("add-on cache series order differs from the main cache for study " + u)
                for b, e in zip(main, eb):
                    b["views"] = b["views"] + [(1, v, st) for v, st in e["views"]]
                    n_add += len(e["views"])
            print("add-on cache %s: %d extra (series, view) blocks for %d studies" % (px_extra, n_add, len(uids)), flush=True)
        if px_override:
            # override cache (sidefix): the listed studies are read entirely from it (same format, own row space), e.g.
            # the studies whose knee side the image check corrected, re-rendered through the inference path
            if px_extra:
                raise SystemExit("--px-override cannot be combined with --px-extra")
            ov = json.load(open(os.path.join(px_override, "index.json"), encoding="utf-8"))
            f_ov = len(self.dirs)
            self.dirs.append(px_override)
            n_ov = 0
            for u in uids:
                ob = ov["blocks"].get(u)
                if ob is not None:
                    self.blocks[u] = [dict(b, views=[(f_ov, v, st) for v, st in b["views"]]) for b in ob]
                    n_ov += 1
            print("override cache %s: %d of %d studies read from it" % (px_override, n_ov, len(uids)), flush=True)
        self.px = self.meta = None
        self.epoch = 0

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        if self.px is None:
            self.px = [np.load(os.path.join(d, "px.u8.npy"), mmap_mode="r") for d in self.dirs]
            self.meta = [np.load(os.path.join(d, "meta.f32.npy"), mmap_mode="r") for d in self.dirs]
        nonce = None
        if isinstance(i, tuple):                    # v9.6 --draw-seed: (index, nonce) from a training sampler (an int index = v9.5's path)
            i, nonce = i
        uid = self.uids[i]
        rng = np.random.default_rng(self.seed + 1000003 * self.epoch + i) if self.mode == "train" else None
        if nonce is not None and self.mode == "train":   # v9.6 --draw-seed: this draw's own stream (repeat draws of a study differ)
            rng = np.random.default_rng([int(self.seed), int(self.epoch), int(i), int(nonce)])
        # 'cover': every series/view is covered at stride cover_stride (x the physical step) with a random phase, so a
        # focal finding spanning >= one stride is in the sampled set of every epoch (uniform sampling of ~10% of the
        # windows misses a 3-slice finding in about half of the epochs).  Rows: (file, centre row, K slice rows k-r..k+r).
        cand = window_rows(self.blocks[uid], lambda f, r: self.meta[f][r], self.mode, self.slices, self.eval_stride,
                           self.cover_stride, self.phys_mm, rng, self.slice_jitter, self.train_sampling)
        if self.mode == "train" and len(cand) > self.n_win:
            cand = [cand[j] for j in np.sort(rng.choice(len(cand), self.n_win, replace=False))]
        rows = np.array(cand, np.int64)
        S = self.px[0].shape[-1]
        img = np.empty((len(rows), self.slices, S, S), np.uint8)
        m = np.empty((len(rows), self.meta[0].shape[1]), np.float32)
        for f in range(len(self.dirs)):
            sel = np.flatnonzero(rows[:, 0] == f)
            if not len(sel):
                continue
            r = rows[sel]
            img[sel] = np.stack([np.asarray(self.px[f][r[:, 2 + d]]) for d in range(self.slices)], 1)
            m[sel] = np.asarray(self.meta[f][r[:, 1]])
        meta = np.stack([m[:, 1], m[:, 3], m[:, 4], m[:, 5], m[:, 6], m[:, 7]], 1)   # slot, view, rel_pos, log_spacing, extent, fluid
        if self.extent_meta:                                   # + series index, plane (0 sag / 1 cor / 2 ax), slices in the series
            meta = np.concatenate([meta, np.stack([m[:, 0], m[:, 2], m[:, 10]], 1)], 1)
        return torch.from_numpy(img), torch.from_numpy(meta.astype(np.float32)), i


def collate(batch):
    W = max(len(b[0]) for b in batch)
    S = batch[0][0].shape[-1]
    img = torch.zeros((len(batch), W, batch[0][0].shape[1], S, S), dtype=torch.uint8)
    meta = torch.zeros((len(batch), W, batch[0][1].shape[1]), dtype=torch.float32)
    mask = torch.zeros((len(batch), W), dtype=torch.bool)
    for j, (x, m, _) in enumerate(batch):
        img[j, :len(x)] = x
        meta[j, :len(x)] = m
        mask[j, :len(x)] = True
    return img, meta, mask, torch.tensor([b[2] for b in batch])


OFL_BACKBONE = "ofl_l"


class OFLEncoder(nn.Module):
    """OrthoFoundation-L (public self-supervised weights) as the window encoder, run exactly as the feature extractor
    runs it: timm Eva vit_large_patch16_dinov3_qkvb, RoPE removed, 5 prefix tokens, ImageNet normalisation (done by
    Net).  Feature = [CLS token, mean patch token] (2048).  The patch embedding, the tokens, norm_pre and the first
    `freeze` of the 24 blocks are frozen and run without a graph; the remaining blocks and the final norm are trained.
    `deploy_state_dict` returns only what differs from the public weights, so a fold model can be shipped small and
    rebuilt on top of the public weights."""

    def __init__(self, init, freeze=16, img_size=256):
        super().__init__()
        import timm
        self.vit = timm.create_model("vit_large_patch16_dinov3_qkvb", pretrained=False, num_classes=0, global_pool="",
                                     img_size=img_size)
        if init:
            if str(init).endswith(".safetensors"):
                from safetensors.torch import load as load_bytes
                with open(str(init), "rb") as fh:          # one sequential read; the mmap loader page-faults 4 KB at a time
                    sd = load_bytes(fh.read())             # (0.4 MB/s from a USB disk: 48 min per fold on the 5090)
            else:
                sd = torch.load(init, map_location="cpu")
            self.vit.load_state_dict(sd, strict=True)
        self.vit.rope = None
        self.freeze = int(freeze)
        self.num_features = 2 * self.vit.embed_dim
        self.frozen_keys = set()
        frozen_modules = [self.vit.patch_embed, self.vit.norm_pre] + list(self.vit.blocks[: self.freeze])
        for mod in frozen_modules:
            for p in mod.parameters():
                p.requires_grad_(False)
        for name in ("cls_token", "reg_token", "pos_embed"):
            t = getattr(self.vit, name, None)
            if isinstance(t, nn.Parameter):
                t.requires_grad_(False)
        for k, p in self.vit.named_parameters():
            if not p.requires_grad:
                self.frozen_keys.add("vit." + k)

    def trunk(self, x):
        v = self.vit
        with torch.no_grad():
            x = v.patch_embed(x)
            x, _ = v._pos_embed(x)
            x = v.norm_pre(x)
            for blk in v.blocks[: self.freeze]:
                x = blk(x, rope=None)
        return x

    def tail(self, x):
        v = self.vit
        for blk in v.blocks[self.freeze:]:
            x = blk(x, rope=None)
        x = v.norm(x)
        return torch.cat([x[:, 0], x[:, v.num_prefix_tokens:].mean(1)], -1)

    def forward(self, x):
        return self.tail(self.trunk(x))


def partial_grad_checkpointing(enc, n):
    """v9.1: recompute the first n transformer blocks of a timm ViT encoder in backward (torch.utils.checkpoint,
    use_reentrant=False, as timm's own grad checkpointing).  Memory saved ~ n/len(blocks) of full checkpointing, extra compute
    the same fraction.  Outputs and gradients are those of the plain forward (a checkpoint only changes WHEN activations are
    computed).  Eval / no_grad passes run the plain forward."""
    import torch.utils.checkpoint as _ckpt
    blocks = getattr(enc, "blocks", None)
    if not isinstance(blocks, (nn.Sequential, nn.ModuleList)):
        raise SystemExit("--grad-ckpt-blocks needs a ViT-style timm encoder with .blocks (use --grad-ckpt for others)")
    n = min(int(n), len(blocks))

    def wrap(f):
        def fwd(x, *args, **kw):
            if torch.is_grad_enabled() and x.requires_grad:
                return _ckpt.checkpoint(f, x, *args, use_reentrant=False, **kw)
            return f(x, *args, **kw)
        return fwd
    for blk in list(blocks)[:n]:
        blk.forward = wrap(blk.forward)
    return n


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


def bce_soft(z, target, weight):
    loss = F.binary_cross_entropy_with_logits(z, target, reduction="none") * weight
    den = weight.sum(0)
    ok = den > 0
    return (loss.sum(0)[ok] / den[ok]).mean()


def lr_loss(z, lr, weight=None):
    """Likelihood-ratio loss on soft targets.  v8: optional per-cell weight [B, C]; the mean stays over cells, so a cell
    with weight 0.5 pushes half as hard (weight None = the v7 loss, bit for bit)."""
    per = -torch.logaddexp(F.logsigmoid(z) + torch.log(lr), F.logsigmoid(-z))
    if weight is None:
        return per.mean(0).mean()
    return (per * weight).mean(0).mean()


def state_loss(sl, bl, st_idx, st_w, bin_t):
    """Masked cross-entropy over report states (per class, weight-normalised) + masked BCE over the binaries.
    st_idx [B, C] long (-1 unknown), st_w [B, C], bin_t [B, nbin] (nan unknown)."""
    total, n = sl.new_zeros(()), 0
    for c in range(sl.shape[1]):
        ok = st_idx[:, c] >= 0
        if ok.any():
            ce = F.cross_entropy(sl[ok, c], st_idx[ok, c], reduction="none") * st_w[ok, c]
            total = total + ce.sum() / st_w[ok, c].sum().clamp_min(1e-6); n += 1
    known = ~torch.isnan(bin_t)
    if known.any():
        t = torch.nan_to_num(bin_t)
        bce = F.binary_cross_entropy_with_logits(bl, t, reduction="none") * known
        per = bce.sum(0) / known.sum(0).clamp_min(1)
        total = total + per[known.any(0)].mean(); n += 1
    return total / max(n, 1)


# ---- v9.3: P1 families indexed by FAMILY (shared refactor) and the four R4d families ---------------------------------------
AGE_CODES = {"age_unstated": 0, "chronic": 1, "acute": 2, "graft": 3, "no_tear": 4}   # --lig-age csv values (the E8 LIG protocol)
OA_SMALL_TARGET = 0.2       # --oa-area-rank: extent-head soft target of a high_grade_small cell (prior 0.33-0.36: < 1 cm^2 more often than not)
OA_L = ("Medial OA", "Lateral OA", "PF OA")


def load_lig_age(path, pool):
    """v9.3: tear-age attribute per (study, class) from a csv with columns study, class, age (AGE_CODES keys): the E8 re-read
    (reader A, provisional) or the adjudicated r4d_inputs/ligament_age.csv.  Cells absent from the csv = age_unstated (0),
    as the coordinator's rule says (the state table's own acuity binary is NOT consulted)."""
    Li = {l: c for c, l in enumerate(LABELS)}
    idx = {u: i for i, u in enumerate(pool)}
    out = np.zeros((len(pool), C), np.int64)
    n_in = n_out = 0
    counts = collections.Counter()
    for r in csv.DictReader(open(path, encoding="utf-8")):
        age, cls = (r.get("age") or "").strip(), (r.get("class") or "").strip()
        if age not in AGE_CODES:
            raise SystemExit("--lig-age: unknown age value %r for %s / %s" % (age, r.get("study"), cls))
        if cls not in Li:
            raise SystemExit("--lig-age: unknown class %r" % cls)
        i = idx.get((r.get("study") or "").strip())
        if i is None:
            n_out += 1
            continue
        out[i, Li[cls]] = AGE_CODES[age]
        n_in += 1
        counts[(cls, age)] += 1
    print("lig-age %s: %d cells on pool studies (%d rows outside the pool) %s" % (path, n_in, n_out, dict(sorted(counts.items()))), flush=True)
    return out


def build_pair_families(spec, sti, bint, bin_names, pair_extra, oa_area_rank=False, oa_sign_decoy=False,
                        acl_acuity_decoy=False, mcl_boundary=False, lig_age=None):
    """v9.3: the P1 families as a list indexed by FAMILY (v8 / v9 indexed them by class: one family per class).  A family is
    {label, pos / neg (state names), cof (co-finding arrays, -1 = unknown -> the study leaves the family), pos_mask / neg_mask
    (optional bool[N]: an extra per-study membership condition beyond the state), quantity ("z" = class logit, "area" =
    log1p(extent area / 100)), name}.  Returns (fams, role[N, F] in {+1, -1, 0}, sig[N, F], pw[N, F] = report prior of the
    study's state in the family's class, fam_class[F]).  With the v9 list and no v9.3 flag, column f of role / sig / pw is
    v9.2's column of that family's class, in the same family order (v93_identity_test.py checks it on the real tables)."""
    cats, N = spec["categories"], len(sti)
    Li = {l: c for c, l in enumerate(LABELS)}
    bni = {(b, l): j for j, (b, l) in enumerate(bin_names)}

    def sset(l, *names):
        return [cats[l].index(x) for x in names]

    def inset(l, idx_):                     # 1 / 0 per study, -1 when the report is silent on l
        v = sti[:, Li[l]]
        return np.where(v < 0, -1, np.isin(v, idx_).astype(np.int64))

    def osteo(l):                           # "osteophytes written or not" (the table has 0 only for a normal compartment)
        v = bint[:, bni[("sign_osteophytes", l)]]
        return (v == 1).astype(np.int64)

    TORN_M, HIGH = ("tear", "macerated_or_destroyed"), ("high_grade_size_unstated", "high_grade_large")
    # v9.3: with a spec that carries high_grade_partial_tear (opusv11c) that officially positive state is an ACL tear both as a
    # co-finding (signature of the meniscus families) and as a positive of the ACL family; with opusv11b the tuple is v9.2's
    acl_torn = tuple(s for s in ("partial_tear_unspecified", "tear_unspecified", "complete_tear", "high_grade_partial_tear") if s in cats["ACL"])
    acl_t = inset("ACL", sset("ACL", *acl_torn))
    mm_t = inset("Medial Meniscus", sset("Medial Meniscus", *TORN_M))
    lm_t = inset("Lateral Meniscus", sset("Lateral Meniscus", *TORN_M))
    hi = {l: inset(l, sset(l, *HIGH)) for l in OA_L}
    cont = inset("Contusion", sset("Contusion", "contusion_or_bruise", "traumatic_marrow_edema"))
    ev = sti[:, Li["Effusion"]]
    eff_b = np.select([ev < 0, ev == 0, ev == 1, np.isin(ev, [2, 3])], [-1, 0, 1, 2], 3)
    fams = [("Lateral Meniscus", TORN_M, ("normal", "degeneration_no_tear"), [acl_t, mm_t, hi["Lateral OA"]]),
            ("Medial Meniscus", TORN_M, ("normal", "degeneration_no_tear"), [acl_t, lm_t, hi["Medial OA"]]),
            ("Lateral OA", HIGH, ("normal", "low_grade_chondral"), [osteo("Lateral OA"), hi["Medial OA"], hi["PF OA"]]),
            ("Medial OA", HIGH, ("normal", "low_grade_chondral"), [osteo("Medial OA"), hi["Lateral OA"], hi["PF OA"]]),
            ("PF OA", HIGH, ("normal", "low_grade_chondral"), [osteo("PF OA"), hi["Medial OA"], hi["Lateral OA"]]),
            ("Synovitis", ("synovitis", "synovial_thickening"), ("other_synovial_finding", "none"), [eff_b]),
            ("MCL", ("partial_tear_or_grade2", "high_grade_partial_tear", "complete_tear"),
             ("normal", "periligamentous_edema_or_grade1", "chronic_or_healed_change"), [mm_t, cont, acl_t]),
            ("ACL", acl_torn,
             ("normal", "signal_or_degeneration", "low_grade_sprain_or_partial"), [cont, lm_t, mm_t])]
    if pair_extra:                          # v9: within-class grade boundaries, no co-finding matching
        fams += [("Effusion", ("moderate", "large"), ("none_or_trace", "small"), []),
                 ("Baker's", ("moderate", "large"), ("none", "small"), []),
                 ("Contusion", ("contusion_or_bruise", "traumatic_marrow_edema"), ("degenerative_or_reactive_edema",), []),
                 ("Fracture", ("acute_fracture", "impaction_or_osteochondral_fracture", "avulsion_fracture"),
                  ("trabecular_microfracture_only", "old_or_healed_fracture"), [])]
    fams = [dict(label=l, pos=p, neg=n, cof=cf, pos_mask=None, neg_mask=None, quantity="z", name=l) for l, p, n, cf in fams]
    if oa_sign_decoy:                       # v9.3: the bone-sign proxy crossed with the grade (signature 0: the crossing is the point)
        for l in OA_L:
            written = np.zeros(N, bool)
            for b in ("sign_osteophytes", "sign_subchondral_marrow", "sign_bone_on_bone"):
                written |= bint[:, bni[(b, l)]] == 1
            fams.append(dict(label=l, pos=HIGH, neg=("low_grade_chondral",), cof=[], pos_mask=~written, neg_mask=written,
                             quantity="z", name=l + "/sign_decoy"))
    if oa_area_rank:                        # v9.3: the official area boundary inside the high grade, asked of the extent head
        for l in OA_L:
            if "high_grade_small" not in cats[l]:
                raise SystemExit("--oa-area-rank needs the state high_grade_small in the spec of %s (state_spec_opusv11c.json)" % l)
            fams.append(dict(label=l, pos=("high_grade_large",), neg=("high_grade_small",), cof=[], pos_mask=None, neg_mask=None,
                             quantity="area", name=l + "/area_rank"))
    if acl_acuity_decoy:                    # v9.3: a tear without acute company > an intact ACL with acute company
        if lig_age is None:
            raise SystemExit("--acl-acuity-decoy needs --lig-age")
        age = lig_age[:, Li["ACL"]]
        tears = tuple(s for s in ("tear_unspecified", "complete_tear", "high_grade_partial_tear") if s in cats["ACL"])
        acute_cof = (cont == 1) | np.isin(ev, sset("Effusion", "moderate", "large"))
        calm = (cont == 0) & np.isin(ev, sset("Effusion", "none_or_trace", "small"))
        pos_mask = (age == AGE_CODES["chronic"]) | ((age == AGE_CODES["age_unstated"]) & calm)
        fams.append(dict(label="ACL", pos=tears, neg=("normal", "signal_or_degeneration", "low_grade_sprain_or_partial"), cof=[],
                         pos_mask=pos_mask, neg_mask=acute_cof, quantity="z", name="ACL/acuity_decoy"))
    if mcl_boundary:                        # v9.3: the official MCL boundary (high-grade partial / complete acute > grade 2)
        if lig_age is None:
            raise SystemExit("--mcl-boundary needs --lig-age")
        age = lig_age[:, Li["MCL"]]
        pos_mask = ~np.isin(age, [AGE_CODES["chronic"], AGE_CODES["graft"], AGE_CODES["no_tear"]])
        fams.append(dict(label="MCL", pos=("high_grade_partial_tear", "complete_tear"), neg=("partial_tear_or_grade2",), cof=[],
                         pos_mask=pos_mask, neg_mask=None, quantity="z", name="MCL/boundary"))
    Fn = len(fams)
    role = np.zeros((N, Fn), np.int64)             # +1 positive / -1 hard negative of the family / 0
    sig = np.zeros((N, Fn), np.int64)              # co-finding signature: pairs only within one signature
    pw = np.zeros((N, Fn), np.float32)              # report prior of the study's state in the family's class (pair weight)
    fam_class = []
    for f, fm in enumerate(fams):
        l = fm["label"]
        c = Li[l]
        v = sti[:, c]
        pm = np.isin(v, sset(l, *fm["pos"]))
        if fm["pos_mask"] is not None:
            pm &= fm["pos_mask"]
        nm = np.isin(v, sset(l, *fm["neg"]))
        if fm["neg_mask"] is not None:
            nm &= fm["neg_mask"]
        role[pm, f] = 1
        role[nm, f] = -1
        unknown_cof = np.zeros(N, bool)
        for arr in fm["cof"]:
            unknown_cof |= np.asarray(arr) < 0
        role[unknown_cof, f] = 0                 # an unknown co-finding cannot be matched: the study leaves this family
        code = np.zeros(N, np.int64)
        for arr in fm["cof"]:
            code = code * 8 + (np.asarray(arr, np.int64) + 1)
        sig[:, f] = code
        ok = v >= 0
        pw[ok, f] = np.asarray(spec["prior"][l], np.float32)[v[ok]]
        fam_class.append(c)
    return fams, role, sig, pw, fam_class


def pair_buckets_of(role, sig, pi):
    """(positive dataset indices, negative dataset indices, probability) per (family, signature) with both present; every
    family gets the same total probability, buckets within it ~ sqrt(n_pos * n_neg).  v9.2's pair_buckets with the family
    axis in place of the class axis: the same list in the same order with the same probabilities for the v9 list."""
    Fn = role.shape[1]
    out = []
    for f in range(Fn):
        r, sg = role[pi, f], sig[pi, f]
        fam = []
        for v in np.unique(sg[r != 0]):
            P_ = np.flatnonzero((r == 1) & (sg == v)); N_ = np.flatnonzero((r == -1) & (sg == v))
            if len(P_) and len(N_):
                fam.append((torch.from_numpy(P_), torch.from_numpy(N_), math.sqrt(len(P_) * len(N_))))
        tot = sum(b[2] for b in fam)
        out += [(p_, n_, w_ / tot / Fn) for p_, n_, w_ in fam]
    return out


def family_summary(fams, role, sig):
    """per family: positives, negatives, buckets (signatures with both sides) and pairable (positive, negative) pairs over the rows given."""
    out = []
    for f, fm in enumerate(fams):
        r, sg = role[:, f], sig[:, f]
        nb = npairs = 0
        for v in np.unique(sg[r != 0]):
            p_, n_ = int(((r == 1) & (sg == v)).sum()), int(((r == -1) & (sg == v)).sum())
            if p_ and n_:
                nb += 1; npairs += p_ * n_
        out.append("%s[%s] pos %d neg %d buckets %d pairs %d" % (fm["name"], fm["quantity"], int((r == 1).sum()), int((r == -1).sum()), nb, npairs))
    return out


def pair_loss_of(z, area, r, sg, pr, cw, fam_class, fam_q):
    """Matched-pair ranking over families: for each ordered (positive i, hard negative j) of family f in the batch with the
    same co-finding signature, w * softplus(q_j - q_i), w = (prior_i - prior_j) * min(sw_i, sw_j); q = the class logit
    (quantity "z") or log1p(area / 100) of the extent head (quantity "area", gradient through the extent read-out and the
    encoder, nothing detached); mean over active families.  v9.2's pair_loss with the family axis in place of the class axis:
    the same operations in the same order for the v9 list."""
    zf = z.float()
    la = None if area is None else torch.log1p(area.float() / 100.0)
    tot, cnt = zf.new_zeros(()), 0
    for f, c in enumerate(fam_class):
        P_ = torch.nonzero(r[:, f] == 1).flatten()
        N_ = torch.nonzero(r[:, f] == -1).flatten()
        if len(P_) == 0 or len(N_) == 0:
            continue
        same = (sg[P_, f][:, None] == sg[N_, f][None, :]).float()
        w = (pr[P_, f][:, None] - pr[N_, f][None, :]).clamp_min(0) * torch.minimum(cw[P_, c][:, None], cw[N_, c][None, :]) * same
        if float(w.sum()) <= 0:
            continue
        q = zf if fam_q[f] == "z" else la
        if q is None:
            raise SystemExit("an 'area' family needs the extent area (--extent)")
        # mean over the valid pairs with w as each pair's weight (dividing by sum(w) would cancel w for a single pair)
        tot = tot + (w * F.softplus(q[N_, c][None, :] - q[P_, c][:, None])).sum() / same.sum()
        cnt += 1
    return tot / max(cnt, 1), cnt


class PairBatchSampler:
    """Training batches of `batch` dataset indices.  With probability `frac` a batch opens with a matched pair -- a positive
    and a hard negative of one pair-family class that share the co-finding signature (bucket drawn with the given
    probabilities) -- and is filled with base draws; otherwise all draws are base draws (weighted with replacement, as
    the rare-boost sampler).  len = n // batch: the same number of steps per epoch as the plain loader."""

    def __init__(self, n, batch, weights, buckets, frac, seed):
        self.n, self.batch, self.frac = int(n), int(batch), float(frac)
        self.w = torch.ones(self.n, dtype=torch.double) if weights is None else torch.as_tensor(weights, dtype=torch.double)
        self.buckets = buckets
        self.bprob = torch.tensor([b[2] for b in buckets], dtype=torch.double) if buckets else None
        self.generator = torch.Generator().manual_seed(int(seed))
        assert self.batch >= 2

    def __len__(self):
        return self.n // self.batch

    def _base(self, k):
        return torch.multinomial(self.w, k, replacement=True, generator=self.generator).tolist() if k > 0 else []

    def __iter__(self):
        g = self.generator
        for _ in range(len(self)):
            if self.buckets and float(torch.rand(1, generator=g)) < self.frac:
                pos, neg, _ = self.buckets[int(torch.multinomial(self.bprob, 1, generator=g))]
                i = pos[int(torch.randint(0, len(pos), (1,), generator=g))]
                j = neg[int(torch.randint(0, len(neg), (1,), generator=g))]
                yield [int(i), int(j)] + self._base(self.batch - 2)
            else:
                yield self._base(self.batch)


# ---- v9.6: --draw-seed (SCAN_A D1) -- every training draw its own window stream ---------------------------------------------------
# Both training samplers draw WITH replacement (pair sampler base draws, rare-boost WeightedRandomSampler), but Studies seeded its rng with
# (seed, epoch, index) only, so a study drawn k times in one epoch was served the same windows k times.  With --draw-seed the training
# samplers below yield (index, nonce), nonce = the draw's position in the epoch (batch * batch size + slot), and Studies seeds
# np.random.default_rng([seed, epoch, index, nonce]).  The subclasses only re-wrap what their parent yields: the parent's generator calls
# -- and so the drawn index sequence -- are unchanged, and the isinstance checks of the checkpoint (sampler stream) keep working.
class NoncePairBatchSampler(PairBatchSampler):
    def __iter__(self):
        for b, batch in enumerate(super().__iter__()):
            yield [(int(i), b * self.batch + s) for s, i in enumerate(batch)]


class NonceWeightedSampler(torch.utils.data.WeightedRandomSampler):
    def __iter__(self):
        for pos, i in enumerate(super().__iter__()):
            yield (int(i), pos)


class NonceRandomSampler(torch.utils.data.RandomSampler):
    def __iter__(self):
        for pos, i in enumerate(super().__iter__()):
            yield (int(i), pos)


# ---- v9.6: --rare-boost-names (SCAN_A D4) -- the rare-boost terms by state name ----------------------------------------------------
# v9.5's rare_weights tests MCL by index (state >= 2), which also boosts index 3 = chronic_or_healed_change, a state the MCL pair family
# treats as a negative.  By name: MCL +2 only for the three tear states below (or the acute-MCL binary, unchanged); the other four terms
# are the names of exactly the indices v9.5 tests (Fracture >= 1, PF OA in {3, 4}, Baker's in {3, 4, 5}, Contusion in {4, 5} under
# opusv11b / opusv11c), asserted at start-up to select the same studies as the index rules under the run's spec, so only MCL changes.
RARE_NAME_TERMS = (("MCL", 2.0, ("partial_tear_or_grade2", "high_grade_partial_tear", "complete_tear")),
                   ("Fracture", 2.0, ("trabecular_microfracture_only", "insufficiency_or_stress_fracture", "avulsion_fracture",
                                      "impaction_or_osteochondral_fracture", "fracture_age_unstated", "acute_fracture", "old_or_healed_fracture")),
                   ("PF OA", 1.0, ("high_grade_size_unstated", "high_grade_large")),
                   ("Baker's", 1.0, ("moderate", "large", "ruptured_or_leaking")),
                   ("Contusion", 1.0, ("traumatic_marrow_edema", "contusion_or_bruise")))
RARE_INDEX_RULES = {"MCL": lambda s: s >= 2, "Fracture": lambda s: s >= 1, "PF OA": lambda s: np.isin(s, (3, 4)),
                    "Baker's": lambda s: np.isin(s, (3, 4, 5)), "Contusion": lambda s: np.isin(s, (4, 5))}   # v9.5's rare_weights, per term


def rare_term_masks(st, cats, acute, by_name):
    """Per rare-boost term a bool [N] over the rows of the state-index matrix st [N, C] (-1 = unknown): by_name -> the RARE_NAME_TERMS name
    sets looked up in the spec's categories, else v9.5's index rules; the MCL term also when the acute-MCL binary is 1 (acute: bool [N])."""
    out = {}
    for l, _, names in RARE_NAME_TERMS:
        s = st[:, LABELS.index(l)]
        m = np.isin(s, [cats[l].index(n_) for n_ in names]) if by_name else np.asarray(RARE_INDEX_RULES[l](s), bool)
        out[l] = (m | acute) if l == "MCL" else m
    return out


def rare_weights_of(masks):
    """v9.5's weight formula on term masks: 1 + 2 MCL + 2 Fracture + PF OA + Baker's + Contusion, capped at 4."""
    v = np.ones(len(masks["MCL"]))
    for l, wt, _ in RARE_NAME_TERMS:
        v = v + wt * masks[l]
    return np.minimum(v, 4.0)


# ---- v9.4: anatomical mirror + output permutation (spec card spec_trainer_v94_mirror_20261010.md) ------------------------------------
MIRROR_PAIRS = (("Medial Meniscus", "Lateral Meniscus"), ("Medial OA", "Lateral OA"))   # the class pairs P exchanges; the rest fixed


def mirror_perm(spec=None, bin_names=None):
    """The output permutation P of a mirrored batch as index lists: perm over the 12 classes (Medial <-> Lateral Meniscus, Medial OA
    <-> Lateral OA, every other class a fixed point; P is its own inverse) and, with a state spec, bin_perm over the binaries (the
    same exchange on the binary's parent class).  Refuses a spec whose two classes of a pair do not share the state category list
    (the state head's per-class K-way outputs would not correspond slot for slot) and a binary that exists for one side only."""
    swap = {}
    for m_, l_ in MIRROR_PAIRS:
        swap[m_], swap[l_] = l_, m_
    perm = [LABELS.index(swap.get(l, l)) for l in LABELS]
    bin_perm = None
    if spec is not None:
        for m_, l_ in MIRROR_PAIRS:
            if spec["categories"][m_] != spec["categories"][l_]:
                raise SystemExit("--mirror needs identical state category lists for %s and %s; the spec has %s vs %s" % (
                    m_, l_, spec["categories"][m_], spec["categories"][l_]))
        if bin_names is not None:
            bin_perm = []
            for b, l in bin_names:
                partner = (b, swap.get(l, l))
                if partner not in bin_names:
                    raise SystemExit("--mirror: the binary %s exists for %s but not for %s" % (b, l, swap[l]))
                bin_perm.append(bin_names.index(partner))
    return perm, bin_perm


def mirror_batch(img, meta):
    """Every window of a training batch as seen in the virtual contralateral knee.  img [B, W, K, S, S] (after augment), meta
    [B, W, >= 8] with the plane in column 7 (0 sagittal / 1 coronal / 2 axial; the loader adds the extent columns for --mirror as
    it does for --extent).  Coronal and axial windows are flipped on the width axis (the patient's left-right axis in both
    planes).  A sagittal window is NOT flipped in-plane (its in-plane axes are anterior-posterior and superior-inferior): its mirror
    runs along the slice axis, which the cache stores in a side-canonical order (px_cache: a right knee's positions are sign-
    reversed before sorting), so the K slice channels [s-r .. s+r] become [s+r .. s-r] and rel_pos (column 2, 2(pos-pos0)/span - 1
    in [-1, 1]) becomes -rel_pos.  The coronal bilateral locals change sides: view id 2 <-> 3 (column 1; coronal_views.py puts them
    at column .25 / .75 of the field, which the flip exchanges).  Slot, spacing, extent, fluid and the extent-head columns are
    unchanged; padded windows (zero meta) are flipped as sagittal zeros and stay zero.  Training forward only: refuses without
    autograd, which is how predict / reeval / bn_recalibrate run their forwards."""
    assert torch.is_grad_enabled(), "v9.4 mirror reached without autograd: the evaluation paths never mirror"
    assert meta.shape[-1] >= 8, "v9.4 mirror needs the plane column (meta column 7)"
    sag = meta[..., 7].round() == 0
    img = torch.where(sag[:, :, None, None, None], img.flip(2), img.flip(-1))
    meta = meta.clone()
    meta[..., 2] = torch.where(sag, -meta[..., 2], meta[..., 2])
    v = meta[..., 1]
    meta[..., 1] = torch.where(v == 2, torch.full_like(v, 3.), torch.where(v == 3, torch.full_like(v, 2.), v))
    return img, meta


def mirror_outputs(z, sl, bl, area, w_main, st_idx, st_w, bin_t, perm, bin_perm, mcl_bins):
    """A mirrored batch's model outputs re-indexed to the canonical class order: z[:, perm] puts the model's lateral output under
    the medial target and vice versa (menisci, OA compartments), the state logits, the binaries and the extent area the same way
    (the syn-decomp columns Synovitis / Effusion are fixed points of P), so the canonical targets, weights, state targets and
    pair-family signatures apply unchanged.  MCL leaves every loss of the batch: main weight 0 (a weight of ones is created when
    the arm carries no cell weights), state weight 0 with the state index unknown, its binaries unknown; the pair loss receives
    the zeroed state weight as its cell weight, so every MCL family's pair weight is 0.  The selections carry gradient as is."""
    MCL = LABELS.index("MCL")
    z = z[:, perm]
    if sl is not None:
        sl, bl = sl[:, perm], bl[:, bin_perm]
    if area is not None:
        area = area[:, perm]
    w_main = torch.ones_like(z) if w_main is None else w_main.clone()
    w_main[:, MCL] = 0
    if st_w is not None:
        st_idx, st_w, bin_t = st_idx.clone(), st_w.clone(), bin_t.clone()
        st_idx[:, MCL] = -1
        st_w[:, MCL] = 0
        if mcl_bins:
            bin_t[:, mcl_bins] = float("nan")
    return z, sl, bl, area, w_main, st_idx, st_w, bin_t


# ---- v9.5: batch translation jitter (spec card spec_trainer_v95_shift_20261010.md) ---------------------------------------------------
def shift_draw(n, gen):
    """One training batch's translation (dy, dx), each uniform on the integers -n .. n, from the shift stream `gen` (its own
    generator: no other augmentation draw moves).  Two draws per batch, dy first."""
    dy = int(torch.randint(-n, n + 1, (1,), generator=gen))
    dx = int(torch.randint(-n, n + 1, (1,), generator=gen))
    return dy, dx


def shift_batch(img, dy, dx, n):
    """Every window of a training batch translated by the same integer (dy, dx) with zero fill.  img [..., H, W] (the cropped uint8
    batch, [B, W, K, crop, crop]) is padded with n zeros on both pixel axes and cropped back at offset (n + dy, n + dx):
    out[..., y, x] = img[..., y + dy, x + dx] where that pixel exists, 0 where it does not (what leaves the field is dropped and the
    entering border is zero -- not torch.roll, which would wrap the opposite side's anatomy in).  One (dy, dx) for every study,
    window, plane and slice channel of the batch.  Pure indexing (no autocast op, no interpolation, the dtype is kept).  Training
    forward only: refuses without autograd, which is how predict / reeval / bn_recalibrate run their forwards (they take the centre
    crop and never reach augment)."""
    assert torch.is_grad_enabled(), "v9.5 shift reached without autograd: the evaluation paths never shift"
    assert -n <= dy <= n and -n <= dx <= n, "v9.5 shift (%d, %d) outside -%d..%d" % (dy, dx, n, n)
    H, W_ = img.shape[-2:]
    return F.pad(img, (n, n, n, n))[..., n + dy:n + dy + H, n + dx:n + dx + W_]


# ---- v9.6: warm start = repair training (spec card spec_r5_repair_training_20261010.md) ----------------------------------------------
# run_config fields that must be equal in the source run (--warm-start DIR) and this run: K and the holdout fold (which studies each fold
# model never saw: the hash folds) and every flag that changes what Net holds (backbone, crop, slices = input channels, spatial attention,
# the extent head and its logit classes, syn-decomp).  A key absent from an older run_config (v9.3 has no extent_classes) is that flag's
# default.  The state spec (by content) and the pool (pool_uids.json) are compared on their own.
WARM_FIELDS = ("folds", "holdout_fold", "backbone", "crop", "slices", "spatial_attn", "extent", "extent_classes", "syn_decomp")
# Net's buffers fall in two groups (Net registers exactly these six).  CONFIG buffers come from flags asserted equal above (input
# normalisation from --slices, the extent logit mask from --extent-classes): a fresh Net of this run must hold exactly the file's values
# (refused otherwise).  SPEC buffers come from the state spec's prior / base values (state prior, valid slots, base logit): this run may
# bring a spec whose categories and binaries are the source's but whose prior / base values are new (R5: re-calibrated labels), so after
# the strict load they are re-set from THIS run's spec -- the loaded values would otherwise silently replace the new spec's -- while every
# learned parameter (state_gate, state_w, state_b, bin_w, bin_b included) keeps the loaded value.  Nothing else in Net comes from the
# spec; the pair-family weights and the extent targets are rebuilt from this run's spec in main() and are not part of the model.
WARM_CONFIG_BUFFERS = ("mean", "std", "ext_mask")
WARM_SPEC_BUFFERS = ("state_prior", "state_valid", "state_base_logit")


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def warm_load(model, path):
    """v9.6 --warm-start: a source run's fold model (the plain tensor dict that Net.deploy_state_dict wrote -- the whole state dict for
    every timm backbone) loaded into the freshly built `model` with strict=True: the encoder, every head, every parameter and buffer; a
    missing, unexpected or reshaped key refuses (never a backbone-only load).  Before loading, each CONFIG buffer of the fresh model must
    equal the file's bit for bit and each SPEC buffer must have the file's dtype and shape (WARM_CONFIG_BUFFERS / WARM_SPEC_BUFFERS); after
    loading, the SPEC buffers are put back to the fresh model's values = this run's state spec.  With the source's own spec that write-back
    restores the very values just loaded (bit for bit a plain strict load).  Returns (tensors loaded, md5 of the file, {spec buffer: per-class
    change vs the loaded value -- max |d| over the slots for state_prior, the number of changed slots for state_valid, |d| for
    state_base_logit}).
    """
    import io
    if not os.path.isfile(path):
        raise SystemExit("--warm-start: fold model %s does not exist" % path)
    with open(path, "rb") as fh:                         # one read: the md5 is that of the very bytes loaded
        raw = fh.read()
    md5 = hashlib.md5(raw).hexdigest()
    sd = torch.load(io.BytesIO(raw), map_location="cpu")
    del raw
    own = model.state_dict()
    for name in WARM_CONFIG_BUFFERS + WARM_SPEC_BUFFERS:
        if (name in own) != (name in sd):
            raise SystemExit("--warm-start: buffer %s is %s this run's Net but %s %s" % (
                name, "in" if name in own else "not in", "in" if name in sd else "not in", path))
        if name in own:
            b_own, b_src = own[name].detach().cpu(), sd[name]
            if b_own.dtype != b_src.dtype or b_own.shape != b_src.shape:
                raise SystemExit("--warm-start: %s holds a %s of dtype / shape %s %s; this run's Net builds %s %s (another state spec "
                                 "category layout, extent classes or slices)" % (path, name, b_src.dtype, tuple(b_src.shape), b_own.dtype,
                                                                                 tuple(b_own.shape)))
            if name in WARM_CONFIG_BUFFERS and not torch.equal(b_own, b_src):
                raise SystemExit("--warm-start: %s holds another %s than this run's Net builds (other extent classes or slices)" % (path, name))
            # with identical categories (asserted at start-up) the valid slots cannot differ unless a spec's prior lists do not match its
            # categories or the source's spec file was edited after the source run: refused (a valid target slot masked to -inf
            # would give an infinite state cross-entropy)
            if name == "state_valid" and not torch.equal(b_own, b_src):
                raise SystemExit("--warm-start: %s holds other state_valid slots than this run's spec builds (a spec whose prior lists do "
                                 "not match its categories, or a source spec file edited after the source run)" % path)
    spec_now = {n_: own[n_].detach().clone() for n_ in WARM_SPEC_BUFFERS if n_ in own}   # this run's spec, before the load overwrites it
    aux_keys = {k_ for k_ in own if k_.startswith("aux_")}   # v9.8: the --fm-aux head (no fold model carries it: deploy_state_dict strips it)
    if not aux_keys:                                       # v9.7's strict load, unchanged
        try:
            model.load_state_dict(sd, strict=True)
        except RuntimeError as e:
            raise SystemExit("--warm-start: %s does not match this run's Net (strict load): %s" % (path, str(e)[:3000]))
    else:                                                  # v9.8: strict except that the missing keys must be EXACTLY the aux_* set
        try:
            missing, unexpected = model.load_state_dict(sd, strict=False)
        except RuntimeError as e:                          # a reshaped key still raises with strict=False
            raise SystemExit("--warm-start: %s does not match this run's Net: %s" % (path, str(e)[:3000]))
        if set(missing) != aux_keys or unexpected:
            raise SystemExit("--warm-start: %s does not match this run's Net: the missing keys must be exactly the auxiliary head %s; "
                             "missing %s, unexpected %s" % (path, sorted(aux_keys), sorted(missing)[:20], sorted(unexpected)[:20]))
    change = {}
    if spec_now:
        bufs = dict(model.named_buffers())
        with torch.no_grad():
            for n_, v_ in spec_now.items():
                old = bufs[n_].detach().clone()                  # the source run's value, as loaded
                bufs[n_].copy_(v_)
                if v_.dtype == torch.bool:
                    change[n_] = (v_ != old).sum(-1).tolist()
                else:
                    d_ = (v_.float() - old.float()).abs()
                    change[n_] = (d_.amax(-1) if d_.ndim > 1 else d_).tolist()
    return len(sd), md5, change


def file_ids(d):
    """{(device, inode): path} of the files directly in d, d/models and d/ckpt (a hard link shares its inode)"""
    out = {}
    for sub in ("", "models", "ckpt"):
        dd = os.path.join(d, sub)
        if os.path.isdir(dd):
            for f in sorted(os.listdir(dd)):
                p = os.path.join(dd, f)
                if os.path.isfile(p):
                    st = os.stat(p)
                    out[(st.st_dev, st.st_ino)] = p
    return out


# ---- v9.7: --save-state-oof -- the state head's read-out next to the class OOF (spec card spec_trainer_v97_state_oof_20261010.md) ---------
def state_layout(spec):
    """The column layout of the state read-out, fixed by the state spec: per class (LABELS order) a block of K_c columns = its categories in
    the spec's order (the state head's slots 0 .. K_c-1), then one column per binary head in Net's order (the spec's binaries dict order, each
    binary's class list order = Net.bin_names).  Refuses a spec whose prior list length differs from its category count (Net takes the
    valid slots from the prior lists and K_c from the categories: the slots would not be the categories)."""
    bad = [l for l in LABELS if len(spec["prior"][l]) != len(spec["categories"][l])]
    if bad:
        raise SystemExit("--save-state-oof: the state spec's prior list length differs from its category count for %s (the state head's "
                         "valid slots would not be the categories)" % bad)
    cols, j = [], 0
    for l in LABELS:
        k_ = len(spec["categories"][l])
        cols.append({"class": l, "categories": list(spec["categories"][l]), "start": j, "stop": j + k_})
        j += k_
    for b, labs in spec["binaries"].items():
        for l in labs:
            cols.append({"binary": b, "class": l, "column": j})
            j += 1
    return {"columns": cols, "width": j}


def state_slots_check(model, layout):
    """v9.7: the model's state head must have exactly the layout's slots -- per class valid slots 0 .. K_c-1 and no other, K_c = the
    layout's block width -- and the layout's binaries (name, class) in the same order; refused otherwise (never a silent column shift)."""
    blocks = [c_ for c_ in layout["columns"] if "categories" in c_]
    valid = model.state_valid.detach().cpu()
    for c, b_ in enumerate(blocks):
        kc = b_["stop"] - b_["start"]
        if b_["class"] != LABELS[c] or model.state_k[c] != kc or not bool(valid[c, :kc].all()) or bool(valid[c, kc:].any()):
            raise SystemExit("--save-state-oof: the state head's slots for %s (K %d, valid %s) are not the layout's %d categories" % (
                LABELS[c], model.state_k[c], valid[c].tolist(), kc))
    if [(c_["binary"], c_["class"]) for c_ in layout["columns"] if "binary" in c_] != [tuple(x) for x in model.bin_names]:
        raise SystemExit("--save-state-oof: the state head's binaries %s are not the layout's" % (model.bin_names,))


def state_probs(model, sl, bl):
    """v9.7: the state read-out of one evaluation batch, [B, D] float32.  sl [B, C, Kmax] / bl [B, nbin] are the state / binary logits that
    Net.head computed in the forward that gave the batch's class logits (return_aux).  Per class: p = sl.softmax(-1) -- exactly the p of
    Net.state_evidence (the -inf invalid slots get 0) -- restricted to its K_c valid slots 0 .. K_c-1; then sigmoid of every binary logit."""
    p = sl.softmax(-1)
    return torch.cat([p[:, c, :kc] for c, kc in enumerate(model.state_k)] + [torch.sigmoid(bl)], 1).float()


def state_files(out_dir, pre, arm, seed):
    """v9.7: the three read-out files of one (arm, seed): pool array, its layout sidecar, gold json."""
    return [os.path.join(out_dir, pre + n_ % (arm, seed)) for n_ in ("oof_state_%s_s%d.npy", "oof_state_%s_s%d.json", "preds_state_%s_s%d.json")]


def write_state_readout(out_dir, pre, arm, seed, layout, spec_path, pool_n, st_pool, gold, gold_fold, st_gold, K, H, only, reeval_of):
    """v9.7: writes the three files of state_files().  st_pool: float32 [pool, D] in pool order (NaN = not predicted here; holdout-fold rows
    = sum over this file's fold models of p / (K - 1)); st_gold: {fold k: float32 [gold, D]} = fold model k's read-out of every gold
    study.  The gold mean (arithmetic, float64, over the fold models in fold order) is written only when the file holds every fold model
    of the run (merge_partial_v2.py writes it for a merged run)."""
    p_npy, p_side, p_gold = state_files(out_dir, pre, arm, seed)
    folds_here = sorted(st_gold)
    expected = [k_ for k_ in range(K) if k_ != H]
    blocks = [c_ for c_ in layout["columns"] if "categories" in c_]
    bcols = [c_["column"] for c_ in layout["columns"] if "binary" in c_]
    common = {"arm": arm, "seed": seed, "labels": LABELS, "state_spec": spec_path, "state_spec_md5": file_md5(spec_path), "folds": K,
              "holdout_fold": H, "only_folds": sorted(only) if only else None, "fold_models": folds_here, "reeval_of": reeval_of,
              "values": "per class: softmax of the state head's logits over the class's valid slots (the p of Net.state_evidence); per binary "
                        "head: sigmoid of its logit; float32, from the bf16-autocast evaluation forward that gives the class logits"}

    def lists(arr):                             # [n, D] -> per class [n][K_c] lists + the binaries [n][nbin] (float32 values, unrounded)
        return {"state": {b_["class"]: arr[:, b_["start"]:b_["stop"]].tolist() for b_ in blocks}, "binary": arr[:, bcols].tolist()}
    np.save(p_npy, st_pool.astype(np.float32))
    side = dict(common, kind="v9.7 state-head read-out: pool, out-of-fold", array=os.path.basename(p_npy), shape=list(st_pool.shape),
                dtype="float32", rows="pool order = pool_uids.json (%d studies)" % pool_n, columns=layout["columns"], width=layout["width"],
                nan_rows="a pool study this process did not predict (another fold's study under --only-folds) is a NaN row",
                holdout_rows=None if H is None else "holdout fold %d: sum over this file's fold models of p / (K - 1) (the class OOF's "
                                                    "convention: the mean over the K - 1 fold models once every fold model is in)" % H)
    json.dump(side, open(p_side, "w"), indent=1)
    g = dict(common, kind="v9.7 state-head read-out: gold", gold_uids=gold, gold_fold=gold_fold,
             categories={b_["class"]: b_["categories"] for b_ in blocks},
             binaries=[[c_["binary"], c_["class"]] for c_ in layout["columns"] if "binary" in c_],
             per_fold_model={str(k_): lists(st_gold[k_]) for k_ in folds_here},
             mean=lists(np.stack([st_gold[k_] for k_ in folds_here]).astype(np.float64).mean(0)) if folds_here == expected else None,
             mean_note="arithmetic mean over the fold models (fold order) of each model's probability vectors; null while the file lacks "
                       "a fold model of the run (a partial file)")
    json.dump(g, open(p_gold, "w"))


# ---- v9.8: --fm-aux -- the fastMRI+ lesion-box auxiliary task on the shared encoder (spec card spec_trainer_v98_fmaux_20261011.md B) ------
FM_CLASSES = ("meniscus_tear", "cart_partial", "cart_full", "bone_edema", "bone_fx_contusion", "effusion", "cyst", "acl_high", "acl_low",
              "collateral", "other")       # A4: auxiliary class ids 0 .. 10 (never merged)
FM_A = len(FM_CLASSES)
FM_VIEWS = (0, 2, 3)            # px_cache v4's coronal views: 0 = the whole field, 2 / 3 = the 100 mm locals (medial / lateral)
FM_SIZE = 384                   # the store's rendering size (as px384_v4)
FM_CELL = 32                    # the encoder's stride: G = crop / 32 cells per side (11 at 352, 12 at 384)
FM_VOLS_PER_STEP = 4            # B2: 4 volumes x M / 4 windows per optimiser step
FM_BOXED_FRAC = 0.75            # B2: share of the volume draws from volumes with boxes (the rest: reviewed volumes without findings)
FM_POS_WEIGHT = 8.0             # B1: pos_weight of the dense BCE (pre-registered, fixed)
FM_PRESENCE_MIN = 0.25          # B1: a box makes its class present when >= 25% of its original area is inside the window
FM_BIAS0 = -4.0                 # aux head bias at init: sigmoid(-4) = 0.018 for sparse targets (ext_b's choice)
FM_GATE = (("meniscus_tear", 0.75), ("effusion", 0.80), ("cart_full", 0.75))   # B5: the "the head learned" diagnostic gate
FM_DONE_RE = re.compile(r"^(partial_)?preds_(?!state_).+_s\d+\.json$")      # E2: a run's done marker (not the v9.7 state read-out's gold json)


class FmStore:
    """v9.8: the fm384_v1 store (spec A4), read and checked at start-up; any inconsistency is refused.  Files in the directory:
      px.u8.npy     uint8 [N, 384, 384] (np.save; read with mmap)
      index.json    {"format": "fm384_v1", "size": 384, "classes": FM_CLASSES (id order), "n_rows": N,
                     "volumes": {<volume>: {"n_slices": n, "views": [[0, s0], [2, s2], [3, s3]], "patient_id": str, "aux_holdout": bool,
                                            optional "excluded": true (skipped, counted); other keys (acquisition, side score / decision,
                                            anchor) are kept for the log only}}}
                    the rows of (volume, view v) are s_v .. s_v + n - 1 in slice order (slice k = the k-th RSS slice, 3 mm apart)
      boxes.json    {"format": "fm384_v1", "n_rows": N, "boxes": {<row>: [[class id, x0, y0, x1, y1, a0], ...]}} (rows without a box absent)
                    x = column, y = row of the row's 384 view frame (pixel (r, c) covers [c, c+1) x [r, r+1)), clipped to [0, 384];
                    a0 = the box's area in that frame BEFORE the clip (the presence rule's "original box area", >= the clipped area)
      manifest.json any json object (input sha256, render code md5, the three check numbers): logged
    Checks: the format tags, classes, shapes; every usable volume has exactly the views {0, 2, 3} (no view-0-only store: that would be
    the scaffold_view0 degradation), row blocks inside N and disjoint; every box row inside a block, class 0..10, 0 <= x0 < x1 <= 384,
    0 <= y0 < y1 <= 384, a0 >= 0.99 x the clipped area - 1 (rounding slack); one aux_holdout value per patient_id; >= min_volumes usable volumes; >= 1 holdout volume,
    >= 1 training volume with boxes and >= 1 without; every class with a box in the training part."""

    def __init__(self, d, min_volumes):
        self.dir = os.path.abspath(d)
        need = ("px.u8.npy", "index.json", "boxes.json", "manifest.json")
        miss = [f_ for f_ in need if not os.path.isfile(os.path.join(self.dir, f_))]
        if miss:
            raise SystemExit("--fm-aux %s: missing %s" % (self.dir, miss))
        self.md5 = {f_: file_md5(os.path.join(self.dir, f_)) for f_ in ("index.json", "boxes.json", "manifest.json")}
        index = json.load(open(os.path.join(self.dir, "index.json"), encoding="utf-8"))
        boxes = json.load(open(os.path.join(self.dir, "boxes.json"), encoding="utf-8"))
        self.manifest = json.load(open(os.path.join(self.dir, "manifest.json"), encoding="utf-8"))
        px = np.load(os.path.join(self.dir, "px.u8.npy"), mmap_mode="r")
        bad = []
        if index.get("format") != "fm384_v1" or boxes.get("format") != "fm384_v1":
            bad.append("format tags %r / %r (want fm384_v1)" % (index.get("format"), boxes.get("format")))
        if index.get("size") != FM_SIZE:
            bad.append("size %r (want %d)" % (index.get("size"), FM_SIZE))
        if list(index.get("classes") or []) != list(FM_CLASSES):
            bad.append("classes %r (want %r)" % (index.get("classes"), list(FM_CLASSES)))
        if px.dtype != np.uint8 or px.ndim != 3 or tuple(px.shape[1:]) != (FM_SIZE, FM_SIZE):
            bad.append("px.u8.npy %s %s (want uint8 [N, %d, %d])" % (px.dtype, tuple(px.shape), FM_SIZE, FM_SIZE))
        N = int(px.shape[0])
        self.n_rows = N
        if index.get("n_rows") != N or boxes.get("n_rows") != N:
            bad.append("n_rows index %r / boxes %r vs px rows %d" % (index.get("n_rows"), boxes.get("n_rows"), N))
        if bad:
            raise SystemExit("--fm-aux %s refused: %s" % (self.dir, " | ".join(bad)))
        vols_all = index.get("volumes") or {}
        self.excluded = sorted(v for v, r in vols_all.items() if r.get("excluded"))
        self.vols = sorted(v for v, r in vols_all.items() if not r.get("excluded"))
        self.n, self.start, self.pid, self.hold = [], {}, [], []
        spans = []
        for vi, v in enumerate(self.vols):
            r = vols_all[v]
            n_ = r.get("n_slices")
            views = r.get("views")
            if not isinstance(n_, int) or n_ < 1:
                bad.append("%s: n_slices %r" % (v, n_)); continue
            try:
                vd = {int(a_): int(b_) for a_, b_ in views}
            except (TypeError, ValueError):
                bad.append("%s: views %r (want [[0, start], [2, start], [3, start]])" % (v, views)); continue
            if sorted(vd) != sorted(FM_VIEWS) or len(views) != len(FM_VIEWS):
                bad.append("%s: views %s (want exactly %s)" % (v, sorted(vd), list(FM_VIEWS))); continue
            for w_, s_ in vd.items():
                if s_ < 0 or s_ + n_ > N:
                    bad.append("%s view %d: rows %d..%d outside 0..%d" % (v, w_, s_, s_ + n_ - 1, N - 1))
                self.start[(vi, w_)] = s_
                spans.append((s_, s_ + n_, v, w_))
            pid_ = r.get("patient_id")
            if not isinstance(pid_, str) or not pid_:
                bad.append("%s: patient_id %r" % (v, pid_))
            if not isinstance(r.get("aux_holdout"), bool):
                bad.append("%s: aux_holdout %r (want true / false)" % (v, r.get("aux_holdout")))
            self.n.append(int(n_)); self.pid.append(pid_); self.hold.append(bool(r.get("aux_holdout")))
            if len(bad) > 20:
                break
        spans.sort()
        for (a0_, a1_, va, wa), (b0_, b1_, vb, wb) in zip(spans, spans[1:]):
            if b0_ < a1_:
                bad.append("row blocks overlap: %s view %d rows %d..%d and %s view %d rows %d..%d" % (va, wa, a0_, a1_ - 1, vb, wb, b0_, b1_ - 1))
                break
        if bad:
            raise SystemExit("--fm-aux %s refused: %s" % (self.dir, " | ".join(bad[:20])))
        self.n = np.array(self.n, np.int64)
        self.hold = np.array(self.hold, bool)
        row_vol = np.full(N, -1, np.int64)                  # row -> volume index (-1: not in a usable volume's block)
        row_k = np.full(N, -1, np.int64)                    # row -> slice index
        for (vi, w_), s_ in self.start.items():
            row_vol[s_:s_ + self.n[vi]] = vi
            row_k[s_:s_ + self.n[vi]] = np.arange(self.n[vi])
        self.row_boxes = {}
        per_cls = np.zeros((2, FM_A), np.int64)             # boxes (rows x boxes) per class: [training part, holdout]
        boxed_k = [set() for _ in self.vols]
        n_box = n_skip = 0
        for key, lst in (boxes.get("boxes") or {}).items():
            try:
                row = int(key)
            except ValueError:
                bad.append("boxes: row key %r" % key); continue
            if not 0 <= row < N:
                bad.append("boxes: row %d outside 0..%d" % (row, N - 1)); continue
            vi = int(row_vol[row])
            if vi < 0:                                     # a row of an excluded volume (or of no block): never read
                n_skip += 1
                continue
            if isinstance(lst, list) and not lst:          # an empty list = a row without a box
                continue
            arr = np.asarray(lst, np.float64)
            if arr.ndim != 2 or arr.shape[1] != 6:
                bad.append("boxes: row %d entry shape %s (want [n, 6] = class, x0, y0, x1, y1, a0)" % (row, arr.shape)); continue
            c_ = arr[:, 0]
            ok = (c_ == np.round(c_)) & (c_ >= 0) & (c_ < FM_A)
            ok &= (arr[:, 1] >= 0) & (arr[:, 1] < arr[:, 3]) & (arr[:, 3] <= FM_SIZE) & (arr[:, 2] >= 0) & (arr[:, 2] < arr[:, 4]) & (arr[:, 4] <= FM_SIZE)
            ok &= arr[:, 5] >= (arr[:, 3] - arr[:, 1]) * (arr[:, 4] - arr[:, 2]) * 0.99 - 1.0   # a0 >= the clipped area (rounding slack)
            if not ok.all():
                bad.append("boxes: row %d has malformed boxes %s" % (row, arr[~ok][:3].tolist())); continue
            if len(arr):
                self.row_boxes[row] = arr
                boxed_k[vi].add(int(row_k[row]))
                n_box += len(arr)
                np.add.at(per_cls[int(self.hold[vi])], c_.astype(np.int64), 1)
            if len(bad) > 20:
                break
        if bad:
            raise SystemExit("--fm-aux %s refused: %s" % (self.dir, " | ".join(bad[:20])))
        self.boxed_slices = [np.array(sorted(s_), np.int64) for s_ in boxed_k]
        self.free_slices = [np.array(sorted(set(range(int(self.n[vi]))) - s_), np.int64) for vi, s_ in enumerate(boxed_k)]
        self.train_boxed = [vi for vi in range(len(self.vols)) if not self.hold[vi] and len(self.boxed_slices[vi])]
        self.train_neg = [vi for vi in range(len(self.vols)) if not self.hold[vi] and not len(self.boxed_slices[vi])]
        self.holdout = [vi for vi in range(len(self.vols)) if self.hold[vi]]
        pid_hold = collections.defaultdict(set)
        for vi in range(len(self.vols)):
            pid_hold[self.pid[vi]].add(bool(self.hold[vi]))
        split_p = sorted(p_ for p_, s_ in pid_hold.items() if len(s_) > 1)
        if split_p:
            bad.append("patients on both sides of aux_holdout: %s" % split_p[:5])
        if len(self.vols) < min_volumes:
            bad.append("%d usable volumes < %d (--fm-aux-min-volumes; spec B5: >= 1,000)" % (len(self.vols), min_volumes))
        if not self.holdout:
            bad.append("no aux_holdout volume (the B5 diagnostic needs them)")
        if not self.train_boxed or not self.train_neg:
            bad.append("training volumes with boxes %d, without %d (both are drawn)" % (len(self.train_boxed), len(self.train_neg)))
        if (per_cls[0] == 0).any():
            bad.append("classes without a box in the training part: %s" % [FM_CLASSES[c] for c in np.flatnonzero(per_cls[0] == 0)])
        if bad:
            raise SystemExit("--fm-aux %s refused: %s" % (self.dir, " | ".join(bad[:20])))
        self.per_cls, self.n_box, self.n_skip = per_cls, n_box, n_skip
        # the holdout rule as information (the index's flag is authoritative): agreement with sha1(patient_id) % 10 == 0, two hash readings
        h_full = np.array([int(hashlib.sha1(p_.encode()).hexdigest(), 16) % 10 == 0 for p_ in self.pid])
        h_8 = np.array([int(hashlib.sha1(p_.encode()).hexdigest()[:8], 16) % 10 == 0 for p_ in self.pid])
        self.hold_agree = (int((h_full == self.hold).sum()), int((h_8 == self.hold).sum()))

    def summary(self):
        return ("%s: %d usable volumes (%d excluded, skipped), %d patients; training %d with boxes + %d without, holdout %d (%d patients); "
                "%d rows, %d boxes (rows x boxes; %d on excluded rows skipped); per class training / holdout %s; aux_holdout vs "
                "sha1(patient_id) %% 10 == 0: %d / %d agree (full hex), %d / %d ([:8]); md5 %s; manifest keys %s" % (
                    self.dir, len(self.vols), len(self.excluded), len(set(self.pid)), len(self.train_boxed), len(self.train_neg),
                    len(self.holdout), len({self.pid[v] for v in self.holdout}), self.n_rows, self.n_box, self.n_skip,
                    {FM_CLASSES[c]: (int(self.per_cls[0, c]), int(self.per_cls[1, c])) for c in range(FM_A)},
                    self.hold_agree[0], len(self.vols), self.hold_agree[1], len(self.vols), self.md5, sorted(self.manifest)[:12]))

    def window_rows(self, vi, view, k):
        """the 3 rows of the window centred on slice k of (volume vi, view): k-1, k, k+1 clamped to the volume (phys step 1, window_rows)"""
        s_, n_ = self.start[(vi, view)], int(self.n[vi])
        return [s_ + min(n_ - 1, max(0, k + d_)) for d_ in (-1, 0, 1)]

    def boxes_of(self, rows):
        """the boxes [n, 6] of a window's rows (each distinct row once: a clamped border row is not counted twice)"""
        got = [self.row_boxes[r] for r in sorted(set(rows)) if r in self.row_boxes]
        return np.concatenate(got) if got else np.zeros((0, 6))


def fm_window_targets(boxes, crop, oy, ox, dy, dx, shift, flip):
    """v9.8: the targets of one auxiliary window.  boxes [n, 6] = (class, x0, y0, x1, y1, a0) of its 3 rows in the 384 view frame.  The window
    is cropped at (oy, ox) to crop x crop, translated by (dy, dx) with zero fill (out[y, x] = in[y + dy, x + dx], v9.5's shift_batch; shift =
    its pad width N, 0 = no translation) and, with flip, mirrored left-right; every box goes through the same maps in the same order and is
    clipped to the field after each (what leaves the field is gone; the entering border is empty).  Returns dense [A, G, G] float32 = per
    class the covered fraction of every 32 x 32 cell by the union of the class's boxes (per pixel, fractional at box edges; G = crop / 32),
    presence [A] float32 = 1 where some box of the class keeps >= 25% of its original area a0 inside the window, and the transformed
    boxes [m, 6] (class, x0, y0, x1, y1, area kept) of the boxes that remain."""
    G = crop // FM_CELL
    dense = np.zeros((FM_A, G, G), np.float32)
    pres = np.zeros(FM_A, np.float32)
    if len(boxes) == 0:
        return dense, pres, np.zeros((0, 6))
    b = np.asarray(boxes, np.float64)
    x0, x1 = np.clip(b[:, 1] - ox, 0, crop), np.clip(b[:, 3] - ox, 0, crop)
    y0, y1 = np.clip(b[:, 2] - oy, 0, crop), np.clip(b[:, 4] - oy, 0, crop)
    if shift:
        x0, x1 = np.clip(x0 - dx, 0, crop), np.clip(x1 - dx, 0, crop)
        y0, y1 = np.clip(y0 - dy, 0, crop), np.clip(y1 - dy, 0, crop)
    if flip:
        x0, x1 = crop - x1, crop - x0
    area = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    masks = {}
    kept = []
    for i in np.flatnonzero(area > 0):
        c = int(b[i, 0])
        m = masks.get(c)
        if m is None:
            m = masks[c] = np.zeros((crop, crop), np.float32)
        c0, c1 = int(math.floor(x0[i])), int(math.ceil(x1[i]))
        r0, r1 = int(math.floor(y0[i])), int(math.ceil(y1[i]))
        cc, rr = np.arange(c0, c1), np.arange(r0, r1)
        wx = np.minimum(x1[i], cc + 1) - np.maximum(x0[i], cc)
        wy = np.minimum(y1[i], rr + 1) - np.maximum(y0[i], rr)
        np.maximum(m[r0:r1, c0:c1], np.outer(wy, wx).astype(np.float32), out=m[r0:r1, c0:c1])
        if area[i] >= FM_PRESENCE_MIN * b[i, 5]:
            pres[c] = 1.0
        kept.append((c, x0[i], y0[i], x1[i], y1[i], area[i]))
    for c, m in masks.items():
        dense[c] = m.reshape(G, FM_CELL, G, FM_CELL).mean((1, 3))
    return dense, pres, np.array(kept, np.float64).reshape(-1, 6)


def fm_window_image(px, rows, crop, oy, ox, dy, dx, shift, flip):
    """v9.8: the uint8 [3, crop, crop] image of one auxiliary window (channels = slices k-1, k, k+1): crop at (oy, ox), translate (zero fill,
    v9.5's convention), mirror left-right -- the maps fm_window_targets applies to the boxes, in the same order."""
    img = np.stack([np.asarray(px[r]) for r in rows])[:, oy:oy + crop, ox:ox + crop]
    if shift:
        img = np.pad(img, ((0, 0), (shift, shift), (shift, shift)))[:, shift + dy:shift + dy + crop, shift + dx:shift + dx + crop]
    if flip:
        img = img[:, :, ::-1]
    return np.ascontiguousarray(img)


class FmAuxWindows(torch.utils.data.Dataset):
    """v9.8: one auxiliary window per index; the index IS the window's spec (volume, view, slice, oy, ox, dy, dx, flip, gamma, contrast)
    drawn in the main process by FmAuxSampler (or the fixed holdout list), so the workers draw nothing.  Returns (uint8 [3, crop, crop],
    dense [A, G, G], presence [A], gamma, contrast); gamma / contrast are applied on the GPU (fm_photometric)."""

    def __init__(self, store, crop, shift):
        self.st, self.crop, self.shift = store, int(crop), int(shift)
        self.px = None

    def __len__(self):
        return 1

    def __getitem__(self, spec):
        if self.px is None:
            self.px = np.load(os.path.join(self.st.dir, "px.u8.npy"), mmap_mode="r")
        vi, view, k, oy, ox, dy, dx, fl, gam, con = spec
        rows = self.st.window_rows(vi, view, k)
        img = fm_window_image(self.px, rows, self.crop, oy, ox, dy, dx, self.shift, fl)
        dense, pres, _ = fm_window_targets(self.st.boxes_of(rows), self.crop, oy, ox, dy, dx, self.shift, fl)
        return torch.from_numpy(img), torch.from_numpy(dense), torch.from_numpy(pres), float(gam), float(con)


def fm_collate(batch):
    return (torch.stack([b_[0] for b_ in batch]), torch.stack([b_[1] for b_ in batch]), torch.stack([b_[2] for b_ in batch]),
            torch.tensor([b_[3] for b_ in batch], dtype=torch.float32), torch.tensor([b_[4] for b_ in batch], dtype=torch.float32))


def fm_epoch_seeds(seed, fold, epoch):
    """B3: the auxiliary streams of one (seed, fold, epoch): sampling 9800 + 17*seed + fold + 1000*epoch, augmentation the same + 2**32 (two
    generators with one seed would emit the same numbers); fresh every epoch, so a resume at an epoch boundary repeats the batches."""
    s_ = 9800 + 17 * int(seed) + int(fold) + 1000 * int(epoch)
    return s_, s_ + (1 << 32)


def fm_fork_seed(seed, fold, epoch, step):
    """B3: the global-stream seed inside the auxiliary step's fork_rng (drop path of the auxiliary forward), f(seed, fold, epoch, step)"""
    return int(hashlib.sha1(("v9.8 fm-aux %d %d %d %d" % (seed, fold, epoch, step)).encode()).hexdigest()[:15], 16)


class FmAuxSampler:
    """v9.8: one epoch of auxiliary batches (batch_sampler of FmAuxWindows): `steps` lists of M window specs, drawn in the main process
    from two private generators.  Per step: FM_VOLS_PER_STEP slots; per slot (sampling stream) boxed = rand < 0.75, a volume uniform among
    the training volumes with boxes (boxed) or without (otherwise), then M / 4 centre slices -- boxed: the first half uniform over the
    volume's slices with a box (any class), the second half over its slices without one (all slices when it has none) -- and a view
    uniform in {0, 2, 3} per window; then (augmentation stream) per window the crop offset oy, ox uniform on 0..384-crop, the translation
    dy, dx uniform on -N..N (only when N = --shift-px > 0), flip = rand < --fm-aux-flip, gamma U(0.8, 1.25), contrast U(0.9, 1.1)
    (augment()'s ranges).  `digest` = sha1 over every spec yielded (the epoch's sequence; test 7)."""

    def __init__(self, store, steps, M, crop, shift, flip_p, seeds):
        self.st, self.steps, self.M, self.crop, self.shift, self.flip_p = store, int(steps), int(M), int(crop), int(shift), float(flip_p)
        self.gs = torch.Generator().manual_seed(int(seeds[0]))
        self.ga = torch.Generator().manual_seed(int(seeds[1]))
        self.h = hashlib.sha1()
        self.yielded = 0
        assert self.M % (2 * FM_VOLS_PER_STEP) == 0

    def __len__(self):
        return self.steps

    def digest(self):
        return self.h.hexdigest()[:16]

    def draw(self):
        st, gs, ga, M = self.st, self.gs, self.ga, self.M
        per = M // FM_VOLS_PER_STEP
        base = []
        for _ in range(FM_VOLS_PER_STEP):
            boxed = float(torch.rand(1, generator=gs)) < FM_BOXED_FRAC
            pool = st.train_boxed if boxed else st.train_neg
            vi = pool[int(torch.randint(0, len(pool), (1,), generator=gs))]
            if boxed:
                pos, neg = st.boxed_slices[vi], st.free_slices[vi]
                if not len(neg):
                    neg = np.arange(int(st.n[vi]))
                ks = np.concatenate([pos[torch.randint(0, len(pos), (per // 2,), generator=gs).numpy()],
                                     neg[torch.randint(0, len(neg), (per - per // 2,), generator=gs).numpy()]])
            else:
                ks = torch.randint(0, int(st.n[vi]), (per,), generator=gs).numpy()
            vw = torch.randint(0, len(FM_VIEWS), (per,), generator=gs).tolist()
            base += [(int(vi), FM_VIEWS[v_], int(k_)) for v_, k_ in zip(vw, ks.tolist())]
        lim = FM_SIZE - self.crop + 1
        oy = torch.randint(0, lim, (M,), generator=ga).tolist()
        ox = torch.randint(0, lim, (M,), generator=ga).tolist()
        if self.shift > 0:
            dy = torch.randint(-self.shift, self.shift + 1, (M,), generator=ga).tolist()
            dx = torch.randint(-self.shift, self.shift + 1, (M,), generator=ga).tolist()
        else:
            dy = dx = [0] * M
        fl = (torch.rand(M, generator=ga) < self.flip_p).tolist()
        gam = (torch.rand(M, generator=ga) * 0.45 + 0.8).tolist()
        con = (torch.rand(M, generator=ga) * 0.2 + 0.9).tolist()
        return [b_ + (oy[j], ox[j], dy[j], dx[j], bool(fl[j]), gam[j], con[j]) for j, b_ in enumerate(base)]

    def __iter__(self):
        for _ in range(self.steps):
            specs = self.draw()
            self.h.update(repr(specs).encode())
            self.yielded += 1
            yield specs


def fm_photometric(img, gam, con):
    """v9.8: augment()'s gamma / contrast arithmetic, per window: img uint8 [M, 3, S, S] on the device, gam / con [M]"""
    x = (img.float() / 255.).clamp_(0, 1).pow(gam.view(-1, 1, 1, 1))
    x = ((x - 0.5) * con.view(-1, 1, 1, 1) + 0.5).clamp_(0, 1) * 255.
    return x.round_().to(torch.uint8)


def fm_auc(y, s):
    """ROC AUC (Mann-Whitney with average ranks for ties); nan when y has no positive or no negative"""
    y = np.asarray(y, bool)
    s = np.asarray(s, np.float64)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    avg = np.cumsum(cnt) - (cnt - 1) / 2.0
    r = avg[inv]
    return float((r[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def fm_clip(model, max_norm):
    """v9.8: ONE clip of the merged gradient over every parameter.  The norm of v9.7's parameters is computed exactly as v9.7's
    clip_grad_norm_(model.parameters()) computes it (same function, same tensors, same order), the auxiliary head's separately, and they
    are combined as sqrt(n_main^2 + n_aux^2) -- the L2 norm over all parameters -- or n_main itself when the head's gradient is 0 (lambda = 0:
    then the clip is v9.7's bit for bit); every gradient (the head's included) is then scaled as clip_grad_norm_ scales it."""
    from torch.nn.utils.clip_grad import _get_total_norm, _clip_grads_with_norm_
    params = list(model.parameters())
    main_g = [p.grad for n, p in model.named_parameters() if not n.startswith("aux_") and p.grad is not None]
    aux_g = [p.grad for n, p in model.named_parameters() if n.startswith("aux_") and p.grad is not None]
    n_main = _get_total_norm(main_g, 2.0, False, None)
    if aux_g:
        n_aux = _get_total_norm(aux_g, 2.0, False, None).to(n_main.device)
        total = torch.where(n_aux == 0, n_main, torch.sqrt(n_main.square() + n_aux.square()))
    else:
        total = n_main
    _clip_grads_with_norm_(params, max_norm, total, None)
    return total


def dynamo_compiles():
    """v9.8 (test 9): (frames dynamo compiled so far = counters['frames']['ok'], recompiles = distinct compile ids 'frame/k' with k >= 1
    in the compilation metrics, i.e. compiles of a frame that had been compiled before); (-1, -1) when the counters are unavailable"""
    try:
        import torch._dynamo.utils as du
        ok = int(du.counters.get("frames", {}).get("ok", 0))
        ids = {str(x_.compile_id) for x_ in du.get_compilation_metrics()}
        rec = sum(1 for s_ in ids if re.match(r"^\[?\d+/(\d+)", s_) and int(re.match(r"^\[?\d+/(\d+)", s_).group(1)) > 0)
        return ok, rec
    except Exception:
        return -1, -1


def fm_make_loader(store, batches, crop, shift, workers, seed_w):
    """v9.8: a DataLoader over FmAuxWindows whose batch_sampler is `batches` (FmAuxSampler, or a fixed list of spec lists for the holdout
    read-out with shift 0); its own workers and its own generator (never the global stream, never the main loader's generator)"""
    return torch.utils.data.DataLoader(FmAuxWindows(store, crop, shift), batch_sampler=batches, num_workers=workers, collate_fn=fm_collate,
                                       pin_memory=True, prefetch_factor=2 if workers > 0 else None, persistent_workers=False,
                                       generator=torch.Generator().manual_seed(int(seed_w)))


def fm_aux_step(model, batch, img_main, enc_params, lam, crop, seed, k, ep, b_i, steps, acc, pw, log_every, dev):
    """v9.8: one auxiliary step, called after the main loss.backward() of step b_i of epoch ep (fold k): the batch (uint8 windows, dense and
    presence targets, gamma, contrast) -> fm_photometric -> model.enc (train mode, bf16 autocast, the compiled module under --compile) ->
    aux_dense -> L_aux = BCE(dense, pos_weight 8) + BCE(presence) -> (lam * L_aux).backward() into the same .grad.  Everything inside
    torch.random.fork_rng(current CUDA device) seeded with fm_fork_seed(seed, k, ep, b_i): the global CPU / CUDA states afterwards are the
    ones before.  Asserts (test 10) the auxiliary and main batches' spatial size == (crop, crop) and the encoder map / targets' grid == G =
    crop / 32.  Every log_every steps prints |lam grad L_aux| and |grad L_main| over enc_params (the main part = the .grad before this
    backward).  acc [3] += (L_dense, L_presence, 1).  Returns (L_dense, L_presence)."""
    xb, yd, yp, gam, con = batch
    g_step = ep * steps + b_i
    log_now = g_step % log_every == 0
    fs = fm_fork_seed(seed, k, ep, b_i)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.random.default_generator.manual_seed(fs)
        torch.cuda.manual_seed(fs)
        xg = fm_photometric(xb.to(dev, non_blocking=True), gam.to(dev, non_blocking=True), con.to(dev, non_blocking=True))
        if tuple(xg.shape[-2:]) != tuple(img_main.shape[-2:]) or tuple(xg.shape[-2:]) != (crop, crop):   # test 10
            raise AssertionError("v9.8: auxiliary batch %s vs main batch %s vs --crop %d: spatial sizes differ" % (
                tuple(xg.shape), tuple(img_main.shape), crop))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            xs = xg.float().div_(255.)
            xs = ((xs - model.mean) / model.std).contiguous(memory_format=torch.channels_last)
            fmap = model.enc(xs)
        G = crop // FM_CELL
        if fmap.ndim != 4 or tuple(fmap.shape[-2:]) != (G, G) or tuple(yd.shape[-2:]) != (G, G):   # test 10
            raise AssertionError("v9.8: encoder map %s / dense target %s vs G = crop / 32 = %d" % (tuple(fmap.shape), tuple(yd.shape), G))
        dz = model.aux_dense(fmap)                                                   # [M, A, G, G]
        l_d = F.binary_cross_entropy_with_logits(dz, yd.to(dev, non_blocking=True), pos_weight=pw)
        pz = torch.logsumexp(dz.flatten(2), -1) - math.log(G * G)                   # presence: log-mean-exp over the G^2 positions
        l_p = F.binary_cross_entropy_with_logits(pz, yp.to(dev, non_blocking=True))
        snap = [torch.zeros_like(p_) if p_.grad is None else p_.grad.detach().clone() for p_ in enc_params] if log_now else None
        (lam * (l_d + l_p)).backward()
    acc += torch.stack([l_d.detach().float(), l_p.detach().float(), torch.ones((), device=dev)])
    if log_now:
        with torch.no_grad():
            n_m = torch.sqrt(sum(s_.float().square().sum() for s_ in snap))
            n_x = torch.sqrt(sum(((p_.grad.float() if p_.grad is not None else torch.zeros_like(s_).float()) - s_.float()).square().sum()
                                 for p_, s_ in zip(enc_params, snap)))
        print("  fm-aux fold %d epoch %d step %d (fold step %d): |lambda grad L_aux| / |grad L_main| over the encoder %.4g (%.4g / %.4g); "
              "L_dense %.4f L_presence %.4f" % (k, ep, b_i, g_step, float(n_x) / max(float(n_m), 1e-30), float(n_x), float(n_m),
                                                 float(l_d.detach()), float(l_p.detach())), flush=True)
    return l_d.detach(), l_p.detach()


@torch.no_grad()
def fm_holdout_eval(model, store, hold_specs, crop, M, workers, dev):
    """v9.8: per-class presence AUC on the store's aux_holdout volumes (the B5 diagnostic; never used to choose anything): every slice of
    every holdout volume in the views {0, 2, 3}, centre crop (center()'s offset), no translation / flip / photometric jitter, batches of M
    windows (the last padded with zeros) through model.enc in eval mode; window level = sigmoid(presence logit) vs the window's presence
    target (the 25% rule), volume level = the max over the volume's windows vs any window present.  Inside fork_rng; the module's train /
    eval mode is restored."""
    was = model.training
    o_ = (FM_SIZE - crop) // 2
    specs = [(vi, w_, k_, o_, o_, 0, 0, False, 1.0, 1.0) for vi, w_, k_ in hold_specs]
    G = crop // FM_CELL
    P, Y = [], []
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        model.eval()
        for xb, yd, yp, _, _ in fm_make_loader(store, [specs[i:i + M] for i in range(0, len(specs), M)], crop, 0, workers, 0):
            n_ = len(xb)
            if n_ < M:
                xb = torch.cat([xb, xb.new_zeros((M - n_,) + tuple(xb.shape[1:]))])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                xs = xb.to(dev, non_blocking=True).float().div_(255.)
                xs = ((xs - model.mean) / model.std).contiguous(memory_format=torch.channels_last)
                fmap = model.enc(xs)
            dz = model.aux_dense(fmap)[:n_]
            P.append(torch.sigmoid(torch.logsumexp(dz.flatten(2), -1) - math.log(G * G)).float().cpu())
            Y.append(yp)
    model.train(was)
    P, Y = torch.cat(P).numpy(), torch.cat(Y).numpy() > 0.5
    win = [fm_auc(Y[:, c], P[:, c]) for c in range(FM_A)]
    vid = np.array([s_[0] for s_ in specs])
    vols = sorted(set(vid.tolist()))
    Pv = np.stack([P[vid == v_].max(0) for v_ in vols])
    Yv = np.stack([Y[vid == v_].any(0) for v_ in vols])
    vol = [fm_auc(Yv[:, c], Pv[:, c]) for c in range(FM_A)]
    return {"window_auc": dict(zip(FM_CLASSES, win)), "volume_auc": dict(zip(FM_CLASSES, vol)), "windows": len(specs), "volumes": len(vols),
            "window_pos": dict(zip(FM_CLASSES, Y.sum(0).astype(int).tolist())), "volume_pos": dict(zip(FM_CLASSES, Yv.sum(0).astype(int).tolist()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--px", required=True)
    ap.add_argument("--px-extra", default=None, help="add-on pixel cache (px_cache --views v2add) read next to --px; its windows join every study")
    ap.add_argument("--px-override", default=None, help="override cache (sidefix): studies listed there are read from it instead of --px")
    ap.add_argument("--targets", required=True)
    ap.add_argument("--extra", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="E_prior")
    ap.add_argument("--backbone", default="resnet34")
    ap.add_argument("--init", default="F:/rsnaknee/pretrained/resnet34-imagenet.pth")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--train-windows", type=int, default=48)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--slices", type=int, default=3, help="adjacent slices per window = input channels (odd; CNN backbones only beyond 3)")
    ap.add_argument("--slice-jitter", type=float, default=0.0, help="probability that a training window uses neighbours two slices apart")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of the trainable weights (0 = off); EMA weights are evaluated and saved")
    ap.add_argument("--bn-recal-batches", type=int, default=0, help="v7: BatchNorm backbones only (CoAtNet, ResNet): after --ema copies the "
                    "averaged weights in, recompute the BatchNorm running statistics for them over this many training batches "
                    "(centre crop, no augmentation) before evaluating and saving; also applied by --reeval-run.  0 = off "
                    "(backbones without BatchNorm are never touched)")
    ap.add_argument("--include-review-holdout", action="store_true", help="add the 120 review-holdout studies (full report targets, "
                    "fold -3) to the training pool; they were held out for the report-reader evaluation, not for image training")
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing in the timm encoder (keeps 208 windows/study on 32 GB cards)")
    ap.add_argument("--syn-decomp", action="store_true", help="v9.2 (10-10): synovitis logit = specific part + beta * stop-grad(effusion logit); "
                    "beta a scalar parameter (init 0); writes oof_synspec_<arm>_s<seed>.npy next to the OOF.  Off = v9.1 bit for bit")
    ap.add_argument("--grad-ckpt-blocks", type=int, default=0, help="v9.1 (10-09): recompute only the first N encoder blocks in backward (ViT "
                    "encoders with .blocks): memory between --grad-ckpt (all blocks) and none, about N/12 of its extra compute; same maths")
    ap.add_argument("--phys-mm", type=float, default=0.0, help="window spacing in mm of anatomy: the covering stride and the K-slice "
                    "neighbour step are multiplied by round(phys_mm / slice spacing), so a 0.4 mm 3-D stack is sampled like a 3.5 mm "
                    "2-D series (0 = per-slice, the old behaviour)")
    ap.add_argument("--rare-boost", action="store_true", help="oversample studies with rare report states (MCL>=2/acute, fracture, high PF OA, Baker>=moderate, contusion); needs --state-targets")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--freeze-blocks", type=int, default=16, help="ofl_l backbone: number of leading transformer blocks kept frozen (of 24)")
    ap.add_argument("--llrd", type=float, default=1.0, help="ViT backbones (ofl_l, timm vit_*): layer-wise learning-rate decay per block from the top (1 = off); block i of n gets lr * llrd**(n - 1 - i), embeddings lr * llrd**n")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval-workers", type=int, default=3)
    ap.add_argument("--forward-only", action="store_true", help="one batch forward/backward, NO optimizer step, then exit")
    ap.add_argument("--compile", action="store_true", help="torch.compile the encoder in place (dynamic shapes); same maths, 1.4x (ConvNeXt-small) / 1.9x (DINOv3-B) on a PRO 6000; weight names unchanged")
    ap.add_argument("--resume", action="store_true", help="continue every fold from its end-of-epoch checkpoint in <out>/ckpt when present")
    ap.add_argument("--ckpt-every", type=int, default=1, help="write the fold checkpoint every N epochs (0 = never); removed when the fold ends")
    ap.add_argument("--stop-after-epoch", type=int, default=-1, help="testing: exit right after the checkpoint of this epoch")
    ap.add_argument("--eval-stride", type=int, default=3, help="evaluation windows: every k-th slice as centre (3 = the validated default; 1 = every slice)")
    ap.add_argument("--train-sampling", default="uniform", choices=["uniform", "cover"],
                    help="uniform = --train-windows random windows of the study; cover = every series/view at --cover-stride with a random phase (capped at --train-windows)")
    ap.add_argument("--cover-stride", type=int, default=3)
    ap.add_argument("--state-targets", default=None, help="state_targets_<tag>.csv from make_state_targets.py (report-state construct)")
    ap.add_argument("--state-spec", default=None, help="state_spec_<tag>.json (categories, prior vectors, binaries)")
    ap.add_argument("--state-weight", type=float, default=0.5, help="weight of the state/binary losses next to the arm loss")
    ap.add_argument("--spatial-attn", action="store_true", help="per-class spatial attention inside each window (encoder keeps its "
                    "grid; class queries start at zero = plain average), then per-class window attention over class-specific tokens")
    ap.add_argument("--pair-rank", type=float, default=0.0, help="weight of the matched-pair ranking loss on the 8 pair families "
                    "(positive vs hard negative of a class with the same co-finding signature); needs --state-targets; 0 = off")
    ap.add_argument("--pair-frac", type=float, default=0.33, help="share of training batches that open with a matched pair (--pair-rank)")
    ap.add_argument("--pair-extra", action="store_true", help="v9: add the within-class grade families Effusion, Baker's, Contusion, "
                    "Fracture (higher grade over lower grade of the same structure) to --pair-rank")
    ap.add_argument("--oa-area-rank", action="store_true", help="v9.3 R4d: per OA class a family that ranks high_grade_large above "
                    "high_grade_small on the extent head's log1p(area/100) (needs --extent and a spec with high_grade_small = opusv11c), "
                    "and the extent-head soft target " + str(OA_SMALL_TARGET) + " for high_grade_small cells")
    ap.add_argument("--oa-sign-decoy", action="store_true", help="v9.3 R4d: per OA class a family that ranks high grade with NO bone sign "
                    "written above low_grade_chondral WITH a sign written (osteophytes / subchondral marrow / bone-on-bone); signature 0")
    ap.add_argument("--acl-acuity-decoy", action="store_true", help="v9.3 R4d: a family that ranks chronic ACL tears (and age-unstated tears "
                    "with no contusion and effusion <= small) above ACL normal / signal / low-grade WITH a contusion or effusion >= moderate; "
                    "needs --lig-age")
    ap.add_argument("--mcl-boundary", action="store_true", help="v9.3 R4d: a family that ranks MCL high_grade_partial_tear / complete_tear "
                    "(not chronic / graft / no_tear per --lig-age) above partial_tear_or_grade2; needs --lig-age")
    ap.add_argument("--lig-age", default=None, help="v9.3: csv with columns study, class, age (chronic / acute / age_unstated / graft / "
                    "no_tear) from the E8 ligament re-read (e8/lig_age_A.csv) or the adjudicated r4d_inputs/ligament_age.csv; a cell absent "
                    "from the csv = age_unstated")
    ap.add_argument("--save-area", action="store_true", help="v9.3 read-out (output only, no effect on training): write oof_area_<arm>_s<seed>.npy "
                    "([pool, 12] extent-head area proxy, OA columns meaningful; needs --extent) next to the OOF, also under --reeval-run")
    ap.add_argument("--save-state-oof", action="store_true", help="v9.7 read-out (output only, no effect on training or on the class "
                    "outputs): the state head's per-class category probabilities (softmax over the class's valid slots, as "
                    "Net.state_evidence uses them) and binary probabilities for every held-out and gold study a fold model predicts -> "
                    "oof_state_<arm>_s<seed>.npy + .json (layout) and preds_state_<arm>_s<seed>.json (gold), also under --reeval-run; "
                    "needs --state-targets.  Off (default) = v9.6 bit for bit")
    ap.add_argument("--seed-base", type=int, default=0, help="first seed (seeds seed_base .. seed_base+seeds-1); a replicate of a recipe")
    ap.add_argument("--extent", action="store_true", help="M4: OA defect extent per series and plane (area proxy) into the OA logits")
    ap.add_argument("--extent-weight", type=float, default=0.2, help="weight of the extent supervision (>= 1 cm^2 stated vs normal/low grade)")
    ap.add_argument("--extent-classes", default="Medial OA,Lateral OA,PF OA", help="v9.4: the OA classes whose logit receives the extent "
                    "head's area term (M4 ablation: 'Medial OA,Lateral OA' leaves PF OA's logit without it; the area, its supervision and "
                    "the area families are unchanged).  Default = all three = v9.3")
    ap.add_argument("--mirror", type=float, default=0.0, help="v9.4 (spec_trainer_v94_mirror_20261010.md): probability that a training "
                    "batch is mirrored into the virtual contralateral knee -- coronal / axial windows flipped left-right, sagittal windows "
                    "reversed along the slice axis (channels reversed, rel_pos negated), coronal view ids 2 <-> 3 -- with the model outputs "
                    "re-indexed Medial <-> Lateral (menisci, OA) before every loss and MCL not trained on that batch.  One draw per batch; "
                    "training forward only; 0 = off (v9.3 bit for bit)")
    ap.add_argument("--shift-px", type=int, default=0, help="v9.5 (spec_trainer_v95_shift_20261010.md): translation jitter of N pixels -- once "
                    "per training batch, after the crop and before the --aug v2 affine, every window of the batch is translated by the "
                    "same integer (dy, dx), each uniform on -N..N, with zero fill (pad N zeros, crop back at N+dy, N+dx); drawn from its "
                    "own stream (9500 + 17*seed + fold), so every other draw is unchanged.  Gives --crop 384 the +-16 px translation that "
                    "--crop 352 takes from its random crop offset.  Training forward only; 0 = off (v9.4 bit for bit)")
    ap.add_argument("--warm-start", default=None, help="v9.6 (spec_r5_repair_training_20261010.md): repair training -- fold k starts from "
                    "DIR/models/<arm>_s<seed>_f<k>.pt (the whole Net state dict, strict=True, loaded before the optimizer and the EMA are "
                    "built, so the EMA starts from it); the optimizer and the LR schedule start fresh over --epochs.  DIR's run_config must "
                    "equal this run in --folds, --holdout-fold, backbone, crop, slices, spatial attention, extent (+ classes) and syn-decomp, "
                    "its state spec must have this run's categories and binaries (prior / base values may differ: the state head's spec "
                    "buffers are re-set from this run's spec after the load), and pool_uids.json the pool (refused at start-up otherwise).  "
                    "Off (default) = v9.5 bit for bit")
    ap.add_argument("--warm-lr-scale", type=float, default=1.0, help="v9.6: with --warm-start, every parameter group's learning rate "
                    "(encoder, layer-wise groups, head) times s (suggested 0.3); the schedule's shape is unchanged (warm-up + cosine over "
                    "--epochs).  Needs --warm-start")
    ap.add_argument("--holdout-fold", type=int, default=None,
                    help="clean ruler fold for pseudo-label rounds: this hash fold is excluded from EVERY training set (no model is trained "
                         "for it); its predictions are the mean of the other folds' models, so a teacher->student chain evaluated on it "
                         "with local_ruler --folds-only is leak-free")
    ap.add_argument("--reeval-run", default=None, help="predict-only: load models/<arm>_s<seed>_f<k>.pt from this run directory and rewrite the out-of-fold "
                    "predictions into --out with the current --eval-stride (no training)")
    ap.add_argument("--aug", default="crop", choices=["crop", "v2"], help="v2 = crop + per-study rotation/scale + gamma/contrast jitter (no flips: laterality)")
    ap.add_argument("--aug-grid-fp32", action="store_true", help="v9.6: --aug v2 only -- the rotation / scale sampling grid (affine_grid) "
                    "and its bilinear resampling (grid_sample) in float32 with autocast off for those two calls (inside the bf16 autocast "
                    "region affine_grid's matmul runs in bf16: ~0.7 px grid error); same random draws, same order.  Off (default) = v9.5 bit for bit")
    ap.add_argument("--draw-seed", action="store_true", help="v9.6 (SCAN_A D1): every TRAINING draw seeds its window choice (cover phase, "
                    "slice jitter, window subset) with [seed, epoch, index, nonce], nonce = the draw's position in the epoch, so a study drawn "
                    "k times in one epoch (the rare-boost / pair samplers draw with replacement) gets k window sets; the samplers' drawn "
                    "index sequence is unchanged; evaluation and BN recalibration keep v9.5's seeding.  Off (default) = v9.5 bit for bit")
    ap.add_argument("--rare-boost-names", action="store_true", help="v9.6 (SCAN_A D4): the --rare-boost terms by state NAME: MCL +2 only for "
                    "partial_tear_or_grade2 / high_grade_partial_tear / complete_tear (or the acute binary) -- v9.5's index rule (>= 2) also "
                    "boosted chronic_or_healed_change; the Fracture / PF OA / Baker's / Contusion terms by the names of v9.5's indices "
                    "(asserted at start-up to select the same studies).  Needs --rare-boost.  Off (default) = v9.5 bit for bit")
    ap.add_argument("--aug-res", type=float, default=0.0, help="v2 only: probability that a training study is down-sampled in-plane by "
                    "1.25-2x and up-sampled back (simulates 0.33-0.5 mm scanners); 0 = off, random stream unchanged")
    ap.add_argument("--only-folds", default=None, help="diagnostic: comma list of folds to train (others skipped); outputs are written with a partial_ prefix and must not be used as OOF")
    ap.add_argument("--check-targets", action="store_true", help="pre-launch check on CPU: build the pool and every arm's targets and the "
                    "state targets with the training code path, print a summary, and exit (no data loading, no model, no GPU)")
    ap.add_argument("--fm-aux", default=None, help="v9.8 (spec_trainer_v98_fmaux_20261011.md B): the fm384_v1 store DIR of the fastMRI+ "
                    "lesion-box auxiliary task -- every step M windows through the same encoder into the aux_* head (dense coverage BCE "
                    "pos_weight 8 + presence BCE), lambda * L_aux back-propagated into the same gradients before the one clip / AdamW / "
                    "EMA step; aux windows cropped to --crop (+ --shift-px), G = crop / 32.  ConvNeXt with --spatial-attn only.  Off "
                    "(default) = v9.7 bit for bit")
    ap.add_argument("--fm-aux-weight", type=float, default=0.3, help="v9.8: lambda of the auxiliary loss (pre-registered 0.3; 0 = the "
                    "isolation test: the auxiliary step runs, the main path is v9.7's bit for bit)")
    ap.add_argument("--fm-aux-windows", type=int, default=96, help="v9.8: M auxiliary windows per step (4 volumes x M/4; a multiple of 8; "
                    ">= 0.25 x --batch x --train-windows)")
    ap.add_argument("--fm-aux-flip", type=float, default=0.5, help="v9.8: probability that an auxiliary window (and its boxes) is mirrored "
                    "left-right")
    ap.add_argument("--fm-aux-workers", type=int, default=2, help="v9.8: loader workers of the auxiliary batches (their own, not the "
                    "main loader's)")
    ap.add_argument("--fm-aux-log-every", type=int, default=100, help="v9.8: every N steps log |lambda grad L_aux| / |grad L_main| over the "
                    "encoder parameters (before the clip)")
    ap.add_argument("--fm-aux-min-volumes", type=int, default=1000, help="v9.8: the store must hold at least this many usable volumes "
                    "(spec B5: >= 1,000; lower only for tests on a synthetic store)")
    ap.add_argument("--i-have-user-authorisation-to-finetune", action="store_true")
    a = ap.parse_args()
    if not a.forward_only and not a.check_targets and not a.i_have_user_authorisation_to_finetune:
        raise SystemExit("refusing to update an encoder: the user has only authorised light heads on cached features. "
                         "Use --forward-only, or pass the authorisation flag after the user says so.")
    if not 0.0 <= a.mirror <= 1.0:                       # v9.4
        raise SystemExit("--mirror is a probability: 0 <= p <= 1")
    if not 0 <= a.shift_px < a.crop:                     # v9.5
        raise SystemExit("--shift-px is a translation in pixels: 0 <= N < --crop")
    if a.aug_grid_fp32 and a.aug != "v2":                # v9.6: the flag changes the v2 affine only; never silently ignored
        raise SystemExit("--aug-grid-fp32 changes the --aug v2 affine grid: needs --aug v2")
    if a.rare_boost_names and not a.rare_boost:          # v9.6: the flag changes --rare-boost's weights only
        raise SystemExit("--rare-boost-names changes the --rare-boost weights: needs --rare-boost")
    if a.warm_start is None and a.warm_lr_scale != 1.0:  # v9.6: the scale applies only to a warm start; never silently ignored
        raise SystemExit("--warm-lr-scale needs --warm-start")
    if a.warm_start is not None:
        if not (math.isfinite(a.warm_lr_scale) and a.warm_lr_scale > 0):
            raise SystemExit("--warm-lr-scale must be a positive number")
        if a.reeval_run:
            raise SystemExit("--warm-start trains from a source run's fold models; --reeval-run only predicts: not both")
        if a.backbone == OFL_BACKBONE:
            raise SystemExit("--warm-start: ofl_l fold models are deploy state dicts without the frozen public weights, so the whole-Net "
                             "strict load is not possible for them (not implemented)")
    ext_classes = tuple(s.strip() for s in a.extent_classes.split(","))      # v9.4: the default tuple is the three OA classes = v9.3
    for l in ext_classes:
        if l not in OA_L:
            raise SystemExit("--extent-classes: %r is not one of %s" % (l, OA_L))
    if set(ext_classes) != set(OA_L) and not a.extent:
        raise SystemExit("--extent-classes changes the extent head's logit term: needs --extent")
    # ---- v9.8: the --fm-aux flags (a companion flag without --fm-aux is refused, never silently ignored)
    fm_flags = ("fm_aux_weight", "fm_aux_windows", "fm_aux_flip", "fm_aux_workers", "fm_aux_log_every", "fm_aux_min_volumes")
    if a.fm_aux is None:
        set_fm = ["--" + f_.replace("_", "-") for f_ in fm_flags if getattr(a, f_) != ap.get_default(f_)]
        if set_fm:
            raise SystemExit("%s change the auxiliary task: need --fm-aux" % ", ".join(set_fm))
    else:
        fm_bad = []
        if a.reeval_run:
            fm_bad.append("--reeval-run only predicts (the auxiliary task trains)")
        if "convnext" not in a.backbone:
            fm_bad.append("backbone %s: the auxiliary path is implemented for ConvNeXt (no BatchNorm: an auxiliary batch would move "
                          "BatchNorm statistics -- CoAtNet / ResNet refused; ViT / OFL encoders return no [N, d, G, G] map)" % a.backbone)
        if not a.spatial_attn:
            fm_bad.append("needs --spatial-attn (the encoder must return its stride-32 map)")
        if a.crop % FM_CELL or not FM_CELL <= a.crop <= FM_SIZE:
            fm_bad.append("--crop %d: the auxiliary windows (rendered at %d) are cropped to it, so it must be a multiple of %d in %d..%d" % (
                a.crop, FM_SIZE, FM_CELL, FM_CELL, FM_SIZE))
        if not (math.isfinite(a.fm_aux_weight) and a.fm_aux_weight >= 0):
            fm_bad.append("--fm-aux-weight must be a finite number >= 0")
        if a.fm_aux_windows <= 0 or a.fm_aux_windows % (2 * FM_VOLS_PER_STEP):
            fm_bad.append("--fm-aux-windows %d: a positive multiple of %d (%d volumes x an even number of windows: half on slices with a "
                          "box)" % (a.fm_aux_windows, 2 * FM_VOLS_PER_STEP, FM_VOLS_PER_STEP))
        elif a.fm_aux_windows < 0.25 * a.batch * a.train_windows:
            fm_bad.append("--fm-aux-windows %d < 0.25 x %d main windows per step (spec B5)" % (a.fm_aux_windows, a.batch * a.train_windows))
        if not 0.0 <= a.fm_aux_flip <= 1.0:
            fm_bad.append("--fm-aux-flip is a probability")
        if a.fm_aux_workers < 0 or a.fm_aux_log_every < 1 or a.fm_aux_min_volumes < 1:
            fm_bad.append("--fm-aux-workers >= 0, --fm-aux-log-every >= 1, --fm-aux-min-volumes >= 1")
        if fm_bad:
            raise SystemExit("--fm-aux refused: " + " | ".join(fm_bad))
    # ---- v9.8 E2: an --out that already holds a finished run is refused before ANY file is written (v9.7 rewrote its run_config.json
    # and pool_uids.json first and only then skipped or refused the done arm); --check-targets writes nothing and is not affected
    if not a.check_targets and os.path.isdir(a.out):
        done_m = sorted(f_ for f_ in os.listdir(a.out) if FM_DONE_RE.match(f_) and os.path.isfile(os.path.join(a.out, f_)))
        if done_m:
            raise SystemExit("--out %s already holds a finished run (done marker %s): refused before writing any file; use a fresh --out" % (
                a.out, ", ".join(done_m)))
    fm_store = None
    if a.fm_aux is not None:                             # v9.8: the store, checked whole before anything is written or built
        fm_store = FmStore(a.fm_aux, a.fm_aux_min_volumes)
        print("v9.8 fm-aux: store %s" % fm_store.summary(), flush=True)
        print("v9.8 fm-aux: lambda %g, M %d windows per step (%d volumes x %d; boxed volumes %.0f%%, half their windows on slices with a box), "
              "views %s uniform, crop %d (offset uniform on 0..%d) -> G %d, shift %s, flip p %.2f, gamma / contrast as augment(); dense BCE "
              "pos_weight %g + presence (log-mean-exp over %d positions, >= %.0f%% of a box's area) BCE; head aux_* (LayerNorm %s -> linear "
              "-> %d); %d loader workers; |lambda grad aux| / |grad main| every %d steps; holdout presence AUC every epoch" % (
                  a.fm_aux_weight, a.fm_aux_windows, FM_VOLS_PER_STEP, a.fm_aux_windows // FM_VOLS_PER_STEP, 100 * FM_BOXED_FRAC,
                  list(FM_VIEWS), a.crop, FM_SIZE - a.crop, a.crop // FM_CELL, ("-%d..%d px (--shift-px)" % (a.shift_px, a.shift_px)) if a.shift_px
                  else "none (--shift-px 0)", a.fm_aux_flip, FM_POS_WEIGHT, (a.crop // FM_CELL) ** 2, 100 * FM_PRESENCE_MIN, "d", FM_A,
                  a.fm_aux_workers, a.fm_aux_log_every), flush=True)
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cpu") if a.check_targets else torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    index = json.load(open(os.path.join(a.px, "index.json"), encoding="utf-8"))
    done = {u for u, v in index["done"].items() if v[1] > 0}
    rows = list(csv.DictReader(open(a.targets, encoding="utf-8")))
    by_uid = {r["StudyInstanceUID"]: r for r in rows}
    if a.extra:
        for r in csv.DictReader(open(a.extra, encoding="utf-8")):
            if r["StudyInstanceUID"] in by_uid:
                by_uid[r["StudyInstanceUID"]].update({k: v for k, v in r.items() if k != "StudyInstanceUID"})
    gold = [r["StudyInstanceUID"] for r in rows if r["is_gold"] == "1" and r["StudyInstanceUID"] in done]
    # a non-gold study whose report labels were blanked (e.g. the report describes another body part) carries no target:
    # it leaves the pool, exactly like a gold row (logged below; bad_report_studies.json lists the blanked studies)
    blanked = {r["StudyInstanceUID"] for r in rows if r["is_gold"] == "0" and any(r.get(L + "|prior", "") == "" for L in LABELS)}
    pool = sorted(r["StudyInstanceUID"] for r in rows if r["is_gold"] == "0" and r["StudyInstanceUID"] in done
                  and r["StudyInstanceUID"] not in blanked
                  and ((r["review_holdout"] == "0" and int(r["fold"]) != -3) or a.include_review_holdout))
    assert not (set(pool) & set(gold))
    for u in gold:
        assert all(by_uid[u][L + "|strict"] == "" and by_uid[u][L + "|prior"] == "" for L in LABELS)
    K = a.folds
    pool_fold = np.array([int(hashlib.sha1(u.encode()).hexdigest()[:8], 16) % K for u in pool])
    print("pool %d gold %d folds %s | non-gold studies with blanked report labels left out: %d" % (
        len(pool), len(gold), np.bincount(pool_fold, minlength=K).tolist(), len(blanked & done)), flush=True)

    def col(uids, suffix, default):
        out = np.full((len(uids), C), default, np.float32)
        for i, u in enumerate(uids):
            for j, L in enumerate(LABELS):
                v = by_uid[u].get(L + "|" + suffix, "")
                if v != "":
                    out[i, j] = float(v)
        return torch.from_numpy(out).to(dev)

    specs = {}
    for item in a.arms.split(","):
        if "=" in item:
            name, spec = item.split("=", 1)
            specs[name] = tuple(spec.split(":"))

    def arm_tensors(arm):
        if arm == "S_strict":
            return "bce", col(pool, "strict", 0.), col(pool, "strict_w", 0.)
        if arm == "P_presence":
            return "bce", col(pool, "presence", 0.), col(pool, "presence_w", 0.)
        # v8: an explicit definition (name=kind:column[:weight_column]) wins over the built-in arm of the same name, so
        # "E_lrprior=lr:lr_prior:w" keeps the arm name (and every downstream file name) while adding cell weights
        spec = specs.get(arm) or {"E_prior": ("bce", "prior"), "E_lrprior": ("lr", "lr_prior")}[arm]
        kind, column = spec[0], spec[1]
        if len(spec) > 2 and spec[2]:                       # name=kind:column:weight_column  (0 = cell carries no information)
            w = col(pool, spec[2], float("nan") if kind == "lr" else 0.)
            t = col(pool, column, float("nan") if kind == "lr" else 0.)
            if kind == "lr":                                # v8: every pool cell needs a target and a weight (log lr)
                assert not torch.isnan(t).any() and not torch.isnan(w).any(), "lr arm with weights: a cell lacks lr or weight"
                assert bool((w >= 0).all()), "negative cell weight"
            return kind, t, w
        t = col(pool, column, float("nan"))
        assert not torch.isnan(t).any()
        return kind, t, torch.ones_like(t)

    bias0 = torch.logit(col(pool, "prior", float("nan")).mean(0).clamp(1e-3, 1 - 1e-3))

    state_spec = None
    if a.state_targets:
        state_spec = json.load(open(a.state_spec, encoding="utf-8"))
        srows = {r["StudyInstanceUID"]: r for r in csv.DictReader(open(a.state_targets, encoding="utf-8"))}
        st_idx = torch.full((len(pool), C), -1, dtype=torch.long)
        st_w = torch.zeros((len(pool), C))
        bin_names = [(b, l) for b, labs in state_spec["binaries"].items() for l in labs]
        bin_t = torch.full((len(pool), len(bin_names)), float("nan"))
        for i, u in enumerate(pool):
            r = srows.get(u)
            if r is None:
                continue
            for c, l in enumerate(LABELS):
                v = r.get(l + "|state", "")
                if v != "":
                    st_idx[i, c] = int(v); st_w[i, c] = float(r.get(l + "|sw") or 1.0)
            for j, (b, l) in enumerate(bin_names):
                v = r.get(l + "|" + b, "")
                if v != "":
                    bin_t[i, j] = float(v)
        st_idx, st_w, bin_t = st_idx.to(dev), st_w.to(dev), bin_t.to(dev)
        print("state targets: known cells per class %s | binaries known %s" % (
            (st_idx >= 0).sum(0).tolist(), (~torch.isnan(bin_t)).sum(0).tolist()), flush=True)

    # ---- v9.7: --save-state-oof -- the read-out's column layout from this run's state spec (refused here, before any data is loaded)
    st_layout = None
    if a.save_state_oof:
        if state_spec is None:
            raise SystemExit("--save-state-oof reads out the state head: needs --state-targets / --state-spec")
        st_layout = state_layout(state_spec)
        print("v9.7 save-state-oof: the state head's read-out of every held-out and gold study a fold model predicts -> %soof_state_<arm>_s<seed>"
              ".npy [%d pool rows x %d columns: %s; binaries %s] + .json (layout), %spreds_state_<arm>_s<seed>.json (gold, per fold model); "
              "per class softmax over the class's valid slots (Net.state_evidence's p), per binary sigmoid" % (
                  "partial_" if a.only_folds else "", len(pool), st_layout["width"],
                  ", ".join("%s %d" % (c_["class"], c_["stop"] - c_["start"]) for c_ in st_layout["columns"] if "categories" in c_),
                  ["%s/%s" % (c_["binary"], c_["class"]) for c_ in st_layout["columns"] if "binary" in c_], "partial_" if a.only_folds else ""),
              flush=True)

    # ---- v9.6: warm start -- the source run must have this run's Net structure on the same hash folds and the same pool, and every fold
    # model to be loaded must exist: refused here, before any data is loaded or any model is built (each file is loaded at its fold's start)
    warm_rec = None
    if a.warm_start is not None:
        wdir = os.path.abspath(a.warm_start)
        for f_ in ("run_config.json", "pool_uids.json"):
            if not os.path.isfile(os.path.join(wdir, f_)):
                raise SystemExit("--warm-start %s refused: no %s there" % (wdir, f_))
        src_cfg = json.load(open(os.path.join(wdir, "run_config.json"), encoding="utf-8"))
        warm_bad = []
        for f_ in WARM_FIELDS:
            mine, theirs = getattr(a, f_), src_cfg.get(f_, ap.get_default(f_))
            if f_ == "extent_classes":                   # a set of class names (the order does not change the mask)
                mine, theirs = sorted(s.strip() for s in mine.split(",")), sorted(s.strip() for s in str(theirs).split(","))
            if mine != theirs:
                warm_bad.append("%s: this run %r, source %r" % (f_, mine, theirs))
        # the state head exists iff --state-targets is given (then built from --state-spec): same presence; and the source's spec must have
        # this run's CATEGORIES (per class the same states in the same order: the state head's slots) and BINARIES (the same names over the
        # same classes in the same order: the binary heads' rows).  Its prior / base values may differ (R5: a re-calibrated spec) -- the three
        # spec buffers are re-set from this run's spec after every load (warm_load)
        sp_mine = a.state_spec if a.state_targets else None
        sp_src = src_cfg.get("state_spec") if src_cfg.get("state_targets") else None
        spec_note = "no state head"
        if (sp_mine is None) != (sp_src is None):
            warm_bad.append("state head: this run %s, source %s" % ("on (" + str(sp_mine) + ")" if sp_mine else "off", "on (" + str(sp_src) + ")" if sp_src else "off"))
        elif sp_mine is not None:
            try:
                spec_src = json.load(open(sp_src, encoding="utf-8"))
            except (OSError, ValueError):
                spec_src = None
            if spec_src is None:
                warm_bad.append("state spec: the source's %s cannot be read, so its categories / binaries cannot be compared" % sp_src)
            else:
                cat_diff = [l for l in LABELS if spec_src.get("categories", {}).get(l) != state_spec["categories"].get(l)]
                if cat_diff:
                    warm_bad.append("state spec categories differ for %s (this run %s, source %s)" % (cat_diff, sp_mine, sp_src))
                if list(spec_src.get("binaries", {}).items()) != list(state_spec["binaries"].items()):
                    warm_bad.append("state spec binaries differ (this run %s %s, source %s %s)" % (
                        sp_mine, list(state_spec["binaries"].items()), sp_src, list(spec_src.get("binaries", {}).items())))
                vals_same = all(spec_src.get(f_, {}).get(l) == state_spec[f_].get(l) for f_ in ("prior", "base") for l in LABELS)
                spec_note = "%s: categories and binaries == the source's (%s); prior / base values %s" % (
                    os.path.basename(sp_mine), os.path.basename(sp_src), "identical" if vals_same else
                    "DIFFER -- state_prior / state_valid / state_base_logit are re-set from this run's spec after every load")
        src_pool = json.load(open(os.path.join(wdir, "pool_uids.json"), encoding="utf-8"))
        if src_pool != pool:
            warm_bad.append("pool: this run %d studies, source %d (%d of this run's not in the source, %d of the source's not in this run)" % (
                len(pool), len(src_pool), len(set(pool) - set(src_pool)), len(set(src_pool) - set(pool))))
        only_w = None if not a.only_folds else {int(x) for x in a.only_folds.split(",")}
        warm_need = [os.path.join(wdir, "models", "%s_s%d_f%d.pt" % (arm_, seed_, k_)) for seed_ in range(a.seed_base, a.seed_base + a.seeds)
                for arm_ in [x.split("=", 1)[0] for x in a.arms.split(",") if x] for k_ in range(K)
                if (only_w is None or k_ in only_w) and k_ != a.holdout_fold]
        warm_miss = [p_ for p_ in warm_need if not os.path.isfile(p_)]
        if warm_miss:
            warm_bad.append("fold models missing: %s" % ", ".join(warm_miss))
        # this run must never write over its source: not the same directory, and no file already in --out (top level, models/, ckpt/)
        # may be the same file as one of the source's (a merged run directory holds hard links to the per-fold originals; this run opens
        # run_config.json, pool_uids.json, preds, OOF, fold models and checkpoints for writing)
        if os.path.realpath(a.out) == os.path.realpath(wdir):
            warm_bad.append("--out is the warm-start directory itself (its run_config, pool and fold models would be overwritten)")
        src_ids, out_ids = file_ids(wdir), file_ids(a.out)
        warm_same = sorted(out_ids[i_] for i_ in set(out_ids) & set(src_ids))
        if warm_same:
            warm_bad.append("file(s) in --out are the same files as the source's (hard links?): %s" % ", ".join(warm_same))
        # this run's spec must be well formed (one prior per category): the state head's valid slots come from the prior lists
        if state_spec is not None:
            bad_len = [l for l in LABELS if len(state_spec["prior"][l]) != len(state_spec["categories"][l])]
            if bad_len:
                warm_bad.append("state spec %s: the prior list length differs from the category count for %s" % (sp_mine, bad_len))
        if warm_bad:
            raise SystemExit("--warm-start %s refused: %s" % (wdir, " | ".join(warm_bad)))
        # the record a warm checkpoint carries (+ the source fold model's md5): a resume needs the same source, scale and state spec file
        warm_rec = {"dir": wdir, "lr_scale": a.warm_lr_scale, "state_spec_md5": file_md5(a.state_spec) if a.state_targets else None}
        print("v9.6 warm-start: source %s matches this run (K %d, holdout fold %s, %s, crop %d, slices %d, spatial %s, extent %s, syn-decomp %s, "
              "pool %d); state spec %s; %d fold model(s) to load; LR x %g; optimizer, LR schedule (over --epochs %d) and EMA start fresh" % (
                  wdir, K, a.holdout_fold, a.backbone, a.crop, a.slices, a.spatial_attn, a.extent, a.syn_decomp, len(pool), spec_note,
                  len(warm_need), a.warm_lr_scale, a.epochs), flush=True)

    # ---- v9.8 E1: --reeval-run predicts with the source's fold models; their state head was trained under the source's state spec, so
    # this run's --state-spec must be that very file content (md5), else the state columns would be read out under the wrong labels.
    # Refused at start-up (before anything is written), with both paths and md5s.  Also moved here from reeval() (E2: every start-up check
    # before run_config.json): the source's pool must be this run's pool.
    if a.reeval_run and not a.check_targets:
        rdir = os.path.abspath(a.reeval_run)
        if not os.path.isfile(os.path.join(rdir, "pool_uids.json")):
            raise SystemExit("--reeval-run %s refused: no pool_uids.json there" % rdir)
        if json.load(open(os.path.join(rdir, "pool_uids.json"), encoding="utf-8")) != pool:
            raise SystemExit("pool of --reeval-run differs from the current pool")
        r_cfg_p = os.path.join(rdir, "run_config.json")
        r_cfg = json.load(open(r_cfg_p, encoding="utf-8")) if os.path.isfile(r_cfg_p) else None
        sp_this = os.path.abspath(a.state_spec) if a.state_targets else None
        if r_cfg is None:
            if sp_this is not None:
                raise SystemExit("--reeval-run %s refused: no run_config.json there, so the state spec its fold models were trained with "
                                 "(this run: %s, md5 %s) cannot be checked" % (rdir, sp_this, file_md5(sp_this)))
        else:
            sp_src = r_cfg.get("state_spec") if r_cfg.get("state_targets") else None
            if (sp_this is None) != (sp_src is None):
                raise SystemExit("--reeval-run %s refused: state head %s in the source run (state_spec %s) but %s in this run (--state-spec "
                                 "%s)" % (rdir, "on" if sp_src else "off", sp_src, "on" if sp_this else "off", sp_this))
            if sp_this is not None:
                if not os.path.isfile(sp_src):
                    raise SystemExit("--reeval-run %s refused: the source run's state spec %s (run_config.json) does not exist, so its md5 "
                                     "cannot be compared with this run's %s (md5 %s)" % (rdir, sp_src, sp_this, file_md5(sp_this)))
                m_src, m_this = file_md5(sp_src), file_md5(sp_this)
                if m_src != m_this:
                    raise SystemExit("--reeval-run %s refused: the source run's state spec %s (md5 %s) is not this run's --state-spec %s "
                                     "(md5 %s): the state columns would carry the wrong labels" % (rdir, sp_src, m_src, sp_this, m_this))

    # ---- v9.6: --rare-boost-names (SCAN_A D4): the name rules under this run's spec, checked here (also under --check-targets); the
    # masks of both rule sets over the pool are kept for make_loader, which cross-checks them against v9.5's rare_weights per training set
    rare_masks_idx = rare_masks_name = pool_row = None
    if a.rare_boost_names:
        if state_spec is None:
            raise SystemExit("--rare-boost-names needs --state-targets (the rare-boost terms read the report states)")
        cats_rb = state_spec["categories"]
        miss_rb = [(l, n_) for l, _, ns in RARE_NAME_TERMS for n_ in ns if n_ not in cats_rb[l]]
        if miss_rb:
            raise SystemExit("--rare-boost-names: state names missing from the spec %s: %s" % (a.state_spec, miss_rb))
        bn_rb = {(b, l): j for j, (b, l) in enumerate(bin_names)}
        acute_rb = (bin_t[:, bn_rb[("acute", "MCL")]].cpu().numpy() == 1.0) if ("acute", "MCL") in bn_rb else np.zeros(len(pool), bool)
        st_rb = st_idx.cpu().numpy()
        rare_masks_idx = rare_term_masks(st_rb, cats_rb, acute_rb, False)
        rare_masks_name = rare_term_masks(st_rb, cats_rb, acute_rb, True)
        diff_rb = [l for l in ("Fracture", "PF OA", "Baker's", "Contusion") if not np.array_equal(rare_masks_idx[l], rare_masks_name[l])]
        if diff_rb:
            raise SystemExit("--rare-boost-names: under %s the name rules of %s select other studies than v9.5's index rules (%s)" % (
                a.state_spec, diff_rb, {l: (int(rare_masks_idx[l].sum()), int(rare_masks_name[l].sum())) for l in diff_rb}))
        pool_row = {u: i for i, u in enumerate(pool)}
        w_rb_idx, w_rb_name = rare_weights_of(rare_masks_idx), rare_weights_of(rare_masks_name)
        mcl_c = LABELS.index("MCL")
        lost_rb = rare_masks_idx["MCL"] & ~rare_masks_name["MCL"]
        print("v9.6 rare-boost-names: terms by state name %s; Fracture / PF OA / Baker's / Contusion select the same pool studies as v9.5's "
              "index rules (%s); MCL boosted by v9.5 but not by name: %d pool studies (their MCL states %s; gained: %d); rare weight "
              "changes for %d of %d pool studies (mean weight %.4f -> %.4f)" % (
                  {l: list(ns) for l, _, ns in RARE_NAME_TERMS},
                  {l: int(rare_masks_name[l].sum()) for l in ("Fracture", "PF OA", "Baker's", "Contusion")}, int(lost_rb.sum()),
                  dict(collections.Counter(cats_rb["MCL"][int(s_)] for s_ in st_rb[lost_rb, mcl_c])),
                  int((rare_masks_name["MCL"] & ~rare_masks_idx["MCL"]).sum()), int((w_rb_idx != w_rb_name).sum()), len(pool),
                  float(w_rb_idx.mean()), float(w_rb_name.mean())), flush=True)
    if a.draw_seed:                                      # v9.6: start-up line of the per-draw window streams
        print("v9.6 draw-seed: every training draw seeds its windows with [seed, epoch, index, nonce = position in the epoch] (repeat "
              "draws of a study in one epoch get their own windows); the samplers' index sequence is unchanged; evaluation and BN "
              "recalibration keep the (seed, epoch, index) seeding", flush=True)

    # ---- v9.4: anatomical mirror -- P over the classes (and the binaries) and the MCL binaries, fixed for the run; the spec check
    # (identical state category lists of the exchanged classes) refuses here, before any data is loaded
    mir_perm = mir_bin_perm = mir_mcl_bins = None
    if a.mirror > 0:
        perm_, bperm_ = mirror_perm(state_spec, bin_names if state_spec is not None else None)
        mir_perm = torch.tensor(perm_, device=dev)
        mir_bin_perm = None if bperm_ is None else torch.tensor(bperm_, device=dev)
        mir_mcl_bins = [j for j, (b, l) in enumerate(bin_names) if l == "MCL"] if state_spec is not None else []
        print("v9.4 mirror: probability %.2f per training batch; output permutation %s%s; MCL not trained on mirrored batches (main / "
              "state / pair weights 0%s)" % (a.mirror, perm_, "" if bperm_ is None else ", binaries " + str(bperm_),
                                            "" if not mir_mcl_bins else ", binaries %s unknown" % mir_mcl_bins), flush=True)
    if a.shift_px > 0:                                   # v9.5: start-up line of the translation jitter
        print("v9.5 shift: one integer translation (dy, dx), each uniform on -%d..%d, per training batch (every study / window / plane / "
              "channel of the batch), zero fill, after the crop%s; own stream 9500 + 17*seed + fold; never in evaluation" % (
                  a.shift_px, a.shift_px, " and before the v2 affine" if a.aug == "v2" else " (--aug crop: no affine)"), flush=True)
    if a.aug_grid_fp32:                                  # v9.6: start-up line of the float32 affine grid
        print("v9.6 aug-grid-fp32: the --aug v2 rotation / scale grid (affine_grid) and its bilinear resampling (grid_sample) run in float32 "
              "with autocast off for those two calls; every random draw and every other operation unchanged", flush=True)

    # ---- P1 pair families (v9.3: built by build_pair_families, indexed by family; v9.2 = the same v9 list indexed by class)
    pair_fams, pair_info, fam_class, fam_q = [], {}, [], []
    new_fam = a.oa_area_rank or a.oa_sign_decoy or a.acl_acuity_decoy or a.mcl_boundary
    if (a.pair_extra or new_fam) and not a.pair_rank > 0:
        raise SystemExit("--pair-extra / --oa-area-rank / --oa-sign-decoy / --acl-acuity-decoy / --mcl-boundary need --pair-rank > 0")
    if a.oa_area_rank and not a.extent:
        raise SystemExit("--oa-area-rank ranks the extent head's area: needs --extent")
    lig_age = load_lig_age(a.lig_age, pool) if a.lig_age else None
    if (a.acl_acuity_decoy or a.mcl_boundary) and lig_age is None:
        raise SystemExit("--acl-acuity-decoy / --mcl-boundary need --lig-age <csv> (columns study, class, age)")
    if a.pair_rank > 0:
        if state_spec is None:
            raise SystemExit("--pair-rank needs --state-targets")
        pair_fams, role, sig, pw, fam_class = build_pair_families(
            state_spec, st_idx.cpu().numpy(), bin_t.cpu().numpy(), bin_names, a.pair_extra, a.oa_area_rank, a.oa_sign_decoy,
            a.acl_acuity_decoy, a.mcl_boundary, lig_age)
        fam_q = [fm["quantity"] for fm in pair_fams]
        pair_info = {"role": role, "sig": sig}
        role_t, sig_t, pw_t = torch.from_numpy(role).to(dev), torch.from_numpy(sig).to(dev), torch.from_numpy(pw).to(dev)
        print("pair families: " + "; ".join("%s pos %d neg %d" % (fm["name"], int((role[:, f] == 1).sum()), int((role[:, f] == -1).sum()))
                                              for f, fm in enumerate(pair_fams)), flush=True)
        if new_fam:
            print("v9.3 families over the pool: " + " | ".join(family_summary(pair_fams, role, sig)), flush=True)

    ext_y = None
    if a.extent:
        if state_spec is None:
            raise SystemExit("--extent needs --state-targets")
        ext_y = torch.full((len(pool), C), float("nan"))
        for l in ("Medial OA", "Lateral OA", "PF OA"):
            c, cats_ = LABELS.index(l), state_spec["categories"][l]
            pos_i, neg_i = cats_.index("high_grade_large"), {cats_.index("normal"), cats_.index("low_grade_chondral")}
            si = st_idx[:, c].cpu()
            ext_y[si == pos_i, c] = 1.0
            for k_ in neg_i:
                ext_y[si == k_, c] = 0.0
            if a.oa_area_rank:                   # v9.3: the small high-grade lesion as the head's "< 1 cm^2 leaning" example
                ext_y[si == cats_.index("high_grade_small"), c] = OA_SMALL_TARGET
        ext_y = ext_y.to(dev)
        print("extent supervision: positives (>= 1 cm^2 stated) %s, negatives %s" % (
            [int((ext_y[:, LABELS.index(l)] == 1).sum()) for l in ("Medial OA", "Lateral OA", "PF OA")],
            [int((ext_y[:, LABELS.index(l)] == 0).sum()) for l in ("Medial OA", "Lateral OA", "PF OA")]), flush=True)
        if a.oa_area_rank:
            print("extent supervision v9.3: high_grade_small cells at soft target %.1f: %s" % (OA_SMALL_TARGET,
                  [int((ext_y[:, LABELS.index(l)] == OA_SMALL_TARGET).sum()) for l in OA_L]), flush=True)

    def extent_loss(area, sel):
        """>= 1 cm^2 stated (1) vs normal / low grade (0), weighted by the report certainty like the state loss (audit 5)."""
        y = ext_y[sel]
        known = ~torch.isnan(y)
        if not bool(known.any()):
            return area.new_zeros(())
        w_ = st_w[sel].float()[known]
        logit_ = 2.0 * (torch.log1p(area.float()) - math.log(101.0))
        l_ = F.binary_cross_entropy_with_logits(logit_[known], y[known], reduction="none")
        return (l_ * w_).sum() / w_.sum().clamp_min(1e-6)

    def pair_buckets(uids):
        """v9.3: pair_buckets_of over the training studies' rows (v9.2's bucket list for the v9 family list)."""
        idx = {u: i for i, u in enumerate(pool)}
        pi = np.array([idx[u] for u in uids])
        return pair_buckets_of(pair_info["role"], pair_info["sig"], pi)

    def pair_loss(z, sel, area=None, cw=None):
        """v9.3: pair_loss_of on the batch's rows; area = the extent head's [B, C] area for the 'area' families (None otherwise).
        v9.4: cw = the batch's cell weights when they differ from st_w[sel] (a mirrored batch: MCL column 0)."""
        return pair_loss_of(z, area, role_t[sel], sig_t[sel], pw_t[sel], st_w[sel] if cw is None else cw, fam_class, fam_q)

    if a.check_targets:
        for item in a.arms.split(","):
            kind, t, w = arm_tensors(item.split("=")[0])
            print("check-targets arm %s (%s): %d x %d cells, finite %s, weighted cells %d, mean target per class %s" % (
                item.split("=")[0], kind, t.shape[0], t.shape[1], bool(torch.isfinite(t).all()), int((w > 0).sum()),
                [round(float(v), 3) for v in t.mean(0)]), flush=True)
        print("check-targets: bias0 finite %s; holdout fold %s has %d pool studies" % (
            bool(torch.isfinite(bias0).all()), a.holdout_fold, int((pool_fold == a.holdout_fold).sum()) if a.holdout_fold is not None else 0))
        if a.pair_rank > 0:                       # v9.3: every family's sizes and buckets over the whole pool and over each fold's training set
            print("check-targets families (pool): " + " | ".join(family_summary(pair_fams, pair_info["role"], pair_info["sig"])), flush=True)
            for k in range(K):
                tr_ = np.flatnonzero((pool_fold != k) & (pool_fold != a.holdout_fold)) if a.holdout_fold is not None else np.flatnonzero(pool_fold != k)
                bk_ = pair_buckets_of(pair_info["role"], pair_info["sig"], tr_)
                print("check-targets fold %d training set %d studies: %d buckets, probability sum %.4f" % (k, len(tr_), len(bk_), sum(b[2] for b in bk_)), flush=True)
        bad = []
        for item in a.arms.split(","):
            kind, t, w = arm_tensors(item.split("=")[0])
            if not bool(torch.isfinite(t).all()):
                bad.append("non-finite targets in arm " + item)
            if kind == "lr" and bool((t <= 0).any()):
                bad.append("non-positive likelihood ratios in arm " + item)
            if kind == "bce" and bool(((t < 0) | (t > 1)).any()):
                bad.append("probability targets outside [0, 1] in arm " + item)
        if not bool(torch.isfinite(bias0).all()):
            bad.append("non-finite bias0 (a blank or non-numeric prior in the pool)")
        if state_spec is not None:
            kc = torch.tensor([len(state_spec["categories"][l]) for l in LABELS], device=st_idx.device)
            if bool((st_idx >= kc[None]).any()):
                bad.append("state index outside its class's category list")
        if bad:
            print("check-targets FAILED: " + "; ".join(bad), flush=True)
            raise SystemExit(1)
        print("check-targets OK", flush=True)
        raise SystemExit(0)

    def rare_weights(uids):
        """1 + 2[MCL state>=2 or acute] + 2[fracture state>=1] + [PF OA high] + [Baker>=moderate] + [traumatic edema/contusion], cap 4."""
        idx = {u: i for i, u in enumerate(pool)}
        L = {l: c for c, l in enumerate(LABELS)}
        bn = {(b, l): j for j, (b, l) in enumerate(bin_names)}
        w = []
        for u in uids:
            i = idx[u]
            st = st_idx[i].tolist()
            v = 1.0
            v += 2.0 * (st[L["MCL"]] >= 2 or (("acute", "MCL") in bn and float(bin_t[i, bn[("acute", "MCL")]]) == 1.0))
            v += 2.0 * (st[L["Fracture"]] >= 1)
            v += 1.0 * (st[L["PF OA"]] in (3, 4))
            v += 1.0 * (st[L["Baker's"]] in (3, 4, 5))
            v += 1.0 * (st[L["Contusion"]] in (4, 5))
            w.append(min(v, 4.0))
        return torch.tensor(w, dtype=torch.double)

    def make_loader(uids, mode, seed, shuffle, nonce=False):
        """v9.6: nonce=True (only run()'s training loader, under --draw-seed) makes the training sampler yield (index, nonce) draws"""
        ds = Studies(a.px, uids, a.train_windows, mode, seed, eval_stride=a.eval_stride,
                     train_sampling=a.train_sampling, cover_stride=a.cover_stride, px_extra=a.px_extra, slices=a.slices,
                     slice_jitter=a.slice_jitter if mode == "train" else 0.0, phys_mm=a.phys_mm, px_override=a.px_override,
                     extent_meta=a.extent or a.mirror > 0)      # v9.4: the mirror reads the plane column (the model ignores the extra columns)
        train = mode == "train"                     # evaluation: one study per batch, few workers, no pinned copies
        sampler = None
        if train and a.rare_boost:
            if state_spec is None:
                raise SystemExit("--rare-boost needs --state-targets")
            w = rare_weights(uids)
            if a.rare_boost_names:                   # v9.6 (SCAN_A D4): the name-rule weights; v9.5's (w) must equal the index-rule replica
                rows_rb = np.array([pool_row[u] for u in uids])
                w_idx_rb = rare_weights_of({l: m_[rows_rb] for l, m_ in rare_masks_idx.items()})
                if not np.array_equal(w_idx_rb, w.numpy()):
                    raise SystemExit("--rare-boost-names: the index-rule replica differs from rare_weights on this training set")
                w_name_rb = torch.tensor(rare_weights_of({l: m_[rows_rb] for l, m_ in rare_masks_name.items()}), dtype=torch.double)
                print("v9.6 rare-boost-names: %d of %d training studies change weight (MCL by name; mean %.4f -> %.4f)" % (
                    int((w_name_rb != w).sum()), len(uids), float(w.mean()), float(w_name_rb.mean())), flush=True)
                w = w_name_rb
            sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=len(uids), replacement=True,
                                                             generator=torch.Generator().manual_seed(seed + 7))
            if nonce:                                # v9.6 --draw-seed: the same sampler (weights, generator), yielding (index, nonce)
                sampler = NonceWeightedSampler(sampler.weights, num_samples=len(uids), replacement=True, generator=sampler.generator)
            shuffle = False
            print("rare-boost: mean weight %.2f, boosted studies %d of %d" % (float(w.mean()), int((w > 1).sum()), len(w)), flush=True)
        if train and a.pair_rank > 0:               # matched pairs on top of the same base distribution
            bk = pair_buckets(uids)
            bs = PairBatchSampler(len(uids), a.batch, None if sampler is None else sampler.weights, bk, a.pair_frac, seed + 7)
            if nonce:                                # v9.6 --draw-seed: the same sampler (fresh generator, same seed), yielding (index, nonce)
                bs = NoncePairBatchSampler(len(uids), a.batch, None if sampler is None else sampler.weights, bk, a.pair_frac, seed + 7)
            print("pair-rank: %d buckets over %d families, %.0f%% of batches drawn as a matched pair (others: base draws, weighted with replacement)" % (
                len(bk), len(pair_fams), 100 * a.pair_frac), flush=True)
            return ds, torch.utils.data.DataLoader(ds, batch_sampler=bs, num_workers=a.workers, collate_fn=collate,
                                                   persistent_workers=False, pin_memory=True, prefetch_factor=2,
                                                   generator=torch.Generator().manual_seed(seed))
        if nonce and train and sampler is None and shuffle:   # v9.6 --draw-seed, plain shuffle: the RandomSampler the DataLoader would
            g_ns = torch.Generator().manual_seed(seed)        # build (same generator object, which also seeds the workers), yielding (index, nonce)
            return ds, torch.utils.data.DataLoader(ds, batch_size=a.batch, sampler=NonceRandomSampler(ds, generator=g_ns),
                                                   num_workers=a.workers, collate_fn=collate, drop_last=True, persistent_workers=False,
                                                   pin_memory=True, prefetch_factor=2, generator=g_ns)
        return ds, torch.utils.data.DataLoader(ds, batch_size=a.batch if train else 1, shuffle=shuffle, sampler=sampler,
                                               num_workers=a.workers if train else min(a.workers, a.eval_workers),
                                               collate_fn=collate, drop_last=train, persistent_workers=False,
                                               pin_memory=train, prefetch_factor=2 if train else 1,
                                               generator=torch.Generator().manual_seed(seed))

    def augment(img, gen, shgen=None, shifts=None):
        B, W, _, S, _ = img.shape
        oy = int(torch.randint(0, S - a.crop + 1, (1,), generator=gen))
        ox = int(torch.randint(0, S - a.crop + 1, (1,), generator=gen))
        img = img[..., oy:oy + a.crop, ox:ox + a.crop]
        if shgen is not None:                   # v9.5: one translation per batch from its own stream, zero fill, before the affine
            dy, dx = shift_draw(a.shift_px, shgen)
            img = shift_batch(img, dy, dx, a.shift_px)
            if shifts is not None:              # the epoch read-out's record
                shifts.append((dy, dx))
        if a.aug != "v2":
            return img
        # per-study rotation (+-10 deg) and scale (0.9-1.1) through one affine grid; gamma 0.8-1.25; contrast 0.9-1.1.
        # No flips: a horizontal flip swaps medial/lateral, a vertical one anterior/posterior.
        Cc = img.shape[2]
        x = img.float().reshape(B * W, Cc, a.crop, a.crop)
        ang = (torch.rand(B, generator=gen) * 20 - 10) * math.pi / 180
        sc = torch.rand(B, generator=gen) * 0.2 + 0.9
        cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
        theta = torch.stack([torch.stack([cos, -sin, torch.zeros(B)], 1), torch.stack([sin, cos, torch.zeros(B)], 1)], 1)
        theta = theta.repeat_interleave(W, 0).to(x.device)
        if a.aug_grid_fp32:                     # v9.6: grid + bilinear resampling in float32, autocast off for these two calls only
            with torch.autocast("cuda", enabled=False):   # (in the bf16 region affine_grid's matmul runs in bf16: ~0.7 px grid error)
                grid = F.affine_grid(theta.float(), x.shape, align_corners=False)
                x = F.grid_sample(x.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        else:                                   # v9.5's two lines, unchanged
            grid = F.affine_grid(theta, x.shape, align_corners=False)
            x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        if a.aug_res > 0:                       # per-study in-plane resolution loss (coarse scanners: 0.33-0.5 mm pixels)
            hit = torch.rand(B, generator=gen) < a.aug_res
            fac = torch.rand(B, generator=gen) * 0.75 + 1.25
            if bool(hit.any()):
                xs = x.view(B, W, Cc, a.crop, a.crop)
                for b in torch.nonzero(hit).flatten().tolist():
                    s_ = max(8, int(round(a.crop / float(fac[b]))))
                    lo = F.interpolate(xs[b], size=(s_, s_), mode="area")
                    xs[b] = F.interpolate(lo, size=(a.crop, a.crop), mode="bilinear", align_corners=False)
                x = xs.view(B * W, Cc, a.crop, a.crop)
        gamma = (torch.rand(B, generator=gen) * 0.45 + 0.8).repeat_interleave(W, 0).to(x.device).view(-1, 1, 1, 1)
        contrast = (torch.rand(B, generator=gen) * 0.2 + 0.9).repeat_interleave(W, 0).to(x.device).view(-1, 1, 1, 1)
        x = (x / 255.).clamp_(0, 1).pow(gamma)
        x = ((x - 0.5) * contrast + 0.5).clamp_(0, 1) * 255.
        return x.round_().to(torch.uint8).reshape(B, W, Cc, a.crop, a.crop)

    def center(img):
        S = img.shape[-1]
        o = (S - a.crop) // 2
        return img[..., o:o + a.crop, o:o + a.crop]

    @torch.no_grad()                                  # evaluation only: no autograd graph (same outputs, less time and memory)
    def predict(model, uids, seed, want_spec=False, want_area=False, want_state=False):
        model.eval()
        ds, dl = make_loader(uids, "eval", seed, False)
        out = torch.zeros((len(uids), C), device=dev)
        spec = torch.zeros((len(uids),), device=dev)
        area = torch.zeros((len(uids), C), device=dev)           # v9.3 --save-area: the extent head's area proxy per study
        if want_state:                                            # v9.7 --save-state-oof: the state head's read-out per study
            state_slots_check(model, st_layout)
            pst = torch.full((len(uids), st_layout["width"]), float("nan"), device=dev)
        for img, meta, mask, idx in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if want_state:                  # v9.7: the same forward, also returning the state / binary logits it computed (z unchanged)
                    z, sl_, bl_ = model(center(img.to(dev, non_blocking=True)), meta.to(dev), mask.to(dev), chunk=PREDICT_CHUNK, return_aux=True)
                    ps_ = state_probs(model, sl_, bl_)
                else:
                    z = model(center(img.to(dev, non_blocking=True)), meta.to(dev), mask.to(dev), chunk=PREDICT_CHUNK)
            out[idx.to(dev)] = z.float()
            if want_state:
                pst[idx.to(dev)] = ps_
            if want_spec and getattr(model, "syn_decomp", False):      # v9.2: the synovium-specific part of the last batch
                spec[idx.to(dev)] = model._syn_spec
            if want_area and getattr(model, "extent", False):          # v9.3: [B, C] area of the last batch (OA columns meaningful)
                area[idx.to(dev)] = model._ext_area.float()
        res = [out] + ([spec] if want_spec else []) + ([area] if want_area else []) + ([pst] if want_state else [])
        return res[0] if len(res) == 1 else tuple(res)

    @torch.no_grad()
    def bn_recalibrate(model, uids, seed, n_batches):
        """--ema copies only the trainable parameters into the shipped model, so BatchNorm running statistics still describe
        the live weights (CoAtNet screen 10-08: holdout 0.61, predictions nearly constant).  Recompute them for the shipped
        weights: cumulative averages over n_batches training batches, centre crop, no augmentation.  The whole model is in
        train mode so that no zero padding enters the statistics (Net.forward pads only in eval mode or without BatchNorm);
        the head's dropout comes after the encoder and does not reach them.  Returns the number of windows seen."""
        bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        if not bns or n_batches <= 0:
            return 0
        mom = [m.momentum for m in bns]
        for m in bns:
            m.reset_running_stats()
            m.momentum = None
        model.train()
        ds_r, dl_r = make_loader(uids, "train", seed, True)
        ds_r.slice_jitter = 0.0                      # audit 10-08 defect 1: no slice-jitter augmentation in the statistics
        seen = used = 0
        stance = torch.compiler.set_stance("force_eager") if hasattr(torch.compiler, "set_stance") else contextlib.nullcontext()
        with stance:                                 # audit defect 2: never through a compiled encoder (guards, recompiles)
            for i, (img, meta, mask, idx) in enumerate(dl_r):
                if i >= n_batches:
                    break
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    model(center(img.to(dev, non_blocking=True)), meta.to(dev), mask.to(dev))
                seen += int(mask.sum())
                used += 1
        for m, v in zip(bns, mom):
            m.momentum = v
        model.eval()
        return used, seen

    # ---- v9.8 --fm-aux: thin wrappers of the module-level fm_make_loader / fm_aux_step / fm_holdout_eval with this run's settings
    def fm_loader(batches, seed_w):
        return fm_make_loader(fm_store, batches, a.crop, a.shift_px if isinstance(batches, FmAuxSampler) else 0, a.fm_aux_workers, seed_w)

    def fm_step(model, batch, img_main, enc_params, seed, k, ep, b_i, steps, acc, pw):
        return fm_aux_step(model, batch, img_main, enc_params, a.fm_aux_weight, a.crop, seed, k, ep, b_i, steps, acc, pw,
                           a.fm_aux_log_every, dev)

    def fm_eval(model, hold_specs):
        return fm_holdout_eval(model, fm_store, hold_specs, a.crop, a.fm_aux_windows, a.fm_aux_workers, dev)

    def fm_auc_str(d_):
        return " ".join("%s %.3f" % (c_, v_) for c_, v_ in d_.items())

    def reeval(arm, seed):
        only = None if not a.only_folds else {int(x) for x in a.only_folds.split(",")}
        pre = "partial_" if only else ""
        path = os.path.join(a.out, pre + "preds_%s_s%d.json" % (arm, seed))
        if os.path.exists(path):
            if a.save_state_oof and not all(os.path.isfile(p_) for p_ in state_files(a.out, pre, arm, seed)):   # v9.7: never a silent skip
                raise SystemExit("--save-state-oof: %s exists (arm %s seed %d is done) but its state read-out does not (made without the flag); "
                                 "re-predict into a fresh --out" % (path, arm, seed))
            return
        src = json.load(open(os.path.join(a.reeval_run, "pool_uids.json"), encoding="utf-8"))
        if src != pool:
            raise SystemExit("pool of --reeval-run differs from the current pool")
        z_gold = torch.zeros((len(gold), C), device=dev)
        z_pool = torch.zeros((len(pool), C), device=dev)
        area_pool = torch.zeros((len(pool), C), device=dev)    # v9.3 --save-area
        t_arm = time.time()
        H = a.holdout_fold
        if a.save_state_oof:                            # v9.7: NaN = a row not predicted here; the holdout fold's rows accumulate / (K - 1)
            st_pool = torch.full((len(pool), st_layout["width"]), float("nan"), device=dev)
            if H is not None:
                st_pool[torch.from_numpy(np.flatnonzero(pool_fold == H)).to(dev)] = 0.0
            st_gold = {}                                # fold k -> fold model k's read-out of every gold study
        for k in range(K):
            if only is not None and k not in only:
                continue
            if H is not None and k == H:                # no model is trained for the holdout fold
                continue
            ho = np.flatnonzero(pool_fold == k)
            model = Net(a.backbone, a.init, freeze_blocks=a.freeze_blocks, crop=a.crop, state_spec=state_spec, in_chans=a.slices, spatial=a.spatial_attn, extent=a.extent, syn_decomp=a.syn_decomp, extent_classes=ext_classes).to(dev).to(memory_format=torch.channels_last)
            sd = torch.load(os.path.join(a.reeval_run, "models", "%s_s%d_f%d.pt" % (arm, seed, k)), map_location="cpu")
            if a.backbone == OFL_BACKBONE:
                model.load_deploy_state_dict(sd)
            else:
                model.load_state_dict(sd)
            if model.has_bn and a.bn_recal_batches > 0:
                tr = np.flatnonzero((pool_fold != k) & (pool_fold != H)) if H is not None else np.flatnonzero(pool_fold != k)
                b_, w_ = bn_recalibrate(model, [pool[i] for i in tr], 9100 + 17 * seed + k, a.bn_recal_batches)
                print("  reeval fold %d: BatchNorm statistics recalibrated over %d batches (%d windows)" % (k, b_, w_), flush=True)
            if a.save_state_oof:                        # v9.7: the same gold forward, plus fold model k's state read-out
                zg_, sg_ = predict(model, gold, seed, want_state=True)
                z_gold += zg_ / (K if H is None else K - 1)
                st_gold[k] = sg_.cpu().numpy()
            else:
                z_gold += predict(model, gold, seed) / (K if H is None else K - 1)
            if a.save_state_oof:                        # v9.7: the same predictions (+ the area proxy when asked), plus the state read-out
                res_ = predict(model, [pool[i] for i in ho], seed, want_area=a.save_area, want_state=True)
                z_pool[torch.from_numpy(ho).to(dev)] = res_[0]
                if a.save_area:
                    area_pool[torch.from_numpy(ho).to(dev)] = res_[1]
                st_pool[torch.from_numpy(ho).to(dev)] = res_[-1]
            elif a.save_area:                           # v9.3: same predictions, plus the area proxy
                zp_, ar_ = predict(model, [pool[i] for i in ho], seed, want_area=True)
                z_pool[torch.from_numpy(ho).to(dev)] = zp_
                area_pool[torch.from_numpy(ho).to(dev)] = ar_
            else:
                z_pool[torch.from_numpy(ho).to(dev)] = predict(model, [pool[i] for i in ho], seed)
            if H is not None:                       # the clean holdout fold, as in training (mean of the fold models)
                hh = np.flatnonzero(pool_fold == H)
                if a.save_state_oof:                    # v9.7: as below, plus each fold model's state read-out / (K - 1)
                    res_ = predict(model, [pool[i] for i in hh], seed, want_area=a.save_area, want_state=True)
                    z_pool[torch.from_numpy(hh).to(dev)] += res_[0] / (K - 1)
                    if a.save_area:
                        area_pool[torch.from_numpy(hh).to(dev)] += res_[1] / (K - 1)
                    st_pool[torch.from_numpy(hh).to(dev)] += res_[-1] / (K - 1)
                elif a.save_area:
                    zh_, ah_ = predict(model, [pool[i] for i in hh], seed, want_area=True)
                    z_pool[torch.from_numpy(hh).to(dev)] += zh_ / (K - 1)
                    area_pool[torch.from_numpy(hh).to(dev)] += ah_ / (K - 1)
                else:
                    z_pool[torch.from_numpy(hh).to(dev)] += predict(model, [pool[i] for i in hh], seed) / (K - 1)
            print("  reeval %s seed %d fold %d done (eval stride %d) %.0fs" % (arm, seed, k, a.eval_stride, time.time() - t_arm), flush=True)
            del model
            torch.cuda.empty_cache()
        np.save(os.path.join(a.out, pre + "oof_%s_s%d.npy" % (arm, seed)), z_pool.cpu().numpy().astype(np.float32))
        if a.save_area:                                      # v9.3 read-out: the extent head's area proxy per pool study
            np.save(os.path.join(a.out, pre + "oof_area_%s_s%d.npy" % (arm, seed)), area_pool.cpu().numpy().astype(np.float32))
        if a.save_state_oof:                                 # v9.7 read-out, written before the preds json (the done marker)
            write_state_readout(a.out, pre, arm, seed, st_layout, a.state_spec, len(pool), st_pool.cpu().numpy(), gold,
                                [int(by_uid[u]["fold"]) for u in gold], st_gold, K, H, only, a.reeval_run)
        out = {"arm": arm, "seed": seed, "reeval_of": a.reeval_run, "eval_stride": a.eval_stride, "gold_uids": gold, "labels": LABELS,
               "holdout_fold": H,
               "only_folds": sorted(only) if only else None,
               "gold_fold": [int(by_uid[u]["fold"]) for u in gold],
               "gold_logits": [[round(float(x), 5) for x in r] for r in z_gold.cpu().numpy()],
               "receipt": [{"arm": arm, "kind": "reeval", "pool_studies": len(pool), "gold_in_training": 0, "trained": False}]}
        json.dump(out, open(path, "w"))

    def run(arm, seed):
        if a.reeval_run:
            return reeval(arm, seed)
        only = None if not a.only_folds else {int(x) for x in a.only_folds.split(",")}
        path = os.path.join(a.out, ("partial_" if only else "") + "preds_%s_s%d.json" % (arm, seed))
        if os.path.exists(path):
            if a.save_state_oof and not all(os.path.isfile(p_) for p_ in state_files(a.out, "partial_" if only else "", arm, seed)):   # v9.7
                raise SystemExit("--save-state-oof: %s exists (arm %s seed %d is done) but its state read-out does not (made without the flag); "
                                 "write it with --reeval-run of this run into a fresh --out" % (path, arm, seed))
            return
        kind, tgt, wgt = arm_tensors(arm)
        cell_w = kind == "lr" and len(specs.get(arm, ())) > 2 and bool(specs[arm][2])   # v8
        z_gold = torch.zeros((len(gold), C), device=dev)
        z_pool = torch.zeros((len(pool), C), device=dev)
        spec_pool = torch.zeros((len(pool),), device=dev)      # v9.2
        area_pool = torch.zeros((len(pool), C), device=dev)    # v9.3 --save-area
        logs, steps_total = [], 0
        warm_md5 = {}                                         # v9.6: fold -> md5 of the source fold model it started from
        fm_res = {}                                           # v9.8: fold -> final holdout read-out and B5 gate (--fm-aux)
        t_arm = time.time()
        H = a.holdout_fold
        if a.save_state_oof:                            # v9.7: NaN = a row not predicted here; the holdout fold's rows accumulate / (K - 1)
            st_pool = torch.full((len(pool), st_layout["width"]), float("nan"), device=dev)
            if H is not None:
                st_pool[torch.from_numpy(np.flatnonzero(pool_fold == H)).to(dev)] = 0.0
            st_gold = {}                                # fold k -> fold model k's read-out of every gold study
        for k in range(K):
            if only is not None and k not in only:
                print("  fold %d skipped (--only-folds)" % k, flush=True)
                continue
            if H is not None and k == H:
                print("  fold %d is the clean holdout: no model trained for it" % k, flush=True)
                continue
            tr = np.flatnonzero((pool_fold != k) & (pool_fold != H)) if H is not None else np.flatnonzero(pool_fold != k)
            ho = np.flatnonzero(pool_fold == k)
            torch.manual_seed(1000 + 17 * seed)          # same init for every fold: heads start identical, so fold models can be souped
            gen = torch.Generator().manual_seed(5000 + 17 * seed + k)
            mgen = torch.Generator().manual_seed(9400 + 17 * seed + k) if a.mirror > 0 else None   # v9.4: the mirror draws' own stream (crops / jitter stay v9.3's)
            # v9.5: the shift draws' own stream (every other draw stays v9.4's); not 'sgen', which the checkpoint block uses for the sampler
            shgen = torch.Generator().manual_seed(9500 + 17 * seed + k) if a.shift_px > 0 else None
            model = Net(a.backbone, a.init, freeze_blocks=a.freeze_blocks, crop=a.crop, state_spec=state_spec, in_chans=a.slices, spatial=a.spatial_attn, extent=a.extent, syn_decomp=a.syn_decomp, extent_classes=ext_classes,
                        fm_aux=FM_A if fm_store is not None else 0).to(dev).to(memory_format=torch.channels_last)   # v9.8: + the aux_* head with --fm-aux
            torch.manual_seed(1000 + 17 * seed + k)      # fold-specific stream for dropout / drop path / augmentation
            if a.grad_ckpt:                              # 32 GB cards: recompute encoder activations in backward (same maths, ~1.3x compute)
                model.enc.set_grad_checkpointing(True)
            elif a.grad_ckpt_blocks > 0:                 # v9.1: only the first N blocks are recomputed (96 GB card at crop 384: N=6 fits)
                partial_grad_checkpointing(model.enc, a.grad_ckpt_blocks)
            if a.grad_ckpt_blocks > 0 and not a.grad_ckpt:
                print("  v9.1 partial gradient checkpointing: first %d encoder blocks recomputed in backward" % a.grad_ckpt_blocks, flush=True)
            if a.compile:                                # fused kernels; Module.compile keeps state_dict keys (checkpoints, deploy unchanged)
                import torch._dynamo as _dynamo          # aliased: a bare 'import torch...' here would make torch a local name
                _dynamo.config.cache_size_limit = max(32, _dynamo.config.cache_size_limit)
                model.enc.compile(dynamic=True)
            with torch.no_grad():
                model.bias.copy_(bias0)
            if warm_rec is not None:                     # v9.6: the source run's fold-k model, whole and strict, after the bias0 copy (which
                # it overwrites) and before the parameter groups, the optimizer and the EMA exist (the EMA below clones the loaded weights)
                wpath = os.path.join(warm_rec["dir"], "models", "%s_s%d_f%d.pt" % (arm, seed, k))
                n_w, warm_md5[k], warm_chg = warm_load(model, wpath)
                # this run's spec buffers as now set: a --resume below must find exactly these in the checkpoint's weights
                warm_spec = {n_: b_.detach().clone() for n_, b_ in model.named_buffers() if n_ in WARM_SPEC_BUFFERS}
                if warm_chg:                             # the state head's spec buffers now hold this run's spec: per-class change log
                    print("  v9.6 warm-start fold %d: state_prior / state_valid / state_base_logit re-set from this run's spec (%s) after the "
                          "load; per class (%s): max |change| state_prior %s; changed valid slots %s; |change| state_base_logit %s; loaded "
                          "state_gate %s" % (
                              k, os.path.basename(a.state_spec), ", ".join(LABELS),
                              ["%.4g" % v_ for v_ in warm_chg["state_prior"]], warm_chg["state_valid"],
                              ["%.4g" % v_ for v_ in warm_chg["state_base_logit"]],
                              ["%.3g" % v_ for v_ in model.state_gate.detach().float().cpu().tolist()]), flush=True)
            enc = [p for n, p in model.named_parameters() if n.startswith("enc.") and p.requires_grad]
            head = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
            groups = [{"params": enc, "lr": a.lr}]
            vit_mod = model.enc.vit if a.backbone == OFL_BACKBONE else (model.enc if hasattr(model.enc, "blocks") else None)
            if vit_mod is not None and a.llrd != 1.0:
                nb = len(vit_mod.blocks)
                groups, seen = [], set()
                for i, blk in enumerate(vit_mod.blocks):
                    ps = [p for p in blk.parameters() if p.requires_grad]
                    if ps:
                        groups.append({"params": ps, "lr": a.lr * (a.llrd ** (nb - 1 - i))}); seen.update(id(p) for p in ps)
                rest = [p for p in enc if id(p) not in seen]          # final norm (top) and, when unfrozen, embeddings (bottom)
                top = [p for n, p in vit_mod.named_parameters() if n.startswith("norm.") and p.requires_grad]
                bottom = [p for p in rest if id(p) not in {id(q) for q in top}]
                if top:
                    groups.append({"params": top, "lr": a.lr})
                if bottom:
                    groups.append({"params": bottom, "lr": a.lr * (a.llrd ** nb)})
            if k == (sorted(only)[0] if only else 0) and seed == 0:
                print("trainable encoder params %.1fM, head params %.2fM, frozen encoder params %.1fM" % (   # v9.8: the aux_* head counted on its own line
                    sum(p.numel() for p in enc) / 1e6, sum(p.numel() for n, p in model.named_parameters() if not n.startswith(("enc.", "aux_"))) / 1e6,
                    sum(p.numel() for n, p in model.named_parameters() if n.startswith("enc.") and not p.requires_grad) / 1e6), flush=True)
            opt = torch.optim.AdamW(groups + [{"params": head, "lr": a.head_lr}], weight_decay=a.wd)
            if warm_rec is not None:                     # v9.6: every group (encoder, layer-wise groups, head) x s, before the LR schedule
                lr_w = [g_["lr"] for g_ in opt.param_groups]                 # records it as the group's initial LR
                for g_ in opt.param_groups:
                    g_["lr"] = g_["lr"] * a.warm_lr_scale
                print("  v9.6 warm-start %s seed %d fold %d: %s loaded strict (%d tensors, md5 %s); EMA starts from the loaded weights; "
                      "LR of the %d parameter groups x %g: %s -> %s" % (arm, seed, k, wpath, n_w, warm_md5[k], len(lr_w), a.warm_lr_scale,
                                                                      ["%.3g" % v_ for v_ in lr_w], ["%.3g" % g_["lr"] for g_ in opt.param_groups]), flush=True)
            ema = None
            if a.ema > 0:                               # EMA over everything that trains (frozen parts are shared anyway)
                ema = {n_: p_.detach().clone() for n_, p_ in model.named_parameters() if p_.requires_grad}
            ds, dl = make_loader([pool[i] for i in tr], "train", 9000 + 17 * seed + k, True, nonce=a.draw_seed)   # v9.6: + nonce
            total = len(dl) * a.epochs
            warm = max(1, int(.05 * total))
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                                      .5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
            tr_t = torch.from_numpy(tr).to(dev)
            log = []
            fm_rec = fm_hist = fm_pw = fm_hold = None
            if fm_store is not None:                     # v9.8: the fold's auxiliary set-up (the record a checkpoint must match on resume)
                if model.has_bn:
                    raise SystemExit("--fm-aux refused: %s has BatchNorm layers (the auxiliary batch would move their statistics)" % a.backbone)
                if sorted(n_ for n_, _ in model.named_parameters() if n_.startswith("aux_")) != ["aux_b", "aux_ln_b", "aux_ln_w", "aux_w"]:
                    raise AssertionError("v9.8: the auxiliary head must be exactly aux_ln_w, aux_ln_b, aux_w, aux_b")
                fm_rec = {"dir": fm_store.dir, "md5": fm_store.md5, "weight": a.fm_aux_weight, "windows": a.fm_aux_windows, "flip": a.fm_aux_flip,
                          "crop": a.crop, "shift_px": a.shift_px}
                fm_hist = []
                fm_pw = torch.tensor(FM_POS_WEIGHT, device=dev)
                fm_hold = [(vi, w_, k_) for vi in fm_store.holdout for w_ in FM_VIEWS for k_ in range(int(fm_store.n[vi]))]
                print("  v9.8 fm-aux %s seed %d fold %d: %d auxiliary batches of %d windows per epoch (one per step) through model.enc "
                      "(%s); head aux_* %d parameters in the head group (lr %g); holdout read-out %d windows of %d volumes" % (
                          arm, seed, k, len(dl), a.fm_aux_windows, "compiled" if a.compile else "eager",
                          sum(p_.numel() for n_, p_ in model.named_parameters() if n_.startswith("aux_")), a.head_lr, len(fm_hold),
                          len(fm_store.holdout)), flush=True)
            # resumable state, written atomically at the end of every --ckpt-every epochs: weights, optimizer, LR schedule, EMA,
            # and every random stream that decides the next epoch (augmentation, sampler, loader, CPU and GPU global streams)
            ckpt_path = os.path.join(a.out, "ckpt", "%s_s%d_f%d.pt" % (arm, seed, k))
            start_ep = 0
            if a.resume and os.path.exists(ckpt_path):
                ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if ck.get("epochs") != a.epochs or ck.get("train_studies") != len(tr):
                    raise SystemExit("checkpoint %s does not match this run (epochs / training set)" % ckpt_path)
                warm_now = None if warm_rec is None else dict(warm_rec, md5=warm_md5[k])
                if ck.get("warm_start") != warm_now:     # v9.6: a checkpoint continues only the warm start it came from (none = none), and
                    # only while the source fold model is the file it started from (md5)
                    raise SystemExit("checkpoint %s does not match this run (warm start %s vs this run %s)" % (ckpt_path, ck.get("warm_start"), warm_now))
                for key_, on_ in (("aug_grid_fp32", a.aug_grid_fp32), ("draw_seed", a.draw_seed), ("rare_boost_names", a.rare_boost_names)):
                    if bool(ck.get(key_, False)) != on_:   # v9.6: one grid precision / draw seeding / boost rule per fold
                        raise SystemExit("checkpoint %s does not match this run (--%s %s vs this run %s)" % (
                            ckpt_path, key_.replace("_", "-"), bool(ck.get(key_, False)), on_))
                if ck.get("fm_aux") != fm_rec:            # v9.8: a fold continues only the auxiliary set-up it started with (none = none)
                    raise SystemExit("checkpoint %s does not match this run (--fm-aux %s vs this run %s)" % (ckpt_path, ck.get("fm_aux"), fm_rec))
                if fm_rec is not None:
                    fm_hist = list(ck.get("fm_hist") or [])
                model.load_state_dict(ck["model"])
                if warm_rec is not None:                 # v9.6: the resumed weights must carry this run's spec buffers (the checkpoint's
                    bufs_ = dict(model.named_buffers())  # would otherwise silently replace the spec this run was launched with)
                    stale = [n_ for n_, v_ in warm_spec.items() if not torch.equal(bufs_[n_], v_)]
                    if stale:
                        raise SystemExit("checkpoint %s holds other %s than this run's state spec (%s) builds: it was written under another "
                                         "spec; resume with that spec or restart the fold" % (ckpt_path, stale, a.state_spec))
                opt.load_state_dict(ck["opt"])
                sched.load_state_dict(ck["sched"])
                if ema is not None:
                    ema = {n_: v.to(dev) for n_, v in ck["ema"].items()}
                gen.set_state(ck["gen"])
                if ck.get("sampler_gen") is not None:
                    (dl.batch_sampler.generator if isinstance(dl.batch_sampler, PairBatchSampler) else dl.sampler.generator).set_state(ck["sampler_gen"])
                if ck.get("loader_gen") is not None:
                    dl.generator.set_state(ck["loader_gen"])
                if mgen is not None and ck.get("mirror_gen") is not None:   # v9.4: the mirror stream continues where it stopped
                    mgen.set_state(ck["mirror_gen"])
                if shgen is not None and ck.get("shift_gen") is not None:   # v9.5: the shift stream continues where it stopped
                    shgen.set_state(ck["shift_gen"])
                torch.set_rng_state(ck["cpu_rng"])
                torch.cuda.set_rng_state_all(ck["cuda_rng"])
                log = list(ck["log"])
                start_ep = int(ck["epoch"]) + 1
                print("  resumed %s seed %d fold %d after epoch %d (%s)" % (arm, seed, k, ck["epoch"], ckpt_path), flush=True)
            for ep in range(start_ep, a.epochs):
                ds.epoch = ep
                model.train()
                tot = n = 0
                pr_batches = pr_terms = 0
                mir_batches = 0                             # v9.4
                shift_ep = []                               # v9.5: the epoch's (dy, dx) draws (read-out only)
                ext_acc = torch.zeros(8, device=dev)        # v9.3: + the high_grade_small cells' mean log1p(A/100) (entries 6, 7)
                if fm_store is not None:                    # v9.8: this epoch's auxiliary batches: fresh generators (seeded from seed, fold,
                    # epoch), one batch per step, own loader; built inside fork_rng (the global CPU stream is not consumed)
                    with torch.random.fork_rng(devices=[]):
                        fm_seeds = fm_epoch_seeds(seed, k, ep)
                        fm_samp = FmAuxSampler(fm_store, len(dl), a.fm_aux_windows, a.crop, a.shift_px, a.fm_aux_flip, fm_seeds)
                        fm_it = iter(fm_loader(fm_samp, fm_seeds[1] + 1))
                    fm_acc = torch.zeros(3, device=dev)
                    fm_b = 0
                for img, meta, mask, idx in dl:
                    sel = tr_t[idx.to(dev)]
                    with torch.autocast("cuda", dtype=torch.bfloat16):   # augment stays inside the autocast region, as in v9.3 (aug v2's grid maths)
                        img_d, meta_d = augment(img.to(dev, non_blocking=True), gen, shgen, shift_ep), meta.to(dev)   # v9.5: + the shift stream
                        # v9.4: one draw per batch (both studies, so both halves of a matched pair, share it): the virtual contralateral
                        # knee goes through the same forward; the outputs are re-indexed to the canonical class order before every loss
                        flip = mgen is not None and float(torch.rand(1, generator=mgen)) < a.mirror
                        if flip:
                            img_d, meta_d = mirror_batch(img_d, meta_d)
                            mir_batches += 1
                        z, sl, bl = model(img_d, meta_d, mask.to(dev), return_aux=True)
                    area = model._ext_area                                          # [B, C] with --extent, else None
                    w_main = wgt[sel] if (kind == "bce" or cell_w) else None
                    sti_b, stw_b, bint_b = (st_idx[sel], st_w[sel], bin_t[sel]) if state_spec is not None else (None, None, None)
                    if flip:
                        z, sl, bl, area, w_main, sti_b, stw_b, bint_b = mirror_outputs(z, sl, bl, area, w_main, sti_b, stw_b, bint_b,
                                                                                       mir_perm, mir_bin_perm, mir_mcl_bins)
                    loss = bce_soft(z, tgt[sel], w_main) if kind == "bce" else lr_loss(z, tgt[sel], w_main)
                    if state_spec is not None:
                        loss = loss + a.state_weight * state_loss(sl, bl, sti_b, stw_b, bint_b)
                    if a.extent:
                        ext_l = extent_loss(area, sel)
                        if a.extent_weight > 0:
                            loss = loss + a.extent_weight * ext_l
                        with torch.no_grad():                     # epoch log of the extent path (audit 6), no sync per step
                            la_, y_ = torch.log1p(area.float() / 100.0), ext_y[sel]
                            ext_acc += torch.stack([(la_ * (y_ == 1)).sum(), (y_ == 1).sum().float(), (la_ * (y_ == 0)).sum(),
                                                    (y_ == 0).sum().float(), ext_l.detach().float(), torch.ones((), device=dev),
                                                    (la_ * (y_ == OA_SMALL_TARGET)).sum(), (y_ == OA_SMALL_TARGET).sum().float()])
                    if a.pair_rank > 0:
                        pl, pc = pair_loss(z, sel, area if a.oa_area_rank else None, stw_b)   # v9.3: area families read the extent head
                        loss = loss + a.pair_rank * pl
                        pr_batches += int(pc > 0); pr_terms += pc
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    if fm_store is not None:                # v9.8: the auxiliary step, after the main backward, into the same .grad
                        with torch.random.fork_rng(devices=[]):   # the fetch (loader iterator) never touches the global CPU stream
                            fm_batch = next(fm_it)
                        fm_ld, fm_lp = fm_step(model, fm_batch, img_d, enc, seed, k, ep, fm_b, len(dl), fm_acc, fm_pw)
                        fm_b += 1
                    if a.forward_only:
                        g = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
                        print("forward-only check: loss %.4f, gradient mass %.3e, windows %d, optimizer steps 0" % (
                            float(loss), g, int(mask.sum())), flush=True)
                        if fm_store is not None:
                            print("forward-only check v9.8 fm-aux: L_dense %.4f L_presence %.4f (lambda %g), %d auxiliary windows; the "
                                  "gradient mass above includes lambda * grad L_aux" % (float(fm_ld), float(fm_lp), a.fm_aux_weight,
                                                                                         a.fm_aux_windows), flush=True)
                        return
                    if fm_store is None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    else:                                   # v9.8: one clip of the merged gradient (v9.7's when lambda = 0)
                        fm_clip(model, 5.)
                    opt.step()
                    sched.step()
                    if ema is not None:
                        with torch.no_grad():
                            d_ = min(a.ema, (1 + sched.last_epoch) / (10 + sched.last_epoch))   # warm EMA early on
                            for n_, p_ in model.named_parameters():
                                if n_ in ema:
                                    ema[n_].mul_(d_).add_(p_.detach(), alpha=1 - d_)
                    tot += float(loss.detach()) * len(idx)
                    n += len(idx)
                log.append(round(tot / max(1, n), 5))
                print("  %s seed %d fold %d epoch %d loss %.4f  %.0fs" % (arm, seed, k, ep, log[-1], time.time() - t_arm), flush=True)
                if a.syn_decomp:                         # v9.2 read-out (audit N1): beta live and EMA, every epoch
                    print("  syn-decomp epoch %d: beta %.4f (ema %s)" % (ep, float(model.syn_beta.detach().cpu()),
                          ("%.4f" % float(ema["syn_beta"].cpu())) if ema is not None and "syn_beta" in ema else "n/a"), flush=True)
                if a.extent:
                    ea = ext_acc.tolist()
                    print("  extent epoch %d: auxiliary loss %.4f (weight %.2f) | gamma MOA/LOA/PFOA %s | mean log1p(A/100) on >= 1 cm^2 stated %.3f vs normal/low grade %.3f" % (
                        ep, ea[4] / max(1.0, ea[5]), a.extent_weight,
                        [round(float(model.ext_gamma[LABELS.index(l)]), 3) for l in ("Medial OA", "Lateral OA", "PF OA")],
                        ea[0] / max(1.0, ea[1]), ea[2] / max(1.0, ea[3])), flush=True)
                    if a.oa_area_rank:
                        print("  extent v9.3 epoch %d: mean log1p(A/100) on high_grade_small (target %.1f) %.3f over %d cells" % (
                            ep, OA_SMALL_TARGET, ea[6] / max(1.0, ea[7]), int(ea[7])), flush=True)
                if a.pair_rank > 0:
                    print("  pair-rank epoch %d: %d of %d batches had at least one matched-pair term (%d class terms; includes base batches that happened to pair)" % (ep, pr_batches, len(dl), pr_terms), flush=True)
                if a.mirror > 0:                         # v9.4 read-out: the mirrored share of the epoch's batches
                    print("  mirror epoch %d: %d of %d batches mirrored (p %.2f)" % (ep, mir_batches, len(dl), a.mirror), flush=True)
                if a.shift_px > 0:                       # v9.5 read-out: the epoch's batch translations
                    dys, dxs = [sh[0] for sh in shift_ep] or [0], [sh[1] for sh in shift_ep] or [0]
                    print("  shift epoch %d: %d of %d batches translated (N %d): dy %d..%d, dx %d..%d, mean |dy| %.2f |dx| %.2f" % (
                        ep, len(shift_ep), len(dl), a.shift_px, min(dys), max(dys), min(dxs), max(dxs),
                        sum(abs(v) for v in dys) / len(dys), sum(abs(v) for v in dxs) / len(dxs)), flush=True)
                if fm_store is not None:                 # v9.8 read-out: the epoch's auxiliary losses, sample digest, holdout AUC, compiles
                    with torch.random.fork_rng(devices=[]):
                        fm_left = next(fm_it, None)
                    if fm_left is not None or fm_samp.yielded != len(dl) or fm_b != len(dl):
                        raise AssertionError("v9.8: epoch %d used %d auxiliary batches, the sampler yielded %d, the loader had more: %s" % (
                            ep, fm_b, fm_samp.yielded, fm_left is not None))
                    fa = fm_acc.tolist()
                    ev_ = fm_eval(model, fm_hold)
                    cmp_ = dynamo_compiles() if a.compile else None
                    fm_hist.append({"epoch": ep, "l_dense": fa[0] / max(1.0, fa[2]), "l_presence": fa[1] / max(1.0, fa[2]), "steps": int(fa[2]),
                                    "digest": fm_samp.digest(), "holdout": ev_, "dynamo": cmp_})
                    print("  fm-aux epoch %d: L_dense %.5f L_presence %.5f over %d steps (lambda %g); sample digest %s (seeds %s)" % (
                        ep, fm_hist[-1]["l_dense"], fm_hist[-1]["l_presence"], int(fa[2]), a.fm_aux_weight, fm_samp.digest(),
                        list(fm_seeds)), flush=True)
                    print("  fm-aux epoch %d holdout presence AUC (live weights; %d windows / %d volumes), window level: %s | volume level: %s" % (
                        ep, ev_["windows"], ev_["volumes"], fm_auc_str(ev_["window_auc"]), fm_auc_str(ev_["volume_auc"])), flush=True)
                    if cmp_ is not None:
                        print("  fm-aux epoch %d dynamo: %d frames compiled, %d recompiles (test 9: <= 4 at the end of epoch 0, no growth after)" % (
                            ep, cmp_[0], cmp_[1]), flush=True)
                if a.ckpt_every > 0 and (ep + 1) % a.ckpt_every == 0 and ep + 1 < a.epochs:
                    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                    sgen = getattr(dl.sampler, "generator", None) if isinstance(dl.sampler, torch.utils.data.WeightedRandomSampler) else None
                    if isinstance(dl.batch_sampler, PairBatchSampler):
                        sgen = dl.batch_sampler.generator
                    ck_ = {"epoch": ep, "epochs": a.epochs, "train_studies": len(tr), "log": log,
                           "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                           "ema": None if ema is None else {n_: v.detach().cpu() for n_, v in ema.items()},
                           "gen": gen.get_state(), "sampler_gen": None if sgen is None else sgen.get_state(),
                           "loader_gen": None if dl.generator is None else dl.generator.get_state(),
                           "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}
                    if mgen is not None:                 # v9.4: the key exists only with --mirror (the checkpoint is otherwise v9.3's)
                        ck_["mirror_gen"] = mgen.get_state()
                    if shgen is not None:                # v9.5: the key exists only with --shift-px > 0 (otherwise v9.4's key set)
                        ck_["shift_gen"] = shgen.get_state()
                    if warm_rec is not None:             # v9.6: the key exists only with --warm-start (otherwise v9.5's key set)
                        ck_["warm_start"] = dict(warm_rec, md5=warm_md5[k])
                    for key_, on_ in (("aug_grid_fp32", a.aug_grid_fp32), ("draw_seed", a.draw_seed), ("rare_boost_names", a.rare_boost_names)):
                        if on_:                          # v9.6: each key exists only with its flag (otherwise v9.5's key set)
                            ck_[key_] = True
                    if fm_rec is not None:               # v9.8: the keys exist only with --fm-aux (otherwise v9.7's key set); the auxiliary
                        ck_["fm_aux"] = fm_rec           # generators restart from their epoch seeds, so they need no state here
                        ck_["fm_hist"] = fm_hist
                    torch.save(ck_, ckpt_path + ".tmp")
                    os.replace(ckpt_path + ".tmp", ckpt_path)
                if a.stop_after_epoch >= 0 and ep == a.stop_after_epoch:
                    print("  stop-after-epoch %d: exiting (test)" % ep, flush=True)
                    sys.exit(3)
            steps_total += total
            if ema is not None:                         # evaluate and ship the EMA weights
                with torch.no_grad():
                    for n_, p_ in model.named_parameters():
                        if n_ in ema:
                            p_.copy_(ema[n_])
                if model.has_bn and a.bn_recal_batches > 0:
                    b_, w_ = bn_recalibrate(model, [pool[i] for i in tr], 9100 + 17 * seed + k, a.bn_recal_batches)
                    print("  BatchNorm statistics recalibrated for the EMA weights: %d batches, %d windows" % (b_, w_), flush=True)
                elif model.has_bn:
                    print("  WARNING: BatchNorm backbone with --ema and no --bn-recal-batches: the saved model pairs EMA weights with "
                          "the live weights' BatchNorm statistics (see the 10-08 CoAtNet screen)", flush=True)
            if fm_store is not None:                     # v9.8: the B5 diagnostic with the shipped weights (EMA when --ema), written per fold
                ev_ = fm_eval(model, fm_hold)
                gate = {c_: (ev_["window_auc"][c_], thr_, bool(ev_["window_auc"][c_] >= thr_)) for c_, thr_ in FM_GATE}
                fm_res[k] = {"final_holdout": ev_, "gate": gate, "gate_pass": all(g_[2] for g_ in gate.values())}
                print("  fm-aux fold %d final holdout presence AUC (%s weights; %d windows / %d volumes), window level: %s | volume level: %s" % (
                    k, "EMA" if ema is not None else "live", ev_["windows"], ev_["volumes"], fm_auc_str(ev_["window_auc"]),
                    fm_auc_str(ev_["volume_auc"])), flush=True)
                print("  fm-aux fold %d B5 gate (the head learned; diagnostic, not a mechanism verdict): %s -> %s" % (
                    k, ", ".join("%s %.3f >= %.2f %s" % (c_, v_[0], v_[1], "ok" if v_[2] else "NOT MET") for c_, v_ in gate.items()),
                    "PASS" if fm_res[k]["gate_pass"] else "FAIL (stop and check the implementation / data; never read as 'mechanism "
                                                         "without effect')"), flush=True)
                json.dump({"arm": arm, "seed": seed, "fold": k, "kind": "v9.8 --fm-aux auxiliary-task log", "store": fm_rec,
                           "classes": list(FM_CLASSES), "epochs": fm_hist, "final_holdout": ev_, "final_weights": "EMA" if ema is not None else "live",
                           "gate": gate, "gate_pass": fm_res[k]["gate_pass"], "pos_weight": FM_POS_WEIGHT, "presence_min_area": FM_PRESENCE_MIN,
                           "volumes_per_step": FM_VOLS_PER_STEP, "boxed_volume_share": FM_BOXED_FRAC},
                          open(os.path.join(a.out, "fm_aux_%s_s%d_f%d.json" % (arm, seed, k)), "w"), indent=1)
            os.makedirs(os.path.join(a.out, "models"), exist_ok=True)
            torch.save(model.deploy_state_dict(),
                       os.path.join(a.out, "models", "%s_s%d_f%d.pt" % (arm, seed, k)))
            if os.path.exists(ckpt_path):                 # the fold is complete: its resumable state is no longer needed
                os.remove(ckpt_path)
            if a.save_state_oof:                        # v9.7: the same gold forward, plus fold model k's state read-out
                zg_, sg_ = predict(model, gold, seed, want_state=True)
                z_gold += zg_ / (K if H is None else K - 1)
                st_gold[k] = sg_.cpu().numpy()
            else:
                z_gold += predict(model, gold, seed) / (K if H is None else K - 1)
            if a.save_state_oof:                        # v9.7: the same held-out forward, plus the state read-out
                zp_, sp_, ar_, ps_ = predict(model, [pool[i] for i in ho], seed, want_spec=True, want_area=True, want_state=True)
                st_pool[torch.from_numpy(ho).to(dev)] = ps_
            else:
                zp_, sp_, ar_ = predict(model, [pool[i] for i in ho], seed, want_spec=True, want_area=True)
            z_pool[torch.from_numpy(ho).to(dev)] = zp_
            if a.syn_decomp:
                spec_pool[torch.from_numpy(ho).to(dev)] = sp_
            if a.save_area:
                area_pool[torch.from_numpy(ho).to(dev)] = ar_
            if H is not None:
                hh = np.flatnonzero(pool_fold == H)
                if a.save_state_oof:                    # v9.7: as below, plus each fold model's state read-out / (K - 1)
                    res_ = predict(model, [pool[i] for i in hh], seed, want_area=a.save_area, want_state=True)
                    z_pool[torch.from_numpy(hh).to(dev)] += res_[0] / (K - 1)
                    if a.save_area:
                        area_pool[torch.from_numpy(hh).to(dev)] += res_[1] / (K - 1)
                    st_pool[torch.from_numpy(hh).to(dev)] += res_[-1] / (K - 1)
                elif a.save_area:                       # v9.3: same predictions, plus the area proxy
                    zh_, ah_ = predict(model, [pool[i] for i in hh], seed, want_area=True)
                    z_pool[torch.from_numpy(hh).to(dev)] += zh_ / (K - 1)
                    area_pool[torch.from_numpy(hh).to(dev)] += ah_ / (K - 1)
                else:
                    z_pool[torch.from_numpy(hh).to(dev)] += predict(model, [pool[i] for i in hh], seed) / (K - 1)
            if fm_store is not None and a.compile:        # v9.8 (test 9): the dynamo count once the fold's predictions are made too
                cmp_ = dynamo_compiles()
                print("  fm-aux fold %d dynamo after the fold's predictions: %d frames compiled, %d recompiles" % (k, cmp_[0], cmp_[1]), flush=True)
            logs.append([log[0], log[-1]])
            del model, opt
            torch.cuda.empty_cache()
        np.save(os.path.join(a.out, ("partial_" if only else "") + "oof_%s_s%d.npy" % (arm, seed)), z_pool.cpu().numpy().astype(np.float32))
        if a.syn_decomp:                                     # v9.2 read-out: the synovium-specific logit per pool study
            np.save(os.path.join(a.out, ("partial_" if only else "") + "oof_synspec_%s_s%d.npy" % (arm, seed)), spec_pool.cpu().numpy().astype(np.float32))
        if a.save_area:                                      # v9.3 read-out: the extent head's area proxy per pool study
            np.save(os.path.join(a.out, ("partial_" if only else "") + "oof_area_%s_s%d.npy" % (arm, seed)), area_pool.cpu().numpy().astype(np.float32))
        if a.save_state_oof:                                 # v9.7 read-out, written before the preds json (the done marker)
            write_state_readout(a.out, "partial_" if only else "", arm, seed, st_layout, a.state_spec, len(pool), st_pool.cpu().numpy(), gold,
                                [int(by_uid[u]["fold"]) for u in gold], st_gold, K, H, only, None)
        out = {"arm": arm, "seed": seed, "epochs": a.epochs, "nested": False, "gold_uids": gold, "only_folds": sorted(only) if only else None,
               "holdout_fold": H,
               "gold_fold": [int(by_uid[u]["fold"]) for u in gold], "labels": LABELS,
               "gold_logits": [[round(float(x), 5) for x in r] for r in z_gold.cpu().numpy()],
               "loss_first_last": logs, "optimizer_steps": steps_total, "train_seconds": round(time.time() - t_arm, 1),
               "receipt": [{"arm": arm, "kind": kind, "pool_studies": len(pool), "gold_in_training": 0,
                            "init_seed": 1000 + 17 * seed, "note": "fold models share one initialisation (soup-compatible)"}]}
        if warm_rec is not None:                             # v9.6: provenance of every fold model's starting point
            out["receipt"][0].update(warm_start=warm_rec["dir"], warm_lr_scale=warm_rec["lr_scale"],
                                     warm_state_spec_md5=warm_rec["state_spec_md5"],
                                     warm_fold_md5={str(k_): v_ for k_, v_ in sorted(warm_md5.items())},
                                     note="v9.6 --warm-start: fold k continued from the source run's fold-k model; optimizer, LR schedule and EMA fresh")
        if fm_store is not None:                             # v9.8: the auxiliary task's provenance and the per-fold B5 gate
            out["receipt"][0]["fm_aux"] = {"store": fm_store.dir, "md5": fm_store.md5, "weight": a.fm_aux_weight, "windows": a.fm_aux_windows,
                                           "flip": a.fm_aux_flip, "crop": a.crop, "shift_px": a.shift_px,
                                           "gate_pass": {str(k_): v_["gate_pass"] for k_, v_ in sorted(fm_res.items())},
                                           "note": "v9.8 --fm-aux: fastMRI+ lesion-box auxiliary task on the shared encoder (training only; the fold "
                                                   "models carry no aux_* key); per-fold log fm_aux_<arm>_s<seed>_f<k>.json"}
        json.dump(out, open(path, "w"))

    if not a.reeval_run:                                     # v9.8 E2: the arms' targets are checked before run_config.json is written
        for arm_ in [x.split("=", 1)[0] for x in a.arms.split(",") if x]:   # (v9.7 checked them in run(), after writing it)
            try:
                arm_tensors(arm_)
            except (AssertionError, KeyError, IndexError, ValueError) as e:
                raise SystemExit("arm %s refused before anything was written: its targets fail the training checks (%s: %s)" % (
                    arm_, type(e).__name__, e))
    cfg = {k: getattr(a, k) for k in vars(a)}
    cfg.update({"pool_studies": len(pool), "gold_studies": len(gold), "feature_studies": len(done), "labels": LABELS,
                "scaffold": "compact fine-tuned image model on the pixel cache"})
    json.dump(cfg, open(os.path.join(a.out, "run_config.json"), "w"), indent=1)
    json.dump(pool, open(os.path.join(a.out, "pool_uids.json"), "w"))
    for seed in range(a.seed_base, a.seed_base + a.seeds):
        for arm in [x.split("=", 1)[0] for x in a.arms.split(",") if x]:
            run(arm, seed)
            if a.forward_only:
                return
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
