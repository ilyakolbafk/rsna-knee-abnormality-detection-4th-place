"""Public FM dataset/sampler with one explicit pixel-reader assignment adapted."""
import os,json,hashlib,ast,inspect,textwrap
from pathlib import Path
import numpy as np
import torch
from fm_loader_reference import *
from fm_chunk_array import ChunkArray

FM_VOLS_PER_STEP=4
FM_BOXED_FRAC=0.75
class FmAuxWindows(torch.utils.data.Dataset):
    """v9.8: one auxiliary window per index; the index IS the window's spec (volume, view, slice, oy, ox, dy, dx, flip, gamma, contrast)
    drawn in the main process by FmAuxSampler (or the fixed holdout list), so the workers draw nothing.  Returns (uint8 [3, crop, crop],
    dense [A, G, G], presence [A], gamma, contrast); gamma / contrast are applied on the GPU (fm_photometric)."""

    def __init__(self, store, crop, shift):
        self.st, self.crop, self.shift = (store, int(crop), int(shift))
        self.px = None

    def __len__(self):
        return 1

    def __getitem__(self, spec):
        if self.px is None:
            self.px = np.load(os.path.join(self.st.dir, 'px.u8.npy'), mmap_mode='r')
        vi, view, k, oy, ox, dy, dx, fl, gam, con = spec
        rows = self.st.window_rows(vi, view, k)
        img = fm_window_image(self.px, rows, self.crop, oy, ox, dy, dx, self.shift, fl)
        dense, pres, _ = fm_window_targets(self.st.boxes_of(rows), self.crop, oy, ox, dy, dx, self.shift, fl)
        return (torch.from_numpy(img), torch.from_numpy(dense), torch.from_numpy(pres), float(gam), float(con))

def fm_collate(batch):
    return (torch.stack([b_[0] for b_ in batch]), torch.stack([b_[1] for b_ in batch]), torch.stack([b_[2] for b_ in batch]), torch.tensor([b_[3] for b_ in batch], dtype=torch.float32), torch.tensor([b_[4] for b_ in batch], dtype=torch.float32))

def fm_epoch_seeds(seed, fold, epoch):
    """B3: the auxiliary streams of one (seed, fold, epoch): sampling 9800 + 17*seed + fold + 1000*epoch, augmentation the same + 2**32 (two
    generators with one seed would emit the same numbers); fresh every epoch, so a resume at an epoch boundary repeats the batches."""
    s_ = 9800 + 17 * int(seed) + int(fold) + 1000 * int(epoch)
    return (s_, s_ + (1 << 32))

def fm_fork_seed(seed, fold, epoch, step):
    """B3: the global-stream seed inside the auxiliary step's fork_rng (drop path of the auxiliary forward), f(seed, fold, epoch, step)"""
    return int(hashlib.sha1(('v9.8 fm-aux %d %d %d %d' % (seed, fold, epoch, step)).encode()).hexdigest()[:15], 16)

class FmAuxSampler:
    """v9.8: one epoch of auxiliary batches (batch_sampler of FmAuxWindows): `steps` lists of M window specs, drawn in the main process
    from two private generators.  Per step: FM_VOLS_PER_STEP slots; per slot (sampling stream) boxed = rand < 0.75, a volume uniform among
    the training volumes with boxes (boxed) or without (otherwise), then M / 4 centre slices -- boxed: the first half uniform over the
    volume's slices with a box (any class), the second half over its slices without one (all slices when it has none) -- and a view
    uniform in {0, 2, 3} per window; then (augmentation stream) per window the crop offset oy, ox uniform on 0..384-crop, the translation
    dy, dx uniform on -N..N (only when N = --shift-px > 0), flip = rand < --fm-aux-flip, gamma U(0.8, 1.25), contrast U(0.9, 1.1)
    (augment()'s ranges).  `digest` = sha1 over every spec yielded (the epoch's sequence; test 7)."""

    def __init__(self, store, steps, M, crop, shift, flip_p, seeds):
        self.st, self.steps, self.M, self.crop, self.shift, self.flip_p = (store, int(steps), int(M), int(crop), int(shift), float(flip_p))
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
        st, gs, ga, M = (self.st, self.gs, self.ga, self.M)
        per = M // FM_VOLS_PER_STEP
        base = []
        for _ in range(FM_VOLS_PER_STEP):
            boxed = float(torch.rand(1, generator=gs)) < FM_BOXED_FRAC
            pool = st.train_boxed if boxed else st.train_neg
            vi = pool[int(torch.randint(0, len(pool), (1,), generator=gs))]
            if boxed:
                pos, neg = (st.boxed_slices[vi], st.free_slices[vi])
                if not len(neg):
                    neg = np.arange(int(st.n[vi]))
                ks = np.concatenate([pos[torch.randint(0, len(pos), (per // 2,), generator=gs).numpy()], neg[torch.randint(0, len(neg), (per - per // 2,), generator=gs).numpy()]])
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
    x = (img.float() / 255.0).clamp_(0, 1).pow(gam.view(-1, 1, 1, 1))
    x = ((x - 0.5) * con.view(-1, 1, 1, 1) + 0.5).clamp_(0, 1) * 255.0
    return x.round_().to(torch.uint8)

def fm_make_loader(store, batches, crop, shift, workers, seed_w):
    """v9.8: a DataLoader over FmAuxWindows whose batch_sampler is `batches` (FmAuxSampler, or a fixed list of spec lists for the holdout
    read-out with shift 0); its own workers and its own generator (never the global stream, never the main loader's generator)"""
    return torch.utils.data.DataLoader(FmAuxWindows(store, crop, shift), batch_sampler=batches, num_workers=workers, collate_fn=fm_collate, pin_memory=True, prefetch_factor=2 if workers > 0 else None, persistent_workers=False, generator=torch.Generator().manual_seed(int(seed_w)))
