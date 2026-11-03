"""AST-exact public FM primitives with all three original views."""
import math,hashlib
import numpy as np
import torch
FM_A=11
FM_CELL=32
FM_SIZE=384
FM_PRESENCE_MIN=.25
FM_VOLS_PER_STEP=4
FM_BOXED_FRAC=.75
FM_VIEWS=(0,2,3)

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
