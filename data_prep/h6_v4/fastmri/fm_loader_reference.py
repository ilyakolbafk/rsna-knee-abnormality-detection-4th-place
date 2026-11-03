"""Unmodified AST-extracted public loader/target QA only. No model or training entry point."""
import os,json,hashlib,math,collections
import numpy as np
FM_CLASSES = ('meniscus_tear', 'cart_partial', 'cart_full', 'bone_edema', 'bone_fx_contusion', 'effusion', 'cyst', 'acl_high', 'acl_low', 'collateral', 'other')

FM_A = len(FM_CLASSES)

FM_VIEWS = (0, 2, 3)

FM_SIZE = 384

FM_CELL = 32

FM_PRESENCE_MIN = 0.25

def file_md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 24), b''):
            h.update(blk)
    return h.hexdigest()

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
        need = ('px.u8.npy', 'index.json', 'boxes.json', 'manifest.json')
        miss = [f_ for f_ in need if not os.path.isfile(os.path.join(self.dir, f_))]
        if miss:
            raise SystemExit('--fm-aux %s: missing %s' % (self.dir, miss))
        self.md5 = {f_: file_md5(os.path.join(self.dir, f_)) for f_ in ('index.json', 'boxes.json', 'manifest.json')}
        index = json.load(open(os.path.join(self.dir, 'index.json'), encoding='utf-8'))
        boxes = json.load(open(os.path.join(self.dir, 'boxes.json'), encoding='utf-8'))
        self.manifest = json.load(open(os.path.join(self.dir, 'manifest.json'), encoding='utf-8'))
        px = np.load(os.path.join(self.dir, 'px.u8.npy'), mmap_mode='r')
        bad = []
        if index.get('format') != 'fm384_v1' or boxes.get('format') != 'fm384_v1':
            bad.append('format tags %r / %r (want fm384_v1)' % (index.get('format'), boxes.get('format')))
        if index.get('size') != FM_SIZE:
            bad.append('size %r (want %d)' % (index.get('size'), FM_SIZE))
        if list(index.get('classes') or []) != list(FM_CLASSES):
            bad.append('classes %r (want %r)' % (index.get('classes'), list(FM_CLASSES)))
        if px.dtype != np.uint8 or px.ndim != 3 or tuple(px.shape[1:]) != (FM_SIZE, FM_SIZE):
            bad.append('px.u8.npy %s %s (want uint8 [N, %d, %d])' % (px.dtype, tuple(px.shape), FM_SIZE, FM_SIZE))
        N = int(px.shape[0])
        self.n_rows = N
        if index.get('n_rows') != N or boxes.get('n_rows') != N:
            bad.append('n_rows index %r / boxes %r vs px rows %d' % (index.get('n_rows'), boxes.get('n_rows'), N))
        if bad:
            raise SystemExit('--fm-aux %s refused: %s' % (self.dir, ' | '.join(bad)))
        vols_all = index.get('volumes') or {}
        self.excluded = sorted((v for v, r in vols_all.items() if r.get('excluded')))
        self.vols = sorted((v for v, r in vols_all.items() if not r.get('excluded')))
        self.n, self.start, self.pid, self.hold = ([], {}, [], [])
        spans = []
        for vi, v in enumerate(self.vols):
            r = vols_all[v]
            n_ = r.get('n_slices')
            views = r.get('views')
            if not isinstance(n_, int) or n_ < 1:
                bad.append('%s: n_slices %r' % (v, n_))
                continue
            try:
                vd = {int(a_): int(b_) for a_, b_ in views}
            except (TypeError, ValueError):
                bad.append('%s: views %r (want [[0, start], [2, start], [3, start]])' % (v, views))
                continue
            if sorted(vd) != sorted(FM_VIEWS) or len(views) != len(FM_VIEWS):
                bad.append('%s: views %s (want exactly %s)' % (v, sorted(vd), list(FM_VIEWS)))
                continue
            for w_, s_ in vd.items():
                if s_ < 0 or s_ + n_ > N:
                    bad.append('%s view %d: rows %d..%d outside 0..%d' % (v, w_, s_, s_ + n_ - 1, N - 1))
                self.start[vi, w_] = s_
                spans.append((s_, s_ + n_, v, w_))
            pid_ = r.get('patient_id')
            if not isinstance(pid_, str) or not pid_:
                bad.append('%s: patient_id %r' % (v, pid_))
            if not isinstance(r.get('aux_holdout'), bool):
                bad.append('%s: aux_holdout %r (want true / false)' % (v, r.get('aux_holdout')))
            self.n.append(int(n_))
            self.pid.append(pid_)
            self.hold.append(bool(r.get('aux_holdout')))
            if len(bad) > 20:
                break
        spans.sort()
        for (a0_, a1_, va, wa), (b0_, b1_, vb, wb) in zip(spans, spans[1:]):
            if b0_ < a1_:
                bad.append('row blocks overlap: %s view %d rows %d..%d and %s view %d rows %d..%d' % (va, wa, a0_, a1_ - 1, vb, wb, b0_, b1_ - 1))
                break
        if bad:
            raise SystemExit('--fm-aux %s refused: %s' % (self.dir, ' | '.join(bad[:20])))
        self.n = np.array(self.n, np.int64)
        self.hold = np.array(self.hold, bool)
        row_vol = np.full(N, -1, np.int64)
        row_k = np.full(N, -1, np.int64)
        for (vi, w_), s_ in self.start.items():
            row_vol[s_:s_ + self.n[vi]] = vi
            row_k[s_:s_ + self.n[vi]] = np.arange(self.n[vi])
        self.row_boxes = {}
        per_cls = np.zeros((2, FM_A), np.int64)
        boxed_k = [set() for _ in self.vols]
        n_box = n_skip = 0
        for key, lst in (boxes.get('boxes') or {}).items():
            try:
                row = int(key)
            except ValueError:
                bad.append('boxes: row key %r' % key)
                continue
            if not 0 <= row < N:
                bad.append('boxes: row %d outside 0..%d' % (row, N - 1))
                continue
            vi = int(row_vol[row])
            if vi < 0:
                n_skip += 1
                continue
            if isinstance(lst, list) and (not lst):
                continue
            arr = np.asarray(lst, np.float64)
            if arr.ndim != 2 or arr.shape[1] != 6:
                bad.append('boxes: row %d entry shape %s (want [n, 6] = class, x0, y0, x1, y1, a0)' % (row, arr.shape))
                continue
            c_ = arr[:, 0]
            ok = (c_ == np.round(c_)) & (c_ >= 0) & (c_ < FM_A)
            ok &= (arr[:, 1] >= 0) & (arr[:, 1] < arr[:, 3]) & (arr[:, 3] <= FM_SIZE) & (arr[:, 2] >= 0) & (arr[:, 2] < arr[:, 4]) & (arr[:, 4] <= FM_SIZE)
            ok &= arr[:, 5] >= (arr[:, 3] - arr[:, 1]) * (arr[:, 4] - arr[:, 2]) * 0.99 - 1.0
            if not ok.all():
                bad.append('boxes: row %d has malformed boxes %s' % (row, arr[~ok][:3].tolist()))
                continue
            if len(arr):
                self.row_boxes[row] = arr
                boxed_k[vi].add(int(row_k[row]))
                n_box += len(arr)
                np.add.at(per_cls[int(self.hold[vi])], c_.astype(np.int64), 1)
            if len(bad) > 20:
                break
        if bad:
            raise SystemExit('--fm-aux %s refused: %s' % (self.dir, ' | '.join(bad[:20])))
        self.boxed_slices = [np.array(sorted(s_), np.int64) for s_ in boxed_k]
        self.free_slices = [np.array(sorted(set(range(int(self.n[vi]))) - s_), np.int64) for vi, s_ in enumerate(boxed_k)]
        self.train_boxed = [vi for vi in range(len(self.vols)) if not self.hold[vi] and len(self.boxed_slices[vi])]
        self.train_neg = [vi for vi in range(len(self.vols)) if not self.hold[vi] and (not len(self.boxed_slices[vi]))]
        self.holdout = [vi for vi in range(len(self.vols)) if self.hold[vi]]
        pid_hold = collections.defaultdict(set)
        for vi in range(len(self.vols)):
            pid_hold[self.pid[vi]].add(bool(self.hold[vi]))
        split_p = sorted((p_ for p_, s_ in pid_hold.items() if len(s_) > 1))
        if split_p:
            bad.append('patients on both sides of aux_holdout: %s' % split_p[:5])
        if len(self.vols) < min_volumes:
            bad.append('%d usable volumes < %d (--fm-aux-min-volumes; spec B5: >= 1,000)' % (len(self.vols), min_volumes))
        if not self.holdout:
            bad.append('no aux_holdout volume (the B5 diagnostic needs them)')
        if not self.train_boxed or not self.train_neg:
            bad.append('training volumes with boxes %d, without %d (both are drawn)' % (len(self.train_boxed), len(self.train_neg)))
        if (per_cls[0] == 0).any():
            bad.append('classes without a box in the training part: %s' % [FM_CLASSES[c] for c in np.flatnonzero(per_cls[0] == 0)])
        if bad:
            raise SystemExit('--fm-aux %s refused: %s' % (self.dir, ' | '.join(bad[:20])))
        self.per_cls, self.n_box, self.n_skip = (per_cls, n_box, n_skip)
        h_full = np.array([int(hashlib.sha1(p_.encode()).hexdigest(), 16) % 10 == 0 for p_ in self.pid])
        h_8 = np.array([int(hashlib.sha1(p_.encode()).hexdigest()[:8], 16) % 10 == 0 for p_ in self.pid])
        self.hold_agree = (int((h_full == self.hold).sum()), int((h_8 == self.hold).sum()))

    def summary(self):
        return '%s: %d usable volumes (%d excluded, skipped), %d patients; training %d with boxes + %d without, holdout %d (%d patients); %d rows, %d boxes (rows x boxes; %d on excluded rows skipped); per class training / holdout %s; aux_holdout vs sha1(patient_id) %% 10 == 0: %d / %d agree (full hex), %d / %d ([:8]); md5 %s; manifest keys %s' % (self.dir, len(self.vols), len(self.excluded), len(set(self.pid)), len(self.train_boxed), len(self.train_neg), len(self.holdout), len({self.pid[v] for v in self.holdout}), self.n_rows, self.n_box, self.n_skip, {FM_CLASSES[c]: (int(self.per_cls[0, c]), int(self.per_cls[1, c])) for c in range(FM_A)}, self.hold_agree[0], len(self.vols), self.hold_agree[1], len(self.vols), self.md5, sorted(self.manifest)[:12])

    def window_rows(self, vi, view, k):
        """the 3 rows of the window centred on slice k of (volume vi, view): k-1, k, k+1 clamped to the volume (phys step 1, window_rows)"""
        s_, n_ = (self.start[vi, view], int(self.n[vi]))
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
        return (dense, pres, np.zeros((0, 6)))
    b = np.asarray(boxes, np.float64)
    x0, x1 = (np.clip(b[:, 1] - ox, 0, crop), np.clip(b[:, 3] - ox, 0, crop))
    y0, y1 = (np.clip(b[:, 2] - oy, 0, crop), np.clip(b[:, 4] - oy, 0, crop))
    if shift:
        x0, x1 = (np.clip(x0 - dx, 0, crop), np.clip(x1 - dx, 0, crop))
        y0, y1 = (np.clip(y0 - dy, 0, crop), np.clip(y1 - dy, 0, crop))
    if flip:
        x0, x1 = (crop - x1, crop - x0)
    area = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    masks = {}
    kept = []
    for i in np.flatnonzero(area > 0):
        c = int(b[i, 0])
        m = masks.get(c)
        if m is None:
            m = masks[c] = np.zeros((crop, crop), np.float32)
        c0, c1 = (int(math.floor(x0[i])), int(math.ceil(x1[i])))
        r0, r1 = (int(math.floor(y0[i])), int(math.ceil(y1[i])))
        cc, rr = (np.arange(c0, c1), np.arange(r0, r1))
        wx = np.minimum(x1[i], cc + 1) - np.maximum(x0[i], cc)
        wy = np.minimum(y1[i], rr + 1) - np.maximum(y0[i], rr)
        np.maximum(m[r0:r1, c0:c1], np.outer(wy, wx).astype(np.float32), out=m[r0:r1, c0:c1])
        if area[i] >= FM_PRESENCE_MIN * b[i, 5]:
            pres[c] = 1.0
        kept.append((c, x0[i], y0[i], x1[i], y1[i], area[i]))
    for c, m in masks.items():
        dense[c] = m.reshape(G, FM_CELL, G, FM_CELL).mean((1, 3))
    return (dense, pres, np.array(kept, np.float64).reshape(-1, 6))

def fm_window_image(px, rows, crop, oy, ox, dy, dx, shift, flip):
    """v9.8: the uint8 [3, crop, crop] image of one auxiliary window (channels = slices k-1, k, k+1): crop at (oy, ox), translate (zero fill,
    v9.5's convention), mirror left-right -- the maps fm_window_targets applies to the boxes, in the same order."""
    img = np.stack([np.asarray(px[r]) for r in rows])[:, oy:oy + crop, ox:ox + crop]
    if shift:
        img = np.pad(img, ((0, 0), (shift, shift), (shift, shift)))[:, shift + dy:shift + dy + crop, shift + dx:shift + dx + crop]
    if flip:
        img = img[:, :, ::-1]
    return np.ascontiguousarray(img)
