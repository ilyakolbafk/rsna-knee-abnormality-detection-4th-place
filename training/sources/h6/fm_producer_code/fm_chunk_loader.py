"""AST-derived public FM checks with exactly two pixel-storage adaptations."""
from fm_loader_reference import *
from fm_chunk_array import ChunkArray

class FmChunkStore:
    """Lossless chunk adapter: original FM structural/targets/sampler checks retained; explicit ChunkArray replaces NPY pixel opening."""

    def __init__(self, d, min_volumes):
        self.dir = os.path.abspath(d)
        need = ('pixel_store.json', 'index.json', 'boxes.json', 'manifest.json')
        miss = [f_ for f_ in need if not os.path.isfile(os.path.join(self.dir, f_))]
        if miss:
            raise SystemExit('--fm-aux %s: missing %s' % (self.dir, miss))
        self.md5 = {f_: file_md5(os.path.join(self.dir, f_)) for f_ in ('index.json', 'boxes.json', 'manifest.json')}
        index = json.load(open(os.path.join(self.dir, 'index.json'), encoding='utf-8'))
        boxes = json.load(open(os.path.join(self.dir, 'boxes.json'), encoding='utf-8'))
        self.manifest = json.load(open(os.path.join(self.dir, 'manifest.json'), encoding='utf-8'))
        px = ChunkArray(self.dir)
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
