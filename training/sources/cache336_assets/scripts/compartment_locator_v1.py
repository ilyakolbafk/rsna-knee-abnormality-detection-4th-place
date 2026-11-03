"""Rule-based compartment localisation prototype on raw Gold58 DICOM (pure numpy; no pydicom/scipy in this VM).
- minimal Explicit-VR-LE DICOM reader (uncompressed pixels only)
- per-series patient geometry from IOP/IPP; plane from slice normal
- joint-line height (patient z) from S-I intensity profiles of condylar sagittal slices and central coronal slices
- notch x from sagittal stack extent; lateral positions per anatomy_roi convention (increase toward lateral)
- contact sheet per study with proposed 90 mm compartment crops
"""
import os, sys, json, struct, tarfile, glob, shutil, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HOME = os.environ['HOME']; W = os.path.join(HOME, 'mnt', 'RSNA Knee'); OUT = os.path.join(W, 'artifacts', 'compartment_locator_v1'); TMP = os.path.join(HOME, 'gold_tmp')
ARCH = os.path.join(W, 'data', 'gold58_mri', 'archives')
SIDE = pd.read_csv(os.path.join(W, '.codex/runs/rsna-oof-core5-next-20260912/compartment_model_20260913/Gold58_native64_side.csv'), dtype=str).set_index('StudyInstanceUID')
SERIES = pd.read_csv(os.path.join(W, 'data/gold58_mri/gold58_series.csv'), dtype=str)
LONG_VR = {b'OB', b'OW', b'OF', b'SQ', b'UT', b'UN', b'OD', b'OL', b'OV', b'SV', b'UV', b'UC', b'UR'}
WANT = {(0x0008, 0x103E): 'desc', (0x0020, 0x000E): 'series', (0x0020, 0x0013): 'instance', (0x0020, 0x0032): 'ipp', (0x0020, 0x0037): 'iop', (0x0028, 0x0030): 'spacing', (0x0028, 0x0010): 'rows', (0x0028, 0x0011): 'cols',
        (0x0028, 0x0100): 'bits', (0x0028, 0x0103): 'pixrep', (0x0028, 0x1052): 'intercept', (0x0028, 0x1053): 'slope', (0x0020, 0x0060): 'laterality', (0x0020, 0x0062): 'image_laterality', (0x0018, 0x0050): 'thickness', (0x7FE0, 0x0010): 'pixels', (0x0008, 0x0070): 'manufacturer', (0x0020, 0x0052): 'frame'}

def read_dicom(path):
    b = open(path, 'rb').read(); assert b[128:132] == b'DICM', path; p = 132; out = {}
    def skip_undefined_sequence(p):
        while True:
            g, e = struct.unpack_from('<HH', b, p); ln = struct.unpack_from('<I', b, p + 4)[0]; p += 8
            if (g, e) == (0xFFFE, 0xE0DD): return p
            if (g, e) == (0xFFFE, 0xE000):
                if ln == 0xFFFFFFFF:
                    while True:  # walk item content until item delimiter
                        g2, e2 = struct.unpack_from('<HH', b, p)
                        if (g2, e2) == (0xFFFE, 0xE00D): p += 8; break
                        vr2 = b[p + 4:p + 6]
                        if vr2 in LONG_VR: ln2 = struct.unpack_from('<I', b, p + 8)[0]; p += 12
                        else: ln2 = struct.unpack_from('<H', b, p + 6)[0]; p += 8
                        if ln2 == 0xFFFFFFFF: p = skip_undefined_sequence(p)
                        else: p += ln2
                else: p += ln
            else: raise ValueError('bad sequence structure')
    while p < len(b):
        g, e = struct.unpack_from('<HH', b, p); vr = b[p + 4:p + 6]
        if vr in LONG_VR: ln = struct.unpack_from('<I', b, p + 8)[0]; hdr = 12
        else: ln = struct.unpack_from('<H', b, p + 6)[0]; hdr = 8
        p += hdr
        if ln == 0xFFFFFFFF:
            if (g, e) == (0x7FE0, 0x0010): raise ValueError('encapsulated pixel data not supported')
            p = skip_undefined_sequence(p); continue
        val = b[p:p + ln]; p += ln
        key = WANT.get((g, e))
        if key is None: continue
        if key == 'pixels': out['pixels_raw'] = val; continue
        if vr in (b'US',): out[key] = struct.unpack_from('<H', val)[0]
        elif vr in (b'DS', b'IS', b'LO', b'CS', b'UI', b'SH'): out[key] = val.decode('latin-1').strip('\x00 ')
        else: out[key] = val
    rows, cols = out['rows'], out['cols']; dt = np.int16 if out.get('pixrep', 0) == 1 else np.uint16
    arr = np.frombuffer(out['pixels_raw'], dtype=dt, count=rows * cols).reshape(rows, cols).astype(np.float32)
    arr = arr * float(out.get('slope', '1') or 1) + float(out.get('intercept', '0') or 0)
    out['pixels'] = arr; del out['pixels_raw']
    out['ipp'] = np.array([float(x) for x in out['ipp'].split('\\')]); out['iop'] = np.array([float(x) for x in out['iop'].split('\\')]); out['spacing'] = [float(x) for x in out['spacing'].split('\\')]
    return out

def load_study(uid):
    arch = glob.glob(os.path.join(ARCH, f'*__{uid}.tar.gz'))[0]
    if os.path.isdir(TMP): shutil.rmtree(TMP)
    os.makedirs(TMP); tarfile.open(arch).extractall(TMP)
    series = {}
    for f in glob.glob(os.path.join(TMP, '**', '*.dcm'), recursive=True):
        try: d = read_dicom(f)
        except Exception as ex: print('skip', f[-40:], ex); continue
        series.setdefault(d['series'], []).append(d)
    for s in series.values():
        n = np.cross(s[0]['iop'][:3], s[0]['iop'][3:])
        s.sort(key=lambda d: float(np.dot(d['ipp'], n)))
    return series

def plane_of(s):
    n = np.abs(np.cross(s[0]['iop'][:3], s[0]['iop'][3:])); return ['Sagittal', 'Coronal', 'Axial'][int(np.argmax(n))]  # LPS: x=L, y=P, z=S

def si_profile(d, band=0.40):
    """Row/col-averaged intensity along the S-I patient axis, with mm coordinates (patient z) for each sample."""
    img = d['pixels']; rowdir = d['iop'][3:]; coldir = d['iop'][:3]  # coldir: direction along columns (x index), rowdir: along rows
    rs, cs = d['spacing']  # row spacing (between rows), column spacing
    if abs(rowdir[2]) >= abs(coldir[2]):  # rows run along S-I
        h, w = img.shape; c0, c1 = int(w * (0.5 - band / 2)), int(w * (0.5 + band / 2)); prof = img[:, c0:c1].mean(1)
        z = d['ipp'][2] + np.arange(h) * rs * rowdir[2] + ((c0 + c1) / 2) * cs * coldir[2]; other_extent = (c0, c1, 'cols')
    else:
        h, w = img.shape; r0, r1 = int(h * (0.5 - band / 2)), int(h * (0.5 + band / 2)); prof = img[r0:r1, :].mean(0)
        z = d['ipp'][2] + np.arange(w) * cs * coldir[2] + ((r0 + r1) / 2) * rs * rowdir[2]; other_extent = (r0, r1, 'rows')
    return z, prof, other_extent

def smooth(x, k):
    k = max(3, int(k) | 1); ker = np.ones(k) / k; return np.convolve(np.pad(x, k // 2, mode='edge'), ker, mode='valid')

def joint_gap(z, prof, fs, step_mm):
    """Find joint gap: on non-FS, marrow bright -> gap is the deepest dip between two bright bone plateaus; on FS marrow dark -> use inverted profile after removing background."""
    p = prof.copy(); p = (p - np.percentile(p, 5)) / (np.percentile(p, 99) - np.percentile(p, 5) + 1e-6)
    if fs: p = 1 - p
    ps = smooth(p, 6 / step_mm)  # 6 mm smoothing
    n = len(ps); lo, hi = int(n * 0.15), int(n * 0.85); best = None
    for i in range(lo, hi):
        if ps[i] <= ps[i - 1] and ps[i] <= ps[i + 1]:
            left = ps[max(0, i - int(45 / step_mm)):i]; right = ps[i + 1:i + 1 + int(45 / step_mm)]
            if len(left) < 3 or len(right) < 3: continue
            lm, rm = left.max(), right.max(); depth = min(lm, rm) - ps[i]
            if lm < 0.45 or rm < 0.45: continue  # both sides must be bone-bright
            score = depth
            if best is None or score > best[0]: best = (score, i, lm, rm)
    if best is None: return None
    score, i, lm, rm = best
    return dict(z=float(z[i]), depth=float(score), left=float(lm), right=float(rm), idx=int(i))

def analyse(uid, side):
    series = load_study(uid); meta = SERIES[SERIES.StudyInstanceUID == uid].set_index('SeriesInstanceUID')
    info = []; cands = []; options = []
    for sid, s in series.items():
        pl = plane_of(s); fs = int(meta.loc[sid, 'Fat_Suppression']) if sid in meta.index else -1; fl = int(meta.loc[sid, 'Fluid_Sensitive']) if sid in meta.index else -1
        n = np.cross(s[0]['iop'][:3], s[0]['iop'][3:]); pos = np.array([float(np.dot(d['ipp'], n)) for d in s])
        centers = np.array([d['ipp'] + d['iop'][:3] * d['spacing'][1] * d['cols'] / 2 + d['iop'][3:] * d['spacing'][0] * d['rows'] / 2 for d in s])
        rec = dict(series=sid, plane=pl, fs=fs, fluid=fl, n=len(s), desc=s[0].get('desc', ''), spacing=s[0]['spacing'][0], span_mm=float(pos.max() - pos.min()), frame=s[0].get('frame', ''))
        sel = []
        if pl == 'Sagittal':
            lat = centers[:, 0] if side == 'L' else -centers[:, 0]; lp = (lat - lat.min()) / max(lat.max() - lat.min(), 1e-6); rec['lateral_positions'] = lp.tolist(); rec['x_centers'] = centers[:, 0].tolist()
            sel = [(k, float(lp[k])) for k in range(len(s)) if 0.15 <= lp[k] <= 0.40 or 0.60 <= lp[k] <= 0.85]  # condylar slices
        if pl == 'Coronal':
            ap = (pos - pos.min()) / max(pos.max() - pos.min(), 1e-6); sel = [(k, None) for k in range(len(s)) if 0.30 <= ap[k] <= 0.70]
        if sel:
            # bright-fat prior: share of central-band pixels above 0.7*p99 (non-FS has bright fat everywhere)
            bright = float(np.mean([(lambda a: (a > 0.7 * np.percentile(a, 99)).mean())(s[k]['pixels']) for k, _ in sel[::max(1, len(sel) // 5)]]))
            trial = {}
            for polarity in (False, True):  # False = marrow bright (non-FS), True = marrow dark (FS/STIR)
                cc = []
                for k, latpos in sel:
                    d = s[k]; z, prof, _ = si_profile(d); g = joint_gap(z, prof, polarity, d['spacing'][0])
                    if g: cc.append(dict(series=sid, plane=pl, fs=fs, polarity_dark=polarity, idx=k, lat=latpos, z=g['z'], depth=g['depth']))
                zs = np.array([c['z'] for c in cc]); mad = float(np.median(np.abs(zs - np.median(zs)))) if len(zs) else np.inf
                trial[polarity] = (cc, mad)
            for polarity, (cc, m) in trial.items():
                if len(cc) >= 3: options.append(dict(series=sid, plane=pl, fs=fs, polarity_dark=polarity, n=len(cc), z=float(np.median([c['z'] for c in cc])), mad=m, cands=cc, bright=bright, desc=rec['desc']))
            rec.update(bright_share=bright, mad_bright=trial[False][1], mad_dark=trial[True][1], n_bright=len(trial[False][0]), n_dark=len(trial[True][0]))
        info.append(rec)
    coherent = [o for o in options if o['mad'] <= 6.0]
    sag_opts = sorted([o for o in coherent if o['plane'] == 'Sagittal'], key=lambda o: o['mad']); cor_opts = sorted([o for o in coherent if o['plane'] == 'Coronal'], key=lambda o: o['mad'])
    res = dict(uid=uid, side=side, n_series=len(series), planes={r['plane']: 0 for r in info}, n_frames=len(set(r['frame'] for r in info)), n_options=len(options), n_coherent=len(coherent))
    for r in info: res['planes'][r['plane']] += 1
    chosen_s = chosen_c = None
    if sag_opts and cor_opts:
        best = None
        for so in sag_opts:
            for co in cor_opts:
                gap = abs(so['z'] - co['z']); cost = gap + 0.5 * (so['mad'] + co['mad']) + (0 if so['polarity_dark'] == (so['fs'] == 1) else 2) + (0 if co['polarity_dark'] == (co['fs'] == 1) else 2)
                if best is None or cost < best[0]: best = (cost, so, co, gap)
        _, chosen_s, chosen_c, gap = best
        res.update(z_sag=chosen_s['z'], z_cor=chosen_c['z'], sag_cor_gap=float(gap), joint_z=float((chosen_s['z'] + chosen_c['z']) / 2) if gap <= 10 else chosen_s['z'], accept='both_planes_agree' if gap <= 10 else 'planes_disagree_use_sag')
    elif sag_opts or cor_opts:
        o = (sag_opts or cor_opts)
        o = sorted(o, key=lambda o: o['mad'] + (0 if o['polarity_dark'] == (o['fs'] == 1) else 2))[0]
        if o['plane'] == 'Sagittal': chosen_s = o; res.update(z_sag=o['z'], z_cor=None)
        else: chosen_c = o; res.update(z_cor=o['z'], z_sag=None)
        res.update(sag_cor_gap=None, joint_z=o['z'], accept='single_plane')
    else:
        res.update(joint_z=None, accept='no_coherent_option')
    for name, o in (('sag', chosen_s), ('cor', chosen_c)):
        if o: res[f'{name}_series'] = o['series']; res[f'{name}_polarity_dark'] = o['polarity_dark']; res[f'{name}_fs_tag'] = o['fs']; res[f'{name}_mad'] = o['mad']; res[f'{name}_n'] = o['n']; res[f'{name}_desc'] = o['desc']
    cands = (chosen_s['cands'] if chosen_s else []) + (chosen_c['cands'] if chosen_c else []); C = pd.DataFrame(cands)
    if chosen_s:
        sc = pd.DataFrame(chosen_s['cands']); med = sc[sc.lat < 0.5]; lat = sc[sc.lat >= 0.5]
        res['z_medial'] = float(np.median(med.z)) if len(med) else None; res['z_lateral'] = float(np.median(lat.z)) if len(lat) else None; res['n_med'] = int(len(med)); res['n_lat'] = int(len(lat))
    res['joint_mad'] = float(np.median(np.abs(C.z - np.median(C.z)))) if len(C) else None; res['n_cand'] = int(len(C))
    sag = [r for r in info if r['plane'] == 'Sagittal']
    if sag:
        s0 = max(sag, key=lambda r: r['n']); xs = np.array(s0['x_centers']); res['notch_x'] = float((xs.min() + xs.max()) / 2); res['sag_span_mm'] = float(abs(xs.max() - xs.min())); res['notch_series'] = s0['series']
    return series, info, C, res

def iso_z_line(ax, d, z, color, lw=1.2, ls='-'):
    """Draw the set of pixels with patient z == z (a straight line in the image, oblique if the plane is tilted)."""
    rowdir, coldir = d['iop'][3:], d['iop'][:3]; rs, cs = d['spacing']; h, w = d['pixels'].shape
    a = rs * rowdir[2]; b = cs * coldir[2]; c = z - d['ipp'][2]  # a*row + b*col = c
    pts = []
    if abs(a) > 1e-9:
        for col in (0, w - 1):
            row = (c - b * col) / a
            if -h <= row <= 2 * h: pts.append((col, row))
    if abs(b) > 1e-9 and len(pts) < 2:
        for row in (0, h - 1):
            col = (c - a * row) / b
            if -w <= col <= 2 * w: pts.append((col, row))
    if len(pts) >= 2: ax.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]], color=color, lw=lw, ls=ls)

def iso_x_line(ax, d, x, color, lw=1.2, ls='--'):
    rowdir, coldir = d['iop'][3:], d['iop'][:3]; rs, cs = d['spacing']; h, w = d['pixels'].shape
    a = rs * rowdir[0]; b = cs * coldir[0]; c = x - d['ipp'][0]; pts = []
    if abs(b) > 1e-9:
        for row in (0, h - 1): pts.append(((c - a * row) / b, row))
    elif abs(a) > 1e-9:
        for col in (0, w - 1): pts.append((col, (c - b * col) / a))
    if len(pts) >= 2: ax.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]], color=color, lw=lw, ls=ls)

def render(uid, series, info, C, res, path):
    side = res['side']; fig, axes = plt.subplots(1, 4, figsize=(20, 5.5)); [a.axis('off') for a in axes]
    z = res.get('joint_z'); nx = res.get('notch_x')
    # coronal central slice (prefer non-FS)
    cor = [r for r in info if r['plane'] == 'Coronal']; cor.sort(key=lambda r: (r['fs'] != 0, -r['n']))
    if cor:
        s = series[cor[0]['series']]; d = s[len(s) // 2]; img = d['pixels']; ax = axes[0]; ax.imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.5))
        rowdir, coldir = d['iop'][3:], d['iop'][:3]; rs, cs = d['spacing']
        if z is not None: iso_z_line(ax, d, z, 'y')
        if res.get('z_cor') is not None: iso_z_line(ax, d, res['z_cor'], 'orange', lw=0.8, ls=':')
        if res.get('z_sag') is not None: iso_z_line(ax, d, res['z_sag'], 'magenta', lw=0.8, ls=':')
        if nx is not None: iso_x_line(ax, d, nx, 'c')
        if nx is not None:
            # 90 mm crop boxes: medial and lateral (lateral = +x for L knee, -x for R knee)
            if z is not None and abs(coldir[0]) >= abs(rowdir[0]) and abs(rowdir[2]) >= abs(coldir[2]):
                lat_sign = 1 if side == 'L' else -1
                for comp, sgn, col in (('lateral', lat_sign, 'r'), ('medial', -lat_sign, 'lime')):
                    x0 = nx - sgn * 20; x1 = nx + sgn * 70; c0 = (x0 - d['ipp'][0]) / (cs * coldir[0]); c1 = (x1 - d['ipp'][0]) / (cs * coldir[0])
                    r0 = (z - 45 - d['ipp'][2]) / (rs * rowdir[2]); r1 = (z + 45 - d['ipp'][2]) / (rs * rowdir[2])
                    ax.add_patch(plt.Rectangle((min(c0, c1), min(r0, r1)), abs(c1 - c0), abs(r1 - r0), fill=False, ec=col, lw=1.5)); ax.text(min(c0, c1) + 3, min(r0, r1) + 14, comp, color=col, fontsize=9)
        ax.set_title(f"coronal {'FS' if cor[0]['fs']==1 else 'non-FS'} mid | z={z:.0f} mad={res.get('joint_mad',0):.1f} n={res.get('n_cand',0)}" if z is not None else 'coronal (no z)', fontsize=9)
    sag = [r for r in info if r['plane'] == 'Sagittal']; sag.sort(key=lambda r: (r['fs'] != 0, -r['n']))
    if sag:
        s = series[sag[0]['series']]; lp = np.array(sag[0]['lateral_positions'])
        for ax, target, name in ((axes[1], 0.25, 'medial'), (axes[2], 0.5, 'notch'), (axes[3], 0.75, 'lateral')):
            k = int(np.argmin(np.abs(lp - target))); d = s[k]; img = d['pixels']; ax.imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.5))
            rowdir, coldir = d['iop'][3:], d['iop'][:3]; rs, cs = d['spacing']
            if z is not None:
                iso_z_line(ax, d, z, 'y')
                if res.get('z_cor') is not None: iso_z_line(ax, d, res['z_cor'], 'orange', lw=0.8, ls=':')
                if abs(rowdir[2]) >= abs(coldir[2]):
                    cmid = img.shape[1] / 2; r0 = (z - 45 - d['ipp'][2] - cmid * cs * coldir[2]) / (rs * rowdir[2]); r1 = (z + 45 - d['ipp'][2] - cmid * cs * coldir[2]) / (rs * rowdir[2])
                    w = img.shape[1]; ax.add_patch(plt.Rectangle((w * 0.5 - 45 / cs, min(r0, r1)), 90 / cs, abs(r1 - r0), fill=False, ec='r' if name == 'lateral' else 'lime' if name == 'medial' else 'w', lw=1.2))
            ax.set_title(f"sag {name} pos={lp[k]:.2f} {'FS' if sag[0]['fs']==1 else 'non-FS'}", fontsize=9)
    fig.suptitle(f"{uid[-12:]} side={side} series={res['n_series']} {res['planes']} notch_x={nx if nx is None else round(nx,1)} span={res.get('sag_span_mm',0):.0f}mm", fontsize=10); fig.tight_layout(); fig.savefig(path, dpi=70); plt.close(fig)

if __name__ == '__main__':
    start, stop = int(sys.argv[1]), int(sys.argv[2]); t0 = time.time()
    gold = pd.read_csv(os.path.join(W, 'data/gold58_mri/gold58_labels.csv'), dtype={'StudyInstanceUID': str}); rows = []
    outp = os.path.join(OUT, 'coverage.jsonl')
    for i, uid in enumerate(gold.StudyInstanceUID.tolist()[start:stop], start=start):
        side = SIDE.loc[uid, 'side'] if uid in SIDE.index else None
        if side not in ('L', 'R'): side = 'R'  # render anyway; flag
        try:
            series, info, C, res = analyse(uid, side); res['side_known'] = bool(uid in SIDE.index and SIDE.loc[uid, 'side'] in ('L', 'R'))
            render(uid, series, info, C, res, os.path.join(OUT, 'sheets', f'gold_{i+1:03d}.png'))
        except Exception as ex: res = dict(uid=uid, error=repr(ex))
        res['gold_index'] = i + 1; rows.append(res); print(i + 1, uid[-10:], {k: (round(v, 1) if isinstance(v, float) else v) for k, v in res.items() if k in ('side', 'joint_z', 'z_sag', 'z_cor', 'sag_cor_gap', 'accept', 'sag_polarity_dark', 'sag_fs_tag', 'cor_polarity_dark', 'cor_fs_tag', 'n_coherent', 'error')}, flush=True)
        with open(outp, 'a') as f: f.write(json.dumps(res, default=str) + '\n')
    print('seconds', round(time.time() - t0, 1))
