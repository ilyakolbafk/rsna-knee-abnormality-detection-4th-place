"""Anterior (patellofemoral) compartment localisation prototype on raw Gold58 DICOM.
Reuses compartment_locator_v1 (DICOM reader, geometry, joint_z / notch_x / side).
Anchors: joint_z (tibiofemoral gap), notch_x (sagittal stack midpoint), anterior skin y at patella level.
Crop proposal: sagittal 110 x 110 mm, A-P from skin-5 to skin+105, S-I from joint_z-15 to joint_z+95, slices at lateral positions 0.30-0.70;
axial 90 x 90 mm at z in [joint_z+10, joint_z+65], x = notch +-50, same A-P window.
Patella detection (QA only): compact extreme-intensity blob in the anterior box on mid-sagittal slices, either polarity, coherent over >=3 slices.
"""
import os, sys, json, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compartment_locator_v1 as CL
OUT = os.path.join(CL.W, 'artifacts', 'anterior_locator_v1')
AP_ANT, AP_POST, SI_DOWN, SI_UP = 5.0, 105.0, 15.0, 95.0  # crop window (mm) relative to skin y and joint z
AX_LO, AX_HI = 10.0, 65.0  # axial PF level relative to joint z

def canon(d):
    """Return image with rows along S-I and cols along the other in-plane axis, plus direction vectors / spacings for the linear map."""
    img = d['pixels']; rowdir, coldir = d['iop'][3:], d['iop'][:3]; rs, cs = d['spacing']
    if abs(rowdir[2]) < abs(coldir[2]): img = img.T; rowdir, coldir = coldir, rowdir; rs, cs = cs, rs
    return img, np.asarray(rowdir, float), np.asarray(coldir, float), float(rs), float(cs)

def patient_coord(d, rows, cols, axis, transposed_pack=None):
    img, rowdir, coldir, rs, cs = transposed_pack if transposed_pack else canon(d)
    return d['ipp'][axis] + np.asarray(rows) * rs * rowdir[axis] + np.asarray(cols) * cs * coldir[axis]

def foreground_threshold(img):
    lo, hi = np.percentile(img, 2), np.percentile(img, 99); return float(lo + 0.10 * (hi - lo))

def skin_anterior(d, jz, band=(10.0, 45.0)):
    """Median anterior skin y (patient) over rows at patella level; returns (y, at_edge_share, n_rows) or None."""
    pack = canon(d); img, rowdir, coldir, rs, cs = pack; h, w = img.shape
    if abs(coldir[1]) < 0.8: return None  # columns must run A-P
    cmid = w / 2; z_rows = d['ipp'][2] + np.arange(h) * rs * rowdir[2] + cmid * cs * coldir[2]
    rows = np.where((z_rows >= jz + band[0]) & (z_rows <= jz + band[1]))[0]
    if len(rows) < 5: return None
    thr = foreground_threshold(img); run = max(2, int(round(3.0 / cs))); follow = max(4, int(round(10.0 / cs))); ys = []; edge = 0; margin = 3
    order = np.arange(w) if coldir[1] > 0 else np.arange(w)[::-1]  # anterior = smaller y: scan from the side where y is smallest
    for r in rows:
        line = img[r, order] > thr; c = None; cnt = 0
        for i in range(margin, w):
            cnt = cnt + 1 if line[i] else 0
            if cnt >= run:
                st = i - run + 1
                if line[st:st + follow].mean() >= 0.6: c = st; break
                cnt = 0
        if c is None: continue
        if c <= margin + 1: edge += 1
        col = order[c]; ys.append(float(d['ipp'][1] + r * rs * rowdir[1] + col * cs * coldir[1]))
    if len(ys) < 5: return None
    return float(np.median(ys)), edge / len(ys), len(ys)

def components(mask):
    """4-connected components on a small boolean grid; returns label array and list of (label, pixel index arrays)."""
    h, w = mask.shape; lab = np.zeros((h, w), int); n = 0; comps = []
    for r0 in range(h):
        for c0 in range(w):
            if mask[r0, c0] and lab[r0, c0] == 0:
                n += 1; stack = [(r0, c0)]; lab[r0, c0] = n; pix = []
                while stack:
                    r, c = stack.pop(); pix.append((r, c))
                    for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                        if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and lab[rr, cc] == 0: lab[rr, cc] = n; stack.append((rr, cc))
                comps.append((n, np.array(pix)))
    return lab, comps

def patella(d, jz, skin_y, prefer_dark):
    """Find the patella blob on one sagittal slice inside the anterior box. Returns dict or None."""
    pack = canon(d); img, rowdir, coldir, rs, cs = pack; h, w = img.shape
    if abs(coldir[1]) < 0.8: return None
    step = 1.0  # mm grid
    ys = np.arange(skin_y + 2, skin_y + 45, step); zs = np.arange(jz - 5, jz + 60, step)
    Yg, Zg = np.meshgrid(ys, zs)  # rows = z, cols = y
    # invert linear map: [y - ipp_y; z - ipp_z] = M [c; r] with M = [[cs*coldir_y, rs*rowdir_y],[cs*coldir_z, rs*rowdir_z]]
    M = np.array([[cs * coldir[1], rs * rowdir[1]], [cs * coldir[2], rs * rowdir[2]]]); Mi = np.linalg.inv(M)
    v = np.stack([Yg.ravel() - d['ipp'][1], Zg.ravel() - d['ipp'][2]]); cr = Mi @ v; c = np.rint(cr[0]).astype(int); r = np.rint(cr[1]).astype(int)
    ok = (c >= 0) & (c < w) & (r >= 0) & (r < h)
    if ok.mean() < 0.5: return None
    box = np.full(Yg.size, np.nan); box[ok] = img[r[ok], c[ok]]; box = box.reshape(Yg.shape)
    lo, hi = np.nanpercentile(img, 1), np.nanpercentile(img, 99); nb = np.clip((box - lo) / max(hi - lo, 1e-6), 0, 1); nb[np.isnan(box)] = 0.5
    def opening(m):
        p = np.pad(m, 1, mode='constant'); e = np.ones_like(m)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1): e &= p[1 + dr:1 + dr + m.shape[0], 1 + dc:1 + dc + m.shape[1]]
        p = np.pad(e, 1, mode='constant'); o = np.zeros_like(m)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1): o |= p[1 + dr:1 + dr + m.shape[0], 1 + dc:1 + dc + m.shape[1]]
        return o
    best = None
    for dark in ([prefer_dark, not prefer_dark]):
        mask = opening((nb < 0.22) if dark else (nb > 0.60))
        lab, comps = components(mask)
        for _, pix in comps:
            area = len(pix) * step * step
            if not (150 <= area <= 1400): continue
            zz = zs[pix[:, 0]]; yy = ys[pix[:, 1]]; hgt = zz.max() - zz.min() + step; wid = yy.max() - yy.min() + step
            if not (18 <= hgt <= 58 and 8 <= wid <= 34): continue
            fill = area / (hgt * wid)
            if fill < 0.45: continue
            depth = yy.mean() - skin_y; dz = zz.mean() - jz
            if not (4 <= depth <= 30) or not (8 <= dz <= 60): continue
            score = fill * min(area, 900) / 900 * (1.0 if dark == prefer_dark else 0.8) * (1.0 if dz >= 15 else 0.7)
            cand = dict(polarity_dark=dark, area=float(area), height=float(hgt), width=float(wid), fill=float(fill), y=float(yy.mean()), z=float(zz.mean()), depth=float(depth), score=float(score))
            if best is None or cand['score'] > best['score']: best = cand
        if best is not None: break
    return best

def analyse_anterior(uid, side):
    series, info, C, res = CL.analyse(uid, side)
    jz = res.get('joint_z'); nx = res.get('notch_x'); out = dict(uid=uid, side=side, joint_z=jz, notch_x=nx, accept_joint=res.get('accept'))
    if jz is None or nx is None: out['accept'] = 'no_joint_or_notch'; return series, info, res, out
    sag = [r for r in info if r['plane'] == 'Sagittal']; sag.sort(key=lambda r: (r['fs'] != 0, -r['n']))
    if not sag: out['accept'] = 'no_sagittal'; return series, info, res, out
    # skin anchor: try non-FS first; slices at 0.35-0.65
    skin = None
    for r in sag:
        s = series[r['series']]; lp = np.array(r['lateral_positions']); ks = [k for k in range(len(s)) if 0.35 <= lp[k] <= 0.65]
        vals = [skin_anterior(s[k], jz) for k in ks]; vals = [v for v in vals if v]
        if len(vals) >= 3:
            ys = np.array([v[0] for v in vals]); skin = dict(series=r['series'], fs=r['fs'], y=float(np.median(ys)), mad=float(np.median(np.abs(ys - np.median(ys)))), edge_share=float(np.mean([v[1] for v in vals])), n=len(vals)); break
    if skin is None: out['accept'] = 'no_skin'; return series, info, res, out
    out.update(skin_y=skin['y'], skin_mad=skin['mad'], skin_edge_share=skin['edge_share'], skin_n=skin['n'], skin_series=skin['series'], skin_fs=skin['fs'])
    # patella QA on the same series
    s = series[skin['series']]; lp = np.array([r for r in sag if r['series'] == skin['series']][0]['lateral_positions'])
    ks = [k for k in range(len(s)) if 0.40 <= lp[k] <= 0.65]; pats = [patella(s[k], jz, skin['y'], prefer_dark=(skin['fs'] == 1)) for k in ks]; pats = [p for p in pats if p]
    if len(pats) >= 3:
        zz = np.array([p['z'] for p in pats]); yy = np.array([p['y'] for p in pats])
        out.update(patella_found=True, patella_n=len(pats), patella_z=float(np.median(zz)), patella_y=float(np.median(yy)), patella_mad=float(np.median(np.abs(zz - np.median(zz)))),
                   patella_dz=float(np.median(zz) - jz), patella_depth=float(np.median(yy) - skin['y']), patella_height=float(np.median([p['height'] for p in pats])), patella_dark=bool(np.mean([p['polarity_dark'] for p in pats]) >= 0.5))
    else: out.update(patella_found=False, patella_n=len(pats))
    # axial availability at PF level
    axs = [r for r in info if r['plane'] == 'Axial']; n_ax = 0; ax_best = None
    for r in axs:
        s2 = series[r['series']]; zc = np.array([float(d['ipp'][2] + d['iop'][:3][2] * d['spacing'][1] * d['cols'] / 2 + d['iop'][3:][2] * d['spacing'][0] * d['rows'] / 2) for d in s2])
        m = (zc >= jz + AX_LO) & (zc <= jz + AX_HI); n_ax += int(m.sum())
        if m.any() and (ax_best is None or (r['fs'] == 0 and ax_best[2] != 0)): ax_best = (r['series'], int(np.argmin(np.abs(zc - (jz + (out.get('patella_dz', 28) if out.get('patella_found') else 28))))), r['fs'])
    out.update(n_axial_pf=n_ax, axial_series=ax_best[0] if ax_best else None, axial_idx=ax_best[1] if ax_best else None)
    # crop windows (patient coords)
    out.update(crop_y0=skin['y'] - AP_ANT, crop_y1=skin['y'] + AP_POST, crop_z0=jz - SI_DOWN, crop_z1=jz + SI_UP, crop_x0=nx - 45, crop_x1=nx + 45, ax_y0=skin['y'] - 5, ax_y1=skin['y'] + 85)
    # acceptance: skin found with low MAD; patella (if found) inside the crop with margin
    inside = True
    if out.get('patella_found'):
        inside = (out['crop_z0'] + 5 <= out['patella_z'] - out['patella_height'] / 2) and (out['patella_z'] + out['patella_height'] / 2 <= out['crop_z1'] - 5) and (out['patella_y'] + 10 <= out['crop_y1'])
    out['accept'] = ('ok' if inside else 'patella_outside_crop') if skin['mad'] <= 6 else 'skin_incoherent'
    return series, info, res, out

def draw_box(ax, d, pts_patient, axes_idx, color, lw=1.5, label=None):
    """pts_patient: list of (a, b) in patient axes axes_idx=(i, j); maps through the slice's linear map to image coords (col, row)."""
    rowdir, coldir = np.asarray(d['iop'][3:], float), np.asarray(d['iop'][:3], float); rs, cs = d['spacing']; i, j = axes_idx
    M = np.array([[cs * coldir[i], rs * rowdir[i]], [cs * coldir[j], rs * rowdir[j]]])
    if abs(np.linalg.det(M)) < 1e-9: return
    Mi = np.linalg.inv(M); xy = []
    for a, b in pts_patient:
        c, r = Mi @ np.array([a - d['ipp'][i], b - d['ipp'][j]]); xy.append((c, r))
    xy.append(xy[0]); ax.plot([p[0] for p in xy], [p[1] for p in xy], color=color, lw=lw)
    if label: ax.text(xy[0][0] + 3, xy[0][1] + 14, label, color=color, fontsize=9)

def render(uid, series, info, res, out, path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5)); [a.axis('off') for a in axes]
    jz, nx = out.get('joint_z'), out.get('notch_x'); sk = out.get('skin_y')
    sag = [r for r in info if r['plane'] == 'Sagittal']
    if sag and out.get('skin_series'):
        r0 = [r for r in sag if r['series'] == out['skin_series']][0]; s = series[r0['series']]; lp = np.array(r0['lateral_positions'])
        for ax, target, name in ((axes[0], 0.40, 'sag 0.40'), (axes[1], 0.52, 'sag mid'), (axes[2], 0.62, 'sag 0.62')):
            k = int(np.argmin(np.abs(lp - target))); d = s[k]; img = d['pixels']; ax.imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.5))
            if jz is not None: CL.iso_z_line(ax, d, jz, 'y', lw=0.8)
            if sk is not None:
                box = [(out['crop_y0'], out['crop_z0']), (out['crop_y1'], out['crop_z0']), (out['crop_y1'], out['crop_z1']), (out['crop_y0'], out['crop_z1'])]
                draw_box(ax, d, box, (1, 2), 'r', label='anterior 110mm')
                draw_box(ax, d, [(sk, out['crop_z0']), (sk, out['crop_z1'])], (1, 2), 'c', lw=0.8)
            if out.get('patella_found'):
                p = patella(d, jz, sk, prefer_dark=bool(out.get('patella_dark')))
                if p: draw_box(ax, d, [(p['y'] - p['width'] / 2, p['z'] - p['height'] / 2), (p['y'] + p['width'] / 2, p['z'] - p['height'] / 2), (p['y'] + p['width'] / 2, p['z'] + p['height'] / 2), (p['y'] - p['width'] / 2, p['z'] + p['height'] / 2)], (1, 2), 'lime', lw=1.0)
            ax.set_title(f"{name} pos={lp[k]:.2f} {'FS' if r0['fs']==1 else 'non-FS'}", fontsize=9)
    if out.get('axial_series') is not None:
        d = series[out['axial_series']][out['axial_idx']]; img = d['pixels']; ax = axes[3]; ax.imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.5))
        box = [(out['crop_x0'], out['ax_y0']), (out['crop_x1'], out['ax_y0']), (out['crop_x1'], out['ax_y1']), (out['crop_x0'], out['ax_y1'])]
        draw_box(ax, d, box, (0, 1), 'r', label='axial PF 90mm'); zc = float(d['ipp'][2] + d['iop'][:3][2] * d['spacing'][1] * d['cols'] / 2 + d['iop'][3:][2] * d['spacing'][0] * d['rows'] / 2)
        ax.set_title(f"axial z-jz={zc - jz:.0f}mm  n_ax_pf={out.get('n_axial_pf')}", fontsize=9)
    else: axes[3].set_title(f"no axial at PF level (n={out.get('n_axial_pf', 0)})", fontsize=9)
    pz = out.get('patella_dz'); pdp = out.get('patella_depth')
    fig.suptitle(f"{uid[-12:]} side={out['side']} jz={jz:.0f} skin_y={sk if sk is None else round(sk,1)} mad={out.get('skin_mad', -1):.1f} edge={out.get('skin_edge_share', 0):.2f} | patella {'dz=%.0f depth=%.0f h=%.0f n=%d' % (pz, pdp, out.get('patella_height', 0), out.get('patella_n', 0)) if out.get('patella_found') else 'not found'} | {out.get('accept')}", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=70); plt.close(fig)

if __name__ == '__main__':
    start, stop = int(sys.argv[1]), int(sys.argv[2]); t0 = time.time()
    gold = pd.read_csv(os.path.join(CL.W, 'data/gold58_mri/gold58_labels.csv'), dtype={'StudyInstanceUID': str}); outp = os.path.join(OUT, 'coverage.jsonl')
    for i, uid in enumerate(gold.StudyInstanceUID.tolist()[start:stop], start=start):
        side = CL.SIDE.loc[uid, 'side'] if uid in CL.SIDE.index else None
        if side not in ('L', 'R'): side = 'R'
        try:
            series, info, res, out = analyse_anterior(uid, side); render(uid, series, info, res, out, os.path.join(OUT, 'sheets', f'gold_{i+1:03d}.png'))
        except Exception as ex:
            import traceback; out = dict(uid=uid, error=repr(ex), tb=traceback.format_exc()[-400:])
        out['gold_index'] = i + 1
        print(i + 1, uid[-10:], {k: (round(v, 1) if isinstance(v, float) else v) for k, v in out.items() if k in ('side', 'joint_z', 'skin_y', 'skin_mad', 'skin_edge_share', 'skin_fs', 'patella_found', 'patella_dz', 'patella_depth', 'patella_height', 'patella_dark', 'n_axial_pf', 'accept', 'error')}, flush=True)
        with open(outp, 'a') as f: f.write(json.dumps(out, default=str) + '\n')
    print('seconds', round(time.time() - t0, 1))
