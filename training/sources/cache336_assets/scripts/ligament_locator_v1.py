"""Ligament (ACL/PCL notch + MCL) localisation prototype on raw Gold58 DICOM. Pure numpy; reuses compartment_locator_v1 / anterior_locator_v1.
Anchors: joint_z (tibiofemoral gap, from CL), notch_x from the intercondylar 'bone dip' along the sagittal stack (band z in [jz+5, jz+22]),
cross-checked on mid coronal slices; medial skin x at joint level on coronal slices; A-P tissue centre on the notch sagittal slice.
Windows (mm, patient coords; medial = -x for L knee, +x for R knee):
  sag_notch : slices |x - notch_x| <= 15 ; crop y in [apc-55, apc+55], z in [jz-45, jz+65]
  cor_notch : slices A-P 0.35-0.65 ; crop x in [notch-55, notch+55], z in [jz-50, jz+60]
  cor_mcl   : same slices ; crop x from notch-80 (medial, outside the condyle) to notch+30 (lateral), z in [jz-65, jz+45]
  sag_mcl   : slices 35..60 mm medial of the notch ; crop y as sag_notch, z in [jz-65, jz+45]
  medial skin is QA only (soft-tissue thickness beyond the assumed condyle edge notch-45)
"""
import os, sys, json, time, traceback
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compartment_locator_v1 as CL, anterior_locator_v1 as AL
OUT = os.path.join(CL.W, 'artifacts', 'ligament_locator_v1'); TILES = os.path.join(OUT, 'tiles')
ZB = (5.0, 22.0)          # condyle band above joint line for the notch profile
NOTCH_HALF = 15.0         # sagittal notch slices: |x - notch| <= 15 mm
CROP = 110.0

def norm_img(img):
    lo, hi = np.percentile(img, 2), np.percentile(img, 99); return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1), 0.10  # threshold in normalised units

def slice_center(d, axis):
    return float(d['ipp'][axis] + d['iop'][:3][axis] * d['spacing'][1] * d['cols'] / 2 + d['iop'][3:][axis] * d['spacing'][0] * d['rows'] / 2)

def tissue_cols(nb, thr_fg, rows):
    """Columns whose share of foreground rows (within `rows`) exceeds 0.5 -> first/last index, or None."""
    fg = (nb[rows] > thr_fg).mean(0) > 0.5; idx = np.where(fg)[0]
    if len(idx) < 5: return None
    return int(idx[0]), int(idx[-1])

def band_value_sag(d, jz):
    """Mean normalised intensity in the condyle band (z in jz+ZB) over the central 50 % of the tissue A-P extent on one sagittal slice."""
    img, rowdir, coldir, rs, cs = AL.canon(d); h, w = img.shape; nb, thr = norm_img(img)
    cmid = w / 2; z_rows = d['ipp'][2] + np.arange(h) * rs * rowdir[2] + cmid * cs * coldir[2]
    rows = np.where((z_rows >= jz + ZB[0]) & (z_rows <= jz + ZB[1]))[0]
    if len(rows) < 3: return np.nan
    tc = tissue_cols(nb, thr, rows)
    if tc is None: return np.nan
    c0, c1 = tc; span = c1 - c0; a, b = int(c0 + 0.25 * span), int(c0 + 0.75 * span)
    if b - a < 3: return np.nan
    return float(nb[rows][:, a:b].mean())

def find_extremum(x, v, dark_marrow, centre_hint=None, half=30.0):
    """Notch along x: peak of v if marrow is dark (FS), dip if marrow bright (non-FS). Returns dict or None."""
    ok = np.isfinite(v); x = np.asarray(x)[ok]; v = np.asarray(v)[ok]
    if len(x) < 7: return None
    o = np.argsort(x); x = x[o]; v = v[o]; step = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
    vs = CL.smooth(v, max(3, 5.0 / max(step, 0.3))); score = vs if dark_marrow else -vs
    if centre_hint is None: lo, hi = int(len(x) * 0.28), int(len(x) * 0.72)
    else:
        m = np.where(np.abs(x - centre_hint) <= half)[0]
        if len(m) < 3: return None
        lo, hi = int(m[0]), int(m[-1]) + 1
    i = lo + int(np.argmax(score[lo:hi]))
    left = score[(x < x[i] - 10) & (x > x[i] - 32)]; right = score[(x > x[i] + 10) & (x < x[i] + 32)]
    if len(left) < 2 or len(right) < 2: return None
    contrast = float(min(score[i] - left.min(), score[i] - right.min()))
    if contrast <= 0: return None
    thr = score[i] - 0.5 * contrast; a = i
    while a > 0 and score[a - 1] >= thr: a -= 1
    b = i
    while b < len(x) - 1 and score[b + 1] >= thr: b += 1
    return dict(x=float(x[i]), contrast=contrast, width_mm=float(x[b] - x[a] + step), idx=int(i), n=int(len(x)))

def polarity_choice(xs, vals, fs, hint, half):
    """Tag-consistent polarity whenever it yields an extremum with contrast >= 0.03; otherwise the other polarity; unknown tag -> best contrast."""
    cands = {}
    for dark in (False, True):
        e = find_extremum(xs, vals, dark, centre_hint=hint, half=half)
        if e: cands[dark] = e
    if not cands: return None, None
    if fs in (0, 1):
        pref = (fs == 1)
        if pref in cands and cands[pref]['contrast'] >= 0.03: return pref, cands[pref]
        if (not pref) in cands: return (not pref), cands[not pref]
        return pref, cands[pref]
    dark = max(cands, key=lambda k: cands[k]['contrast']); return dark, cands[dark]

def notch_from_sagittal(series, info, jz, hint, half=22.0):
    """Condyle-band profile along every sagittal series; per-series polarity by tag; contrast-weighted mean of the per-series notch x."""
    picks = []; profiles = {}
    for r in sorted([r for r in info if r['plane'] == 'Sagittal'], key=lambda r: (r['fs'] != 0, -r['n'])):
        s = series[r['series']]; xs = np.array([slice_center(d, 0) for d in s]); vals = np.array([band_value_sag(d, jz) for d in s])
        profiles[r['series']] = (xs, vals)
        dark, e = polarity_choice(xs, vals, r['fs'], hint, half)
        if e: picks.append(dict(series=r['series'], fs=r['fs'], dark=dark, **e))
    if not picks: return None, profiles
    w = np.array([p['contrast'] for p in picks]); x = np.array([p['x'] for p in picks])
    best = max(picks, key=lambda p: p['contrast'])
    return dict(x=float((w * x).sum() / w.sum()), spread=float(x.max() - x.min()), n_series=len(picks), contrast=float(best['contrast']), width_mm=float(best['width_mm']), series=best['series'], fs=best['fs'], dark=best['dark'], n=best['n'], picks=picks), profiles

def coronal_profile(d, jz):
    img, rowdir, coldir, rs, cs = AL.canon(d); h, w = img.shape; nb, thr = norm_img(img)
    cmid = w / 2; z_rows = d['ipp'][2] + np.arange(h) * rs * rowdir[2] + cmid * cs * coldir[2]
    rows = np.where((z_rows >= jz + ZB[0]) & (z_rows <= jz + ZB[1]))[0]
    if len(rows) < 3: return None
    xs = d['ipp'][0] + np.arange(w) * cs * coldir[0] + (rows.mean()) * rs * rowdir[0]; vals = nb[rows].mean(0)
    return xs, vals

def notch_from_coronal(series, info, jz, hint, half=22.0):
    out = []
    for r in sorted([r for r in info if r['plane'] == 'Coronal'], key=lambda r: (r['fs'] != 0, -r['n'])):
        s = series[r['series']]; n = np.cross(s[0]['iop'][:3], s[0]['iop'][3:]); pos = np.array([float(np.dot(d['ipp'], n)) for d in s]); ap = (pos - pos.min()) / max(pos.max() - pos.min(), 1e-6)
        ks = [k for k in range(len(s)) if 0.40 <= ap[k] <= 0.60][:5]
        for k in ks:
            pr = coronal_profile(s[k], jz)
            if pr is None: continue
            dark, e = polarity_choice(pr[0], pr[1], r['fs'], hint, half)
            if e: out.append(dict(series=r['series'], fs=r['fs'], dark=dark, k=k, **e))
        if out: break
    if not out: return None
    return dict(x=float(np.median([o['x'] for o in out])), contrast=float(np.median([o['contrast'] for o in out])), n=len(out), dark=bool(np.mean([o['dark'] for o in out]) >= 0.5), series=out[0]['series'], fs=out[0]['fs'])

def skin_x(d, jz, med_sign, x_centre=None, half=95.0, band=15.0):
    """Medial and lateral skin x at joint level on one coronal slice (rows z in jz+-band), searched only within |x - x_centre| <= half."""
    img, rowdir, coldir, rs, cs = AL.canon(d); h, w = img.shape
    if abs(coldir[0]) < 0.8: return None
    nb, thr = norm_img(img); cmid = w / 2; z_rows = d['ipp'][2] + np.arange(h) * rs * rowdir[2] + cmid * cs * coldir[2]
    rows = np.where((z_rows >= jz - band) & (z_rows <= jz + band))[0]
    if len(rows) < 5: return None
    run = max(2, int(round(3.0 / cs))); follow = max(4, int(round(10.0 / cs))); margin = 3
    xcol = d['ipp'][0] + np.arange(w) * cs * coldir[0]
    cols = np.arange(w) if x_centre is None else np.where(np.abs(xcol - x_centre) <= half)[0]
    if len(cols) < 20: return None
    def first_tissue(line, order):
        cnt = 0
        for i in range(margin, len(order)):
            cnt = cnt + 1 if line[order[i]] else 0
            if cnt >= run:
                st = i - run + 1
                if line[order[st:st + follow]].mean() >= 0.6: return st, order[st]
                cnt = 0
        return None
    med_order = cols[np.argsort(xcol[cols] * med_sign)[::-1]]  # most medial column first
    lat_order = med_order[::-1]
    xm, xl = [], []; edge_m = edge_l = 0
    for r in rows:
        line = nb[r] > thr; a = first_tissue(line, med_order); b = first_tissue(line, lat_order)
        if a is not None: xm.append(float(xcol[a[1]])); edge_m += int(a[0] * cs <= 5.0)
        if b is not None: xl.append(float(xcol[b[1]])); edge_l += int(b[0] * cs <= 5.0)
    if len(xm) < 5: return None
    return dict(x_med=float(np.median(xm)), mad_med=float(np.median(np.abs(np.array(xm) - np.median(xm)))), x_lat=float(np.median(xl)) if len(xl) >= 5 else None, n=len(xm), edge_med=edge_m / len(xm), edge_lat=(edge_l / len(xl)) if len(xl) else None)

def skin_from_sagittal(series, info, jz, med_sign, band=15.0):
    """Medial/lateral skin x from the sagittal stack: first slice (from each end) whose tissue share in rows z in jz+-band exceeds 0.25."""
    sag = sorted([r for r in info if r['plane'] == 'Sagittal'], key=lambda r: -r['n'])
    if not sag: return None
    s = series[sag[0]['series']]; xs = np.array([slice_center(d, 0) for d in s]); frac = []
    for d in s:
        img, rowdir, coldir, rs, cs = AL.canon(d); h, w = img.shape; nb, thr = norm_img(img); cmid = w / 2
        z_rows = d['ipp'][2] + np.arange(h) * rs * rowdir[2] + cmid * cs * coldir[2]; rows = np.where((z_rows >= jz - band) & (z_rows <= jz + band))[0]
        frac.append(float((nb[rows] > thr).mean()) if len(rows) >= 3 else np.nan)
    frac = np.array(frac); order = np.argsort(xs * med_sign)[::-1]  # medial first
    step = float(np.median(np.abs(np.diff(xs[order])))) if len(xs) > 1 else 3.0
    def first(order_):
        for j, k in enumerate(order_):
            if np.isfinite(frac[k]) and frac[k] >= 0.25: return k, j
        return None
    m = first(order); l = first(order[::-1])
    if m is None: return None
    x_med = float(xs[m[0]] + med_sign * step / 2); x_lat = float(xs[l[0]] - med_sign * step / 2) if l else None
    return dict(x_med=x_med, x_lat=x_lat, edge_med=float(m[1] == 0), edge_lat=float(l[1] == 0) if l else None, series=sag[0]['series'], step=step)

def ap_centre(d, jz):
    img, rowdir, coldir, rs, cs = AL.canon(d); h, w = img.shape
    if abs(coldir[1]) < 0.8: return None
    nb, thr = norm_img(img); cmid = w / 2; z_rows = d['ipp'][2] + np.arange(h) * rs * rowdir[2] + cmid * cs * coldir[2]
    rows = np.where((z_rows >= jz - 10) & (z_rows <= jz + 10))[0]
    if len(rows) < 3: return None
    tc = tissue_cols(nb, thr, rows)
    if tc is None: return None
    y0 = d['ipp'][1] + tc[0] * cs * coldir[1]; y1 = d['ipp'][1] + tc[1] * cs * coldir[1]
    return dict(apc=float((y0 + y1) / 2), ap_extent=float(abs(y1 - y0)), y_ant=float(min(y0, y1)), y_post=float(max(y0, y1)))

def fov_fraction(d, axes_idx, a0, a1, b0, b1, step=2.0):
    img, rowdir, coldir, rs, cs = AL.canon(d); h, w = img.shape; i, j = axes_idx
    A, B = np.meshgrid(np.arange(a0, a1, step), np.arange(b0, b1, step)); M = np.array([[cs * coldir[i], rs * rowdir[i]], [cs * coldir[j], rs * rowdir[j]]])
    if abs(np.linalg.det(M)) < 1e-9: return None
    cr = np.linalg.inv(M) @ np.stack([A.ravel() - d['ipp'][i], B.ravel() - d['ipp'][j]]); c = np.rint(cr[0]); r = np.rint(cr[1])
    return float(((c >= 0) & (c < w) & (r >= 0) & (r < h)).mean())

def resample(d, axes_idx, a0, a1, b0, b1, step=1.0):
    """Crop window of slice d in patient axes (i, j) -> image on a 1 mm grid (rows = j descending, cols = i ascending)."""
    img, rowdir, coldir, rs, cs = AL.canon(d); h, w = img.shape; i, j = axes_idx
    A, B = np.meshgrid(np.arange(a0, a1, step), np.arange(b1, b0, -step)); M = np.array([[cs * coldir[i], rs * rowdir[i]], [cs * coldir[j], rs * rowdir[j]]])
    cr = np.linalg.inv(M) @ np.stack([A.ravel() - d['ipp'][i], B.ravel() - d['ipp'][j]]); c = np.rint(cr[0]).astype(int); r = np.rint(cr[1]).astype(int)
    ok = (c >= 0) & (c < w) & (r >= 0) & (r < h); t = np.zeros(A.size, np.float32); t[ok] = img[r[ok], c[ok]]; return t.reshape(A.shape)

def analyse_ligament(uid, side):
    series, info, C, res = CL.analyse(uid, side); jz = res.get('joint_z'); med_sign = -1.0 if side == 'L' else 1.0
    out = dict(uid=uid, side=side, joint_z=jz, accept_joint=res.get('accept'), notch_mid_x=res.get('notch_x'), sag_span_mm=res.get('sag_span_mm'))
    if jz is None: out['accept'] = 'no_joint'; return series, info, res, out, {}
    skin = None
    for r in sorted([r for r in info if r['plane'] == 'Coronal'], key=lambda r: (r['fs'] != 0, -r['n'])):
        s = series[r['series']]; n = np.cross(s[0]['iop'][:3], s[0]['iop'][3:]); pos = np.array([float(np.dot(d['ipp'], n)) for d in s]); ap = (pos - pos.min()) / max(pos.max() - pos.min(), 1e-6)
        vals = [skin_x(s[k], jz, med_sign, x_centre=res.get('notch_x')) for k in range(len(s)) if 0.35 <= ap[k] <= 0.65]; vals = [v for v in vals if v]
        if len(vals) >= 3:
            xm = np.array([v['x_med'] for v in vals]); xl = np.array([v['x_lat'] for v in vals if v['x_lat'] is not None])
            skin = dict(series=r['series'], fs=r['fs'], x_med=float(np.median(xm)), mad=float(np.median(np.abs(xm - np.median(xm)))), x_lat=float(np.median(xl)) if len(xl) >= 3 else None, n=len(vals), edge_med=float(np.mean([v['edge_med'] for v in vals])), edge_lat=float(np.mean([v['edge_lat'] for v in vals if v['edge_lat'] is not None])) if any(v['edge_lat'] is not None for v in vals) else None); break
    sk_sag = skin_from_sagittal(series, info, jz, med_sign)
    out['skin_sag_med_x'] = None if sk_sag is None else sk_sag['x_med']; out['skin_sag_edge'] = None if sk_sag is None else sk_sag['edge_med']
    out['skin_cor_med_x'] = None if skin is None else skin['x_med']; out['skin_cor_edge'] = None if skin is None else skin['edge_med']
    cor_ok = skin is not None and skin['edge_med'] < 0.5 and skin['x_lat'] is not None and 60 <= abs(skin['x_lat'] - skin['x_med']) <= 150
    sag_ok = sk_sag is not None and sk_sag['edge_med'] < 0.5
    out['skin_cor_sag_gap'] = float(abs(skin['x_med'] - sk_sag['x_med'])) if (skin is not None and sk_sag is not None) else None
    if cor_ok and (not sag_ok or out['skin_cor_sag_gap'] <= 12): out['skin_source'] = 'coronal'
    elif sag_ok:
        out['skin_source'] = 'sagittal'; skin = dict(series=sk_sag['series'], fs=-1, x_med=sk_sag['x_med'], mad=0.0, x_lat=sk_sag['x_lat'], n=0, edge_med=sk_sag['edge_med'], edge_lat=sk_sag['edge_lat']) if skin is None else dict(skin, x_med=sk_sag['x_med'], x_lat=sk_sag['x_lat'] if sk_sag['x_lat'] is not None else skin['x_lat'], edge_med=sk_sag['edge_med'])
    elif skin is not None: out['skin_source'] = 'coronal_flagged'
    else: out['accept'] = 'no_medial_skin'; return series, info, res, out, {}
    width = abs(skin['x_lat'] - skin['x_med']) if skin['x_lat'] is not None else None
    knee_centre = (skin['x_med'] + skin['x_lat']) / 2 if (width is not None and 60 <= width <= 150) else res.get('notch_x')
    out.update(skin_series=skin['series'], skin_fs=skin['fs'], skin_med_x=skin['x_med'], skin_mad=skin['mad'], skin_lat_x=skin['x_lat'], knee_width_mm=width, knee_centre_x=knee_centre, skin_edge_med=skin['edge_med'], skin_edge_lat=skin['edge_lat'])
    best, profiles = notch_from_sagittal(series, info, jz, hint=knee_centre)
    if best is None:
        out.update(notch_x=float(knee_centre), notch_accept='fallback_knee_centre', notch_series=None)
    else:
        out.update(notch_series=best['series'], notch_fs=best['fs'], notch_dark=best['dark'], notch_x_sag=best['x'], notch_sag_spread=best['spread'], notch_sag_nseries=best['n_series'], notch_contrast=best['contrast'], notch_width_mm=best['width_mm'], notch_n_slices=best['n'])
        cor = notch_from_coronal(series, info, jz, hint=knee_centre); mid = res.get('notch_x')
        votes = [best['x']] + ([cor['x']] if cor else []) + ([mid] if mid is not None else [])
        if cor:
            out.update(notch_x_cor=cor['x'], notch_cor_contrast=cor['contrast'], notch_cor_n=cor['n'], notch_cor_series=cor['series'], notch_cor_dark=cor['dark'])
            gap = abs(cor['x'] - best['x']); out['notch_gap'] = float(gap)
        else: out['notch_gap'] = None
        # final estimate: mean of the sagittal condyle-band profile and the sagittal stack midpoint (Gold58 review: the coronal x-profile is
        # unreliable and is kept as information only; the profile alone misses by 10-15 mm in ~2/58, the midpoint by 20+ mm in ~1/58; the mean keeps every
        # reviewed case within ~15 mm, which the +-15 mm slice window and the 110 mm crops tolerate)
        out['notch_x'] = float((best['x'] + mid) / 2) if mid is not None else float(best['x']); out['notch_votes_spread_mm'] = float(max(votes) - min(votes)); out['notch_source'] = 'mean_sag_mid' if mid is not None else 'sag'
        out['notch_sag_mid_gap_mm'] = float(abs(best['x'] - mid)) if mid is not None else None
        out['notch_accept'] = 'sag_mid_agree' if (out['notch_sag_mid_gap_mm'] or 0) <= 12 else 'sag_mid_disagree_mean'
    out['notch_vs_mid_mm'] = float(out['notch_x'] - res['notch_x']) if res.get('notch_x') is not None else None
    out['notch_vs_centre_mm'] = float(out['notch_x'] - knee_centre)
    out['notch_from_med_skin_mm'] = float((out['notch_x'] - skin['x_med']) * (-med_sign))
    if out.get('notch_series') is None:
        sagr = sorted([r for r in info if r['plane'] == 'Sagittal'], key=lambda r: (r['fs'] != 0, -r['n']))
        if not sagr: out['accept'] = 'no_sagittal'; return series, info, res, out, profiles
        out['notch_series'] = sagr[0]['series']; out['notch_fs'] = sagr[0]['fs']; out.setdefault('notch_contrast', 0.0); out.setdefault('notch_width_mm', 0.0)
    s = series[out['notch_series']]; xs = np.array([slice_center(d, 0) for d in s]); k_n = int(np.argmin(np.abs(xs - out['notch_x']))); apc = ap_centre(s[k_n], jz)
    if apc is None: out['accept'] = 'no_ap_centre'; return series, info, res, out, profiles
    out.update(**apc, notch_slice_idx=k_n)
    nx, sm = out['notch_x'], skin['x_med']; xin0, xin1 = nx + med_sign * 80, nx - med_sign * 30  # MCL crop: 80 mm medial of the notch .. 30 mm lateral of it
    out['soft_tissue_medial_mm'] = float((sm - nx) * med_sign - 45.0)  # medial skin beyond the assumed condyle edge (notch - 45 mm)
    out.update(sagn_y0=apc['apc'] - 55, sagn_y1=apc['apc'] + 55, sagn_z0=jz - 45, sagn_z1=jz + 65,
               corn_x0=nx - 55, corn_x1=nx + 55, corn_z0=jz - 50, corn_z1=jz + 60,
               mcl_x0=min(xin0, xin1), mcl_x1=max(xin0, xin1), mcl_z0=jz - 65, mcl_z1=jz + 45)
    cnt = {}
    for r in [r for r in info if r['plane'] == 'Sagittal']:
        ss = series[r['series']]; xx = np.array([slice_center(d, 0) for d in ss]); inward = (xx - sm) * (-med_sign)
        med_dist = (xx - nx) * med_sign  # mm medial of the notch (positive on the medial side)
        cnt[r['series']] = dict(fs=r['fs'], n_notch=int((np.abs(xx - nx) <= NOTCH_HALF).sum()), n_mcl=int(((med_dist >= 35) & (med_dist <= 60)).sum()), n=len(ss))
    out['sag_counts'] = cnt; out['n_notch_primary'] = cnt[out['notch_series']]['n_notch']; out['n_mcl_primary'] = cnt[out['notch_series']]['n_mcl']
    cors = [r for r in info if r['plane'] == 'Coronal']; out['n_cor_mid'] = int(sum(max(1, int(round(0.30 * r['n']))) for r in cors))
    sc = series[skin['series']]; d_cor = sc[len(sc) // 2]
    out['mcl_fov'] = fov_fraction(d_cor, (0, 2), out['mcl_x0'], out['mcl_x1'], out['mcl_z0'], out['mcl_z1']); out['corn_fov'] = fov_fraction(d_cor, (0, 2), out['corn_x0'], out['corn_x1'], out['corn_z0'], out['corn_z1'])
    out['sagn_fov'] = fov_fraction(s[k_n], (1, 2), out['sagn_y0'], out['sagn_y1'], out['sagn_z0'], out['sagn_z1'])
    flags = []
    if out['notch_contrast'] < 0.04: flags.append('weak_notch')
    if (out.get('notch_sag_mid_gap_mm') or 0) > 15: flags.append('notch_sag_mid_gap')
    out['skin_qa'] = 'edge' if skin['edge_med'] > 0.5 else ('unverified' if out.get('skin_source') == 'coronal_flagged' else 'ok')
    if out['n_notch_primary'] < 4: flags.append('few_notch_slices')
    if out['mcl_fov'] is not None and out['mcl_fov'] < 0.85: flags.append('mcl_crop_outside_fov')
    if out.get('skin_source') != 'coronal_flagged' and skin['edge_med'] < 0.5 and not (25 <= out['notch_from_med_skin_mm'] <= 85): flags.append('skin_notch_distance_odd')
    out['flags'] = flags; out['accept'] = 'ok' if not flags else 'review:' + ','.join(flags)
    np.save(os.path.join(TILES, f"{uid}_mcl_cor.npy"), resample(d_cor, (0, 2), out['mcl_x0'], out['mcl_x1'], out['mcl_z0'], out['mcl_z1']))
    np.save(os.path.join(TILES, f"{uid}_notch_sag.npy"), resample(s[k_n], (1, 2), out['sagn_y0'], out['sagn_y1'], out['sagn_z0'], out['sagn_z1']))
    np.save(os.path.join(TILES, f"{uid}_notch_cor.npy"), resample(d_cor, (0, 2), out['corn_x0'], out['corn_x1'], out['corn_z0'], out['corn_z1']))
    return series, info, res, out, profiles

def render(uid, series, info, res, out, profiles, path):
    fig, axes = plt.subplots(1, 4, figsize=(21, 5.6)); [a.axis('off') for a in axes[:3]]
    jz = out.get('joint_z'); nx = out.get('notch_x'); sm = out.get('skin_med_x')
    if out.get('skin_series'):
        sc = series[out['skin_series']]; d = sc[len(sc) // 2]; img = d['pixels']; ax = axes[0]; ax.imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.5))
        CL.iso_z_line(ax, d, jz, 'y', lw=0.8); CL.iso_x_line(ax, d, nx, 'c'); CL.iso_x_line(ax, d, sm, 'm', ls=':')
        if out.get('notch_x_sag') is not None: CL.iso_x_line(ax, d, out['notch_x_sag'], 'c', lw=0.6, ls=':')
        AL.draw_box(ax, d, [(out['corn_x0'], out['corn_z0']), (out['corn_x1'], out['corn_z0']), (out['corn_x1'], out['corn_z1']), (out['corn_x0'], out['corn_z1'])], (0, 2), 'w', lw=1.0, label='notch cor')
        AL.draw_box(ax, d, [(out['mcl_x0'], out['mcl_z0']), (out['mcl_x1'], out['mcl_z0']), (out['mcl_x1'], out['mcl_z1']), (out['mcl_x0'], out['mcl_z1'])], (0, 2), 'lime', lw=1.5, label='MCL cor')
        ax.set_title(f"coronal mid {'FS' if out['skin_fs']==1 else 'non-FS'} | width={out.get('knee_width_mm') or 0:.0f} notch-skin={out.get('notch_from_med_skin_mm', 0):.0f}", fontsize=9)
    if out.get('notch_series'):
        s = series[out['notch_series']]; xs = np.array([slice_center(d, 0) for d in s]); med_sign = -1.0 if out['side'] == 'L' else 1.0
        k = out['notch_slice_idx']; d = s[k]; img = d['pixels']; ax = axes[1]; ax.imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.5))
        CL.iso_z_line(ax, d, jz, 'y', lw=0.8); AL.draw_box(ax, d, [(out['sagn_y0'], out['sagn_z0']), (out['sagn_y1'], out['sagn_z0']), (out['sagn_y1'], out['sagn_z1']), (out['sagn_y0'], out['sagn_z1'])], (1, 2), 'c', lw=1.5, label='notch sag')
        ax.set_title(f"sag notch x={xs[k]:.0f} (notch={nx:.0f}) {'FS' if out['notch_fs']==1 else 'non-FS'} c={out['notch_contrast']:.2f} w={out['notch_width_mm']:.0f}mm", fontsize=9)
        inward = (xs - nx) * med_sign; km = [i for i in range(len(s)) if 35 <= inward[i] <= 60]
        if km:
            k2 = km[len(km) // 2]; d = s[k2]; img = d['pixels']; ax = axes[2]; ax.imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.5))
            CL.iso_z_line(ax, d, jz, 'y', lw=0.8); AL.draw_box(ax, d, [(out['sagn_y0'], out['mcl_z0']), (out['sagn_y1'], out['mcl_z0']), (out['sagn_y1'], out['mcl_z1']), (out['sagn_y0'], out['mcl_z1'])], (1, 2), 'lime', lw=1.5, label='MCL sag')
            ax.set_title(f"sag MCL slice {inward[k2]:.0f}mm medial of notch (n={len(km)})", fontsize=9)
        else: axes[2].set_title('no sagittal slice 35-60 mm medial of the notch', fontsize=9)
        ax = axes[3]
        for sid, (px, pv) in profiles.items():
            ok = np.isfinite(pv)
            if ok.sum() > 3: ax.plot(px[ok], pv[ok], '.-', ms=3, lw=0.8, label=f"{sid[-6:]} {'FS' if [r for r in info if r['series']==sid][0]['fs']==1 else 'nFS'}")
        ax.axvline(nx, color='c'); ax.axvline(res['notch_x'], color='gray', ls=':'); ax.axvline(sm, color='m', ls=':')
        if out.get('notch_x_cor') is not None: ax.axvline(out['notch_x_cor'], color='orange', ls='--', lw=0.8)
        ax.set_title('condyle-band profile along sagittal stack (cyan=notch, gray=mid, orange=cor, magenta=medial skin)', fontsize=8); ax.legend(fontsize=6); ax.set_xlabel('patient x, mm')
    fig.suptitle(f"{uid[-12:]} side={out['side']} jz={jz:.0f} | notch {out.get('notch_accept')} gap={out.get('notch_gap') if out.get('notch_gap') is None else round(out['notch_gap'],1)} vs_mid={out.get('notch_vs_mid_mm') if out.get('notch_vs_mid_mm') is None else round(out['notch_vs_mid_mm'],1)} | skin mad={out.get('skin_mad', -1):.1f} | n_notch={out.get('n_notch_primary')} n_mcl={out.get('n_mcl_primary')} | fov mcl={out.get('mcl_fov') or 0:.2f} | {out.get('accept')}", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=70); plt.close(fig)

if __name__ == '__main__':
    start, stop = int(sys.argv[1]), int(sys.argv[2]); t0 = time.time()
    gold = pd.read_csv(os.path.join(CL.W, 'data/gold58_mri/gold58_labels.csv'), dtype={'StudyInstanceUID': str}); outp = os.path.join(OUT, 'coverage.jsonl')
    for i, uid in enumerate(gold.StudyInstanceUID.tolist()[start:stop], start=start):
        side = CL.SIDE.loc[uid, 'side'] if uid in CL.SIDE.index else None
        if side not in ('L', 'R'): side = 'R'
        try:
            series, info, res, out, profiles = analyse_ligament(uid, side); render(uid, series, info, res, out, profiles, os.path.join(OUT, 'sheets', f'gold_{i+1:03d}.png'))
        except Exception as ex: out = dict(uid=uid, error=repr(ex), tb=traceback.format_exc()[-600:])
        out['gold_index'] = i + 1
        print(i + 1, uid[-10:], {k: (round(v, 1) if isinstance(v, float) else v) for k, v in out.items() if k in ('side', 'joint_z', 'notch_x', 'notch_gap', 'notch_vs_mid_mm', 'notch_contrast', 'notch_width_mm', 'notch_dark', 'notch_fs', 'skin_mad', 'knee_width_mm', 'notch_from_med_skin_mm', 'n_notch_primary', 'n_mcl_primary', 'mcl_fov', 'accept', 'error')}, flush=True)
        with open(outp, 'a') as f: f.write(json.dumps(out, default=str) + '\n')
    print('seconds', round(time.time() - t0, 1))
