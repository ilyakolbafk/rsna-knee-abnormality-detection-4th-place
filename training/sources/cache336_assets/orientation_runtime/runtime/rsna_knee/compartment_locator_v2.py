"""Portable pydicom adapter for the approved two-plane compartment locator.

The numerical profile/option-selection functions are ported from the recorded
prototype without threshold tuning. Unknown side can produce a height estimate,
but never an eligible anatomical compartment. No trained segmentation is used.
"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import pydicom

PROTOTYPE_SHA256 = '9da547d451f0ce9287d9527b1860e0b3f6471c3d38a6dcbcec783f0ade089ae6'


def read_dicom(path, *, expected_sha256=None):
    path = Path(path)
    if expected_sha256 is not None:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256
    ds = pydicom.dcmread(path)
    px = ds.pixel_array
    if px.ndim != 2:
        raise ValueError('Only single-frame monochrome DICOM is supported')
    if int(getattr(ds, 'SamplesPerPixel', 1)) != 1:
        raise ValueError('Unsupported multichannel pixels')
    arr = px.astype(np.float32)
    arr = arr * float(getattr(ds, 'RescaleSlope', 1) or 1) + float(getattr(ds, 'RescaleIntercept', 0) or 0)
    iop = np.asarray(ds.ImageOrientationPatient, float)
    ipp = np.asarray(ds.ImagePositionPatient, float)
    spacing = np.asarray(ds.PixelSpacing, float)
    if iop.shape != (6,) or ipp.shape != (3,) or spacing.shape != (2,) or not np.isfinite(np.r_[iop,ipp,spacing]).all() or (spacing <= 0).any():
        raise ValueError('Invalid DICOM geometry')
    if abs(np.dot(iop[:3],iop[3:])) > .02 or not np.allclose(np.linalg.norm(iop.reshape(2,3),axis=1),1,atol=.02):
        raise ValueError('Invalid orientation vectors')
    syntax = ds.file_meta.TransferSyntaxUID
    return dict(pixels=arr, iop=iop, ipp=ipp, spacing=spacing,
                rows=arr.shape[0], cols=arr.shape[1], series=str(ds.SeriesInstanceUID),
                uid=str(ds.StudyInstanceUID), sop=str(getattr(ds,'SOPInstanceUID','')),
                instance=str(getattr(ds,'InstanceNumber','')), desc=str(getattr(ds,'SeriesDescription','')),
                frame=str(getattr(ds,'FrameOfReferenceUID','')), laterality=str(getattr(ds,'Laterality','')),
                image_laterality=str(getattr(ds,'ImageLaterality','')),
                transfer_syntax_uid=str(syntax), compressed=bool(syntax.is_compressed), source_file=str(path))


def load_study(root, expected=None):
    series = {}
    for path in sorted(Path(root).glob('*/*.dcm')):
        want = expected.get(path.name) if expected else None
        d = read_dicom(path, expected_sha256=want)
        series.setdefault(d['series'], []).append(d)
    if not series:
        raise ValueError('No source DICOM found')
    for sid, frames in series.items():
        normal = np.cross(frames[0]['iop'][:3],frames[0]['iop'][3:])
        frames.sort(key=lambda d:(float(np.dot(d['ipp'],normal)), d['sop']))
    return series


def plane_of(s):
    n = np.abs(np.cross(s[0]['iop'][:3], s[0]['iop'][3:])); return ['Sagittal', 'Coronal', 'Axial'][int(np.argmax(n))]

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

def analyse_series(uid, series, meta, side):
    # Unknown-side ordering is used only to estimate a side-independent height.
    # It never makes an unknown side eligible for a compartment crop.
    side_known = side in ('L', 'R')
    recorded_side = side if side_known else None
    side = side if side_known else 'R'
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
    res['side'] = recorded_side
    res['side_known'] = side_known
    res['crop_eligible'] = bool(side_known and res.get('joint_z') is not None)
    res['crop_exclusion'] = None if res['crop_eligible'] else ('unknown_side' if not side_known else 'no_coherent_option')
    if not side_known:
        res['z_medial'] = res['z_lateral'] = None
    res['compartment_z_used'] = bool(side_known and res.get('z_medial') is not None and res.get('z_lateral') is not None and abs(res['z_medial']-res['z_lateral']) <= 10)
    for comp in ['medial','lateral']:
        res['crop_z_'+comp] = res.get('z_'+comp) if res['compartment_z_used'] else res.get('joint_z')
    res['prototype_sha256'] = PROTOTYPE_SHA256
    return series, info, C, res
