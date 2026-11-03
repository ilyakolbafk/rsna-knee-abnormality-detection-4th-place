#!/usr/bin/env python3
"""Render the visual-reader input ("menu B") for one study: overview pages with every frame + detail sheets.

Same rendering math as the panels used for the VM labels (window = per-series [0.5, 99.7] percentiles of every 16th
pixel; tiles fitted by PixelSpacing; overview 4x4 pages of 420 px tiles; detail 2x2 pages of 640-960 px tiles; one
fluid-sensitive series per plane in detail, 20 evenly spaced frames on coronal/sagittal, all frames on axial; an
extra wide window on the axial FS series). The production renderer that built the 4,356 reader packets is
labels/vm_reader/build_shard.py; this file is a compact copy of its rendering primitives for figures and
for re-creating a reader packet for a single study.

Usage: python labels/vm_reader/render_reader_panels.py --study-dir <test_series/StudyUID> --series-csv test_series.csv --out out_dir
"""
import argparse
import csv
import math
import statistics
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image, ImageDraw


def window(arrays):
    sample = np.concatenate([a.ravel()[::16] for a in arrays])
    sample = sample[sample > sample.min()]
    lo, hi = np.percentile(sample, [.5, 99.7])
    return [float(lo), float(hi)]


def gray(a, w, photometric):
    a = np.clip((a - w[0]) / (w[1] - w[0]), 0, 1)
    if photometric == 'MONOCHROME1':
        a = 1 - a
    return np.round(a * 255).astype(np.uint8)


def fit(a, spacing, tile):
    ratio = a.shape[1] * spacing[1] / (a.shape[0] * spacing[0])
    width = tile if ratio >= 1 else max(1, round(tile * ratio))
    height = max(1, round(tile / ratio)) if ratio >= 1 else tile
    im = Image.fromarray(a).resize((width, height), Image.Resampling.BILINEAR)
    canvas = Image.new('L', (tile, tile))
    canvas.paste(im, ((tile - width) // 2, (tile - height) // 2))
    return canvas


def even_indices(n, maximum):
    if n <= maximum:
        return list(range(n))
    return sorted({round(j * (n - 1) / (maximum - 1)) for j in range(maximum)})


def load_series(study_dir, rows):
    out = []
    for row in rows:
        frames = []
        for f in sorted((study_dir / row['SeriesInstanceUID']).glob('*.dcm')):
            d = pydicom.dcmread(f)
            iop = np.array([float(x) for x in d.ImageOrientationPatient])
            normal = np.cross(iop[:3], iop[3:])
            z = float(np.dot(np.array([float(x) for x in d.ImagePositionPatient]), normal))
            arr = d.pixel_array.astype(np.float32) * float(getattr(d, 'RescaleSlope', 1) or 1) + float(getattr(d, 'RescaleIntercept', 0) or 0)
            frames.append(dict(z=z, arr=arr, spacing=[float(x) for x in d.PixelSpacing], photometric=str(d.PhotometricInterpretation),
                               matrix=max(int(d.Rows), int(d.Columns))))
        frames.sort(key=lambda r: r['z'])
        steps = np.diff([r['z'] for r in frames]).tolist()
        out.append(dict(uid=row['SeriesInstanceUID'], plane=row['Anatomical_Plane'], fs=row['Fluid_Sensitive'], frames=frames,
                        step=float(statistics.median(steps)) if steps else 3.0, matrix=frames[0]['matrix'], spacing=frames[0]['spacing']))
    return out


def rank(s):
    return (-max(s['spacing']), s['matrix'], -s['step'])


def page(series, indices, tile, cols, w, title):
    rows = math.ceil(len(indices) / cols)
    c = min(cols, len(indices))
    canvas = Image.new('L', (c * tile, 40 + rows * (tile + 24)))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), title, fill=255)
    for pos, i in enumerate(indices):
        r = series['frames'][i]
        x, y = (pos % c) * tile, 40 + (pos // c) * (tile + 24)
        canvas.paste(fit(gray(r['arr'], w, r['photometric']), r['spacing'], tile), (x, y + 24))
        draw.text((x + 5, y + 5), f'i={i:03d} z={r["z"]:.2f}', fill=255)
    return canvas


def render(study_dir, series_csv, out):
    study_dir, out = Path(study_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    uid = study_dir.name
    rows = [r for r in csv.DictReader(open(series_csv)) if r['StudyInstanceUID'] == uid]
    series = load_series(study_dir, rows)
    panels = []
    windows = {}
    for n, s in enumerate(series, 1):  # overview: every frame of every series
        w = windows[s['uid']] = window([f['arr'] for f in s['frames']])
        idx = list(range(len(s['frames'])))
        for p in range(math.ceil(len(idx) / 16)):
            sel = idx[p * 16:(p + 1) * 16]
            im = page(s, sel, 420, 4, w, f'S{n} {s["plane"]} FS={s["fs"]} overview')
            panels.append(('overview', s['plane'], n, im))
    for plane in ('Axial', 'Coronal', 'Sagittal'):  # detail: one series per plane, fluid-sensitive first
        cands = [s for s in series if s['plane'] == plane and s['fs'] == '1'] or [s for s in series if s['plane'] == plane]
        if not cands:
            continue
        s = max(cands, key=rank)
        n = series.index(s) + 1
        if plane == 'Axial':
            stride = max(1, math.ceil(3.0 / s['step'])) if s['step'] < 1.5 else 1
            idx = sorted(set(range(0, len(s['frames']), stride)) | {len(s['frames']) - 1})
        else:
            idx = even_indices(len(s['frames']), 20)
        tile = int(max(640, min(960, s['matrix'])))
        for off in range(0, len(idx), 4):
            panels.append(('detail', plane, n, page(s, idx[off:off + 4], tile, 2, windows[s['uid']], f'S{n} {plane} detail')))
        if plane == 'Axial' and s['fs'] == '1':
            w = windows[s['uid']]
            w2 = [w[0], w[0] + 1.6 * (w[1] - w[0])]
            idx2 = even_indices(len(s['frames']), 8)
            for off in range(0, len(idx2), 4):
                panels.append(('second_window', plane, n, page(s, idx2[off:off + 4], tile, 2, w2, f'S{n} Axial second window')))
    for k, (mode, plane, n, im) in enumerate(panels):
        im.save(out / f'{k:02d}_{mode}_{plane}_S{n}.png')
    return panels


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--study-dir', required=True)
    ap.add_argument('--series-csv', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    ps = render(a.study_dir, a.series_csv, a.out)
    print(len(ps), 'panels:', {m: sum(1 for p in ps if p[0] == m) for m in ('overview', 'detail', 'second_window')})
