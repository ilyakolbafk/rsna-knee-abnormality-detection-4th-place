#!/usr/bin/env python3
"""Re-render the frozen compact menu B (Codex STEP1_MENU_SPEC) for Gold58 cases from the DICOM archives.

Reproduces prepare_step1.py (select_detail, even_indices, detail_panels) and reader_technical.py
(window, gray, fit, baseline_fullframe420 overview) without the run-ledger guards.
Never reads label columns: train.csv is opened only for StudyInstanceUID and Report.

Usage: render_menu_b.py --root "<repo root>" --out "<out dir>" --cases G011 G013 ... [--allowlist-order]
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, math, statistics, tarfile, time
from pathlib import Path
import numpy as np
import pydicom
from PIL import Image, ImageDraw

BUDGET = 42
TAGS = ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID', 'Rows', 'Columns', 'NumberOfFrames',
        'ImageOrientationPatient', 'ImagePositionPatient', 'PixelSpacing', 'SliceThickness', 'SpacingBetweenSlices',
        'Laterality', 'ImageLaterality', 'InstanceNumber', 'PhotometricInterpretation', 'RescaleSlope',
        'RescaleIntercept', 'BitsAllocated', 'BitsStored', 'PixelRepresentation', 'SamplesPerPixel']


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def scalar(d, k):
    v = getattr(d, k, None)
    if v is None: return None
    if isinstance(v, str) and not v.strip(): return None
    if isinstance(v, (list, pydicom.multival.MultiValue)): return [float(x) for x in v]
    if isinstance(v, (int, float)): return float(v)
    return str(v)


# ---- renderer primitives (identical math to reader_technical.py) ----
def window(arrays):
    sample = np.concatenate([a.ravel()[::16] for a in arrays]); sample = sample[sample > sample.min()]
    if not len(sample): raise ValueError('constant_or_empty_sample')
    lo, hi = np.percentile(sample, [.5, 99.7])
    if not hi > lo: raise ValueError('constant_or_empty_sample')
    return [float(lo), float(hi)]


def gray(a, w, photometric):
    a = np.clip((a - w[0]) / (w[1] - w[0]), 0, 1)
    if photometric == 'MONOCHROME1': a = 1 - a
    return np.round(a * 255).astype(np.uint8)


def fit(a, spacing, tile):
    ratio = a.shape[1] * spacing[1] / (a.shape[0] * spacing[0])
    width = tile if ratio >= 1 else max(1, round(tile * ratio))
    height = max(1, round(tile / ratio)) if ratio >= 1 else tile
    im = Image.fromarray(a).resize((width, height), Image.Resampling.BILINEAR)
    canvas = Image.new('L', (tile, tile)); canvas.paste(im, ((tile - width) // 2, (tile - height) // 2))
    return canvas, [width, height]


# ---- menu functions (prepare_step1.py) ----
def step(s):
    values = s.get('steps_mm') or []
    return float(statistics.median(values)) if values else 3.0


def native_matrix(s):
    first = s['frames'][0]; return max(first['Rows'], first['Columns'])


def rank(s):
    first = s['frames'][0]
    return (-max(first['PixelSpacing']), first['Rows'] * first['Columns'], -step(s), s['series_id'])


def even_indices(n, maximum):
    if n <= maximum: return list(range(n))
    return sorted({round(j * (n - 1) / (maximum - 1)) for j in range(maximum)})


def select_detail(series):
    by = {}
    for plane in ('Axial', 'Coronal', 'Sagittal'):
        for fs in ('1', '0'):
            available = [s for s in series if s['anatomical_plane'] == plane and s['fluid_sensitive'] == fs]
            if available: by[(plane, fs)] = max(available, key=rank)
    selections = []; primaries = {}
    for plane in ('Axial', 'Coronal', 'Sagittal'):
        chosen = by.get((plane, '1')) or by.get((plane, '0'))
        if chosen is None: continue
        primaries[plane] = chosen
        if plane == 'Axial':
            spacing = step(chosen)
            stride = max(1, math.ceil(3.0 / spacing)) if spacing < 1.5 else 1
            indices = sorted(set(range(0, chosen['n_frames'], stride)) | {chosen['n_frames'] - 1})
        else:
            indices = even_indices(chosen['n_frames'], 20)
        selections.append({'series_id': chosen['series_id'], 'mode': f'detail_{plane.lower()}', 'indices': indices, 'alternate_window': False})
    axial_fs = by.get(('Axial', '1'))
    if axial_fs is not None:
        selections.append({'series_id': axial_fs['series_id'], 'mode': 'axial_second_window', 'indices': even_indices(axial_fs['n_frames'], 8), 'alternate_window': True})
    axial_other = by.get(('Axial', '0'))
    if axial_other is not None and axial_other is not primaries.get('Axial') and native_matrix(axial_other) >= 720:
        selections.append({'series_id': axial_other['series_id'], 'mode': 'axial_anatomic_aux', 'indices': even_indices(axial_other['n_frames'], 16), 'alternate_window': False})
    for plane in ('Sagittal', 'Coronal'):
        other, primary = by.get((plane, '0')), primaries.get(plane)
        if other is not None and primary is not None and other is not primary and native_matrix(other) >= 1.25 * native_matrix(primary):
            selections.append({'series_id': other['series_id'], 'mode': f'{plane.lower()}_anatomic_aux', 'indices': even_indices(other['n_frames'], 8), 'alternate_window': False})
    return selections


# ---- DICOM loading ----
def load_case(archive, uid, allowed):
    """allowed: ordered list of dicts {series_uid, fluid_sensitive, fat_suppression, anatomical_plane}."""
    groups = {s['series_uid']: [] for s in allowed}; pix = {}
    with tarfile.open(archive, 'r:gz') as t:
        for m in t.getmembers():
            if not m.isfile() or not m.name.lower().endswith('.dcm'): continue
            raw = t.extractfile(m).read()
            d = pydicom.dcmread(io.BytesIO(raw))
            assert str(d.StudyInstanceUID) == uid
            suid = str(d.SeriesInstanceUID)
            if suid not in groups: continue  # series outside the allowlist are ignored
            assert int(getattr(d, 'NumberOfFrames', 1)) == 1
            row = {k: scalar(d, k) for k in TAGS}
            row.update(member=m.name, source_sha256=hashlib.sha256(raw).hexdigest())
            iop = np.array(row['ImageOrientationPatient'], float); ipp = np.array(row['ImagePositionPatient'], float)
            normal = np.cross(iop[:3], iop[3:]); assert abs(np.linalg.norm(normal) - 1) < 0.01
            assert row['PhotometricInterpretation'] in ['MONOCHROME1', 'MONOCHROME2'] and int(row['SamplesPerPixel']) == 1
            row['physical_coordinate'] = float(np.dot(ipp, normal)); row['normal'] = normal.tolist()
            slope = float(getattr(d, 'RescaleSlope', 1) or 1); intercept = float(getattr(d, 'RescaleIntercept', 0) or 0)
            pix[m.name] = d.pixel_array.astype(np.float32) * slope + intercept
            groups[suid].append(row)
    series = []
    for n, s in enumerate(allowed, 1):
        rows = sorted(groups[s['series_uid']], key=lambda r: r['physical_coordinate'])
        assert rows, ('empty series', s['series_uid'])
        assert len({r['physical_coordinate'] for r in rows}) == len(rows)
        assert len({(r['Rows'], r['Columns'], str(r['PixelSpacing'])) for r in rows}) == 1
        normal = rows[0]['normal']; plane = ['Sagittal', 'Coronal', 'Axial'][int(np.argmax(np.abs(normal)))]
        assert plane == s['anatomical_plane'], (s['series_uid'], plane, s['anatomical_plane'])
        for i, r in enumerate(rows): r['physical_index'] = i
        series.append(dict(s, series_id=f"{s['case_id']}_S{n}", n_frames=len(rows),
                           steps_mm=np.diff([r['physical_coordinate'] for r in rows]).tolist(), frames=rows,
                           arrays=[pix[r['member']] for r in rows]))
    return series


# ---- panels ----
def overview_panels(case, s, out):
    arrays = s['arrays']; w = window(arrays); tile = 420; cols = 4; perpage = 16; recs = []
    indices = list(range(len(arrays)))
    for page in range(math.ceil(len(indices) / perpage)):
        selected = indices[page * perpage:(page + 1) * perpage]; c = min(cols, len(selected)); rows = math.ceil(len(selected) / c)
        canvas = Image.new('L', (c * tile, 40 + rows * (tile + 24))); draw = ImageDraw.Draw(canvas)
        draw.text((8, 10), f'{s["series_id"]} {s["anatomical_plane"]} FS={s["fat_suppression"]} win={w[0]:.1f},{w[1]:.1f}', fill=255)
        frames = []
        for pos, i in enumerate(selected):
            r = s['frames'][i]; im, size = fit(gray(arrays[i], w, r['PhotometricInterpretation']), r['PixelSpacing'], tile)
            x = (pos % c) * tile; y = 40 + (pos // c) * (tile + 24)
            canvas.paste(im, (x, y + 24)); draw.text((x + 5, y + 5), f'i={i:03d} z={r["physical_coordinate"]:.2f}', fill=255)
            frames.append({'index': i, 'sop_uid': r['SOPInstanceUID'], 'source_sha256': r['source_sha256'], 'resized_dimensions': size})
        variant = hashlib.sha256(json.dumps({'indices': selected, 'crop': None, 'tile': tile, 'window': w, 'columns': c}, sort_keys=True).encode()).hexdigest()[:10]
        pid = f'{s["series_id"]}_{selected[0]:03d}-{selected[-1]:03d}_full_{variant}'
        path = out / 'panels' / f'{pid}.png'; canvas.save(path)
        recs.append({'panel_id': pid, 'mode': 'overview_all_frames', 'path': str(path), 'sha256': sha(path), 'dimensions': list(canvas.size),
                     'series_id': s['series_id'], 'series_uid': s['series_uid'], 'indices': selected, 'window': w, 'tile_px': tile, 'frames': frames})
    return recs, w


def detail_panels(case, series, menu, out, windows):
    by = {s['series_id']: s for s in series}; recs = []
    for item in menu:
        s = by[item['series_id']]; arrays = s['arrays']; nw = windows[s['series_id']]
        w = nw if not item['alternate_window'] else [nw[0], nw[0] + 1.6 * (nw[1] - nw[0])]
        tile = int(max(640, min(960, native_matrix(s)))); indices = item['indices']
        for off in range(0, len(indices), 4):
            page = indices[off:off + 4]; cols = min(2, len(page)); rows = math.ceil(len(page) / cols)
            canvas = Image.new('L', (cols * tile, 40 + rows * (tile + 24))); draw = ImageDraw.Draw(canvas)
            draw.text((8, 10), f"{s['series_id']} {s['anatomical_plane']} {item['mode']} win={w[0]:.1f},{w[1]:.1f}", fill=255)
            frames = []
            for pos, index in enumerate(page):
                r = s['frames'][index]; im, size = fit(gray(arrays[index], w, r['PhotometricInterpretation']), r['PixelSpacing'], tile)
                x = (pos % cols) * tile; y = 40 + (pos // cols) * (tile + 24)
                canvas.paste(im, (x, y + 24)); draw.text((x + 5, y + 5), f'i={index:03d} z={r["physical_coordinate"]:.2f}', fill=255)
                frames.append({'index': index, 'sop_uid': r['SOPInstanceUID'], 'source_sha256': r['source_sha256'], 'resized_dimensions': size})
            identity = hashlib.sha256(json.dumps({'series': s['series_id'], 'mode': item['mode'], 'indices': page, 'window': w, 'tile': tile}, sort_keys=True).encode()).hexdigest()[:10]
            pid = f"{s['series_id']}_{item['mode']}_{page[0]:03d}-{page[-1]:03d}_{identity}"
            path = out / 'panels' / f'{pid}.png'; canvas.save(path)
            recs.append({'panel_id': pid, 'mode': item['mode'], 'path': str(path), 'sha256': sha(path), 'dimensions': list(canvas.size),
                         'series_id': s['series_id'], 'series_uid': s['series_uid'], 'indices': page, 'window': w, 'tile_px': tile, 'frames': frames})
    return recs


def report_for(root, uid):
    p = root / '.codex/runs/rsna-knee-research/metadata/train.csv'
    with p.open(newline='') as f:
        r = csv.reader(f); hdr = next(r); iu = hdr.index('StudyInstanceUID'); ir = hdr.index('Report')
        for row in r:
            if row[iu] == uid: return row[ir]
    raise KeyError(uid)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--cases', nargs='+', required=True); ap.add_argument('--allowlist', help='dir with <case>/inputs/SERIES_ALLOWLIST.json to fix series order')
    ap.add_argument('--data-dir', default='data/gold58_mri', help='dataset dir with <prefix>_manifest.csv, <prefix>_series.csv, archives/')
    ap.add_argument('--prefix', default='gold58', help='manifest/series file prefix: gold58 or studies')
    ap.add_argument('--index-col', default='gold_study_index', help='manifest index column (gold_study_index)')
    ap.add_argument('--case-prefix', default='G', help='case id letter: G for gold, P for pilot')
    a = ap.parse_args(); root = Path(a.root); out_root = Path(a.out)
    data_dir = root / a.data_dir
    manifest = {int(r[a.index_col]): r for r in csv.DictReader((data_dir / f'{a.prefix}_manifest.csv').open())}
    series_csv = list(csv.DictReader((data_dir / f'{a.prefix}_series.csv').open()))
    for case in a.cases:
        t0 = time.time(); idx = int(case[1:]); m = manifest[idx]; uid = m['StudyInstanceUID']
        archive = data_dir / m['archive_file']
        assert archive.is_file() and archive.stat().st_size == int(float(m['archive_bytes']))
        if a.allowlist and (Path(a.allowlist) / case / 'inputs/SERIES_ALLOWLIST.json').is_file():
            allowed = json.loads((Path(a.allowlist) / case / 'inputs/SERIES_ALLOWLIST.json').read_text())
        else:
            allowed = [{'case_id': case, 'study_uid': uid, 'series_uid': r['SeriesInstanceUID'], 'fluid_sensitive': r['Fluid_Sensitive'],
                        'fat_suppression': r['Fat_Suppression'], 'anatomical_plane': r['Anatomical_Plane']} for r in series_csv if r['StudyInstanceUID'] == uid]
        assert allowed, case
        out = out_root / case; (out / 'panels').mkdir(parents=True, exist_ok=True)
        series = load_case(archive, uid, allowed)
        menu = select_detail(series)
        predicted = sum(math.ceil(s['n_frames'] / 16) for s in series) + sum(math.ceil(len(m['indices']) / 4) for m in menu)
        dropped = []; used = {m['series_id'] for m in menu}; skip_overview = set()
        if predicted > BUDGET:
            for s in series:
                if s['fluid_sensitive'] == '0' and s['series_id'] not in used and predicted > BUDGET:
                    skip_overview.add(s['series_id']); predicted -= math.ceil(s['n_frames'] / 16); dropped.append(f"overview:{s['series_id']}")
        if predicted > BUDGET:
            sw = [m for m in menu if m['mode'] == 'axial_second_window']
            for item in sw:
                menu.remove(item); predicted -= math.ceil(len(item['indices']) / 4); dropped.append(f"axial_second_window:{item['series_id']}")
        panels = []; windows = {}
        for s in series:
            if s['series_id'] in skip_overview:
                windows[s['series_id']] = window(s['arrays']); continue
            recs, w = overview_panels(case, s, out); panels.extend(recs); windows[s['series_id']] = w
        panels.extend(detail_panels(case, series, menu, out, windows))
        assert len({p['panel_id'] for p in panels}) == len(panels) == predicted, (len(panels), predicted)
        reader_input = {'case_id': case, 'study_uid': uid,
                        'series': [{'series_id': s['series_id'], 'plane': s['anatomical_plane'], 'fluid_sensitive': s['fluid_sensitive'], 'frames': s['n_frames'], 'physical_step_mm': step(s)} for s in series],
                        'panel_count': len(panels), 'panel_budget': BUDGET, 'panels_dropped_for_budget': dropped,
                        'panels': [{'order': i + 1, 'panel_id': p['panel_id'], 'file': f'/home/claude/opus_b20/cases/{case}/panels/{p["panel_id"]}.png', 'mode': p['mode'],
                                    'series_id': p['series_id'], 'indices': p['indices'], 'window': p['window'], 'tile_px': p['tile_px'], 'png_dimensions': p['dimensions']} for i, p in enumerate(panels)]}
        (out / 'READER_INPUT.json').write_text(json.dumps(reader_input, ensure_ascii=False, indent=1))
        (out / 'REPORT.txt').write_text(report_for(root, uid))
        pm = {'case_id': case, 'study_uid': uid, 'archive_sha256': m['archive_sha256'], 'menu': 'STEP1_MENU_SPEC compact B (re-rendered)',
              'series': [{k: s[k] for k in ('series_id', 'series_uid', 'anatomical_plane', 'fluid_sensitive', 'fat_suppression', 'n_frames')} | {'physical_step_mm': step(s)} for s in series],
              'detail_selections': menu, 'panels': [{k: v for k, v in p.items() if k != 'frames'} | {'source_frame_sha256': [f['source_sha256'] for f in p['frames']]} for p in panels],
              'panel_count': len(panels), 'panel_budget': BUDGET, 'panels_dropped_for_budget': dropped}
        (out / 'PANEL_MANIFEST.json').write_text(json.dumps(pm, ensure_ascii=False, indent=1))
        for s in series: s['arrays'] = None
        print(f'{case}: {len(panels)} panels (dropped {len(dropped)}), {len(series)} series, {sum(s["n_frames"] for s in series)} frames, {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
