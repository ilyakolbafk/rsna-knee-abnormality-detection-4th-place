"""Opt-in, auditable Native64 side and acquisition metadata adapter.

LPS X is not a patient midline. Untagged side is explicitly a geometric
hypothesis, never a verified anatomical label. No diagnosis is read here.
"""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = ('relative_position', 'offset_mm_over_65', 'position_valid',
                 'fs_unknown', 'fs_no', 'fs_yes')


def tag_consensus(values):
    values = {str(x).strip().upper() for x in values if x is not None}
    sides = {dict(L='L', LEFT='L', R='R', RIGHT='R')[x]
             for x in values if x in ('L', 'LEFT', 'R', 'RIGHT')}
    if len(sides) > 1:
        return '', 'tag_conflict'
    if values & {'B', 'BOTH', 'BILATERAL'}:
        return '', 'bilateral'
    return (next(iter(sides)), 'tag_consistent') if sides else ('', 'unknown')


def frame_center(tags):
    iop = np.asarray(tags['ImageOrientationPatient'], float)
    ipp = np.asarray(tags['ImagePositionPatient'], float)
    spacing = np.asarray(tags['PixelSpacing'], float)
    if iop.shape != (6,) or ipp.shape != (3,) or spacing.shape != (2,):
        raise ValueError('Invalid image-plane geometry')
    if (not np.isfinite(iop).all() or not np.allclose(np.linalg.norm(iop.reshape(2,3),axis=1),1,atol=.001)
            or abs(iop[:3] @ iop[3:]) > .001):
        raise ValueError('Invalid orientation')
    result = (ipp + iop[:3] * spacing[1] * (int(tags['Columns']) - 1) / 2
              + iop[3:] * spacing[0] * (int(tags['Rows']) - 1) / 2)
    if not np.isfinite(result).all() or np.any(spacing <= 0):
        raise ValueError('Nonfinite image-plane geometry')
    return result


def side_record(metadata, tag_values=()):
    side, status = tag_consensus(tag_values)
    selected = {sid for sid, s in metadata['series'].items() if s['legacy_slot'] is not None}
    xs = {sid: [] for sid in selected}
    complete_geometry = bool(selected) and all(metadata['series'][sid]['geometry_qc']['within_series_valid'] for sid in selected)
    for p in metadata['planes']:
        if p['SeriesInstanceUID'] in selected:
            if not p.get('geometry_valid', False):
                complete_geometry = False
                continue
            try:
                xs[p['SeriesInstanceUID']].append(float(frame_center(p['geometry_tags'])[0]))
            except (KeyError, ValueError, TypeError):
                complete_geometry = False
    medians = [float(np.median(xs[sid])) for sid in sorted(xs) if xs[sid]]
    x = float(np.median(medians)) if medians else 0.
    gate = (complete_geometry and len(medians) == len(selected) > 0 and abs(x) >= 20
            and (all(v > 0 for v in medians) or all(v < 0 for v in medians)))
    guess = ('L' if x > 0 else 'R') if gate else ''
    if status == 'unknown' and guess:
        side, source = guess, 'geometry_hypothesis'
    else:
        source = status
    return dict(side=side, source=source, tag_status=status,
                geometry_side=guess, median_center_x_mm=x,
                geometry_gate=bool(gate),
                tag_geometry_disagreement=bool(status == 'tag_consistent' and guess and side != guess))


def transform_spec(iop, plane, side, *, min_cosine=.8):
    """Lossless transpose/flips to standard plane axes, then right-to-left.

    Columns/rows target: sagittal +Y/-Z, coronal +X/-Z, axial +X/+Y.
    Depth targets are separate from the reflected image's handedness.
    """
    axes = {'Sagittal': ((1, 1), (2, -1), (0, -1)),
            'Coronal': ((0, 1), (2, -1), (1, 1)),
            'Axial': ((0, 1), (1, 1), (2, 1))}
    if side not in ('L', 'R') or plane not in axes:
        return None
    try:
        col, row = np.asarray(iop, float).reshape(2, 3)
    except (ValueError, TypeError):
        return None
    if not np.isfinite(col).all() or not np.isfinite(row).all():
        return None
    if abs(np.linalg.norm(col)-1) > .02 or abs(np.linalg.norm(row)-1) > .02 or abs(col @ row) > .02:
        return None
    (cx, cs), (ry, rs), (dz, ds) = axes[plane]
    transpose = abs(row[cx]) > abs(col[cx])
    c, r = (row, col) if transpose else (col, row)
    if abs(c[cx]) < min_cosine or abs(r[ry]) < min_cosine:
        return None
    # Right-sided reflection changes in-plane LR for Cor/Ax, depth for Sag.
    if side == 'R':
        if plane == 'Sagittal':
            ds *= -1
        else:
            cs *= -1
    normal = np.cross(col, row)
    if abs(normal[dz]) < min_cosine:
        return None
    return dict(transpose=bool(transpose), flip_x=bool(c[cx]*cs < 0),
                flip_y=bool(r[ry]*rs < 0), reverse_channels=bool(normal[dz]*ds < 0))


def transform_spec_v3(iop, plane, side):
    """Keep obliquity, but allow a unique dominant-axis lossless orientation.

    This is not an orthogonal reslice. Near-45-degree ties still abstain. The
    historical .8 gate remains the default for existing models and caches.
    """
    legacy = transform_spec(iop, plane, side)
    if legacy is not None:
        return legacy
    candidate = transform_spec(iop, plane, side, min_cosine=2 ** -.5)
    if candidate is None:
        return None
    col, row = np.asarray(iop, float).reshape(2, 3)
    c, r = (row, col) if candidate['transpose'] else (col, row)
    axes = {'Sagittal': (1, 2, 0), 'Coronal': (0, 2, 1), 'Axial': (0, 1, 2)}
    for vector, axis in zip((c, r, np.cross(col, row)), axes[plane]):
        magnitudes = np.abs(vector)
        if magnitudes[axis] - max(np.delete(magnitudes, axis)) < .1:
            return None
    return candidate


def adapt_inputs(tokens, arrays, metadata, sample, options, record):
    """Keep centers/masks fixed. Return model input, metadata and audit records.

    Position is relative to the complete acquired series, not an inferred joint
    center. FS is the supplied acquisition annotation; fluid-sensitive != FS.
    """
    import torch
    features = np.zeros((len(tokens), len(FEATURE_NAMES)), np.float32)
    features[:, 3] = 1
    by_ref = {int(p['bank_index']): p for p in metadata['planes']}
    by_index = {int(s['series_index']): (sid, s) for sid, s in metadata['series'].items()}
    projection = {}
    for sid, s in metadata['series'].items():
        values = []
        for ref in range(int(s['offset']), int(s['offset']+s['source_count'])):
            p = by_ref.get(ref)
            if p is None or not p.get('geometry_valid', False):
                continue
            try:
                g = p['geometry_tags']; iop = np.asarray(g['ImageOrientationPatient'], float)
                frame_center(g)  # fail closed on malformed/nonfinite geometry
                n = np.cross(iop[:3], iop[3:])
                values.append((ref, float(np.asarray(g['ImagePositionPatient'], float) @ n)))
            except (KeyError, ValueError, TypeError):
                continue
        if values and s['geometry_qc']['within_series_valid']:
            coords = dict(values); lo, hi = min(coords.values()), max(coords.values())
            projection[sid] = coords, lo, hi
    audit = []
    for t, ref in enumerate(sample['center_refs']):
        if ref < 0 or not sample['token_mask'][t]:
            continue
        sid, series = by_index[int(arrays['series_index'][ref])]
        plane = metadata['legacy_slots'][int(sample['slot_ids'][t])]['plane']
        spec = None; p = by_ref.get(int(ref))
        if options.get('canonical_side') and p and p.get('geometry_valid') and series['geometry_qc']['within_series_valid']:
            orient = transform_spec_v3 if options.get('canonical_policy_version') == 3 else transform_spec
            spec = orient(p['geometry_tags']['ImageOrientationPatient'], plane, record['side'])
        if spec is not None:
            x = tokens[t]
            if spec['transpose']: x = x.transpose(-1, -2)
            if spec['flip_x']: x = x.flip(-1)
            if spec['flip_y']: x = x.flip(-2)
            if spec['reverse_channels']: x = x.flip(0)
            tokens[t] = x.clone()
        if options.get('position') and sid in projection and ref in projection[sid][0]:
            coords, lo, hi = projection[sid]
            if hi > lo:
                offset = coords[int(ref)] - (lo+hi)/2
                if spec and spec['reverse_channels']: offset *= -1
                features[t, :3] = [2*offset/(hi-lo), np.clip(offset/65., -4, 4), 1]
        if options.get('fat_suppression'):
            fs = series.get('source_inventory', {}).get('fat_suppression')
            features[t, 3:6] = [0, 0, 0]
            features[t, 4+int(fs) if fs in (0, 1) else 3] = 1
        channel_refs = np.asarray(sample.get('bank_refs', np.tile([-1,-1,-1], (len(tokens),1)))[t])
        if spec and spec['reverse_channels']:
            channel_refs = channel_refs[::-1]
        audit.append(dict(token=t, center_ref=int(ref), transform=spec,
                          model_bank_refs=channel_refs.tolist(),
                          side=record['side'], side_source=record['source']))
    return tokens, torch.from_numpy(features), audit
