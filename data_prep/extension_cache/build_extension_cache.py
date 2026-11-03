#!/usr/bin/env python3
"""Additional K24 native-336 bank. Never modifies the frozen 64-token bank.

Run with the exported, verified v8.3 runtime. Gold archives are expanded one
study at a time; train/test use their original DICOM directories. No network IO.
"""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

FORMAT = 'k24-allseries-ext-v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def boolean(x):
    return str(x).strip().lower() in ('true', '1', 'yes')


def bind_runtime(root):
    root = Path(root)
    manifest = json.loads((root.parent / 'MANIFEST.json').read_text())
    for name, digest in manifest['files'].items():
        if sha(root / name) != digest:
            raise RuntimeError(f'Frozen runtime mismatch: {name}')
    sys.path[:0] = [str(root), str(root / 'rsna_knee')]
    global DecodedStudy, SLOTS, norm, select_series_for_cache, sample_window_indices
    global select_centers, source_triplets, apply_tokens, adapt_inputs, transform_spec, transform_spec_v3
    from raw_inputs import DecodedStudy, SLOTS, norm
    from rsna_knee.grayscale_cache import select_series_for_cache, sample_window_indices
    from rsna_knee.central_sampling import select_centers, source_triplets
    from h1_adapter import apply_tokens
    from canonical_inputs import adapt_inputs, transform_spec, transform_spec_v3


def apply_extra_tokens(raw, arrays, metadata, sample, decision):
    """Exact adapter sequence with the approved v3 geometry option, ext only.

    The original production adapter is immutable and retains the legacy gate
    for all 64 base tokens. Both pixel and positional paths use v3 here.
    """
    options = dict(canonical_side=True, position=True, fat_suppression=True,
        canonical_policy_version=3)
    generic, _, _ = adapt_inputs(torch.as_tensor(raw).clone(), arrays, metadata,
        sample, options, {'side':'L','source':'generic_geometry_base_v3'})
    _, features, _ = adapt_inputs(torch.zeros_like(torch.as_tensor(raw)), arrays,
        metadata, sample, options, {'side':decision['side'],'source':'laterality_v8.3_v3'})
    bysid={s['sid']:s for s in decision['series']}
    byref={int(p['bank_index']):p for p in metadata['planes']}
    chosen=np.flatnonzero(sample['token_mask'])
    for t in chosen:
        s=bysid[byref[int(sample['center_refs'][t])]['SeriesInstanceUID']]
        im=generic[t];pose=s.get('apply_pose');plane=s['plane']
        if pose:
            im=torch.rot90(im,int(pose['rot90_k']),(-2,-1))
            if pose.get('flip_x'):im=im.flip(-1)
        if s.get('appearance_side',decision['side'])=='R':
            im=im.flip(-1) if plane in ('Axial','Coronal') else im.flip(0)
        if s.get('reverse_depth_extra'):
            im=im.flip(0);features[t,:2]*=-1
        generic[t]=im
    return (generic[chosen].float()/255.,torch.ones(len(chosen),dtype=torch.bool),features[chosen]),[]


def render(arrays, metadata, centers, slots, decision, *, extension_v3=False):
    """Common assembly for both selected-series parity and extra series."""
    sample = source_triplets(arrays, metadata, centers, slots)
    sample['center_refs'] = centers
    refs = sample['bank_refs']
    raw = np.zeros((len(centers), 3, 336, 336), np.uint8)
    ok = refs >= 0
    raw[ok] = arrays['images'][refs[ok]]
    valid = np.flatnonzero(sample['token_mask'])
    native = np.zeros(len(centers), bool)
    orient=np.full(len(centers),'legacy',dtype='U6')
    if extension_v3:
        byref = {int(p['bank_index']): p for p in metadata['planes']}
        for t in valid:
            p = byref[int(centers[t])]
            plane = metadata['legacy_slots'][int(slots[t])]['plane']
            iop=p['geometry_tags'].get('ImageOrientationPatient')
            native[t] = (not p['geometry_valid'] or transform_spec_v3(iop,plane,'L') is None)
            if transform_spec(iop,plane,'L') is None:orient[t]='v3'
        sample['token_mask'] &= ~native
        valid=np.flatnonzero(sample['token_mask'])
    _, features, _ = adapt_inputs(torch.zeros_like(torch.from_numpy(raw)), arrays,
        metadata, sample, {'position': True, 'fat_suppression': True},
        {'side': decision['side'], 'source': 'v8.3_native_abstention'})
    pixels = raw.copy()
    canonical = sample['token_mask'] & ~native
    if canonical.any():
        canonical_sample = dict(sample, token_mask=canonical)
        adapter=apply_extra_tokens if extension_v3 else apply_tokens
        (p, _, m), _ = adapter(raw, arrays, metadata, canonical_sample, decision)
        pixels[canonical] = np.rint(p.numpy() * 255).clip(0, 255).astype(np.uint8)
        features[canonical] = m
    return (pixels[valid], features.numpy()[valid].astype(np.float32),
            sample['bank_refs'][valid], orient[valid], sample['neighbor_status'][valid],
            centers[valid], int(native.sum()))


def plane_from_iop(s):
    normals = []
    for tags, valid in zip(s['tags'], s['geometry_valid']):
        if valid and len(tags.get('ImageOrientationPatient', [])) == 6:
            iop = np.asarray(tags['ImageOrientationPatient'], float)
            n = np.cross(iop[:3], iop[3:])
            if np.isfinite(n).all() and np.linalg.norm(n) > .9:
                normals.append(n / np.linalg.norm(n))
    if not normals:
        return s['row']['Anatomical_Plane'], None
    n = normals[0]
    return ('Sagittal', 'Coronal', 'Axial')[int(np.argmax(np.abs(n)))], n


def normalization_spec(row):
    plane = row['Anatomical_Plane']
    fluid = int(float(row['Fluid_Sensitive']))
    for spec in SLOTS:
        if spec['plane'] == plane and spec['fluid_sensitive'] == fluid:
            return spec
    if plane == 'Axial':
        return SLOTS[4]  # axial non-FS uses the identical axial statistics frames
    raise ValueError(f'Unsupported series type: {plane}/{fluid}')


def series_arrays(study, sid):
    s = study.series[sid]
    spec = normalization_spec(s['row'])
    ids = sample_window_indices(len(s['valid']), spec['slices'], *spec['window'])
    q, low, high = norm(s['float140'], s['valid'], ids)
    plane, normal = plane_from_iop(s)
    fs = pd.to_numeric(s['row'].get('Fat_Suppression'), errors='coerce')
    fs = int(fs) if fs in (0, 1) else -1
    n = len(q)
    record = {'offset': 0, 'source_count': n, 'series_index': 0,
        'legacy_slot': 0, 'geometry_qc': s['qc'],
        'source_inventory': {'fat_suppression': fs if fs >= 0 else None},
        'normalization_low': low, 'normalization_high': high}
    arrays = {'images': q, 'decode_valid': s['valid'],
        'geometry_valid': s['geometry_valid'], 'series_index': np.zeros(n, np.int32),
        'source_index': np.arange(n, dtype=np.int32)}
    metadata = {'legacy_slots': [dict(spec, plane=plane)], 'series': {sid: record},
        'uid': study.uid, 'planes': [dict(bank_index=i, SeriesInstanceUID=sid,
        geometry_tags=g, geometry_valid=bool(s['geometry_valid'][i]))
        for i, g in enumerate(s['tags'])]}
    return arrays, metadata, plane, normal, fs, ids


def pack(path, pixels, meta6, columns, identity):
    lookup, bank, refs = {}, [], []
    for frame in pixels.reshape(-1, 336, 336):
        key = array_sha(frame)
        if key not in lookup:
            lookup[key] = len(bank)
            bank.append(frame)
        elif not np.array_equal(frame, bank[lookup[key]]):
            raise RuntimeError('Frame hash collision')
        refs.append(lookup[key])
    typed_columns = {}
    for k, v in columns.items():
        dtype = ('U128' if k in ('series_uid', 'plane', 'dcls','orient') else
            bool if k in ('is_oblique', 'native_orientation') else np.int32)
        typed_columns[k] = np.asarray(v, dtype=dtype)
    typed_columns['neighbor_status'] = typed_columns['neighbor_status'].reshape(-1, 3)
    payload = dict(format=np.asarray(FORMAT),
        bank=np.stack(bank) if bank else np.zeros((0, 336, 336), np.uint8),
        refs=np.asarray(refs, np.int32).reshape(-1, 3), mask=np.ones(len(pixels), bool),
        meta6=meta6, **typed_columns,
        **{k: np.asarray(v) for k, v in identity.items()})
    tmp = path.with_suffix('.npz.partial')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **payload)
    with np.load(tmp, allow_pickle=False) as z:
        if not np.array_equal(z['bank'][z['refs']], pixels):
            raise RuntimeError('Lossless bank roundtrip failed')
        for k, v in payload.items():
            if not np.array_equal(z[k], v):
                raise RuntimeError(f'Roundtrip mismatch: {k}')
    tmp.replace(path)
    return {'sha256': sha(path), 'bytes': path.stat().st_size,
        'unique_frames': len(bank), 'pixels_sha256': array_sha(pixels),
        'meta6_sha256': array_sha(meta6)}


def parity(study, decision, reference_row, reference_root=None):
    arrays, metadata = study.h1_bank(decision)
    centers, slots, _ = select_centers(arrays, metadata, budget=64,
        trim_each_side=[.1, .1, .1, .1, 0], slot_weights=[1] * 5)
    pixels, meta, *_ = render(arrays, metadata, centers, slots, decision)
    # Exercise the extension-specific normalization and metadata constructor on
    # the ORIGINAL base centers; a second call to h1_bank would miss that path.
    cross_pixels = np.zeros((64, 3, 336, 336), np.uint8)
    cross_meta = np.zeros((64, 6), np.float32)
    for sid, rec in metadata['series'].items():
        at = np.flatnonzero((slots == rec['legacy_slot']) & (centers >= 0))
        a, m2, _, _, _, _ = series_arrays(study, sid)
        local_centers = centers[at] - rec['offset']
        px, mt, *_ = render(a, m2, local_centers,
            np.zeros(len(at), np.int64), decision)
        cross_pixels[at] = px
        cross_meta[at] = mt
    padded = np.zeros((64, 3, 336, 336), np.uint8)
    padded[:len(pixels)] = pixels
    m = np.zeros((64, 6), np.float32)
    m[:len(meta)] = meta
    mask = np.arange(64) < len(meta)
    hashes = {'pixels_sha256': array_sha(padded), 'meta_sha256': array_sha(m),
        'mask_sha256': array_sha(mask)}
    matches = {k: value == reference_row[k] for k, value in hashes.items()}
    result = {'StudyInstanceUID': study.uid, 'exact_hash_match': all(matches.values()),
        'matches': matches, 'cross_path_pixels_exact': bool(np.array_equal(cross_pixels,padded)),
        'cross_path_meta_max_error': float(np.abs(cross_meta-m).max()), **hashes}
    if reference_root is not None:
        path = Path(reference_root) / reference_row['cache_file']
        with np.load(path, allow_pickle=False) as z:
            ref = z['bank'][z['refs']]
            delta = np.abs(padded.astype(np.int16) - ref.astype(np.int16))
            result.update(exact_frame_fraction=float((delta == 0).all(axis=(-2,-1)).mean()),
                total_frames=int(np.prod(delta.shape[:2])),
                nonexact_frames=int((delta != 0).any(axis=(-2,-1)).sum()),
                nonexact_pixels=int((delta != 0).sum()),
                max_lsb=int(delta.max()), max_meta_error=float(np.abs(m-z['meta']).max()),
                mask_exact=bool(np.array_equal(mask, z['mask'])))
    numeric_ok = result['exact_hash_match'] or (reference_root is not None
        and result['max_lsb'] <= 1 and result['max_meta_error'] <= 1e-6
        and result['mask_exact'])
    result['pass'] = bool(numeric_ok and result['cross_path_pixels_exact']
        and result['cross_path_meta_max_error'] <= 1e-6)
    return result


def train_decision(manifest_rows, op_rows):
    routes = manifest_rows.side.tolist()
    series = []
    for row in op_rows.to_dict('records'):
        pose = row['apply_pose'].strip()
        series.append(dict(sid=row['sid'], plane=row['plane'],
            assigned_side=row['assigned_side'], appearance_side=row['appearance_side'],
            apply_pose=ast.literal_eval(pose) if pose else None,
            reverse_depth_extra=boolean(row['reverse_depth_extra'])))
    return dict(side=routes[0], bilateral=len(routes) > 1, routes=routes,
        confidence_tier=manifest_rows.side_tier.iloc[0], series=series)


def extract_archive(path, destination):
    with tarfile.open(path) as tf:
        for m in tf.getmembers():
            if not m.isfile() or not (destination / m.name).resolve().is_relative_to(destination.resolve()):
                raise RuntimeError(f'Unsafe archive member: {m.name}')
        tf.extractall(destination)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runtime', type=Path, required=True)
    ap.add_argument('--series-csv', type=Path, required=True)
    ap.add_argument('--inventory', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--cohort', choices=['gold58', 'train', 'test'], required=True)
    ap.add_argument('--images-root', type=Path)
    ap.add_argument('--archives', type=Path)
    ap.add_argument('--training-manifest', type=Path)
    ap.add_argument('--operations', type=Path)
    ap.add_argument('--parent-shards', type=Path)
    ap.add_argument('--shard', type=int)
    ap.add_argument('--reference-manifest', type=Path)
    ap.add_argument('--reference-root', type=Path)
    ap.add_argument('--parity-parents', type=Path)
    ap.add_argument('--contact-parents', type=Path)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    bind_runtime(args.runtime)
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError('Output must be empty; never overwrite a bank')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'decisions').mkdir()
    started = time.monotonic()
    frame = pd.read_csv(args.series_csv, dtype=str).fillna('')
    inventory = pd.read_csv(args.inventory, dtype=str).fillna('') if args.inventory else None
    inventory = inventory[inventory.source_cohort.eq(args.cohort)] if inventory is not None else None
    training = pd.read_csv(args.training_manifest, dtype=str).fillna('') if args.training_manifest else None
    ops = pd.read_csv(args.operations, dtype=str).fillna('') if args.operations else None
    references = pd.read_csv(args.reference_manifest, dtype=str).fillna('') if args.reference_manifest else None
    parity_uids = set(pd.read_csv(args.parity_parents, dtype=str).ParentStudyInstanceUID) if args.parity_parents else set()
    contacts = pd.read_csv(args.contact_parents,dtype=str) if args.contact_parents else pd.DataFrame()
    contact_uids = set(contacts.uid) if len(contacts) else set()
    groups = {uid: g for uid, g in frame.groupby('StudyInstanceUID', sort=True)}
    uids = sorted(groups)
    if args.cohort == 'train':
        if training is None or ops is None:
            raise RuntimeError('Training requires the frozen soft-v2 overlay')
        uids = sorted(set(uids) & set(training.ParentStudyInstanceUID))
    if args.shard is not None:
        shards = pd.read_csv(args.parent_shards, dtype=str)
        keep = set(shards.loc[shards.shard.astype(int).eq(args.shard), 'ParentStudyInstanceUID'])
        uids = [uid for uid in uids if uid in keep]
    if args.limit:
        uids = uids[:args.limit]
    resolver = None
    if args.cohort != 'train':
        from resolver import LateralityResolver
        resolver = LateralityResolver(args.runtime, device='cpu')
    rows_out, series_out, parity_out, excluded = [], [], [], []
    for i, uid in enumerate(uids, 1):
        with tempfile.TemporaryDirectory(prefix='knee-ext-') as scratch:
            if args.archives:
                found = list(args.archives.glob(f'*__{uid}.tar.gz'))
                if len(found) != 1:
                    raise RuntimeError(f'Expected one archive: {uid}')
                extract_archive(found[0], Path(scratch))
                images_root = Path(scratch) / 'train_series'
            elif args.images_root:
                images_root = args.images_root
            else:
                raise RuntimeError('DICOM images or archives required')
            rows = groups[uid]
            if args.cohort == 'train':
                mr = training[training.ParentStudyInstanceUID.eq(uid)]
                decision = train_decision(mr, ops[ops.uid.eq(uid)])
                if decision['bilateral']:
                    excluded.extend(dict(SampleID=s, StudyInstanceUID=uid,
                        status='excluded_bilateral') for s in mr.SampleID)
                    continue
                sample_id = mr.SampleID.iloc[0]
            else:
                decision, sample_id = None, uid
            choices = select_series_for_cache(rows, images_root, SLOTS)
            selected = {c['series_uid'] for c in choices if c is not None}
            extra = sorted(set(rows.SeriesInstanceUID) - selected)
            if inventory is not None:
                inv = inventory[inventory.uid.eq(uid)]
                expected = set(inv.loc[~inv.sel.map(boolean), 'sid'])
                if set(extra) != expected:
                    raise RuntimeError(f'Series selection differs from inventory: {uid}')
            needs_parity = references is not None and (args.cohort == 'gold58' or uid in parity_uids)
            decode_rows = rows if resolver is not None or needs_parity or uid in contact_uids else rows[rows.SeriesInstanceUID.isin(extra)]
            study = DecodedStudy(uid, decode_rows.to_dict('records'), images_root) if len(decode_rows) else None
            if resolver is not None:
                decision = study.side_inputs(resolver)
                if decision.get('bilateral') or len(decision.get('routes', [])) != 1:
                    raise RuntimeError(f'Unexpected bilateral eval case: {uid}')
            atomic_json(args.output / 'decisions' / f'{uid}.json', decision)
            if needs_parity:
                ref = references[references.SampleID.eq(sample_id)]
                if len(ref) != 1:
                    raise RuntimeError(f'Unique parity reference missing: {sample_id}')
                pr = parity(study, decision, ref.iloc[0], args.reference_root)
                parity_out.append(pr)
                atomic_json(args.output / 'PARITY.json', parity_out)
                if not pr['pass']:
                    raise RuntimeError(f'Selected-series pixel/meta parity FAIL: {uid}')
            all_pixels, all_meta = [], []
            columns = {k: [] for k in ['series_uid', 'plane', 'fs', 'is_oblique',
                'dcls', 'orient', 'native_orientation', 'center_source_index', 'neighbor_status']}
            for sid in extra:
                if sid not in study.series:
                    raise RuntimeError(f'Missing decoded extra series: {sid}')
                errors = [e for e in study.errors if e.get('sid') == sid]
                if errors:
                    raise RuntimeError(f'Extra series decode errors: {sid}: {errors}')
                arrays, metadata, plane, normal, fs, stat_ids = series_arrays(study, sid)
                centers, slots, _ = select_centers(arrays, metadata, budget=12,
                    trim_each_side=0 if plane == 'Axial' else .1)
                pixels, meta6, _, orient, neighbor, kept_centers, dropped = render(arrays, metadata,
                    centers, slots, decision, extension_v3=True)
                # Frozen inventory reference: lexicographically greatest selected
                # SID in this plane (the original metadata JSON used sorted keys).
                reference_sids = [bs for bs in sorted(selected, reverse=True)
                    if rows[rows.SeriesInstanceUID.eq(bs)].iloc[0].Anatomical_Plane == plane][:1]
                base_normals = []
                for base_sid in reference_sids:
                    if base_sid in study.series:
                        bp, bn = plane_from_iop(study.series[base_sid])
                        if bp == plane and bn is not None:
                            base_normals.append(bn)
                # For train non-parity rows selected headers are read without pixel decode.
                if not base_normals and normal is not None:
                    from rsna_knee.native_neighbor_bank import read_ordered_headers
                    from rsna_knee.cache import resolve_series_dir
                    for bsid in reference_sids:
                        r = rows[rows.SeriesInstanceUID.eq(bsid)].iloc[0]
                        if r.Anatomical_Plane != plane:
                            continue
                        entries, _ = read_ordered_headers(resolve_series_dir(images_root, uid, bsid))
                        iop = np.asarray(getattr(entries[0]['header'], 'ImageOrientationPatient', []), float)
                        if iop.shape == (6,):
                            bn = np.cross(iop[:3], iop[3:])
                            if np.linalg.norm(bn) > .9:
                                base_normals.append(bn / np.linalg.norm(bn))
                angles = [float(np.degrees(np.arccos(np.clip(abs(normal @ bn), 0, 1))))
                    for bn in base_normals] if normal is not None else []
                angle = angles[0] if angles else 0.
                oblique = angle >= 15.
                inv_row = inventory[inventory.sid.eq(sid)].iloc[0] if inventory is not None else None
                dcls = inv_row.dcls if inv_row is not None else 'test_unselected'
                n = len(pixels)
                values = dict(series_uid=[sid]*n, plane=[plane]*n, fs=[fs]*n,
                    is_oblique=[oblique]*n, dcls=[dcls]*n,
                    orient=orient.tolist(),native_orientation=[False]*n, center_source_index=kept_centers.tolist(),
                    neighbor_status=neighbor.tolist())
                for k, v in values.items():
                    columns[k].extend(v)
                all_pixels.append(pixels)
                all_meta.append(meta6)
                series_out.append(dict(StudyInstanceUID=uid, SampleID=sample_id,
                    series_uid=sid, plane=plane, fs=fs, dcls=dcls, windows=n,
                    native_windows=0, v3_windows=int((orient=='v3').sum()),
                    excluded_orientation_windows=dropped, excluded_series=(n==0),
                    exclusion_reason='transform_spec_v3_abstention' if dropped else '',
                    is_oblique=oblique, angle_degrees=angle,
                    normalization_source_indices=json.dumps(stat_ids.tolist()),
                    normalization_low=metadata['series'][sid]['normalization_low'],
                    normalization_high=metadata['series'][sid]['normalization_high']))
            pixels = np.concatenate(all_pixels) if all_pixels else np.zeros((0,3,336,336), np.uint8)
            meta = np.concatenate(all_meta) if all_meta else np.zeros((0,6), np.float32)
            if uid in contact_uids:
                from PIL import Image, ImageDraw
                cr=contacts[contacts.uid.eq(uid)].iloc[0]
                positions=np.flatnonzero(np.asarray(columns['series_uid'])==cr.sid)
                if len(positions):
                    oblique=pixels[positions[len(positions)//2],1]
                    ba,bm=study.h1_bank(decision)
                    bc,bs,_=select_centers(ba,bm,budget=64,
                        trim_each_side=[.1,.1,.1,.1,0],slot_weights=[1]*5)
                    bp,_,*_=render(ba,bm,bc,bs,decision)
                    present=np.flatnonzero(bc>=0)
                    rec=bm['series'][cr.base_sid]
                    indices=np.flatnonzero(bs[present]==rec['legacy_slot'])
                    main=bp[indices[len(indices)//2],1]
                    panel=Image.new('RGB',(672,376),'white')
                    panel.paste(Image.fromarray(main).convert('RGB'),(0,40))
                    panel.paste(Image.fromarray(oblique).convert('RGB'),(336,40))
                    draw=ImageDraw.Draw(panel)
                    draw.text((5,4),f'{cr.side} | main coronal / oblique | {uid[-16:]}',fill='black')
                    draw.text((5,20),'Geometry target: medial LEFT, lateral RIGHT in both panels',fill='black')
                    cp=args.output/'contacts';cp.mkdir(exist_ok=True)
                    panel.save(cp/f'{cr.side}__{uid}.png')
                else:
                    raise RuntimeError(f'Prespecified contact case excluded by v3: {uid}')
            name = f'{sample_id}.ext.npz'
            packed = pack(args.output / name, pixels, meta, columns,
                dict(sample_id=sample_id, study_uid=uid, side=decision['side'],
                    orientation_policy='knee-laterality-v8.3', orientation_state='canonical_L_v8_3',
                    source_image_size=336, model_image_size=384, crop_mm=140))
            rows_out.append(dict(SampleID=sample_id, StudyInstanceUID=uid,
                cache_file=name, source_series=len(extra), series=len(set(columns['series_uid'])), windows=len(pixels),
                native_windows=int(sum(columns['native_orientation'])), **packed))
            pd.DataFrame(rows_out).to_csv(args.output / 'EXT_MANIFEST.csv', index=False)
            pd.DataFrame(series_out).to_csv(args.output / 'SERIES_MANIFEST.csv', index=False)
            live = dict(status='BUILDING', done=i, total=len(uids), windows=sum(r['windows'] for r in rows_out),
                bytes=sum(r['bytes'] for r in rows_out), elapsed_seconds=time.monotonic()-started)
            atomic_json(args.output / 'LIVE.json', live)
            print(json.dumps(live), flush=True)
            del study
            gc.collect()
    if inventory is not None:
        excluded_uids = {r['StudyInstanceUID'] for r in excluded}
        expected = set(inventory.loc[inventory.uid.isin(set(uids)-excluded_uids) & ~inventory.sel.map(boolean), 'sid'])
        actual = {r['series_uid'] for r in series_out}
        if actual != expected:
            raise RuntimeError('Final extra-series coverage FAIL')
    atomic_json(args.output / 'EXCLUDED_BILATERAL.json', excluded)
    atomic_json(args.output / 'PARITY.json', parity_out)
    parity_frames = sum(r.get('total_frames', 192) for r in parity_out)
    parity_nonexact = sum(r.get('nonexact_frames', 0) for r in parity_out)
    exact_fraction = 1 - parity_nonexact/parity_frames if parity_frames else None
    if exact_fraction is not None and exact_fraction < .999:
        raise RuntimeError(f'Aggregate 99.9% frame parity FAIL: {exact_fraction}')
    receipt = dict(status='PASS_LOCAL_EXT_BANK' if args.cohort == 'gold58' else 'PASS_EXT_SHARD',
        format=FORMAT, cohort=args.cohort, shard=args.shard, parents=len(uids), samples=len(rows_out),
        source_series=len(series_out),series=sum(r['windows']>0 for r in series_out), windows=sum(r['windows'] for r in rows_out),
        v3_windows=sum(r['v3_windows'] for r in series_out),
        excluded_orientation_windows=sum(r['excluded_orientation_windows'] for r in series_out),
        excluded_orientation_series=sum(r['excluded_series'] for r in series_out),
        native_windows=sum(r['native_windows'] for r in rows_out),
        parity_parents=len(parity_out), parity_pass=all(r['pass'] for r in parity_out),
        parity_exact=all(r['exact_hash_match'] for r in parity_out),
        parity_exact_frame_fraction=exact_fraction, parity_nonexact_frames=parity_nonexact,
        excluded_bilateral_samples=len(excluded), bytes=sum(r['bytes'] for r in rows_out),
        source_image_size=336, model_image_size=384, crop_mm=140,
        orientation_policy=json.loads((args.runtime.parent/'ORIENTATION_RECEIPT.json').read_text())['orientation_policy'],
        orientation_state='canonical_L_v8_3', extension_geometry_policy='transform_spec_v3',
        unknown_orientation_policy='exclude_and_audit',
        builder_sha256=sha(__file__), series_sha256=sha(args.series_csv),
        inventory_sha256=sha(args.inventory) if args.inventory else None,
        manifest_sha256=sha(args.output/'EXT_MANIFEST.csv'),
        training_started=False, full_train_coverage_verified=False,
        elapsed_seconds=time.monotonic()-started)
    atomic_json(args.output / 'CACHE_COMPLETE.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        print(json.dumps(dict(status='FAIL', type=type(exc).__name__, message=str(exc))), flush=True)
        raise
