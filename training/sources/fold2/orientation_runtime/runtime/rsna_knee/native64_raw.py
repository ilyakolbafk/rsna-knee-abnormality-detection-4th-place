"""Test DICOM to Native64 pixels plus optional side/acquisition metadata.

Pixel path copied unchanged from the Kaggle-verified singlefold v2 runtime.
"""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import pydicom
from PIL import Image  # Require the real bilinear implementation; no fallback.
from rsna_knee.grayscale_cache import select_series_for_cache, preprocess_grayscale_slot
from rsna_knee.native_neighbor_bank import read_ordered_headers, normalization_bounds, quantize_with_bounds
from rsna_knee.dicom import dicom_pixels, pixel_spacing, center_crop_mm, resize_image, discover_dicom_files
from rsna_knee.cache import resolve_series_dir
from rsna_knee.canonical_inputs import side_record


def build_arrays(uid, series_rows, images_root, data):
    started = time.monotonic()
    tag_values = []
    side_header_errors = []
    # Include unselected series: a decisive side tag can exist only there.
    for row in series_rows:
        try:
            directory = resolve_series_dir(images_root, uid, str(row['SeriesInstanceUID']))
            files = discover_dicom_files(directory)
            header = pydicom.dcmread(files[0], stop_before_pixels=True, force=True)
            if str(getattr(header,'StudyInstanceUID','')) != uid or str(getattr(header,'SeriesInstanceUID','')) != str(row['SeriesInstanceUID']):
                raise ValueError('Side header UID mismatch')
            tag_values.append(str(getattr(header, 'Laterality', '')).strip())
        except Exception as exc:
            side_header_errors.append(dict(series_uid=str(row['SeriesInstanceUID']),error=type(exc).__name__))
    plane_records = []
    selected = select_series_for_cache(pd.DataFrame(series_rows), images_root, data['slots'])
    images, decode, geometry, series_index, source_index = [], [], [], [], []
    records, errors, slot_audit = {}, [], []
    offset = 0
    for slot, choice in enumerate(selected):
        if choice is None:
            continue
        spec = data['slots'][slot]; sid = choice['series_uid']; directory = choice['directory']
        if sid in records:
            raise ValueError('One source series assigned to multiple semantic slots')
        old, old_valid, _, indices, names, old_stats = preprocess_grayscale_slot(
            directory, count=spec['slices'], image_size=336, crop_mm=data['crop_mm'],
            percentile_low=data['percentile_low'], percentile_high=data['percentile_high'],
            window=spec['window'], centering_mode='geometric', crop_profile='local_rectangular_mm')
        if not old_valid.any():
            slot_audit.append({'slot':slot,'series_uid':sid,'status':'no_valid_legacy_normalization_centers'})
            continue
        ordered, qc = read_ordered_headers(directory); n = len(ordered)
        if n != old_stats['source_slices'] or [ordered[i]['path'].name for i in indices] != names:
            raise ValueError('Legacy and native DICOM ordering differs')
        resized = np.zeros((n,336,336),np.float32); valid = np.zeros(n,bool)
        for i, entry in enumerate(ordered):
            try:
                ds = pydicom.dcmread(entry['path'],force=True)
                resized[i] = resize_image(center_crop_mm(dicom_pixels(ds),pixel_spacing(ds),data['crop_mm']),336)
                valid[i] = True
            except Exception as exc:
                errors.append({'slot':slot,'source_index':i,'error_type':type(exc).__name__})
        if not np.array_equal(valid[indices],old_valid):
            raise ValueError('Legacy center validity changed between decodes')
        low,high = normalization_bounds(resized[indices],old_valid,(data['percentile_low'],data['percentile_high']))
        pixels = quantize_with_bounds(resized,low,high); pixels[~valid] = 0
        delta = np.abs(pixels[indices].astype(np.int16)-old.astype(np.int16))
        if delta.max(initial=0)>1 or np.count_nonzero(delta)>old.size*1e-5:
            raise ValueError('Native versus legacy center numerical mismatch')
        # On test, canonical legacy bytes are freshly built by the unchanged pipeline.
        # Historical train-bank byte correction is separately checked on raw/FSZ pairs.
        for index in np.unique(indices):
            expected = old[indices==index]
            if not np.all(expected==expected[0]):
                raise ValueError('Repeated legacy center bytes disagree')
            pixels[index] = expected[0]
        si = len(records)
        source_row = next(r for r in series_rows if str(r['SeriesInstanceUID']) == sid)
        fs = pd.to_numeric(source_row.get('Fat_Suppression'), errors='coerce')
        fs = int(fs) if fs in (0, 1) else None
        for i, entry in enumerate(ordered):
            plane_records.append(dict(bank_index=offset+i, SeriesInstanceUID=sid,
                geometry_tags={key: (list(getattr(entry['header'], key)) if key in ('ImageOrientationPatient','ImagePositionPatient','PixelSpacing') else int(getattr(entry['header'], key)))
                    for key in ('ImageOrientationPatient','ImagePositionPatient','PixelSpacing','Rows','Columns') if getattr(entry['header'],key,None) is not None},
                geometry_valid=bool(entry['geometry']['valid'] and qc['within_series_valid'])))
        records[sid] = {'offset':offset,'source_count':n,'series_index':si,'legacy_slot':slot,
            'geometry_qc':qc,'source_inventory':{'fat_suppression':fs},'normalization_low':float(low),'normalization_high':float(high)}
        images.append(pixels);decode.extend(valid.tolist())
        geometry.extend(bool(e['geometry']['valid'] and qc['within_series_valid']) for e in ordered)
        series_index.extend([si]*n);source_index.extend(range(n));offset += n
        slot_audit.append({'slot':slot,'series_uid':sid,'source_count':n,'decoded':int(valid.sum()),
            'legacy_centers':int(old_valid.sum()),'geometry_valid':qc['within_series_valid'],
            'center_recompute_differences':int(np.count_nonzero(delta)), 'status':'built'})
    if not images:
        raise ValueError('No usable semantic slot; refuse fabricated prediction')
    arrays = {'images':np.concatenate(images),'decode_valid':np.array(decode,bool),
        'geometry_valid':np.array(geometry,bool),'series_index':np.array(series_index,np.int32),
        'source_index':np.array(source_index,np.int32)}
    metadata = {'legacy_slots':data['slots'],'series':records,'uid':uid,'planes':plane_records}
    metadata['side_record'] = side_record(metadata, tag_values)
    if side_header_errors:
        metadata['side_record'].update(side='',source='side_header_read_error')
    audit = {'uid':uid,'slots':slot_audit,'decode_errors':errors,'seconds':time.monotonic()-started,
        'side_header_errors':side_header_errors,
        'source_planes':offset,'normalization':'legacy-center occurrences, including duplicate indices'}
    return arrays,metadata,audit


def build_to_file(request):
    uid,rows,images_root,data,output = request
    arrays,metadata,audit = build_arrays(uid,rows,images_root,data)
    path = Path(output)/(uid+'.npz')
    with path.open('xb') as f:
        np.savez(f,**arrays,metadata=np.array(json.dumps(metadata)),audit=np.array(json.dumps(audit)))
    return str(path)
