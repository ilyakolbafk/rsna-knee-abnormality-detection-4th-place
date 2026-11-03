"""Source DICOM -> optional image-side estimate -> canonical-left array plan.

No disease labels or per-study registry are accepted. Input studies must be
separate. The output is a model-input plan, not a rewritten clinical DICOM.
"""
from pathlib import Path
import hashlib
import numpy as np
import pydicom
from .canonical_inputs import transform_spec_v3
from .compartment_locator_v2 import read_dicom
from .laterality_inference import reconcile_series_sides


def dicom_record(ds,path):
    if not getattr(ds,'StudyInstanceUID',None) or not getattr(ds,'SOPInstanceUID',None):
        raise ValueError('missing_source_identity')
    return dict(path=str(path),study_uid=str(ds.StudyInstanceUID),
            series_uid=str(getattr(ds,'SeriesInstanceUID','')),
            sop_instance_uid=str(ds.SOPInstanceUID),
            frame_of_reference_uid=str(getattr(ds,'FrameOfReferenceUID','')),
            iop=list(map(float,getattr(ds,'ImageOrientationPatient',[]))),
            ipp=list(map(float,getattr(ds,'ImagePositionPatient',[]))),
            rows=int(getattr(ds,'Rows',0)),cols=int(getattr(ds,'Columns',0)),
            pixel_spacing=list(map(float,getattr(ds,'PixelSpacing',[]))),
            tags=[str(getattr(ds,'Laterality','')),str(getattr(ds,'ImageLaterality',''))],
            anatomical_orientation=str(getattr(ds,'AnatomicalOrientationType','BIPED')))


def inventory_dicom(paths):
    return [dicom_record(pydicom.dcmread(path,stop_before_pixels=True),path)
            for path in sorted(map(Path,paths))]


def select_axial_sources(records):
    records=list(records)
    if not records or len({r['study_uid'] for r in records})!=1:
        raise ValueError('exactly_one_study_required')
    if len({r['sop_instance_uid'] for r in records})!=len(records):
        raise ValueError('duplicate_source_sop')
    groups={}
    for r in records:
        groups.setdefault(r['series_uid'],[]).append(r)
    selected=[];skipped=[]
    for sid,rows in sorted(groups.items()):
        first=rows[0];iop=np.asarray(first['iop'],float)
        if iop.shape!=(6,):
            skipped.append(dict(series_uid=sid,reason='missing_iop'));continue
        normal=np.cross(iop[:3],iop[3:])
        if int(np.argmax(np.abs(normal)))!=2:
            continue
        error=None
        if not sid or len(rows)<10:error='insufficient_axial_series'
        if len({r['frame_of_reference_uid'] for r in rows})!=1:error='mixed_frame_of_reference'
        if any(r['anatomical_orientation'] not in ('','BIPED') for r in rows):error='unsupported_anatomical_orientation'
        for r in rows:
            io=np.asarray(r['iop'],float);ip=np.asarray(r['ipp'],float)
            if (io.shape!=(6,) or ip.shape!=(3,) or not np.isfinite(ip).all()
                or not np.allclose(io,iop,atol=1e-3)
                or transform_spec_v3(io,'Axial','L') is None):
                error='inconsistent_or_unsupported_geometry'
        if error:
            skipped.append(dict(series_uid=sid,reason=error));continue
        inferior=normal*np.sign(normal[2])
        rows=sorted(rows,key=lambda r:float(np.dot(r['ipp'],inferior)))
        projection=np.asarray([np.dot(r['ipp'],inferior) for r in rows])
        if np.any(np.diff(projection)<=1e-3):
            skipped.append(dict(series_uid=sid,reason='duplicate_slice_positions'));continue
        indices=np.rint(np.array([0,1/9,2/9])*(len(rows)-1)).astype(int)
        selected.append(dict(series_uid=sid,source_count=len(rows),
            source_indices=indices.tolist(),frames=[rows[i] for i in indices]))
    return selected,skipped


def predict_study(records,predictor,*,enable_image_estimate=False,prefer_unclipped_series=False):
    records=list(records)
    selected,skipped=select_axial_sources(records)
    results=[]
    for series in selected:
        rows=[];evidence=[]
        for r in series['frames']:
            p=read_dicom(r['path'])
            if (p['uid'],p['series'],p['sop'])!=(r['study_uid'],r['series_uid'],r['sop_instance_uid']):
                raise ValueError('source_identity_changed')
            rows.append(dict(r,pixels=p['pixels'],iop=p['iop']))
            evidence.append(dict(path=r['path'],sop_instance_uid=r['sop_instance_uid'],
                sha256=hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()))
        try:
            legacy=predictor.predict(rows,padding='legacy')
            # Recompute only if the crop actually crosses the FOV boundary.
            views=[legacy]
            if any(q['crop_extends_fov'] for q in legacy['input_qc']):
                views.append(predictor.predict(rows,padding='constant'))
            results.append(dict(series_uid=series['series_uid'],source_indices=series['source_indices'],
                sources=evidence,views=views))
        except ValueError as exc:
            results.append(dict(series_uid=series['series_uid'],sources=evidence,error=str(exc)))
    tags=[tag for r in records for tag in r['tags']]
    decision=reconcile_series_sides(results,tags,enable_image_estimate=enable_image_estimate,
        prefer_unclipped_series=prefer_unclipped_series)
    plan=[]
    if decision['side']:
        for sid in sorted({r['series_uid'] for r in records}):
            rows=[r for r in records if r['series_uid']==sid]
            specs=[]
            for r in rows:
                io=np.asarray(r['iop'],float)
                if io.shape!=(6,):specs.append(None);continue
                plane=['Sagittal','Coronal','Axial'][int(np.argmax(np.abs(np.cross(io[:3],io[3:]))))]
                specs.append(transform_spec_v3(io,plane,decision['side']))
            consistent=all(s is not None and s==specs[0] for s in specs)
            plan.append(dict(series_uid=sid,transform=specs[0] if consistent else None,
                usable=consistent,reason='' if consistent else 'series_transform_not_consistent',
                depth_input_order='ascending dot(IPP, cross(IOP_col, IOP_row))'))
    return dict(decision,study_uid=records[0]['study_uid'],series_results=results,
        skipped_series=skipped,canonical_input_plan=plan,
        all_series_transformable=bool(plan) and all(r['usable'] for r in plan),
        note='Research estimate; original DICOM files and current training inputs unchanged.')


def apply_series_transform(volume,spec):
    """Apply to source stack in the explicitly recorded ascending normal order."""
    if spec is None:raise ValueError('verified_transform_spec_required')
    x=np.asarray(volume)
    if x.ndim!=3:raise ValueError('expected_depth_rows_columns')
    if spec['transpose']:x=x.transpose(0,2,1)
    if spec['flip_x']:x=x[:,:,::-1]
    if spec['flip_y']:x=x[:,::-1,:]
    if spec['reverse_channels']:x=x[::-1]
    return np.ascontiguousarray(x)
