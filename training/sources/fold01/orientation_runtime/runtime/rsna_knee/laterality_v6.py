"""Three evidence fields and physical source-frame selection.

Absolute X is an empirical scanner convention, not a DICOM midline guarantee.
Geometry agreement is reported as such. Original DICOM is never rewritten.
"""
from collections import defaultdict
import numpy as np
from .canonical_inputs import tag_consensus, transform_spec_v3


def frame_center(record):
    io=np.asarray(record.get('iop'),float)
    ip=np.asarray(record.get('ipp'),float)
    sp=np.asarray(record.get('pixel_spacing'),float)
    rows,cols=record.get('rows',0),record.get('cols',0)
    if (io.shape!=(6,) or ip.shape!=(3,) or sp.shape!=(2,)
        or not np.isfinite(np.r_[io,ip,sp]).all() or (sp<=0).any()
        or rows<1 or cols<1 or abs(io[:3]@io[3:])>.02
        or not np.allclose(np.linalg.norm(io.reshape(2,3),axis=1),1,atol=.02)):
        raise ValueError('invalid_patient_geometry')
    return ip+io[:3]*sp[1]*(cols-1)/2+io[3:]*sp[0]*(rows-1)/2


def geometry_vote(records, *, min_abs_x_mm=20.):
    records=list(records)
    if not records or len({r.get('study_uid') for r in records})!=1:
        raise ValueError('exactly_one_study_required')
    groups=defaultdict(list)
    issues=[]
    for r in records:
        try:
            if r.get('anatomical_orientation','BIPED') not in ('','BIPED'):
                raise ValueError('unsupported_anatomical_orientation')
            groups[r['series_uid']].append(float(frame_center(r)[0]))
        except (ValueError,TypeError,KeyError) as e:
            issues.append(str(e))
    xs={sid:float(np.median(v)) for sid,v in groups.items()}
    median=float(np.median(list(xs.values()))) if xs else None
    agrees=bool(xs) and (all(x>0 for x in xs.values()) or all(x<0 for x in xs.values()))
    valid=not issues and agrees and abs(median)>=min_abs_x_mm
    return dict(side=('L' if median>0 else 'R') if valid else '',
                valid=bool(valid),median_center_x_mm=median,series_center_x_mm=xs,
                threshold_mm=min_abs_x_mm,issues=sorted(set(issues)),
                evidence='empirical_absolute_X_convention_not_anatomical_truth')


def resolve_three_votes(tag_values=(), geometry=None, image=None, *, enable_image_estimate=False):
    tag,status=tag_consensus(tag_values)
    geometry=geometry or {};image=image or {}
    geo=geometry.get('side','') if geometry.get('valid',False) else ''
    visual=image.get('side','') if image.get('valid',False) else ''
    if geo not in ('','L','R') or visual not in ('','L','R'):
        raise ValueError('invalid_side_vote')
    base=dict(tag_side=tag,tag_status=status,geometry_side=geo,image_side=visual,
              tag_geometry_conflict=bool(tag and geo and tag!=geo),
              image_conflicts_with_agreement=bool(tag==geo and tag and visual and visual!=tag),
              canonical_target='L',canonicalization_verified=False,
              scores_are_calibrated_probabilities=False)
    if status=='bilateral':
        return dict(base,side='',source='bilateral_not_single_knee',orientations=[],needs_image=False)
    if tag and tag==geo:
        return dict(base,side=tag,source='tag_geometry_agreement',orientations=[tag],needs_image=False)
    if visual and enable_image_estimate:
        return dict(base,side=visual,source='image_decides_missing_or_conflicting_metadata',orientations=[visual],needs_image=False)
    return dict(base,side='',source='insufficient_or_conflicting_evidence',orientations=['L','R'],needs_image=True)


def select_anatomical_sources(records, joint_references=()):
    """Axial band relative to a registered joint height; posterior coronal views.

    Each joint reference must carry its source FrameOfReferenceUID. Identical
    numeric Z or the same study UID cannot substitute for registration.
    """
    records=list(records)
    if not records or len({r['study_uid'] for r in records})!=1:
        raise ValueError('exactly_one_study_required')
    if len({r['sop_instance_uid'] for r in records})!=len(records):
        raise ValueError('duplicate_source_sop')
    groups=defaultdict(list)
    for r in records:groups[r['series_uid']].append(r)
    selected=[];skipped=[]
    for sid,rows in sorted(groups.items()):
        try:
            centers=np.asarray([frame_center(r) for r in rows])
            io=np.asarray(rows[0]['iop'])
            plane=['Sagittal','Coronal','Axial'][int(np.argmax(np.abs(np.cross(io[:3],io[3:]))))]
            if plane=='Sagittal':continue
            if len(rows)<3 or any(not np.allclose(r['iop'],io,atol=1e-3) for r in rows):
                raise ValueError('inconsistent_or_short_series')
            if transform_spec_v3(io,plane,'L') is None:
                raise ValueError('unsupported_orientation')
            fors={r.get('frame_of_reference_uid','') for r in rows}
            if len(fors)!=1 or not next(iter(fors)):
                raise ValueError('missing_or_mixed_frame_of_reference')
            order=np.argsort(centers[:,2] if plane=='Axial' else centers[:,1])
            if plane=='Axial':
                refs=[j for j in joint_references if j.get('frame_of_reference_uid') in fors and j.get('valid',False)
                      and j.get('series_uid') in groups
                      and all(r.get('frame_of_reference_uid')==j['frame_of_reference_uid'] for r in groups[j['series_uid']])]
                if not refs:raise ValueError('no_registered_joint_height')
                zs=np.asarray([j['joint_z'] for j in refs],float)
                if not np.isfinite(zs).all() or np.ptp(zs)>10:
                    raise ValueError('conflicting_joint_heights')
                jz=float(np.median(zs));low,high=jz-35,jz-10
                candidates=[int(i) for i in order if low<=centers[i,2]<=high]
                if len(candidates)<3:raise ValueError('no_three_anatomical_axial_frames')
                chosen=[candidates[i] for i in np.rint(np.linspace(0,len(candidates)-1,3)).astype(int)]
                detail=dict(joint_z=jz,z_band_mm=[low,high],joint_source_series=[j['series_uid'] for j in refs])
            else:
                if np.ptp(centers[:,1])<5:raise ValueError('insufficient_coronal_depth')
                chosen=[int(order[i]) for i in np.rint(np.array([.65,.8,.95])*(len(rows)-1)).astype(int)]
                if len(set(chosen))!=3:raise ValueError('no_three_distinct_posterior_frames')
                detail=dict(posterior_fractions=[.65,.8,.95])
            selected.append(dict(series_uid=sid,plane=plane,source_indices=chosen,
                                 source_count=len(rows),frames=[rows[i] for i in chosen],**detail))
        except (ValueError,TypeError,KeyError) as e:
            skipped.append(dict(series_uid=sid,reason=str(e)))
    return selected,skipped


def predict_study_three_votes(records, predictor_factory=None, *, joint_references=(),
                               geometry_records=None, audit_images=False,
                               enable_image_estimate=False, high_score=.9):
    """Lazy image inference: agreed tag/geometry costs no backbone forward.

    The factory returns a dict keyed by Axial/Coronal. A predictor bundle must
    match its plane, sampling, and input size; the old axial bundle stays 224.
    """
    import hashlib
    from pathlib import Path
    from .compartment_locator_v2 import read_dicom
    from .study_laterality import select_axial_sources
    from .laterality_inference import reconcile_series_sides
    records=list(records)
    if not records or len({r['study_uid'] for r in records})!=1:
        raise ValueError('exactly_one_study_required')
    if len({r['sop_instance_uid'] for r in records})!=len(records):
        raise ValueError('duplicate_source_sop')
    tags=[t for r in records for t in r.get('tags',[])]
    if geometry_records is not None:
        geometry_records=list(geometry_records)
        allowed={r['sop_instance_uid'] for r in records}
        if (not geometry_records or any(r['study_uid']!=records[0]['study_uid'] or r['sop_instance_uid'] not in allowed for r in geometry_records)):
            raise ValueError('geometry_records_must_belong_to_input_study')
    geometry=geometry_vote(records if geometry_records is None else geometry_records)
    decision=resolve_three_votes(tags,geometry)
    results=[];skipped=[]
    if (decision['needs_image'] or audit_images) and predictor_factory is not None:
        models=predictor_factory()
        legacy,skipped=select_axial_sources(records)
        anatomy,extra=select_anatomical_sources(records,joint_references)
        skipped+=extra
        for plane,model in models.items():
            if model.bundle.get('plane','Axial')!=plane:
                raise ValueError('predictor_plane_mismatch')
            selection=model.bundle.get('selection','legacy_lower_three')
            series=(legacy if selection=='legacy_lower_three' and plane=='Axial' else
                    [s for s in anatomy if s['plane']==plane])
            for s in series:
                frames=[];sources=[];decode_error=None
                for r in s['frames']:
                    source_sha=hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()
                    try:
                        d=read_dicom(r['path'])
                    except ValueError as exc:
                        if not str(exc).startswith('The number of bytes of pixel data is less than expected'):
                            raise
                        decode_error='source_pixel_decode_error: '+str(exc)
                        sources.append(dict(path=r['path'],sop_instance_uid=r['sop_instance_uid'],sha256=source_sha,error=decode_error))
                        break
                    if (d['uid'],d['series'],d['sop'])!=(r['study_uid'],r['series_uid'],r['sop_instance_uid']):
                        raise ValueError('source_identity_changed')
                    frames.append(dict(r,pixels=d['pixels'],iop=d['iop']))
                    sources.append(dict(path=r['path'],sop_instance_uid=d['sop'],sha256=source_sha))
                if decode_error:
                    results.append(dict(series_uid=s['series_uid'],plane=plane,error=decode_error,sources=sources))
                    continue
                try:
                    padding=model.bundle.get('padding','legacy')
                    view=model.predict(frames,padding=padding);views=[view]
                    if padding=='legacy' and any(q['crop_extends_fov'] for q in view['input_qc']):
                        views.append(model.predict(frames,padding='constant'))
                    results.append(dict(series_uid=s['series_uid'],plane=plane,views=views,sources=sources))
                except ValueError as e:
                    results.append(dict(series_uid=s['series_uid'],plane=plane,error=str(e),sources=sources))
        image=reconcile_series_sides(results,enable_image_estimate=True,high_score=high_score)
        decision=resolve_three_votes(tags,geometry,dict(side=image['side'],valid=bool(image['side'])),enable_image_estimate=enable_image_estimate)
    plan=[]
    if decision['side']:
        for sid in sorted({r['series_uid'] for r in records}):
            specs=[]
            for r in [x for x in records if x['series_uid']==sid]:
                io=np.asarray(r['iop'],float)
                if io.shape!=(6,):specs.append(None);continue
                plane=['Sagittal','Coronal','Axial'][int(np.argmax(np.abs(np.cross(io[:3],io[3:]))))]
                specs.append(transform_spec_v3(io,plane,decision['side']))
            ok=bool(specs) and all(s is not None and s==specs[0] for s in specs)
            plan.append(dict(series_uid=sid,usable=ok,transform=specs[0] if ok else None,
                depth_input_order='ascending dot(IPP,cross(IOP_col,IOP_row))'))
    return dict(decision,study_uid=records[0]['study_uid'],method_version='laterality_v6',
                geometry_scope='all_input_series' if geometry_records is None else 'explicit_input_subset',
                geometry=geometry,image_was_evaluated=bool(results),series_results=results,
                skipped_series=skipped,canonical_input_plan=plan,
                all_series_transformable=bool(plan) and all(r['usable'] for r in plan),
                note='Model-input transform only; original DICOM preserved; empirical side estimate.')
