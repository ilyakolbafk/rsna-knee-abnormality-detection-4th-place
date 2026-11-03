"""Opt-in side classifier inputs and conservative inference policy.

All images are first oriented to patient +X right, +Y down. Neither absolute
patient coordinates, pathology scores, identifiers nor scanner/site strings
are classifier features. A model must be independently approved before its
votes may override DICOM tags; a candidate fit is not such an approval.
"""
from __future__ import annotations
import numpy as np
from .canonical_inputs import transform_spec_v3, tag_consensus


def axial_body_image(pixels, iop, *, size=224, padding='legacy', return_qc=False, plane='Axial'):
    """Return a body-centred axial view in patient axes, before side reflection."""
    import cv2
    if plane not in ('Axial','Coronal'):
        raise ValueError('unsupported_side_image_plane')
    spec=transform_spec_v3(iop,plane,'L')
    if spec is None:
        raise ValueError('unsupported_axial_orientation')
    a=np.asarray(pixels,dtype=np.float32)
    if a.ndim!=2 or min(a.shape)<32 or not np.isfinite(a).all():
        raise ValueError('invalid_pixels')
    if spec['transpose']:a=a.T
    if spec['flip_x']:a=a[:,::-1]
    if spec['flip_y']:a=a[::-1]
    lo,hi=np.percentile(a,[1,99.5])
    if hi<=lo:raise ValueError('empty_dynamic_range')
    a=np.uint8(np.clip((a-lo)/(hi-lo)*255,0,255))
    blur=cv2.GaussianBlur(a,(7,7),1.5)
    binary=np.uint8(blur>max(12,float(np.percentile(a,75))*.2))
    binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    n,lab,stats,centroids=cv2.connectedComponentsWithStats(binary)
    candidates=[i for i in range(1,n) if stats[i,cv2.CC_STAT_AREA]>.08*a.size
        and .15*a.shape[1]<centroids[i,0]<.85*a.shape[1]
        and .15*a.shape[0]<centroids[i,1]<.85*a.shape[0]]
    if not candidates:raise ValueError('no_central_body_component')
    k=max(candidates,key=lambda i:stats[i,cv2.CC_STAT_AREA])
    x,y,w,h,_=stats[k];side=int(np.ceil(max(w,h)*1.1))
    cx,cy=x+(w-1)/2,y+(h-1)/2
    # Legacy getRectSubPix repeats the edge beyond the FOV. Keep it for exact
    # reproduction; the constant-padding variant never fabricates edge tissue.
    bounds=[float(cx-(side-1)/2),float(cy-(side-1)/2),
            float(cx+(side-1)/2),float(cy+(side-1)/2)]
    qc=dict(source_shape=list(a.shape),crop_bounds_xyxy=bounds,output_size=size,plane=plane,
            crop_extends_fov=bool(bounds[0]<0 or bounds[1]<0
                or bounds[2]>a.shape[1]-1 or bounds[3]>a.shape[0]-1),
            body_touches_fov=bool(x==0 or y==0 or x+w==a.shape[1] or y+h==a.shape[0]),
            padding=padding)
    if padding=='constant':
        a=cv2.copyMakeBorder(a,side,side,side,side,cv2.BORDER_CONSTANT,value=0)
        cx+=side;cy+=side
    elif padding!='legacy':
        raise ValueError('unsupported_padding_policy')
    a=cv2.getRectSubPix(a,(side,side),(float(cx),float(cy)))
    image=cv2.resize(a,(size,size),interpolation=cv2.INTER_AREA)
    return (image,qc) if return_qc else image


def antisymmetric_hog(images):
    """Frame features satisfy feature(flip(image)) == -feature(image)."""
    import cv2
    def hog(a):
        gy,gx=np.gradient(a.astype(np.float32))
        magnitude=np.hypot(gx,gy)
        direction=(np.arctan2(gy,gx)%np.pi)*(9/np.pi)
        low=np.floor(direction).astype(int)%9;weight=direction-np.floor(direction)
        cy,cx=np.indices(a.shape)//8
        hist=np.zeros((12,12,9),np.float32)
        np.add.at(hist,(cy.ravel(),cx.ravel(),low.ravel()),(magnitude*(1-weight)).ravel())
        np.add.at(hist,(cy.ravel(),cx.ravel(),((low+1)%9).ravel()),(magnitude*weight).ravel())
        blocks=[]
        for y in range(11):
            for x in range(11):
                v=hist[y:y+2,x:x+2].ravel();v=v/np.sqrt(np.dot(v,v)+1e-8)
                v=np.minimum(v,.2);blocks.append(v/np.sqrt(np.dot(v,v)+1e-8))
        return np.concatenate(blocks)
    rows=[]
    for a in images:
        a=cv2.resize(np.asarray(a,dtype=np.uint8),(96,96),interpolation=cv2.INTER_AREA)
        mirrored=np.ascontiguousarray(a[:,::-1])
        rows.append(hog(a)-hog(mirrored))
    return np.asarray(rows,dtype=np.float32)


def predict_linear_frames(features, bundle):
    """Evaluate the antisymmetric linear score; no claim of calibration."""
    scale=np.asarray(bundle['rms'],dtype=float)
    coef=np.asarray(bundle['coef'],dtype=float)
    x=np.asarray(features,dtype=float)
    if (x.ndim<1 or coef.ndim!=1 or x.shape[-1]!=len(coef)
        or scale.shape!=coef.shape or not np.isfinite(x).all()
        or not np.isfinite(coef).all() or not np.isfinite(scale).all()
        or not (scale>0).all()):
        raise ValueError('invalid_feature_contract')
    logits=(x/scale)@coef
    return 1/(1+np.exp(-np.clip(logits,-50,50)))


def resolve_inference_side(tag_values=(), visual_frames=(), *, model_approved=False,
                           high_score=.9, min_frames=2):
    """Resolve at study level, with explicit two-orientation fallback.

    `visual_frames` are dicts with unique SOP identity and an image-only score
    for L, not pathology probabilities. Weak absolute-X guesses are not accepted.
    The caller must validate the model bundle before model_approved can be True.
    """
    tag,status=tag_consensus(tag_values)
    seen={}; invalid=False
    for r in visual_frames:
        sop=r.get('sop_instance_uid');score=r.get('left_score')
        if not sop or score is None or not np.isfinite(score) or not 0<=score<=1:
            invalid=True;continue
        if sop in seen and seen[sop]!=float(score):invalid=True
        seen[sop]=float(score)
    values=np.array(list(seen.values()),dtype=float)
    confident_left=int((values>=high_score).sum())
    confident_right=int((values<=1-high_score).sum())
    visual=''
    if not invalid and len(values)>=min_frames:
        if confident_left>=min_frames and not (values<.5).any():visual='L'
        if confident_right>=min_frames and not (values>.5).any():visual='R'
    conflict=bool(visual and tag and visual!=tag)
    base=dict(tag_side=tag,tag_status=status,visual_candidate=visual,
              visual_tag_conflict=conflict,distinct_frames=len(values),
              scores_are_calibrated_probabilities=False,canonical_target='L')
    if model_approved and visual:
        return dict(base,side=visual,source='approved_image_side_model',orientations=[visual],
                    status='image_model_estimate',canonicalization_verified=False)
    if tag and not conflict and not invalid:
        return dict(base,side=tag,source='dicom_tag_consensus',orientations=[tag],
                    status='tag_resolved_anatomy_not_verified',canonicalization_verified=False)
    return dict(base,side='',source='unresolved',orientations=['L','R'],
                status='two_orientation_inference_required',canonicalization_verified=False)


class AxialSidePredictor:
    """Reusable study scorer for already selected lower axial source frames.

    Construct once and reuse across studies. Candidate bundles remain diagnostic:
    their scores do not override tags or certify an anatomical orientation.
    The same three inferior frames sampled from ten evenly spaced series frames
    must be supplied, as in the frozen experiment. No study registry is queried.
    """

    def __init__(self, bundle, *, weights=None, device='cpu'):
        self.bundle=bundle
        self.kind=bundle['feature_type']
        self.device=device
        self.model=None
        if self.kind=='imagenet':
            import hashlib
            from pathlib import Path
            import timm
            if weights is None or not Path(weights).is_file():
                raise ValueError('local_imagenet_weights_required')
            expected=bundle.get('feature_weights_sha256')
            if not expected or hashlib.sha256(Path(weights).read_bytes()).hexdigest()!=expected:
                raise ValueError('feature_weights_hash_mismatch')
            self.model=timm.create_model(
                'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k',pretrained=True,
                num_classes=0,img_size=384,
                pretrained_cfg_overlay={'file':str(weights)}).eval().to(device)
        elif self.kind!='hog':
            raise ValueError('unsupported_feature_type')

    def predict(self, frames, *, tag_values=(), padding='legacy'):
        """Frames: source SOP, study/series UID, pixels, IOP; exactly 3 unique."""
        rows=list(frames)
        if len(rows)!=3 or len({r['sop_instance_uid'] for r in rows})!=3:
            raise ValueError('three_distinct_source_frames_required')
        for key in ('study_uid','series_uid'):
            values={r.get(key) for r in rows}
            if len(values)!=1 or not next(iter(values)):
                raise ValueError('mixed_or_missing_'+key)
        size=int(self.bundle.get('input_size',224))
        if size not in (224,384):raise ValueError('unsupported_input_size')
        plane=self.bundle.get('plane','Axial')
        views=[axial_body_image(r['pixels'],r['iop'],size=size,plane=plane,padding=padding,return_qc=True) for r in rows]
        images=np.stack([v[0] for v in views])
        if self.kind=='hog':
            features=antisymmetric_hog(images)
        else:
            import torch
            from torch.nn import functional as F
            with torch.inference_mode():
                x=torch.from_numpy(images.copy()).float().div_(255)
                x=x[:,None].expand(-1,3,-1,-1).to(self.device)
                if x.shape[-2:]!=(384,384):
                    x=F.interpolate(x,size=(384,384),mode='bilinear',align_corners=False)
                f=self.model.forward_features((torch.cat([x,x.flip(-1)])-.5)/.5)
                f=F.adaptive_avg_pool2d(f,(2,2)).flatten(1).cpu().numpy()
                features=f[:3]-f[3:]
        scores=predict_linear_frames(features,self.bundle)
        visual=[dict(sop_instance_uid=r['sop_instance_uid'],left_score=float(p))
                for r,p in zip(rows,scores)]
        decision=resolve_inference_side(tag_values,visual,
            model_approved=self.bundle.get('approved_for_override') is True)
        return dict(decision,frame_scores=visual,feature_type=self.kind,
                    automatic_override_enabled=self.bundle.get('approved_for_override') is True,
                    input_qc=[v[1] for v in views])


def reconcile_series_sides(series_results, tag_values=(), *, enable_image_estimate=False,
                           prefer_unclipped_series=False, high_score=.9):
    """A study must not silently inherit the first series' side.

    Confidence thresholds are fixed scores, not calibrated probabilities.
    Optional paired preprocessing views must agree. An unresolved/conflicting
    series blocks the visual estimate; an enabled estimate can override tags.
    """
    rows=list(series_results)
    # Input-quality selection happens before reading L/R scores. A body mask
    # touching the FOV is only a proxy for incomplete coverage. Retain the
    # rejected series in the result, and preserve the strict policy as default.
    excluded=[]
    if prefer_unclipped_series:
        def interior_count(r):
            views=r.get('views',[])
            qc=views[0].get('input_qc',[]) if views else []
            if len(qc)!=3 or any('body_touches_fov' not in q for q in qc):return None
            return sum(not q['body_touches_fov'] for q in qc)
        counts=[interior_count(r) for r in rows]
        if any(c is not None and c>0 for c in counts):
            excluded=[dict(series_uid=r.get('series_uid'),reason='all_three_body_masks_touch_fov_better_covered_series_available')
                      for r,c in zip(rows,counts) if c==0]
            rows=[r for r,c in zip(rows,counts) if c!=0]
    votes=[];issues=[]
    for r in rows:
        sid=r.get('series_uid')
        if not sid:issues.append('missing_series_uid');continue
        if r.get('error'):issues.append(str(r['error']));continue
        views=r.get('views',[])
        if not views:issues.append('missing_views');continue
        for view in views:
            v=resolve_inference_side([],view.get('frame_scores',[]),model_approved=True,high_score=high_score)
            if not v['side']:issues.append('weak_or_conflicting_frames')
            else:votes.append(v['side'])
    if len({r.get('series_uid') for r in rows})!=len(rows):issues.append('duplicate_series')
    sides=set(votes)
    if len(sides)>1:issues.append('series_or_preprocessing_conflict')
    tag,tag_status=tag_consensus(tag_values)
    if tag_status=='bilateral':issues.append('bilateral_tags')
    estimate=next(iter(sides)) if len(sides)==1 and not issues else ''
    conflict=bool(tag and estimate and tag!=estimate)
    result=dict(visual_candidate=estimate,tag_side=tag,tag_status=tag_status,
        visual_tag_conflict=conflict,series_count=len(rows),issues=sorted(set(issues)),
        excluded_series=excluded,coverage_preference_enabled=prefer_unclipped_series,
        canonical_target='L',canonicalization_verified=False,
        scores_are_calibrated_probabilities=False)
    if estimate and enable_image_estimate:
        return dict(result,side=estimate,source='image_side_estimate',orientations=[estimate])
    if tag and not conflict and not issues:
        return dict(result,side=tag,source='dicom_tag_consensus',orientations=[tag])
    return dict(result,side='',source='unresolved',orientations=['L','R'])
