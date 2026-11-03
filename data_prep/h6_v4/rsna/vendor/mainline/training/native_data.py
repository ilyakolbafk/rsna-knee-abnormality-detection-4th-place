"""Runtime whole-field/local views with matching full-depth slice coverage."""
import json
import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .common import UID,LABELS
from .supervision import targets_from_row
from .input_provenance import view_geometry,source_record,window_provenance,stack_provenance


def covering_centers(length, phase=1):
    if length<1:raise ValueError('empty Series')
    centers=list(range(min(phase,length-1),length,3))
    if centers[0]>1:centers.insert(0,0)
    if centers[-1]<length-2:centers.append(length-1)
    return np.asarray(centers,dtype=int)


def curriculum_centers(centers,epoch,warmup_epochs):
    """A training-only context schedule; final training/inference stay complete."""
    if warmup_epochs<=1 or epoch>=warmup_epochs-1:return centers
    fraction=.5+.5*max(0,epoch)/(warmup_epochs-1)
    count=min(len(centers),max(2,int(np.ceil(len(centers)*fraction))))
    return centers[np.unique(np.linspace(0,len(centers)-1,count).round().astype(int))]


def retained_series(rows,rng,dropout):
    if not 0<=dropout<1:raise ValueError('series dropout must lie in [0,1)')
    keep=rng.random(len(rows))>=dropout if dropout else np.ones(len(rows),bool)
    # Never remove a plane entirely; remove all views of a dropped acquisition.
    for plane in {r['plane'] for r in rows}:
        indices=[i for i,r in enumerate(rows) if r['plane']==plane]
        if not keep[indices].any():keep[int(rng.choice(indices))]=True
    return keep


def render_view(triplet, spacing, output_size, *, crop_mm=None, center=(.5,.5), window=None,return_geometry=False,
                intensity_window_policy='series_percentile'):
    """One resize from genuine source pixels; full view preserves all four edges."""
    geometry=view_geometry(triplet.shape[1:],spacing,output_size,crop_mm=crop_mm,center=center)
    x,y,x1,y1=geometry.crop_xyxy;px,py,_,_=geometry.valid_xyxy
    triplet=triplet[:,y:y1,x:x1];oh,ow=geometry.resized_shape
    resized=cv2.resize(triplet.transpose(1,2,0).astype(np.float32),(ow,oh),interpolation=cv2.INTER_AREA)
    low,high=window if window is not None else np.percentile(triplet,[1,99])
    from .intensity_mapping import normalize_intensity,mapping_receipt
    resized=normalize_intensity(resized,(low,high),intensity_window_policy)
    result=np.zeros((output_size,output_size,3),np.float32)
    result[py:py+oh,px:px+ow]=resized
    output=(result.transpose(2,0,1).copy(),geometry.extent)
    if return_geometry:
        receipt=geometry.receipt()
        receipt['display_window_uint16']=np.asarray([low,high],np.float64)
        receipt['intensity_gain']=np.float32(1.)
        receipt.update(mapping_receipt((low,high),intensity_window_policy))
        return (*output,receipt)
    return output


def select_display_window(series,policy='series_percentile'):
    """Fixed per-Series choices; full range preserves highs but reduces contrast.

    Neither the native cache nor existing checkpoint defaults are changed.
    Full range is an input candidate, not a demonstrated classification gain.
    """
    if policy in ('series_percentile','series_companded'):return tuple(series['intensity']['display_window_uint16'])
    if policy=='native_full_range':return (0.,65535.)
    raise ValueError('unknown native intensity window policy: '+str(policy))


def render_probabilities(prob,native_shape,spacing,image_size,*,crop_mm=None,center=(.5,.5),grid=32):
    """Same source crop and letterbox as image rendering; no intensity transform."""
    geometry=view_geometry(native_shape,spacing,image_size,crop_mm=crop_mm,center=center)
    h,w=geometry.native_shape;ch,cw=geometry.crop_shape;oh,ow=geometry.resized_shape
    x,y,_,_=geometry.crop_xyxy;px,py,_,_=geometry.valid_xyxy
    gy,gx=np.mgrid[:grid,:grid].astype(np.float32);iy=(gy+.5)*image_size/grid;ix=(gx+.5)*image_size/grid
    valid=(iy>=py)&(iy<py+oh)&(ix>=px)&(ix<px+ow)
    mapx=((x+(ix-px)*cw/ow)*prob.shape[-1]/w-.5).astype(np.float32)
    mapy=((y+(iy-py)*ch/oh)*prob.shape[-2]/h-.5).astype(np.float32)
    return np.stack([cv2.remap(a.astype(np.float32),mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)*valid for a in prob])


def anatomical_center(anatomy):
    """One fixed centre per real Series keeps neighbouring slices aligned."""
    if anatomy is None:return (.5,.5),False
    p=anatomy['tissues'].astype(np.float32);q=anatomy['tissue_quality']
    mask=(p*(p>.5)*q[None,:,None,None]).max(1).sum(0)
    if mask.sum()<1:return (.5,.5),False
    y,x=np.mgrid[:mask.shape[0],:mask.shape[1]]
    cy=float(((y+.5)*mask).sum()/mask.sum()/mask.shape[0]);cx=float(((x+.5)*mask).sum()/mask.sum()/mask.shape[1])
    if not (.15<cy<.85 and .15<cx<.85):return (.5,.5),False
    return (cy,cx),True


def kneex_sagittal_receipt(receipt,size):
    """Compose the exact source-model AP reflection with the native pull map."""
    result=dict(receipt)
    flip=np.array([[-1.,0.,size-1.],[0.,1.,0.],[0.,0.,1.]],np.float64)
    result['output_to_native']=receipt['output_to_native']@flip
    result['input_valid_mask']=receipt['input_valid_mask'][:,:,::-1].copy()
    for name in ('rendered_valid_xyxy','valid_image_xyxy'):
        x0,y0,x1,y1=receipt[name]
        result[name]=np.array([size-x1,y0,size-x0,y1],np.int64)
    return result


class NativeStudyDataset(Dataset):
    def __init__(self,manifest,labels=None,*,ids=None,training=False,image_size=384,
                 views='global_local',sampling='cover',local_mm=100.,seed=42,geometry_dir=None,anatomy_dir=None,
                 context_curriculum_epochs=0,series_dropout=0.,roi_error_profile=None,load_pixel_supervision=True,
                 source_input_convention='canonical',intensity_window_policy='series_percentile',anatomy_weights=None,
                 exact_duplicate_policy='keep',local_evidence_inputs=False,coronal_bilateral_context=False,**unused):
        self.manifest_path=Path(manifest);self.manifest=json.loads(self.manifest_path.read_text(encoding='utf-8'))
        if self.manifest['config'].get('format')!='native_series_uint16_v1':raise ValueError('native cache required')
        index={s[UID]:s for s in self.manifest['studies']}
        self.ids=list(index) if ids is None else list(map(str,ids))
        self.studies=[index[u] for u in self.ids]
        self.labels=labels.set_index(UID) if labels is not None else None
        self.geometry_dir=Path(geometry_dir) if geometry_dir else None
        self.anatomy_dir=Path(anatomy_dir) if anatomy_dir else None
        self.anatomy_reader=None
        if self.anatomy_dir:
            from .anatomy_cache import AnatomyReader
            self.anatomy_reader=AnatomyReader(self.anatomy_dir,weights=anatomy_weights)
        self.load_pixel_supervision=bool(load_pixel_supervision)
        self.local_evidence_inputs=bool(local_evidence_inputs)
        self.coronal_bilateral_context=bool(coronal_bilateral_context);self.local_mm=float(local_mm)
        if self.local_evidence_inputs and (self.geometry_dir is None or self.anatomy_dir is None):
            raise ValueError('local evidence requires native geometry and rejectable anatomical proposals')
        if exact_duplicate_policy not in ('keep','collapse_exact'):raise ValueError('unknown exact duplicate policy')
        self.exact_duplicate_policy=exact_duplicate_policy
        from .exact_duplicates import ExactDuplicateResolver
        self.duplicate_resolver=ExactDuplicateResolver(self.geometry_dir,self.anatomy_dir,self.load_pixel_supervision,
                                                      anatomy_reader=self.anatomy_reader)
        from .intensity_mapping import POLICIES
        if intensity_window_policy not in POLICIES:
            raise ValueError('unknown native intensity window policy')
        self.intensity_window_policy=intensity_window_policy
        if source_input_convention not in ('canonical','kneex_ap'):raise ValueError('unknown source input convention')
        self.source_input_convention=source_input_convention
        self.size=image_size;self.training=training;self.seed=seed;self.epoch=0;self.sampling=sampling
        self.context_curriculum_epochs=int(context_curriculum_epochs);self.series_dropout=float(series_dropout)
        if self.context_curriculum_epochs<0 or not 0<=self.series_dropout<1:raise ValueError('invalid input schedule')
        self.roi_error_profile=None
        if training and roi_error_profile:
            profile=json.loads(Path(roi_error_profile).read_text(encoding='utf-8'))
            if profile.get('format')!='sourced_roi_error_profile_v1' or not profile.get('samples'):
                raise ValueError('no sourced ROI-error distribution')
            if self.labels is None or 'group_id' not in self.labels:raise ValueError('patient grouping is required for ROI-error calibration')
            groups=set(self.labels.loc[self.ids,'group_id'].astype(str))
            if set(profile['calibration_groups'])-groups:raise ValueError('ROI profile includes patients outside this training partition')
            self.roi_error_profile=profile
        definitions={'center':[(1,local_mm,(.5,.5))],'global':[(0,None,(.5,.5))],
                     'global_local':[(0,None,(.5,.5)),(1,local_mm,(.5,.5))],
                     'global_anatomic':[(0,None,(.5,.5)),(1,local_mm,(.5,.5))],
                     'global_grid':[(0,None,(.5,.5)),(2,local_mm,(.3,.3)),(3,local_mm,(.3,.7)),
                                    (4,local_mm,(.7,.3)),(5,local_mm,(.7,.7))]}
        if views not in definitions or sampling not in ('cover','legacy'):raise ValueError('unknown input recipe')
        self.views=definitions[views]
        self.anatomical_crop=views=='global_anatomic'
        if self.anatomical_crop and self.anatomy_dir is None:raise ValueError('anatomical crop requires actual tissue proposals')
        if training and any(not s['series'] for s in self.studies):raise ValueError('training study has no MRI')

    def __len__(self):return len(self.studies)

    def __getitem__(self,index):
        i,draw=index if isinstance(index,tuple) else (index,0)
        study=self.studies[i];rng=np.random.default_rng(self.seed+self.epoch*1000003+i*97+draw*1009)
        images=[];slots=[];series=[];position=[];geometry=[];view_ids=[];names=[];original_series=[];source_depth=[]
        spatial_valid=[];regions=[];region_quality=[];tissues=[];tissue_quality=[];pixel_support=[]
        crop_centers=[];localized=[];provenance=[];evidence_ids=[]
        local_regions=[];local_tissues=[];local_tissue_quality=[]
        sources=[source_record(s) for s in study['series']]
        from .exact_duplicates import keep_plan
        duplicates=self.duplicate_resolver.resolve(study) if self.exact_duplicate_policy=='collapse_exact' else keep_plan(study)
        selected=duplicates['representative_indices']
        for source_index,source in enumerate(sources):source['evidence_series']=duplicates['canonical_index'][source_index]
        keep=retained_series([study['series'][j] for j in selected],rng,self.series_dropout if self.training else 0.)
        for selected_index,real_sid in enumerate(selected):
            if not keep[selected_index]:continue
            s=study['series'][real_sid]
            key=hashlib.sha256((study[UID]+'/'+s['SeriesInstanceUID']).encode()).hexdigest()[:24]
            aligned=False;anatomy=None;native_affines=None
            if self.geometry_dir:
                gp=self.geometry_dir/(key+'.json')
                if not gp.is_file():raise FileNotFoundError('physical geometry missing: '+str(gp))
                physical=json.loads(gp.read_text());aligned=bool(physical['quality']['aligned'])
                native_affines=np.asarray(physical['affines_col_row_origin_lps_mm'],np.float64)
                if native_affines.shape!=(len(s['positions']),3,3) or not np.isfinite(native_affines).all():
                    raise ValueError('native affine/slice mapping is incomplete')
            if self.anatomy_dir:
                anatomy=self.anatomy_reader.load(study[UID],s,
                    self.geometry_dir/(key+'.json') if self.geometry_dir else None,
                    include_pixels=self.load_pixel_supervision)
            volume=np.load(s['path'],mmap_mode='r',allow_pickle=False)
            if self.intensity_window_policy in ('native_full_range','series_companded') and volume.dtype!=np.uint16:
                raise ValueError('full encoded range requires the native uint16 cache')
            pos=np.asarray(s['positions'],float);order=np.arange(len(pos))
            if volume.ndim!=3 or len(pos)!=len(volume) or not len(pos) or not np.isfinite(pos).all():
                raise ValueError('native volume/slice-position mapping is incomplete')
            sources[real_sid]['native_shape']=list(volume.shape)
            source_sagittal=self.source_input_convention=='kneex_ap' and s['plane']=='Sagittal'
            if s['plane']=='Sagittal' and s.get('side')=='R':pos=-pos
            order=np.argsort(pos,kind='stable');pos=pos[order]
            if self.sampling=='cover':
                centers=covering_centers(len(pos),int(rng.integers(0,3)) if self.training else 1)
            else:
                n=min(len(pos),32 if self.training else 128,max(8 if self.training else 16,
                    int(np.ceil(len(pos)/4)) if self.training else int(np.ceil((len(pos)+2)/3))))
                centers=np.unique(np.linspace(0,len(pos)-1,n).round().astype(int))
            if self.training:centers=curriculum_centers(centers,self.epoch,self.context_curriculum_epochs)
            spacing=float(np.median(np.diff(pos))) if len(pos)>1 else 0.
            bilateral=self.coronal_bilateral_context and s['plane']=='Coronal'
            if bilateral:
                from .coronal_views import coronal_bilateral_views
                series_views=coronal_bilateral_views(volume.shape[-2:],s['spacing_mm'],self.size,self.local_mm)
            else:series_views=self.views
            for view_id,crop,center_xy in series_views:
                located=False
                if self.anatomical_crop and crop is not None and not bilateral:
                    from .detail_view import choose_detail_roi
                    center_xy,crop,located,_=choose_detail_roi(anatomy,volume.shape[-2:],s['spacing_mm'][0],crop)
                if self.roi_error_profile and located and crop is not None:
                    from .roi_error_profile import perturb_center
                    sy,sx=map(float,s['spacing_mm'][0]);h,w=volume.shape[-2:]
                    center_xy=perturb_center(center_xy,(h*sy,w*sx),s['plane'],self.roi_error_profile,rng)
                sid=len(names);names.append(s['SeriesInstanceUID'])
                for c in centers:
                    native=order[[max(0,c-1),c,min(len(pos)-1,c+1)]]
                    triplet=np.asarray(volume[native])
                    image,extent,receipt=render_view(triplet,s['spacing_mm'][int(order[c])],self.size,
                        crop_mm=crop,center=center_xy,window=select_display_window(s,self.intensity_window_policy),return_geometry=True,
                        intensity_window_policy=self.intensity_window_policy)
                    if source_sagittal:
                        image=image[:,:,::-1].copy()
                        receipt=kneex_sagittal_receipt(receipt,self.size)
                    provenance.append(window_provenance(s,native,receipt,native_affines))
                    evidence_ids.append(np.stack([np.full(3,duplicates['canonical_index'][real_sid],np.int64),native],axis=-1))
                    images.append(image);slots.append(s['slot']);series.append(sid);original_series.append(real_sid)
                    position.append(pos[c]);view_ids.append(view_id)
                    relative=2*(pos[c]-pos[0])/max(pos[-1]-pos[0],1e-6)-1
                    geometry.append([relative,np.log1p(abs(spacing)),extent/130.,s.get('fluid_sensitive',-1)])
                    source_depth.append([relative/2.,1./max(1,len(pos)-1)])
                    crop_centers.append(center_xy);localized.append(located)
                    spatial_valid.append(aligned)
                    if anatomy is not None:
                        kw=dict(crop_mm=crop,center=center_xy)
                        regions.append(render_probabilities(anatomy['regions'][int(order[c])],volume.shape[-2:],s['spacing_mm'][int(order[c])],self.size,**kw))
                        target=anatomy['native_tissues_u8'][int(order[c])].astype(np.float32)/255 if 'native_tissues_u8' in anatomy else anatomy['tissues'][int(order[c])]
                        support=anatomy['supervision_support_u8'][int(order[c])].astype(np.float32)/255 if 'supervision_support_u8' in anatomy else np.zeros_like(target)
                        from .masked_supervision import render_masked_targets
                        observed_target,observed_weight=render_masked_targets(target,support,volume.shape[-2:],
                            s['spacing_mm'][int(order[c])],self.size,grid=96,**kw)
                        tissues.append(observed_target);pixel_support.append(observed_weight)
                        if source_sagittal:
                            # All masks share the image's final AP reflection;
                            # cropping and source identities remain native.
                            regions[-1]=regions[-1][:,:,::-1].copy()
                            tissues[-1]=tissues[-1][:,:,::-1].copy()
                            pixel_support[-1]=pixel_support[-1][:,:,::-1].copy()
                        region_quality.append(anatomy['region_quality']);tissue_quality.append(anatomy.get('supervision_quality',np.zeros(4)))
                        if self.local_evidence_inputs:
                            # Every channel is a distinct native observation (or
                            # an explicitly repeated identity). Never copy the
                            # centre mask onto both neighbouring source images.
                            evidence_regions=[];evidence_tissues=[]
                            for frame in native:
                                channel_kw=dict(crop_mm=crop,center=center_xy)
                                evidence_regions.append(render_probabilities(anatomy['regions'][int(frame)][[2,3,4,5,6]],
                                    volume.shape[-2:],s['spacing_mm'][int(order[c])],self.size,grid=32,**channel_kw))
                                # Candidate anatomy and masked pixel targets
                                # have separate semantics. Keep proposals the
                                # same whether auxiliary targets are loaded.
                                proposal=anatomy['tissues'][int(frame)]
                                evidence_tissues.append(render_probabilities(proposal,volume.shape[-2:],
                                    s['spacing_mm'][int(order[c])],self.size,grid=96,**channel_kw))
                            rr=np.stack(evidence_regions);tt=np.stack(evidence_tissues)
                            if source_sagittal:rr=rr[...,::-1].copy();tt=tt[...,::-1].copy()
                            local_regions.append(rr);local_tissues.append(tt)
                            local_tissue_quality.append(anatomy['tissue_quality'])
        y=np.zeros(12,np.float32);w=np.zeros(12,np.float32)
        extra=stack_provenance(provenance,self.size)
        extra['channel_evidence_id']=torch.from_numpy(np.stack(evidence_ids) if evidence_ids else np.zeros((0,3,2),np.int64))
        if self.labels is not None:
            row=self.labels.loc[study[UID]];y=np.nan_to_num(row[LABELS].to_numpy(np.float32))
            w=row[[x+'__weight' for x in LABELS]].to_numpy(np.float32)
            extra.update(targets_from_row(row))
        if self.geometry_dir:extra['spatial_valid']=torch.tensor(spatial_valid,dtype=torch.bool)
        if self.anatomy_dir:
            for name,array,shape in [('anatomy_regions',regions,(12,32,32)),('region_quality',region_quality,(12,)),
                                     ('tissue_targets',tissues,(4,96,96)),('tissue_quality',tissue_quality,(4,)),('tissue_pixel_weight',pixel_support,(4,96,96))]:
                extra[name]=torch.from_numpy(np.stack(array).astype(np.float32) if array else np.zeros((0,*shape),np.float32))
        if self.local_evidence_inputs:
            for name,array,shape in [('local_evidence_regions',local_regions,(3,5,32,32)),
                                     ('local_evidence_tissues',local_tissues,(3,4,96,96)),
                                     ('local_evidence_tissue_quality',local_tissue_quality,(4,))]:
                extra[name]=torch.from_numpy(np.stack(array).astype(np.float32) if array else np.zeros((0,*shape),np.float32))
        return {'images':torch.from_numpy(np.stack(images) if images else np.zeros((0,3,self.size,self.size),np.float32)),
                'slots':torch.tensor(slots,dtype=torch.long),'series':torch.tensor(series,dtype=torch.long),
                'position':torch.tensor(position,dtype=torch.float32),'geometry':torch.tensor(geometry,dtype=torch.float32).reshape(-1,4),
                'view_ids':torch.tensor(view_ids,dtype=torch.long),'original_series':torch.tensor(original_series,dtype=torch.long),
                'source_depth':torch.tensor(source_depth,dtype=torch.float32).reshape(-1,2),
                'crop_center':torch.tensor(crop_centers,dtype=torch.float32).reshape(-1,2),'localized':torch.tensor(localized,dtype=torch.bool),
                'target':torch.from_numpy(y),'weight':torch.from_numpy(w),UID:study[UID],
                'series_names':names,'source_series':sources,'source_series_aliases':duplicates['aliases'],
                'duplicate_resolution':duplicates,'fallback':not bool(images),**extra}


def dataset_for(manifest,labels,config,*,ids=None,training=False):
    from .data import StudyDataset
    meta=json.loads(Path(manifest).read_text(encoding='utf-8'))
    if meta['config'].get('format')=='native_series_uint16_v1':
        pixels=config.get('load_pixel_supervision',config['model'].get('anatomy_auxiliary',False))
        if training and config['model'].get('anatomy_auxiliary') and not pixels:
            raise ValueError('anatomy auxiliary training cannot disable its pixel targets')
        return NativeStudyDataset(manifest,labels,ids=ids,training=training,image_size=config['model']['image_size'],
            views=config.get('views','global_local'),sampling=config.get('sampling','cover'),
            local_mm=config.get('local_mm',100.),seed=config.get('seed',42),
            geometry_dir=config.get('geometry_dir'),anatomy_dir=config.get('anatomy_dir'),
            anatomy_weights=config.get('anatomy_weights'),
            context_curriculum_epochs=config.get('context_curriculum_epochs',0),
            series_dropout=config.get('series_dropout',0.),roi_error_profile=config.get('roi_error_profile'),load_pixel_supervision=pixels,
            source_input_convention=config.get('source_input_convention','canonical'),
            intensity_window_policy=config.get('intensity_window_policy','series_percentile'),
            exact_duplicate_policy=config.get('exact_duplicate_policy','keep'),
            coronal_bilateral_context=config.get('coronal_bilateral_context',False),
            local_evidence_inputs=config['model'].get('local_evidence',False))
    if config.get('source_input_convention','canonical')!='canonical':
        raise ValueError('source input adaptation requires the native cache and its slice mapping')
    if config.get('intensity_window_policy','series_percentile')!='series_percentile':
        raise ValueError('alternative intensity windows require the full-range native uint16 cache')
    if config.get('exact_duplicate_policy','keep')!='keep':raise ValueError('exact evidence collapse requires the native cache')
    if config.get('coronal_bilateral_context',False):raise ValueError('coronal peripheral views require the full native field')
    return StudyDataset(manifest,labels,ids=ids,training=training,seed=config.get('seed',42),
        windows_per_series=config['train_windows_per_series' if training else 'eval_windows_per_series'],
        max_windows_per_series=config.get('max_train_windows_per_series' if training else 'max_eval_windows_per_series'),
        neighbor_mm=config.get('neighbor_mm',3.),cache_read_mode=config.get('cache_read_mode','mmap'))
