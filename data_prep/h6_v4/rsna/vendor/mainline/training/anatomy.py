"""Offline SKM-TEA tissue proposals, physical transport and soft disease regions.

The released model has FOUR merged tissues, no ACL/MCL masks. Region priors
are explicitly geometric; only the four tissue probabilities supervise pixels.
Reliability below is a rejection heuristic, never a calibrated accuracy score.
"""
from pathlib import Path
import argparse,hashlib,json,sys
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import map_coordinates,gaussian_filter
from .common import UID,save_json,file_hash
from .geometry_bank import resolve_anatomy_laterality
from .anatomy_runtime import native_volume

TISSUES=['patellar_cartilage','femoral_cartilage','tibial_cartilage','meniscus']
VERSION='two-model-canonical-AP-regions-v4'


def topology_reliability(prob,affines,quality):
    """Reject anatomically inconsistent centroids; passing is not validation."""
    world=world_grid(affines,prob.shape[-1]);centers=[];reasons=[];q=quality.copy()
    for c in range(4):
        w=prob[:,c]*(prob[:,c]>.5);centers.append((world*w[:,None]).sum((0,2,3))/max(float(w.sum()),1e-6))
    pc,fc,tc,men=np.array(centers)
    if q[1]>0 and q[2]>0 and not 0<fc[2]-tc[2]<35:
        q[1:3]=0;reasons.append('femoral/tibial superior-inferior relationship failed')
    if q[3]>0:
        if min(q[1:3])<=0 or abs(men[1]-tc[1])>20 or not -10<men[2]-tc[2]<20:
            q[3]=0;reasons.append('meniscus is not adjacent to the tibiofemoral joint')
    if q[0]>0 and q[1]>0 and not (pc[1]<fc[1] and pc[2]>tc[2]):
        q[0]=0;reasons.append('patellar cartilage anterior/superior relationship failed')
    return q,{'centroids_lps_mm':np.asarray(centers).tolist(),'rejections':reasons}


def key_for(uid,sid):return hashlib.sha256((uid+'/'+sid).encode()).hexdigest()[:24]


def load_segmenter(weights,device):
    vendor=Path(__file__).parent/'vendor_runtime/monai060'
    if not vendor.is_dir():raise FileNotFoundError('isolated MONAI 0.6 runtime missing')
    sys.path.insert(0,str(vendor.resolve()))
    from monai.networks.nets import VNet
    model=VNet(spatial_dims=2,in_channels=1,out_channels=4,
               act=('elu',{'inplace':True}),dropout_prob=0.,dropout_dim=2)
    state=torch.load(weights,map_location='cpu',weights_only=True)['state_dict']
    model.load_state_dict({k.removeprefix('model.net.'):v for k,v in state.items()},strict=True)
    return model.to(device).eval().requires_grad_(False)


@torch.no_grad()
def segment_series(model,series,geometry,device,grid=128):
    if series['plane']!='Sagittal':raise ValueError('the released model was trained on sagittal images')
    volume=native_volume(series['path']);mean=float(volume.mean());std=max(float(volume.std()),1.)
    aff=np.asarray(geometry['affines_col_row_origin_lps_mm'])
    # Official annotation orientation is SI,AP,LR. Our canonical sagittal
    # arrays are SI,PA: physically inspect and reverse AP before inference.
    if abs(aff[0,1,0])/np.linalg.norm(aff[0,:,0])<.97 or abs(aff[0,2,1])/np.linalg.norm(aff[0,:,1])<.97:
        raise ValueError('sagittal obliquity exceeds supported transfer orientation')
    flip_x=aff[0,1,0]<0;flip_y=aff[0,2,1]>0
    probabilities=[];agreements=[];native=[];native_count=np.zeros(4);native_conf=np.zeros(4);lost=np.zeros(4);confident_count=np.zeros(4)
    for z in range(len(volume)):
        x=(np.asarray(volume[z],np.float32)-mean)/std
        if flip_x:x=x[:,::-1]
        if flip_y:x=x[::-1]
        sy,sx=series['spacing_mm'][z]
        shape=(max(16,round(x.shape[1]*sx/.3125)),max(16,round(x.shape[0]*sy/.3125)))
        x=cv2.resize(x,shape,interpolation=cv2.INTER_LINEAR)
        t=torch.from_numpy(x.copy())[None,None].to(device)
        h,w=t.shape[-2:];t=F.pad(t,(0,(-w)%16,0,(-h)%16),mode='replicate')
        # Only intensity perturbation, no anatomically invalid mirror ensemble.
        p=model(t).sigmoid()[...,:h,:w];q=model(t*.9).sigmoid()[...,:h,:w]
        p=F.interpolate(p,size=volume.shape[-2:],mode='bilinear',align_corners=False)[0].cpu().numpy()
        q=F.interpolate(q,size=volume.shape[-2:],mode='bilinear',align_corners=False)[0].cpu().numpy()
        if flip_x:p=p[:,:,::-1];q=q[:,:,::-1]
        if flip_y:p=p[:,::-1];q=q[:,::-1]
        native.append(np.rint(np.clip(p,0,1)*255).astype(np.uint8))
        foreground=p>.5;native_count+=foreground.sum((1,2));native_conf+=(p*foreground).sum((1,2))
        area=np.stack([cv2.resize(a,(grid,grid),interpolation=cv2.INTER_AREA) for a in p])
        restored=np.stack([cv2.resize(a,(p.shape[-1],p.shape[-2])) for a in area])
        confident_count+=(p>.8).sum((1,2));lost+=((p>.8)&(restored<.8)).sum((1,2))
        # This is a localization envelope, NOT a calibrated low-resolution
        # probability or a pixel target. Native probabilities teach pixels.
        envelope=F.adaptive_max_pool2d(torch.from_numpy(p.copy()),(grid,grid)).numpy()
        probabilities.append(envelope)
        agreements.append(np.stack([cv2.resize(abs(a-b),(grid,grid),interpolation=cv2.INTER_AREA) for a,b in zip(p,q)]))
    p=np.stack(probabilities);difference=np.stack(agreements)
    foreground=p>.5;count=foreground.sum((0,2,3));fraction=native_count/np.prod(volume.shape)
    confidence=native_conf/np.maximum(native_count,1)
    stable=1-(difference*foreground).sum((0,2,3))/np.maximum(count,1)
    plausible=(fraction>.00005)&(fraction<.12)&(confidence>.65)&(native_count>30)
    reliability=np.clip((confidence-.5)*2,0,1)*np.clip(stable,0,1)*plausible
    if not geometry['quality']['aligned']:reliability*=0
    return p,reliability.astype(np.float32),{'fraction':fraction.tolist(),'confidence':confidence.tolist(),
        'intensity_stability':stable.tolist(),'normalization':'whole-series mean/std; 0.3125mm; SI/AP',
        'native_confident_voxels':confident_count.tolist(),'old_area_resize_lost_confident_fraction':(lost/np.maximum(confident_count,1)).tolist(),
        'limits':'qDESS to clinical-protocol transfer unvalidated; heuristic reliability, not accuracy'},np.stack(native)


def grid_affines(series,geometry,grid):
    a=np.asarray(geometry['affines_col_row_origin_lps_mm'],float).copy()
    h,w=series['native_shape'][-2:]
    # OpenCV half-pixel mapping, including centre offset.
    a[:,:,2]+=a[:,:,0]*(w/grid/2-.5)+a[:,:,1]*(h/grid/2-.5)
    a[:,:,0]*=w/grid;a[:,:,1]*=h/grid
    return a


def world_grid(affines,grid):
    y,x=np.mgrid[:grid,:grid]
    return (affines[:,:,0,None,None]*x+affines[:,:,1,None,None]*y+affines[:,:,2,None,None])


def transport(prob,source_aff,target_aff,grid):
    """Tri-linear physical mapping; points outside acquired volume get zero."""
    normal=np.cross(source_aff[0,:,0],source_aff[0,:,1]);normal/=np.linalg.norm(normal)
    depth=source_aff[:,:,2]@normal;order=np.argsort(depth);depth=depth[order]
    source_aff=source_aff[order];prob=prob[order]
    world=world_grid(target_aff,grid);relative=world-source_aff[0,:,2][None,:,None,None]
    uv=np.einsum('ij,djhw->dihw',np.linalg.pinv(source_aff[0,:,:2]),relative)
    d=np.einsum('j,djhw->dhw',normal,world)
    z=np.interp(d,depth,np.arange(len(depth)),left=-10,right=len(depth)+10)
    return np.stack([map_coordinates(prob[:,c],[z,uv[:,1],uv[:,0]],order=1,mode='constant',cval=0.)
                     for c in range(4)],axis=1).astype(np.float32)


def regions_from_tissues(prob,world,reliability,side):
    """Soft regional evidence. No disease diagnosis or artificial ligament GT."""
    d,_,h,w=prob.shape;regions=np.zeros((d,12,h,w),np.float32);quality=np.zeros(12,np.float32)
    joint=np.maximum(prob[:,1],np.maximum(prob[:,2],prob[:,3]))
    weight=joint*(joint>.3);total=weight.sum()
    if total<1:return regions,quality
    center=(world*weight[:,None]).sum((0,2,3))/total
    delta=world-center[None,:,None,None]
    from .ligament_atlas import proposals
    atlas_path=Path(__file__).parent/'data/ligament_atlas.json'
    atlas=json.loads(atlas_path.read_text(encoding='utf-8'))
    ligament,frame=proposals(prob,world,reliability,side,atlas)
    if frame is not None:
        delta=world-frame['origin'][None,:,None,None]
        delta=np.einsum('ij,djhw->dihw',frame['basis_medial_posterior_superior'].T,delta)
    if side in ('L','R'):
        # LPS x increases towards patient's left; medial is towards midline.
        medial_coordinate=delta[:,0] if frame is not None else delta[:,0]*(-1 if side=='L' else 1)
        medial=1/(1+np.exp(np.clip(-medial_coordinate/2.,-30,30)))
        lateral=1-medial
        for label,mask,tissue in [(2,medial,3),(3,lateral,3),(4,medial,1),(5,lateral,1)]:
            base=prob[:,tissue] if tissue==3 else np.maximum(prob[:,1],prob[:,2])
            regions[:,label]=base*mask
            quality[label]=reliability[tissue] if tissue==3 else min(reliability[1:3])
        if frame is None:quality[2:6]=0  # merged tissues without a supported compartment split
        regions[:,9]=medial*np.exp(-((delta[:,1]-30)/22)**2-(delta[:,2]/30)**2)
        quality[9]=.2*min(reliability[1:])
    for name,label in [('ACL',0),('MCL',1)]:
        if name in ligament:regions[:,label],quality[label]=ligament[name]
    pc=prob[:,0];pc_weight=pc*(pc>.3)
    if pc_weight.sum()>1:
        pc_center=(world*pc_weight[:,None]).sum((0,2,3))/pc_weight.sum()
        distance=((world-pc_center[None,:,None,None])**2).sum(1)
        regions[:,6]=np.maximum(pc,prob[:,1]*np.exp(-distance/(2*25**2)))
        quality[6]=min(reliability[:2])
    shell=gaussian_filter(joint,sigma=(0,3,3));shell/=max(float(shell.max()),1e-6)
    regions[:,7]=shell;regions[:,8]=shell;quality[7:9]=.2*min(reliability[1:])
    # Contusion/fracture remain whole-field. No invented marrow/cortex mask.
    return regions,quality


def select_source(candidates,target,geometry,side,registration_cache=None):
    """Select same-frame evidence or verify cross-frame correspondence on MRI."""
    if not geometry or not geometry['quality']['aligned']:return None,None,[]
    sid=target['SeriesInstanceUID']
    options=[x for x in candidates if not (side in ('L','R') and x[2].get('explicit_side') in ('L','R') and side!=x[2]['explicit_side'])]
    compatible=[x for x in options if x[1]['SeriesInstanceUID']==sid or
                (geometry.get('frame_of_reference_uid') and x[2].get('frame_of_reference_uid')==geometry['frame_of_reference_uid'])]
    existing=max(compatible,key=lambda x:x[0]) if compatible else None
    if existing is not None and existing[0]>0:return existing,None,[]
    from .series_registration_rigid import source_center,register_pair
    from .anatomy_runtime import active_runtime
    runtime=active_runtime()
    attempts=[]
    for candidate in sorted((x for x in options if x[0]>0 and x[1]['SeriesInstanceUID']!=sid),key=lambda x:x[0],reverse=True)[:2]:
        _,source,source_geometry,p,q,_=candidate
        try:
            center=source_center(source,source_geometry,{'tissues':p,'tissue_quality':q})
            hashes=({'source_sha256':runtime.file_hash(source['path']),'target_sha256':runtime.file_hash(target['path'])}
                    if runtime is not None else {})
            result=register_pair(source,target,source_geometry,geometry,center,cache_dir=registration_cache,**hashes)
        except ValueError as exc:
            attempts.append({'source_series':source['SeriesInstanceUID'],'accepted':False,'reason':str(exc)})
            continue
        attempts.append({k:result.get(k) for k in ('source_series','accepted','reason','input_sha256','validation_score','heldout_coverage')})
        if result['accepted']:return candidate,result,attempts
    return existing,None,attempts


def project_source(source,target,geometry,side,grid,registration=None):
    """Regional feature evidence only; registration never creates pixel truth."""
    _,ss,sg,p,q,source_detail=source
    sa=grid_affines(ss,sg,grid);ta=grid_affines(target,geometry,grid)
    sampling_affines=ta
    detail=dict(source_detail)
    if registration is not None:
        from .series_registration_rigid import transformed_affines
        sampling_affines=transformed_affines(ta,registration)
        q=q*.5  # explicit conservative discount, not calibrated accuracy
        detail['cross_series_registration']={**registration,'regional_quality_multiplier':.5,
            'consumer_scope':'regional pooling/cropping only; no target dense-pixel supervision'}
    p=transport(p,sa,sampling_affines,grid) if target['SeriesInstanceUID']!=ss['SeriesInstanceUID'] else p
    regions,rq=regions_from_tissues(p,world_grid(ta,grid),q,side)
    from .anatomy_quality import uncertainty_regions
    spacing=(np.linalg.norm(ta[0,:,1]),np.linalg.norm(ta[0,:,0]))
    regions=uncertainty_regions(regions,rq,spacing)
    return p,q,regions,rq,detail,ss['SeriesInstanceUID']


def publish_manifest(output,contract,rows,*,append=False):
    """Atomically publish completed Series without hiding previous batches."""
    output=Path(output);path=output/'manifest.json';combined={}
    if append and path.exists():
        previous=json.loads(path.read_text(encoding='utf-8'))
        combined={(r[UID],r['SeriesInstanceUID']):r for r in previous.get('series',[])}
    combined.update({(r[UID],r['SeriesInstanceUID']):r for r in rows})
    values=list(combined.values())
    save_json(path,{'contract':contract,'series':values,'studies':len({r[UID] for r in values}),
        'active_producer_signature':contract.get('producer_signature'),
        'per_series_contracts':'Each Series JSON records its exact producer; previous completed batches are retained.',
        'scope':'external pretrained proposals; no knee-classifier training; no anatomical GT claim'})


def prepare_study(study,geometry_dir,runtime,grid=128):
    """Frozen GPU inference and same CPU consensus; one complete Study payload."""
    with runtime.activate(),runtime.profile.stage('prepare_study',gpu=True):
        return _prepare_study(study,geometry_dir,runtime,grid)


def _prepare_study(study,geometry_dir,runtime,grid):
    model,localizer,independent=runtime.models();device=runtime.device
    from .anatomy_quality import localize_series,check_against_joint,independent_tissues,consensus_supervision
    candidates=[];geometries={};uid=study[UID];joints={};dense={};rejections=[]
    for s in study['series']:
        key=key_for(uid,s['SeriesInstanceUID']);gpath=Path(geometry_dir)/(key+'.json')
        if not gpath.is_file():continue
        g=json.loads(gpath.read_text());geometries[s['SeriesInstanceUID']]=g
        if s['plane']!='Sagittal' or not g['quality']['aligned']:continue
        try:
            joint=localize_series(localizer,s);joints[s['SeriesInstanceUID']]=joint
            prob,reliability,detail,native=segment_series(model,s,g,device,grid)
            other,observed=independent_tissues(independent,s,joint,device)
            target,support,sq,agreement=consensus_supervision(native,other,observed,reliability)
            # Regional readout must not keep trusting a tissue that pixel
            # supervision rejected. Localize on joint support from BOTH
            # segmenters, then apply physical plausibility checks.
            shared=[]
            for first,second in zip(native,other):
                consensus=np.sqrt(first.astype(np.float32)/255*(second.astype(np.float32)/255))
                shared.append(F.adaptive_max_pool2d(torch.from_numpy(consensus),(grid,grid)).numpy())
            prob=np.stack(shared)
            reliability=sq*2  # localization confidence, before the extra pixel-loss discount
            reliability,topology=topology_reliability(prob,grid_affines(s,g,grid),reliability)
            detail['topology_check']=topology
            reliability,check=check_against_joint(prob,reliability,joint);detail['independent_joint_check']=check
            sq=reliability*.5
            detail['pixel_supervision']=agreement
            dense[s['SeriesInstanceUID']]=(target,support,sq)
            candidates.append((float(reliability.mean()),s,g,prob,reliability,detail))
        except ValueError as exc:rejections.append({'SeriesInstanceUID':s['SeriesInstanceUID'],'reason':str(exc)})
    side,side_evidence=resolve_anatomy_laterality(geometries)
    arrays=[x for row in candidates for x in (row[3],row[4])]+[x for row in dense.values() for x in row]
    payload_nbytes=sum(x.nbytes for x in {id(x):x for x in arrays}.values())
    return {'study':study,'candidates':candidates,'geometries':geometries,'joints':joints,
            'dense':dense,'rejections':rejections,'side':side,'side_evidence':side_evidence,'grid':grid,
            'payload_array_bytes':payload_nbytes}


def finish_study(prepared,geometry_dir,output,contract,runtime=None):
    """Same registration, regional readout and lossless writer; no model forward."""
    if runtime is not None:
        with runtime.activate(),runtime.profile.stage('finish_study'):
            return _finish_study(prepared,geometry_dir,output,contract,runtime)
    return _finish_study(prepared,geometry_dir,output,contract,None)


def _finish_study(prepared,geometry_dir,output,contract,runtime):
    study=prepared['study'];uid=study[UID];out=Path(output);out.mkdir(parents=True,exist_ok=True)
    candidates=prepared['candidates'];geometries=prepared['geometries'];joints=prepared['joints']
    dense=prepared['dense'];rejections=prepared['rejections'];side=prepared['side'];side_evidence=prepared['side_evidence'];grid=prepared['grid']
    rows=[];hash_file=runtime.file_hash if runtime is not None else file_hash
    for s in study['series']:
        sid=s['SeriesInstanceUID'];g=geometries.get(sid);key=key_for(uid,sid)
        source,registration,attempts=select_source(candidates,s,g,side,out/'registration')
        if source:
            p,q,regions,rq,detail,source_sid=project_source(source,s,g,side,grid,registration)
            detail['registration_attempts']=attempts
        else:
            p=np.zeros((len(s['positions']),4,grid,grid),np.float32);q=np.zeros(4,np.float32)
            regions=np.zeros((len(p),12,grid,grid),np.float32);rq=np.zeros(12,np.float32)
            detail={'fallback':'no supported same-frame or image-verified sagittal source','rejected_sources':rejections,
                    'registration_attempts':attempts};source_sid=None
        path=out/(key+'.npz')
        payload=dict(tissues=p.astype(np.float16),tissue_quality=q,regions=regions.astype(np.float16),region_quality=rq)
        joint=joints.get(sid,{});payload['joint_quality']=np.float32(joint.get('quality',0))
        if 'box_xyxy_normalized' in joint:payload['joint_box']=np.asarray(joint['box_xyxy_normalized'],np.float32)
        payload['supervision_quality']=np.zeros(4,np.float32)
        if sid in dense:
            target,support,sq=dense[sid];payload.update(native_tissues_u8=target,supervision_support_u8=support,supervision_quality=sq)
        runtime.save(path,payload) if runtime is not None else np.savez_compressed(path,**payload)
        row={UID:uid,'SeriesInstanceUID':sid,'source_series':source_sid,'path':str(path.resolve()),
             'producer_signature':contract['producer_signature'],
             'region_quality':rq.tolist(),'tissue_quality':q.tolist(),'detail':detail,
             'explicit_side':side,'native_sha256':hash_file(s['path']),
             'laterality_evidence':side_evidence,
             'joint_localization':joint,'supervision_quality':payload['supervision_quality'].tolist()}
        gp=Path(geometry_dir)/(key+'.json');row['geometry_sha256']=hash_file(gp) if gp.exists() else None
        save_json(path.with_suffix('.json'),{**row,'contract':contract});rows.append(row)
    return rows


def build(manifest,geometry_dir,weights,output,*,limit=None,device='cuda',assets_dir=None,append=False,runtime=None):
    """Backward-compatible entry; pass one runtime across successive batches."""
    from .anatomy_runtime import AnatomyRuntime
    owned=runtime is None
    runtime=runtime or AnatomyRuntime(weights,device,assets_dir)
    runtime.validate(weights,device,assets_dir)
    try:
        with runtime.activate(),runtime.profile.stage('build_total',gpu=True):
            return _build(manifest,geometry_dir,weights,output,limit=limit,device=device,assets_dir=assets_dir,append=append,runtime=runtime)
    finally:
        if owned:runtime.close()


def _build(manifest,geometry_dir,weights,output,*,limit=None,device='cuda',assets_dir=None,append=False,runtime):
    torch.set_num_threads(4);cv2.setNumThreads(1)
    data=json.loads(Path(manifest).read_text(encoding='utf-8'));out=Path(output);out.mkdir(parents=True,exist_ok=True)
    rows=[];hash_file=runtime.file_hash
    assets=Path(assets_dir or (Path(weights).parent.parent/'kneexnet' if Path(weights).parent.name=='skm_tea' else Path(weights).parent/'kneexnet'))
    contract={'version':VERSION,'weights_sha256':hash_file(weights),'code_sha256':hash_file(__file__),
              'source_manifest_sha256':hash_file(manifest),'tissues':TISSUES,'grid':128,
              'localizer_sha256':hash_file(assets/'localizer.pt'),
              'independent_segmenter_sha256':hash_file(assets/'best_model_256_noise_notebook_5_0.15.pth'),
              'quality_code_sha256':hash_file(Path(__file__).with_name('anatomy_quality.py'))}
    contract['ligament_atlas_sha256']=hash_file(Path(__file__).parent/'data/ligament_atlas.json')
    contract['ligament_code_sha256']=hash_file(Path(__file__).with_name('ligament_atlas.py'))
    contract['registration_code_sha256']=hash_file(Path(__file__).with_name('series_registration.py'))
    contract['registration_rigid_code_sha256']=hash_file(Path(__file__).with_name('series_registration_rigid.py'))
    from .anatomy_contract import producer_signature,cache_matches
    contract['producer_signature']=producer_signature(weights,assets,hash_function=hash_file)
    studies=data['studies'][:limit] if limit else data['studies']
    for study in studies:
        reusable=[]
        for s in study['series']:
            path=out/(key_for(study[UID],s['SeriesInstanceUID'])+'.json')
            if not path.exists():break
            previous=json.loads(path.read_text(encoding='utf-8'));old=previous.get('contract',{})
            gp=Path(geometry_dir)/(key_for(study[UID],s['SeriesInstanceUID'])+'.json')
            if (not Path(previous['path']).is_file() or not cache_matches(previous,contract['producer_signature'])
                or previous.get('geometry_sha256')!=(hash_file(gp) if gp.exists() else None)
                or previous.get('native_sha256')!=hash_file(s['path'])):break
            reusable.append({k:v for k,v in previous.items() if k!='contract'})
        if len(reusable)==len(study['series']) and reusable:
            rows.extend(reusable);continue
        prepared=prepare_study(study,geometry_dir,runtime)
        rows.extend(finish_study(prepared,geometry_dir,out,contract,runtime))
        print(f"anatomy {len(rows)} Series; candidate scans {len(prepared['candidates'])}",flush=True)
        publish_manifest(out,contract,rows,append=append)
        del prepared
    publish_manifest(out,contract,rows,append=append)
    return {'rows':rows,'runtime':runtime.snapshot()}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',default='F:/rsnaknee/cache_native/manifest.json')
    p.add_argument('--geometry',default='F:/rsnaknee/geometry');p.add_argument('--weights',default='F:/rsnaknee/pretrained/skm_tea/model-raw.ckpt')
    p.add_argument('--output',default='F:/rsnaknee/anatomy');p.add_argument('--limit',type=int);p.add_argument('--device',default='cuda')
    a=p.parse_args();build(a.manifest,a.geometry,a.weights,a.output,limit=a.limit,device=a.device)
