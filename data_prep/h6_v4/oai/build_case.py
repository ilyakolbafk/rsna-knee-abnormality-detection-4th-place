"""Declared v8.3/LPS adaptation of public v4; never claims private hrepro parity."""
from common import *
import argparse, ast, gc, importlib.util, io, tarfile, sys, zipfile
import numpy as np
import torch
import cv2
import pydicom
import psutil
import pandas as pd

SOURCE=RUN/'component_discovery/source'
ORIENT=HANDOFF/'orientation_v8_3/runtime'
PUBLIC=HANDOFF/'reference/public_v4'
sys.path[:0]=[str(SOURCE/'cloud_code'),str(ORIENT),str(ORIENT/'rsna_knee')]
from training.dicom_geometry import pixels_from_dataset
from training.input_provenance import view_geometry
from training.intensity_mapping import normalize_intensity
from canonical_inputs import transform_spec,side_record
from h1_adapter import apply_tokens
from resolver import LateralityResolver
from rsna_knee.dicom import center_crop_mm,resize_image
from rsna_knee.grayscale_cache import sample_window_indices
from rsna_knee.native_neighbor_bank import normalization_bounds,quantize_with_bounds
from rsna_knee.compartment_locator_v2 import analyse_series

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

KB=module('oai_public_anchor_v2',PUBLIC/'knee_anchor_bank_v2.py')
FAMILIES=['sagiw','sagds','cormpr','coriw','axmpr']
PLANE={'sagiw':'Sagittal','sagds':'Sagittal','cormpr':'Coronal','coriw':'Coronal','axmpr':'Axial'}
MODEL=None;RESOLVER=None
META_COLS=['series_idx','slot','plane','view_id','rel_pos','log_spacing','extent_130','fluid','fatsat','slice_order','n_slices','real_series_idx']

def exact_function(path,name,scope):
    tree=ast.parse(Path(path).read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),scope)
    return scope[name]

quantize_series=exact_function(SOURCE/'cloud_code/training/native_cache.py','quantize_series',{'np':np})

def resources():
    global MODEL,RESOLVER
    if MODEL is None:
        config=json.loads((RUN/'BUILD_CONFIG.frozen.json').read_text())
        for rel,expected in config['component_sha256'].items():
            if sha(RUN/rel)!=expected:raise ValueError('STALE_RECIPE_COMPONENT:'+rel)
        torch.set_num_threads(2);cv2.setNumThreads(1)
        loader=exact_function(SOURCE/'cloud_code/training/anatomy.py','load_segmenter',
                              {'Path':Path,'sys':sys,'torch':torch,'__file__':str(SOURCE/'cloud_code/training/anatomy.py')})
        assert sha(SOURCE/'anatomy.pt')=='5904dd16eb9a4fd7a1caf90ac977f8fe2ef364908598c23c209db3fe7fd1563e'
        MODEL=loader(SOURCE/'anatomy.pt','cpu');RESOLVER=LateralityResolver(ORIENT,device='cpu')

def source_paths(row):
    lookup=json.loads((RUN/'LOCAL_RAW_LOOKUP.json').read_text());out={}
    for item in row['objects']:
        p=RUN/'scratch'/row['case_id']/'downloads'/item['object']['download_alias']
        if not p.is_file():
            candidates=lookup.get(item['archive_sha256'],[])
            p=next((Path(x['path']) for x in candidates if Path(x['path']).is_file()),p)
        if not p.is_file():raise RuntimeError('RAW_NOT_AVAILABLE')
        if p.stat().st_size!=int(item['object']['archive_bytes']) or sha(p)!=item['archive_sha256']:
            raise RuntimeError('RAW_SHA_CONFLICT')
        out[item['object']['family']]=p
    return out

def decode(row):
    paths=source_paths(row);series=[];patients=set();dates=set();sops=set()
    groups=json.loads((HANDOFF/'reference/oai/SOURCE_GROUPS.frozen.json').read_text())['groups']
    group=groups.get(row['case_id'])
    for item in sorted(row['objects'],key=lambda x:FAMILIES.index(x['object']['family'])):
        o=item['object'];family=o['family'];frames=[];excluded=[];count=0
        with tarfile.open(paths[family],'r:gz') as archive:
            for member in archive:
                if member.issym() or member.islnk() or Path(member.name).is_absolute() or '..' in Path(member.name).parts:
                    raise ValueError('UNSAFE_ARCHIVE_MEMBER')
                if not member.isfile():continue
                if member.size>20_000_000:raise ValueError('SOURCE_FRAME_SIZE')
                raw=archive.extractfile(member).read();ds=pydicom.dcmread(io.BytesIO(raw),force=True)
                if ds.Modality!='MR' or int(getattr(ds,'NumberOfFrames',1))!=1:raise ValueError('EXPECTED_CLASSIC_MR')
                if str(ds.PatientID)!=o['src_subject_id'] or str(ds.StudyDate)!=o['acquisition_date_path']:
                    raise ValueError('PATIENT_DATE_CONFLICT')
                if (str(ds.StudyInstanceUID),str(ds.SeriesInstanceUID))!=(item['source_study_uid'],item['source_series_uid']):
                    raise ValueError('REAL_UID_CONFLICT')
                for tag in ['Laterality','ImageLaterality']:
                    v=str(getattr(ds,tag,'')).strip().upper()
                    if v in ('L','R') and v!=row['side']:raise ValueError('SOURCE_SIDE_CONFLICT')
                patients.add(str(ds.PatientID));dates.add(str(ds.StudyDate));count+=1
                sop=str(ds.SOPInstanceUID)
                if sop in sops:raise ValueError('DUPLICATE_SOP')
                sops.add(sop)
                rec={'sop_uid':sop,'archive_member':member.name,'sha256':hashlib.sha256(raw).hexdigest()}
                if 'PROJECTION IMAGE' in [str(x) for x in ds.get('ImageType',[])]:
                    if family not in ('cormpr','axmpr'):raise ValueError('UNEXPECTED_PROJECTION')
                    excluded.append(rec);continue
                payload=pixels_from_dataset(ds);iop=np.asarray(ds.ImageOrientationPatient,float);ipp=np.asarray(ds.ImagePositionPatient,float)
                normal=np.cross(iop[:3],iop[3:]);plane=['Sagittal','Coronal','Axial'][int(abs(normal).argmax())]
                if plane!=PLANE[family]:raise ValueError('PLANE_CONFLICT')
                aff=np.stack([iop[:3]*payload.spacing[1],iop[3:]*payload.spacing[0],ipp],axis=1)
                tags={k:list(getattr(ds,k)) for k in ['ImageOrientationPatient','ImagePositionPatient','PixelSpacing']}
                tags.update(Rows=int(ds.Rows),Columns=int(ds.Columns))
                frames.append({**rec,'image':payload.image,'tags':tags,'aff':aff,'pos':float(ipp@normal),
                               'laterality':str(ds.get('Laterality','')),
                               'frame_uid':str(ds.get('FrameOfReferenceUID','')),'description':str(ds.get('SeriesDescription',''))})
        if count!=item['source_frames'] or len(frames)!=item['diagnostic_frames']:raise ValueError('SOURCE_FRAME_COUNT')
        frames.sort(key=lambda f:f['pos'])
        if np.min(np.diff([f['pos'] for f in frames]))<1e-4:raise ValueError('DUPLICATE_OR_DEGENERATE_POSITION')
        aff=np.stack([f['aff'] for f in frames])
        gaps=np.diff([f['pos'] for f in frames])
        if np.max(gaps)>1.5*np.median(gaps)+1e-4:raise ValueError('LARGE_SLICE_INTERVAL')
        if len({f['frame_uid'] for f in frames})!=1 or not frames[0]['frame_uid']:raise ValueError('MIXED_OR_MISSING_FRAME_OF_REFERENCE')
        if any(not np.allclose(f['tags']['ImageOrientationPatient'],frames[0]['tags']['ImageOrientationPatient'],atol=1e-4,rtol=0) for f in frames):raise ValueError('MIXED_ORIENTATION')
        if any(not np.allclose(f['tags']['PixelSpacing'],frames[0]['tags']['PixelSpacing'],atol=1e-4,rtol=0) for f in frames):raise ValueError('MIXED_PIXEL_SPACING')
        if len({f['image'].shape for f in frames})!=1:raise ValueError('MIXED_SOURCE_SHAPE')
        if not np.allclose(aff[:,:,:2],aff[0,:,:2],atol=1e-3):raise ValueError('INCONSISTENT_INPLANE_GEOMETRY')
        normal=np.cross(aff[0,:,0],aff[0,:,1]);normal/=np.linalg.norm(normal)
        offsets=aff[:,:,2]-aff[0,:,2];drift=offsets-np.outer(offsets@normal,normal)
        if np.max(np.linalg.norm(drift,axis=1))>=1.:raise ValueError('INPLANE_DRIFT_EXCEEDS_ORIGINAL_GEOMETRY_GATE')
        if group:
            e=group['families'][family]
            if (e['source_study_uid'],e['series_uid'],e['archive_sha256'])!=(item['source_study_uid'],item['source_series_uid'],item['archive_sha256']):raise ValueError('COMPOSITE_MAP_CONFLICT')
        series.append({'family':family,'plane':PLANE[family],'sid':item['source_series_uid'],'source_study_uid':item['source_study_uid'],
                       'archive_sha256':item['archive_sha256'],'archive_bytes':int(o['archive_bytes']),
                       'frames':frames,'aff':aff,'excluded':excluded,'volume':np.stack([f['image'] for f in frames]),
                       'fluid':int(family=='sagiw'),'fatsat':int(family!='coriw')})
    if len(patients)!=1 or len(dates)!=1:raise ValueError('CROSS_PATIENT_OR_DATE_MIX')
    study_ids={s['source_study_uid'] for s in series}
    if len(study_ids)>1 and (not group or not group['composite_allowed']):raise ValueError('COMPOSITE_NOT_AUTHORIZED')
    if group and (patients!={group['participant']} or dates!={group['acquisition_date']} or row['side']!=group['side'] or row['visit']!=group['visit']):raise ValueError('GROUP_IDENTITY_CONFLICT')
    frame_ids={s['frames'][0]['frame_uid'] for s in series}
    if len(frame_ids)!=1:raise ValueError('CROSS_SERIES_FRAME_OF_REFERENCE_REQUIRES_REGISTRATION_EVIDENCE')
    return series

def decide(series,row):
    scored=[];records={};planes=[];tags=[];offset=0
    for ix,s in enumerate(series):
        sp=[f['tags']['PixelSpacing'] for f in s['frames']]
        side_images=np.stack([resize_image(center_crop_mm(im,spacing,130.),336) for im,spacing in zip(s['volume'],sp)])
        n=len(side_images);low,high=(.12,.88) if ix in (0,1,4) else (.2,.8)
        ids=sample_window_indices(n,[12,12,8,6,8][ix],low,high)
        lo,hi=normalization_bounds(side_images[ids],np.ones(len(ids),bool),(.5,99.5))
        q=quantize_with_bounds(side_images,lo,hi)
        sr=RESOLVER.score_source_series(q,[f['tags'] for f in s['frames']],s['plane'],bool(s['fluid']),
              valid=np.ones(n,bool),sop_ids=[f['sop_uid'] for f in s['frames']],uid=row['study_uid'],sid=s['sid'])
        scored.append(sr);records[s['sid']]={'legacy_slot':ix,'geometry_qc':{'within_series_valid':True}}
        planes.extend({'bank_index':offset+i,'SeriesInstanceUID':s['sid'],'geometry_tags':f['tags'],'geometry_valid':True} for i,f in enumerate(s['frames']))
        tags.extend(f['laterality'] for f in s['frames'][:1]);offset+=n
    old=side_record({'series':records,'planes':planes},tags)
    loc={s['sid']:[{'pixels':f['image'],'iop':np.asarray(f['tags']['ImageOrientationPatient'],float),
                   'ipp':np.asarray(f['tags']['ImagePositionPatient'],float),'spacing':np.asarray(f['tags']['PixelSpacing'],float),
                   'rows':f['tags']['Rows'],'cols':f['tags']['Columns'],'desc':f['description'],'frame':f['frame_uid']} for f in s['frames']] for s in series}
    locmeta=pd.DataFrame([{'SeriesInstanceUID':s['sid'],'Anatomical_Plane':s['plane'],
                          'Fat_Suppression':s['fatsat'],'Fluid_Sensitive':s['fluid']} for s in series]).set_index('SeriesInstanceUID')
    try:
        _,_,_,lr=analyse_series(row['study_uid'],loc,locmeta,None)
        locator={k:lr.get(k) for k in ['joint_z','accept','sag_cor_gap','sag_series','cor_series']}
    except Exception as e:locator={'joint_z':None,'error':type(e).__name__}
    d=RESOLVER.resolve_scored(scored,old,locator['joint_z']);d['locator']=locator
    if d['bilateral'] or d['side']!=row['side']:raise ValueError('V83_ROSTER_SIDE_DISAGREEMENT')
    return d

def orient(s,decision):
    vol,encoding=quantize_series(s['volume']);n,h,w=vol.shape
    frames=s['frames'];mid=n//2;sid=s['sid']
    meta={'series':{sid:{'offset':0,'source_count':n,'series_index':0,'legacy_slot':0,
                'geometry_qc':{'within_series_valid':True},'source_inventory':{'fat_suppression':s['fatsat']}}},
          'planes':[{'bank_index':i,'SeriesInstanceUID':sid,'geometry_tags':f['tags'],'geometry_valid':True} for i,f in enumerate(frames)],
          'legacy_slots':[{'plane':s['plane']}]}
    sample={'token_mask':np.array([True]),'center_refs':np.array([mid]),'slot_ids':np.array([0]),'bank_refs':np.arange(n)[None]}
    (out,_,_),audit=apply_tokens(vol.astype(np.float32)[None],{'series_index':np.zeros(n,int)},meta,sample,decision)
    oriented=np.rint(out[0].numpy()*255).astype(np.uint16)
    spec=transform_spec(frames[mid]['tags']['ImageOrientationPatient'],s['plane'],'L')
    sd=next(x for x in decision['series'] if x['sid']==sid)
    def inplane(a):
        if spec:
            if spec['transpose']:a=np.swapaxes(a,0,1)
            if spec['flip_x']:a=a[:,::-1]
            if spec['flip_y']:a=a[::-1]
        pose=sd.get('apply_pose')
        if pose:
            a=np.rot90(a,int(pose['rot90_k']),axes=(0,1))
            if pose.get('flip_x'):a=a[:,::-1]
        if s['plane'] in ('Axial','Coronal') and sd.get('appearance_side',decision['side'])=='R':a=a[:,::-1]
        return a
    depth=bool(spec and spec['reverse_channels']) ^ bool(s['plane']=='Sagittal' and sd.get('appearance_side',decision['side'])=='R') ^ bool(sd.get('reverse_depth_extra'))
    expected=np.stack([inplane(x) for x in vol])
    order=np.arange(n)[::-1] if depth else np.arange(n)
    expected=expected[order]
    if not np.array_equal(oriented,expected):raise ValueError('V83_FULLFIELD_ADAPTER_PARITY')
    yy,xx=np.mgrid[:h,:w];coords=inplane(np.stack([xx,yy,np.ones_like(xx)],axis=-1)).astype(float)
    pull=np.stack([coords[0,1]-coords[0,0],coords[1,0]-coords[0,0],coords[0,0]],axis=1)
    aff=s['aff'][order]@pull
    s.update(volume=oriented,aff=aff,frames=[frames[i] for i in order],orientation_audit=audit,
             pull_stored_to_original=pull.tolist(),native_encoding=encoding,
             window=np.percentile(oriented[:,::4,::4],[1,99]).astype(float).tolist())
    return s

class MemoryAnchorBank(KB.AnchorBank):
    def __init__(self,model,geometry):super().__init__(model,'cpu');self.entries=geometry
    def geometry(self,uid,sid,geo_dir=None):return self.entries.get(sid)

def views(s,anchor,row):
    joint=np.asarray(anchor['joint_lps'],float);pat=anchor.get('patella_lps');plane=s['plane']
    out=[(0,None,None)]
    if plane=='Sagittal':out += [(1,100.,joint),(4,100.,np.asarray(pat) if pat is not None else joint+[0,-40,25]),(5,100.,joint+[0,35,-5])]
    elif plane=='Axial':out += [(1,100.,joint),(4,100.,np.asarray(pat) if pat is not None else joint+[0,-40,0]),(5,100.,joint+[0,35,0])]
    else:
        lateral=30 if row['side']=='L' else -30
        out += [(2,100.,joint+[-lateral,0,0]),(3,100.,joint+[lateral,0,0])]
    result=[]
    for view,crop,point in out:
        if point is None:center=(.5,.5)
        else:
            _,c,r,_=KB.project(s['aff'],np.asarray(point,float));h,w=s['volume'].shape[-2:]
            center=(float(np.clip(r/max(h-1,1),0,1)),float(np.clip(c/max(w-1,1),0,1)))
        result.append((view,crop,center,None if point is None else np.asarray(point,float).tolist()))
    return result

def render(series,anchor,row):
    px=[];meta=[];blocks=[];lineage=[];geometries=[]
    for real,s in enumerate(series):
        vol=s['volume'];aff=s['aff'];n=len(vol);delta=aff[-1,:,2]-aff[0,:,2];axis=delta/max(np.linalg.norm(delta),1e-9)
        pos=aff[:,:,2]@axis;spacing=float(np.median(np.diff(pos)));indices=np.arange(n)
        chunks=[indices[i:i+1] for i in range(n)]
        if 0<spacing<1.75:
            g=max(1,int(round(3.5/spacing)));nn=n//g+int(n%g>=max(1,g//2))
            if nn>=2:chunks=[indices[j*g:min(n,(j+1)*g)] for j in range(nn)]
        slab=np.stack([np.rint(vol[c].astype(np.float32).mean(0)).astype(vol.dtype) for c in chunks])
        slabpos=np.asarray([pos[c].mean() for c in chunks]);spacing=float(np.median(np.diff(slabpos)))
        block={'series':s['sid'],'source_study_uid':s['source_study_uid'],'family':s['family'],'plane':s['plane'],'n_slices':len(chunks),'views':[]}
        block['slabs']=[{'source_indices':c.tolist(),'source_sops':[s['frames'][i]['sop_uid'] for i in c],'weights':[1/len(c)]*len(c),'affine_lps':np.mean(aff[c],axis=0).tolist()} for c in chunks]
        used={int(i) for c in chunks for i in c};block['dropped_tail_source_sops']=[s['frames'][i]['sop_uid'] for i in range(n) if i not in used]
        for vid,crop,center,point in views(s,anchor,row):
            block['views'].append([vid,len(px)])
            for k,c in enumerate(chunks):
                sp=(float(np.linalg.norm(aff[c[0],:,1])),float(np.linalg.norm(aff[c[0],:,0])))
                vg=view_geometry(slab.shape[1:],sp,384,crop_mm=crop,center=center)
                x,y,x1,y1=vg.crop_xyxy;ox,oy,_,_=vg.valid_xyxy;oh,ow=vg.resized_shape
                resized=cv2.resize(slab[k,y:y1,x:x1].astype(np.float32),(ow,oh),interpolation=cv2.INTER_AREA)
                im=np.zeros((384,384),np.uint8);im[oy:oy+oh,ox:ox+ow]=np.rint(np.clip(normalize_intensity(resized,s['window'],'series_companded'),0,1)*255).astype(np.uint8)
                px.append(im);rel=2*(slabpos[k]-slabpos[0])/max(slabpos[-1]-slabpos[0],1e-6)-1
                slot={('Sagittal',1):1,('Sagittal',0):2,('Coronal',1):3,('Coronal',0):4,('Axial',1):5}[(s['plane'],s['fatsat'])]
                meta.append([real,slot,{'Sagittal':0,'Coronal':1,'Axial':2}[s['plane']],vid,rel,np.log1p(abs(spacing)),vg.extent/130,s['fluid'],s['fatsat'],k,len(chunks),real])
                receipt=vg.receipt();geometries.append({'row':len(px)-1,
                    'output_to_lps':(np.mean(aff[c],axis=0)@receipt['output_to_native']).tolist(),'valid_xyxy':list(vg.valid_xyxy),'crop_xyxy':list(vg.crop_xyxy),'fov_mm':receipt['source_crop_fov_mm'].tolist(),'anchor_lps':point})
        blocks.append(block)
        lineage.append({'family':s['family'],'source_study_uid':s['source_study_uid'],'series_uid':s['sid'],'archive_sha256':s['archive_sha256'],'archive_bytes':s['archive_bytes'],
             'orientation_applications':1,'orientation_audit':s['orientation_audit'],'native_encoding':s['native_encoding'],'display_window_uint16':s['window'],
             'stored_affines_lps':aff.tolist(),'frames':[{k:v for k,v in f.items() if k not in ('image','aff')} for f in s['frames']],'excluded_reference_frames':s['excluded']})
    return np.stack(px),np.asarray(meta,np.float32),{'blocks':blocks,'lineage':lineage,'view_geometry':geometries}

def build(case):
    resources();start=time.time();row=next(x for x in roster() if x['case_id']==case)
    work=RUN/'build_scratch'/case;work.mkdir(parents=True,exist_ok=True)
    series=decode(row);decision=decide(series,row)
    series=[orient(s,decision) for s in series]
    geometry={};native=[]
    for s in series:
        p=work/f"{s['family']}.npy";np.save(p,s['volume'])
        native.append({'path':str(p),'SeriesInstanceUID':s['sid'],'native_shape':list(s['volume'].shape),'plane':s['plane'],
                       'fluid_sensitive':s['fluid'],'spacing_mm':[[float(np.linalg.norm(a[:,1])),float(np.linalg.norm(a[:,0]))] for a in s['aff']]})
        geometry[s['sid']]={'affines_col_row_origin_lps_mm':s['aff'].tolist(),'quality':{'aligned':True}}
    anchor=MemoryAnchorBank(MODEL,geometry).record(row['study_uid'],{'series':native})
    if anchor['joint_lps'] is None:raise ValueError('NO_ANATOMICAL_OR_SLAB_PRIOR_ANCHOR')
    px,meta,audit=render(series,anchor,row)
    out=RUN/'cache'/f'{case}.npz';out.parent.mkdir(exist_ok=True);tmp=out.with_suffix('.npz.partial')
    config=json.loads((RUN/'BUILD_CONFIG.frozen.json').read_text())
    payload={'px':px,'meta':meta,'orientation_state':np.asarray('canonical_L_v8_3'),'case_id':np.asarray(case),
             'format':np.asarray('oai-v4-v83-lps-adapt-v1'),'roster_sha256':np.asarray(sha(ROSTER)),
             'recipe_sha256':np.asarray(config['semantic_sha256']),
             'orientation_policy_json':np.asarray(json.dumps(config['orientation_policy'],sort_keys=True))}
    with tmp.open('wb') as raw:
        with zipfile.ZipFile(raw,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
            for k,v in payload.items():
                with z.open(k+'.npy','w',force_zip64=True) as f:np.lib.format.write_array(f,v,allow_pickle=False)
        raw.flush();os.fsync(raw.fileno())
    with np.load(tmp,allow_pickle=False) as z:
        if not np.array_equal(z['px'],px) or not np.array_equal(z['meta'],meta):raise ValueError('CACHE_READER_BITEXACT_FAIL')
    os.replace(tmp,out)
    receipt={'case_id':case,'status':'CANARY_RENDERED_PENDING_GEOMETRY_REVIEW','bytes':out.stat().st_size,'sha256':sha(out),
        'pixel_sha256':hashlib.sha256(px.tobytes()).hexdigest(),'metadata_sha256':hashlib.sha256(meta.tobytes()).hexdigest(),
        'rows':len(px),'metadata_columns':META_COLS,'roster_sha256':sha(ROSTER),'source_pool_sha256':'ec7a85668e3c0ae2caf445be7756f247e9970c6078008065e91dfff88e2527a8',
        'code_sha256':sha(__file__),'orientation_policy':json.loads((HANDOFF/'orientation_v8_3/ORIENTATION_RECEIPT.json').read_text())['orientation_policy'],
        'recipe_sha256':config['semantic_sha256'],'config_sha256':sha(RUN/'BUILD_CONFIG.frozen.json'),
        'orientation_state':'canonical_L_v8_3','decision':decision,'anchor':anchor,'audit':audit,
        'reader_bitexact':True,'private_hrepro_parity':False,'adaptation':'oai-v4-v83-lps-adapt-v1',
        'known_runtime_adaptation':'official MONAI0.6 module.py; author patch unavailable due429',
        'seconds':time.time()-start,'rss_bytes':psutil.Process().memory_info().rss,
        'peak_rss_bytes':__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss,
        'labels_read':False}
    dump(RUN/'cache'/f'{case}.audit.json',receipt)
    print(json.dumps({k:receipt[k] for k in ['case_id','bytes','rows','seconds','peak_rss_bytes','status']}),flush=True)
    return receipt

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('case');a=ap.parse_args();build(a.case)
