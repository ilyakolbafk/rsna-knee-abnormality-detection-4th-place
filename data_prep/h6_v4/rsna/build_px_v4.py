#!/usr/bin/env python3
"""RSNA-only fullfield v4 cache builder. No labels/training/download of DICOM."""
from support import *
import argparse,csv,tempfile,traceback,io,gc
import numpy as np,pandas as pd,pydicom,psutil
import runtime_core as C
from cache_store import ChunkWriter,CacheReader
from rsna_knee.native_neighbor_bank import read_ordered_headers
from rsna_knee.cache import resolve_series_dir
from rsna_knee.grayscale_cache import select_series_for_cache,sample_window_indices
from rsna_knee.dicom import center_crop_mm,resize_image
from rsna_knee.compartment_locator_v2 import analyse_series
from raw_inputs import norm

class SourceStudy(C.DecodedStudy):
 def __init__(self,uid,rows,root,max_mib):
  self.uid=uid;self.rows=rows;self.images_root=root;self.series={};self.tag_values=[];self.errors=[];self.locator={};self.full=[];self.source_records=[];self.decoded=0
  frame=pd.DataFrame(rows);chosen=select_series_for_cache(frame,root,C.SLOTS);slots={c['series_uid']:i for i,c in enumerate(chosen)if c};seen=set();estimated=0
  pending=[]
  for row in rows:
   sid=row['SeriesInstanceUID'];plane=row['Anatomical_Plane']
   if plane not in ('Sagittal','Coronal','Axial'):
    self.errors.append({'sid':sid,'excluded':'NON_DIAGNOSTIC_PLANE_METADATA'});continue
   entries,qc=read_ordered_headers(resolve_series_dir(root,uid,sid))
   if not entries or not qc['within_series_valid']:raise ValueError('INVALID_SOURCE_GEOMETRY:'+sid)
   estimated+=sum(int(e['header'].Rows)*int(e['header'].Columns)*12 for e in entries)
   pending.append((row,entries,qc))
  reserve=max(2*2**30,int(psutil.virtual_memory().total*.15));available=psutil.virtual_memory().available
  if estimated>max_mib*2**20 or estimated+reserve>available:raise MemoryError(f'RESOURCE_DEFERRED estimated={estimated} available={available}')
  for row,entries,qc in pending:
   sid=row['SeriesInstanceUID'];plane=row['Anatomical_Plane'];frames=[];images=[];loc=[]
   for e in entries:
    p=Path(e['path']);raw=p.read_bytes();ds=pydicom.dcmread(io.BytesIO(raw),force=True)
    if str(ds.StudyInstanceUID)!=uid or str(ds.SeriesInstanceUID)!=sid:raise ValueError('DICOM_UID_CONFLICT')
    sop=str(ds.SOPInstanceUID)
    if sop in seen:raise ValueError('DUPLICATE_SOURCE_SOP')
    seen.add(sop)
    if int(getattr(ds,'NumberOfFrames',1))!=1:raise ValueError('MULTIFRAME_NOT_SUPPORTED_BY_ORIGINAL_PRODUCER')
    payload=C.pixels_from_dataset(ds);im=np.asarray(payload.image,np.float32)
    if im.ndim!=2 or not np.isfinite(im).all():raise ValueError('INVALID_DECODED_PIXELS')
    tags={k:list(getattr(ds,k))for k in ['ImageOrientationPatient','ImagePositionPatient','PixelSpacing']};tags.update(Rows=int(ds.Rows),Columns=int(ds.Columns))
    iop=np.asarray(ds.ImageOrientationPatient,float);ipp=np.asarray(ds.ImagePositionPatient,float);sp=np.asarray(ds.PixelSpacing,float)
    expected_plane=['Sagittal','Coronal','Axial'][int(abs(np.cross(iop[:3],iop[3:])).argmax())]
    if plane!=expected_plane:raise ValueError('METADATA_DICOM_PLANE_CONFLICT')
    if frames and (im.shape!=images[0].shape or not np.allclose(iop,frames[0]['tags']['ImageOrientationPatient'],atol=1e-4,rtol=0)or not np.allclose(sp,frames[0]['tags']['PixelSpacing'],atol=1e-4,rtol=0)):raise ValueError('MIXED_SERIES_GEOMETRY')
    aff=np.stack([iop[:3]*sp[1],iop[3:]*sp[0],ipp],axis=1)
    rec={'path':str(p.resolve()),'stat':stat_id(p),'sha256':hashlib.sha256(raw).hexdigest(),'sop_uid':sop}
    self.source_records.append(rec);images.append(im)
    frames.append({**rec,'tags':tags,'laterality':str(ds.get('Laterality','')),'frame_uid':str(ds.get('FrameOfReferenceUID','')),'description':str(ds.get('SeriesDescription',''))})
    loc.append({'pixels':im,'iop':iop,'ipp':ipp,'spacing':sp,'rows':im.shape[0],'cols':im.shape[1],'desc':str(ds.get('SeriesDescription','')),'frame':str(ds.get('FrameOfReferenceUID',''))})
   vol=np.stack(images)
   for j,entry in enumerate(loc):entry['pixels']=vol[j] # avoid retaining a second full float32 series
   del images
   n=len(vol);sp=[f['tags']['PixelSpacing']for f in frames];slot=slots.get(sid);spec=C.SLOTS[slot]if slot is not None else None
   ids=sample_window_indices(n,spec['slices'],*spec['window'])if spec else np.arange(n)
   side=np.stack([resize_image(center_crop_mm(im,spacing,130.),336)for im,spacing in zip(vol,sp)])
   valid=np.ones(n,bool);q,lo,hi=norm(side,valid,ids)
   self.series[sid]={'images130':q,'valid':valid,'geometry_valid':np.ones(n,bool),'tags':[f['tags']for f in frames],'sops':[f['sop_uid']for f in frames],'qc':qc,'row':row,'slot':slot}
   self.tag_values.extend(f['laterality']for f in frames[:1]);self.locator[sid]=loc
   fs=int(row['Fat_Suppression'])if str(row['Fat_Suppression'])in('0','1')else-1;fluid=int(row['Fluid_Sensitive'])if str(row['Fluid_Sensitive'])in('0','1')else 0
   aff=np.stack([np.stack([np.asarray(f['tags']['ImageOrientationPatient'])[:3]*float(f['tags']['PixelSpacing'][1]),np.asarray(f['tags']['ImageOrientationPatient'])[3:]*float(f['tags']['PixelSpacing'][0]),np.asarray(f['tags']['ImagePositionPatient'])],axis=1)for f in frames])
   if len(vol)>1:
    normal=np.cross(aff[0,:,0],aff[0,:,1]);normal/=np.linalg.norm(normal);pos=aff[:,:,2]@normal
    if np.max(pos)-np.min(pos)<=1e-4:raise ValueError('DEGENERATE_DEPTH')
   self.full.append({'sid':sid,'plane':plane,'family':sid,'frames':frames,'volume':vol,'aff':aff,'fatsat':fs,'fluid':fluid,'source_study_uid':uid})
   self.decoded+=n
  if not self.full:raise ValueError('NO_USABLE_SERIES')
  try:
   _,_,_,r=analyse_series(uid,self.locator,frame.set_index('SeriesInstanceUID'),None);self.joint_z=r.get('joint_z');self.locator_audit={k:r.get(k)for k in ['joint_z','accept','sag_cor_gap','sag_series','cor_series']}
  except Exception as e:self.joint_z=None;self.locator_audit={'error':type(e).__name__,'joint_z':None}
  self.locator.clear()

def render_sample(series,anchor,route,writer):
 blocks=[];lineage=[]
 for real,s in enumerate(series):
  vol=s['volume'];aff=s['aff'];n=len(vol)
  axis=aff[-1,:,2]-aff[0,:,2]if n>1 else np.cross(aff[0,:,0],aff[0,:,1]);axis=axis/max(np.linalg.norm(axis),1e-9)
  pos=aff[:,:,2]@axis;spacing=float(np.median(np.diff(pos)))if n>1 else 0.
  chunks=C.slab_chunks(n,spacing);slabpos=np.asarray([pos[c].mean()for c in chunks]);spacing=float(np.median(np.diff(slabpos)))if len(chunks)>1 else 0.
  block={'series':s['sid'],'plane':s['plane'],'n_slices':len(chunks),'views':[],'view_geometry':{},'slabs':[{'source_indices':c.tolist(),'source_sops':[s['frames'][i]['sop_uid']for i in c],'weights':[1/len(c)]*len(c),'affine_lps':np.mean(aff[c],axis=0).tolist()}for c in chunks]}
  used={int(i)for c in chunks for i in c};block['dropped_tail_source_sops']=[s['frames'][i]['sop_uid']for i in range(n)if i not in used]
  for vid,crop,center,point in C.views(s,anchor,route):
   block['views'].append([vid,writer.rows]);geoms=[]
   for k,c in enumerate(chunks):
    slab=np.rint(vol[c].astype(np.float32).mean(0)).astype(vol.dtype);sp=(float(np.linalg.norm(aff[c[0],:,1])),float(np.linalg.norm(aff[c[0],:,0])))
    vg=C.view_geometry(slab.shape,sp,384,crop_mm=crop,center=center);x,y,x1,y1=vg.crop_xyxy;ox,oy,_,_=vg.valid_xyxy;oh,ow=vg.resized_shape
    resized=C.cv2.resize(slab[y:y1,x:x1].astype(np.float32),(ow,oh),interpolation=C.cv2.INTER_AREA)
    px=np.zeros((384,384),np.uint8);px[oy:oy+oh,ox:ox+ow]=np.rint(np.clip(C.normalize_intensity(resized,s['window'],'series_companded'),0,1)*255).astype(np.uint8)
    rel=2*(slabpos[k]-slabpos[0])/max(slabpos[-1]-slabpos[0],1e-6)-1 if len(chunks)>1 else 0.
    slot=C.SOURCE_SLOTS[(s['plane'],s['fatsat'])]
    writer.add(px,[real,slot,{'Sagittal':0,'Coronal':1,'Axial':2}[s['plane']],vid,rel,np.log1p(abs(spacing)),vg.extent/130,s['fluid'],s['fatsat'],k,len(chunks),real])
    if k==0:
     receipt=vg.receipt();block['view_geometry'][str(vid)]={'output_to_native':np.asarray(receipt['output_to_native']).tolist(),'fov_mm':np.asarray(receipt['source_crop_fov_mm']).tolist(),'valid_xyxy':list(vg.valid_xyxy),'crop_xyxy':list(vg.crop_xyxy),'anchor_lps':point}
   writer.flush()
  blocks.append(block);lineage.append({'series_uid':s['sid'],'plane':s['plane'],'orientation_applications':1,'orientation_audit':s['orientation_audit'],'pull_stored_to_original':s['pull_stored_to_original'],'native_encoding':s['native_encoding'],'window_uint16':s['window'],'frames':s['frames']})
 return blocks,lineage

def roots_for(parent,roots):
 for root in roots:
  for base in [root,root/'train_images',root/'test_images',root/'test_series',root/'train_series',root/'images']:
   if(base/parent).is_dir():return base
 raise FileNotFoundError('MISSING_STUDY:'+parent)

def expected_samples():
 e=json.loads((PACKAGE/'EXPECTED_RSNA_SAMPLES.json').read_text());m={r['SampleID']:r for r in csv.DictReader((PACKAGE/'RSNA_SAMPLE_MAPPING.csv').open())};out={}
 for sample in e['fit_sample_ids']+e['diagnostic_Gold58_sample_ids']:
  row=m.get(sample,{'ParentStudyInstanceUID':sample,'SampleID':sample,'report_group':sample,'side':None})
  out[sample]={'sample_id':sample,'parent':row['ParentStudyInstanceUID'],'report_group':row['report_group'],'diagnostic_Gold58':sample in e['diagnostic_Gold58_sample_ids'],'expected_side_for_scope_check_only':row['side']}
 return e,out

def source_unchanged(folder,binding):
 p=folder/'index.json'
 if not p.exists():return False
 index=json.loads(p.read_text())
 if index['binding']!=binding:raise ValueError('EXISTING_CACHE_RECIPE_DIFFERS')
 qa=folder/'QA.json'
 if not qa.exists()or json.loads(qa.read_text()).get('verified_index_sha256')!=sha(p):return False
 for directory,names in index.get('source_directory_inventory',{}).items():
  if sorted(f.name for f in Path(directory).glob('*.dcm'))!=names:return False
 for r in index['source_files']:
  if not Path(r['path']).is_file()or stat_id(r['path'])!=r['stat']:return False
 for c in index['chunks']:
  if not(folder/c['path']).is_file()or stat_id(folder/c['path'])!=c['stat']:return False
 return True

def build_parent(parent,samples,rows,roots,out,binding,max_mib):
 folders={s['sample_id']:out/'samples'/key(s['sample_id'])for s in samples}
 if all(source_unchanged(folders[s['sample_id']],binding)for s in samples):return {'parent':parent,'resumed_verified':True}
 root=roots_for(parent,roots);study=SourceStudy(parent,rows,root,max_mib);model,resolver=C.resources();decision=study.side_inputs(resolver)
 routes=list(decision['routes']);required={s['expected_side_for_scope_check_only']for s in samples if s['expected_side_for_scope_check_only']}
 if len(samples)==2 and(set(routes)!={'L','R'}or required!={'L','R'}):raise ValueError('BILATERAL_SCOPE_MISMATCH')
 if len(samples)==1 and len(routes)!=1:raise ValueError('UNEXPECTED_BILATERAL_SCOPE')
 results=[]
 for sample in samples:
  route=sample['expected_side_for_scope_check_only']if len(samples)==2 else routes[0]
  # Frozen mapping checks routing only; it never determines a physical transform.
  if required and route not in required:raise ValueError('RESOLVED_SCOPE_SIDE_DIFFERS')
  selected=[s for s in study.full if not decision['bilateral']or next(z for z in decision['series']if z['sid']==s['sid'])['assigned_side']==route]
  selected=[C.orient(s,decision,route)for s in selected];geometry={};native=[]
  with tempfile.TemporaryDirectory(prefix='rsna_native_',dir=out/'scratch')as work:
   for s in selected:
    p=Path(work)/(key(s['sid'])+'.npy');np.save(p,s['volume']);native.append({'path':str(p),'SeriesInstanceUID':s['sid'],'native_shape':list(s['volume'].shape),'plane':s['plane'],'fluid_sensitive':s['fluid'],'spacing_mm':[[float(np.linalg.norm(a[:,1])),float(np.linalg.norm(a[:,0]))]for a in s['aff']]});quality=C.alignment_quality(s['aff'])
    if not quality['aligned']:raise ValueError('SOURCE_ALIGNMENT_QUALITY_FAILED:'+s['sid'])
    geometry[s['sid']]={'affines_col_row_origin_lps_mm':s['aff'].tolist(),'quality':quality}
   anchor=C.MemoryAnchorBank(model,geometry).record(parent,{'series':native})
   if anchor['joint_lps']is None:raise ValueError('NO_SOURCE_ANATOMY_OR_SLAB_PRIOR_ANCHOR')
   folder=folders[sample['sample_id']];writer=ChunkWriter(folder,binding);blocks,lineage=render_sample(selected,anchor,route,writer)
   index=writer.finish({'sample':sample,'resolved_route':route,'decision':decision,'anchor':anchor,'blocks':blocks,'lineage':lineage,'source_files':study.source_records,'source_directory_inventory':{str(d):sorted(f.name for f in d.glob('*.dcm'))for d in {Path(z['path']).parent for z in study.source_records}},'geometry_quality':geometry,'frame_of_reference_audit':{'source_UIDs':sorted({f['frame_uid']for s in selected for f in s['frames']}),'cross_series_registration_proven':False,'method':'original producer projects native scanner LPS; no inferred registration'},'excluded_series':study.errors,'metadata_columns':C.META_COLS,'private_hrepro_parity':False,'adaptations':['explicitRSNAorchestration','v8.3orientationwithLPSoffsets','boundedlosslesschunks','MONAI060officialmodulefallback']})
   qa=CacheReader(folder).verify();qa['verified_index_sha256']=sha(folder/'index.json');dump(folder/'QA.json',qa);results.append({'sample_id':sample['sample_id'],'rows':index['rows'],'chunks':len(index['chunks']),'anchor_method':anchor.get('method'),'QA':qa['status']})
 return {'parent':parent,'samples':results}

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--dicom-root',action='append',required=True);ap.add_argument('--out',required=True);ap.add_argument('--samples',default='test3',help='test3 | all | comma-separated expected SampleIDs');ap.add_argument('--max-study-mib',type=int,default=8192);ap.add_argument('--resource-backoff-seconds',type=float,default=30);ap.add_argument('--max-resource-wait-seconds',type=float,default=1800);args=ap.parse_args()
 package_sha=verify_package();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True);(out/'scratch').mkdir(exist_ok=True)
 with lock(out/'builder.lock'):
  e,samples=expected_samples();selected=e['public_test3']if args.samples=='test3'else list(samples)if args.samples=='all'else args.samples.split(',')
  if not set(selected)<=set(samples):raise ValueError('SAMPLE_OUTSIDE_FROZEN_SCOPE')
  metadata={}
  for p in sorted((PACKAGE/'metadata').glob('*.csv')):
   for row in csv.DictReader(p.open()):
    k=(row['StudyInstanceUID'],row['SeriesInstanceUID'])
    if k in metadata and metadata[k]!=row:raise ValueError('METADATA_CONFLICT')
    metadata[k]=row
  runtime_files=[p for p in PACKAGE.rglob('*') if p.is_file() and ('vendor'in p.relative_to(PACKAGE).parts or 'orientation_v8_3'in p.relative_to(PACKAGE).parts or 'weights'in p.relative_to(PACKAGE).parts or p.name in ('support.py','runtime_core.py','cache_store.py','build_px_v4.py','knee_anchor_bank_v2.py','compact_train.py','RSNA_SAMPLE_MAPPING.csv','EXPECTED_RSNA_SAMPLES.json')) and '__pycache__'not in p.parts]
  binding={'recipe':RECIPE,'runtime_components_sha256':digest({str(p.relative_to(PACKAGE)):sha(p)for p in sorted(runtime_files)}),'metadata_sha256':digest(sorted((list(k),v)for k,v in metadata.items())),'expected_sha256':sha(PACKAGE/'EXPECTED_RSNA_SAMPLES.json')}
  # Tuple keys in the metadata dictionary are represented explicitly for hashing.
  binding['metadata_sha256']=digest(sorted((list(k),v)for k,v in metadata.items()))
  cfg=out/'CONFIG.frozen.json'
  if cfg.exists()and json.loads(cfg.read_text())!=binding:raise ValueError('OUTPUT_CONFIG_DIFFERS')
  dump(cfg,binding);groups={}
  for sample in selected:groups.setdefault(samples[sample]['parent'],[]).append(samples[sample])
  done=[];errors=[]
  for parent,group in groups.items():
   try:
    rows=[r for(uv,sid),r in metadata.items()if uv==parent]
    if not rows:raise ValueError('MISSING_OFFICIAL_SERIES_METADATA')
    wait_started=time.time()
    while True:
     try:
      result=build_parent(parent,group,rows,[Path(x).resolve()for x in args.dicom_root],out,binding,args.max_study_mib);break
     except MemoryError as deferred:
      if 'RESOURCE_DEFERRED'not in str(deferred)or time.time()-wait_started>args.max_resource_wait_seconds:raise
      dump(out/'STATE.json',{'state':'RESOURCE_DEFERRED','parent':parent,'reason':str(deferred),'pid':os.getpid(),'at_unix':time.time()})
      gc.collect();time.sleep(args.resource_backoff_seconds)
    done.extend(s['sample_id']for s in group);print(json.dumps(result,ensure_ascii=False),flush=True)
   except Exception as error:
    rec={'parent':parent,'error_type':type(error).__name__,'error':str(error)};errors.append(rec);dump(out/'errors'/(key(parent)+'.json'),rec);print(json.dumps(rec),flush=True)
   dump(out/'STATE.json',{'state':'BUILDING','pid':os.getpid(),'at_unix':time.time(),'requested':len(selected),'built_or_resumed':len(done),'errors':errors,'last_parent':parent});gc.collect()
  coverage={'status':'PASS_REQUESTED_SCOPE'if not errors and len(done)==len(selected)else'INCOMPLETE','full_frozen_scope_complete':len(done)==len(samples)and not errors,'expected_full':len(samples),'requested':len(selected),'built_or_resumed':done,'missing_requested':sorted(set(selected)-set(done)),'errors':errors,'binding':binding}
  dump(out/'COVERAGE.json',coverage)
  if errors:raise SystemExit(2)
if __name__=='__main__':main()
