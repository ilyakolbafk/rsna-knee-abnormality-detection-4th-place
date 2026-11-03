"""Three-field FM adaptation candidate. Anatomy M/L gate remains explicitly closed."""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import collections,hashlib,json,math,time,sys,fcntl
from pathlib import Path
import numpy as np
from fm_pixels import geometries,geometry_receipt,transform_box,normalize_volume,display_slice,render_display_slice
ROOT=Path('/workspace/knee_v4_cache_20261021/fastmri_v1');OUT=ROOT/'candidate_pixel_store'
EXPECTED_VOLUMES=1172

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def arr_sha(a):
 h=hashlib.sha256()
 for i in range(0,len(a),16):h.update(np.ascontiguousarray(a[i:i+16]).tobytes())
 return h.hexdigest()
def syncdir(p):
 fd=os.open(p,os.O_RDONLY);os.fsync(fd);os.close(fd)
def write(p,d):
 import tempfile
 fd,name=tempfile.mkstemp(prefix=p.name+'.tmp.',dir=p.parent)
 with os.fdopen(fd,'w') as f:json.dump(d,f,indent=2,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(name,p);syncdir(p.parent)
def source_stat_matches(p,v):
 s=p.stat();return all(getattr(s,k)==v[n] for k,n in [('st_ino','inode'),('st_size','bytes'),('st_mtime_ns','mtime_ns'),('st_ctime_ns','ctime_ns')])
def build(progress=None):
 if progress is None:raise RuntimeError('USE_FINISH_PIPELINE_WITH_SINGLETON_AND_SHARED_ADMISSION')
 with (ROOT/'candidate_writer.lock').open('a') as writer_lock:
  fcntl.flock(writer_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  return _build_locked(progress)
def _build_locked(progress):
 if progress is None:raise RuntimeError('USE_FINISH_PIPELINE_WITH_SINGLETON_AND_SHARED_ADMISSION')
 admission=json.loads((ROOT.parent/'shared/admission.json').read_text())['tasks']['fastmri_v1']
 assert admission['PID']==os.getpid() and admission.get('active') and admission.get('convert_workers')==1,'CANDIDATE_REQUIRES_OWN_ADMISSION'
 full=ROOT/'receipts/SOURCE_FULLSHA_MANIFEST.json';src=json.loads(full.read_text());assert src['report']['state']=='PASS_SOURCE_STAGE_ONLY' and len(src['volumes'])==EXPECTED_VOLUMES
 spec=json.loads((ROOT/'inputs/FM_ADAPTATION_SPEC.json').read_text());classes=spec['class_mapping']['classes'];mapping=spec['class_mapping']['mapping'];assert len(classes)==11
 inputs=sorted(src['volumes'],key=lambda x:x['stem']);N=sum(x['n_slices']*3 for x in inputs);OUT.mkdir(exist_ok=True);(OUT/'volume_receipts').mkdir(exist_ok=True)
 bind={'source_manifest_sha256':sha(full),'adaptation_spec_sha256':sha(ROOT/'inputs/FM_ADAPTATION_SPEC.json'),'code_sha256':{n:sha(ROOT/'code'/n) for n in ['build_candidate.py','fm_geometry.py','fm_pixels.py']},'shape':[N,384,384],'dtype':'uint8','candidate_semantics':'screen_left/right100mm; no anatomical M/L proof','training_ready':False,'versions':{'numpy':np.__version__,'opencv':__import__('cv2').__version__}}
 bp=OUT/'BUILD_BINDING.json'
 if bp.exists():assert json.loads(bp.read_text())==bind,'CANDIDATE_RESUME_BINDING_MISMATCH'
 else:write(bp,bind)
 if (OUT/'COMPLETION_RECEIPT.json').exists():
  result=json.loads((OUT/'COMPLETION_RECEIPT.json').read_text());assert result['binding']==bind
  for n,v in result['files'].items():assert source_stat_matches(OUT/n,v['stat']) and sha(OUT/n)==v['sha256']
  return result
 temp=OUT/'px.u8.npy.partial';final=OUT/'px.u8.npy'
 if final.exists():assert not temp.exists() and all((OUT/'volume_receipts'/(r['stem']+'.json')).exists() for r in inputs),'UNRECEIPTED_FINAL_PIXELS_WITH_INCOMPLETE_ROWS'
 working=final if final.exists() else temp
 px=np.lib.format.open_memmap(working,mode='r+' if working.exists() else 'w+',dtype=np.uint8,shape=(N,384,384));assert px.shape==(N,384,384) and px.dtype==np.uint8
 start=0;index={};rowboxes={};catcounts=collections.Counter();visible=collections.Counter();norms={};outside110=collections.Counter()
 for i,r in enumerate(inputs):
  stem=r['stem'];n=r['n_slices'];p=ROOT/'source_rss'/stem
  if progress and progress.get('abort_reason'):raise RuntimeError('STOP_AFTER_RESOURCE_OR_HEARTBEAT_FAILURE')
  assert sha(p/'receipt.json')==r['receipt_sha256'],'SOURCE_RECEIPT_CHANGED_AFTER_VERIFICATION'
  for name,v in r['files'].items():assert source_stat_matches(p/name,v['stat']),'SOURCE_CHANGED_AFTER_VERIFICATION'
  meta=json.loads((p/'metadata.json').read_text());boxes=json.loads((p/'boxes.source.json').read_text());assert meta['patient_id']==r['patient_id'] and meta['rss_shape']==[n,320,320]
  spatial=boxes['spatial'];artifact_only=bool(boxes['study_level']) and not spatial;no_findings=not spatial and not boxes['study_level']
  gs=geometries((320,320),meta['spacing_xy_mm']);vd={0:start,2:start+n,3:start+2*n};assert list(gs)==[0,2,3]
  transformed={};counts=collections.Counter();recovered=collections.Counter()
  dx,dy=meta['spacing_xy_mm'];oldcw=round(110/dx);oldch=round(110/dy);oldx=(320-oldcw)//2;oldy=(320-oldch)//2
  for b in spatial:
   raw=b['label_raw'];assert raw in mapping,'UNKNOWN_FINDING_PRESERVE_AND_STOP';c=classes.index(mapping[raw]);z=b['slice'];assert 0<=z<n
   counts[raw]+=1;x,y,w,h=b['source_xywh'];old_empty=(min(x+w,oldx+oldcw)<=max(x,oldx) or min(y+h,oldy+oldch)<=max(y,oldy))
   for view,g in gs.items():
    v,info=transform_box(b['source_xywh'],c,g)
    if v is not None:
     key=str(vd[view]+z);transformed.setdefault(key,[]).append(v);visible[(view,c)]+=1
     if old_empty:recovered[view]+=1
  assert no_findings or artifact_only or any(str(vd[0]+z) in transformed for z in range(n)),'SPATIAL_VOLUME_WITH_NO_FULLFIELD_BOX_NOT_NEGATIVE'
  rp=OUT/'volume_receipts'/(stem+'.json');rowbind={'stem':stem,'start':start,'n_slices':n,'source_receipt_sha256':r['receipt_sha256'],'build_binding_sha256':sha(bp)}
  if rp.exists():
   vr=json.loads(rp.read_text());assert vr['row_binding']==rowbind,'ROW_RESUME_BINDING_MISMATCH';assert arr_sha(px[start:start+3*n])==vr['row_sha256'],'ROW_RESUME_SHA_MISMATCH';norm=vr['normalization']
  else:
   raw=np.load(p/'rss.f32.npy',mmap_mode='r');scale,norm=normalize_volume(raw)
   for z in range(n):
    display=display_slice(raw,z,scale)
    for view,g in gs.items():px[vd[view]+z]=render_display_slice(display,g)
   px.flush()
   with working.open('rb') as f:os.fsync(f.fileno())
   vr={'row_binding':rowbind,'row_sha256':arr_sha(px[start:start+3*n]),'normalization':norm,'created_unix':time.time()};write(rp,vr)
  catcounts.update(counts);outside110.update(recovered);rowboxes.update(transformed);norms[stem]=norm
  index[stem]={'n_slices':n,'views':[[v,vd[v]] for v in [0,2,3]],'patient_id':r['patient_id'],'aux_holdout':bool(r['patient_safe_closure_holdout']),'stem_rule_holdout':bool(r['stem_holdout']),'excluded':artifact_only,'exclude_reason':'STUDY_LEVEL_ARTIFACT_ONLY_NOT_SPATIAL_NEGATIVE' if artifact_only else None,'reviewed_no_findings':no_findings,'source_split':r['split'],'source_raw_h5_sha256':r['raw_h5_sha256'],'source_receipt_sha256':r['receipt_sha256'],'source_spacing_xy_mm':meta['spacing_xy_mm'],'view_geometry':{str(v):geometry_receipt(g) for v,g in gs.items()},'view_names':{'0':'fullfield','2':'screen_left100mm','3':'screen_right100mm'},'anatomical_medial_lateral_known':False}
  start+=3*n
  if progress is not None:progress.update(phase='building_three_field_adaptation_candidate',built_volumes=i+1,expected_volumes=EXPECTED_VOLUMES,valid_rows=start,expected_rows=N,ready=False)
  if (i+1)%25==0:print(json.dumps({'candidate_volumes':i+1,'valid_rows':start,'expected_rows':N}),flush=True)
 assert start==N and len(index)==EXPECTED_VOLUMES
 px.flush();del px
 with working.open('rb') as f:os.fsync(f.fileno())
 if working==temp:os.rename(temp,final);syncdir(OUT)
 head={'format':'fm384_v1','size':384,'classes':classes,'n_rows':N,'training_ready':False,'adaptation_id':spec['adaptation_id'],'private_pixel_parity':False,'anatomical_medial_lateral_known':False,'semantic_gate':'BLOCKED_ANATOMICAL_ML_UNVERIFIED'}
 write(OUT/'index.json',dict(head,volumes=index));write(OUT/'boxes.json',dict(head,boxes=rowboxes))
 manifest={'binding':bind,'format':'fm384_v1','status':'GEOMETRIC_ADAPTATION_CANDIDATE_ONLY','n_rows':N,'physical_slices':N//3,'reviewed_volumes':EXPECTED_VOLUMES,'usable_volumes':sum(not x['excluded'] for x in index.values()),'spatial_volumes':sum(not x['excluded'] and not x['reviewed_no_findings'] for x in index.values()),'reviewed_no_findings_volumes':sum(x['reviewed_no_findings'] for x in index.values()),'excluded_study_level_only':sum(x['excluded'] for x in index.values()),'all_source_category_counts':dict(catcounts),'mapped_visible_boxes_per_view_class':{str(k):v for k,v in visible.items()},'boxes_recovered_that_were_completely_outside_old110':dict(outside110),'normalization':norms,'view_names':{'0':'fullfield','2':'screen_left100mm','3':'screen_right100mm'},'no_RGB_or_crop352_or_augmentation_or_ImageNet_or_dense_baked':True,'private_pixel_parity':False,'training_ready':False,'remaining_blocker':'Standalone coronal RSS has no proven anatomical M/L identity or matching-private producer. View2/3 are screen positions from recovered author code.','source_manifest_sha256':sha(full)}
 write(OUT/'manifest.json',manifest)
 files={}
 for name in ['px.u8.npy','index.json','boxes.json','manifest.json']:
  p=OUT/name;s=p.stat();digest=sha(p);t=p.stat();assert (s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==(t.st_ino,t.st_size,t.st_mtime_ns,t.st_ctime_ns)
  files[name]={'sha256':digest,'stat':{'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}}
 receipt={'state':'PIXEL_CANDIDATE_BUILT_FULLSHA_NOT_ANATOMICAL_ML_READY','binding':bind,'files':files,'n_rows':N,'physical_slices':N//3,'reviewed_volumes':EXPECTED_VOLUMES,'actual_store_bytes':sum(x['stat']['bytes'] for x in files.values()),'uncompressed_uint8_payload_bytes':N*384*384,'compression':'none, mmap-readable NPY','ready':False,'private_pixel_parity':False,'completed_unix':time.time()}
 write(OUT/'COMPLETION_RECEIPT.json',receipt);return receipt
if __name__=='__main__':raise SystemExit('Use finish_pipeline.py; standalone producer launch is refused')
