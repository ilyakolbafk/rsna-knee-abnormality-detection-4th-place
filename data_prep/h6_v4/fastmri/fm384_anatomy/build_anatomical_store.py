"""Atomic, source-bound anatomical 3-view adaptation. Consumer gate remains closed until QA."""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import json,hashlib,time,collections
from pathlib import Path
import numpy as np
from anatomy_job import ROOT,BASE,OwnedJob,write
from fm_anatomy_pixels import geometries,source_box_to_view,geometry_receipt,render,normalize_volume
OUT=ROOT/'training_pixel_store'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def arrsha(a):
 h=hashlib.sha256()
 for k in range(0,len(a),16):h.update(np.ascontiguousarray(a[k:k+16]).tobytes())
 return h.hexdigest()
def same_stat(p,s):
 v=Path(p).stat();return all(getattr(v,k)==s[n] for k,n in [('st_ino','inode'),('st_size','bytes'),('st_mtime_ns','mtime_ns'),('st_ctime_ns','ctime_ns')])
def fsyncdir(p):
 fd=os.open(p,os.O_RDONLY);os.fsync(fd);os.close(fd)
def commit_pixels(px,work,rp,row_binding,source_pixels,gs,flip):
 start=row_binding['start'];n=row_binding['n_slices'];vd={0:start,2:start+n,3:start+2*n}
 if rp.exists():
  rec=json.loads(rp.read_text());assert rec['row_binding']==row_binding,'ROW_BINDING_CHANGED';assert arrsha(px[start:start+3*n])==rec['row_sha256'],'ROW_RESUME_SHA_CHANGED';return rec['normalization']
 raw=np.load(source_pixels,mmap_mode='r');scale,norm=normalize_volume(raw)
 for z in range(n):
  for v,g in gs.items():px[vd[v]+z]=render(raw,z,scale,g,flip)
 px.flush()
 with work.open('rb') as f:os.fsync(f.fileno())
 write(rp,{'row_binding':row_binding,'row_sha256':arrsha(px[start:start+3*n]),'normalization':norm,'created_unix':time.time()});return norm
def build(job):
 decisions=json.loads((ROOT/'ANATOMY_DECISIONS.json').read_text());assert decisions['gate_state']=='PASS' and decisions['training_started'] is False
 assert json.loads((ROOT/'qa/ANATOMICAL_CANARY.json').read_text())['state']=='PASS'
 src=BASE/'receipts/SOURCE_FULLSHA_MANIFEST.json';source=json.loads(src.read_text());entries=sorted(source['volumes'],key=lambda r:r['stem']);assert len(entries)==1172 and set(decisions['volumes'])=={r['stem'] for r in entries}
 spec=json.loads((BASE/'inputs/FM_ADAPTATION_SPEC.json').read_text());classes=spec['class_mapping']['classes'];mapping=spec['class_mapping']['mapping'];D=decisions['volumes'];N=sum(3*r['n_slices'] for r in entries if D[r['stem']]['accepted']);assert sum(d['accepted'] for d in D.values())>=1000
 OUT.mkdir(exist_ok=True);(OUT/'volume_receipts').mkdir(exist_ok=True)
 bind={'source_manifest_sha256':sha(src),'adaptation_spec_sha256':sha(BASE/'inputs/FM_ADAPTATION_SPEC.json'),'anatomy_decisions_sha256':sha(ROOT/'ANATOMY_DECISIONS.json'),'anatomical_canary_sha256':sha(ROOT/'qa/ANATOMICAL_CANARY.json'),'code_sha256':{n:sha(ROOT/n) for n in ['build_anatomical_store.py','fm_anatomy_pixels.py','fm_geometry.py','fm_pixels.py']},'shape':[N,384,384],'dtype':'uint8','orientation_policy':'fm-rss-anatomy-v1','orientation_state':'relative_lateral_plus_column_head_up','private_pixel_parity':False,'normalization':'volume_p99.9_linear_before_resize_clip01_round_to_even_uint8'};bp=OUT/'BUILD_BINDING.json'
 if bp.exists():assert json.loads(bp.read_text())==bind,'STORE_RESUME_BINDING_CHANGED'
 else:write(bp,bind)
 if (OUT/'COMPLETION_RECEIPT.json').exists():
  c=json.loads((OUT/'COMPLETION_RECEIPT.json').read_text());assert c['binding']==bind
  for n,f in c['files'].items():assert sha(OUT/n)==f['sha256'],'FINAL_FULLSHA_CHANGED'
  return c
 final=OUT/'px.u8.npy';partial=OUT/'px.u8.npy.partial';work=final if final.exists() else partial
 if final.exists():assert not partial.exists() and all((OUT/'volume_receipts'/(r['stem']+'.json')).exists() for r in entries if D[r['stem']]['accepted']),'FINAL_WITH_MISSING_ROW_RECEIPTS'
 px=np.lib.format.open_memmap(work,mode='r+' if work.exists() else 'w+',dtype=np.uint8,shape=(N,384,384));start=0;index={};rowboxes={};counts=collections.Counter();visible=collections.Counter();norms={}
 for i,r in enumerate(entries):
  job.check();s=r['stem'];d=D[s];n=r['n_slices'];p=BASE/'source_rss'/s;assert sha(p/'receipt.json')==r['receipt_sha256']
  for name,v in r['files'].items():assert same_stat(p/name,v['stat']),'SOURCE_FILE_CHANGED'
  m=json.loads((p/'metadata.json').read_text());bs=json.loads((p/'boxes.source.json').read_text());sp=bs['spatial'];artifact=bool(bs['study_level']) and not sp;neg=not sp and not bs['study_level'];assert not (d['accepted'] and artifact),'ARTIFACT_ONLY_NEVER_NEGATIVE'
  ent={'n_slices':n,'views':[],'patient_id':r['patient_id'],'aux_holdout':bool(r['patient_safe_closure_holdout']),'stem_rule_holdout':bool(r['stem_holdout']),'excluded':not d['accepted'],'exclude_reason':d.get('exclude_reason'),'reviewed_no_findings':neg,'source_split':r['split'],'source_raw_h5_sha256':r['raw_h5_sha256'],'source_receipt_sha256':r['receipt_sha256'],'source_spacing_xy_mm':m['spacing_xy_mm'],'source_interslice_distance_mm':None,'orientation_policy':bind['orientation_policy'],'orientation_state':bind['orientation_state'],'anatomy_decision':d}
  if not d['accepted']:index[s]=ent;continue
  flip=bool(d['hflip_to_lateral_pluscol']);anchor=d['joint_anchor_canonical_display_row_col'];gs=geometries((320,320),m['spacing_xy_mm'],anchor);vd={0:start,2:start+n,3:start+2*n};ent.update(views=[[v,vd[v]] for v in [0,2,3]],view_names={'0':'fullfield','2':'medial_local100mm','3':'lateral_local100mm'},anatomical_medial_lateral_known=True,view_geometry={str(v):geometry_receipt(g,flip,anchor,v) for v,g in gs.items()})
  for b in sp:
   assert b['label_raw'] in mapping;klass=classes.index(mapping[b['label_raw']]);z=b['slice'];assert 0<=z<n;counts[b['label_raw']]+=1
   for v,g in gs.items():
    box,info=source_box_to_view(b['source_xywh'],klass,g,flip)
    if box is not None:rowboxes.setdefault(str(vd[v]+z),[]).append(box);visible[(v,klass)]+=1
  assert neg or any(str(vd[0]+z) in rowboxes for z in range(n)),'BOXED_VOLUME_LOST_FULLFIELD_ANNOTATIONS'
  rp=OUT/'volume_receipts'/(s+'.json');rb={'stem':s,'start':start,'n_slices':n,'build_binding_sha256':sha(bp),'source_receipt_sha256':r['receipt_sha256']}
  norm=commit_pixels(px,work,rp,rb,p/'rss.f32.npy',gs,flip)
  norms[s]=norm;index[s]=ent;start+=3*n;job.progress.update(phase='anatomical_store_render',source_volumes_seen=i+1,reviewed_expected=1172,valid_rows=start,expected_rows=N)
  if (i+1)%25==0:print(json.dumps(job.progress),flush=True)
 assert start==N and len(index)==1172;px.flush();del px
 with work.open('rb') as f:os.fsync(f.fileno())
 if work==partial:os.rename(partial,final);fsyncdir(OUT)
 head={'format':'fm384_v1','size':384,'classes':classes,'n_rows':N,'training_ready':False,'semantic_gate':'QA_PENDING','adaptation_id':'fm-rss-anatomy-v1','orientation_policy':bind['orientation_policy'],'orientation_state':bind['orientation_state'],'anatomical_medial_lateral_known':True,'private_pixel_parity':False}
 write(OUT/'index.json',dict(head,volumes=index));write(OUT/'boxes.json',dict(head,boxes=rowboxes));write(OUT/'manifest.json',{'binding':bind,'status':'ANATOMICAL_ADAPTATION_PENDING_FINAL_QA','reviewed_volumes':1172,'usable_volumes':sum(d['accepted'] for d in D.values()),'physical_slices_all_sources':sum(r['n_slices'] for r in entries),'physical_slices_usable':N//3,'normalization':norms,'accepted_raw_category_counts':dict(counts),'visible_boxes_per_view_class':{str(k):v for k,v in visible.items()},'no_RGB_crop352_augmentation_ImageNet_or_dense_maps_baked':True,'mapping_is_explicit_adaptation_not_author_private_parity':True,'H5_patient_id_scope':'official anonymized examination group, not proven person identity across exams','no_exam_group_cross_aux_holdout':True,'source_AP_spacing_claimed':False,'training_ready':False,'private_pixel_parity':False})
 files={}
 for n in ['px.u8.npy','index.json','boxes.json','manifest.json']:
  p=OUT/n;s=p.stat();digest=sha(p);assert (s.st_size,s.st_mtime_ns,s.st_ctime_ns)==(p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ctime_ns)
  files[n]={'sha256':digest,'stat':{'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}}
 result={'state':'ANATOMICAL_PIXELS_BUILT_FINAL_QA_PENDING','binding':bind,'files':files,'n_rows':N,'actual_store_bytes':sum(v['stat']['bytes'] for v in files.values()),'uncompressed_uint8_payload_bytes':N*384*384,'compression':'none,mmap NPY','training_ready':False,'completed_unix':time.time()};write(OUT/'COMPLETION_RECEIPT.json',result);return result
if __name__=='__main__':
 with OwnedJob('anatomical_store_render',cpu=4,ram_gib=4,disk_gib=19) as job:build(job);job.progress['phase']='anatomical_store_built_QA_pending'
