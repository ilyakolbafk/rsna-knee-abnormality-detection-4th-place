"""Resumable lossless anatomical adaptation. No training; promotion gate remains closed."""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import json,time,collections,hashlib,tempfile
from pathlib import Path
import numpy as np
from anatomy_job import ROOT,BASE,OwnedJob,write,G
from fm_proof_chain import sha,load,require,verify_canary,POLICY,ORIENTATION
from fm_anatomy_pixels import geometries,source_box_to_view,geometry_receipt,render,normalize_volume
from fm_delta_codec import encode,decode,CODEC
OUT=ROOT/'training_pixel_store'

def fsyncdir(p):
 fd=os.open(p,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def atomic_bytes(p,data):
 fd,n=tempfile.mkstemp(prefix=p.name+'.tmp.',dir=p.parent)
 with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 os.replace(n,p);fsyncdir(p.parent)
def source_unchanged(row):
 p=BASE/'source_rss'/row['stem'];require(sha(p/'receipt.json')==row['receipt_sha256'],'SOURCE_RECEIPT_CHANGED')
 for name,v in row['files'].items():
  st=(p/name).stat();require(all(getattr(st,k)==v['stat'][n] for k,n in [('st_ino','inode'),('st_size','bytes'),('st_mtime_ns','mtime_ns'),('st_ctime_ns','ctime_ns')]),'SOURCE_STAT_CHANGED:'+row['stem']+'/'+name)
 return p
def render_block(p,n,gs,flip):
 raw=np.load(p/'rss.f32.npy',mmap_mode='r');scale,norm=normalize_volume(raw);a=np.empty((3*n,384,384),np.uint8)
 for j,v in enumerate([0,2,3]):
  for z in range(n):a[j*n+z]=render(raw,z,scale,gs[v],flip)
 return a,norm
def commit_chunk(out,row_binding,p,n,gs,flip,before_write=None):
 stem=row_binding['stem'];cp=out/'chunks'/(stem+'.dxz');rp=out/'volume_receipts'/(stem+'.json')
 if rp.exists():
  rec=load(rp);require(rec['row_binding']==row_binding,'CHUNK_RESUME_BINDING_CHANGED');wire=cp.read_bytes();require(hashlib.sha256(wire).hexdigest()==rec['chunk']['compressed_sha256'],'COMMITTED_COMPRESSED_SHA_CHANGED');a=decode(wire,rec['chunk']['shape']);require(hashlib.sha256(a.tobytes()).hexdigest()==rec['chunk']['uncompressed_sha256'],'COMMITTED_DECODED_SHA_CHANGED');return rec
 a,norm=render_block(p,n,gs,flip);wire,digest=encode(a);decoded=decode(wire,a.shape);require(np.array_equal(a,decoded),'LOSSLESS_PIXEL_MISMATCH')
 if before_write is not None:before_write(len(wire))
 atomic_bytes(cp,wire)
 require(sha(cp)==hashlib.sha256(wire).hexdigest(),'CHUNK_RECEIVER_SHA_MISMATCH')
 rec={'row_binding':row_binding,'normalization':norm,'chunk':{'stem':stem,'start':row_binding['start'],'shape':list(a.shape),'path':'chunks/'+cp.name,'compressed_bytes':len(wire),'compressed_sha256':hashlib.sha256(wire).hexdigest(),'uncompressed_sha256':digest},'every_decoded_pixel_compared':True,'created_unix':time.time()};write(rp,rec);return rec

def binding(d,c,N):
 return {'inputs':d['inputs'],'source_manifest_sha256':d['inputs']['source_manifest_sha256'],'adaptation_spec_sha256':d['inputs']['adaptation_spec_sha256'],'anatomy_decisions_sha256':sha(ROOT/'ANATOMY_DECISIONS.json'),'anatomical_canary_sha256':sha(ROOT/'qa/ANATOMICAL_CANARY.json'),'producer_code_sha256':sha(Path(__file__)),'shape':[N,384,384],'dtype':'uint8','codec':CODEC,'orientation_policy':POLICY,'orientation_state':ORIENTATION,'private_pixel_parity':False,'normalization':'volume_p99.9_linear_before_resize_clip01_round_to_even_uint8'}
def build(job):
 require(not (OUT/'CORE_BINDING.json').exists() and not (OUT/'PROMOTION_RECEIPT.json').exists() and not (OUT/'_metadata_freeze_stage/TRANSACTION.json').exists(),'FROZEN_NAMESPACE_USE_CONSUMER_PREFLIGHT_DO_NOT_REBUILD')
 d,c=verify_canary(ROOT,BASE);D=d['volumes'];rows=sorted(load(BASE/'receipts/SOURCE_FULLSHA_MANIFEST.json')['volumes'],key=lambda x:x['stem']);spec=load(BASE/'inputs/FM_ADAPTATION_SPEC.json');classes=spec['class_mapping']['classes'];mapping=spec['class_mapping']['mapping'];N=sum(3*r['n_slices'] for r in rows if D[r['stem']]['accepted']);bind=binding(d,c,N)
 OUT.mkdir(exist_ok=True)
 for n in ['chunks','volume_receipts']: (OUT/n).mkdir(exist_ok=True)
 bp=OUT/'BUILD_BINDING.json'
 if bp.exists():require(load(bp)==bind,'BUILD_RESUME_INPUTS_CHANGED')
 else:write(bp,bind)
 # Row ranges are immutable even across partial-file recovery.
 start=0;index={};rowboxes={};counts=collections.Counter();visible=collections.Counter();norms={};chunks=[];wire_bytes=0
 for i,r in enumerate(rows):
  job.check();s=r['stem'];a=D[s];p=source_unchanged(r);n=r['n_slices'];m=load(p/'metadata.json');bs=load(p/'boxes.source.json');sp=bs['spatial'];negative=not sp and not bs['study_level'];require(not(a['accepted'] and bs['study_level'] and not sp),'ARTIFACT_NEGATIVE')
  ent={'n_slices':n,'views':[],'patient_id':r['patient_id'],'aux_holdout':bool(r['patient_safe_closure_holdout']),'stem_rule_holdout':bool(r['stem_holdout']),'excluded':not a['accepted'],'exclude_reason':a['exclude_reason'],'reviewed_no_findings':negative,'source_split':r['split'],'source_raw_h5_sha256':r['raw_h5_sha256'],'source_receipt_sha256':r['receipt_sha256'],'source_spacing_xy_mm':m['spacing_xy_mm'],'source_interslice_distance_mm':None,'orientation_policy':POLICY,'orientation_state':ORIENTATION,'anatomy_decision':a}
  if not a['accepted']:index[s]=ent;continue
  flip=a['hflip_to_lateral_pluscol'];anchor=a['joint_anchor_canonical_display_row_col'];gs=geometries((320,320),m['spacing_xy_mm'],anchor);vd={0:start,2:start+n,3:start+2*n};ent.update(views=[[v,vd[v]] for v in [0,2,3]],view_names={'0':'fullfield','2':'medial_local100mm','3':'lateral_local100mm'},anatomical_medial_lateral_known=True,view_geometry={str(v):geometry_receipt(g,flip,anchor,v) for v,g in gs.items()})
  for b in sp:
   require(b['label_raw'] in mapping,'UNKNOWN_RAW_CLASS');klass=classes.index(mapping[b['label_raw']]);z=b['slice'];require(0<=z<n,'BOX_SLICE_OOB');counts[b['label_raw']]+=1
   for v,g in gs.items():
    box,info=source_box_to_view(b['source_xywh'],klass,g,flip)
    if box is not None:rowboxes.setdefault(str(vd[v]+z),[]).append(box);visible[(v,klass)]+=1
  require(negative or any(str(vd[0]+z) in rowboxes for z in range(n)),'LOST_FULLFIELD_ANNOTATIONS')
  rb={'stem':s,'start':start,'n_slices':n,'build_binding_sha256':sha(bp),'source_receipt_sha256':r['receipt_sha256']};
  def admit_exact_wire(wire_length):
   remaining_raw=(N-start-3*n)*384*384;job.disk=int(wire_length+remaining_raw*.70+256*1024**2);job.register(True);job.check()
  rec=commit_chunk(OUT,rb,p,n,gs,flip,before_write=admit_exact_wire);chunks.append(rec['chunk']);wire_bytes+=rec['chunk']['compressed_bytes'];norms[s]=rec['normalization'];index[s]=ent;start+=3*n
  # Admission reserves future bytes, not a second copy of already committed bytes.
  remaining=(N-start)*384*384
  job.disk=int(remaining*.70+256*1024**2);job.progress.update(phase='anatomical_lossless_chunks',reviewed_volumes_seen=i+1,expected_sources=1172,valid_rows=start,expected_rows=N,committed_bytes=wire_bytes,remaining_reserved_bytes=job.disk)
  if (i+1)%25==0:job.register(True);print(json.dumps(job.progress),flush=True)
 require(start==N and len(index)==1172,'ROW_MEMBERSHIP')
 header={'format':'fm384_v1','size':384,'classes':classes,'n_rows':N,'training_ready':False,'semantic_gate':'ANATOMICAL_PASS_FINAL_READER_QA_PENDING','adaptation_id':POLICY,'orientation_policy':POLICY,'orientation_state':ORIENTATION,'anatomical_medial_lateral_known':True,'private_pixel_parity':False}
 write(OUT/'pixel_store.json',{'format':'fm384_lossless_volume_chunks_v1','codec':CODEC,'dtype':'uint8','shape':[N,384,384],'chunks':chunks,'is_numpy_mmap':False,'reader':'fm_chunk_array.ChunkArray','build_binding_sha256':sha(bp)})
 write(OUT/'index.json',dict(header,volumes=index));write(OUT/'boxes.json',dict(header,boxes=rowboxes))
 write(OUT/'manifest.json',{'binding':bind,'status':'ANATOMICAL_PIXELS_FINAL_READER_QA_PENDING','reviewed_volumes':1172,'usable_volumes':sum(a['accepted'] for a in D.values()),'excluded_volumes':sum(not a['accepted'] for a in D.values()),'physical_slices_all_sources':sum(r['n_slices'] for r in rows),'physical_slices_usable':N//3,'normalization':norms,'accepted_raw_category_counts':dict(counts),'visible_boxes_per_view_class':{str(k):v for k,v in visible.items()},'no_RGB_crop352_augmentation_ImageNet_or_dense_maps_baked':True,'mapping_is_explicit_adaptation_not_author_private_parity':True,'H5_patient_id_scope':'official anonymized examination group, not proven person identity across exams','no_exam_group_cross_aux_holdout':True,'source_AP_spacing_claimed':False,'training_ready':False,'private_pixel_parity':False})
 core=['BUILD_BINDING.json','pixel_store.json','index.json','boxes.json','manifest.json'];files={n:sha(OUT/n) for n in core}
 receipt={'state':'ANATOMICAL_PIXELS_BUILT_FINAL_QA_PENDING','build_binding_sha256':sha(bp),'files_sha256':files,'n_rows':N,'accepted_volumes':len(chunks),'compressed_pixel_bytes':wire_bytes,'uncompressed_pixel_bytes':N*384*384,'all_chunks_every_pixel_decode_equal_and_receiver_fullsha':True,'training_ready':False,'completed_unix':time.time()};write(OUT/'COMPLETION_RECEIPT.json',receipt);job.disk=128*1024**2;return receipt
if __name__=='__main__':
 # Start reservation is conservative based on four-case measured maximum0.600; actual guard checked each25 volumes.
 d=load(ROOT/'ANATOMY_DECISIONS.json');src=load(BASE/'receipts/SOURCE_FULLSHA_MANIFEST.json');N=sum(3*r['n_slices'] for r in src['volumes'] if d['volumes'][r['stem']]['accepted']);already=sum(p.stat().st_size for p in (OUT/'chunks').glob('*.dxz')) if (OUT/'chunks').exists() else 0
 reserve=max(256*1024**2,int(N*384*384*.70+256*1024**2)-already)
 with OwnedJob('anatomical_lossless_chunks',cpu=3,ram_gib=4,disk_gib=reserve/G) as job:
  result=build(job);job.progress.update(phase='anatomical_chunks_built_final_QA_pending',**{k:result[k] for k in ['n_rows','accepted_volumes','compressed_pixel_bytes']});print(json.dumps(result),flush=True)
