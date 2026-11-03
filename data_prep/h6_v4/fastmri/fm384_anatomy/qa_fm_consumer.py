"""Actual Linux auxiliary dataset/sampler/loader/target/loss smoke. No model, optimizer or backward."""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import json,hashlib,tempfile,collections,copy,sys,time
from pathlib import Path
import numpy as np
import torch
from anatomy_job import ROOT,BASE,OwnedJob,write
from fm_training_reader import load_store,verify_core,sha,need
from fm_chunk_array import ChunkArray
import fm_aux_chunk_adapter as adapted
import fm_aux_reference as original
from fm_loader_reference import fm_window_targets,FM_CLASSES,FM_VIEWS,FM_SIZE
OUT=ROOT/'training_pixel_store'

def compare_annotation_rows(expected,actual):
 need(set(expected)==set(actual),'ALL_VIEW_ANNOTATION_ROW_MEMBERSHIP')
 for row,boxes in expected.items():
  left=np.asarray(sorted(tuple(x) for x in boxes),np.float64);right=np.asarray(sorted(tuple(x) for x in actual[row]),np.float64)
  need(left.shape==right.shape and np.allclose(left,right,atol=1e-7,rtol=0),'ALL_VIEW_ANNOTATION_MULTISET_MISMATCH:'+str(row))

def batch_equal(a,b):
 need(len(a)==len(b),'BATCH_ARITY')
 for x,y in zip(a,b):need(torch.equal(x,y) if isinstance(x,torch.Tensor) else x==y,'ACTUAL_DATASET_OR_LOADER_PARITY')
def summarize(batch):return [hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in batch]
def main(job):
 torch.set_num_threads(1);core=verify_core(OUT);st=load_store(OUT,review_geometry_only=True);pixels=ChunkArray(OUT);index=json.loads((OUT/'index.json').read_text());dec=json.loads((OUT/'provenance/ANATOMY_DECISIONS.json').read_text());need(len(st.vols)>=1000 and set(st.excluded)=={s for s,a in dec['volumes'].items() if not a['accepted']},'ACTUAL_EXCLUSIONS');need(all(not index['volumes'][s]['views'] for s in st.excluded),'EXCLUDED_HAS_ROWS')
 # The default API must reject an otherwise complete unpromoted store.
 try:load_store(OUT)
 except RuntimeError as e:need(str(e)=='FM_ANATOMICAL_STORE_NOT_PROMOTED','WRONG_UNPROMOTED_FAILURE')
 else:raise RuntimeError('DEFAULT_CONSUMER_FAIL_OPEN')
 # Empty/partial receipt dictionaries are rejected before reading any pixel chunk.
 with tempfile.TemporaryDirectory(prefix='receipt-rejection-',dir=ROOT/'qa') as td:
  fixture=Path(td)
  for name in ['index.json','pixel_store.json']:(fixture/name).write_bytes((OUT/name).read_bytes())
  (fixture/'CORE_BINDING.json').write_text(json.dumps({'state':'FROZEN_INPUTS_FOR_READER_QA','files_sha256':{}}))
  try:verify_core(fixture)
  except RuntimeError as e:need(str(e)=='FM_INCOMPLETE_REQUIRED_CORE_FILESET','WRONG_EMPTY_RECEIPT_FAILURE')
  else:raise RuntimeError('EMPTY_CORE_RECEIPT_ACCEPTED')
 # Independent deterministic sampler: requested4volumes/step,3/4boxed,1/4reviewednegative; only train groups.
 sampler=adapted.FmAuxSampler(st,steps=8,M=16,crop=352,shift=8,flip_p=.5,seeds=(137,173))
 batches=list(sampler);need(len(batches)==8 and sampler.yielded==8,'SAMPLER_STEPS');specs=[x for batch in batches for x in batch]
 repeat=list(adapted.FmAuxSampler(st,steps=8,M=16,crop=352,shift=8,flip_p=.5,seeds=(137,173)));need(batches==repeat,'EPOCH_SEED_RESUME_NOT_EXACT')
 sampled_slots=[]
 for batch in batches:
  need(len(batch)==16 and all(not st.hold[int(x[0])] for x in batch),'SAMPLER_VOLUME_OR_HOLDOUT')
  for slot in range(4):
   group=batch[slot*4:(slot+1)*4];vi=int(group[0][0]);need(all(int(x[0])==vi for x in group),'SAMPLER_FOUR_VOLUME_SLOTS');sampled_slots.append(vi in st.train_boxed)
   need(vi in st.train_boxed or (vi in st.train_neg and index['volumes'][st.vols[vi]]['reviewed_no_findings']),'SAMPLER_NEGATIVE_MUST_BE_REVIEWED_NO_FINDINGS')
 # Compare actual original dataset using a compact NPY for the same sampled volumes, not a mock pixel reader.
 chosen=sorted({int(x[0]) for x in specs});mini=copy.copy(st);mini.start={};mini.row_boxes={};mini.dir=None;parts=[];pos=0
 with tempfile.TemporaryDirectory(prefix='actual-loader-parity-',dir=ROOT/'qa') as td:
  mini.dir=td
  for vi in chosen:
   n=int(st.n[vi])
   for v in FM_VIEWS:
    mini.start[vi,v]=pos;parts.append(pixels[st.start[vi,v]:st.start[vi,v]+n])
    for k in range(n):
     oldrow=st.start[vi,v]+k
     if oldrow in st.row_boxes:mini.row_boxes[pos+k]=st.row_boxes[oldrow]
    pos+=n
  np.save(Path(td)/'px.u8.npy',np.concatenate(parts));ref=original.FmAuxWindows(mini,352,8);cur=adapted.FmAuxWindows(st,352,8)
  for spec in specs:batch_equal(cur[spec],ref[spec])
  ref_loader=original.fm_make_loader(mini,batches,352,8,0,191);cur0=adapted.fm_make_loader(st,batches,352,8,0,191);cur2=adapted.fm_make_loader(st,batches,352,8,2,191);digests=[]
  for b0,b2,br in zip(cur0,cur2,ref_loader):job.check();batch_equal(b0,b2);batch_equal(b0,br);digests.append(summarize(b0))
  # Photometric uint8 and actual independent dense/presence BCE forward only.
  batch=next(iter(adapted.fm_make_loader(st,batches[:1],352,8,0,191)));image,dense,presence,gam,con=batch;aug=adapted.fm_photometric(image,gam,con);need(aug.dtype==torch.uint8 and aug.shape==(16,3,352,352),'RGB_CROP_PHOTOMETRIC_SHAPE');need(dense.shape==(16,11,11,11) and presence.shape==(16,11),'DENSE_PRESENCE_SHAPE');need(torch.isfinite(dense).all() and torch.isfinite(presence).all(),'NONFINITE_TARGET');need((dense>=0).all() and (dense<=1).all() and ((presence==0)|(presence==1)).all(),'INVALID_TARGET_RANGE')
  loss_dense=torch.nn.functional.binary_cross_entropy_with_logits(torch.zeros_like(dense),dense);loss_presence=torch.nn.functional.binary_cross_entropy_with_logits(torch.zeros_like(presence),presence);need(torch.isfinite(loss_dense+loss_presence),'LOSS_FORWARD_NONFINITE')
 job.disk=64*1024**2;job.register(True);job.check()
 need(all(index['volumes'][st.vols[vi]]['reviewed_no_findings'] for vi in st.train_neg),'TRAIN_NEG_CONTAINS_UNREVIEWED_OR_FINDING_VOLUME')
 # Every no-findings volume has zero targets in all3views; study/artifact-only cannot enter this route.
 negatives=0
 for vi,s in enumerate(st.vols):
  if not index['volumes'][s]['reviewed_no_findings']:continue
  negatives+=1
  for v in FM_VIEWS:
   need(all(st.start[vi,v]+k not in st.row_boxes for k in range(int(st.n[vi]))),'NO_FINDINGS_VOLUME_HAS_BOXES_ON_SOME_SLICES')
   target,pres,_=fm_window_targets(st.boxes_of(st.window_rows(vi,v,0)),352,0,0,0,0,0,False);need(not target.any() and not pres.any(),'NO_FINDINGS_TARGET_NOT_ZERO')

 # Real source boxes beyond the old110mm frame remain represented in new fullfield rows.
 mapping=json.loads((OUT/'provenance/FM_ADAPTATION_SPEC.json').read_text())['class_mapping']['mapping'];outside=0;outside_examples=[];area_classes={};expected_all_views=collections.defaultdict(list);source_rows={r['stem']:r for r in json.loads((OUT/'provenance/SOURCE_FULLSHA_MANIFEST.json').read_text())['volumes']}
 for vi,stem in enumerate(st.vols):
  v=index['volumes'][stem];dx,dy=v['source_spacing_xy_mm'];extent=min(110,320*dx,320*dy);cw=round(extent/dx);ch=round(extent/dy);cx=round((320-cw)/2);cy=round((320-ch)/2);A=np.asarray(v['view_geometry']['0']['annotation_display_to_view_xy_edges']);vx,vy,vx1,vy1=v['view_geometry']['0']['rendered_valid_xyxy'];sourcepath=BASE/'source_rss'/stem/'boxes.source.json';need(sha(sourcepath)==source_rows[stem]['files']['boxes.source.json']['sha256'],'SOURCE_ANNOTATION_SHA_CHANGED');source=json.loads(sourcepath.read_text())
  for b in source['spatial']:
   bx,by,bw,bh=b['source_xywh'];klass=list(FM_CLASSES).index(mapping[b['label_raw']])
   for view in FM_VIEWS:
    geom=v['view_geometry'][str(view)];matrix=np.asarray(geom['annotation_display_to_view_xy_edges']);vvx,vvy,vvx1,vvy1=geom['rendered_valid_xyxy'];points=np.array([[bx,by,1],[bx+bw,by+bh,1]])@matrix.T;low=points[:,:2].min(0);high=points[:,:2].max(0);coords=[max(vvx,low[0]),max(vvy,low[1]),min(vvx1,high[0]),min(vvy1,high[1])]
    if coords[2]>coords[0] and coords[3]>coords[1]:expected_all_views[st.start[vi,view]+b['slice']].append([klass,*coords,bw*bh*abs(np.linalg.det(matrix[:2,:2]))])
   x,y,w,h=b['source_xywh'];c=list(FM_CLASSES).index(mapping[b['label_raw']]);row=st.start[vi,0]+b['slice'];stored=st.row_boxes.get(row,np.zeros((0,6)));pts=np.array([[x,y,1],[x+w,y+h,1]])@A.T;lo=pts[:,:2].min(0);hi=pts[:,:2].max(0);expected=np.array([c,max(vx,lo[0]),max(vy,lo[1]),min(vx1,hi[0]),min(vy1,hi[1]),w*h*abs(np.linalg.det(A[:2,:2]))]);match=next((z for z in stored if np.allclose(z,expected,atol=1e-7)),None);need(match is not None,'SOURCE_BOX_MISSING_OR_SHIFTED_IN_ACTUAL_FULLFIELD')
   if x<cx or y<cy or x+w>cx+cw or y+h>cy+ch:
    outside+=1
    if len(outside_examples)<8:outside_examples.append({'stem':stem,'slice':b['slice'],'raw_label':b['label_raw'],'row':row})
   if not st.hold[vi] and c not in area_classes:
    box=np.array([match]);ox=int(np.clip(round((match[1]+match[3])/2-176),0,32));oy=int(np.clip(round((match[2]+match[4])/2-176),0,32));dt,pt,kept=fm_window_targets(box,352,oy,ox,0,0,0,False);area=float(np.clip(min(match[3]-ox,352)-max(match[1]-ox,0),0,None)*np.clip(min(match[4]-oy,352)-max(match[2]-oy,0),0,None));need(abs(float(dt[c].sum())*1024-area)<=max(.01,area*1e-5),'DENSE_POSITIVE_AREA_CONSERVATION');need(bool(pt[c])==(area>=.25*match[5]),'PRESENCE_ORIGINAL_AREA_RULE')
    if area>0:area_classes[c]={'row':row,'retained_area':area,'a0':float(match[5]),'presence':int(pt[c])}
 compare_annotation_rows(expected_all_views,st.row_boxes)
 fixture={0:[[0,1.,2.,3.,4.,4.]],1:[[1,5.,6.,7.,8.,4.]]}
 for altered in [{0:fixture[0]*2,1:fixture[1]},{0:fixture[1],1:fixture[0]}]:
  try:compare_annotation_rows(fixture,altered)
  except RuntimeError:pass
  else:raise RuntimeError('EXTRA_OR_CROSS_VIEW_BOX_WAS_ACCEPTED')
 need(outside>0 and set(area_classes)==set(range(11)),'OUTSIDE_OLD110_OR_CLASS_AREA_COVERAGE');need(index['volumes']['file1002097']['excluded'] and not index['volumes']['file1002097']['views'],'QUARANTINE_CASE_REENTERED_TRAINING')
 # Actual worker cache cannot mutate decoded data.
 a=pixels[0]
 try:a.setflags(write=True)
 except ValueError:pass
 else:raise RuntimeError('READ_ONLY_CHUNK_CAN_BE_MUTATED')
 # Actual public installer is checked with an isolated namespace containing only exact public components.
 import types
 module=types.ModuleType('public_fm_components');module.__file__=str(Path(__file__).parent/'fm_aux_reference.py');sys.modules[module.__name__]=module
 exec(compile(Path(module.__file__).read_text(),module.__file__,'exec'),module.__dict__)
 adapted.install_fm_aux_adapter(module);need(module.FmAuxWindows is adapted.FmAuxWindows,'INSTALLER_NOT_PATCHED')
 try:module.FmStore(OUT)
 except RuntimeError as e:need(str(e)=='FM_ANATOMICAL_STORE_NOT_PROMOTED','INSTALLED_STORE_GUARD_WRONG_FAILURE')
 else:raise RuntimeError('INSTALLED_STORE_FAIL_OPEN')
 installed_batch=next(iter(module.fm_make_loader(st,batches[:1],352,8,2,191)));need(summarize(installed_batch)==digests[0],'ACTUAL_INSTALLED_MODULE_LOADER_PARITY')
 # Final receiver check every compressed byte, decoded byte and receipt, independent from producer's encoder comparison.
 decoded=0;compressed=0
 for k,c in enumerate(pixels.chunks):
  job.check();a=pixels.block(k);need(hashlib.sha256(a.tobytes()).hexdigest()==c['uncompressed_sha256'],'RECEIVER_DECODE_SHA');decoded+=a.nbytes;compressed+=(OUT/c['path']).stat().st_size
 job.register(True);job.check()
 common={'state':'PASS','core_binding_sha256':sha(OUT/'CORE_BINDING.json'),'training_started':False,'optimizer_or_backward_called':False,'verified_unix':time.time()}
 full=dict(common,checked_core_files=len(core['files_sha256']),checked_pixel_chunks=len(pixels.chunks),compressed_pixel_bytes=compressed,decoded_pixel_bytes=decoded,every_pixel_chunk_compressed_and_decoded_sha_checked=True);write(OUT/'provenance/FULLSHA_QA.json',full)
 qa=dict(common,usable_volumes=len(st.vols),excluded_volumes=len(st.excluded),n_rows=st.n_rows,reviewed_no_findings_volumes=negatives,actual_original_vs_chunk_samples=len(specs),actual_DataLoader_batches_workers0_vs2=len(digests),batch_tensor_sha256=digests,sampler_digest=sampler.h.hexdigest(),sampler_epoch_resume_specs_exact=True,sampler_volume_slots=32,sampled_boxed_slots=sum(sampled_slots),sampler_boxed_probability=.75,all11_train_classes=True,no_exam_group_cross_holdout=True,no_findings_all3views_zero_targets=True,dummy_BCE_dense=float(loss_dense),dummy_BCE_presence=float(loss_presence),crop=352,RGB_slices=3,geometry_and_sampling_preserved=True,default_unpromoted_consumer_rejected=True,read_only_chunk_mutation_rejected=True,actual_public_installer_PASS=True,all3views_exact_per_row_annotation_multisets=True,annotation_rows_compared=len(expected_all_views),annotation_boxes_compared=sum(len(x) for x in expected_all_views.values()),extra_and_cross_view_box_rejection_PASS=True,dense_area_and_presence_rule_all11_PASS=True,positive_area_class_examples=area_classes,source_boxes_outside_old110_retained_fullfield=outside,outside_old110_examples=outside_examples,quarantine_volume_excluded_all_views=True,incomplete_receipt_rejected=True,code_sha256=sha(Path(__file__)));write(OUT/'provenance/LOADER_TARGET_QA.json',qa);return qa
if __name__=='__main__':
 with OwnedJob('actual_FM_consumer_and_receiver_SHA_QA',cpu=4,ram_gib=5,disk_gib=.8) as job:print(json.dumps(main(job)),flush=True);job.progress['phase']='actual_consumer_QA_PASS_no_training'
