"""Real structural/target QA; refuses promoting screen views to anatomical M/L."""
import collections,json,os,time
from pathlib import Path
import numpy as np
from build_candidate import ROOT,OUT,sha,write
from fm_candidate_reader import load_store
from fm_loader_reference import fm_window_targets,fm_window_image,FM_CLASSES

def main(progress=None):
 idx=json.loads((OUT/'index.json').read_text());assert idx['training_ready'] is False and idx['semantic_gate']=='BLOCKED_ANATOMICAL_ML_UNVERIFIED'
 try:load_store(OUT)
 except RuntimeError as e:assert str(e)=='CANDIDATE_NOT_PROMOTED_ANATOMICAL_ML_UNVERIFIED'
 else:raise AssertionError('DEFAULT_READER_DID_NOT_REFUSE_CANDIDATE')
 st=load_store(OUT,review_geometry_only=True,min_volumes=1000);px=np.load(OUT/'px.u8.npy',mmap_mode='r');assert px.shape==(idx['n_rows'],384,384) and px.dtype==np.uint8
 visited=0;dense_classes=set();sampled=[];nonzero_targets=0
 for vi,v in enumerate(st.vols):
  n=int(st.n[vi])
  for view in [0,2,3]:
   start=st.start[(vi,view)]
   for k in range(n):
    rows=st.window_rows(vi,view,k);assert rows==[start+max(0,k-1),start+k,start+min(n-1,k+1)];assert all(start<=r<start+n for r in rows);visited+=1
 for row,b in st.row_boxes.items():
  classes={int(x) for x in b[:,0]}
  if not classes-dense_classes:continue
  dense,pres,kept=fm_window_targets(b,384,0,0,0,0,0,False)
  for c in classes:
   if dense[c].sum()>0:dense_classes.add(c)
  nonzero_targets+=int(dense.sum()>0);sampled.append(row)
  if len(dense_classes)==11:break
 assert dense_classes==set(range(11)),'ALL11_POSITIVE_AREA_DENSE_TARGETS'
 # Actual mmap windows and lazy crop352/RGB3/augmentation geometry.
 for vi in [0,len(st.vols)//2,len(st.vols)-1]:
  for view in [0,2,3]:
   for k in [0,int(st.n[vi])//2,int(st.n[vi])-1]:
    rows=st.window_rows(vi,view,k);boxes=st.boxes_of(rows)
    for shift,dy,dx,flip in [(0,0,0,False),(8,3,-4,True)]:
     image=fm_window_image(px,rows,352,16,16,dy,dx,shift,flip);dense,pres,b=fm_window_targets(boxes,352,16,16,dy,dx,shift,flip)
     assert image.shape==(3,352,352) and image.dtype==np.uint8 and image.flags.c_contiguous
     assert dense.shape==(11,11,11) and pres.shape==(11,) and np.isfinite(dense).all() and ((dense>=0)&(dense<=1)).all()
 pids=collections.defaultdict(set)
 for pid,flag in zip(st.pid,st.hold):pids[pid].add(bool(flag))
 assert all(len(v)==1 for v in pids.values())
 excluded=set(st.excluded);assert len(excluded)==5
 for v in excluded:assert idx['volumes'][v]['exclude_reason']=='STUDY_LEVEL_ARTIFACT_ONLY_NOT_SPATIAL_NEGATIVE'
 assert all(idx['volumes'][st.vols[v]]['reviewed_no_findings'] for v in st.train_neg),'EMPTY_BOX_VOLUME_NOT_VERIFIED_NEGATIVE'
 spans=sorted((off,off+v['n_slices']) for v in idx['volumes'].values() for _,off in v['views']);assert spans[0][0]==0 and spans[-1][1]==len(px) and all(a1==b0 for (_,a1),(b0,_) in zip(spans,spans[1:]))
 receipt=json.loads((OUT/'COMPLETION_RECEIPT.json').read_text())
 for name,v in receipt['files'].items():
  s=(OUT/name).stat();assert s.st_ino==v['stat']['inode'] and s.st_size==v['stat']['bytes'] and s.st_mtime_ns==v['stat']['mtime_ns'] and s.st_ctime_ns==v['stat']['ctime_ns']
 report={'state':'PASS_STRUCTURAL_AND_TARGET_QA_ONLY','candidate_training_ready':False,'anatomical_ML_ready':False,'private_pixel_parity':False,'default_guarded_reader_refusal':'PASS','stock_public_FmStore_flags_are_advisory':True,'public_FmStore_read_in_explicit_geometry_review_mode':'PASS','usable_volumes':len(st.vols),'excluded_volumes':len(st.excluded),'H5_exam_groups':len(set(st.pid)),'holdout_usable_volumes':len(st.holdout),'training_boxed_volumes':len(st.train_boxed),'training_reviewed_no_findings_volumes':len(st.train_neg),'n_rows':len(px),'all_valid_window_positions_checked':visited,'all11_dense_classes_positive_area':True,'crop352_RGB3_shift_flip_lazy_smoke':'PASS','no_cross_volume_or_view_windows':'PASS','no_H5_exam_group_on_both_holdout_sides':'PASS','all_global_rows_contiguous':True,'actual_store_bytes':receipt['actual_store_bytes'],'payload_bytes':receipt['uncompressed_uint8_payload_bytes'],'allocated_pixel_bytes':(OUT/'px.u8.npy').stat().st_blocks*512,'receiver_fullSHA_manifest':'COMPLETION_RECEIPT.json','verified_unix':time.time(),'remaining_blocker':'No anatomical medial/lateral correspondence proof; candidate2/3 are declared screen fields, not validated M/L views'}
 write(ROOT/'qa/LOADER_TARGET_QA.json',report)
 if progress is not None:progress.update(phase='source_and_pixel_candidate_complete_semantic_blocked',committed=1172,built_volumes=1172,valid_rows=len(px),ready=False,blocker=report['remaining_blocker'])
 print(json.dumps(report,indent=2));return report
if __name__=='__main__':main()
