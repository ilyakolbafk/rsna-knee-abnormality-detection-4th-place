"""Bound anatomy proposals on owned GPUs. No training or readiness promotion."""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',nnUNet_compile='false',nnUNet_def_n_proc='2')
import json,time,hashlib,multiprocessing as mp,argparse
from pathlib import Path
from anatomy_job import ROOT,BASE,OwnedJob,write

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def worker(gpu,q,res,binding,selection):
 try:
  if hasattr(os,'sched_getaffinity'):
   cores=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,cores[gpu*2:gpu*2+2])
  import numpy as np
  import torch
  from bone_geometry import segment_canonical,propose_joint
  from coronal_graph import CoronalGraph
  torch.set_num_threads(2);torch.cuda.set_device(gpu);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  cornet=None;mr=None
  def mrseg(raw,flip,spacing):
   nonlocal mr
   if mr is None:
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    mr=nnUNetPredictor(tile_step_size=1.,use_gaussian=True,use_mirroring=False,perform_everything_on_device=True,device=torch.device('cuda',gpu),verbose=False,verbose_preprocessing=False,allow_tqdm=False)
    mr.initialize_from_trained_model_folder(str(ROOT/'safe_task002'),use_folds=(0,),checkpoint_name='checkpoint_final.pth')
   v=raw[::-1,:,:]
   if flip:v=v[:,:,::-1]
   data=np.ascontiguousarray(v.transpose(1,0,2)[None],dtype=np.float32)
   seg=mr.predict_single_npy_array(data,{'spacing':[spacing[1],3.,spacing[0]]}).transpose(1,0,2)[::-1,:,:]
   # seg is already width-canonical. Convert rawrow into displayrow only.
   seg=np.ascontiguousarray(seg[:,::-1,:],dtype=np.uint8);generic=np.zeros(seg.shape,np.uint8);generic[seg==1]=7;generic[seg==4]=8
   p=propose_joint(generic,spacing);counts={str(k):int(v) for k,v in zip(*np.unique(seg,return_counts=True))};foreground=seg!=0;other=int(((seg>=5)&foreground).sum())/max(1,int(foreground.sum()))
   p['all_model_class_counts']=counts;p['other_foreground_fraction']=other
   if other>.5:p['geometry_issues'].append('OTHER_ANATOMICAL_REGION_DOMINATES')
   return seg,p
  while True:
   case=q.get()
   if case is None:break
   stem=case['stem'];dest=ROOT/'anchor_proposals'/(stem+'.json');p=BASE/'source_rss'/stem;rb={'binding_sha256':binding,'source_receipt_sha256':case['receipt_sha256']}
   if dest.exists():
    prev=json.loads(dest.read_text());assert prev['row_binding']==rb,'ANCHOR_RESUME_BINDING_MISMATCH';res.put({'stem':stem,'reused':True,'issues':prev['proposal']['geometry_issues']});continue
   assert sha(p/'receipt.json')==case['receipt_sha256'],'SOURCE_RECEIPT_CHANGED'
   for name,v in case['files'].items():
    s=(p/name).stat();assert all(getattr(s,k)==v['stat'][n] for k,n in [('st_ino','inode'),('st_size','bytes'),('st_mtime_ns','mtime_ns'),('st_ctime_ns','ctime_ns')]),'SOURCE_STAT_CHANGED'
   raw=np.load(p/'rss.f32.npy',mmap_mode='r');m=json.loads((p/'metadata.json').read_text());h=json.loads((ROOT/'SIDE_HYPOTHESES.json').read_text())['volumes'][stem];flip=h['source_mean_logit']<0;t=time.time();fallback=None
   if m['h5_attrs']['acquisition']=='CORPDFS_FBK':
    if cornet is None:cornet=CoronalGraph(ROOT/'weights/coronal_best_model.h5').eval().to(torch.device('cuda',gpu))
    # segment_canonical calls .cuda(); current CUDA device is explicitly bound.
    seg=segment_canonical(raw,flip,cornet);proposal=propose_joint(seg,m['spacing_xy_mm']);method='DOSMA_coronal_qDESS_literal_graph'
    if proposal['geometry_issues']:
     fallback=proposal;seg,proposal=mrseg(raw,flip,m['spacing_xy_mm']);method='MRAnnotator_PD_PD_FS_generic_bones_fallback'
   else:seg,proposal=mrseg(raw,flip,m['spacing_xy_mm']);method='MRAnnotator_PD_PD_FS_generic_bones'
   a=proposal['joint_anchor_canonical_display_row_col'];source_anchor=[a[0],raw.shape[2]-1-a[1] if flip else a[1]] if a else None
   out={'row_binding':rb,'stem':stem,'method':method,'hflip_to_lateral_pluscol':flip,'source_anchor_display_row_col':source_anchor,'proposal':proposal,'failed_qDESS_proposal':fallback,'seconds':time.time()-t,'worker_gpu':gpu,'source_AP_distance_claimed':False,'native_LPS_or_LAS_claimed':False,'training_ready':False}
   if stem in selection:
    path=ROOT/'anchor_validation_segmentation'/(stem+'.seg.npy');np.save(path,seg);out['validation_segmentation_sha256']=sha(path);out['validation_segmentation_frame']='canonical head-up lateral+column; unchanged source slice order'
   write(dest,out);res.put({'stem':stem,'reused':False,'method':method,'seconds':out['seconds'],'issues':proposal['geometry_issues']})
 except Exception as e:
  import traceback;res.put({'worker_error':str(e),'traceback':traceback.format_exc()[-4000:]})

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['validation','corpus'],default='validation');a=ap.parse_args()
 with OwnedJob('joint_anchor_'+a.phase+'_inference',cpu=6,ram_gib=24,disk_gib=2,gpus=[0,1]) as job:
  assert json.loads((ROOT/'qa/CORONAL_GRAPH_PARITY.json').read_text())['state']=='PASS_NUMERIC_GRAPH_PARITY_ONLY';assert (ROOT/'pilot/mr_joints/COMPLETE.json').exists()
  if a.phase=='corpus':assert json.loads((ROOT/'ANCHOR_VALIDATION_PASS.json').read_text())['state']=='PASS','INDEPENDENT_ANCHOR_GATE_REQUIRED'
  full=BASE/'receipts/SOURCE_FULLSHA_MANIFEST.json';source=json.loads(full.read_text());sel=json.loads((ROOT/'VALIDATION_SELECTION.json').read_text());selection={c['stem'] for c in sel['cases']};entries=[r for r in source['volumes'] if a.phase=='corpus' or r['stem'] in selection]
  out=ROOT/'anchor_proposals';out.mkdir(exist_ok=True);(ROOT/'anchor_validation_segmentation').mkdir(exist_ok=True)
  bind={'source_manifest_sha256':sha(full),'policy_sha256':sha(ROOT/'JOINT_ANCHOR_POLICY.v1.json'),'side_hypotheses_sha256':sha(ROOT/'SIDE_HYPOTHESES.json'),'model_sha256':{n:sha(ROOT/n) for n in ['weights/coronal_best_model.h5','safe_task002/fold_0/checkpoint_final.pth']},'code_sha256':{n:sha(ROOT/n) for n in ['infer_joint_anchors.py','coronal_graph.py','bone_geometry.py']},'numpy_version':__import__('numpy').__version__,'training_ready':False};bp=ROOT/'ANCHOR_BUILD_BINDING.json'
  if bp.exists():assert json.loads(bp.read_text())==bind,'ANCHOR_BINDING_CHANGED'
  else:write(bp,bind)
  ctx=mp.get_context('spawn');q=ctx.Queue();res=ctx.Queue();procs=[ctx.Process(target=worker,args=(g,q,res,sha(bp),selection)) for g in [0,1]]
  for p in procs:p.start()
  for c in entries:q.put(c)
  for p in procs:q.put(None)
  done=0;stats=[]
  try:
   while done<len(entries):
    job.check()
    try:r=res.get(timeout=15)
    except __import__('queue').Empty:
     assert all(p.is_alive() for p in procs),'ANCHOR_WORKER_EXITED_EARLY';continue
    if 'worker_error' in r:raise RuntimeError(r['worker_error']+'\n'+r['traceback'])
    done+=1;stats.append(r);job.progress.update(phase='joint_anchor_'+a.phase+'_inference',completed=done,expected=len(entries),last_stem=r['stem'])
    if done%4==0:print(json.dumps({'completed':done,'expected':len(entries),'last':r}),flush=True)
   for p in procs:p.join(30);assert p.exitcode==0
  finally:
   for p in procs:
    if p.is_alive():p.terminate();p.join()
  write(ROOT/('ANCHOR_'+a.phase.upper()+'_COMPLETE.json'),{'state':'PROPOSALS_COMPLETE_NOT_VISUAL_PROMOTION','binding_sha256':sha(bp),'completed':done,'entries':stats,'training_ready':False});job.progress['phase']='anchor_'+a.phase+'_proposals_complete'
if __name__=='__main__':main()
