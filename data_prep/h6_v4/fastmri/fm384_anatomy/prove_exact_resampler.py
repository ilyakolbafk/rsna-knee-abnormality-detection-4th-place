"""Numeric parity on actual MR logits and raw RSS; returns original result during proof."""
import os
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',nnUNet_compile='false',nnUNet_def_n_proc='4')
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from anatomy_job import ROOT,BASE,OwnedJob,write
from exact_ap_resampler import install,resample_exact
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
 return h.hexdigest()

def main():
 with OwnedJob('exact_AP_resampling_numeric_parity',cpu=4,ram_gib=12,disk_gib=1,gpus=[0]) as job:
  torch.set_num_threads(4);torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;checks=[];current=[None]
  def prove(data,new_shape,is_seg,axis,order,do_separate_z,order_z,dtype_out,original):
   t=time.time();ref=original(data,new_shape,is_seg,axis,order,do_separate_z,order_z,dtype_out);rt=time.time()-t;t=time.time();fast=resample_exact(data,new_shape,is_seg,axis,order,do_separate_z,order_z,dtype_out,original=original);ft=time.time()-t
   eq=np.array_equal(ref,fast);rec={'stem':current[0],'input_shape':list(data.shape),'output_shape':list(ref.shape),'input_dtype':str(data.dtype),'dtype_out':str(ref.dtype),'order':order,'order_z':order_z,'axis':np.asarray(axis).tolist(),'array_equal':eq,'max_abs_error':0. if eq else float(np.max(np.abs(ref.astype(float)-fast.astype(float)))),'original_seconds':rt,'optimized_seconds':ft,'reference_sha256':hashlib.sha256(ref.tobytes()).hexdigest(),'optimized_sha256':hashlib.sha256(fast.tobytes()).hexdigest()};checks.append(rec);write(ROOT/'qa/EXACT_RESAMPLE_PARITY_PARTIAL.json',checks);assert eq,'NUMERIC_RESAMPLE_PARITY_FAILED';return ref
  install(prove)
  from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
  pred=nnUNetPredictor(tile_step_size=1.,use_gaussian=True,use_mirroring=False,perform_everything_on_device=True,device=torch.device('cuda',0),verbose=False,verbose_preprocessing=False,allow_tqdm=False);pred.initialize_from_trained_model_folder(str(ROOT/'safe_task002'),use_folds=(0,),checkpoint_name='checkpoint_final.pth');hyp=json.loads((ROOT/'SIDE_HYPOTHESES.json').read_text())['volumes'];results={}
  for stem in ['file1000000','file1001479','file1001949','file1001698']:
   current[0]=stem;job.check();p=BASE/'source_rss'/stem;raw=np.load(p/'rss.f32.npy');m=json.loads((p/'metadata.json').read_text());flip=hyp[stem]['source_mean_logit']<0;v=raw[::-1,:,:]
   if flip:v=v[:,:,::-1]
   seg=pred.predict_single_npy_array(np.ascontiguousarray(v.transpose(1,0,2)[None],dtype=np.float32),{'spacing':[m['spacing_xy_mm'][1],3.,m['spacing_xy_mm'][0]]}).transpose(1,0,2)[::-1,:,:]
   if flip:seg=seg[:,:,::-1]
   baseline=np.load(ROOT/'pilot/mr_joints'/(stem+'_AP3.0_tile1.0.seg.npy'));eq=np.array_equal(baseline,seg);results[stem]={'original_pilot_segmentation_equal':eq,'source_receipt_sha256':sha(p/'receipt.json')};assert eq,'ACTUAL_MODEL_SEGMENTATION_CHANGED';print(json.dumps({'stem':stem,'checks':len(checks),'pilot_segmentation_equal':eq}),flush=True)
  assert len(checks)==8 and {r['order'] for r in checks}=={1,3}
  proof={'state':'PASS_EXACT_NUMERIC_PARITY','checks':checks,'actual_RSS_and_actual_model_logits_cases':results,'resampler_code_sha256':sha(ROOT/'exact_ap_resampler.py'),'proof_code_sha256':sha(__file__),'original_resampling_source_sha256':sha(Path(__import__('nnunetv2.preprocessing.resampling.default_resampling',fromlist=['x']).__file__)),'anchor_binding_sha256':sha(ROOT/'ANCHOR_BUILD_BINDING.json'),'no_interpolation_order_or_model_change':True,'training_ready':False,'completed_unix':time.time()};write(ROOT/'qa/EXACT_RESAMPLE_PARITY.json',proof);job.progress['phase']='exact_resampler_numerically_verified_no_promotion'
if __name__=='__main__':main()
