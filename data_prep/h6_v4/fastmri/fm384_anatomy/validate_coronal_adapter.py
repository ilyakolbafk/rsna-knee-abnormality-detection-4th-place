"""Numeric GPU adapter parity against a separate CPU Keras runtime."""
import os
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import json,hashlib,time,subprocess
from pathlib import Path
import numpy as np
import torch
from skimage.transform import resize
from anatomy_job import ROOT,BASE,OwnedJob,write
from coronal_graph import CoronalGraph
torch.set_num_threads(4)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 with OwnedJob('validate_coronal_GPU_adapter_against_CPU_Keras',gpus=[0]) as job:
  q=ROOT/'qa';q.mkdir(exist_ok=True);xs=[np.random.default_rng(20261021).normal(size=(1,160,512)).astype(np.float32)];refs=[{'synthetic_seed':20261021}]
  for stem,z in [('file1000000',20),('file1000000',24),('file1001479',20),('file1001949',24)]:
   p=BASE/'source_rss'/stem;raw=np.load(p/'rss.f32.npy');v=np.ascontiguousarray(raw[:,::-1,:].transpose(2,1,0));v=resize(v,(160,512,v.shape[-1]),order=3);v=(v-np.mean(v))/(np.std(v)+1e-8);xs.append(v[:,:,z][None].astype(np.float32));refs.append({'stem':stem,'source_slice':z,'source_receipt_sha256':sha(p/'receipt.json'),'preprocess':'one flipud; model image axes(col,row), resize160x512 cubic, whole-volume whiten eps1e-8'})
  x=np.stack(xs);np.save(q/'coronal_parity_inputs.npy',x)
  binding={'weight_sha256':sha(ROOT/'weights/coronal_best_model.h5'),'inputs_sha256':sha(q/'coronal_parity_inputs.npy'),'graph_code_sha256':sha(ROOT/'coronal_graph.py'),'reference_code_sha256':sha(ROOT/'coronal_tf_reference.py'),'input_references':refs,'tolerances':{'max_abs':.0005,'mean_abs':.000005,'argmax_agreement':.99995}}
  write(q/'CORONAL_PARITY_BINDING.json',binding)
  job.check();subprocess.run([str(ROOT/'tf_reference_venv/bin/python'),str(ROOT/'coronal_tf_reference.py')],check=True)
  tf=np.load(q/'coronal_TF_reference_outputs.npy');assert tf.shape==(len(x),10,160,512)
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  net=CoronalGraph(ROOT/'weights/coronal_best_model.h5').eval().cuda();ys=[]
  with torch.inference_mode():
   for a in x:ys.append(net(torch.from_numpy(a[None]).cuda()).cpu().numpy())
  y=np.concatenate(ys);assert y.shape==tf.shape and np.isfinite(y).all();diff=np.abs(y-tf);agreement=float((y.argmax(1)==tf.argmax(1)).mean())
  result={'binding':binding,'max_absolute_error':float(diff.max()),'mean_absolute_error':float(diff.mean()),'argmax_agreement':agreement,'per_sample_max_abs':[float(z.max()) for z in diff],'training_ready':False,'completed_unix':time.time()}
  ok=result['max_absolute_error']<=.0005 and result['mean_absolute_error']<=.000005 and agreement>=.99995;result['state']='PASS_NUMERIC_GRAPH_PARITY_ONLY' if ok else 'FAIL_NUMERIC_GRAPH_PARITY';write(q/'CORONAL_GRAPH_PARITY.json',result);assert ok,'CORONAL_GRAPH_PARITY_FAILED';job.progress.update(phase='coronal_graph_numeric_parity_pass_domain_QA_pending')
if __name__=='__main__':main()
