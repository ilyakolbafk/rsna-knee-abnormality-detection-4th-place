"""Pinned coronal anatomy checkpoint and non-image architecture metadata."""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PIP_CACHE_DIR='/workspace/knee_v4_cache_20261021/fastmri_v1/anatomy_v1/package_cache')
import urllib.request,json,hashlib,time
import h5py
from anatomy_job import ROOT,OwnedJob,write
def main():
 with OwnedJob('receive_coronal_bone_joint_model',cpu=2,downloads=1) as job:
  url='https://huggingface.co/aagatti/dosma_bones/resolve/ff5e0b5d1bf8b850ee6258210e05418889d7133f/coronal_best_model.h5';dst=ROOT/'weights/coronal_best_model.h5';tmp=dst.with_suffix('.h5.partial');h=hashlib.sha256();n=0
  with urllib.request.urlopen(url,timeout=45) as response,tmp.open('wb') as f:
   assert response.status==200
   for b in iter(lambda:response.read(4*1024**2),b''):job.check();f.write(b);h.update(b);n+=len(b)
   f.flush();os.fsync(f.fileno())
  assert n==173475456 and h.hexdigest()=='bb63556a712255531da10e9342202571c1f3fd247e5756b7c0ed160af2ad47c2';os.replace(tmp,dst)
  with h5py.File(dst,'r') as f:
   meta={'file_attrs':{},'weight_shapes':{}}
   for k,v in f.attrs.items():
    if isinstance(v,bytes):v=v.decode()
    if isinstance(v,str):
     try:v=json.loads(v)
     except json.JSONDecodeError:pass
    meta['file_attrs'][k]=v
   def visit(name,obj):
    if isinstance(obj,h5py.Dataset):meta['weight_shapes'][name]=list(obj.shape)
   f.visititems(visit)
  write(ROOT/'CORONAL_MODEL_CONFIG.json',meta);write(ROOT/'CORONAL_MODEL_RECEIPT.json',{'source_url':url,'bytes':n,'receiver_sha256':h.hexdigest(),'license_repo_tag':'MIT; separate from parent DOSMA GPL code','class_map':'pc/fc/mtc/ltc/med_men/lat_men/fem/tib/pat; no fibula','input_domain':'coronal reformat of SKM-TEA qDESS, new RSS contrast requires validation','training_ready':False,'received_unix':time.time()});job.progress.update(phase='coronal_model_received_config_extracted')
if __name__=='__main__':main()
