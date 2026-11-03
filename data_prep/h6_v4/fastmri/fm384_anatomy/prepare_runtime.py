"""Prepare isolated pinned inference wheels; no training, medical input, or GPU work."""
import os
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',UV_CONCURRENT_DOWNLOADS='1',UV_CONCURRENT_INSTALLS='1',UV_CONCURRENT_BUILDS='1')
import subprocess,sys,json,time,hashlib
from pathlib import Path
from anatomy_job import ROOT,OwnedJob,write
def main():
 with OwnedJob('preparing_scoped_inference_runtime',downloads=1,disk_gib=35) as job:
  env=ROOT/'inference_venv'
  if not (env/'bin/python').exists():subprocess.run([sys.executable,'-m','venv',str(env)],check=True)
  py=str(env/'bin/python')
  commands=[[py,'-m','pip','install','--disable-pip-version-check','--retries','2','--timeout','40','torch==2.8.0','torchvision==0.23.0','--index-url','https://download.pytorch.org/whl/cu128'],[py,'-m','pip','install','--disable-pip-version-check','--retries','2','--timeout','40','numpy==2.2.6','scipy==1.16.2','opencv-python-headless==4.12.0.88','pillow==11.0.0','h5py==3.14.0','requests==2.32.5']]
  for c in commands:
   job.check();subprocess.run(c,check=True)
  v=subprocess.run([py,'-m','pip','freeze'],check=True,capture_output=True,text=True).stdout
  (ROOT/'RUNTIME_FREEZE.txt').write_text(v)
  write(ROOT/'RUNTIME_COMPLETE.json',{'state':'PASS_SCOPED_RUNTIME_INSTALLED','freeze_sha256':hashlib.sha256(v.encode()).hexdigest(),'python':py,'prepared_unix':time.time(),'no_training':True})
  job.progress.update(phase='scoped_runtime_prepared_no_inference')
if __name__=='__main__':main()
