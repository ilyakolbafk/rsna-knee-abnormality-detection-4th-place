"""Separate CPU-only TensorFlow reference for validating an inference adapter."""
import os
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PIP_CACHE_DIR='/workspace/knee_v4_cache_20261021/fastmri_v1/anatomy_v1/package_cache')
import subprocess,sys,time
from anatomy_job import ROOT,OwnedJob,write
def main():
 with OwnedJob('prepare_CPU_TensorFlow_reference_only',downloads=1) as job:
  env=ROOT/'tf_reference_venv'
  if not (env/'bin/python').exists():subprocess.run([sys.executable,'-m','venv',str(env)],check=True)
  subprocess.run([str(env/'bin/python'),'-m','pip','install','--disable-pip-version-check','--retries','2','--timeout','40','tensorflow-cpu==2.20.0','numpy==2.2.6','h5py==3.14.0'],check=True)
  write(ROOT/'TF_REFERENCE_RUNTIME.json',{'state':'SCOPED_CPU_REFERENCE_INSTALLED','python':str(env/'bin/python'),'prepared_unix':time.time(),'training_ready':False});job.progress.update(phase='TF_CPU_reference_runtime_prepared')
if __name__=='__main__':main()
