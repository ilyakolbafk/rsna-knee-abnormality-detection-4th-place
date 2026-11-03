"""Retire completed own GPU inference runtime after installing/QAing small pinned CPU consumer runtime."""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PIP_DISABLE_PIP_VERSION_CHECK='1')
import json,time,fcntl,shutil,threading,subprocess,hashlib,socket,signal
from pathlib import Path
from anatomy_job import ROOT,BASE,SHARED,write,ticks,live,G
KEY='fastmri_loader_runtime';reserve=int(1.7*G);phase='INSTALL_PINNED_CPU_CONSUMER_RUNTIME'
def register(active):
 with (SHARED/'admission.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);p=SHARED/'admission.json';d=json.loads(p.read_text());old=d['tasks'].get(KEY,{})
  assert old.get('PID') in [None,os.getpid()] or not live(old['PID'],old.get('ctime'))
  if active:
   peer=[v for k,v in d['tasks'].items() if k!=KEY];l=d['limits'];assert sum(v.get('cpu_threads',0) for v in peer)+1<=l['max_total_cpu_threads'];assert sum(v.get('download_workers',0) for v in peer)+1<=l['max_total_download_workers'];assert shutil.disk_usage(BASE).free>=l['disk_reserve_bytes']+sum(v.get('reserved_disk_bytes',0) for v in peer)+reserve,'LOADER_RUNTIME_DISK_ADMISSION'
  d['tasks'][KEY]={'PID':os.getpid(),'ctime':ticks(os.getpid()),'run_id':'fm-loader-runtime-20261021','active':active,'phase':phase if active else 'CPU_CONSUMER_RUNTIME_TERMINAL','cpu_threads':1 if active else 0,'download_workers':1 if active else 0,'reserved_disk_bytes':reserve if active else 0,'reserved_ram_bytes':2*G if active else 0,'gpu_indices':[],'heartbeat_unix':time.time(),'scratch_paths':[str(ROOT/'loader_venv'),str(ROOT/'loader_wheels')],'argv':Path('/proc/self/cmdline').read_bytes().decode().strip('\0').split('\0')};write(p,d)
def child(args):
 p=subprocess.Popen(args,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
 try:
  while True:
   try:out=p.communicate(timeout=15)[0];break
   except subprocess.TimeoutExpired:register(True)
  print(out[-10000:],flush=True);assert p.returncode==0,'RUNTIME_CHILD_FAILURE'
  return out
 except BaseException:
  if p.poll() is None:os.killpg(p.pid,signal.SIGTERM);p.wait(10)
  raise
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def main():
 t=json.loads((BASE/'TARGET_VERIFIED.json').read_text());assert socket.gethostname()==t['hostname'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==t['boot_id'];complete=ROOT/'ANCHOR_CORPUS_COMPLETE.json';assert json.loads(complete.read_text())['completed']==1172
 inf=ROOT/'inference_venv';dest=ROOT/'loader_venv';wheels=ROOT/'loader_wheels';wheels.mkdir(exist_ok=True)
 def no_inference_processes():
  for p in Path('/proc').iterdir():
   if not p.name.isdigit():continue
   try:
    cmd=(p/'cmdline').read_bytes().split(b'\0');exe=cmd[0].decode()
    if exe.startswith(str(inf)+'/') and '\nState:\tZ' not in (p/'status').read_text():raise RuntimeError('INFERENCE_RUNTIME_STILL_LIVE:'+p.name)
   except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 no_inference_processes();freeze=child([str(inf/'bin/python'),'-m','pip','freeze']);(ROOT/'qa/INFERENCE_RUNTIME_FINAL_PIP_FREEZE.txt').write_text(freeze)
 child([str(BASE/'render_venv/bin/python'),'-m','venv',str(dest)])
 pure=next((dest/'lib').glob('python*/site-packages'));source=next((BASE/'render_venv/lib').glob('python*/site-packages'));(pure/'fm_render_runtime.pth').write_text(str(source)+'\n')
 child([str(dest/'bin/python'),'-m','pip','download','--no-cache-dir','--only-binary=:all:','--dest',str(wheels),'--index-url','https://download.pytorch.org/whl/cpu','torch==2.8.0+cpu'])
 inventory={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in wheels.glob('*.whl')};write(ROOT/'qa/CPU_LOADER_WHEEL_MANIFEST.json',inventory)
 child([str(dest/'bin/python'),'-m','pip','install','--no-cache-dir','--no-index','--find-links',str(wheels),'torch==2.8.0+cpu'])
 versions=child([str(dest/'bin/python'),'-c',"import torch,numpy,cv2,PIL,json;assert torch.__version__=='2.8.0+cpu' and numpy.__version__=='2.2.6';print(json.dumps({'torch':torch.__version__,'numpy':numpy.__version__,'opencv':cv2.__version__,'pillow':PIL.__version__,'cuda_available':torch.cuda.is_available()}))"])
 cpufreeze=child([str(dest/'bin/python'),'-m','pip','freeze']);(ROOT/'qa/CPU_LOADER_PIP_FREEZE.txt').write_text(cpufreeze)
 no_inference_processes();assert json.loads(complete.read_text())['completed']==1172;before=shutil.disk_usage(BASE).free;inference_bytes=sum(p.lstat().st_blocks*512 for p in inf.rglob('*') if p.is_file() and not p.is_symlink());note={'state':'CPU_RUNTIME_QA_PASS_GPU_INFERENCE_ENV_RETIREMENT_PREPARED','corpus_completion_sha256':sha(complete),'inference_pip_freeze_sha256':sha(ROOT/'qa/INFERENCE_RUNTIME_FINAL_PIP_FREEZE.txt'),'cpu_pip_freeze_sha256':sha(ROOT/'qa/CPU_LOADER_PIP_FREEZE.txt'),'cpu_runtime':str(dest),'read_only_numpy_cv2_PIL_runtime':str(source),'versions':json.loads(versions.strip().splitlines()[-1]),'retired_only_own_paths':[str(inf),str(wheels)],'inference_allocated_bytes':inference_bytes,'source_or_weights_or_peer_deletions':False,'training_started':False,'created_unix':time.time()};write(ROOT/'qa/CPU_LOADER_RUNTIME_RECEIPT.json',note);shutil.rmtree(inf);shutil.rmtree(wheels);note.update(state='CPU_CONSUMER_RUNTIME_READY_OWN_GPU_ENV_RETIRED',disk_free_delta_bytes=shutil.disk_usage(BASE).free-before);write(ROOT/'qa/CPU_LOADER_RUNTIME_RECEIPT.json',note);print(json.dumps(note,indent=2),flush=True)
if __name__=='__main__':
 register(True)
 try:main()
 finally:register(False)
