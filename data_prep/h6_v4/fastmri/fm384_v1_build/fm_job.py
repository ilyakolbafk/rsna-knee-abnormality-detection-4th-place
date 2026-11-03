"""One owned finalizer, same shared admission as OAI and the RSS collector."""
import os,json,time,fcntl,threading,shutil,socket
from pathlib import Path
from build_candidate import write
ROOT=Path('/workspace/knee_v4_cache_20261021/fastmri_v1');SHARED=ROOT.parent/'shared';G=1024**3

def ticks(pid):return Path('/proc/%s/stat'%pid).read_text().rsplit(')',1)[1].split()[19]
def process_live(pid,ctime,argv=None):
 try:
  p=Path('/proc/%s'%pid)
  if ticks(pid)!=str(ctime) or '\nState:\tZ' in p.joinpath('status').read_text():return False
  actual=p.joinpath('cmdline').read_bytes().split(b'\0');actual=[x.decode() for x in actual if x]
  if argv and not all(x in actual for x in argv):raise RuntimeError('LIVE_PID_ARGV_MISMATCH')
  return True
 except (FileNotFoundError,ProcessLookupError):return False
class OwnedJob:
 def __init__(self):self.progress={'phase':'source_fullSHA_verification','ready':False,'expected_volumes':1172};self.stop=threading.Event();self.lock=None;self.thread=None
 def register(self,active=True):
  with (SHARED/'admission.lock').open('a') as f:
   fcntl.flock(f,fcntl.LOCK_EX);p=SHARED/'admission.json';d=json.loads(p.read_text());assert d['version']==1 and int(d['target']['instance_id'])==56967269
   old=d['tasks'].get('fastmri_v1',{});pid=old.get('PID')
   if pid and pid!=os.getpid() and process_live(pid,old.get('ctime'),old.get('argv')):raise RuntimeError('OTHER_LIVE_FASTMRI_WORKER')
   own={'run_id':'fm384-v1-56967269-20261021','PID':os.getpid(),'ctime':ticks(os.getpid()),'argv':[str(ROOT/'render_venv/bin/python'),str(ROOT/'code/finish_pipeline.py')],'heartbeat_unix':time.time(),'reserved_disk_bytes':26*G if active else 0,'reserved_ram_bytes':4*G if active else 0,'cpu_threads':2 if active else 0,'download_workers':0,'convert_workers':1 if active else 0,'scratch_paths':[str(ROOT/'candidate_pixel_store')],'active':active,'phase':self.progress['phase']}
   if active:
    peers=[v for k,v in d['tasks'].items() if k!='fastmri_v1'];limits=d['limits'];assert sum(v.get('cpu_threads',0) for v in peers)+2<=limits['max_total_cpu_threads'],'CPU_ADMISSION'
    mi={a:int(b.split()[0])*1024 for a,b in (x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines())};cg=Path('/sys/fs/cgroup/memory');mst=dict(x.split() for x in (cg/'memory.stat').read_text().splitlines());head=int((cg/'memory.limit_in_bytes').read_text())-int((cg/'memory.usage_in_bytes').read_text())+int(mst.get('total_inactive_file',0))
    if min(head,mi['MemAvailable'])<limits['memory_reserve_bytes']+sum(v.get('reserved_ram_bytes',0) for v in peers)+4*G:raise RuntimeError('RAM_ADMISSION')
    if shutil.disk_usage(ROOT).free<limits['disk_reserve_bytes']+sum(v.get('reserved_disk_bytes',0) for v in peers)+26*G:raise RuntimeError('DISK_ADMISSION')
   d['tasks']['fastmri_v1']=own;write(p,d)
 def __enter__(self):
  b=json.loads((ROOT/'TARGET_VERIFIED.json').read_text());assert socket.gethostname()==b['hostname'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==b['boot_id'] and Path('/root/.vast_containerlabel').read_text().strip()=='C.56967269'
  self.lock=(ROOT/'singleton.lock').open('a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  while True:
   try:self.register();break
   except RuntimeError as e:
    if str(e) not in ['RAM_ADMISSION','DISK_ADMISSION']:raise
    write(ROOT/'PROGRESS.json',dict(self.progress,phase='waiting_resource_admission',reason=str(e),updated_unix=time.time()));self.stop.wait(20)
  def heartbeat():
   while not self.stop.wait(40):
    try:
     self.register();write(ROOT/'STATE.json',dict(self.progress,PID=os.getpid(),ctime=ticks(os.getpid()),updated_unix=time.time()))
    except Exception as e:self.progress['abort_reason']=str(e)
  self.thread=threading.Thread(target=heartbeat,daemon=True);self.thread.start();return self
 def __exit__(self,ty,exc,tb):
  self.stop.set()
  if self.thread:self.thread.join()
  if exc:self.progress.update(phase='finalizer_failed',error_type=type(exc).__name__,error=str(exc),ready=False)
  write(ROOT/'STATE.json',dict(self.progress,PID=os.getpid(),ctime=ticks(os.getpid()),updated_unix=time.time(),process_terminal=True));self.register(active=False)
  if self.lock:self.lock.close()
