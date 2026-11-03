"""Owned anatomy inference/preparation admission; never changes peer records."""
import os,json,time,fcntl,threading,socket,shutil,tempfile
from pathlib import Path
BASE=Path('/workspace/knee_v4_cache_20261021/fastmri_v1')
ROOT=BASE/'anatomy_v1'
SHARED=BASE.parent/'shared'
G=1024**3
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 fd,n=tempfile.mkstemp(prefix=path.name+'.tmp.',dir=path.parent)
 with os.fdopen(fd,'w') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(n,path)
 directory_fd=os.open(path.parent,os.O_RDONLY)
 try:os.fsync(directory_fd)
 finally:os.close(directory_fd)
def ticks(pid):return Path('/proc/%s/stat'%pid).read_text().rsplit(')',1)[1].split()[19]
def live(pid,ctime):
 try:return ticks(pid)==str(ctime) and '\nState:\tZ' not in Path('/proc/%s/status'%pid).read_text()
 except (FileNotFoundError,ProcessLookupError):return False
class OwnedJob:
 def __init__(self,phase,cpu=4,ram_gib=8,disk_gib=35,downloads=0,gpus=()):
  self.progress={'phase':phase,'training_ready':False};self.cpu=cpu;self.ram=ram_gib*G;self.disk=disk_gib*G;self.downloads=downloads;self.gpus=list(gpus);self.stop=threading.Event();self.error=None
 def register(self,active):
  with (SHARED/'admission.lock').open('a') as f:
   fcntl.flock(f,fcntl.LOCK_EX);p=SHARED/'admission.json';d=json.loads(p.read_text());assert int(d['target']['instance_id'])==56967269
   key='fastmri_anatomy_v1';old=d['tasks'].get(key,{})
   if old.get('PID') not in [None,os.getpid()] and live(old['PID'],old.get('ctime')):raise RuntimeError('OTHER_LIVE_ANATOMY_JOB')
   if active:
    peers=[v for k,v in d['tasks'].items() if k!=key];l=d['limits']
    assert sum(v.get('cpu_threads',0) for v in peers)+self.cpu<=l['max_total_cpu_threads'],'CPU_ADMISSION'
    assert sum(v.get('download_workers',0) for v in peers)+self.downloads<=l['max_total_download_workers'],'DOWNLOAD_ADMISSION'
    assert shutil.disk_usage(BASE).free>=l['disk_reserve_bytes']+sum(v.get('reserved_disk_bytes',0) for v in peers)+self.disk,'DISK_ADMISSION'
    mi={a:int(b.split()[0])*1024 for a,b in (x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines())}
    assert mi['MemAvailable']>=l['memory_reserve_bytes']+sum(v.get('reserved_ram_bytes',0) for v in peers)+self.ram,'RAM_ADMISSION'
    for v in peers:assert not set(self.gpus).intersection(v.get('gpu_indices',[])),'GPU_ADMISSION'
   d['tasks'][key]={'run_id':'fm-anatomy-ready-56967269-20261021','PID':os.getpid(),'ctime':ticks(os.getpid()),'argv':Path('/proc/self/cmdline').read_bytes().decode().strip('\0').split('\0'),'active':active,'phase':self.progress['phase'],'cpu_threads':self.cpu if active else 0,'download_workers':self.downloads if active else 0,'convert_workers':1 if active else 0,'reserved_disk_bytes':self.disk if active else 0,'reserved_ram_bytes':self.ram if active else 0,'gpu_indices':self.gpus if active else [],'scratch_paths':[str(ROOT)],'heartbeat_unix':time.time()};write(p,d)
 def __enter__(self):
  t=json.loads((BASE/'TARGET_VERIFIED.json').read_text());assert socket.gethostname()==t['hostname'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==t['boot_id'] and Path('/root/.vast_containerlabel').read_text().strip()=='C.56967269'
  ROOT.mkdir(exist_ok=True);self.lock=(ROOT/'singleton.lock').open('a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB);self.register(True)
  def beat():
   while not self.stop.wait(30):
    try:self.register(True);write(ROOT/'STATE.json',dict(self.progress,PID=os.getpid(),ctime=ticks(os.getpid()),updated_unix=time.time()))
    except Exception as e:self.error=type(e).__name__+': '+str(e);self.progress['abort_reason']=self.error
  self.thread=threading.Thread(target=beat,daemon=True);self.thread.start();return self
 def check(self):
  if self.error:raise RuntimeError(self.error)
 def __exit__(self,ty,exc,tb):
  self.stop.set();self.thread.join()
  if exc:self.progress.update(phase='failed',error_type=type(exc).__name__,error=str(exc))
  write(ROOT/'STATE.json',dict(self.progress,PID=os.getpid(),ctime=ticks(os.getpid()),process_terminal=True,updated_unix=time.time()));self.register(False);self.lock.close()
