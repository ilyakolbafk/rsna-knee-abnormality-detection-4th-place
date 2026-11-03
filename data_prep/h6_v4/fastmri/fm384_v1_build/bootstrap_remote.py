import json,subprocess,sys
from pathlib import Path
from remote import run,ROOT
binding=json.loads((ROOT/'TARGET_VERIFIED.json').read_text())
code='''import pathlib,json,os,fcntl,socket,shutil,time
P=pathlib.Path
binding=PAYLOAD
assert socket.gethostname()==binding['hostname'] and P('/proc/sys/kernel/random/boot_id').read_text().strip()==binding['boot_id']
assert P('/root/.vast_containerlabel').read_text().strip()=='C.56967269'
base=P('/workspace/knee_v4_cache_20261021');shared=base/'shared';shared.mkdir(parents=True,exist_ok=True)
def write(p,d):
 t=p.with_suffix(p.suffix+'.tmp.'+str(os.getpid()))
 with t.open('w') as f:json.dump(d,f,indent=2);f.write('\\n');f.flush();os.fsync(f.fileno())
 os.replace(t,p);fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
with (shared/'admission.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 f=shared/'admission.json'
 if f.exists():
  d=json.loads(f.read_text());assert d['version']==1
  assert int(d['target']['instance_id'])==56967269
 else:
  mi={a:int(b.split()[0])*1024 for a,b in (x.split(':',1) for x in P('/proc/meminfo').read_text().splitlines())}
  cm=int(P('/sys/fs/cgroup/memory/memory.limit_in_bytes').read_text());ram=min(cm,mi['MemTotal']);cp=float(P('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read_text())/float(P('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read_text());total=shutil.disk_usage(base).total
  d={'version':1,'target':{'instance_id':56967269,'machine_id':1569,'host_id':18,'hostname':binding['hostname'],'boot_id':binding['boot_id']},'limits':{'measured_cpu_quota':cp,'max_total_cpu_threads':int(cp/2),'max_total_download_workers':2,'initial_download_workers_per_task':1,'initial_convert_workers_per_task':1,'memory_limit_bytes':ram,'memory_reserve_bytes':max(8*1024**3,int(.15*ram)),'disk_total_bytes':total,'disk_reserve_bytes':max(30*1024**3,int(.10*total))},'tasks':{},'created_unix':time.time()};write(f,d)
 own=base/'fastmri_v1'
 if own.exists():
  owner=own/'OWNER.json';assert owner.exists() and json.loads(owner.read_text())['run_id']=='fm384-v1-56967269-20261021','existing namespace needs reconciliation'
 else:
  own.mkdir(mode=0o700);write(own/'OWNER.json',{'run_id':'fm384-v1-56967269-20261021','owner':'fastmri builder','instance_id':56967269,'created_unix':time.time()})
 for x in ['code','inputs','source_rss','scratch','qa','receipts','private']: (own/x).mkdir(exist_ok=True,mode=0o700)
 write(own/'TARGET_VERIFIED.json',binding)
 print(json.dumps({'shared':d,'namespace':str(own)}))
'''.replace('PAYLOAD',repr(binding))
print(run(code))
