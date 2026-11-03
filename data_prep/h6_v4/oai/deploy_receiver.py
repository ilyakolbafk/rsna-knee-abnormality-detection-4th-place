from common import *
import tarfile

b=binding()
bundle=RUN/'receiver_bundle.tar.gz'
names=['TARGET_VERIFIED.json','CONTRACT.json','BUILD_CONFIG.frozen.json','CANARY_QA.frozen.json','RESOURCE_PLAN.json','ADAPTATION_DIFF.json']
with tarfile.open(bundle,'w:gz') as tar:
    for p in sorted((RUN/'code').glob('*.py')):tar.add(p,arcname='code/'+p.name)
    for n in names:tar.add(RUN/n,arcname=n)
    for n in ['GEOMETRY_BUILD_CONFIG.frozen.json','COLLECTION_ACTIVE.json','COLLECTION_READER_QA.json']:
        if (RUN/n).exists():tar.add(RUN/n,arcname=n)
    for folder in ['collections','geometry_v2/recipes','geometry_v2/qa','geometry_v2/geometry_composite_registration','registration_recovery/source']:
        for p in sorted((RUN/folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:tar.add(p,arcname=str(p.relative_to(RUN)))
print(json.dumps(remote_python(f'''
from pathlib import Path
import os,json,time,signal,fcntl,sys
p=Path({REMOTE!r});assert Path('/root/.vast_containerlabel').read_text().strip()=='C.56967269'
assert os.uname().nodename=={b['hostname']!r}
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()=={b['boot_id']!r}
reg=json.loads((p.parent/'shared/admission.json').read_text());own=reg['tasks']['oai_v4'];pid=own['PID']
assert own['argv'][-1] in [str(p/'code/source_service.py'),str(p/'code/receiver_service.py')]
if Path(f'/proc/{{pid}}/stat').exists():
 assert Path(f'/proc/{{pid}}/stat').read_text().split()[21]==own['ctime']
 os.kill(pid,signal.SIGTERM)
for _ in range(180):
 try:
  with (p/'singleton.lock').open('a') as lk:fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
  break
 except BlockingIOError:time.sleep(.25)
else:raise RuntimeError('OWN_SERVICE_LOCK_STILL_HELD')
with (p.parent/'shared/admission.lock').open('a') as lk:
 fcntl.flock(lk,fcntl.LOCK_EX);reg=json.loads((p.parent/'shared/admission.json').read_text())
 assert reg['tasks']['oai_v4']['PID']==pid
 reg.setdefault('task_history',[]).append({{'task':'oai_v4','reason':'own_runtime_upgrade_preserving_pixels','prior':reg['tasks']['oai_v4'],'unix':time.time()}})
 reg['tasks']['oai_v4'].update(active=False,reserved_disk_bytes=0,reserved_ram_bytes=0,cpu_threads=0,download_workers=0,convert_workers=0)
 sys.path.insert(0,str(p/'code'));from common import dump
 dump(p.parent/'shared/admission.json',reg)
print(json.dumps({{'status':'OWN_RECEIVER_SERVICE_DRAINED','PID':pid,'neighbor_unchanged':True}}))
''')))
p=subprocess.run(['scp','-q','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o',f'UserKnownHostsFile="{RUN}/target_known_hosts"',
 '-i',SSH_KEY,'-P',SSH_PORT,str(bundle),f'{SSH_HOST}:{REMOTE}/receiver_bundle.tar.gz'],capture_output=True,text=True,timeout=120)
if p.returncode:raise RuntimeError(p.stderr[-1000:])
code=f'''
from pathlib import Path
import os,json,tarfile,subprocess,hashlib
p=Path({REMOTE!r});bundle=p/'receiver_bundle.tar.gz'
assert hashlib.sha256(bundle.read_bytes()).hexdigest()=={sha(bundle)!r}
with tarfile.open(bundle) as t:
 for m in t.getmembers():
  q=Path(m.name)
  assert not q.is_absolute() and '..' not in q.parts and m.isfile()
 t.extractall(p)
log=(p/'receiver_service.log').open('a')
proc=subprocess.Popen(['python3',str(p/'code/receiver_service.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,
env={{**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}})
print(json.dumps({{'status':'CACHE_RECEIVER_DISPATCHED','PID':proc.pid}}))
'''
print(json.dumps(remote_python(code)))
