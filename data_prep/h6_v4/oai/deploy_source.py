from common import *
import shlex

b=binding()
code=f'''
from pathlib import Path
import os,json
assert Path('/root/.vast_containerlabel').read_text().strip()=='C.56967269'
assert os.uname().nodename=={b['hostname']!r}
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()=={b['boot_id']!r}
p=Path({REMOTE!r})
if p.exists() and list(p.iterdir()):raise RuntimeError('OWN_NAMESPACE_ALREADY_OCCUPIED_INSPECT_FIRST')
p.mkdir(parents=True,exist_ok=True)
print(json.dumps({{'status':'OWN_NAMESPACE_CREATED','path':str(p)}}))
'''
print(json.dumps(remote_python(code)))
for source,dest in [(RUN/'code',REMOTE+'/code'),(RUN/'TARGET_VERIFIED.json',REMOTE+'/TARGET_VERIFIED.json'),
                    (RUN/'CONTRACT.json',REMOTE+'/CONTRACT.json'),(RUN/'handoff',REMOTE+'/handoff'),
                    (RUN/'SOURCE_COVERAGE.local.json',REMOTE+'/SOURCE_COVERAGE.origin.json')]:
    p=subprocess.run(['scp','-q','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
       '-o',f'UserKnownHostsFile="{RUN}/target_known_hosts"','-i',SSH_KEY,
       '-P',SSH_PORT,'-r',str(source),SSH_HOST+':'+dest],capture_output=True,text=True,timeout=300)
    if p.returncode:raise RuntimeError(f'SCP_EXIT_{p.returncode}: '+p.stderr[-800:])
code=f'''
import subprocess,json,os
from pathlib import Path
p=Path({REMOTE!r});log=(p/'source_service.log').open('a')
proc=subprocess.Popen(['python3',str(p/'code/source_service.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,env={{**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}})
print(json.dumps({{'status':'SOURCE_SERVICE_DISPATCHED','PID':proc.pid}}))
'''
print(json.dumps(remote_python(code)))
