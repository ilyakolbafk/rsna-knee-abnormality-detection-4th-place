import base64,json
from remote import run,ROOT
payload=base64.b64encode((ROOT/'finish_pipeline.py').read_bytes()).decode()
code='''import json,base64,os,subprocess,time,fcntl,hashlib
from pathlib import Path
r=Path('/workspace/knee_v4_cache_20261021/fastmri_v1');data=base64.b64decode(PAYLOAD);p=r/'code/finish_pipeline.py'
if p.exists():assert p.read_bytes()==data,'Different launched finalizer needs explicit version'
else:p.write_bytes(data)
lock=(r/'finish_controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.close()
assert json.loads((r/'qa/FINALIZER_RESUME_TESTS.json').read_text())['returncode']==0
with (r/'finish.log').open('ab',buffering=0) as log:
 child=subprocess.Popen([str(r/'render_venv/bin/python'),'-u',str(p)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
rec={'PID':child.pid,'ctime':Path('/proc/%s/stat'%child.pid).read_text().rsplit(')',1)[1].split()[19],'argv':[str(r/'render_venv/bin/python'),'-u',str(p)],'code_sha256':hashlib.sha256(data).hexdigest(),'unix':time.time(),'purpose':'waitsource1172;fullSHA;realcanary;3-fieldcandidate;loaderQA;no training;semanticgateunresolved'};(r/'FINISHER_LAUNCH.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec))
'''.replace('PAYLOAD',repr(payload))
s=run(code);(ROOT/'FINISHER_LAUNCH.json').write_text(s);print(s)
