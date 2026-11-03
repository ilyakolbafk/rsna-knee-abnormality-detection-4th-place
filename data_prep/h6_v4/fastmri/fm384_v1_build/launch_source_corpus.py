import json
from remote import run,ROOT
code="""import subprocess,json,os,time,fcntl,hashlib
from pathlib import Path
r=Path('/workspace/knee_v4_cache_20261021/fastmri_v1');lock=(r/'singleton.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
s=json.loads((r/'STATE.json').read_text());assert s['phase']=='canary_source_complete' and s['committed']==2
assert json.loads((r/'qa/SOURCE_STAGE_TESTS_V4.json').read_text())['returncode']==0
cs=hashlib.sha256((r/'code/source_collect.py').read_bytes()).hexdigest()
for p in (r/'source_rss').glob('*/receipt.json'):
 b=json.loads(p.read_text())['binding'];assert b['policy']=='official-rss-lossless-source-stage-v4' and b['code_sha256']==cs
old=json.loads((r.parent/'shared/admission.json').read_text())['tasks']['fastmri_v1'];assert old['active'] is False
lock.close()
with (r/'source.log').open('ab',buffering=0) as f:
 p=subprocess.Popen([str(r/'venv/bin/python'),'-u',str(r/'code/source_collect.py')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
launch={'phase':'source_corpus_launch_pending','PID':p.pid,'ctime':Path('/proc/%s/stat'%p.pid).read_text().rsplit(')',1)[1].split()[19],'argv':[str(r/'venv/bin/python'),'-u',str(r/'code/source_collect.py')],'launch_unix':time.time(),'code_sha256':cs,'goal':'1172 complete RSS sources; finalfm384 blocked until valid producer/adaptation','ready':False}
t=r/'LAUNCH_SOURCE_CORPUS.tmp';t.write_text(json.dumps(launch,indent=2)+'\\n');os.replace(t,r/'LAUNCH_SOURCE_CORPUS.json')
print(json.dumps(launch))
"""
s=run(code);(ROOT/'LAUNCH_SOURCE_CORPUS.json').write_text(s);print(s)
