from remote import run
print(run("""import subprocess,json,os,time
from pathlib import Path
r=Path('/workspace/knee_v4_cache_20261021/fastmri_v1')
assert not list((r/'source_rss').glob('file*/receipt.json')),'Canary already exists; inspect state'
with (r/'canary.log').open('ab',buffering=0) as f:
 p=subprocess.Popen([str(r/'venv/bin/python'),'-u',str(r/'code/source_collect.py'),'--canary-per-split','1'],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
print(json.dumps({'PID':p.pid,'ctime':Path('/proc/%s/stat'%p.pid).read_text().rsplit(')',1)[1].split()[19],'argv':[str(r/'venv/bin/python'),'-u',str(r/'code/source_collect.py'),'--canary-per-split','1'],'launch_unix':time.time()}))
"""))
