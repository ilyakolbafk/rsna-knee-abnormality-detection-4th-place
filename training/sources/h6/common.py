from pathlib import Path
import hashlib,json,os

ROOT=Path(__file__).resolve().parent
TARGET={'instance':56967269,'machine':1569,'hostname':'56dc3869206e'}
GPU_UUIDS=['GPU-a38ae157-e4d9-b27f-62d8-6883b6a33659','GPU-7923cc48-86fc-6e33-e54b-0e832fc540a0']
ORIENTATION_SPEC=json.loads((ROOT/'assets/ORIENTATION_POLICY.json').read_text())
POLICY={k:ORIENTATION_SPEC[k] for k in ['id','release_manifest_sha256','algorithm_sha256','canonical_target']}
def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.partial.'+str(os.getpid()))
    with tmp.open('w') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path);fd=os.open(str(path.parent),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
def pin_check():
    pins=read(ROOT/'PACKAGE_PINS.json')
    for rel,digest in pins['files'].items():
        if sha(ROOT/rel)!=digest:raise ValueError('PACKAGE_PIN_MISMATCH:'+rel)
    if pins['state'] not in ['PREPARED_CPU_QA','ARMED']:raise ValueError('PACKAGE_NOT_PREPARED')
    return sha(ROOT/'PACKAGE_PINS.json')
