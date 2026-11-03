"""Synchronize reviewed metadata/source closures; active collection pointer last."""
from common import *
from recipe_registry import collection
import fcntl
import base64

def sync():
    with (RUN/'collection_sync.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        data=collection();assert data is not None
        pointer=json.loads((RUN/'COLLECTION_ACTIVE.json').read_text())
        names={'COLLECTION_READER_QA.json',pointer['path']}
        for entry in data['recipes'].values():
            names.add(entry['config']);cp=RUN/entry['config'];cfg=json.loads(cp.read_text())
            for rel in cfg.get('adaptation_component_sha256',{}):names.add(str((cp.parent/'components'/rel).relative_to(RUN)))
        for ref in data['case_qa'].values():names.add(ref['path'])
        payload={n:{'base64':base64.b64encode((RUN/n).read_bytes()).decode('ascii'),'sha256':sha(RUN/n)} for n in sorted(names)}
        b=binding()
        result=remote_python(f'''
from pathlib import Path
import json,os,sys,fcntl,hashlib,base64
p=Path({REMOTE!r});sys.path.insert(0,str(p/'code'));from common import dump
assert Path('/root/.vast_containerlabel').read_text().strip()=='C.56967269'
assert os.uname().nodename=={b['hostname']!r} and Path('/proc/sys/kernel/random/boot_id').read_text().strip()=={b['boot_id']!r}
with (p/'collection_sync.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 for name,record in {payload!r}.items():
  q=p/name;assert q.resolve().is_relative_to(p.resolve());q.parent.mkdir(parents=True,exist_ok=True)
  value=base64.b64decode(record['base64'],validate=True)
  assert hashlib.sha256(value).hexdigest()==record['sha256']
  temp=q.with_name(q.name+'.sync.writing');temp.write_bytes(value);os.replace(temp,q)
  assert hashlib.sha256(q.read_bytes()).hexdigest()==record['sha256']
 dump(p/'COLLECTION_ACTIVE.json',{pointer!r})
print(json.dumps({{'status':'COLLECTION_METADATA_SHA_VERIFIED_POINTER_LAST','collection_sha256':{data['semantic_sha256']!r},'files':{len(payload)}}}))
''',timeout=60)
        dump(RUN/'geometry_v2/LAST_COLLECTION_SYNC.json',result);return result

if __name__=='__main__':print(json.dumps(sync()))
