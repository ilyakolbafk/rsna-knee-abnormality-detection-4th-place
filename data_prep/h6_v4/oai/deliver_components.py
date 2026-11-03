from common import *
import tarfile,shlex

if __name__=='__main__':
    bundle=RUN/'source_components.tar.gz'
    with tarfile.open(bundle,'w:gz',compresslevel=1) as tar:
        for p in sorted((RUN/'component_discovery/source').rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':
                assert not p.is_symlink();tar.add(p,arcname=str(p.relative_to(RUN)))
        for p in sorted((RUN/'code').glob('*.py')):tar.add(p,arcname='code/'+p.name)
        for n in ['BUILD_CONFIG.frozen.json','CANARY_QA.frozen.json','RESOURCE_PLAN.json','ADAPTATION_DIFF.json','CONTRACT.json']:
            tar.add(RUN/n,arcname=n)
    p=subprocess.run(['rsync','-a','--partial','--timeout=120','-e',shlex.join(SSH[:-1]),str(bundle),f'{SSH_HOST}:{REMOTE}/source_components.tar.gz'],capture_output=True,text=True,timeout=600)
    if p.returncode:raise RuntimeError('COMPONENT_RSYNC_EXIT_'+str(p.returncode))
    b=binding()
    result=remote_python(f'''
from pathlib import Path
import json,os,tarfile,hashlib,sys,time
p=Path({REMOTE!r})
assert Path('/root/.vast_containerlabel').read_text().strip()=='C.56967269' and os.uname().nodename=={b['hostname']!r}
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()=={b['boot_id']!r}
bundle=p/'source_components.tar.gz';h=hashlib.sha256(bundle.read_bytes()).hexdigest();assert h=={sha(bundle)!r}
with tarfile.open(bundle) as t:
 for m in t.getmembers():
  q=Path(m.name);assert m.isfile() and not q.is_absolute() and '..' not in q.parts
 t.extractall(p)
sys.path.insert(0,str(p/'code'));from common import sha,dump,roster
cfg=json.loads((p/'BUILD_CONFIG.frozen.json').read_text())
for rel,expected in cfg['component_sha256'].items():assert sha(p/rel)==expected,rel
result={{'status':'PASS_ALL_FROZEN_COMPONENT_SHA','files':len(cfg['component_sha256']),'recipe_sha256':cfg['semantic_sha256'],'bundle_sha256':h,'unix':time.time(),'private_hrepro_parity':False}}
dump(p/'COMPONENT_RECEIVER_QA.json',result)
byid={{r['case_id']:r for r in roster()}}
for rp in (p/'cache_receipts').glob('*.json'):
 r=json.loads(rp.read_text());case=r['case_id'];out=p/'source_receipts'/f'{{case}}.json'
 if out.exists():continue
 ap=p/'cache'/f'{{case}}.audit.json';assert sha(ap)==r['audit_sha256'];a=json.loads(ap.read_text());row=byid[case]
 dump(out,{{'case_id':case,'status':'SOURCE_FULLSHA_VERIFIED_ON_MAC','subjectkey':row['subjectkey'],'side':row['side'],'visit':row['visit'],
  'recipe_sha256':cfg['semantic_sha256'],'cache_sha256':r['sha256'],'audit_sha256':r['audit_sha256'],'raw_bytes_present_on_target':False,
  'objects':[{{'family':s['family'],'source_study_uid':s['source_study_uid'],'source_series_uid':s['series_uid'],'archive_bytes':s['archive_bytes'],'archive_sha256':s['archive_sha256']}} for s in a['audit']['lineage']], 'unix':time.time()}})
print(json.dumps(result))
''',timeout=120)
    dump(RUN/'COMPONENT_RECEIVER_QA.json',result)
    bundle.unlink()
    print(json.dumps(result))
