"""Fail closed on exact case sets, every recipe and QA before publishing COMPLETE."""
from common import *
import fcntl

if __name__=='__main__':
    lock=(RUN/'finalize.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    b=binding()
    result=remote_python(f'''
from pathlib import Path
import sys,json,os,time,fcntl
p=Path({REMOTE!r});sys.path.insert(0,str(p/'code'))
from common import dump,roster,sha
from receiver_service import snapshot
from recipe_registry import collection,config_for_case,validate_audit,recipe_counts,BASE
assert Path('/root/.vast_containerlabel').read_text().strip()=='C.56967269'
assert os.uname().nodename=={b['hostname']!r} and Path('/proc/sys/kernel/random/boot_id').read_text().strip()=={b['boot_id']!r}
lock=(p/'finalize.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
rows=roster();expected={{r['case_id'] for r in rows}}
for folder in ['cache_receipts','source_receipts']:
 assert {{f.stem for f in (p/folder).glob('*.json')}}==expected,'EXACT_RECEIPT_CASE_SET_CONFLICT:'+folder
assert {{f.stem for f in (p/'cache').glob('*.npz')}}==expected,'EXACT_CACHE_CASE_SET_CONFLICT'
assert {{f.name.removesuffix('.audit.json') for f in (p/'cache').glob('*.audit.json')}}==expected,'EXACT_AUDIT_CASE_SET_CONFLICT'
assert not list((p/'rejections').glob('*.json')),'REJECTIONS_REMAIN'
s=snapshot(publish=False);assert s['status']=='COMPLETE_ADAPTED_PUBLIC_V4' and s['expected']==s['received']==s['built']==s['verified']==3254 and not s['stale_receipts'] and s['rejected']==0
assert json.loads((p/'COMPONENT_RECEIVER_QA.json').read_text())['status']=='PASS_ALL_FROZEN_COMPONENT_SHA'
base=json.loads((p/'BUILD_CONFIG.frozen.json').read_text())
for name,digest in base['component_sha256'].items():assert sha(p/name)==digest,'FROZEN_BASE_COMPONENT_CHANGED:'+name
data=collection();recipes={{BASE:{{'config':'BUILD_CONFIG.frozen.json'}}}} if data is None else data['recipes']
for recipe,entry in recipes.items():
 cp=p/entry['config'];cfg=json.loads(cp.read_text());assert cfg['semantic_sha256']==recipe
 if recipe!=BASE:
  for name,digest in cfg['adaptation_component_sha256'].items():assert sha(cp.parent/'components'/name)==digest,'FROZEN_ADAPTATION_COMPONENT_CHANGED:'+name
for row in rows:
 case=row['case_id'];a=json.loads((p/'cache'/f'{{case}}.audit.json').read_text());config=validate_audit(case,a)
 receipt=json.loads((p/'cache_receipts'/f'{{case}}.json').read_text());assert receipt['case_id']==case and receipt['recipe_sha256']==config['semantic_sha256']
 assert receipt['audit_sha256']==sha(p/'cache'/f'{{case}}.audit.json')
 source=json.loads((p/'source_receipts'/f'{{case}}.json').read_text());assert source['case_id']==case
 actual={{x['family']:(x['source_study_uid'],x['source_series_uid'],x['archive_sha256'],x['archive_bytes']) for x in source['objects']}}
 want={{x['object']['family']:(x['source_study_uid'],x['source_series_uid'],x['archive_sha256'],int(x['object']['archive_bytes'])) for x in row['objects']}}
 assert len(source['objects'])==5 and actual==want,'FINAL_SOURCE_LINEAGE_CONFLICT:'+case
manifest={{'summary':s,'rows':[{{'case_id':r['case_id'],'expected':True,'received':True,'built':True,'verified':True,'rejected':False,'recipe_sha256':config_for_case(r['case_id'])['semantic_sha256']}} for r in rows]}}
missing={{'missing':[],'rejected':[]}}
r={{'status':'COMPLETE_ADAPTED_PUBLIC_V4','expected':3254,'received':3254,'built':3254,'verified':3254,'rejected':0,'cache_bytes':s['bytes'],
 'private_hrepro_parity':False,'format':'oai-v4-collection-v1','orientation_state':'canonical_L_v8_3','raw_processing_location':'Mac','raw_bytes_on_target':False,
 'collection_semantic_sha256':None if data is None else data['semantic_sha256'],'recipe_counts':recipe_counts(),'recipes':recipes,'unix':time.time()}}
dump(p/'EXPECTED_RECEIVED_BUILT_VERIFIED.json',manifest);dump(p/'MISSING_REJECTED.json',missing)
dump(p/'FINAL_MANIFEST.json',r)
print(json.dumps({{'result':r,'manifest':manifest,'missing':missing}}))
''',timeout=120)
    assert not result['missing']['missing'] and not result['missing']['rejected']
    for name,key in [('EXPECTED_RECEIVED_BUILT_VERIFIED.final.json','manifest'),('MISSING_REJECTED.final.json','missing'),('FINAL_MANIFEST.json','result')]:dump(RUN/name,result[key])
    print(json.dumps(result['result']))
