"""Read-only final reconciliation and exact-byte compact metadata copy to Mac."""
from common import *
import base64

if __name__ == '__main__':
    binding()
    result = remote_python(r'''
from pathlib import Path
import sys,json,os,time,fcntl,hashlib,base64,shutil
p=Path('/workspace/knee_v4_cache_20261021/oai_v4');sys.path.insert(0,str(p/'code'))
from common import sha,roster
from source_service import identity
from receiver_service import snapshot
from recipe_registry import collection,validate_audit,recipe_counts,BASE,canonical_hash
identity()
lock=(p/'finalize.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
rows=roster();expected={r['case_id'] for r in rows};assert len(rows)==len(expected)==3254
sets={
 'cache_npz':{q.stem for q in (p/'cache').glob('*.npz')},
 'cache_audit':{q.name.removesuffix('.audit.json') for q in (p/'cache').glob('*.audit.json')},
 'cache_receipts':{q.stem for q in (p/'cache_receipts').glob('*.json')},
 'source_receipts':{q.stem for q in (p/'source_receipts').glob('*.json')}}
assert all(v==expected for v in sets.values()),'EXACT_CASESETS'
assert not list((p/'rejections').glob('*.json'))
assert not list((p/'incoming').glob('*.ready.json')) and not list((p/'incoming').glob('*.processing.json'))
s=snapshot(publish=False)
assert s['expected']==s['received']==s['built']==s['verified']==3254 and s['rejected']==0 and not s['stale_receipts']
f=json.loads((p/'FINAL_MANIFEST.json').read_bytes());data=collection()
assert f['status']=='COMPLETE_ADAPTED_PUBLIC_V4' and f['expected']==f['received']==f['built']==f['verified']==3254 and f['rejected']==0
assert f['collection_semantic_sha256']==data['semantic_sha256'] and f['recipe_counts']==recipe_counts()
records={};npzbytes=wirebytes=delta=0
for row in rows:
 case=row['case_id'];npz=p/'cache'/f'{case}.npz';ap=p/'cache'/f'{case}.audit.json';rp=p/'cache_receipts'/f'{case}.json';sp=p/'source_receipts'/f'{case}.json'
 a=json.loads(ap.read_bytes());rr=json.loads(rp.read_bytes());sr=json.loads(sp.read_bytes());cfg=validate_audit(case,a)
 st=npz.stat();ast=ap.stat()
 assert (st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns,ast.st_ino,ast.st_mtime_ns)==(rr['inode'],rr['bytes'],rr['mtime_ns'],rr['ctime_ns'],rr['audit_inode'],rr['audit_mtime_ns']),case
 assert rr['case_id']==case and rr['recipe_sha256']==cfg['semantic_sha256']
 assert rr['status']=='RECEIVER_FULLSHA_AND_ARRAYS_VERIFIED' and rr['sha256']==a['sha256'] and rr['audit_sha256']==sha(ap) and rr['rows']==a['rows']
 nonce=rr['request_nonce'];assert len(nonce)==32 and all(c in '0123456789abcdef' for c in nonce)
 assert rr['source_objects']==5 and rr['orientation_state']==a['orientation_state']=='canonical_L_v8_3'
 assert len(a['pixel_sha256'])==len(a['metadata_sha256'])==64
 assert sr['case_id']==case and sr['cache_sha256']==rr['sha256'] and sr['audit_sha256']==rr['audit_sha256'] and sr['recipe_sha256']==cfg['semantic_sha256']
 actual={x['family']:(x['source_study_uid'],x['source_series_uid'],x['archive_sha256'],x['archive_bytes']) for x in sr['objects']}
 want={x['object']['family']:(x['source_study_uid'],x['source_series_uid'],x['archive_sha256'],int(x['object']['archive_bytes'])) for x in row['objects']}
 assert len(sr['objects'])==5 and actual==want,'SOURCE_LINEAGE:'+case
 records[case]={'cache_receipt_sha256':sha(rp),'source_receipt_sha256':sha(sp),'npz_sha256':rr['sha256'],'audit_sha256':rr['audit_sha256'],'nonce':nonce}
 npzbytes+=st.st_size;wirebytes+=rr.get('wire_bytes',rr['bytes']);delta+=rr.get('original_npz_reconstructed_fullsha') is True
assert npzbytes==f['cache_bytes']==s['bytes']
base=json.loads((p/'BUILD_CONFIG.frozen.json').read_bytes());assert len(base['component_sha256'])==333
for rel,digest in base['component_sha256'].items():assert sha(p/rel)==digest,rel
for recipe,ref in data['recipes'].items():
 cp=p/ref['config'];cfg=json.loads(cp.read_bytes());assert sha(cp)==ref['config_sha256'] and cfg['semantic_sha256']==recipe
 if recipe!=BASE:
  for rel,digest in cfg['adaptation_component_sha256'].items():assert sha(cp.parent/'components'/rel)==digest,rel
reader=json.loads((p/'COLLECTION_READER_QA.json').read_bytes())
assert reader['status']=='PASS_MIXED_RECIPES_TWO_WORKER_READER_AND_IDENTITY_GUARDS' and set(reader['covered_recipes'])==set(data['recipes'])
assert reader['collection_reader_sha256']==sha(p/'code/collection_reader.py') and reader['recipe_registry_sha256']==sha(p/'code/recipe_registry.py')
pointer=json.loads((p/'COLLECTION_ACTIVE.json').read_bytes())
names=['FINAL_MANIFEST.json','EXPECTED_RECEIVED_BUILT_VERIFIED.json','MISSING_REJECTED.json','STATE.json','COMPONENT_RECEIVER_QA.json','CANARY_QA.frozen.json','COLLECTION_READER_QA.json','COLLECTION_ACTIVE.json',pointer['path']]
files={n:{'base64':base64.b64encode((p/n).read_bytes()).decode(),'sha256':sha(p/n)} for n in names}
qa={'status':'PASS_FINAL_EXACT3254_CASESETS_RECEIPT_FULLSHA_ARRAY_AUDIT_SOURCE_LINEAGE_FRESH_STAT_AND_ALL_QA','unix':time.time(),
 'case_set_counts':{k:len(v) for k,v in sets.items()},'verified':3254,'missing':[],'rejected':[],'stale':[],
 'source_archive_objects':3254*5,'cache_npz_bytes':npzbytes,'receipt_wire_bytes_or_original_bytes':wirebytes,'delta_reconstructed_fullsha_cases':delta,
 'collection_semantic_sha256':data['semantic_sha256'],'recipe_counts':recipe_counts(),'geometry_cases':len(data['case_qa']),
 'base_333_components':'PASS_FRESH_FULLSHA','active_frozen_adaptations':'PASS_FRESH_FULLSHA','mixed_two_worker_reader':'PASS',
 'verification_basis':'receiver_fullSHA_and_array_receipts_with_fresh_nonce_inode_stat_and_audit_SHA;no_new_allNPZ_rescan',
 'record_manifest_sha256':canonical_hash(records),'file_sha256':{n:x['sha256'] for n,x in files.items()},
 'finalizer_sha256':sha(p/'code/finalize.py'),'receiver_owner':json.loads((p.parent/'shared/admission.json').read_bytes())['tasks']['oai_v4'],
 'incoming_ready':0,'incoming_processing':0,'disk_free_bytes':shutil.disk_usage(p).free}
print(json.dumps({'qa':qa,'files':files,'records':records}))
''',timeout=120)
    out=RUN/'final_handoff';out.mkdir(exist_ok=True)
    for rel,entry in result['files'].items():
        data=base64.b64decode(entry['base64']);assert hashlib.sha256(data).hexdigest()==entry['sha256']
        q=out/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(data);assert sha(q)==entry['sha256']
    assert sha(RUN/'FINAL_MANIFEST.json')==sha(out/'FINAL_MANIFEST.json')
    dump(out/'RECEIPT_IDENTITY_MANIFEST.json',result['records'])
    dump(out/'FINAL_INTEGRITY_QA.json',result['qa']);dump(RUN/'FINAL_INTEGRITY_QA.json',result['qa'])
    print(json.dumps({k:v for k,v in result['qa'].items() if k!='receiver_owner'}))
