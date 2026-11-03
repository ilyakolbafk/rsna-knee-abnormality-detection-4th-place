"""Fresh read-only target proof for every activated geometry case."""
from common import *
from recipe_registry import collection

def check():
    cases=list(collection()['case_qa'])
    result=remote_python(f'''
import sys,json,time
from pathlib import Path
p=Path({REMOTE!r});sys.path.insert(0,str(p/'code'))
from common import sha
from source_service import identity
from receiver_service import snapshot
from recipe_registry import validate_audit
identity();checks={{}}
for c in {cases!r}:
 rp=p/'cache_receipts'/f'{{c}}.json';sp=p/'source_receipts'/f'{{c}}.json';npz=p/'cache'/f'{{c}}.npz';ap=p/'cache'/f'{{c}}.audit.json'
 if not all(q.exists() for q in [rp,sp,npz,ap]):checks[c]={{'status':'PENDING'}};continue
 receipt=json.loads(rp.read_text());a=json.loads(ap.read_text());validate_audit(c,a)
 st=npz.stat();ast=ap.stat()
 assert (st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)==(receipt['inode'],receipt['bytes'],receipt['mtime_ns'],receipt['ctime_ns'])
 assert (ast.st_ino,ast.st_mtime_ns)==(receipt['audit_inode'],receipt['audit_mtime_ns'])
 assert sha(npz)==receipt['sha256']==a['sha256'] and sha(ap)==receipt['audit_sha256']
 assert len(receipt['request_nonce'])==32
 checks[c]={{'status':'PASS_FRESH_FULLSHA_NONCE_STAT_AND_CASE_RECIPE_QA','sha256':receipt['sha256'],'recipe_sha256':receipt['recipe_sha256'],'nonce':receipt['request_nonce'],'bytes':receipt['bytes'],'inode':st.st_ino,'source_receipt_sha256':sha(sp),'verified_unix':time.time()}}
print(json.dumps({{'checks':checks,'state':snapshot(publish=False),'unix':time.time()}}))
''',timeout=60)
    dump(RUN/'geometry_v2/DELIVERY_QA.json',result)
    return {'passed':[c for c,q in result['checks'].items() if q['status'].startswith('PASS')],'pending':[c for c,q in result['checks'].items() if q['status']=='PENDING'],'target_state':result['state']}

if __name__=='__main__':print(json.dumps(check()))
