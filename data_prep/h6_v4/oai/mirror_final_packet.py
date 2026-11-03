"""Mirror only owned compact metadata, preserving exact source bytes and SHA."""
from common import *
import base64

binding()
paths=[p for p in (RUN/'final_handoff').rglob('*') if p.is_file()]
paths += [RUN/n for n in ['ledger.md','README.md','NEXT_REPAIR_PLAN.md','ACTIVE_OPERATION.json','MONITOR_LAST.json','AUTONOMY_APPROVAL_AUDIT.json','FINAL_INTEGRITY_QA.json','RUNTIME_RETIREMENT_QA.json','geometry_v2/RECIPE_QA.json','geometry_v2/DELIVERY_QA.json','code/final_handoff_check.py','code/prepare_final_packet.py','code/mirror_final_packet.py']]
payload=[{'path':str(p.relative_to(RUN)),'base64':base64.b64encode(p.read_bytes()).decode(),'sha256':sha(p)} for p in paths]
result=remote_python(f'''
from pathlib import Path
import sys,json,base64,hashlib,uuid,os,time
p=Path({REMOTE!r});sys.path.insert(0,str(p/'code'))
from common import sha,dump
from source_service import identity
identity();out={{}}
for v in {payload!r}:
 q=p/v['path'];assert q.resolve().is_relative_to(p.resolve())
 b=base64.b64decode(v['base64']);assert hashlib.sha256(b).hexdigest()==v['sha256']
 q.parent.mkdir(parents=True,exist_ok=True);tmp=q.with_name(q.name+'.'+uuid.uuid4().hex+'.writing')
 with tmp.open('wb') as fp:fp.write(b);fp.flush();os.fsync(fp.fileno())
 os.replace(tmp,q);assert sha(q)==v['sha256'];out[v['path']]=v['sha256']
assert json.loads((p/'FINAL_MANIFEST.json').read_bytes())['verified']==3254
assert sha(p/'FINAL_MANIFEST.json')==sha(p/'final_handoff/FINAL_MANIFEST.json')
qa={{'status':'PASS_EXACT_BINARY_FINAL_PACKET_MAC_TARGET_MIRROR','unix':time.time(),'files':out,'count':len(out)}}
dump(p/'HANDOFF_MIRROR_QA.json',qa);print(json.dumps(qa))
''',timeout=120)
dump(RUN/'HANDOFF_MIRROR_QA.json',result)
print(json.dumps({'status':result['status'],'count':result['count'],'unix':result['unix']}))
