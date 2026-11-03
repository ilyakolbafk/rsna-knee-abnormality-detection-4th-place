import base64,hashlib,json,subprocess
from pathlib import Path
from remote import run,ROOT,ARGV
h=Path('.codex/runs/oai_fastmri_handoff_56967269_20261021/handoff')
files={'code/source_collect.py':ROOT/'source_collect.py','inputs/knee.csv':h/'reference/fastmri/annotations_spatial_valid/knee.csv','inputs/knee_file_list.csv':h/'reference/fastmri/annotations_spatial_valid/knee_file_list.csv','inputs/quarantined_rows.json':h/'reference/fastmri/annotations_spatial_valid/quarantined_rows.json','inputs/knee.original.csv':h/'reference/fastmri/annotations_original/knee.csv','inputs/FM_CLASSES.frozen.json':h/'inputs/FM_CLASSES.frozen.json','inputs/TASK_CONTRACT.json':ROOT/'TASK_CONTRACT.json','inputs/SOURCE_EVIDENCE.yaml':ROOT/'SOURCE_EVIDENCE.yaml'}
payload={k:base64.b64encode(p.read_bytes()).decode() for k,p in files.items()}
code='''import json,base64,hashlib,os
from pathlib import Path
root=Path('/workspace/knee_v4_cache_20261021/fastmri_v1');assert json.loads((root/'OWNER.json').read_text())['run_id']=='fm384-v1-56967269-20261021'
files=PAYLOAD
out={}
for name,b64 in files.items():
 p=root/name;data=base64.b64decode(b64)
 if p.exists():assert p.read_bytes()==data,'DEPLOY_DIFFERENT_EXISTING_BYTES'
 else:
  with p.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 out[name]={'bytes':len(data),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
print(json.dumps(out))
'''.replace('PAYLOAD',repr(payload))
result=json.loads(run(code));(ROOT/'DEPLOYMENT_RECEIPT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
# Private source links copied independently via stdin. Never part of package/manifest/logs.
links=Path('.codex/runs/fastmri-acquisition-20261020/nyu_private_links.json').read_text()
expected={'train':'8bd70a1ec71a71e0042eea3827e3b14e137108d0b27555cf225286b27ef565d6','val':'688c841f03ebbeab5f7f2ba2fa3fa7d3ed910d9e15ffe461c78c73dee7a4236b'}
assert {k:hashlib.sha256(v.encode()).hexdigest() for k,v in json.loads(links).items()}==expected
cmd="umask 077; cat > /workspace/knee_v4_cache_20261021/fastmri_v1/private/links.json"
x=subprocess.run(ARGV+[cmd],input=links,text=True,capture_output=True,timeout=30);assert x.returncode==0,'private link transfer failed';print('private source link delivery: two email-verified links; public output redacted')
