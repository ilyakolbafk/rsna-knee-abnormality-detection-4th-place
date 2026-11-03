"""Terminal readback; reuses verified SHA evidence only with unchanged file stats."""
import json
from remote import ROOT,run
out=json.loads(run('''import json,time,socket,hashlib
from pathlib import Path
r=Path('/workspace/knee_v4_cache_20261021/fastmri_v1')
def read(n):return json.loads((r/n).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def live(v):
 p=Path('/proc/%s'%v['PID'])
 try:return p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]==str(v['ctime']) and '\\nState:\\tZ' not in p.joinpath('status').read_text()
 except FileNotFoundError:return False
t=read('TARGET_VERIFIED.json');assert socket.gethostname()==t['hostname'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==t['boot_id'];assert Path('/root/.vast_containerlabel').read_text().strip()=='C.56967269'
assert not live(read('LAUNCH_SOURCE_CORPUS.json')) and not live(read('FINISHER_LAUNCH.json'))
a=json.loads((r.parent/'shared/admission.json').read_text())['tasks']['fastmri_v1'];assert not a['active'] and a['cpu_threads']==a['download_workers']==a['convert_workers']==a['reserved_disk_bytes']==a['reserved_ram_bytes']==0
s=read('STATE.json');assert s['process_terminal'] and s['phase']=='source_and_candidate_verified_anatomical_gate_unresolved'
f=read('FINAL_STATUS.json');assert not f['training_ready'] and not f['goal_complete']
assert f['source']['actual_volumes']==1172 and f['source']['source_split_volumes']=={'train':973,'val':199}
assert f['candidate_QA']['state']=='PASS_STRUCTURAL_AND_TARGET_QA_ONLY'
p=r/'candidate_pixel_store';rc=read('candidate_pixel_store/COMPLETION_RECEIPT.json')
for n,v in rc['files'].items():
 st=(p/n).stat();assert all(getattr(st,k)==v['stat'][key] for k,key in [('st_ino','inode'),('st_size','bytes'),('st_mtime_ns','mtime_ns'),('st_ctime_ns','ctime_ns')])
assert sha(p/'COMPLETION_RECEIPT.json')==f['candidate_receipt_sha256']
assert sha(r/'receipts/SOURCE_FULLSHA_MANIFEST.json')==f['source_manifest_sha256']
assert sha(r/'inputs/FM_ADAPTATION_SPEC.json')==f['adaptation_spec_sha256']
source=read('receipts/SOURCE_FULLSHA_MANIFEST.json');sr=read('receipts/SOURCE_VERIFICATION.json');assert source['report']==sr==f['source']
complete=read('receipts/SOURCE_VERIFICATION_COMPLETE.json')
assert complete=={'source_terminal_receipt_sha256':sha(r/'receipts/SOURCE_TERMINAL_RECEIPT.json'),'source_manifest_sha256':sha(r/'receipts/SOURCE_FULLSHA_MANIFEST.json'),'source_verification_sha256':sha(r/'receipts/SOURCE_VERIFICATION.json'),'verifier_code_sha256':sha(r/'code/verify_source.py')}
assert sr['source_code_sha256']==sha(r/'code/source_collect.py')
terminal_source=read('receipts/SOURCE_TERMINAL_RECEIPT.json')['binding'];assert terminal_source['source_launch_sha256']==sha(r/'LAUNCH_SOURCE_CORPUS.json') and terminal_source['source_code_sha256']==sr['source_code_sha256']
for sp in ['train','val']:
 proof_path=r/'receipts'/('ARCHIVE_COMPLETE_'+sp+'.json');assert sha(proof_path)==terminal_source['archive_proof_sha256'][sp]
 assert {k:v for k,v in json.loads(proof_path.read_text()).items() if k!='stems'}==sr['archive_proofs'][sp]
for vol in source['volumes']:
 base=r/'source_rss'/vol['stem'];assert sha(base/'receipt.json')==vol['receipt_sha256']
 for n,v in vol['files'].items():
  st=(base/n).stat();assert all(getattr(st,k)==v['stat'][key] for k,key in [('st_ino','inode'),('st_size','bytes'),('st_mtime_ns','mtime_ns'),('st_ctime_ns','ctime_ns')]),str(base/n)
canary=read('qa/REAL_CANARY_QA.json');assert canary['records_sha256']==sha(r/'qa/real_canary/records.json')
assert canary['binding']['source_manifest_sha256']==f['source_manifest_sha256'] and canary['binding']['adaptation_spec_sha256']==f['adaptation_spec_sha256']
for n,v in canary['binding']['code_sha256'].items():assert sha(r/'code'/n)==v
for v in canary['source_flipud_display_and_views_montages']:assert sha(Path(v['path']))==v['sha256'] and Path(v['path']).stat().st_size==v['bytes']
assert not [p for p in (r/'scratch').iterdir() if p.name.endswith(('.h5','.h5.partial','.txn'))]
print(json.dumps({'state':'PASS_TERMINAL_RECEIPT_AND_UNCHANGED_STATS','target':56967269,'source_volumes':1172,'source_and_finisher_terminal':True,'own_admission_released':True,'all_source_receipt_SHA_and_file_stats_unchanged':True,'source_verification_completion_binding_checked':True,'source_manifest_sha256':f['source_manifest_sha256'],'candidate_receipt_sha256':f['candidate_receipt_sha256'],'candidate_files':rc['files'],'training_ready':False,'anatomical_gate':'UNRESOLVED','verified_unix':time.time()}))
'''))
(ROOT/'TERMINAL_READBACK.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='candidate_files'}))
