"""Copy an explicit metadata-only final bundle; never MRI arrays or montages."""
import hashlib,json,subprocess,tarfile
from pathlib import Path
from remote import ROOT,ARGV,run

required=['FINAL_STATUS.json','STATE.json','ledger.md','receipts/ARCHIVE_COMPLETE_train.json','receipts/ARCHIVE_COMPLETE_val.json','receipts/SOURCE_TERMINAL_RECEIPT.json','receipts/SOURCE_FULLSHA_MANIFEST.json','receipts/SOURCE_VERIFICATION.json','receipts/SOURCE_VERIFICATION_COMPLETE.json','qa/REAL_CANARY_QA.json','qa/real_canary/records.json','qa/LOADER_TARGET_QA.json','candidate_pixel_store/BUILD_BINDING.json','candidate_pixel_store/index.json','candidate_pixel_store/boxes.json','candidate_pixel_store/manifest.json','candidate_pixel_store/COMPLETION_RECEIPT.json','inputs/FM_ADAPTATION_SPEC.json']
optional=['PROGRESS.json','FINISHER_LAUNCH.json','LAUNCH_SOURCE_CORPUS.json','qa/SOURCE_STAGE_TESTS_V4.json','qa/FINALIZER_INTEGRATION_TESTS.json','qa/FINALIZER_RESUME_TESTS.json']
code='''import json,tarfile,hashlib,os
from pathlib import Path
r=Path('/workspace/knee_v4_cache_20261021/fastmri_v1');required=REQUIRED;optional=OPTIONAL
for n in required:assert (r/n).is_file(),n
state=json.loads((r/'STATE.json').read_text());assert state.get('process_terminal') and state['phase']=='source_and_candidate_verified_anatomical_gate_unresolved'
paths=required+[n for n in optional if (r/n).is_file()];files={}
for n in paths:
 p=r/n;assert p.resolve().is_relative_to(r) and not p.is_symlink() and p.stat().st_size<64*1024**2
 assert not any(x in n.lower() for x in ['private','links','password','token','.npy','.png','.jpg'])
 files[n]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
manifest=r/'qa/COMPACT_METADATA_MANIFEST.json';manifest.write_text(json.dumps({'scope':'Explicit metadata allowlist; no MRI pixels, source links or credentials','files':files},indent=2))
out=r/'qa/compact_report_bundle.tar.gz'
with tarfile.open(out,'w:gz') as t:
 for n in paths:t.add(r/n,arcname=n,recursive=False)
 t.add(manifest,arcname='COMPACT_METADATA_MANIFEST.json',recursive=False)
assert out.stat().st_size<64*1024**2
print(json.dumps({'path':str(out),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'members':len(paths)+1}))
'''.replace('REQUIRED',repr(required)).replace('OPTIONAL',repr(optional))
meta=json.loads(run(code,60));bundle=ROOT/'server_metadata.tar.gz'
with bundle.open('wb') as f:
 p=subprocess.run(ARGV+['cat /workspace/knee_v4_cache_20261021/fastmri_v1/qa/compact_report_bundle.tar.gz'],stdout=f,stderr=subprocess.PIPE,timeout=120)
assert p.returncode==0,p.stderr.decode()[-1000:]
assert bundle.stat().st_size==meta['bytes'] and hashlib.sha256(bundle.read_bytes()).hexdigest()==meta['sha256']
dest=ROOT/'server_artifacts';dest.mkdir(exist_ok=True)
allowed=set(required+optional+['COMPACT_METADATA_MANIFEST.json'])
with tarfile.open(bundle,'r:gz') as t:
 for m in t.getmembers():assert m.isfile() and m.name in allowed,m.name
 t.extractall(dest,filter='data')
manifest=json.loads((dest/'COMPACT_METADATA_MANIFEST.json').read_text())
for n,v in manifest['files'].items():
 p=dest/n;assert p.stat().st_size==v['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256']
(ROOT/'MAC_METADATA_RECEIPT.json').write_text(json.dumps(dict(meta,all_selected_receiver_SHA_verified=True,medical_arrays_and_images_transferred=False),indent=2)+'\n')
print(json.dumps({'metadata_files':len(manifest['files']),'bundle_bytes':meta['bytes'],'receiver_SHA':'PASS','images_or_arrays_transferred':False}))
