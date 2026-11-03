"""Compact code/metadata only. Every received file SHA checked; never export medical pixels or weights."""
from pathlib import Path
import sys,json,hashlib,shutil,zipfile,datetime
T=Path(__file__).resolve().parent
sys.path.insert(0,str(T));from fetch_metadata_exact import fetch
from remote import run

def main():
 paths=json.loads(run("""
from pathlib import Path
import json
r=Path('/workspace/knee_v4_cache_20261021/fastmri_v1/anatomy_v1');ready=json.loads((r/'FINAL_READINESS.json').read_text());assert ready['training_ready'] and not ready['training_started'];core=json.loads((r/'training_pixel_store/CORE_BINDING.json').read_text());paths=['training_pixel_store/'+n for n in core['files_sha256'] if not n.startswith('chunks/')]
paths+=['training_pixel_store/'+n for n in ['CORE_BINDING.json','PROMOTION_RECEIPT.json','provenance/FULLSHA_QA.json','provenance/LOADER_TARGET_QA.json','provenance/CORE_BINDING.before_AST_schema_fix.json','provenance/CORE_BINDING.before_all_view_annotation_QA.json','provenance/CANDIDATE_COMPLETION_RECEIPT.json']]
paths+=['FINAL_READINESS.json','FINAL_RELEASE_SUMMARY.json','STATE.json','README.md','AUX_DATASET_ADAPTATION.json','CHUNK_LOADER_ADAPTATION.json','TASK_CONTRACT.v2.json','SOURCE_EVIDENCE.yaml','ledger.md','ANATOMY_DECISIONS.json','qa/CROSS_PYTHON_AST_PARITY.json','qa/CONSUMER_CORE_REVISION.v2.json','qa/CONSUMER_CORE_REVISION.v3.json','qa/CPU_LOADER_RUNTIME_RECEIPT.json','qa/CPU_LOADER_WHEEL_MANIFEST.json','qa/CPU_LOADER_PIP_FREEZE.txt','qa/INFERENCE_RUNTIME_FINAL_PIP_FREEZE.txt','qa/CHUNK_COMMIT_RESUME_SYNTHETIC.json','qa/GEOMETRY_SYNTHETIC.json','qa/COMMIT_RESUME_SYNTHETIC.json','qa/CORONAL_GRAPH_PARITY.json','qa/consumer_core_v2/LOADER_TARGET_QA.json','qa/consumer_core_v2/FULLSHA_QA.json','promote_anatomical_store.py','prepare_final_store.py','revise_consumer_AST_v2.py','revise_qa_harness_v3.py','prepare_loader_runtime.py']
bind=json.loads((r/'ANCHOR_BUILD_BINDING.json').read_text());paths+=list(bind['code_sha256']);aux=json.loads((r/'ANCHOR_AUXILIARY_BINDING.json').read_text());paths+=list(aux['config_files']);paths+=['anchor_proposals/'+s+'.json' for s in json.loads((r/'ANCHOR_CORPUS_BINDING_AUDIT.json').read_text())['proposal_sha256']];print(json.dumps(sorted(set(paths))))
""",timeout=60))
 assert all(Path(n).suffix in ['.json','.py','.md','.yaml','.txt','.csv'] for n in paths)
 rec=fetch(paths);dest=T/'delivery';dest.mkdir(exist_ok=True)
 for n in paths:
  src=T/'remote_metadata'/n;target=dest/n;target.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(src,target);assert hashlib.sha256(target.read_bytes()).hexdigest()==rec[n]['sha256']
 manifest={'state':'PASS_METADATA_DELIVERY_RECEIVER_FULLSHA','server_pixel_store':'/workspace/knee_v4_cache_20261021/fastmri_v1/anatomy_v1/training_pixel_store','scope':'Code/provenance/QA/metadata only. Pixel chunks and model weights remain server-side. This local bundle is not a second copy of the medical cache.','files':rec,'medical_images_or_pixel_arrays_or_weights_exported':False,'local_verified_bytes':sum(x['bytes'] for x in rec.values()),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(dest/'METADATA_DELIVERY_RECEIVER_FULLSHA.json').write_text(json.dumps(manifest,indent=2)+'\n');archive=T/'fm384_anatomical_ready_metadata_56967269.zip'
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(dest.rglob('*')):
   if p.is_file():z.write(p,p.relative_to(dest))
 summary={'state':'PASS','archive':str(archive),'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'archive_bytes':archive.stat().st_size,'metadata_files_received':len(rec),'uncompressed_metadata_bytes':manifest['local_verified_bytes'],'medical_images_exported':False,'training_started':False};(T/'LOCAL_DELIVERY_RECEIPT.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2));return summary
if __name__=='__main__':main()
