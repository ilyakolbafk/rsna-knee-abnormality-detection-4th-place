"""Strengthen pre-promotion annotation QA; preserve successful previous QA and core bytes."""
from pathlib import Path
import json,time
from anatomy_job import ROOT,OwnedJob,write
from fm_training_reader import sha,need,verify_core
from build_anatomical_chunks import atomic_bytes
OUT=ROOT/'training_pixel_store'
def main(job):
 need(not (OUT/'PROMOTION_RECEIPT.json').exists(),'PROMOTED_STORE_IMMUTABLE');old=json.loads((OUT/'CORE_BINDING.json').read_text());need(old.get('revision')==2,'EXPECTED_CORE_REVISION2');oldsha=sha(OUT/'CORE_BINDING.json');hist=ROOT/'qa/consumer_core_v2';hist.mkdir(exist_ok=True)
 for n,h in old['files_sha256'].items():job.check();need(sha(OUT/n)==h,'PREVIOUS_CORE_CHANGED:'+n)
 archived={}
 for n in ['LOADER_TARGET_QA.json','FULLSHA_QA.json']:
  p=OUT/'provenance'/n;q=json.loads(p.read_text());need(q['state']=='PASS' and q['core_binding_sha256']==oldsha,'PREVIOUS_QA_NOT_PASS_OR_NOT_BOUND');atomic_bytes(hist/n,p.read_bytes());need(sha(hist/n)==sha(p),'PREVIOUS_QA_HISTORY_SHA');archived[n]=sha(p)
 job.register(True);job.check();atomic_bytes(OUT/'provenance/CORE_BINDING.before_all_view_annotation_QA.json',(OUT/'CORE_BINDING.json').read_bytes());atomic_bytes(OUT/'code/qa_fm_consumer.py',(ROOT/'qa_fm_consumer.py').read_bytes());files=dict(old['files_sha256']);files['code/qa_fm_consumer.py']=sha(ROOT/'qa_fm_consumer.py');new=dict(old,revision=3,previous_core_binding_sha256=oldsha,reason='Strengthened actual source SHA-bound per-row annotation multiset equality for every0/2/3view; no cache/target/sampler behavior change.',files_sha256=files,created_unix=time.time());write(OUT/'CORE_BINDING.json',new);verify_core(OUT)
 for n,h in archived.items():
  p=OUT/'provenance'/n;need(sha(p)==h,'PREVIOUS_QA_CHANGED_DURING_ARCHIVE');p.unlink()
 note={'state':'PASS_UNPROMOTED_QA_HARNESS_REVISION','old_core_binding_sha256':oldsha,'new_core_binding_sha256':sha(OUT/'CORE_BINDING.json'),'changed_file':'code/qa_fm_consumer.py','changed_code_sha256':sha(ROOT/'qa_fm_consumer.py'),'prior_successful_qa_archived_exact_sha256':archived,'all_other2356_core_files_unchanged':True,'no_pixels_metadata_source_or_geometry_changed':True,'training_started':False,'created_unix':time.time()};write(ROOT/'qa/CONSUMER_CORE_REVISION.v3.json',note);return note
if __name__=='__main__':
 with OwnedJob('strengthen_unpromoted_actual_annotation_QA',cpu=1,ram_gib=1,disk_gib=.03) as job:print(json.dumps(main(job),indent=2),flush=True)
