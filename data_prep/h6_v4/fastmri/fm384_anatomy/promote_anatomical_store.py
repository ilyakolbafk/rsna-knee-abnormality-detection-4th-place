"""Atomic final training-input marker after actual consumer and receiver gates; no training."""
from pathlib import Path
import json,time
from anatomy_job import ROOT,OwnedJob,write
from fm_training_reader import load_store,verify_core,sha,need
OUT=ROOT/'training_pixel_store'
def main():
 core=verify_core(OUT);digest=sha(OUT/'CORE_BINDING.json')
 for n in ['LOADER_TARGET_QA.json','FULLSHA_QA.json']:
  q=json.loads((OUT/'provenance'/n).read_text());need(q['state']=='PASS' and q['core_binding_sha256']==digest and q['training_started'] is False,'FINAL_QA_NOT_BOUND:'+n)
 q=json.loads((OUT/'provenance/LOADER_TARGET_QA.json').read_text());need(q['code_sha256']==sha(ROOT/'qa_fm_consumer.py')==sha(OUT/'code/qa_fm_consumer.py'),'STALE_CONSUMER_QA_CODE')
 names=set(core['files_sha256'])|{'CORE_BINDING.json','provenance/LOADER_TARGET_QA.json','provenance/FULLSHA_QA.json'};files={n:sha(OUT/n) for n in sorted(names)}
 receipt={'state':'PASS','artifact':'fm384_v1 relative anatomical adaptation','orientation_policy':'fm-rss-anatomy-v1','orientation_state':'relative_lateral_plus_column_head_up','core_binding_sha256':digest,'files_sha256':files,'training_ready':True,'training_started':False,'private_pixel_parity':False,'no_LPS_v8_3_or_population100_percent_accuracy_claim':True,'promoted_unix':time.time()}
 marker=OUT/'PROMOTION_RECEIPT.json'
 if marker.exists():
  old=json.loads(marker.read_text());need(old['files_sha256']==files and old['core_binding_sha256']==digest,'PROMOTED_ARTIFACT_CHANGED');receipt=old
 else:write(marker,receipt)
 st=load_store(OUT);result={'state':'PASS_TRAINING_INPUT_PREFLIGHT','store':str(OUT),'promotion_receipt_sha256':sha(marker),'core_binding_sha256':digest,'usable_volumes':len(st.vols),'excluded_volumes':len(st.excluded),'physical_slices_usable':st.n_rows//3,'cache_rows':st.n_rows,'train_boxed_volumes':len(st.train_boxed),'train_reviewed_no_findings_volumes':len(st.train_neg),'holdout_volumes':len(st.holdout),'anonymized_exam_groups':len(set(st.pid)),'class_training_boxes':{c:int(st.per_cls[0,k]) for k,c in enumerate(__import__('fm_loader_reference').FM_CLASSES)},'training_ready':True,'training_started':False,'private_pixel_parity':False,'created_unix':time.time()};write(ROOT/'FINAL_READINESS.json',result);return result
if __name__=='__main__':
 with OwnedJob('final_anatomical_store_promotion_and_preflight',cpu=2,ram_gib=2,disk_gib=.1) as job:
  r=main();job.progress.update(phase='anatomical_cache_ready_no_training',training_ready=True,promotion_receipt_sha256=r['promotion_receipt_sha256']);print(json.dumps(r,indent=2),flush=True)
