"""Journaled, idempotent metadata freeze for consumer QA. No promotion marker created."""
import shutil,json,time
from pathlib import Path
from anatomy_job import ROOT,BASE,write,OwnedJob
from fm_proof_chain import verify_canary,sha,load,require
from fm_training_reader import CODE_FILES,PROOF_FILES,expected_core_files
from build_anatomical_chunks import atomic_bytes
OUT=ROOT/'training_pixel_store'
def main():
 verify_canary(ROOT,BASE)
 if (OUT/'CORE_BINDING.json').exists():
  from fm_training_reader import verify_core
  return {'state':'ALREADY_FROZEN_NO_MUTATION','core_binding_sha256':sha(OUT/'CORE_BINDING.json'),'files':len(verify_core(OUT)['files_sha256'])}
 require(not (OUT/'PROMOTION_RECEIPT.json').exists(),'PROMOTION_WITHOUT_CORE')
 stage=OUT/'_metadata_freeze_stage';stage.mkdir(exist_ok=True);jp=stage/'TRANSACTION.json'
 if not jp.exists():
  old=load(OUT/'COMPLETION_RECEIPT.json');require(old['state']=='ANATOMICAL_PIXELS_BUILT_FINAL_QA_PENDING','PIXEL_BUILD_NOT_COMPLETE')
  for n,h in old['files_sha256'].items():require(sha(OUT/n)==h,'CANDIDATE_BUILD_FILE_CHANGED:'+n)
  (OUT/'provenance').mkdir(exist_ok=True);atomic_bytes(OUT/'provenance/CANDIDATE_COMPLETION_RECEIPT.json',(OUT/'COMPLETION_RECEIPT.json').read_bytes())
  for n in ['index.json','boxes.json','manifest.json']:
   x=load(OUT/n);x.update(training_ready=True,default_consumer_requires_PROMOTION_RECEIPT=True)
   if n=='manifest.json':x['status']='FINAL_METADATA_FROZEN_READER_QA_REQUIRED'
   else:x['semantic_gate']='PASS'
   write(stage/n,x)
  final=dict(old);final.update(state='PIXELS_FROZEN_FOR_FINAL_READER_QA',files_sha256={n:sha(stage/n if n in ['index.json','boxes.json','manifest.json'] else OUT/n) for n in ['BUILD_BINDING.json','pixel_store.json','index.json','boxes.json','manifest.json']},training_ready=False,metadata_frozen_unix=time.time());write(stage/'COMPLETION_RECEIPT.json',final)
  txn={'original_sha256':dict(old['files_sha256'],**{'COMPLETION_RECEIPT.json':sha(OUT/'COMPLETION_RECEIPT.json')}),'final_sha256':{n:sha(stage/n) for n in ['index.json','boxes.json','manifest.json','COMPLETION_RECEIPT.json']},'state':'PREPARED_NO_LIVE_METADATA_CHANGED'};write(jp,txn)
 txn=load(jp)
 for n,h in txn['final_sha256'].items():
  require(sha(stage/n)==h,'FROZEN_STAGE_CHANGED');require(sha(OUT/n) in [txn['original_sha256'][n],h],'LIVE_METADATA_NEITHER_ORIGINAL_NOR_COMMITTED:'+n)
 for n,h in txn['original_sha256'].items():
  if n not in txn['final_sha256']:require(sha(OUT/n)==h,'UNSTAGED_PIXEL_INPUT_CHANGED')
 (OUT/'code').mkdir(exist_ok=True);(OUT/'provenance').mkdir(exist_ok=True)
 for n in CODE_FILES:atomic_bytes(OUT/'code'/n,(ROOT/n).read_bytes())
 for n in PROOF_FILES:
  source=(BASE/'receipts'/n if n=='SOURCE_FULLSHA_MANIFEST.json' else BASE/'inputs'/n if n in ['FM_ADAPTATION_SPEC.json','knee.original.csv','knee_file_list.csv','quarantined_rows.json','FM_CLASSES.frozen.json'] else ROOT/'qa'/n if n in ['ANATOMICAL_CANARY.json','ANATOMICAL_CANARY_PACKET.json','ANATOMICAL_CANARY_VISUAL_REVIEW.json','EXACT_RESAMPLE_PARITY.json'] else ROOT/n)
  atomic_bytes(OUT/'provenance'/n,source.read_bytes());require(sha(source)==sha(OUT/'provenance'/n),'PROOF_RECEIVER_SHA')
 for n,h in txn['final_sha256'].items():
  if sha(OUT/n)!=h:atomic_bytes(OUT/n,(stage/n).read_bytes())
 core={'state':'FROZEN_INPUTS_FOR_READER_QA','files_sha256':{n:sha(OUT/n) for n in sorted(expected_core_files(OUT))},'training_started':False,'created_unix':time.time()};write(OUT/'CORE_BINDING.json',core)
 return {'state':'FROZEN_REQUIRES_READER_QA_AND_PROMOTION','core_binding_sha256':sha(OUT/'CORE_BINDING.json'),'files':len(core['files_sha256'])}
if __name__=='__main__':
 with OwnedJob('freeze_final_metadata_for_reader_QA',cpu=2,ram_gib=2,disk_gib=.15) as job:print(json.dumps(main()),flush=True)
