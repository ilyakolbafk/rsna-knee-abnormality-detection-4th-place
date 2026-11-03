"""Wait for the owned source job, verify, canary, build and QA. Never trains."""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import fcntl,json,time,sys,traceback,datetime
from pathlib import Path
from pathlib import Path
from build_candidate import ROOT,OUT,sha,write,build
from fm_job import OwnedJob,ticks,process_live

def source_terminal_barrier(launch,state):
 assert not process_live(launch['PID'],launch['ctime']), 'SOURCE_PROCESS_STILL_LIVE'
 assert len(list((ROOT/'source_rss').glob('file*/receipt.json')))==1172,'SOURCE_COUNT_INCOMPLETE'
 binding={'source_launch_sha256':sha(ROOT/'LAUNCH_SOURCE_CORPUS.json'),'source_code_sha256':sha(ROOT/'code/source_collect.py'),'archive_proof_sha256':{sp:sha(ROOT/'receipts'/('ARCHIVE_COMPLETE_'+sp+'.json')) for sp in ['train','val']}}
 assert binding['source_code_sha256']==launch['code_sha256']
 p=ROOT/'receipts/SOURCE_TERMINAL_RECEIPT.json'
 if p.exists():assert json.loads(p.read_text())['binding']==binding,'SOURCE_TERMINAL_BINDING_CHANGED'
 else:write(p,{'binding':binding,'state_at_barrier':state,'source_terminal_unix':time.time(),'note':'Barrier permits full receiver verification only; no source or pixel PASS by itself.'})
 return sha(p)

def verified_source(progress,terminal_sha):
 import verify_source
 manifest=ROOT/'receipts/SOURCE_FULLSHA_MANIFEST.json';report=ROOT/'receipts/SOURCE_VERIFICATION.json';complete=ROOT/'receipts/SOURCE_VERIFICATION_COMPLETE.json'
 if complete.exists():
  c=json.loads(complete.read_text());assert c=={'source_terminal_receipt_sha256':terminal_sha,'source_manifest_sha256':sha(manifest),'source_verification_sha256':sha(report),'verifier_code_sha256':sha(ROOT/'code/verify_source.py')},'SOURCE_VERIFICATION_RESUME_BINDING_CHANGED'
 else:
  assert not (OUT/'BUILD_BINDING.json').exists(),'CANDIDATE_BOUND_WITHOUT_ATOMIC_SOURCE_VERIFICATION_PROOF'
  verify_source.main(progress)
  write(complete,{'source_terminal_receipt_sha256':terminal_sha,'source_manifest_sha256':sha(manifest),'source_verification_sha256':sha(report),'verifier_code_sha256':sha(ROOT/'code/verify_source.py')})
 source=json.loads(report.read_text());assert source==json.loads(manifest.read_text())['report'] and source['state']=='PASS_SOURCE_STAGE_ONLY'
 return source

def main():
 controller=(ROOT/'finish_controller.lock').open('a');fcntl.flock(controller,fcntl.LOCK_EX|fcntl.LOCK_NB)
 launch=json.loads((ROOT/'LAUNCH_SOURCE_CORPUS.json').read_text());assert sha(ROOT/'code/source_collect.py')==launch['code_sha256']
 while True:
  s=json.loads((ROOT/'STATE.json').read_text());active=process_live(launch['PID'],launch['ctime']);now=time.time();elapsed=max(1,now-s.get('started_unix',now));transferred=s.get('bytes_transferred_this_run',0);rate=transferred/elapsed
  progress={'phase':'waiting_source_corpus','source_phase':s.get('phase'),'source_PID':launch['PID'],'source_process_live':active,'expected_volumes':1172,'committed':len(list((ROOT/'source_rss').glob('file*/receipt.json'))),'slices_committed':s.get('slices_committed'),'bytes_transferred_this_run':transferred,'reused_source_output_bytes':s.get('reused_source_rss_bytes',0),'transfer_Bps':rate,'ETA_source_seconds':max(0,(94067027900-transferred)/rate) if rate else None,'ready':False,'controller_PID':os.getpid(),'controller_ctime':ticks(os.getpid()),'updated_unix':now}
  write(ROOT/'PROGRESS.json',progress)
  if not active:
   if progress['committed']==1172 and all((ROOT/'receipts'/('ARCHIVE_COMPLETE_'+sp+'.json')).exists() for sp in ['train','val']):
    terminal_sha=source_terminal_barrier(launch,s);break
   raise RuntimeError('OWN_SOURCE_JOB_TERMINAL_WITHOUT_COMPLETE1172_ARCHIVE_PROOF: '+str(s.get('phase')))
  if s.get('phase')=='failed':raise RuntimeError('OWN_SOURCE_JOB_FAILED_REQUIRES_DIAGNOSIS: '+str(s.get('error_type')))
  if now-s.get('updated_unix',0)>150:raise RuntimeError('OWN_SOURCE_HEARTBEAT_STALE_REQUIRES_DIAGNOSIS')
  time.sleep(20)
 with OwnedJob() as job:
  tests=json.loads((ROOT/'qa/FINALIZER_INTEGRATION_TESTS.json').read_text());assert all(x['returncode']==0 for x in tests['results'])
  job.progress.update(committed=1172,downloaded_volumes=1172,bytes_transferred_this_run=s.get('bytes_transferred_this_run'),source_process_terminal=True,controller_PID=os.getpid())
  source=verified_source(job.progress,terminal_sha)
  job.progress.update(phase='real_canary_before_candidate_corpus',slices_committed=source['physical_slices'])
  import qa_real_canary
  qa_real_canary.main()
  canary=json.loads((ROOT/'qa/REAL_CANARY_QA.json').read_text());assert canary['state']=='PASS_NUMERIC_REAL_CANARY_MONTAGE_GENERATED' and canary['all11_positive_area_dense_targets']
  assert canary['binding']['source_manifest_sha256']==sha(ROOT/'receipts/SOURCE_FULLSHA_MANIFEST.json') and canary['binding']['adaptation_spec_sha256']==sha(ROOT/'inputs/FM_ADAPTATION_SPEC.json')
  assert canary['records_sha256']==sha(ROOT/'qa/real_canary/records.json')
  for m in canary['source_flipud_display_and_views_montages']:assert Path(m['path']).stat().st_size==m['bytes'] and sha(m['path'])==m['sha256']
  build(job.progress)
  job.progress.update(phase='loader_target_QA')
  import qa_candidate
  qa=qa_candidate.main(job.progress)
  result={'state':'SOURCE_AND_GEOMETRIC_CANDIDATE_VERIFIED_FINAL_ANATOMICAL_GATE_UNRESOLVED','training_ready':False,'goal_complete':False,'source':source,'candidate_QA':qa,'candidate_receipt_sha256':sha(OUT/'COMPLETION_RECEIPT.json'),'source_manifest_sha256':sha(ROOT/'receipts/SOURCE_FULLSHA_MANIFEST.json'),'adaptation_spec_sha256':sha(ROOT/'inputs/FM_ADAPTATION_SPEC.json'),'medical_images_and_arrays_remain_on_server':True,'no_training_or_old_queue_changes':True,'completed_unix':time.time(),'blocker':'Private fm384 producer absent and no validated standalone RSS anatomical medial/lateral mapping; delivered view2/3 remain explicitly screen-left/right100mm.'}
  write(ROOT/'FINAL_STATUS.json',result)
  job.progress.update(phase='source_and_candidate_verified_anatomical_gate_unresolved',ready=False,goal_complete=False,blocker=result['blocker'])
  ledger='# fastMRI source + fm384 geometry candidate\n\n'+json.dumps({'target':56967269,'source_volumes':source['actual_volumes'],'physical_slices':source['physical_slices'],'candidate_rows':qa['n_rows'],'source_archive_proofs':source['archive_proofs'],'candidate_store':str(OUT),'QA':'PASS_STRUCTURAL_AND_TARGET_QA_ONLY','training_ready':False,'goal_complete':False,'blocker':result['blocker'],'external_writes':'Own fastmri namespace and locked shared admission only; no training/queue/rental/publication','last_source_refresh_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'resume_command':str(ROOT/'render_venv/bin/python')+' '+str(ROOT/'code/finish_pipeline.py')},indent=2)+'\n'
  (ROOT/'ledger.md').write_text(ledger)
 print(json.dumps({'state':result['state'],'source_volumes':source['actual_volumes'],'rows':qa['n_rows'],'training_ready':False,'goal_complete':False}),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  write(ROOT/'FINISHER_FAILURE.json',{'type':type(e).__name__,'reason':str(e),'unix':time.time(),'PID':os.getpid(),'ctime':ticks(os.getpid())});raise
