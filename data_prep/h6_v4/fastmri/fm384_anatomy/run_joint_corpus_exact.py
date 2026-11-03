"""Orchestration-only tail-race repair; invoke immutable bound worker implementation."""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',nnUNet_compile='false',nnUNet_def_n_proc='2')
import json,time,multiprocessing as mp,argparse,queue
from pathlib import Path
from anatomy_job import ROOT,BASE,OwnedJob,write
from infer_joint_anchors import sha,worker as original_worker

def optimized_worker(gpu,q,res,binding,selection):
 proof=json.loads((ROOT/'qa/EXACT_RESAMPLE_PARITY.json').read_text());assert proof['state']=='PASS_EXACT_NUMERIC_PARITY' and len(proof['checks'])==8 and all(x['array_equal'] and x['max_abs_error']==0 for x in proof['checks']);assert proof['resampler_code_sha256']==sha(ROOT/'exact_ap_resampler.py') and proof['anchor_binding_sha256']==binding
 from exact_ap_resampler import install
 install()
 original_worker(gpu,q,res,binding,selection)

from audit_anchor_proposals import audit

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['validation','corpus'],default='validation');a=ap.parse_args()
 with OwnedJob('joint_anchor_'+a.phase+'_inference_repaired_orchestration',cpu=6,ram_gib=24,disk_gib=2,gpus=[0,1]) as job:
  bind=json.loads((ROOT/'ANCHOR_BUILD_BINDING.json').read_text());assert all(sha(ROOT/n)==v for n,v in bind['code_sha256'].items());assert sha(BASE/'receipts/SOURCE_FULLSHA_MANIFEST.json')==bind['source_manifest_sha256']
  # All completed rows/configs/source are audited before reuse. Missing unfinished rows allowed here.
  aux=json.loads((ROOT/'ANCHOR_AUXILIARY_BINDING.json').read_text());assert sha(ROOT/'VALIDATION_SELECTION.json')==aux['selection_sha256']
  for n,f in aux['config_files'].items():assert sha(ROOT/n)==f['sha256']
  if a.phase=='corpus':
   gate=json.loads((ROOT/'ANCHOR_VALIDATION_PASS.json').read_text());assert gate['state']=='PASS' and gate['binding_sha256']==sha(ROOT/'ANCHOR_BUILD_BINDING.json') and gate['selection_sha256']==sha(ROOT/'VALIDATION_SELECTION.json'),'INDEPENDENT_ANCHOR_GATE_REQUIRED'
  if a.phase=='corpus':
   assert gate['policy_sha256']==sha(ROOT/'JOINT_ANCHOR_POLICY.v1.json') and gate['comparison_code_sha256']==sha(ROOT/'validate_anatomy.py')
   assert gate['technical_binding_audit_sha256']==sha(ROOT/'ANCHOR_VALIDATION_BINDING_AUDIT.json') and gate['source_review_binding_audit_sha256']==sha(ROOT/'SOURCE_REVIEW_BINDING_AUDIT.json')
   review=json.loads((ROOT/'SOURCE_REVIEW_BINDING_AUDIT.json').read_text());assert review['state']=='PASS' and review['code_sha256']==sha(ROOT/'validate_review_bindings.py')
   from validate_review_bindings import verify_document
   for n,v in review['files'].items():assert sha(ROOT/n)==v['sha256'] and verify_document(n)['sha256']==v['sha256']
  full=json.loads((BASE/'receipts/SOURCE_FULLSHA_MANIFEST.json').read_text());sel={c['stem'] for c in json.loads((ROOT/'VALIDATION_SELECTION.json').read_text())['cases']};entries=[r for r in full['volumes'] if a.phase=='corpus' or r['stem'] in sel]
  ctx=mp.get_context('spawn');q=ctx.Queue();res=ctx.Queue();ps=[ctx.Process(target=optimized_worker,args=(g,q,res,sha(ROOT/'ANCHOR_BUILD_BINDING.json'),sel)) for g in [0,1]]
  for p in ps:p.start()
  for c in entries:q.put(c)
  for p in ps:q.put(None)
  done=0;stats=[]
  try:
   while done<len(entries):
    job.check()
    try:r=res.get(timeout=15)
    except queue.Empty:
     assert all(p.exitcode in [None,0] for p in ps),'WORKER_NONZERO_EXIT'
     assert any(p.is_alive() for p in ps),'ALL_WORKERS_EXITED_WITH_UNDELIVERED_ROWS'
     continue
    if 'worker_error' in r:raise RuntimeError(r['worker_error']+'\n'+r['traceback'])
    done+=1;stats.append(r);job.progress.update(phase='joint_anchor_'+a.phase+'_inference',completed=done,expected=len(entries),last_stem=r['stem'])
    if done%8==0:print(json.dumps({'completed':done,'expected':len(entries),'last':r}),flush=True)
   for p in ps:p.join(30);assert p.exitcode==0
  finally:
   for p in ps:
    if p.is_alive():p.terminate();p.join()
  technical=audit(a.phase)
  write(ROOT/('ANCHOR_'+a.phase.upper()+'_COMPLETE.json'),{'state':'PROPOSALS_COMPLETE_NOT_VISUAL_PROMOTION','binding_sha256':sha(ROOT/'ANCHOR_BUILD_BINDING.json'),'orchestrator_sha256':sha(__file__),'exact_resampler_sha256':sha(ROOT/'exact_ap_resampler.py'),'exact_resampler_proof_sha256':sha(ROOT/'qa/EXACT_RESAMPLE_PARITY.json'),'technical_audit_sha256':sha(ROOT/('ANCHOR_'+a.phase.upper()+'_BINDING_AUDIT.json')),'repair_scope':'Immutable validated worker/model/geometry; paired actual-array-proven nearestAP resampling evaluation order; safe normal sentinel exit handling','completed':done,'entries':stats,'training_ready':False});job.progress['phase']='anchor_'+a.phase+'_proposals_complete_with_binding_audit'
if __name__=='__main__':main()
