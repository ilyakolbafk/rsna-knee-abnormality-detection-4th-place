"""Fail-closed consumer for validated relative FM anatomy. Never launches training."""
import json,hashlib
from pathlib import Path
from fm_chunk_loader import FmChunkStore
POLICY='fm-rss-anatomy-v1'
STATE='relative_lateral_plus_column_head_up'
CORE_FILES=('BUILD_BINDING.json','COMPLETION_RECEIPT.json','pixel_store.json','index.json','boxes.json','manifest.json')
CODE_FILES=('fm_training_reader.py','fm_loader_reference.py','fm_chunk_loader.py','fm_chunk_array.py','fm_delta_codec.py','fm_aux_chunk_adapter.py','fm_aux_reference.py','qa_fm_consumer.py','anatomy_job.py')
PROOF_FILES=('ANATOMY_DECISIONS.json', 'SIDE_VALIDATION_PASS.json', 'ANCHOR_VALIDATION_PASS.json', 'ANATOMICAL_CANARY.json', 'SOURCE_FULLSHA_MANIFEST.json', 'FM_ADAPTATION_SPEC.json', 'ANCHOR_CORPUS_COMPLETE.json', 'ANCHOR_CORPUS_BINDING_AUDIT.json', 'SOURCE_REVIEW_BINDING_AUDIT.json', 'DECISION_POLICY.v1.json', 'JOINT_ANCHOR_POLICY.v1.json', 'ANATOMICAL_QA_PROTOCOL.v1.json', 'ANCHOR_BUILD_BINDING.json', 'ANCHOR_AUXILIARY_BINDING.json', 'SIDE_HYPOTHESES.json', 'VALIDATION_SELECTION.json', 'assemble_anatomy_decisions.py', 'fm_proof_chain.py', 'build_anatomical_chunks.py', 'fm_anatomy_pixels.py', 'fm_geometry.py', 'fm_pixels.py', 'fm_delta_codec.py', 'anatomical_canary.py', 'ANATOMICAL_CANARY_PACKET.json', 'ANATOMICAL_CANARY_VISUAL_REVIEW.json', 'EXACT_RESAMPLE_PARITY.json', 'BLIND_VALIDATION_A.json', 'BLIND_VALIDATION_B.json', 'BLIND_SUPPLEMENT_A.json', 'BLIND_SUPPLEMENT_B.json', 'BLIND_SUPPLEMENT_C011_A.json', 'BLIND_SUPPLEMENT_C011_B.json','run_joint_corpus_exact.py','exact_ap_resampler.py','validate_anatomy.py','validate_review_bindings.py','ANCHOR_VALIDATION_BINDING_AUDIT.json','knee.original.csv','knee_file_list.csv','quarantined_rows.json','FM_CLASSES.frozen.json')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def need(ok,msg):
 if not ok:raise RuntimeError(msg)
def safe_path(root,name):
 p=root/name;need(isinstance(name,str) and not Path(name).is_absolute() and '..' not in Path(name).parts and p.is_file() and not p.is_symlink() and root.resolve() in p.resolve().parents,'FM_INVALID_OR_MISSING_BOUND_PATH:'+str(name));return p

def expected_core_files(root):
 spec=load(root/'pixel_store.json');index=load(root/'index.json');files=set(CORE_FILES)|{'code/'+n for n in CODE_FILES}|{'provenance/'+n for n in PROOF_FILES}
 accepted={s:v for s,v in index['volumes'].items() if not v['excluded']};need({c['stem'] for c in spec['chunks']}==set(accepted),'FM_PIXEL_MEMBERSHIP_MISMATCH')
 for c in spec['chunks']:
  need(c['path']=='chunks/'+c['stem']+'.dxz','FM_UNEXPECTED_CHUNK_NAME');files.add(c['path']);files.add('volume_receipts/'+c['stem']+'.json')
 return files

def verify_core(path,verify_pixels=True):
 p=Path(path).resolve();core=load(p/'CORE_BINDING.json');need(core.get('state')=='FROZEN_INPUTS_FOR_READER_QA','FM_CORE_BINDING_NOT_FROZEN');files=core.get('files_sha256');need(isinstance(files,dict) and set(files)==expected_core_files(p),'FM_INCOMPLETE_REQUIRED_CORE_FILESET')
 for n,digest in files.items():
  target=safe_path(p,n)
  if verify_pixels or not n.startswith('chunks/'):need(sha(target)==digest,'FM_BOUND_FILE_SHA_CHANGED:'+n)
 # Verify executing consumer, not only an unused copy on disk.
 for n in CODE_FILES:need(sha(Path(__file__).parent/n)==files['code/'+n],'FM_EXECUTING_CONSUMER_CODE_DIFFERS:'+n)
 idx=load(p/'index.json');bind=load(p/'BUILD_BINDING.json');done=load(p/'COMPLETION_RECEIPT.json');dec=load(p/'provenance/ANATOMY_DECISIONS.json');canary=load(p/'provenance/ANATOMICAL_CANARY.json');side=load(p/'provenance/SIDE_VALIDATION_PASS.json');anchor=load(p/'provenance/ANCHOR_VALIDATION_PASS.json');spec=load(p/'pixel_store.json');src=load(p/'provenance/SOURCE_FULLSHA_MANIFEST.json');sr={r['stem']:r for r in src['volumes']}
 need(done['state']=='PIXELS_FROZEN_FOR_FINAL_READER_QA' and done['all_chunks_every_pixel_decode_equal_and_receiver_fullsha'] is True,'FM_PIXEL_BUILD_INCOMPLETE')
 need(done['build_binding_sha256']==sha(p/'BUILD_BINDING.json')==spec['build_binding_sha256'],'FM_BUILD_BINDING_MISMATCH')
 for n in ['BUILD_BINDING.json','pixel_store.json','index.json','boxes.json','manifest.json']:need(done['files_sha256'][n]==files[n],'FM_COMPLETION_FILE_BINDING_MISMATCH:'+n)
 need(bind['anatomy_decisions_sha256']==files['provenance/ANATOMY_DECISIONS.json'] and bind['anatomical_canary_sha256']==files['provenance/ANATOMICAL_CANARY.json'],'FM_STALE_DECISIONS_OR_CANARY')
 need(bind['producer_code_sha256']==files['provenance/build_anatomical_chunks.py'],'FM_PRODUCER_CODE_MISMATCH')
 for n,h in dec['inputs']['renderer_sha256'].items():need(files['provenance/'+n]==h,'FM_RENDERER_CODE_MISMATCH:'+n)
 need(bind['inputs']==dec['inputs']==canary['inputs'] and dec['gate_state']==canary['state']=='PASS' and canary['numeric_checks_pass'] and canary['visual_review_state']=='PASS','FM_ANATOMY_INPUT_CHAIN_CLOSED')
 need(canary['anatomy_decisions_sha256']==sha(p/'provenance/ANATOMY_DECISIONS.json'),'FM_CANARY_DECISION_MISMATCH')
 need(canary['code_sha256']==files['provenance/anatomical_canary.py'] and canary['packet_sha256']==files['provenance/ANATOMICAL_CANARY_PACKET.json'] and canary['visual_review_sha256']==files['provenance/ANATOMICAL_CANARY_VISUAL_REVIEW.json'],'FM_CANARY_REVIEW_OR_CODE_MISMATCH')
 for n,h in side['frozen_observations_sha256'].items():need(files['provenance/'+n]==h,'FM_FROZEN_SOURCE_OBSERVATION_MISMATCH:'+n)
 review=load(p/'provenance/ANATOMICAL_CANARY_VISUAL_REVIEW.json');packet=load(p/'provenance/ANATOMICAL_CANARY_PACKET.json');need(review['state']=='PASS' and review['packet_sha256']==canary['packet_sha256'] and review['panels_sha256']==packet['panels_sha256'] and set(review['reviewed_stems'])=={x['stem'] for x in packet['cases']},'FM_CANARY_REVIEW_COVERAGE_MISMATCH')
 need(side['state']==anchor['state']=='PASS','FM_INDEPENDENT_ANATOMY_GATES_CLOSED')
 done_anchor=load(p/'provenance/ANCHOR_CORPUS_COMPLETE.json')
 for field,n in [('orchestrator_sha256','run_joint_corpus_exact.py'),('exact_resampler_sha256','exact_ap_resampler.py'),('exact_resampler_proof_sha256','EXACT_RESAMPLE_PARITY.json'),('technical_audit_sha256','ANCHOR_CORPUS_BINDING_AUDIT.json')]:need(done_anchor[field]==files['provenance/'+n],'FM_ANCHOR_PROOF_LINK_MISMATCH:'+n)
 for g in [side,anchor]:
  need(g['comparison_code_sha256']==files['provenance/validate_anatomy.py'] and g['source_review_binding_audit_sha256']==files['provenance/SOURCE_REVIEW_BINDING_AUDIT.json'],'FM_COMPARISON_PROOF_LINK_MISMATCH')
 need(anchor['technical_binding_audit_sha256']==files['provenance/ANCHOR_VALIDATION_BINDING_AUDIT.json'],'FM_VALIDATION_AUDIT_MISMATCH')
 source_review=load(p/'provenance/SOURCE_REVIEW_BINDING_AUDIT.json');need(source_review['code_sha256']==files['provenance/validate_review_bindings.py'] and source_review['protocol_sha256']==files['provenance/ANATOMICAL_QA_PROTOCOL.v1.json'],'FM_SOURCE_REVIEW_PROTOCOL_MISMATCH')
 need(side['source_manifest_sha256']==bind['source_manifest_sha256']==files['provenance/SOURCE_FULLSHA_MANIFEST.json'],'FM_SOURCE_MANIFEST_BINDING_MISMATCH')
 need(bind['adaptation_spec_sha256']==files['provenance/FM_ADAPTATION_SPEC.json'],'FM_MAPPING_SPEC_MISMATCH')
 for n in dec['inputs']['root_files_sha256']:
  need(dec['inputs']['root_files_sha256'][n]==files['provenance/'+n],'FM_DECISION_PROOF_BINDING_MISMATCH:'+n)
 need(set(idx['volumes'])==set(dec['volumes'])==set(sr) and len(sr)==1172,'FM_SOURCE_VOLUME_MEMBERSHIP')
 need(idx['orientation_policy']==POLICY and idx['orientation_state']==STATE and idx['training_ready'] is True and idx['semantic_gate']=='PASS','FM_ANATOMICAL_METADATA_NOT_FROZEN')
 pos=0
 for chunk in spec['chunks']:
  s=chunk['stem'];v=idx['volumes'][s];rec=load(p/'volume_receipts'/(s+'.json'));need(chunk==rec['chunk'] and rec['every_decoded_pixel_compared'] is True,'FM_CHUNK_RECEIPT_MISMATCH')
  need(rec['row_binding']=={'stem':s,'start':pos,'n_slices':v['n_slices'],'build_binding_sha256':sha(p/'BUILD_BINDING.json'),'source_receipt_sha256':v['source_receipt_sha256']},'FM_VOLUME_ROW_BINDING_MISMATCH')
  need(v['n_slices']==sr[s]['n_slices'] and v['views']==[[0,pos],[2,pos+v['n_slices']],[3,pos+2*v['n_slices']]],'FM_SOURCE_VOLUME_ROW_OFFSETS_MISMATCH:'+s)
  need(chunk['start']==pos and chunk['shape']==[3*v['n_slices'],384,384] and files[chunk['path']]==chunk['compressed_sha256'],'FM_CHUNK_BLOCK_MISMATCH');pos+=chunk['shape'][0]
 for s,v in idx['volumes'].items():
  a=dec['volumes'][s];r=sr[s];need(v['reviewed_no_findings']==a['reviewed_no_findings'],'FM_NO_FINDINGS_STATUS_DRIFT:'+s)
  need(v['excluded']==(not a['accepted']) and v['anatomy_decision']==a,'FM_EXCLUSION_OR_DECISION_DRIFT:'+s)
  need(v['source_receipt_sha256']==r['receipt_sha256'] and v['patient_id']==r['patient_id'] and v['aux_holdout']==r['patient_safe_closure_holdout'],'FM_SOURCE_EXAM_GROUP_DRIFT:'+s)
  if v['excluded']:need(v['views']==[],'FM_EXCLUDED_VOLUME_HAS_ROWS');continue
  need(v['anatomical_medial_lateral_known'] is True and v['orientation_policy']==POLICY and v['orientation_state']==STATE and [x[0] for x in v['views']]==[0,2,3] and v['view_names']=={'0':'fullfield','2':'medial_local100mm','3':'lateral_local100mm'},'FM_VIEW_ANATOMY_MISMATCH:'+s)
 need(pos==idx['n_rows']==spec['shape'][0],'FM_ROW_COVERAGE_MISMATCH');return core

def load_store(path,*,review_geometry_only=False,min_volumes=1000):
 p=Path(path).resolve()
 if not review_geometry_only:need((p/'PROMOTION_RECEIPT.json').is_file(),'FM_ANATOMICAL_STORE_NOT_PROMOTED')
 core=verify_core(p,verify_pixels=True)
 if not review_geometry_only:
  marker=p/'PROMOTION_RECEIPT.json';need(marker.is_file(),'FM_ANATOMICAL_STORE_NOT_PROMOTED');receipt=load(marker);need(receipt.get('state')=='PASS' and receipt.get('core_binding_sha256')==sha(p/'CORE_BINDING.json'),'FM_PROMOTION_CORE_MISMATCH')
  required=set(core['files_sha256'])|{'CORE_BINDING.json','provenance/LOADER_TARGET_QA.json','provenance/FULLSHA_QA.json'};need(set(receipt.get('files_sha256',{}))==required,'FM_INCOMPLETE_PROMOTION_FILESET')
  for n,digest in receipt['files_sha256'].items():need(sha(safe_path(p,n))==digest,'FM_PROMOTED_FILE_SHA_MISMATCH:'+n)
  for n in ['LOADER_TARGET_QA.json','FULLSHA_QA.json']:
   q=load(p/'provenance'/n);need(q['state']=='PASS' and q['core_binding_sha256']==sha(p/'CORE_BINDING.json'),'FM_REQUIRED_QA_GATE_NOT_BOUND:'+n)
 st=FmChunkStore(str(p),min_volumes=min_volumes);st._fm_guard_mode='review' if review_geometry_only else 'training';st._fm_pixel_backend='lossless_volume_chunks';st._fm_pixel_store_sha256=sha(p/'pixel_store.json');st._fm_promotion_receipt_sha256=None if review_geometry_only else sha(p/'PROMOTION_RECEIPT.json');return st
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('store');ap.add_argument('--review-geometry-only',action='store_true');a=ap.parse_args();st=load_store(a.store,review_geometry_only=a.review_geometry_only);print(json.dumps({'state':'PASS_REVIEW_ONLY' if a.review_geometry_only else 'PASS_TRAINING_INPUT_PREFLIGHT','usable_volumes':len(st.vols),'excluded_volumes':len(st.excluded),'n_rows':st.n_rows,'training_started':False}))
