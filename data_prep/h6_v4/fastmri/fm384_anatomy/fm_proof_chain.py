"""Source-bound prebuild gates. A PASS from different inputs never authorizes a build."""
import json,hashlib
from pathlib import Path
SOURCE_SHA='eb823926e1ea8ffd28056d80888eb39ed3d5231e5ad869fd56f90396a85b8fc4'
POLICY='fm-rss-anatomy-v1'
ORIENTATION='relative_lateral_plus_column_head_up'
TRUSTED_GATES={'SIDE_VALIDATION_PASS.json': '9804096b03397c180673e04f265d96215dbbe6b4810af225787f20b9f8acc79b', 'ANCHOR_VALIDATION_PASS.json': '3d162f468cac871cdd70a2d1548fae706934687e17c6eef7ef0ea33ddd20782e'} # populated from exact, already frozen independent validation bytes
RENDER_CODE=('fm_anatomy_pixels.py','fm_geometry.py','fm_pixels.py','fm_delta_codec.py')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def require(ok,msg):
 if not ok:raise RuntimeError(msg)
def require_sha(path,digest):require(sha(path)==digest,'BOUND_SHA_CHANGED:'+str(path))
def current_inputs(root,base):
 root=Path(root);base=Path(base)
 for n,d in TRUSTED_GATES.items():require_sha(root/n,d)
 side=load(root/'SIDE_VALIDATION_PASS.json');anchor=load(root/'ANCHOR_VALIDATION_PASS.json')
 require(side['state']==anchor['state']=='PASS','INDEPENDENT_ANATOMY_GATE_CLOSED')
 require_sha(base/'receipts/SOURCE_FULLSHA_MANIFEST.json',SOURCE_SHA)
 bind=load(root/'ANCHOR_BUILD_BINDING.json');audit=load(root/'ANCHOR_CORPUS_BINDING_AUDIT.json');done=load(root/'ANCHOR_CORPUS_COMPLETE.json')
 require(done['completed']==audit['checked_volumes']==1172 and done['state']=='PROPOSALS_COMPLETE_NOT_VISUAL_PROMOTION' and audit['state']=='PASS_TECHNICAL_BINDING_ONLY','INCOMPLETE_CORPUS')
 require_sha(root/'ANCHOR_CORPUS_BINDING_AUDIT.json',done['technical_audit_sha256']);require_sha(root/'ANCHOR_BUILD_BINDING.json',done['binding_sha256'])
 require_sha(root/'run_joint_corpus_exact.py',done['orchestrator_sha256']);require_sha(root/'exact_ap_resampler.py',done['exact_resampler_sha256']);require_sha(root/'qa/EXACT_RESAMPLE_PARITY.json',done['exact_resampler_proof_sha256'])
 require(audit['source_manifest_sha256']==bind['source_manifest_sha256']==SOURCE_SHA and audit['binding_sha256']==sha(root/'ANCHOR_BUILD_BINDING.json'),'CORPUS_SOURCE_BINDING_MISMATCH')
 for bucket in ['code_sha256','model_sha256']:
  for n,d in bind[bucket].items():require_sha(root/n,d)
 aux=load(root/'ANCHOR_AUXILIARY_BINDING.json')
 for n,v in aux['config_files'].items():require_sha(root/n,v['sha256'])
 review=load(root/'SOURCE_REVIEW_BINDING_AUDIT.json');require_sha(root/'ANATOMICAL_QA_PROTOCOL.v1.json',review['protocol_sha256']);require_sha(root/'validate_review_bindings.py',review['code_sha256'])
 require_sha(root/'SIDE_HYPOTHESES.json',bind['side_hypotheses_sha256']);require_sha(root/'JOINT_ANCHOR_POLICY.v1.json',bind['policy_sha256'])
 require(side['source_manifest_sha256']==SOURCE_SHA and side['side_hypotheses_sha256']==bind['side_hypotheses_sha256'] and anchor['binding_sha256']==sha(root/'ANCHOR_BUILD_BINDING.json'),'ANATOMY_GATE_INPUT_MISMATCH')
 for g in [side,anchor]:
  require_sha(root/'SOURCE_REVIEW_BINDING_AUDIT.json',g['source_review_binding_audit_sha256']);require_sha(root/'VALIDATION_SELECTION.json',g['selection_sha256']);require_sha(root/'validate_anatomy.py',g['comparison_code_sha256'])
  for n,d in g['frozen_observations_sha256'].items():require_sha(root/n,d)
 require_sha(root/'JOINT_ANCHOR_POLICY.v1.json',anchor['policy_sha256']);require_sha(root/'ANCHOR_VALIDATION_BINDING_AUDIT.json',anchor['technical_binding_audit_sha256'])
 src=load(base/'receipts/SOURCE_FULLSHA_MANIFEST.json');require(len(src['volumes'])==1172,'SOURCE_CARDINALITY')
 expected={x['stem'] for x in src['volumes']};require(set(audit['proposal_sha256'])==expected,'PROPOSAL_MEMBERSHIP')
 for stem,digest in audit['proposal_sha256'].items():require_sha(root/'anchor_proposals'/(stem+'.json'),digest)
 names=('SIDE_VALIDATION_PASS.json','ANCHOR_VALIDATION_PASS.json','SOURCE_REVIEW_BINDING_AUDIT.json','ANATOMICAL_QA_PROTOCOL.v1.json','ANCHOR_BUILD_BINDING.json','ANCHOR_AUXILIARY_BINDING.json','ANCHOR_CORPUS_COMPLETE.json','ANCHOR_CORPUS_BINDING_AUDIT.json','SIDE_HYPOTHESES.json','VALIDATION_SELECTION.json','JOINT_ANCHOR_POLICY.v1.json','DECISION_POLICY.v1.json','assemble_anatomy_decisions.py','fm_proof_chain.py')
 return {'source_manifest_sha256':SOURCE_SHA,'adaptation_spec_sha256':sha(base/'inputs/FM_ADAPTATION_SPEC.json'),'root_files_sha256':{n:sha(root/n) for n in names},'renderer_sha256':{n:sha(root/n) for n in RENDER_CODE},'policy':POLICY,'orientation':ORIENTATION}
def verify_decisions(root,base):
 root=Path(root);d=load(root/'ANATOMY_DECISIONS.json');require(d['gate_state']=='PASS' and d['training_started'] is False,'DECISIONS_NOT_PASS');require(d['inputs']==current_inputs(root,base),'STALE_DECISIONS_INPUTS')
 require(set(d['volumes'])=={x['stem'] for x in load(Path(base)/'receipts/SOURCE_FULLSHA_MANIFEST.json')['volumes']},'DECISIONS_MEMBERSHIP')
 return d
def verify_canary(root,base):
 root=Path(root);d=verify_decisions(root,base);c=load(root/'qa/ANATOMICAL_CANARY.json')
 require(c['state']=='PASS' and c['inputs']==d['inputs'] and c['anatomy_decisions_sha256']==sha(root/'ANATOMY_DECISIONS.json'),'STALE_CANARY_INPUTS')
 require_sha(root/'anatomical_canary.py',c['code_sha256']);require_sha(root/'qa/ANATOMICAL_CANARY_PACKET.json',c['packet_sha256']);require_sha(root/'qa/ANATOMICAL_CANARY_VISUAL_REVIEW.json',c['visual_review_sha256']);review=load(root/'qa/ANATOMICAL_CANARY_VISUAL_REVIEW.json');packet=load(root/'qa/ANATOMICAL_CANARY_PACKET.json');require(review['state']=='PASS' and review['packet_sha256']==c['packet_sha256'] and review['panels_sha256']==packet['panels_sha256'] and set(review['reviewed_stems'])=={x['stem'] for x in packet['cases']},'STALE_OR_INCOMPLETE_VISUAL_REVIEW');
 for n,h in packet['panels_sha256'].items():require_sha(root/'qa/anatomical_panels'/n,h)
 require(c['numeric_checks_pass'] and c['visual_review_state']=='PASS','CANARY_NOT_REVIEWED')
 return d,c
