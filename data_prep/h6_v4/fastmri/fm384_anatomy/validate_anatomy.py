"""Deterministic comparison with frozen blinded source observations; no clinical accuracy claim."""
from pathlib import Path
import json,hashlib,datetime,sys
import numpy as np
T=Path(__file__).resolve().parent;M=T/'remote_metadata'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(n,remote=False):return json.loads(((M if remote else T)/n).read_text())
def source_observations(filename,group):
 from validate_review_bindings import verify_document
 verify_document(filename)
 d=load(filename);assert 'PARTIAL' not in d['state'] and 'PROGRESS' not in d['state'],'SOURCE_REVIEW_NOT_FROZEN'
 if group=='A':
  assert d.get('individual_model_hypotheses_or_segmentations_or_other_reviewers_read') is False
  pairs=[(c['observation'],c['provenance']) for c in d['cases']]
 else:
  assert (all(v is False for v in d['blinding'].values()) if 'blinding' in d else d.get('individual_model_or_other_reviewer_results_read') is False);pairs=[(c,c) for c in d['observations']]
 out={}
 for c,p in pairs:
  alias=c['alias'];assert alias not in out;side=c.get('side',c.get('fibula_image_side'));w=c.get('source_slice_witnesses',[v['source_slice'] for v in c.get('witnesses',[])])
  joint=c.get('approx_joint_line',c.get('joint_center'))
  if joint is None:row=col=None;jw=[]
  else:
   row=joint.get('rows',joint.get('display_row_interval'));col=joint.get('cols',joint.get('display_col_interval'));jw=joint.get('source_indices',joint.get('source_slice_witnesses'));assert len(set(jw))>=2 and len(row)==len(col)==2 and all(0<=float(v)<=319 for v in row+col)
  out[alias]={'side':side,'witnesses':w,'resolved':side in ['left','right'] and len(set(w))>=2 and not c.get('needs_more_frames',False),'confidence':c['confidence'],'joint_row_interval':row,'joint_col_interval':col,'joint_witnesses':jw,'source_record':c,'source_provenance':p}
 return out

def compare_anchor(r,obs,spacing):
 a=r['source_anchor_display_row_col'];intervals=[obs['joint_row_interval'],obs['joint_col_interval']];sp=[spacing[1],spacing[0]]
 if a is None:return {'resolved':False,'reason':'NO_MODEL_ANCHOR'}
 dist=[max(float(lo)-v,0.,v-float(hi))*s for v,(lo,hi),s in zip(a,intervals,sp)];mid=[abs(v-(float(lo)+float(hi))/2)*s for v,(lo,hi),s in zip(a,intervals,sp)]
 return {'resolved':not r['proposal']['geometry_issues'],'distance_to_source_interval_row_col_mm':dist,'distance_to_source_interval_midpoint_row_col_mm':mid,'within_frozen_bounds':max(dist)<=5 and max(mid)<=10,'source_anchor':a,'source_joint_intervals':intervals,'source_witnesses':obs['joint_witnesses'],'model_geometry_issues':r['proposal']['geometry_issues']}

def anchor_gate(A,B):
 sel=load('VALIDATION_SELECTION.json');audit=load('ANCHOR_VALIDATION_BINDING_AUDIT.json',True);assert audit['state']=='PASS_TECHNICAL_BINDING_ONLY' and audit['checked_volumes']==112
 assert audit['selection_sha256']==sha(T/'VALIDATION_SELECTION.json');assert audit['policy_sha256']==sha(M/'JOINT_ANCHOR_POLICY.v1.json');assert audit['binding_sha256']==sha(M/'ANCHOR_BUILD_BINDING.json')
 results={};resolved=0;mismatches=[];uncertain=[]
 for c in sel['cases']:
  if not c['alias'].startswith('V'):continue
  o=(A if int(c['alias'][1:])<49 else B)[c['alias']];assert o['joint_row_interval'] is not None and len(set(o['joint_witnesses']))>=2;f=M/'anchor_proposals'/(c['stem']+'.json');assert sha(f)==audit['proposal_sha256'][c['stem']];r=json.loads(f.read_text());assert r['row_binding']['source_receipt_sha256']==c['source_receipt_sha256'];p=compare_anchor(r,o,c['spacing_xy_mm']);results[c['alias']]={'stem':c['stem'],'method':r['method'],**p}
  if not p['resolved']:uncertain.append(c['alias'])
  else:
   resolved+=1
   if not p['within_frozen_bounds']:mismatches.append(c['alias'])
 assert len(results)==96
 report={'state':'PASS' if resolved>=90 and not mismatches else 'FAIL','protocol':'JOINT_ANCHOR_POLICY.v1.json','policy_sha256':sha(M/'JOINT_ANCHOR_POLICY.v1.json'),'binding_sha256':audit['binding_sha256'],'technical_binding_audit_sha256':sha(M/'ANCHOR_VALIDATION_BINDING_AUDIT.json'),'selection_sha256':audit['selection_sha256'],'frozen_observations_sha256':{n:sha(T/n) for n in ['BLIND_VALIDATION_A.json','BLIND_VALIDATION_B.json']},'source_review_binding_audit_sha256':sha(T/'SOURCE_REVIEW_BINDING_AUDIT.json'),'comparison_code_sha256':sha(__file__),'primary_volumes':96,'distinct_exam_groups':96,'resolved_model_anchors':resolved,'geometry_uncertain_aliases':uncertain,'mismatched_resolved_aliases':mismatches,'all_resolved_coarse_joint_centers_within_frozen_bounds':not mismatches,'measurement_scope':'Named model source-visual coarse joint intervals with>=2source frames; not clinician labels or segmented surface error; no all1172 accuracy claim','results':results,'training_started':False,'verified_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (T/'ANCHOR_VALIDATION_PASS.json').write_text(json.dumps(report,indent=2)+'\n');return report

def side_gate(A,B):
 # Initial observations remain immutable; supplementary source observations resolve abstentions.
 files=['BLIND_VALIDATION_A.json','BLIND_VALIDATION_B.json','BLIND_SUPPLEMENT_A.json','BLIND_SUPPLEMENT_B.json'];SA=source_observations(files[2],'A');SB=source_observations(files[3],'B');A=dict(A,**SA);B=dict(B,**SB)
 for group,obs in [('A',A),('B',B)]:
  n='BLIND_SUPPLEMENT_C011_'+group+'.json'
  if (T/n).exists():obs.update(source_observations(n,group));files.append(n)
 hyp=load('SIDE_HYPOTHESES.json',True);binding=load('ANCHOR_BUILD_BINDING.json',True);assert binding['side_hypotheses_sha256']==sha(M/'SIDE_HYPOTHESES.json');selection=load('VALIDATION_SELECTION.json');results={};mismatch=[];unresolved=[];manual={};primary=0;resolved=0
 for c in selection['cases']:
  alias=c['alias'];h=hyp['volumes'][c['stem']];assert h['source_receipt_sha256']==c['source_receipt_sha256'];pred='right' if h['source_mean_logit']>0 else 'left';auto=abs(h['source_mean_logit'])>=3.5 and abs(h['hflip_mean_logit'])>=3.5 and h['all_source_frames_same_sign'] and h['all_mirror_frames_same_sign'] and h['mirror_sign_concordant']
  if alias.startswith('V'):
   assert auto,'PRIMARY_CASE_NOT_ELIGIBLE_FOR_AUTOMATIC_POLICY';primary+=1;o=(A if int(alias[1:])<49 else B)[alias]
   if not o['resolved']:unresolved.append(alias)
   else:
    resolved+=1
    if o['side']!=pred:mismatch.append(alias)
   results[alias]={'stem':c['stem'],'source_side':o['side'],'model_side':pred,'resolved_source':o['resolved'],'automatic_policy_eligible':bool(auto),'source_witnesses':o['witnesses'],'confidence':o['confidence']}
  else:
   aa=A[alias];bb=B[alias];ok=aa['resolved'] and bb['resolved'] and aa['side']==bb['side'];manual[c['stem']]={'alias':alias,'accepted':bool(ok),'source_side':aa['side'] if ok else None,'reviewer_A_side':aa['side'],'reviewer_B_side':bb['side'],'reviewer_A_witnesses':aa['witnesses'],'reviewer_B_witnesses':bb['witnesses'],'reason':None if ok else 'NO_TWO_INDEPENDENT_SOURCE_SIDES_AGREE'}
 assert primary==96 and len(manual)==16
 report={'state':'PASS' if resolved>=90 and not mismatch else 'FAIL','protocol_semantic_sha256':load('ANATOMICAL_QA_PROTOCOL.v1.json')['semantic_sha256'] if 'semantic_sha256' in load('ANATOMICAL_QA_PROTOCOL.v1.json') else load('ANATOMICAL_QA_PROTOCOL.v1.json').get('sha256'),'source_manifest_sha256':binding['source_manifest_sha256'],'side_hypotheses_sha256':sha(M/'SIDE_HYPOTHESES.json'),'selection_sha256':sha(T/'VALIDATION_SELECTION.json'),'frozen_observations_sha256':{n:sha(T/n) for n in files},'source_review_binding_audit_sha256':sha(T/'SOURCE_REVIEW_BINDING_AUDIT.json'),'comparison_code_sha256':sha(__file__),'primary_volumes':96,'primary_resolved':resolved,'primary_unresolved':unresolved,'resolved_side_mismatches':mismatch,'named_visual_model_review_not_clinician_certified':True,'no_1172_population_accuracy_guarantee':True,'manual_challenge_accepted':sum(x['accepted'] for x in manual.values()),'manual_challenge_excluded':sum(not x['accepted'] for x in manual.values()),'manual_decisions':manual,'primary_results':results,'training_started':False,'verified_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (T/'SIDE_VALIDATION_PASS.json').write_text(json.dumps(report,indent=2)+'\n');return report
if __name__=='__main__':
 from validate_review_bindings import main as verify_all_reviews
 verify_all_reviews()
 A=source_observations('BLIND_VALIDATION_A.json','A');B=source_observations('BLIND_VALIDATION_B.json','B');assert len(A)==len(B)==64
 ar=anchor_gate(A,B);print(json.dumps({k:v for k,v in ar.items() if k!='results'},indent=2))
 if '--anchor-only' not in sys.argv:
  sr=side_gate(A,B);print(json.dumps({k:v for k,v in sr.items() if k not in ['primary_results','manual_decisions']},indent=2))
