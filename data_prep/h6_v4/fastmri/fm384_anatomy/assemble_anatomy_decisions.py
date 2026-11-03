"""Deterministic whole-volume admission. No disease labels enter side/joint estimation."""
from pathlib import Path
import json,collections,time
import numpy as np
from anatomy_job import ROOT,BASE,write
from fm_proof_chain import current_inputs,load,sha,require

def main():
 inp=current_inputs(ROOT,BASE);src=load(BASE/'receipts/SOURCE_FULLSHA_MANIFEST.json');S=load(ROOT/'SIDE_VALIDATION_PASS.json');A=load(ROOT/'ANCHOR_VALIDATION_PASS.json');H=load(ROOT/'SIDE_HYPOTHESES.json')['volumes'];audit=load(ROOT/'ANCHOR_CORPUS_BINDING_AUDIT.json');D={}
 side_unreadable={S['primary_results'][a]['stem'] for a in S['primary_unresolved']};joint_failed={A['results'][a]['stem'] for a in A['geometry_uncertain_aliases']};manual=S['manual_decisions'];pol=load(ROOT/'DECISION_POLICY.v1.json')
 for r in sorted(src['volumes'],key=lambda r:r['stem']):
  require(r['files']['rss.f32.npy']['stat']['bytes']>0,'EMPTY_SOURCE');s=r['stem'];h=H[s];p=load(ROOT/'anchor_proposals'/(s+'.json'));reasons=[];b=load(BASE/'source_rss'/s/'boxes.source.json');sp=b['spatial'];study=b['study_level'];require(h['source_receipt_sha256']==r['receipt_sha256']==p['row_binding']['source_receipt_sha256'],'SOURCE_BINDING')
  auto=abs(h['source_mean_logit'])>=3.5 and abs(h['hflip_mean_logit'])>=3.5 and h['all_source_frames_same_sign'] and h['all_mirror_frames_same_sign'] and h['mirror_sign_concordant']
  if s in manual:
   m=manual[s];side=m['source_side'] if m['accepted'] else None;side_method='two_independent_frozen_source_fibula_reviews'
  elif auto:side='right' if h['source_mean_logit']>0 else 'left';side_method='validated_R6_consistent_source_and_mirror'
  else:side=None;side_method='unresolved'
  if s in side_unreadable:reasons.append('INDEPENDENT_SOURCE_SIDE_UNREADABLE');side=None
  if side not in ['left','right']:reasons.append('ANATOMICAL_LATERAL_SIDE_UNRESOLVED')
  if study and not sp:reasons.append('ARTIFACT_ONLY_STUDY_FINDINGS_NO_SPATIAL_NEGATIVE_LABELS')
  if s in pol['malformed_annotation']:reasons.append('MALFORMED_SPATIAL_ANNOTATION_CANNOT_MASK_IN_FIXED_LOADER')
  if s in joint_failed:reasons.append('INDEPENDENT_PRIMARY_JOINT_GEOMETRY_UNCERTAIN')
  issues=list(p['proposal']['geometry_issues'])
  if issues:reasons.append('JOINT_GEOMETRY_UNCERTAIN:'+','.join(issues))
  meta=load(BASE/'source_rss'/s/'metadata.json');require(meta['rss_shape'][1:]==[320,320],'NATIVE_RSS_SHAPE_CHANGED');sa=p['source_anchor_display_row_col']
  if sa is None or len(sa)!=2 or not np.isfinite(sa).all():reasons.append('NO_FINITE_SOURCE_JOINT_ANCHOR')
  flip=side=='left';ca=[float(sa[0]),float(319-sa[1] if flip else sa[1])] if sa is not None else None
  D[s]={'accepted':not reasons,'exclude_reason':';'.join(reasons) if reasons else None,'source_receipt_sha256':r['receipt_sha256'],'proposal_sha256':audit['proposal_sha256'][s],'source_lateral_image_side':side,'side_method':side_method,'hflip_to_lateral_pluscol':flip if side else None,'joint_method':p['method'],'joint_anchor_source_display_row_col':sa,'joint_anchor_canonical_display_row_col':ca,'joint_geometry_issues':issues,'reviewed_no_findings':not sp and not study,'manual_challenge_alias':manual.get(s,{}).get('alias'),'no_pathology_inputs_used_for_side_or_joint':True}
 require(sum(d['accepted'] for d in D.values())>=1000,'FEWER_THAN_1000_ANATOMICAL_VOLUMES')
 counts=collections.Counter(reason for d in D.values() for reason in (d['exclude_reason'] or '').split(';') if reason)
 out={'gate_state':'PASS','inputs':inp,'volumes':D,'accepted_volumes':sum(d['accepted'] for d in D.values()),'excluded_volumes':sum(not d['accepted'] for d in D.values()),'exclusion_reason_counts':dict(counts),'uncertainty_handling':'actual_whole_volume_exclusion_all_views_no_pixel_rows','private_pixel_parity':False,'training_started':False,'created_unix':time.time()}
 path=ROOT/'ANATOMY_DECISIONS.json'
 if path.exists():
  old=load(path);require(old['inputs']==inp and old['volumes']==D,'FROZEN_DECISION_CHANGE_REQUIRES_NEW_NAMESPACE');return old
 write(path,out);return out
if __name__=='__main__':
 x=main();print(json.dumps({k:v for k,v in x.items() if k not in ['volumes','inputs']},indent=2))
