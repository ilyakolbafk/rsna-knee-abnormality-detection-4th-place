"""Total, auditable routing. Disease labels are never inputs."""
import math
import numpy as np

def logit(q):
 q=float(np.clip(q,.00001,.99999));return float(np.clip(math.log(q/(1-q)),-3,3))
def sig(x):return 1/(1+math.exp(-float(np.clip(x,-30,30))))
def evidence(series,joint_z=None):
 out=[];sagspans=[s.get('span_mm',0) for s in series if s['plane']=='Sagittal'];maxspan=max(sagspans or [0])
 for s0 in series:
  s=dict(s0);plane=s['plane'];frames=s.get('frames',[]);s.update(v8_q=None,v8_weight=0.,v8_reason='no_valid_frames',v8_selected_sops=[])
  poseok=s.get('orientation_action')=='corrected_candidate';prefix='corrected_' if poseok else ''
  if plane=='Sagittal':
   q=s.get(prefix+'sequence_p_left',s.get('sequence_p_left'));full=s.get('span_mm',0)>=45 and s.get('span_mm',0)>=.65*maxspan
   if q is not None and np.isfinite(q):s.update(v8_q=float(q),v8_weight=1. if full else .1,v8_reason='full_sagittal_sequence' if full else 'partial_sagittal_sequence',v8_selected_sops=[r.get('sop','') for r in frames])
  elif frames and plane in ('Axial','Coronal'):
   if plane=='Axial':
    selected=[r for r in frames if joint_z is not None and np.isfinite(joint_z) and -35<=float(r['center_lps'][2])-joint_z<=-10]
    reason='joint_z_minus35_to_minus10' if selected else 'inferior_third_fallback'
    if not selected:selected=[r for r in frames if r['percentile']<=.34]
   else:selected=[r for r in frames if .5<=r['percentile']<=.85];reason='posterior_coronal_half'
   vals=[]
   for r in selected:
    q=r.get(prefix+'p_left',r.get('p_left'));qm=r.get(prefix+'horizontal_mirror_p_left',r.get('horizontal_mirror_p_left'))
    if q is None or qm is None:continue
    if np.isfinite(q) and np.isfinite(qm):vals.append(.5*(q+1-qm))
   if vals:
    weight=1.
    if plane=='Axial' and reason=='inferior_third_fallback' and joint_z is not None and np.isfinite(joint_z):
     zz=[float(r['center_lps'][2])-joint_z for r in frames]
     covered=min(zz)<=-10 and max(zz)>=-35
     weight=.25 if covered else .1
     reason+=';sparse_anatomical_sampling' if covered else ';no_fibular_level_coverage'
    s.update(v8_q=float(np.mean(vals)),v8_weight=weight,v8_reason=reason,v8_selected_sops=[r.get('sop','') for r in selected])
  # A weak pose check is not proof of an abnormal image. Only positive contrary evidence reduces its vote.
  if s.get('image_plane')!=plane and s.get('image_plane_confidence',0)>=.9:
   s['v8_weight']*=.1;s['v8_reason']+=';confident_plane_mismatch'
  if s.get('orientation_action')=='review_failed_cycle':s['v8_weight']*=.1;s['v8_reason']+=';pose_cycle_failed'
  s['v8_image_side']='L' if s['v8_q'] is not None and s['v8_q']>=.8 and s['v8_weight']>=.5 else 'R' if s['v8_q'] is not None and s['v8_q']<=.2 and s['v8_weight']>=.5 else ''
  out.append(s)
 return out

def ap_pattern(series):
 ax=any(s['plane']=='Axial' and s.get('orientation_action')=='corrected_candidate' and s.get('pose_class')==2 for s in series)
 sag=any(s['plane']=='Sagittal' and s.get('orientation_action')=='corrected_candidate' and s.get('pose_class')==4 for s in series)
 cor=[s for s in series if s['plane']=='Coronal' and s.get('depth_forward_probability') is not None]
 return bool(ax and sag and len(cor)>=2 and all(s['depth_forward_probability']<=.05 for s in cor))

def resolve(series,metadata,prior_decision=None,joint_z=None):
 ss=evidence(series,joint_z);pattern=ap_pattern(ss);votes={'L':set(),'R':set()};plane_scores={}
 for s in ss:
  if s['v8_image_side']:votes[s['v8_image_side']].add(s['plane'])
 for plane in ['Axial','Coronal','Sagittal']:
  g=[s for s in ss if s['plane']==plane and s['v8_q'] is not None and s['v8_weight']>0]
  if g:plane_scores[plane]=sum(logit(s['v8_q'])*s['v8_weight'] for s in g)/max(1.,sum(s['v8_weight'] for s in g))
 # Distinct knees require image support in at least two planes for each side.
 bilateral=not pattern and all(len(votes[x])>=2 for x in ['L','R'])
 oldside=metadata.get('side','');tag=oldside if metadata.get('tag_status')=='tag_consistent' else '';geo=metadata.get('geometry_side','') if metadata.get('geometry_gate') in (True,'True',1) else ''
 if pattern:
  # Physical side from coronal/Sag agreement; Ax appearance is separately canonicalized.
  total=plane_scores.get('Coronal',0)+plane_scores.get('Sagittal',0)
 else:total=2*plane_scores.get('Axial',0)+plane_scores.get('Coronal',0)+plane_scores.get('Sagittal',0)
 if tag:total+=.5*(1 if tag=='L' else -1)
 if geo:total+=.25*(1 if geo=='L' else -1)
 side='L' if total>0 else 'R' if total<0 else oldside if oldside in ('L','R') else 'L'
 prior=prior_decision or {};preserved=prior.get('side') in ('L','R') and not pattern and not bilateral
 if preserved:side=prior['side']
 opposite='R' if side=='L' else 'L';high=len(votes[side])>=2 and not votes[opposite];medium=any(abs(v)>2.2 for v in plane_scores.values()) and not votes[opposite]
 tier='high_agreement' if high else 'medium_evidence' if medium or tag==geo==side else 'weak_best_effort'
 if pattern:tier='cross_plane_pose_correction'
 if bilateral:tier='bilateral_partition'
 for s in ss:
  s['assigned_side']=s['v8_image_side'] or ('L' if (s['v8_q'] or .5)>=.5 else 'R') if bilateral else side
  if bilateral and not s['v8_image_side']:
   # Low evidence series: use its own continuous score, then series geometry only for an exact tie.
   q=s['v8_q'];s['assigned_side']='L' if q is not None and q>.5 else 'R' if q is not None and q<.5 else 'L' if s.get('median_center_x_mm',0)>0 else 'R'
  s['appearance_side']=s['assigned_side']
  if pattern and s['plane']=='Axial':s['appearance_side']='L' if plane_scores.get('Axial',0)>0 else 'R'
  s['apply_pose']=s.get('orientation_transform') if s.get('orientation_action')=='corrected_candidate' else None
  s['reverse_depth_extra']=bool(pattern and s['plane']=='Coronal')
  s['depth_status']='corrected_by_cross_plane_AP_pattern' if s['reverse_depth_extra'] else 'reverse_candidate_not_applied' if s.get('depth_forward_probability',1)<=.05 else 'uncertain' if s.get('depth_forward_probability',1)<.95 else 'not_independently_scored' if s.get('depth_forward_probability') is None else 'dicom_order_supported'
 return {'side':side,'confidence_tier':tier,'bilateral':bilateral,'routes':['L','R'] if bilateral else [side],'selection_reason':'v7_supported_decision_preserved' if preserved else 'cross_plane_AP_pattern' if pattern else 'coverage_and_anatomical_frame_fusion','evidence_score':total,'evidence_p_left':sig(total),'score_is_calibrated':False,'plane_scores':plane_scores,'planes_left':sorted(votes['L']),'planes_right':sorted(votes['R']),'series':ss,'old_side':oldside,'ap_pattern':pattern,'no_evidence_fallback':not plane_scores}
