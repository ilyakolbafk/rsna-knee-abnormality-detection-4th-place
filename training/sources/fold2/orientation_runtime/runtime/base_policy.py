"""Frozen v7 candidate routing. Confidence is evidence agreement, not guaranteed truth."""
import math

def series_decision(s):
 r=dict(s);r['image_side']='';r['side_reason']='insufficient_image_evidence';act=s.get('orientation_action','not_scored');r['pose_usable']=act in ('normal','corrected_candidate')
 if not r['pose_usable']:r['side_reason']=act;return r
 corrected=act=='corrected_candidate';pre='corrected_' if corrected else ''
 if s['plane']=='Sagittal':
  q=s.get(pre+'sequence_p_left')
  if q is not None:r['image_side']='L' if q>=.95 else 'R' if q<=.05 else '';r['side_score']=q
 else:
  q=s.get(pre+'frame_mean_p_left');err=s.get(pre+'mirror_complement_error',1);nl=s.get(pre+'strong_left_frames',0);nr=s.get(pre+'strong_right_frames',0)
  if q is not None:
   r['side_score']=q
   if err<=.15:
    if q>=.95 and nl>=3 and nr==0:r['image_side']='L'
    if q<=.05 and nr>=3 and nl==0:r['image_side']='R'
 if r['image_side']:r['side_reason']='ordered7_frames' if s['plane']=='Sagittal' else 'frame_and_mirror_agreement'
 return r

def study_decision(series,old):
 votes={'L':set(),'R':set()};by={'L':[],'R':[]}
 for s in series:
  side=s.get('image_side','')
  if side:votes[side].add(s['plane']);by[side].append(s['sid'])
 good=[s for s in ('L','R') if len(votes[s])>=2];conflict=bool(votes['L'] and votes['R']);new='';status='unresolved';support='';reason='insufficient_agreement'
 if len(good)==2:status='bilateral_candidate';reason='both_sides_supported_in_multiple_planes'
 elif len(good)==1 and not conflict:new=good[0];status='resolved_image';support='two_or_three_planes';reason='multi_plane_agreement'
 elif conflict:status='series_conflict';reason='opposite_series_do_not_yet_form_two_complete_knees'
 else:
  tag=old.get('side','') if old.get('tag_status')=='tag_consistent' else '';geom=old.get('geometry_side','') if old.get('geometry_gate') in (True,'True','true',1) else ''
  # Metadata fallback only preserves existing side; cannot silently reverse a case.
  if tag and tag==geom and not votes['R' if tag=='L' else 'L']:
   new=tag;status='resolved_metadata';support='tag_and_geometry';reason='agreed_metadata_without_image_opposition'
  elif geom and len(votes[geom])>=1 and not votes['R' if geom=='L' else 'L']:
   new=geom;status='resolved_geometry_image';support='geometry_and_image_plane';reason='image_supports_geometry'
  elif tag and len(votes[tag])>=1 and not votes['R' if tag=='L' else 'L']:
   new=tag;status='resolved_tag_image';support='tag_and_image_plane';reason='image_supports_consistent_tag'
 return {'side':new,'status':status,'support':support,'reason':reason,'planes_left':sorted(votes['L']),'planes_right':sorted(votes['R']),'series_left':by['L'],'series_right':by['R'],'orientation_candidates':[s['sid'] for s in series if s.get('orientation_action')=='corrected_candidate'],'plane_review':[s['sid'] for s in series if s.get('orientation_action')=='review_plane'],'series_conflict':conflict,'fallback':'old_H1_input' if not new else None}
