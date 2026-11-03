"""Label-free total laterality routing. High agreement is not calibrated correctness."""
from pathlib import Path
import numpy as np
from base_runtime import LateralityV7
from base_policy import series_decision,study_decision
from canonical_inputs import frame_center
from policy_v8 import resolve

class LateralityResolver(LateralityV7):
 def score_source_series(self,images336,geometry_tags,plane,fluid=False,valid=None,sop_ids=None,**metadata):
  r=self.predict_source_series(images336,geometry_tags,plane,fluid,valid,**metadata)
  picks=r.get('sampled_source_frames',[]);frames=[]
  for fs,pr in zip(r.get('frame_scores',[]),picks):
   i=pr['source_index'];frames.append(dict(fs,horizontal_mirror_p_left=fs['mirror_p_left'],center_lps=np.asarray(frame_center(geometry_tags[i]),float).tolist(),sop=sop_ids[i] if sop_ids else str(i)))
  r['frames']=frames;ax={'Axial':2,'Coronal':1,'Sagittal':0}[plane]
  r['span_mm']=float(np.ptp([f['center_lps'][ax] for f in frames])) if frames else 0.
  r['median_center_x_mm']=float(np.median([f['center_lps'][0] for f in frames])) if frames else 0.
  return r
 def resolve_scored(self,series,metadata_record=None,joint_z=None):
  metadata_record=metadata_record or {};pr=study_decision([series_decision(s) for s in series],metadata_record)
  return resolve(series,metadata_record,pr,joint_z)
 def predict_bank(self,arrays,metadata,joint_z=None):
  binding=metadata.get('binding',{})
  if binding.get('crop_mm')!=130:raise ValueError('Side model requires130mm source bank, not H1 140mm')
  plane_by={int(p['bank_index']):p for p in metadata['planes']};out=[]
  for sid,s in metadata['series'].items():
   inv=s['source_inventory'];plane=inv.get('plane',inv.get('Anatomical_Plane'))
   if plane not in ('Axial','Coronal','Sagittal'):continue
   ids=np.arange(s['offset'],s['offset']+s['source_count']);tags=[plane_by[int(i)]['geometry_tags'] for i in ids]
   r=self.score_source_series(arrays['images'][ids],tags,plane,bool(inv.get('fluid_sensitive',inv.get('Fluid_Sensitive',0))),valid=(arrays['decode_valid'][ids]&arrays['geometry_valid'][ids]),sop_ids=[plane_by[int(i)].get('SOPInstanceUID',str(i)) for i in ids],uid=metadata.get('uid',metadata['planes'][0].get('StudyInstanceUID')),sid=sid)
   out.append(r)
  return self.resolve_scored(out,metadata.get('side_record',{}),joint_z)
