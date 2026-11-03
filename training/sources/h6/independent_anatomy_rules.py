"""Header/output-based checks of the public prior and patella selection/project."""
import numpy as np

def validate(native,anchors):
 per={}
 for s in native['series']:
  if not s['geometry']['quality']['aligned']:continue
  aff=np.asarray(s['affines_col_row_origin_lps_mm'],float);h,w=s['native_shape'][-2:];point=aff[len(aff)//2]@np.array([(w-1)/2,(h-1)/2,1.]);per.setdefault(s['plane'],[]).append(point)
 assert per,'NO_PRIOR_GEOMETRY'
 expected=[]
 for ax,planes in enumerate([['Sagittal'],['Coronal'],['Sagittal','Coronal']]):
  cs=[c for pl in planes for c in per.get(pl,[])] or [c for pl in per for c in per[pl]];expected.append(float(np.median([c[ax] for c in cs])))
 assert np.allclose(expected,anchors['slab_prior_lps'],rtol=0,atol=1e-9),'PUBLIC_HEADER_PRIOR'
 candidates=list(anchors['audit']['candidate_audit'].values());accepted=[c for c in candidates if c.get('accepted_after_prior_gate')]
 if accepted:
  accepted.sort(key=lambda c:-c['joint_mass_pixels']);best=accepted[0];close=[c for c in accepted if np.linalg.norm(np.asarray(c['joint_lps'])-best['joint_lps'])<=20.]
  pc=[c for c in close if c.get('patella_lps') is not None] or [c for c in accepted if c.get('patella_lps') is not None];Xp=max(pc,key=lambda c:c['patella_mass_pixels'])['patella_lps'] if pc else None
 else:Xp=None
 assert (Xp is None)==(anchors['patella_lps'] is None),'PATELLA_SELECTION_NULL'
 if Xp is not None:
  assert np.array_equal(Xp,anchors['patella_lps']),'PATELLA_SELECTION_MAX_MASS'
  for s in native['series']:
   aff=np.asarray(s['affines_col_row_origin_lps_mm'],float);origins=aff[:,:,2];normal=np.cross(aff[0,:,0],aff[0,:,1]);normal/=np.linalg.norm(normal);z=int(np.argmin(np.abs((origins-Xp)@normal)));cr=np.linalg.lstsq(aff[z,:,:2],np.asarray(Xp)-origins[z],rcond=None)[0];h,w=s['native_shape'][-2:];fraction=[cr[1]/max(1,h-1),cr[0]/max(1,w-1)];assert np.allclose(fraction,anchors['series'][s['SeriesInstanceUID']]['patella'],rtol=0,atol=1e-10),'PATELLA_PROJECTION'
 return True
