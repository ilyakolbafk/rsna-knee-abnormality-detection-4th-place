"""Image-relative joint proposals from generic femur/tibia labels, no disease data."""
import numpy as np
import torch
from skimage.transform import resize
from scipy import ndimage as ndi
def segment_canonical(raw,flip,net,batch=4):
 display=np.asarray(raw[:,::-1,:],dtype=np.float32)
 if flip:display=display[:,:,::-1]
 vol=np.ascontiguousarray(display.transpose(2,1,0));v=resize(vol,(160,512,vol.shape[-1]),order=3);v=(v-np.mean(v))/(np.std(v)+1e-8);x=np.ascontiguousarray(v.transpose(2,0,1)[:,None],dtype=np.float32);labels=[]
 with torch.inference_mode():
  for i in range(0,len(x),batch):labels.append(net(torch.from_numpy(x[i:i+batch]).cuda()).argmax(1).cpu().numpy().astype(np.uint8))
 small=np.concatenate(labels);back=resize(small.transpose(1,2,0),vol.shape,order=0,preserve_range=True,anti_aliasing=False).transpose(2,1,0).astype(np.uint8)
 return np.ascontiguousarray(back)
def propose_joint(seg,spacing_xy):
 n,H,W=seg.shape;dx,dy=map(float,spacing_xy);counts={str(k):int(v) for k,v in zip(*np.unique(seg,return_counts=True))};per=[]
 for z in range(round(.25*(n-1)),round(.8*(n-1))+1):
  f=seg[z]==7;t=seg[z]==8;okf=f.any(0);okt=t.any(0);cols=np.flatnonzero(okf&okt)
  if len(cols)<max(8,20/dx):continue
  fend=(H-1)-np.argmax(f[::-1],axis=0);tstart=np.argmax(t,axis=0);gap=(tstart[cols]-fend[cols])*dy;good=(gap>=-2)&(gap<=15);cols=cols[good];gap=gap[good]
  if len(cols)<max(8,20/dx):continue
  lo,hi=np.percentile(cols,[5,95]);row=float(np.median((fend[cols]+tstart[cols])/2));col=float((lo+hi)/2)
  per.append({'source_slice':z,'display_row_col':[row,col],'joint_gap_median_mm':float(np.median(gap)),'common_joint_width_mm':float((hi-lo+1)*dx),'valid_columns':len(cols)})
 if per:
  coords=np.array([x['display_row_col'] for x in per]);med=np.median(coords,axis=0);mad=np.median(np.abs(coords-med),axis=0)*np.array([dy,dx]);anchor=med.tolist()
 else:anchor=None;mad=np.array([np.inf,np.inf])
 issues=[]
 if counts.get('7',0)<seg.size*.015:issues.append('FEMUR_VOLUME_LOW')
 if counts.get('8',0)<seg.size*.01:issues.append('TIBIA_VOLUME_LOW')
 if len(per)<3:issues.append('FEWER_THAN3_JOINT_FRAMES')
 if mad[0]>8 or mad[1]>6:issues.append('JOINT_ANCHOR_UNSTABLE')
 if anchor and not (.2*H<anchor[0]<.85*H and .2*W<anchor[1]<.8*W):issues.append('JOINT_OUTSIDE_EXPECTED_IMAGE_INTERIOR')
 return {'class_counts':counts,'joint_anchor_canonical_display_row_col':anchor,'joint_anchor_MAD_row_col_mm':[float(x) if np.isfinite(x) else None for x in mad],'valid_joint_frames':per,'geometry_issues':issues,'state':'PROPOSAL_GEOMETRY_PASS_NOT_VISUAL_VALIDATED' if not issues else 'UNCERTAIN_BONE_ANCHOR','model_classes_used':{'7':'femur','8':'tibia'},'no_pathology_labels_used':True}
