"""Inference-only laterality/pose adapter. No disease labels or UID-specific rules."""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision.models import resnet50
from canonical_inputs import transform_spec_v3,frame_center
from base_policy import series_decision,study_decision
PLANES=['Axial','Coronal','Sagittal']
TYPES=['Sagittal_fluid','Sagittal_structural','Coronal_fluid','Coronal_structural','Axial_fluid','Axial_structural']
class Net(nn.Module):
 def __init__(self):
  super().__init__();self.backbone=nn.Sequential(*list(resnet50(weights=None).children())[:9]);self.side=nn.Sequential(nn.Linear(2055,256),nn.ReLU(),nn.Dropout(.1),nn.Linear(256,1));self.rotation=nn.Sequential(nn.Linear(2048,128),nn.ReLU(),nn.Linear(128,4))
class Sequence(nn.Module):
 def __init__(self):super().__init__();self.f=nn.Sequential(nn.Linear(6144,128),nn.GELU(),nn.Linear(128,1))
 def forward(self,d):return ((self.f(d)-self.f(-d))/2).squeeze(-1)
class Pose(nn.Module):
 def __init__(self):
  super().__init__();self.body=nn.Sequential(nn.Linear(2048,256),nn.GELU(),nn.Dropout(.1));self.plane=nn.Linear(256,3);self.rot=nn.Linear(256,4);self.sagpose=nn.Linear(256,8)
 def forward(self,f):z=self.body(f);return self.plane(z),self.rot(z),self.sagpose(z)
def delta(z):return torch.cat([z[:,6]-z[:,0],z[:,5]-z[:,1],z[:,4]-z[:,2]],1)
def file_sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
 return h.hexdigest()
def pose_summary(pl,po,indices):
 p=pl[indices].mean(0);r=po[indices].mean(0);return {'image_plane':PLANES[int(p.argmax())],'image_plane_confidence':float(p.max()),'pose_class':int(r.argmax()),'pose_confidence':float(r.max())}
def normalize_resized_series(float_images,normalization_indices=None,valid=None):
 """130mm->336 float32 images; exact legacy selected-center occurrences or all extra-series frames."""
 x=np.asarray(float_images,np.float32);assert x.ndim==3 and x.shape[1:]==(336,336)
 v=np.ones(len(x),bool) if valid is None else np.asarray(valid,bool);ids=np.arange(len(x)) if normalization_indices is None else np.asarray(normalization_indices,int);values=x[ids][v[ids]];values=values[np.isfinite(values)]
 if not len(values):raise ValueError('No valid normalization pixels')
 lo,hi=np.percentile(values,[.5,99.5]);lo,hi=np.float64(lo),np.float64(hi)
 q=np.zeros_like(x,dtype=np.uint8) if hi<=lo else np.rint(np.clip((x-lo)/(hi-lo),0,1).astype(np.float32)*255).clip(0,255).astype(np.uint8);q[~v]=0
 return q,{'low':float(lo),'high':float(hi),'normalization':'legacy_center_occurrences' if normalization_indices is not None else 'all_valid_extra_series_frames'}
class LateralityV7:
 def __init__(self,root=None,device='cpu'):
  self.root=Path(root or Path(__file__).parent);self.device=torch.device(device);manifest=json.loads((self.root/'MANIFEST.json').read_text())
  for n,r in manifest['weights'].items():
   if file_sha(self.root/n)!=r['sha256']:raise ValueError('Laterality checkpoint identity mismatch: '+n)
  self.net=Net().to(self.device);self.net.load_state_dict(torch.load(self.root/'side.pt',map_location='cpu',weights_only=True)['model']);self.net.eval()
  ck=torch.load(self.root/'sagittal.pt',map_location='cpu',weights_only=True);self.seq=Sequence().to(self.device);self.seq.load_state_dict(ck['head']);self.seq.eval();self.seq_scale=ck['scale'].to(self.device)
  ck=torch.load(self.root/'pose.pt',map_location='cpu',weights_only=True);self.pose=Pose().to(self.device);self.pose.load_state_dict(ck['head']);self.pose.eval();self.pose_scale=ck['scale'].to(self.device);self.depth={}
  if (self.root/'depth.pt').exists():
   for plane,ck in torch.load(self.root/'depth.pt',map_location='cpu',weights_only=True)['heads'].items():
    h=Sequence().to(self.device);h.load_state_dict(ck['head']);h.eval();self.depth[plane]=(h,ck['scale'].to(self.device))
 @torch.inference_mode()
 def _features(self,a):
  x=torch.tensor(np.ascontiguousarray(a),device=self.device).float()[:,None].expand(-1,3,-1,-1)/127.5-1
  dtype=torch.bfloat16 if self.device.type=='cuda' and torch.cuda.get_device_capability(self.device)[0]>=8 and torch.cuda.is_bf16_supported() else torch.float16
  with torch.autocast(self.device.type,dtype=dtype,enabled=self.device.type=='cuda'):
   return self.net.backbone(x.contiguous(memory_format=torch.channels_last)).flatten(1).float()
 @torch.inference_mode()
 def _score(self,images,percentiles,plane,fluid):
  f=self._features(images);ff=self._features(images[:,:,::-1]);kind=TYPES.index(plane+('_fluid' if fluid else '_structural'));k=torch.full((len(images),),kind,device=self.device);p=torch.tensor(percentiles,device=self.device,dtype=torch.float32);md=torch.cat((nn.functional.one_hot(k,6).float(),p[:,None]),1);q=self.net.side(torch.cat((f,md),1)).squeeze(1).sigmoid().cpu().numpy();qm=self.net.side(torch.cat((ff,md),1)).squeeze(1).sigmoid().cpu().numpy();pl,r,s=self.pose(f/self.pose_scale);pl=pl.softmax(1).cpu().numpy();po=(s if plane=='Sagittal' else r).softmax(1).cpu().numpy();central=sorted(set(int(np.argmin(abs(np.asarray(percentiles)-v))) for v in [.25,.5,.75]));res=pose_summary(pl,po,central)
  res.update(frame_mean_p_left=float(q.mean()),strong_left_frames=int((q>=.9).sum()),strong_right_frames=int((q<=.1).sum()),mirror_complement_error=float(abs(q+qm-1).mean()),frame_scores=[{'percentile':float(p),'p_left':float(a),'mirror_p_left':float(b),'plane_prob':c.tolist(),'pose_prob':d.tolist()} for p,a,b,c,d in zip(percentiles,q,qm,pl,po)])
  if len(images)==7:
   if plane=='Sagittal':res['sequence_p_left']=float(self.seq(delta(f[None])/self.seq_scale).sigmoid()[0])
   elif plane in self.depth:
    h,sc=self.depth[plane];res['depth_forward_probability']=float(h(delta(f[None])/sc).sigmoid()[0])
  return res
 @torch.inference_mode()
 def predict_prepared(self,images224,percentiles,plane,fluid=False,**metadata):
  a=np.asarray(images224);assert a.dtype==np.uint8 and a.shape[1:]==(224,224) and len(a)>=3;assert plane in PLANES
  result={**metadata,'plane':plane,**self._score(a,percentiles,plane,fluid)};act='review_plane' if result['image_plane']!=plane or result['image_plane_confidence']<.9 else 'review_pose' if result['pose_confidence']<.9 else 'normal' if result['pose_class']==0 else 'candidate_inverse_d4';result['orientation_action']=act
  if act=='candidate_inverse_d4':
   aug=result['pose_class'];fixed=np.rot90(a,-(aug%4),axes=(-2,-1))
   if plane=='Sagittal' and aug>=4:fixed=fixed[:,:,::-1]
   check=self._score(fixed,percentiles,plane,fluid);result['post_correction_pose']={k:check[k] for k in ('image_plane','image_plane_confidence','pose_class','pose_confidence')};result['orientation_transform']={'rot90_k':-(aug%4),'flip_x':plane=='Sagittal' and aug>=4,'order':'rotate_then_optional_flip'};result['orientation_action']='corrected_candidate' if check['image_plane']==plane and check['image_plane_confidence']>=.9 and check['pose_class']==0 and check['pose_confidence']>=.9 else 'review_failed_cycle'
   for key in ['frame_mean_p_left','strong_left_frames','strong_right_frames','mirror_complement_error','sequence_p_left']: 
    if key in check:result['corrected_'+key]=check[key]
  return series_decision(result)
 def predict_source_series(self,images336,geometry_tags,plane,fluid=False,valid=None,**metadata):
  """Input intensity must follow130mm source-bank contract; never reuse140mm H1 intensity normalization."""
  a=np.asarray(images336);assert a.dtype==np.uint8 and a.shape[1:]==(336,336) and len(a)==len(geometry_tags);axis={'Sagittal':0,'Coronal':1,'Axial':2}[plane];entries=[]
  for i,g in enumerate(geometry_tags):
   if valid is not None and not valid[i]:continue
   try:sp=transform_spec_v3(g.get('ImageOrientationPatient'),plane,'L');c=frame_center(g)
   except (KeyError,ValueError,TypeError):continue
   if sp is not None and c is not None:entries.append((float(c[axis]),i,sp))
  if len(entries)<3:return {**metadata,'plane':plane,'image_side':'','orientation_action':'not_scored','side_reason':'less_than3_valid_frames'}
  entries.sort();co=np.array([x[0] for x in entries]);lo,hi=co[0],co[-1]
  if hi-lo<1e-3 or min(np.diff(co))<1e-4:return {**metadata,'plane':plane,'image_side':'','orientation_action':'not_scored','side_reason':'duplicate_or_degenerate_geometry'}
  picks=sorted(set(int(np.argmin(abs(co-x))) for x in np.linspace(lo,hi,7)));ims=[];pct=[];provenance=[]
  for ix in picks:
   coord,i,sp=entries[ix];im=a[i]
   if sp['transpose']:im=im.T
   if sp['flip_x']:im=im[:,::-1]
   if sp['flip_y']:im=im[::-1,:]
   ims.append(np.asarray(Image.fromarray(np.ascontiguousarray(im)).resize((224,224),Image.Resampling.BILINEAR)));pct.append(float((coord-lo)/(hi-lo)));provenance.append({'source_index':i,'generic_transform':sp})
  r=self.predict_prepared(np.stack(ims),pct,plane,fluid,**metadata);r['sampled_source_frames']=provenance;return r
 @staticmethod
 def decide_study(series,metadata_record):return study_decision(series,metadata_record)
