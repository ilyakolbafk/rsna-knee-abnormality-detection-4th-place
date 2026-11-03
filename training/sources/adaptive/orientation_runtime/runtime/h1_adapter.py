"""Canonical L token adapter shared by train overlays and H1 Kaggle inference.
Pixel input is ORIGINAL140mm H1 uint8, not the side classifier130mm crop.
"""
import numpy as np
import torch
from canonical_inputs import adapt_inputs

def apply_tokens(raw_tokens,arrays,metadata,sample,decision,route=None):
 side=route or decision['side']
 if side not in ('L','R'):raise ValueError('Missing resolved side')
 if decision['bilateral'] and route is None:raise ValueError('Bilateral input requires a separate route for each knee')
 bysid={s['sid']:s for s in decision['series']};byref={int(p['bank_index']):p for p in metadata['planes']}
 raw=torch.as_tensor(raw_tokens).clone();record={'side':'L','source':'generic_geometry_base'}
 generic,_,_=adapt_inputs(raw,arrays,metadata,sample,{'canonical_side':True,'position':True,'fat_suppression':True},record)
 _,features,_=adapt_inputs(torch.zeros_like(raw),arrays,metadata,sample,{'canonical_side':True,'position':True,'fat_suppression':True},{'side':side,'source':'laterality_v8'})
 mask=np.array(sample['token_mask'],bool).copy();audit=[]
 for token in np.flatnonzero(mask):
  ref=int(sample['center_refs'][token]);sid=byref[ref]['SeriesInstanceUID'];s=bysid[sid]
  if decision['bilateral'] and s['assigned_side']!=side:mask[token]=False;continue
  im=generic[token];pose=s.get('apply_pose');plane=s['plane']
  if pose:
   im=torch.rot90(im,int(pose['rot90_k']),(-2,-1))
   if pose.get('flip_x'):im=im.flip(-1)
  appearance=s.get('appearance_side',side)
  if plane in ('Axial','Coronal') and appearance=='R':im=im.flip(-1)
  if plane=='Sagittal' and appearance=='R':im=im.flip(0)
  if s.get('reverse_depth_extra'):im=im.flip(0);features[token,:2]*=-1
  generic[token]=im;audit.append({'token':int(token),'sid':sid,'side':side,'appearance_side':appearance,'pose':pose,'extra_depth_reverse':bool(s.get('reverse_depth_extra'))})
 chosen=np.flatnonzero(mask)
 if not len(chosen):raise ValueError('No usable windows for selected knee: rebuild five semantic slots within this knee before inference')
 return (generic[chosen].float()/255.,torch.ones(len(chosen),dtype=torch.bool),features[chosen]),audit
