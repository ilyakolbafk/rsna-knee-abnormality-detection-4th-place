"""Public v2 augmentation; stateless private per-update streams for exact resume."""
import math
import torch
from torch.nn import functional as F
PERM=[0,1,3,2,5,4,6,7,8,9,10,11]

def apply(batch,seed,epoch,step,force_mirror=None,generators=None):
    gen,mirror_gen=generators if generators is not None else (
        torch.Generator().manual_seed(5000+17*seed+epoch*1000003+step),
        torch.Generator().manual_seed(9400+17*seed+epoch*1000003+step))
    img=batch['x'];B,W,C,S,_=img.shape
    oy=int(torch.randint(0,S-352+1,(1,),generator=gen));ox=int(torch.randint(0,S-352+1,(1,),generator=gen))
    x=img[...,oy:oy+352,ox:ox+352].float().reshape(B*W,C,352,352)
    angle=(torch.rand(B,generator=gen)*20-10)*math.pi/180;scale=torch.rand(B,generator=gen)*.2+.9
    cos,sin=angle.cos()/scale,angle.sin()/scale
    theta=torch.stack([torch.stack([cos,-sin,torch.zeros(B)],1),torch.stack([sin,cos,torch.zeros(B)],1)],1)
    theta=theta.repeat_interleave(W,0).to(x.device)
    with torch.autocast('cuda',enabled=False):
        grid=F.affine_grid(theta.float(),x.shape,align_corners=False)
        x=F.grid_sample(x.float(),grid,mode='bilinear',padding_mode='zeros',align_corners=False)
    gamma=(torch.rand(B,generator=gen)*.45+.8).repeat_interleave(W).to(x.device)[:,None,None,None]
    contrast=(torch.rand(B,generator=gen)*.2+.9).repeat_interleave(W).to(x.device)[:,None,None,None]
    x=(((x/255).clamp(0,1).pow(gamma)-.5)*contrast+.5).clamp(0,1).mul(255).round().to(torch.uint8).reshape(B,W,C,352,352)
    out=dict(batch);out['x']=x
    mirror=bool(torch.rand((),generator=mirror_gen)<.5) if force_mirror is None else force_mirror
    if mirror and bool((batch['meta'][...,7][batch['mask']]>=0).all()):
        sag=batch['meta'][...,7].round()==0
        out['x']=torch.where(sag[:,:,None,None,None],x.flip(2),x.flip(-1))
        out['meta']=batch['meta'].clone();out['meta'][...,2]=torch.where(sag,-out['meta'][...,2],out['meta'][...,2])
        view = out['meta'][...,1].clone()
        out['meta'][...,1] = torch.where(view == 2, 3, torch.where(view == 3, 2, view))
        for key in ['main_lr','main_weight','main_probability','oai_vm_lr','oai_vm_weight','oai_vm_probability']:
            if key in batch:out[key]=batch[key][:,PERM].clone()
        out['main_weight'][:,1]=0;out['oai_vm_weight'][:,1]=0
        out['mirrored']=True
    else:out['mirrored']=False
    return out

class Augmentor:
    def __init__(self,seed):
        self.seed=seed
        self.generators=(torch.Generator().manual_seed(5000+17*seed),torch.Generator().manual_seed(9400+17*seed))
    def __call__(self,batch,epoch,step):return apply(batch,self.seed,epoch,step,generators=self.generators)
    def state_dict(self):return [g.get_state().clone() for g in self.generators]
    def load_state_dict(self,state):
        assert len(state)==2
        for g,s in zip(self.generators,state):g.set_state(s.cpu())
