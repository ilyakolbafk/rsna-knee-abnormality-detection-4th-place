"""Original H1, trim02 H1, and independent auxiliary attention heads."""
from contextlib import nullcontext
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import h1_base as h

AUX_ARMS={'AUX60','AUXV3','TRIM02_AUX60','TRIM02_AUXV3'}
TRIM02_ARMS={'TRIM02','TRIM02_AUX60','TRIM02_AUXV3'}


class Bags(h.Bags):
    def __init__(self,*args,aux_names=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.aux_names=aux_names or []
        if self.aux_names:
            y=self.frame[['aux__'+n for n in self.aux_names]].to_numpy(np.float32)
            conf=self.frame[['confidence__aux__'+n for n in self.aux_names]].to_numpy(np.float32)
            mask=self.frame[['mask__aux__'+n for n in self.aux_names]].to_numpy(np.float32)
            valid=np.isfinite(y)&(y>=0)&(y<=1)&(conf>0)&(mask>0)
            self.y=np.concatenate([self.y,np.where(valid,y,0)],axis=1)
            self.label_mask=np.concatenate([self.label_mask.astype(np.float32),np.where(valid,conf*mask,0)],axis=1)
    def __getitem__(self,index):
        uid=str(self.frame.iloc[index].StudyInstanceUID)
        prefix='n2' if self.arm in TRIM02_ARMS else 'n'
        with np.load(self.root/(uid+'.npz'),allow_pickle=False) as z:
            pool=np.flatnonzero(z[prefix+'_mask']);metadata=z[prefix+'_meta']
            rng=random.Random(int(h.semantic([self.seed,self.epoch,uid])[:16],16))
            chosen=h.random_windows(pool,12,rng) if self.train else pool
            refs=z[prefix+'_refs'][chosen];pixels=np.zeros((*refs.shape,336,336),np.uint8)
            ok=refs>=0;pixels[ok]=z['n_pixels'][refs[ok]]
            for i,spec in enumerate(z[prefix+'_transform'][chosen]):
                if spec[0]:pixels[i]=pixels[i].transpose(0,2,1)
                if spec[1]:pixels[i]=pixels[i,:,:,::-1]
                if spec[2]:pixels[i]=pixels[i,:,::-1,:]
                if spec[3]:pixels[i]=pixels[i,::-1]
            meta=metadata[chosen]
        x=torch.from_numpy(np.ascontiguousarray(pixels)).float().div_(255)
        gain=1+(random.Random(int(h.semantic([self.seed,self.epoch,uid,'gain'])[:16],16)).random()-.5)*.2 if self.train else 1.
        return x,torch.from_numpy(meta),torch.from_numpy(self.y[index]),torch.from_numpy(self.label_mask[index]),index,gain


class H1Aux(h.H1):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        d=self.clsW.shape[1]
        self.aux_att=nn.Sequential(nn.Linear(d,256),nn.Tanh(),nn.Dropout(.2),nn.Linear(256,48))
        self.auxW=nn.Parameter(torch.zeros(48,d));self.auxb=nn.Parameter(torch.zeros(48))
        nn.init.trunc_normal_(self.auxW,std=.02)
    def forward(self,x,mask,meta,gain=None):
        if not mask.any(1).all():raise ValueError('Empty study')
        b,k=mask.shape;idx=mask.flatten().nonzero().flatten()
        selected=x.flatten(0,1).index_select(0,idx);values=[]
        if gain is None:gain=x.new_ones(b)
        gains=gain[:,None].expand(b,k).reshape(-1).index_select(0,idx)
        for xx,gg in zip(selected.split(self.chunk),gains.split(self.chunk)):
            values.append(self.features(xx,gg))
        encoded=torch.cat(values).float()
        padded=encoded.new_zeros(b*k,encoded.shape[-1]).index_copy(0,idx,encoded).view(b,k,-1)
        with torch.autocast(x.device.type,enabled=False):
            hidden=self.norm(padded+self.meta_projection(meta.float()))
            alpha=self.att(hidden).masked_fill(~mask[:,:,None],-torch.inf).softmax(1)
            pooled=torch.einsum('bkn,bkf->bnf',alpha,hidden)
            main=(pooled*self.clsW).sum(-1)+self.clsb
            alpha_aux=self.aux_att(hidden).masked_fill(~mask[:,:,None],-torch.inf).softmax(1)
            pooled_aux=torch.einsum('bkn,bkf->bnf',alpha_aux,hidden)
            return torch.cat([main,(pooled_aux*self.auxW).sum(-1)+self.auxb],dim=1)


class DinoH1(h.H1):
    def __init__(self,weights,**kwargs):
        import timm
        backbone=timm.create_model('vit_small_patch16_dinov3.lvd1689m',pretrained=True,
            pretrained_cfg_overlay={'file':str(weights)},num_classes=0,global_pool='token',img_size=384,drop_path_rate=0.)
        super().__init__(backbone=backbone,**kwargs)
    def features(self,x,gain):
        x=F.interpolate(x,size=(384,384),mode='bilinear',align_corners=False)
        x=(x*gain[:,None,None,None]).clamp_(0,1)
        mean=x.new_tensor([.485,.456,.406])[None,:,None,None]
        std=x.new_tensor([.229,.224,.225])[None,:,None,None]
        return self.backbone((x-mean)/std)


def make_model(arm,assets,chunk=96):
    if arm=='DINO':return DinoH1(assets/'dinov3.safetensors',chunk=chunk,checkpointing=True)
    if arm in AUX_ARMS:return H1Aux(assets/'coatnet_imagenet.safetensors',chunk=chunk,checkpointing=True)
    assert arm in ['TRIM02','H1']
    return h.H1(assets/'coatnet_imagenet.safetensors',chunk=chunk,checkpointing=True)


def optimizer(model,schedule_epochs,steps,dino=False):
    if not dino:return h.optimizer_scheduler(model,schedule_epochs,steps)
    groups=[];bb_ids=set()
    for name,param in model.backbone.named_parameters():
        bb_ids.add(id(param));layer=0
        if name.startswith('blocks.'):layer=int(name.split('.')[1])+1
        elif name.startswith(('norm.','fc_norm.','head.')):layer=13
        lr=1.5e-5*.8**(13-layer)
        groups.append({'params':[param],'lr':lr,'max_lr':lr})
    groups.append({'params':[p for p in model.parameters() if id(p) not in bb_ids],'lr':1e-3,'max_lr':1e-3})
    opt=torch.optim.AdamW(groups,weight_decay=.02)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[g['max_lr'] for g in groups],total_steps=schedule_epochs*steps,pct_start=.15)
    return opt,sched


def loss_parts(z,y,mask,pw,ordinal_index=33,aux_weight=.25):
    raw=F.binary_cross_entropy_with_logits(z[:,:12],y[:,:12],pos_weight=pw[:12],reduction='none')
    main=(raw*mask[:,:12]).sum(1)/mask[:,:12].sum(1).clamp_min(1)
    if z.shape[1]==12:return main,torch.zeros_like(main),main
    aux=F.binary_cross_entropy_with_logits(z[:,12:],y[:,12:],pos_weight=pw[12:],reduction='none')
    # effusion_size is an ordered severity v/3, not a binary diagnosis.
    aux[:,ordinal_index]=F.smooth_l1_loss(z[:,12+ordinal_index].sigmoid(),y[:,12+ordinal_index],reduction='none')
    aux=(aux*mask[:,12:]).sum(1)/mask[:,12:].sum(1).clamp_min(1)
    return main,aux,main+aux_weight*aux


def positive_weights(frame,names):
    main=h.positive_weights(frame)
    if not names:return main
    y=frame[['aux__'+n for n in names]].to_numpy(float)
    conf=frame[['confidence__aux__'+n for n in names]].to_numpy(float)
    mask=frame[['mask__aux__'+n for n in names]].to_numpy(float)
    valid=np.isfinite(y)&(y>=0)&(y<=1)&(conf>0)&(mask>0)
    w=np.where(valid,conf*mask,0);y=np.where(valid,y,0)
    totals=w.sum(0);active=totals>0
    # Kappa-filtered v3 keeps the full 48-head schema but gives excluded heads
    # exactly zero loss weight.  Their positive weight is therefore inert.
    prev=np.full(len(names),.5,dtype=float)
    prev[active]=np.clip((w[:,active]*y[:,active]).sum(0)/totals[active],.03,.7)
    p=np.ones(len(names),dtype=float);p[active]=np.clip((1-prev[active])/prev[active],1,10)
    p[names.index('effusion_size')]=1
    return torch.cat([main,torch.tensor(p,dtype=torch.float32)])
