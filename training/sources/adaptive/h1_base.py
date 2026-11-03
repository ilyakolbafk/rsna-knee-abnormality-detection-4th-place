"""Isolated H1: fixed bags, random 12 windows, raw BF16 weights, 24-epoch OneCycle.

No imported training entry point can start an old experiment. Gold/report-group
exclusion and cache identity are enforced again at execution, not just in YAML.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, time
from contextlib import contextmanager, nullcontext
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint

TARGETS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
           'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()

def semantic(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)

def atomic_torch(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.partial')
    torch.save(obj,tmp);os.replace(tmp,path)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def eligible_manifest(frame):
    if frame.StudyInstanceUID.duplicated().any() or frame.report_group.isna().any():
        raise ValueError('Unique studies and non-null report groups required')
    gold=frame.is_gold.astype(str).str.lower().isin(['true','1'])
    excluded=set(frame.loc[gold,'report_group'])
    result=frame.loc[~frame.report_group.isin(excluded)].copy()
    if result.report_group.groupby(result.report_group).size().empty:
        raise ValueError('Empty eligible population')
    if result.groupby('report_group').fold.nunique().max()!=1:
        raise ValueError('Report group crosses folds')
    return result,sorted(excluded)

def random_windows(valid,count,rng):
    valid=list(map(int,valid))
    if not valid or count<1: raise ValueError('Empty window pool')
    # Match author repetition+shuffle, including duplicate draws for short bags.
    pool=valid*(count//len(valid)+1);rng.shuffle(pool)
    return np.asarray(pool[:count],np.int64)

def label_arrays(frame):
    y=frame[TARGETS].to_numpy(np.float32)
    masks=frame[['label_mask__'+t for t in TARGETS]].to_numpy(bool,copy=True)
    masks &= np.isfinite(y) & (y>=0) & (y<=1)
    return np.nan_to_num(y),masks

def positive_weights(frame):
    y,m=label_arrays(frame);counts=m.sum(0)
    if np.any(counts==0): raise ValueError('No training labels for a target')
    prevalence=np.clip((y*m).sum(0)/counts,.03,.7)
    return torch.tensor(np.clip((1-prevalence)/prevalence,1,10),dtype=torch.float32)

def auc_report(frame,predictions):
    from sklearn.metrics import roc_auc_score
    y,m=label_arrays(frame)
    c=frame[['confidence__'+t for t in TARGETS]].to_numpy(float)
    m &= (c>=.5) & (np.abs(y-.5)>0.0)
    scores={};counts={}
    for j,t in enumerate(TARGETS):
        labels=(y[m[:,j],j]>=.5).astype(int)
        scores[t]=float(roc_auc_score(labels,predictions[m[:,j],j])) if len(np.unique(labels))==2 else None
        counts[t]={'n':len(labels),'positive':int(labels.sum())}
    if any(v is None for v in scores.values()): raise ValueError('All 12 validation targets must be evaluable')
    return {'pseudo_macro_auc':float(np.mean(list(scores.values()))),'per_target':scores,'counts':counts}

class Bags(Dataset):
    def __init__(self,frame,cache_root,arm,*,train=False,seed=20260908,epoch=0):
        self.frame=frame.reset_index(drop=True);self.root=Path(cache_root)
        self.arm=arm;self.train=train;self.seed=seed;self.epoch=epoch
        self.y,self.label_mask=label_arrays(self.frame)
    def __len__(self): return len(self.frame)
    def __getitem__(self,index):
        uid=str(self.frame.iloc[index].StudyInstanceUID)
        with np.load(self.root/(uid+'.npz'),allow_pickle=False) as z:
            if self.arm=='R':
                volume=z['r_volume'];mask=z['r_mask'];valid=np.flatnonzero(mask)
                if len(valid)<3: raise ValueError('R study has fewer than 3 usable frames: '+uid)
                centers=list(range(int(valid.min())+1,int(valid.max())))
                # 62 uniformly spread centers reproduce the published all-window eval.
                centers=np.asarray(centers)[np.linspace(0,len(centers)-1,62).round().astype(int)] if not self.train else np.asarray(centers)
                pool=np.arange(len(centers));metadata=np.zeros((len(pool),6),np.float32)
            else:
                pool=np.flatnonzero(z['n_mask']);metadata=z['n_meta']
            rng=random.Random(int(semantic([self.seed,self.epoch,uid])[:16],16))
            chosen=random_windows(pool,12,rng) if self.train else pool
            if self.arm=='R':
                refs=centers[chosen,None]+np.arange(-1,2)
                pixels=volume[refs];meta=metadata[chosen]
            else:
                refs=z['n_refs'][chosen]
                pixels=np.zeros((*refs.shape,336,336),np.uint8)
                ok=refs>=0;pixels[ok]=z['n_pixels'][refs[ok]]
                # Canonical transforms are frozen per-window, computed from full geometry.
                for i,spec in enumerate(z['n_transform'][chosen]):
                    if spec[0]: pixels[i]=pixels[i].transpose(0,2,1)
                    if spec[1]: pixels[i]=pixels[i,:,:,::-1]
                    if spec[2]: pixels[i]=pixels[i,:,::-1,:]
                    if spec[3]: pixels[i]=pixels[i,::-1]
                meta=metadata[chosen]
        x=torch.from_numpy(np.ascontiguousarray(pixels)).float().div_(255)
        # Resize on GPU in the model: identical bilinear math, much less worker traffic.
        gain=1+(random.Random(int(semantic([self.seed,self.epoch,uid,'gain'])[:16],16)).random()-.5)*.2 if self.train else 1.
        return x,torch.from_numpy(meta),torch.from_numpy(self.y[index]),torch.from_numpy(self.label_mask[index]),index,gain

def collate(rows):
    b=len(rows);k=max(len(r[0]) for r in rows)
    x=torch.zeros((b,k,3,336,336));meta=torch.zeros((b,k,6));mask=torch.zeros((b,k),dtype=torch.bool)
    for i,(pixels,m,*_) in enumerate(rows):
        x[i,:len(pixels)]=pixels;meta[i,:len(m)]=m;mask[i,:len(pixels)]=True
    return x,mask,meta,torch.stack([r[2] for r in rows]),torch.stack([r[3] for r in rows]),torch.tensor([r[4] for r in rows]),torch.tensor([r[5] for r in rows])

@contextmanager
def preserve_bn(module):
    values=[(b,b.clone()) for n,b in module.named_buffers() if n.endswith(('running_mean','running_var','num_batches_tracked'))]
    try: yield
    finally:
        with torch.no_grad():
            for b,v in values:b.copy_(v)

class H1(nn.Module):
    def __init__(self,weights=None,*,chunk=48,checkpointing=False,backbone=None):
        super().__init__()
        if backbone is None:
            import timm
            opts=dict(pretrained=weights is not None,num_classes=0,img_size=384,drop_path_rate=0.)
            if weights:opts['pretrained_cfg_overlay']={'file':str(weights)}
            backbone=timm.create_model('coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k',**opts)
        self.backbone=backbone;self.chunk=chunk;self.checkpointing=checkpointing
        d=int(backbone.num_features)
        self.norm=nn.LayerNorm(d)
        self.att=nn.Sequential(nn.Linear(d,256),nn.Tanh(),nn.Dropout(.2),nn.Linear(256,12))
        self.clsW=nn.Parameter(torch.zeros(12,d));self.clsb=nn.Parameter(torch.zeros(12))
        nn.init.trunc_normal_(self.clsW,std=.02)
        # Same parameter set/initialization in both arms. R metadata is all zeros.
        self.meta_projection=nn.Linear(6,d,bias=False);nn.init.zeros_(self.meta_projection.weight)
    def features(self,x,gain):
        x=F.interpolate(x,size=(384,384),mode='bilinear',align_corners=False)
        x=(x*gain[:,None,None,None]).clamp_(0,1)
        x=(x-.5)/.5
        if self.training and self.checkpointing and hasattr(self.backbone,'stages'):
            # Match timm's block granularity; preserve BN's one-update semantics.
            x=self.backbone.stem(x)
            for stage in self.backbone.stages:
                for block in stage.blocks:
                    x=checkpoint(block,x,use_reentrant=False,context_fn=lambda block=block:(nullcontext(),preserve_bn(block)))
            return self.backbone.forward_head(self.backbone.norm(x))
        return self.backbone(x)
    def forward(self,x,mask,meta,gain=None):
        if not mask.any(1).all():raise ValueError('Empty study')
        b,k=mask.shape;idx=mask.flatten().nonzero().flatten()
        selected=x.flatten(0,1).index_select(0,idx);values=[]
        if gain is None:gain=x.new_ones(b)
        gains=gain[:,None].expand(b,k).reshape(-1).index_select(0,idx)
        for xx,gg in zip(selected.split(self.chunk),gains.split(self.chunk)):
            if self.training and self.checkpointing and not hasattr(self.backbone,'stages'):
                f=checkpoint(self.features,xx,gg,use_reentrant=False,context_fn=lambda:(nullcontext(),preserve_bn(self.backbone)))
            else:f=self.features(xx,gg)
            values.append(f)
        encoded=torch.cat(values).float()
        padded=encoded.new_zeros(b*k,encoded.shape[-1]).index_copy(0,idx,encoded).view(b,k,-1)
        # Keep attention and pooled logits FP32 for stable long-bag evaluation.
        with torch.autocast(x.device.type,enabled=False):
            hidden=self.norm(padded+self.meta_projection(meta.float()))
            scores=self.att(hidden).masked_fill(~mask[:,:,None],-torch.inf)
            alpha=scores.softmax(1);pooled=torch.einsum('bkn,bkf->bnf',alpha,hidden)
            return (pooled*self.clsW).sum(-1)+self.clsb

def optimizer_scheduler(model,epochs,steps_per_epoch):
    bb=list(model.backbone.parameters());bb_ids={id(x) for x in bb}
    head=[p for p in model.parameters() if id(p) not in bb_ids]
    opt=torch.optim.AdamW([{'params':bb,'lr':3e-5},{'params':head,'lr':1e-3}],weight_decay=.02)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[3e-5,1e-3],total_steps=epochs*steps_per_epoch,pct_start=.15)
    return opt,sched

def average_states(states):
    if len(states)!=3 or any(set(s)!=set(states[0]) for s in states):raise ValueError('Exactly 3 matching states required')
    result={}
    for key,first in states[0].items():
        values=[s[key] for s in states]
        if any(v.shape!=first.shape or v.dtype!=first.dtype for v in values):raise ValueError('State schema drift')
        result[key]=(sum(v.double() for v in values)/3).to(first.dtype) if first.is_floating_point() else first.clone()
    return result

def load_checkpoint(path): return torch.load(path,map_location='cpu',weights_only=False)

def loader(dataset,batch,workers,seed,shuffle):
    opts=dict(batch_size=batch,shuffle=shuffle,num_workers=workers,collate_fn=collate,pin_memory=True,
              generator=torch.Generator().manual_seed(seed))
    if workers:opts.update(prefetch_factor=1,persistent_workers=False)
    return DataLoader(dataset,**opts)

@torch.no_grad()
def predict(model,data,device):
    model.eval();p=np.full((len(data.dataset),12),np.nan,np.float32)
    for x,m,meta,y,lm,idx,gain in data:
        if time.time()>=getattr(model,'deadline',float('inf')):raise RuntimeError('Budget deadline reached during validation')
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            out=model(x.to(device,non_blocking=True),m.to(device),meta.to(device),gain.to(device)).sigmoid()
        p[idx.numpy()]=out.float().cpu().numpy()
    if not np.isfinite(p).all():raise ValueError('Incomplete predictions')
    return p
