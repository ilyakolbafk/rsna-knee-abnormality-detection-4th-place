#!/usr/bin/env python3
"""VM20 fold3, changed main labels only; shared36 disease-head weakCore6 variant.

Historical feature/augmentation/data/optimizer modules are copied unchanged.
Gold is evaluation-only. All target joins and the fixed SWA window are frozen.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, fcntl, hashlib, json, math, os, random, time
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
import h1_base as h
import models
from core_v2_a_model import make
from token_slot_map import TokenSlotMap
from v83_aug_data import V83AugBags, AugConfig, loader
from orientation_policy import require_cache_orientation

K = Path(__file__).resolve().parent
A = K/'assets'
T = h.TARGETS
N = json.loads((A/'aux_names.json').read_text())
SEED, FOLD, BATCH, SCHEDULE, STOP = 20260908, 3, 8, 24, 20
ARMS = ['joint36_core6']
CORE_FRAME = None
CORE6_ALPHA = 0.10
IDS = ['SampleID','StudyInstanceUID','ParentStudyInstanceUID','report_group','fold']
def H(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def J(p): return json.loads(Path(p).read_text())
def atomic(p, x):
    p=Path(p); tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n'); os.replace(tmp,p)
def guard():
    b = J(K/'BUDGET.json')
    if datetime.now(timezone.utc) >= datetime.fromisoformat(b['stop_no_later_utc']):
        raise RuntimeError('BUDGET_DEADLINE_STOP')
    assert b['instance_id'] > 0 and b['total_usd_limit'] > 0
    assert Path('/root/.vast_containerlabel').read_text().strip() == 'C.' + str(b['instance_id'])
    if torch.cuda.device_count() != 1: raise RuntimeError('Exactly one GPU required')
    return b
def verify_package():
    p = J(K/'PACKAGE_MANIFEST.json')
    for name, digest in p['files'].items(): assert H(K/name) == digest, name
    assert p['training_contract_semantic_sha256'] == J(A/'JOINT36_TRAINING_CONTRACT.json')['semantic_sha256']
    assert len(N) == 48 and N[33] == 'effusion_size'
    root = Path('/root/rsna_joint36_core6_f3_20261012/cache')
    transport = J(root/'TRAINING_STAGE_RECEIPT.json')
    assert transport['status'] == 'PASS_TRAINING_STAGE_TRANSPORT_ONLY' and transport['npz_files'] == 4352
    inv = J(A/'TRAINING_INVENTORY.json')
    for row in inv['metadata']:
        assert H(root/row['relative_path']) == row['sha256'], row['relative_path']
    policy = J(A/'ORIENTATION_POLICY.json'); provenance = J(A/'PREPARATION_PROVENANCE.json')
    assert inv['builder_sha256'] == provenance['builder_sha256']
    assert inv['orientation_policy'] == provenance['orientation_policy']['id']
    require_cache_orientation({'orientation_policy': provenance['orientation_policy'], 'orientation_state': inv['orientation_state']}, policy)
    return p
def read_frame(p): return pd.read_csv(p,dtype={x:str for x in IDS if x!='fold'})
def inputs():
    base=read_frame(A/'BASE_FRAME.csv'); vm=read_frame(A/'VM_MAIN.csv'); e=read_frame(A/'E_MAIN.csv'); mean=read_frame(A/'MEAN_MAIN.csv')
    assert len(base)==len(vm)==len(e)==len(mean)==4352
    for f in [base,vm,e,mean]: assert f.SampleID.is_unique
    for f in [vm,e,mean]: assert f.SampleID.tolist()==base.SampleID.tolist()
    assert base.groupby('report_group').fold.nunique().max()==1
    gold=read_frame(A/'GOLD58_LABELS.csv'); gm=pd.read_csv(A/'GOLD_CACHE_MANIFEST.csv',dtype={'SampleID':str,'StudyInstanceUID':str})
    assert gm.StudyInstanceUID.is_unique and (gm.SampleID==gm.StudyInstanceUID).all()
    if 'SampleID' not in gold: gold['SampleID']=gold.StudyInstanceUID
    assert len(gold)==len(gm)==58 and gold.SampleID.is_unique and gm.SampleID.is_unique
    assert set(gold.SampleID)==set(gm.SampleID)
    gold['cache_path']=gold.SampleID.map(gm.set_index('SampleID').cache_file).map(lambda s:'/root/rsna_joint36_core6_f3_20261012/gold_stage/'+s)
    if 'ParentStudyInstanceUID' not in gold: gold['ParentStudyInstanceUID']=gold.StudyInstanceUID
    if 'report_group' not in gold: gold['report_group']=gold.StudyInstanceUID
    gold['fold']=-1
    for t in T:
        assert gold[t].isin([0,1]).all()
        gold['label_mask__'+t]=gold['confidence__'+t]=1
    for n in N:
        gold['aux__'+n]=gold['confidence__aux__'+n]=gold['mask__aux__'+n]=0
    assert not set(gold.StudyInstanceUID)&set(base.ParentStudyInstanceUID)
    assert not set(gold.report_group)&set(base.report_group)
    for p in base.cache_path.tolist()+gold.cache_path.tolist(): assert Path(p).is_file(),p
    assert (base.fold==3).sum()==873 and (base.fold!=3).sum()==3479
    global CORE_FRAME
    CORE_FRAME = read_frame(A/'CORE6_MAIN.csv')
    assert CORE_FRAME.SampleID.tolist() == base.SampleID.tolist()
    assert CORE_FRAME.groupby('report_group').fold.nunique().max() == 1
    return base,vm,e,mean,gold
def arrays(f):
    y=f[T].to_numpy(np.float32); m=f[['label_mask__'+t for t in T]].to_numpy(np.float32); c=f[['confidence__'+t for t in T]].to_numpy(np.float32)
    ok=np.isfinite(y)&(y>=0)&(y<=1)&np.isfinite(c)&(c>0)&np.isfinite(m)&(m>0)
    return np.where(ok,y,0).astype(np.float32),np.where(ok,c*m,0).astype(np.float32)
class Bags(V83AugBags):
    def __init__(self, frame, arm, evidence=None, **kwargs):
        super().__init__(frame, N, **kwargs)
        assert arm == 'joint36_core6'
        assert evidence is not None and evidence.SampleID.tolist() == self.frame.SampleID.tolist()
        y, w = arrays(self.frame); ey, ew = arrays(evidence)
        assert CORE_FRAME is not None
        cf = CORE_FRAME.set_index('SampleID').reindex(self.frame.SampleID)
        cy, cw = arrays(cf)
        self.y = np.concatenate([y, ey, cy, self.y[:,12:]], 1)
        self.label_weight = np.concatenate([w, ew, cw, self.label_weight[:,12:]], 1)
        assert self.y.shape[1] == self.label_weight.shape[1] == 84
def network(arm):
    m=make(A,chunk=96)
    if arm=='joint36_core6':
        old=m.att[-1]; rng=torch.get_rng_state(); expanded=nn.Linear(old.in_features,36)
        torch.set_rng_state(rng)  # New cloned head must not shift VM20 training RNG.
        with torch.no_grad():
            expanded.weight.copy_(old.weight.repeat(3,1)); expanded.bias.copy_(old.bias.repeat(3))
        m.att[-1]=expanded
        m.clsW=nn.Parameter(m.clsW.detach().repeat(3,1)); m.clsb=nn.Parameter(m.clsb.detach().repeat(3))
    return m
def weights(arm,device):
    d=J(A/'POS_WEIGHT_f3.json'); vals=[d[t] for t in T]+[d[n] for n in N]
    if arm=='joint36_core6': vals=vals[:12]+vals[:12]+vals[:12]+vals[12:]
    return torch.tensor(vals,dtype=torch.float32,device=device)
def losses(z, y, w, pw, arm):
    assert arm == 'joint36_core6' and z.shape[1] == y.shape[1] == w.shape[1] == 84
    def branch(start):
        return (torch.cat([v[:,start:start+12],v[:,36:]],1) for v in [z,y,w])
    wz,wy,ww = branch(0); ez,ey,ew = branch(12); cz,cy,cw = branch(24)
    wp = torch.cat([pw[:12],pw[36:]]); ep = torch.cat([pw[12:24],pw[36:]]); cp = torch.cat([pw[24:36],pw[36:]])
    wm,aux,_ = models.loss_parts(wz,wy,ww,wp,aux_weight=.25)
    em,_,_ = models.loss_parts(ez,ey,ew,ep,aux_weight=.25)
    cm,_,_ = models.loss_parts(cz,cy,cw,cp,aux_weight=.25)
    alpha = CORE6_ALPHA * (w[:,24:36].sum(1) > 0).to(z.dtype)
    main = (1-alpha)*.5*(wm+em) + alpha*cm
    return main, aux, main+.25*aux
def views(logits, arm):
    assert arm == 'joint36_core6'
    p = torch.from_numpy(logits[:,:36]).sigmoid().numpy()
    vm,e,c = p[:,:12],p[:,12:24],p[:,24:36]
    mean = .5*(vm+e)
    return {'primary': mean, 'VM_head': vm, 'Evidence_head': e, 'Core6_head': c,
            'Core6_5pct': .95*mean+.05*c}
def auc(y,p):
    if len(y)==0 or len(np.unique(y))!=2:return None
    return float(roc_auc_score(y,p))
def summarize(scores,counts):
    finite=[x for x in scores.values() if x is not None]
    return {'macro_auc':float(np.mean(finite)) if finite else None,'per_target':scores,'counts':counts,'evaluable_targets':len(finite)}
def pseudo_metrics(vm,e,probs):
    assert vm.SampleID.tolist()==e.SampleID.tolist() and len(vm)==len(probs)
    vy,vw=arrays(vm);ey,ew=arrays(e)
    # Same per-target eligible SampleIDs for both target sources and every model/readout.
    common=(vw>0)&(ew>0)&np.isfinite(vm[T].to_numpy(float))&np.isfinite(e[T].to_numpy(float))
    common&=(vm[['confidence__'+t for t in T]].to_numpy(float)>=.5)&(e[['confidence__'+t for t in T]].to_numpy(float)>=.5)&(vy!=.5)&(ey!=.5)
    out={}
    for source,y in [('VM',vy),('Evidence',ey)]:
        grain={}
        for grouped in [False,True]:
            scores={};counts={}
            for j,t in enumerate(T):
                use=common[:,j]
                if grouped:
                    rows=pd.DataFrame({'uid':vm.ParentStudyInstanceUID.astype(str),'p':probs[:,j],'y':np.where(use,(y[:,j]>.5).astype(float),np.nan)})
                    q=rows.groupby('uid',sort=False).agg(p=('p','max'),y=('y','max')).dropna(subset=['y'])
                    yy=q.y.to_numpy(int);pp=q.p.to_numpy(float)
                else: yy=(y[use,j]>.5).astype(int);pp=probs[use,j]
                scores[t]=auc(yy,pp);counts[t]={'n':int(len(yy)),'positive':int(yy.sum())}
            grain['parent' if grouped else 'sample']=summarize(scores,counts)
        out[source]=grain
    return out
def gold_metrics(gold,p):
    scores={};counts={}
    for j,t in enumerate(T):
        yy=gold[t].to_numpy(int);scores[t]=auc(yy,p[:,j]);counts[t]={'n':len(yy),'positive':int(yy.sum())}
    assert counts['Effusion']=={'n':58,'positive':35}
    return summarize(scores,counts)
def make_data(frame,arm,evidence,train,epoch,slot,workers):
    ds=Bags(frame,arm,evidence=evidence,train=train,seed=SEED,epoch=epoch,slot_map=slot if train else None,config=AugConfig())
    return loader(ds,BATCH if train else 1,workers,SEED+epoch,train)
@torch.inference_mode()
def predict(m,frame,arm,evidence,workers):
    m.eval(); result=np.full((len(frame),84),np.nan,np.float32)
    for x,mask,meta,_y,_w,index,gain,_params in make_data(frame,arm,evidence,False,0,None,workers):
        guard()
        with torch.autocast('cuda',dtype=torch.bfloat16): z=m(x.cuda(non_blocking=True),mask.cuda(non_blocking=True),meta.cuda(non_blocking=True),gain.cuda(non_blocking=True))
        result[index.numpy()]=z.float().cpu().numpy()
    assert np.isfinite(result).all()
    return result
def save_views(out,stem,frame,logits,arm):
    for name,p in views(logits,arm).items():
        d=frame[IDS].copy()
        for j,t in enumerate(T):d[t]=p[:,j]
        d.to_csv(out/(stem+'.'+name+'.csv'),index=False)
    np.savez_compressed(out/(stem+'.logits.npz'),logits=logits,SampleID=frame.SampleID.to_numpy(str))
def overlay(base,f):
    assert base.SampleID.tolist()==f.SampleID.tolist(); r=base.copy()
    for t in T:
        for col in [t,'label_mask__'+t,'confidence__'+t]:r[col]=f[col].to_numpy()
    return r
def select(base,vm,e,mean,gold,arm):
    f=overlay(base,{'evidence12':e,'mean12':mean,'joint36_core6':vm}[arm]); vf=base.fold==3; ff=~vf
    def cut(x,mask):return x.loc[mask].copy().reset_index(drop=True)
    return cut(f,ff),cut(f,vf),cut(e,ff),cut(e,vf),cut(vm,vf),gold
def evaluation(m,val,ve,vw,gold,arm,workers,out,epoch):
    result={}
    for name,frame,ef in [('fold',val,ve),('gold',gold,gold)]:
        z=predict(m,frame,arm,ef,workers);save_views(out,f'{name}_predictions_{epoch}',frame,z,arm)
        result[name]={v:(pseudo_metrics(vw,ve,p) if name=='fold' else gold_metrics(gold,p)) for v,p in views(z,arm).items()}
    return result
def smoke(data,workers):
    base,vm,e,mean,gold=data; device=torch.device('cuda'); reports=[]
    for arm in ARMS:
        h.seed_all(SEED);m=network(arm).to(device);fit,val,fe,ve,vw,g=select(*data,arm)
        pw=weights(arm,device);dl=make_data(fit,arm,fe,True,0,TokenSlotMap(A/'TOKEN_SLOT_MAP.csv'),workers)
        for step,(x,mask,meta,y,w,index,gain,params) in enumerate(dl):
            guard();m.train();m.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                z=m(x.cuda(),mask.cuda(),meta.cuda(),gain.cuda(),params.cuda());_,_,ls=losses(z,y.cuda(),w.cuda(),pw,arm);loss=ls.mean()
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),3.,error_if_nonfinite=True)
            reports.append({'arm':arm,'batch':step+1,'loss':float(loss.detach()),'logits':int(z.shape[1])})
            if step==1:break
        assert step==1
        # Explicit zero-gradient masked-cell check for actual branch layout, independent of encoder.
        z=torch.randn(2,84,device=device,requires_grad=True);y=torch.rand_like(z);w=torch.ones_like(z);w[:,0]=0;w[:,24]=0
        if arm=='joint36_core6':w[:,12]=0
        losses(z,y,w,pw,arm)[2].mean().backward();assert (z.grad[:,0]==0).all()
        if arm=='joint36_core6':assert (z.grad[:,12]==0).all() and (z.grad[:,24]==0).all()
        del m,dl,z;torch.cuda.empty_cache()
    atomic(K/'SMOKE_RESULT.json',{'status':'PASS_CUDA_FINITE_AND_MASKED_GRADIENT','code_sha256':H(__file__),'package_manifest_sha256':H(K/'PACKAGE_MANIFEST.json'),'batches':reports,'no_smoke_weights_reused':True})
def required_outputs(arm):
    names=['primary','VM_head','Evidence_head','Core6_head','Core6_5pct']
    fs={'IDENTITY.json','history.json','fit_ids.csv','validation_ids.csv','averaged.pt'}
    fs.update(f'weights-e{e:02d}.pt' for e in range(1,21))
    for ep in [f'e{e:02d}' for e in range(1,21)]+['swa18_20']:
        for role in ['fold','gold']:
            stem=f'{role}_predictions_{ep}';fs.add(stem+'.logits.npz')
            fs.update(stem+'.'+name+'.csv' for name in names)
    return fs
def completed(out,arm):
    q=J(out/'RESULT.json');identity=J(out/'IDENTITY.json')
    assert q['status']=='COMPLETE' and q['arm']==arm and q['epochs']==20
    assert identity['arm']==arm and identity['code_sha256']==H(__file__) and identity['package_manifest_sha256']==H(K/'PACKAGE_MANIFEST.json')
    assert [x['epoch'] for x in J(out/'history.json')]==list(range(1,21))
    assert set(q['outputs'])==required_outputs(arm)
    for n,digest in q['outputs'].items():assert H(out/n)==digest,n
    assert H(out/'averaged.pt')==q['averaged_sha256']
    return True
def train(data,arm,workers):
    guard();sm=J(K/'SMOKE_RESULT.json');assert sm['status']=='PASS_CUDA_FINITE_AND_MASKED_GRADIENT' and sm['code_sha256']==H(__file__) and sm['package_manifest_sha256']==H(K/'PACKAGE_MANIFEST.json')
    fit,val,fe,ve,vw,gold=select(*data,arm)
    assert not set(fit.report_group)&set(val.report_group)
    h.seed_all(SEED);m=network(arm).cuda();opt,sched=models.optimizer(m,SCHEDULE,math.ceil(len(fit)/BATCH));pw=weights(arm,torch.device('cuda'))
    out=K.parent/'training_outputs'/arm;out.mkdir(parents=True,exist_ok=True)
    identity={'arm':arm,'baseline':'core_v2a_main_V_short20_f3','fold':3,'seed':SEED,'schedule':24,'stop':20,'batch':8,'main_heads':36,'core6_loss_alpha':CORE6_ALPHA,'Core6_bilateral_masked':True,'aux_heads':48,'aux_weight':.25,'fixed_swa':[18,19,20],'fit':3479,'validation':873,'package_manifest_sha256':H(K/'PACKAGE_MANIFEST.json'),'code_sha256':H(__file__),'gold_fit':False,'gold_selection':False,'primary_joint':'VM_Evidence_probability_mean_all12','secondary_joint':'VM_Evidence95pct_plus_learnedCore6_5pct','orientation_policy':'knee-laterality-v8.3','orientation_state':'canonical_L_v8_3','training_contract_sha256':H(A/'JOINT36_TRAINING_CONTRACT.json'),'budget':guard()}
    history=[];start=updates=0;latest=out/'latest.pt'
    if latest.exists():
        q=h.load_checkpoint(latest);assert q['identity']==identity
        m.load_state_dict(q['model']);opt.load_state_dict(q['optimizer']);sched.load_state_dict(q['scheduler']);history=q['history'];start=q['epoch'];updates=q['updates']
        torch.set_rng_state(q['rng']);torch.cuda.set_rng_state_all(q['cuda_rng']);random.setstate(q['python_rng']);np.random.set_state(q['numpy_rng'])
    atomic(out/'IDENTITY.json',identity);fit[IDS].to_csv(out/'fit_ids.csv',index=False);val[IDS].to_csv(out/'validation_ids.csv',index=False)
    slot=TokenSlotMap(A/'TOKEN_SLOT_MAP.csv')
    for ei in range(start,STOP):
        ep=ei+1;m.train();opt.zero_grad(set_to_none=True);begin=time.monotonic();total=n=0;ml=al=0
        dl=make_data(fit,arm,fe,True,ei,slot,workers)
        for step,(x,mask,meta,y,w,index,gain,params) in enumerate(dl):
            guard();x,mask,meta,y,w,gain,params=[t.cuda(non_blocking=True) for t in (x,mask,meta,y,w,gain,params)]
            with torch.autocast('cuda',dtype=torch.bfloat16):z=m(x,mask,meta,gain,params);main,aux,ls=losses(z,y,w,pw,arm);loss=ls.sum()
            assert torch.isfinite(loss);loss.backward()
            for parameter in m.parameters():
                if parameter.grad is not None:parameter.grad.div_(len(x))
            torch.nn.utils.clip_grad_norm_(m.parameters(),3.,error_if_nonfinite=True);opt.step();sched.step();opt.zero_grad(set_to_none=True);updates+=1
            n+=len(x);total+=float(loss.detach());ml+=float(main.sum().detach());al+=float(aux.sum().detach())
            if step%25==0:atomic(out/'LIVE.json',{'phase':'train','arm':arm,'epoch':ep,'batch':step+1,'batches':len(dl),'loss':total/n,'utc':datetime.now(timezone.utc).isoformat()})
        torch.cuda.synchronize();record={'epoch':ep,'train_seconds':time.monotonic()-begin,'train_loss':total/n,'main_loss':ml/n,'aux_loss':al/n,'updates':updates}
        atomic(out/'LIVE.json',{'phase':'evaluation','arm':arm,'epoch':ep,'utc':datetime.now(timezone.utc).isoformat()})
        record['evaluation']=evaluation(m,val,ve,vw,gold,arm,workers,out,f'e{ep:02d}');history.append(record)
        cpu={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        h.atomic_torch(out/f'weights-e{ep:02d}.pt',{'model':cpu,'identity':identity,'epoch':ep})
        h.atomic_torch(latest,{'model':cpu,'identity':identity,'epoch':ep,'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'history':history,'updates':updates,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state()});atomic(out/'history.json',history)
        print(json.dumps({'arm':arm,'epoch':ep,'loss':total/n,'seconds':record['train_seconds']}),flush=True)
    states=[h.load_checkpoint(out/f'weights-e{ep:02d}.pt')['model'] for ep in [18,19,20]]
    averaged={}
    for k,v in states[0].items():
        assert all(s[k].shape==v.shape and s[k].dtype==v.dtype for s in states)
        averaged[k]=(sum(s[k].double() for s in states)/3).to(v.dtype) if v.is_floating_point() else v.clone()
    m.load_state_dict(averaged);h.atomic_torch(out/'averaged.pt',{'model':averaged,'identity':identity,'epochs':[18,19,20]})
    result={'status':'COMPLETE','arm':arm,'epochs':20,'fixed_swa_evaluation':evaluation(m,val,ve,vw,gold,arm,workers,out,'swa18_20'),'averaged_sha256':H(out/'averaged.pt')};result['outputs']={n:H(out/n) for n in sorted(required_outputs(arm))};atomic(out/'RESULT.json',result);assert completed(out,arm);atomic(out/'LIVE.json',{'phase':'complete','arm':arm,'utc':datetime.now(timezone.utc).isoformat()})
    del m,opt,sched;torch.cuda.empty_cache()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['smoke','train','all']);ap.add_argument('--arm',choices=ARMS);ap.add_argument('--workers',type=int,default=24);args=ap.parse_args()
    lock=open(K.parent/'training_pipeline.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    guard();verify_package();torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    assert torch.cuda.is_bf16_supported();data=inputs()
    if args.mode in ['smoke','all']:smoke(data,args.workers)
    if args.mode=='train':assert args.arm;train(data,args.arm,args.workers)
    if args.mode=='all':
        for arm in ARMS:
            result=K.parent/'training_outputs'/arm/'RESULT.json'
            if result.exists():assert completed(result.parent,arm);continue
            train(data,arm,args.workers)
if __name__=='__main__':main()
