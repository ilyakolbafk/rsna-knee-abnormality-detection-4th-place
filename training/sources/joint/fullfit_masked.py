"""One fresh K24/336 Joint36 fullfit; source masks and equal ranked readout."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, fcntl, hashlib, json, math, os, random, time
import numpy as np
import pandas as pd
import torch
import historical_joint36 as t
import v83_aug_data as data

K=Path(__file__).resolve().parent;A=K/'assets';ROOT=K.parent
T=t.T;N=t.N;IDS=t.IDS;ARM='joint36_core6'
ROLE='descriptive_in_sample; Gold58 included in fit; no heldout quality claim'
def H(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def J(p):return json.loads(Path(p).read_text())
def atomic(p,x):t.atomic(p,x)
def contract():return J(A/'TRAINING_CONTRACT.json')
def guard():
    b=J(K/'BUDGET.json')
    assert Path('/root/.vast_containerlabel').read_text().strip()=='C.'+str(b['instance_id'])
    if time.time()>=b['stop_no_later_unix']:raise RuntimeError('BUDGET_DEADLINE')
    assert torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 5090'
    return b
def verify():
    m=J(K/'PACKAGE_MANIFEST.json')
    for n,s in m['files'].items():assert H(K/n)==s,n
    assert m['semantic_sha256']==contract()['semantic_sha256']
    assert data.WINDOWS==24
    inv=J(A/'TRAINING_INVENTORY.json');policy=J(A/'ORIENTATION_POLICY.json');prov=J(A/'PREPARATION_PROVENANCE.json')
    t.require_cache_orientation({'orientation_policy':prov['orientation_policy'],'orientation_state':inv['orientation_state']},policy)
    return m
def frames(paths=False):
    f=t.read_frame(A/'FULLFIT_BASE.csv');v=t.read_frame(A/'FULLFIT_MAIN.csv');e=t.read_frame(A/'FULLFIT_EVIDENCE.csv');c=t.read_frame(A/'FULLFIT_CORE6.csv')
    assert len(f)==4413 and f.SampleID.is_unique and f.ParentStudyInstanceUID.nunique()==4406
    for q in [v,e,c]:assert q.SampleID.tolist()==f.SampleID.tolist()
    if paths:
        for path in f.cache_path:assert Path(path).is_file(),path
    t.CORE_FRAME=c
    return f,v,e,c
def overlay(base,source):return t.overlay(base,source)
def bags(f,v,e,train=False,epoch=0):
    return t.Bags(overlay(f,v),ARM,evidence=e,train=train,seed=t.SEED,epoch=epoch,
        slot_map=t.TokenSlotMap(A/'FULLFIT_TOKEN_SLOT_MAP.csv'),config=data.AugConfig())
def losses(z,y,w,pw):
    # Keep historical 45/45/10 branch weights; renormalize only row-missing branches.
    ms=[];active=[];aux=None
    for i in range(3):
        start=12*i
        zz=torch.cat([z[:,start:start+12],z[:,36:]],1);yy=torch.cat([y[:,start:start+12],y[:,36:]],1)
        ww=torch.cat([w[:,start:start+12],w[:,36:]],1);pp=torch.cat([pw[start:start+12],pw[36:]])
        mm,aa,_=t.models.loss_parts(zz,yy,ww,pp,aux_weight=.25)
        ms.append(mm);active.append((ww[:,:12].sum(1)>0).to(z.dtype))
        if aux is None:aux=aa
    aa=torch.stack(active,1);coeff=z.new_tensor([.45,.45,.10])[None,:]*aa
    coeff=coeff/coeff.sum(1,keepdim=True).clamp_min(1e-12)
    main=(torch.stack(ms,1)*coeff).sum(1)
    return main,aux,main+.25*aux
def ranked(frame,logits):
    assert np.isfinite(logits).all() and logits.shape[0]==len(frame)
    # Aggregate knees within each learned head BEFORE cohort percentile ranks.
    uid=frame.ParentStudyInstanceUID.astype(str).to_numpy();unique=list(dict.fromkeys(uid))
    allowed=np.asarray(J(A/'SOURCE_MASKS.json')['matrix'],bool)
    output=np.zeros((len(unique),12),np.float64)
    for source in range(3):
        z=pd.DataFrame(logits[:,source*12:(source+1)*12],columns=T);z.insert(0,'StudyInstanceUID',uid)
        mx=z.groupby('StudyInstanceUID',sort=False)[T].max().reindex(unique)
        # Monotonic logits preserve exact ranking and avoid sigmoid saturation ties.
        rr=mx.rank(axis=0,method='average',pct=True).to_numpy(float)
        output+=rr*allowed[source][None,:]/allowed.sum(0)[None,:]
    df=pd.DataFrame(output,columns=T);df.insert(0,'StudyInstanceUID',unique)
    return df
def predict(m,f,v,e,workers):
    m.eval();ds=bags(f,v,e);dl=data.loader(ds,1,workers,t.SEED,False)
    result=np.full((len(f),84),np.nan,np.float32)
    with torch.inference_mode():
        for x,mask,meta,y,w,index,gain,params in dl:
            guard()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                z=m(x.cuda(),mask.cuda(),meta.cuda(),gain.cuda(),params.cuda())
            result[index.numpy()]=z.float().cpu().numpy()
    assert np.isfinite(result).all();return result
def label_qa():
    verify();f,v,e,c=frames();ds=bags(f,v,e)
    w=ds.label_weight;allowed=np.asarray(J(A/'SOURCE_MASKS.json')['matrix'],bool).reshape(36)
    assert w.shape==ds.y.shape==(4413,84) and np.isfinite(w).all() and np.isfinite(ds.y).all()
    assert (w[:,:36][:,~allowed]==0).all()
    bilateral=f.ParentStudyInstanceUID.duplicated(keep=False).to_numpy()
    assert bilateral.sum()==14 and (w[bilateral,24:36]==0).all()
    test3=J(A/'TEST3_BUILD_RECEIPT.json')['records'];si=f.SampleID.isin([x['SampleID'] for x in test3]).to_numpy()
    assert si.sum()==3 and (f.loc[si,'fold']==-1).all()
    assert (w[si,:12]==1).all() and (w[si,12:]==0).all()
    assert np.array_equal(ds.y[si,:12],v.loc[si,T].to_numpy(np.float32))
    slot=t.TokenSlotMap(A/'FULLFIT_TOKEN_SLOT_MAP.csv')
    for r in test3:assert len(slot.slot_ids(r['SampleID'],64))==64
    torch.manual_seed(7);z=torch.randn(4413,84,requires_grad=True);yy=torch.from_numpy(ds.y);ww=torch.from_numpy(w);pw=t.weights(ARM,torch.device('cpu'))
    main,aux,ls=losses(z,yy,ww,pw);ls.mean().backward()
    assert torch.isfinite(ls).all() and (z.grad[:,:36][:,~allowed]==0).all()
    assert (z.grad[si,:12]!=0).all() and (z.grad[si,12:]==0).all()
    vm_loss=t.models.loss_parts(torch.cat([z[:,:12],z[:,36:]],1),torch.cat([yy[:,:12],yy[:,36:]],1),torch.cat([ww[:,:12],ww[:,36:]],1),torch.cat([pw[:12],pw[36:]]),aux_weight=.25)[0]
    assert torch.equal(main[si],vm_loss[si]) and (aux[si]==0).all()
    # All masks and rank routing verified without a GPU optimizer action.
    fake=np.random.default_rng(22).normal(size=(len(f),84)).astype(np.float32);r1=ranked(f,fake)
    changed=fake.copy();changed[:,:36][:,~allowed]=1e20;r2=ranked(f,changed)
    pd.testing.assert_frame_equal(r1,r2);assert r1.StudyInstanceUID.is_unique and len(r1)==4406
    result={'status':'PASS_CPU_MASK_GRADIENT_RANK_ROUTING_TEST3','fit':len(f),'parent_count':4406,'active_main_heads':int(allowed.sum()),'disabled_main_heads':int((~allowed).sum()),
       'test3_vm_cells':36,'test3_VM_loss_multiplier':1,'test3_E_Core6_aux_zero':True,'bilateral_Core6_masked':14,
       'class_weights_sha256':H(A/'POS_WEIGHT_f3.json'),'package_sha256':H(K/'PACKAGE_MANIFEST.json'),'code_sha256':H(__file__)}
    atomic(ROOT/'CPU_QA.json',result);print(json.dumps(result),flush=True);return result
def cache_qa():
    guard();verify();f,v,e,c=frames(True)
    inv=J(A/'TRAINING_INVENTORY.json');root=ROOT/'cache'
    for row in inv['metadata']:assert H(root/row['relative_path'])==row['sha256'],row['relative_path']
    rows=[(root/x['relative_path'],x) for x in inv['files']]
    rows += [(ROOT/'gold_stage'/x['cache_file'],x) for x in J(A/'EXPECTED_GOLD_OBJECTS.json')]
    rows += [(ROOT/'test3'/x['cache_file'],x) for x in J(A/'TEST3_BUILD_RECEIPT.json')['records']]
    assert len(rows)==4413
    slot=t.TokenSlotMap(A/'FULLFIT_TOKEN_SLOT_MAP.csv');start=time.monotonic()
    for i,(p,row) in enumerate(rows):
        guard();assert p.stat().st_size==row['bytes'] and H(p)==row['sha256'],str(p)
        with np.load(p,allow_pickle=False) as z:
            assert str(z['format'])=='native64-token-bank-v1' and str(z['sample_id'])==row['SampleID']
            b,r,m,meta=z['bank'],z['refs'],z['mask'],z['meta']
            assert b.dtype==np.uint8 and b.shape[1:]==(336,336) and r.shape==(64,3) and m.shape==(64,) and meta.shape==(64,6)
            assert np.isfinite(meta).all() and r.min()>=0 and r.max()<len(b)
            assert len(slot.slot_ids(row['SampleID'],int(m.sum())))==int(m.sum())
        if (i+1)%250==0:print(json.dumps({'phase':'cache_QA','done':i+1,'total':4413}),flush=True)
    # Real test3 cache must traverse the family loader and sampler before GPU allocation.
    ff=f.iloc[-3:].reset_index(drop=True);vv=v.iloc[-3:].reset_index(drop=True);ee=e.iloc[-3:].reset_index(drop=True)
    ds=bags(ff,vv,ee,True);seen=[]
    for x,mask,meta,y,w,index,gain,params in data.loader(ds,3,0,t.SEED,False):
        assert x.shape[0]==3 and 0<x.shape[1]<=24 and x.shape[2:]==(3,336,336)
        assert torch.isfinite(x).all() and torch.isfinite(meta).all() and (w[:,:12]==1).all() and (w[:,12:]==0).all()
        zz=torch.zeros((3,84),requires_grad=True);_,aa,ll=losses(zz,y,w,t.weights(ARM,torch.device('cpu')));ll.mean().backward()
        assert torch.isfinite(ll).all() and (aa==0).all() and (zz.grad[:,:12]!=0).all() and (zz.grad[:,12:]==0).all()
        seen.extend(ff.iloc[index.numpy()].SampleID.astype(str))
    assert set(seen)==set(ff.SampleID) and len(seen)==3
    atomic(ROOT/'CACHE_QA.json',{'status':'PASS_4413_CACHE_SHA_SCHEMA_SLOT_V83','seconds':time.monotonic()-start,'test3_actual_CPU_loader_sampler_loss':'PASS_36_VM_gradients_aux_E_Core6_zero','package_sha256':H(K/'PACKAGE_MANIFEST.json')})
def smoke(workers):
    guard();label_qa();qa=J(ROOT/'CACHE_QA.json');assert qa['status']=='PASS_4413_CACHE_SHA_SCHEMA_SLOT_V83' and qa['package_sha256']==H(K/'PACKAGE_MANIFEST.json')
    f,v,e,c=frames(True);indices=list(range(5))+[4410,4411,4412]
    ff=f.iloc[indices].reset_index(drop=True);vv=v.iloc[indices].reset_index(drop=True);ee=e.iloc[indices].reset_index(drop=True)
    # Disable token dropout for a second worst-case K24 batch while keeping training recipe untouched.
    t.h.seed_all(t.SEED);m=t.network(ARM).cuda();pw=t.weights(ARM,torch.device('cuda'))
    opt,sched=t.models.optimizer(m,24,math.ceil(4413/8));timings=[]
    for worst in [False,True]:
        ds=bags(ff,vv,ee,True)
        if worst:ds.config=data.AugConfig(series_drop=0,token_drop=0,jitter=1,params=True)
        dl=data.loader(ds,8,workers,t.SEED,False);x,mask,meta,y,w,index,gain,params=next(iter(dl))
        assert x.shape[1]<=24 and (not worst or x.shape[1]==24)
        torch.cuda.reset_peak_memory_stats();start=time.monotonic();m.train();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):z=m(x.cuda(),mask.cuda(),meta.cuda(),gain.cuda(),params.cuda());main,aux,ls=losses(z,y.cuda(),w.cuda(),pw);loss=ls.mean()
        assert torch.isfinite(loss) and torch.isfinite(z).all();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),3.,error_if_nonfinite=True)
        opt.step();sched.step();torch.cuda.synchronize()
        timings.append({'worst_case':worst,'tokens':int(x.shape[1]),'loss':float(loss.detach()),'peak_memory_bytes':torch.cuda.max_memory_allocated(),'seconds':time.monotonic()-start})
    atomic(ROOT/'CUDA_QA.json',{'status':'PASS_CUDA_REAL_OPTIMIZER_AND_K24_WORSTCASE','steps':timings,'smoke_weights_discarded':True,'package_sha256':H(K/'PACKAGE_MANIFEST.json'),'code_sha256':H(__file__)})
def evaluate(m,f,v,e,out,tag,workers):
    # Frozen descriptive cohort excludes test3 from quality, includes every original parent before rank.
    keep=np.arange(4410);ff=f.iloc[keep].reset_index(drop=True);vv=v.iloc[keep].reset_index(drop=True);ee=e.iloc[keep].reset_index(drop=True)
    z=predict(m,ff,vv,ee,workers);rr=ranked(ff,z);rr.to_csv(out/(tag+'.ranked.csv'),index=False)
    np.savez_compressed(out/(tag+'.logits.npz'),logits=z,SampleID=ff.SampleID.to_numpy(str))
    gold=t.read_frame(A/'GOLD58_LABELS.csv');gg=rr.set_index('StudyInstanceUID').loc[gold.StudyInstanceUID]
    values={target:t.auc(gold[target].to_numpy(int),gg[target].to_numpy(float)) for target in T}
    return {'ranked_Gold58_in_sample':{'macro_auc':float(np.mean([x for x in values.values() if x is not None])),'per_target':values},'role':ROLE,'rank_pool_parents':len(rr),'test3_excluded_from_quality':True}
def train(workers):
    guard();verify();qa=J(ROOT/'CUDA_QA.json');cpu=J(ROOT/'CPU_QA.json')
    assert qa['status']=='PASS_CUDA_REAL_OPTIMIZER_AND_K24_WORSTCASE' and qa['package_sha256']==H(K/'PACKAGE_MANIFEST.json') and qa['code_sha256']==H(__file__)
    assert cpu['status']=='PASS_CPU_MASK_GRADIENT_RANK_ROUTING_TEST3' and cpu['package_sha256']==H(K/'PACKAGE_MANIFEST.json') and cpu['code_sha256']==H(__file__)
    out=ROOT/'training_outputs';assert not out.exists(),'No implicit restart/resume';out.mkdir()
    f,v,e,c=frames(True);t.h.seed_all(t.SEED);m=t.network(ARM).cuda();opt,sched=t.models.optimizer(m,24,math.ceil(len(f)/8));pw=t.weights(ARM,torch.device('cuda'))
    identity={**contract(),'code_sha256':H(__file__),'package_sha256':H(K/'PACKAGE_MANIFEST.json'),'instance_id':guard()['instance_id'],'quality_role':ROLE}
    atomic(out/'IDENTITY.json',identity);f[IDS].to_csv(out/'fit_ids.csv',index=False);history=[];updates=0
    for ei in range(20):
        ep=ei+1;begin=time.monotonic();m.train();total=ml=al=0.;n=0
        dl=data.loader(bags(f,v,e,True,ei),8,workers,t.SEED+ei,True)
        for step,(x,mask,meta,y,w,index,gain,params) in enumerate(dl):
            guard();opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):z=m(x.cuda(),mask.cuda(),meta.cuda(),gain.cuda(),params.cuda());main,aux,ls=losses(z,y.cuda(),w.cuda(),pw);loss=ls.sum()
            assert torch.isfinite(loss);loss.backward()
            for parameter in m.parameters():
                if parameter.grad is not None:parameter.grad.div_(len(x))
            torch.nn.utils.clip_grad_norm_(m.parameters(),3.,error_if_nonfinite=True);opt.step();sched.step();updates+=1
            n+=len(x);total+=float(loss.detach());ml+=float(main.sum().detach());al+=float(aux.sum().detach())
            if step%25==0:atomic(out/'LIVE.json',{'phase':'train','epoch':ep,'step':step+1,'steps':len(dl),'updates':updates,'samples':n,'loss':total/n,'utc':datetime.now(timezone.utc).isoformat()})
        torch.cuda.synchronize();record={'epoch':ep,'updates':updates,'train_loss':total/n,'main_loss':ml/n,'aux_loss':al/n,'seconds':time.monotonic()-begin};history.append(record)
        state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        t.h.atomic_torch(out/f'weights-e{ep:02d}.pt',{'model':state,'identity':identity,'epoch':ep})
        t.h.atomic_torch(out/'latest.pt',{'model':state,'identity':identity,'epoch':ep,'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'history':history,'updates':updates,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state()})
        atomic(out/'history.json',history);print(json.dumps(record),flush=True)
    states=[t.h.load_checkpoint(out/f'weights-e{ep:02d}.pt')['model'] for ep in [18,19,20]]
    averaged={k:(sum(s[k].double() for s in states)/3).to(v.dtype) if v.is_floating_point() else v.clone() for k,v in states[0].items()}
    m.load_state_dict(averaged);t.h.atomic_torch(out/'averaged.pt',{'model':averaged,'identity':identity,'epochs':[18,19,20]})
    atomic(out/'LIVE.json',{'phase':'ranked_in_sample_readout','utc':datetime.now(timezone.utc).isoformat()})
    metrics=evaluate(m,f,v,e,out,'swa18_20',workers)
    result={'status':'COMPLETE','epochs':20,'updates':updates,'selected_epochs':[18,19,20],'averaged_sha256':H(out/'averaged.pt'),'evaluation':metrics,'role':ROLE,'code_sha256':H(__file__),'package_sha256':H(K/'PACKAGE_MANIFEST.json')}
    atomic(out/'RESULT.json',result);atomic(out/'LIVE.json',{'phase':'complete','utc':datetime.now(timezone.utc).isoformat()})
def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['cpu','cache','smoke','train']);ap.add_argument('--workers',type=int,default=8);args=ap.parse_args()
    lock=(ROOT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    if args.mode=='cpu':label_qa()
    elif args.mode=='cache':cache_qa()
    elif args.mode=='smoke':smoke(args.workers)
    else:train(args.workers)
if __name__=='__main__':main()
