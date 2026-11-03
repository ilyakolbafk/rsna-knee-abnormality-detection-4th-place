"""Matched main-label adapter for the preserved Core-v2A recipe."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import os
import time
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
os.environ['V83_PACKAGE_ROOT'] = str(ROOT)
import train_core_v2_a as core
import train_v83 as base
import h1_base as h
import models
from v83_aug_data import V83AugBags, AugConfig, loader
from token_slot_map import TokenSlotMap
from orientation_policy import require_cache_orientation

TARGETS = h.TARGETS
IDS = ['SampleID', 'StudyInstanceUID', 'ParentStudyInstanceUID', 'report_group', 'fold']
MAIN = TARGETS + ['label_mask__'+t for t in TARGETS] + ['confidence__'+t for t in TARGETS]

def sha(path):
    return base.sha(path)

def semantic(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

def verify_cache(cache):
    pins_manifest=json.loads((ROOT/'SOURCE_PACKAGE_MANIFEST.json').read_text())
    for name,digest in pins_manifest['files'].items(): assert sha(ROOT/name)==digest,name
    receipt = json.loads((cache/'GLOBAL_RELEASE_RECEIPT.json').read_text())
    provenance = json.loads((ROOT/'PREPARATION_PROVENANCE.json').read_text())
    pins = json.loads((ROOT/'FROZEN_COMPONENT_SHA256.json').read_text())
    assert {x['role']:x['cache_complete_sha256'] for x in receipt['components']} == pins
    assert receipt['builder_sha256'] == provenance['builder_sha256']
    for role, digest in pins.items():
        paths = list((cache/'downloads'/role).rglob('CACHE_COMPLETE.json'))
        assert len(paths)==1 and sha(paths[0])==digest
        assert json.loads(paths[0].read_text())['asset_manifest_sha256']==provenance['asset_manifest_sha256']
    metadata = {'orientation_policy':provenance['orientation_policy'],
                'orientation_state':receipt['orientation_state'],
                'cache_release_sha256':sha(cache/'GLOBAL_RELEASE_RECEIPT.json')}
    require_cache_orientation(metadata,json.loads((ROOT/'ORIENTATION_POLICY.json').read_text()))
    return metadata

def effective_arrays(frame):
    y = frame[TARGETS].to_numpy(np.float32)
    mask = frame[['label_mask__'+t for t in TARGETS]].to_numpy(float)
    conf = frame[['confidence__'+t for t in TARGETS]].to_numpy(np.float32)
    valid = np.isfinite(y)&(y>=0)&(y<=1)&np.isfinite(mask)&(mask>0)&np.isfinite(conf)&(conf>0)
    return np.where(valid,y,0),np.where(valid,mask*conf,0).astype(np.float32)

class ConfidenceBags(V83AugBags):
    def __init__(self,frame,aux_names,**kwargs):
        super().__init__(frame,aux_names,**kwargs)
        y,w = effective_arrays(self.frame)
        self.y[:,:12] = y
        self.label_weight[:,:12] = w

def prepare(cache):
    base.guard()
    assert not (ROOT/'assets/TARGET_FREEZE.json').exists(), 'Frozen targets already exist; never overwrite'
    orient = verify_cache(cache)
    names = json.loads((ROOT/'assets/aux_names.json').read_text())
    r,gold,receipt = base.load_data(cache,names)
    over = pd.read_csv(ROOT/'assets/V_INPUT.csv',dtype={'SampleID':str}).set_index('SampleID')
    assert over.index.is_unique and set(over.index)==set(r.SampleID.astype(str)) and len(over)==4352
    v = r.copy()
    for col in MAIN: v[col] = over.loc[r.SampleID.astype(str),col].to_numpy()
    bilateral = r.ParentStudyInstanceUID.astype(str).duplicated(keep=False).to_numpy()
    assert int(bilateral.sum())==14
    for col in MAIN:
        assert np.array_equal(r.loc[bilateral,col].to_numpy(),v.loc[bilateral,col].to_numpy(),equal_nan=True),col
    ry,rw = effective_arrays(r);vy,vw = effective_arrays(v)
    support = (rw>0)&(vw>0)
    assert np.array_equal(rw[support],vw[support]), 'Main confidence/mask differs'
    original = r[IDS+MAIN].copy()
    for j,t in enumerate(TARGETS):
        mask='label_mask__'+t
        r[mask] = np.where(support[:,j],r[mask],0)
        v[mask] = r[mask].to_numpy()
        v['confidence__'+t] = r['confidence__'+t].to_numpy()
    for teacher,frame in [('R',r),('V',v)]:
        frame[IDS+MAIN].to_csv(ROOT/'assets'/f'{teacher}_MAIN.csv',index=False)
    original.to_csv(ROOT/'assets/FROZEN_REFERENCE.csv',index=False)
    _,w = effective_arrays(r)
    weights = {}
    for fold in [3,0]:
        fit = r.loc[r.fold!=fold].copy()
        y,fw = effective_arrays(fit)
        assert (fw.sum(0)>0).all()
        prevalence = np.clip((y*fw).sum(0)/fw.sum(0),.03,.7)
        main = torch.tensor(np.clip((1-prevalence)/prevalence,1,10),dtype=torch.float32)
        full = models.positive_weights(fit,names)
        full[:12] = main
        path = ROOT/'assets'/f'POS_WEIGHT_f{fold}.json'
        base.atomic_json(path,dict(zip(TARGETS+names,full.tolist())))
        weights[str(fold)] = sha(path)
    assert np.array_equal(effective_arrays(r)[1],effective_arrays(v)[1])
    assert not set(gold.StudyInstanceUID.astype(str)) & set(r.ParentStudyInstanceUID.astype(str))
    slot = TokenSlotMap(ROOT/'assets/TOKEN_SLOT_MAP.csv')
    # Current bank mask token count is independently checked for every mapped sample.
    checked=0
    for row in r.itertuples(index=False):
        with np.load(row.cache_path,allow_pickle=False) as z:
            slot.slot_ids(str(row.SampleID),int(z['mask'].sum()))
        checked+=1
    columns = ['label_mask__'+t for t in TARGETS]+['confidence__'+t for t in TARGETS]
    support_binding = semantic(r[['SampleID']+columns].to_dict(orient='records'))
    record = {'status':'PASS_MATCHED_TARGET_FREEZE','cache_train_labels_sha256':sha(cache/'TRAIN_LABELS.csv'),
              'cache_release_sha256':sha(cache/'GLOBAL_RELEASE_RECEIPT.json'),
              'v_input_sha256':sha(ROOT/'assets/V_INPUT.csv'),'samples':len(r),'parents':r.ParentStudyInstanceUID.nunique(),
              'bilateral_rows':int(bilateral.sum()),'additional_common_masked_cells':int(((rw>0)&~support).sum()),
              'confidence025_active_cells':int(((w==.25)&support).sum()),'common_support_sha256':support_binding,
              'alpha_rule':'R fit only; confidence*mask weighted soft prevalence clipped[.03,.7], (1-p)/p clipped[1,10]; float32; same vector R/V. Auxiliary weights unchanged historical formula.',
              'reference':'Untouched original R cache before common-support masking; every evaluation route restores it',
              'orientation':orient,'token_slot_samples_checked':checked,'weights':weights,
              'files':{n:sha(ROOT/'assets'/n) for n in ['R_MAIN.csv','V_MAIN.csv','FROZEN_REFERENCE.csv']}}
    base.atomic_json(ROOT/'assets/TARGET_FREEZE.json',record)
    return record

def verify_freeze(cache):
    freeze=json.loads((ROOT/'assets/TARGET_FREEZE.json').read_text())
    assert freeze['status']=='PASS_MATCHED_TARGET_FREEZE'
    assert sha(cache/'TRAIN_LABELS.csv')==freeze['cache_train_labels_sha256']
    assert sha(cache/'GLOBAL_RELEASE_RECEIPT.json')==freeze['cache_release_sha256']
    verify_cache(cache)
    for n,digest in freeze['files'].items(): assert sha(ROOT/'assets'/n)==digest
    for fold,digest in freeze['weights'].items(): assert sha(ROOT/'assets'/f'POS_WEIGHT_f{fold}.json')==digest
    return freeze

def frame_for(cache,names,teacher):
    freeze=verify_freeze(cache)
    frame,gold,receipt=base.load_data(cache,names)
    for col in MAIN: frame['reference__'+col]=frame[col]
    over=pd.read_csv(ROOT/'assets'/f'{teacher}_MAIN.csv',dtype={'SampleID':str}).set_index('SampleID')
    assert over.index.is_unique and set(over.index)==set(frame.SampleID.astype(str))
    for col in MAIN: frame[col]=over.loc[frame.SampleID.astype(str),col].to_numpy()
    assert not set(gold.report_group.astype(str)) & set(frame.report_group.astype(str))
    return frame,gold,receipt

def original_reference_eval(frame,prob):
    original=frame.copy()
    assert all('reference__'+col in original for col in MAIN), 'Original reference missing'
    for col in MAIN:
        original[col]=original['reference__'+col]
    return base.evaluate_fold(original,prob)

def fixed_weights(fold,names):
    weights=json.loads((ROOT/'assets'/f'POS_WEIGHT_f{fold}.json').read_text())
    return torch.tensor([weights[t] for t in TARGETS+names],dtype=torch.float32)

def smoke(cache,steps):
    base.guard(); freeze=verify_freeze(cache)
    names=json.loads((ROOT/'assets/aux_names.json').read_text())
    r,gold,_=frame_for(cache,names,'R');v,_,_=frame_for(cache,names,'V')
    ds=[ConfidenceBags(f,names,train=True,seed=base.SEED,epoch=0,slot_map=TokenSlotMap(ROOT/'assets/TOKEN_SLOT_MAP.csv')) for f in [r,v]]
    assert np.array_equal(ds[0].label_weight,ds[1].label_weight)
    assert np.array_equal(ds[0].y[:,12:],ds[1].y[:,12:])
    assert all(np.isfinite(d.y).all() and np.isfinite(d.label_weight).all() for d in ds)
    weak=ds[0].label_weight[:,:12]==.25
    assert int(weak.sum())==freeze['confidence025_active_cells']
    for i in [0,1,100]+np.flatnonzero(r.ParentStudyInstanceUID.astype(str).duplicated(keep=False))[:2].tolist():
        left,right=ds[0][i],ds[1][i]
        for j in [0,1,5,6]:
            assert torch.equal(left[j],right[j]) if torch.is_tensor(left[j]) else left[j]==right[j]
    fixture=r.iloc[:3].copy();fixture[TARGETS[0]]=float('nan')
    fixture['label_mask__'+TARGETS[0]]=1;fixture['confidence__'+TARGETS[0]]=1
    probe=ConfidenceBags(fixture,names,train=False)
    assert np.all(probe.y[:,0]==0) and np.all(probe.label_weight[:,0]==0)
    logits=torch.randn(3,60,requires_grad=True)
    y=torch.from_numpy(probe.y);w=torch.from_numpy(probe.label_weight)
    _,_,loss=models.loss_parts(logits,y,w,fixed_weights(3,names));loss.mean().backward()
    assert torch.equal(logits.grad[:,0],torch.zeros(3)) and torch.isfinite(logits.grad).all()
    h.seed_all(base.SEED);torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    model=core.make(ROOT/'assets').cuda();model.train()
    fit=r.loc[r.fold!=3].reset_index(drop=True)
    data=loader(ConfidenceBags(fit,names,train=True,seed=base.SEED,epoch=0,slot_map=TokenSlotMap(ROOT/'assets/TOKEN_SLOT_MAP.csv')),8,24,base.SEED,True)
    opt,scheduler=models.optimizer(model,core.SCHEDULE,len(data),False);pw=fixed_weights(3,names).cuda()
    times=[];torch.cuda.reset_peak_memory_stats()
    for i,batch in enumerate(data):
        if i>=steps:break
        base.guard();begin=time.monotonic()
        x,mask,meta,y,w,_idx,gain,params=batch
        x,mask,meta,y,w,gain,params=[t.cuda(non_blocking=True) for t in [x,mask,meta,y,w,gain,params]]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=model(x,mask,meta,gain,params);_,_,per=models.loss_parts(out,y,w,pw,aux_weight=.25);loss=per.sum()
        assert torch.isfinite(loss)
        loss.backward()
        for param in model.parameters():
            if param.grad is not None:param.grad.div_(len(x))
        torch.nn.utils.clip_grad_norm_(model.parameters(),3,error_if_nonfinite=True)
        opt.step();scheduler.step();opt.zero_grad(set_to_none=True);torch.cuda.synchronize()
        times.append(time.monotonic()-begin)
        if i%10==0:print(json.dumps({'smoke_batch':i+1,'loss':float(loss.detach())/len(x),'seconds':times[-1]}),flush=True)
    assert len(times)==steps
    report={'status':'PASS_CUDA_MATCHED_CORE_SMOKE','steps':len(times),'batch':8,'train_windows_before_dropout':12,
            'steady_step_median_seconds':float(np.median(times[5:])),'steady_step_p90_seconds':float(np.quantile(times[5:],.9)),
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'full_steps_per_epoch':len(data),
            'common_bag_sample_check':5,'masked_main_gradient_zero':True,'main_confidence025_used':True,
            'no_smoke_weights_used_for_training':True,'torch':torch.__version__,'cuda':torch.version.cuda,
            'orientation':freeze['orientation'],'code_sha256':sha(__file__),
            'source_package_manifest_sha256':sha(ROOT/'SOURCE_PACKAGE_MANIFEST.json'),
            'recipe_schedule':core.SCHEDULE,'recipe_stop':core.STOP,'selected_epochs':core.SWA_WINDOW}
    base.atomic_json(ROOT/'SMOKE_RESULT.json',report)
    return report

def train(cache,teacher,fold,out,run):
    freeze=verify_freeze(cache);names=json.loads((ROOT/'assets/aux_names.json').read_text())
    smoke_result=json.loads((ROOT/'SMOKE_RESULT.json').read_text())
    assert smoke_result['status']=='PASS_CUDA_MATCHED_CORE_SMOKE' and smoke_result['code_sha256']==sha(__file__)
    core.ARM='CORE_V2_A_MAIN_'+teacher
    core.V83AugBags=ConfidenceBags
    core.load_data=lambda cache,names:frame_for(cache,names,teacher)
    core.evaluate_fold=original_reference_eval
    core.positive_weights=lambda fit,names:fixed_weights(fold,names)
    original_hashes=core.code_hashes
    core.code_hashes=lambda:{**original_hashes(),'matched_core.py':sha(__file__),'orientation_policy.py':sha(ROOT/'orientation_policy.py')}
    core.IDENTITY_EXTRA={'recipe_variant':'short20','teacher':teacher,'gold_in_training':False,'cache_train_labels_sha256':freeze['cache_train_labels_sha256'],
                         'target_freeze_sha256':sha(ROOT/'assets/TARGET_FREEZE.json'),
                         'main_targets_sha256':freeze['files'][teacher+'_MAIN.csv'],'common_support_sha256':freeze['common_support_sha256'],
                         'fixed_pos_weight_sha256':freeze['weights'][str(fold)],'original_reference_sha256':freeze['files']['FROZEN_REFERENCE.csv'],
                         'orientation_identity':freeze['orientation']['orientation_policy'],'main_confidence_used':True}
    args=SimpleNamespace(cache_root=str(cache),slot_map=str(ROOT/'assets/TOKEN_SLOT_MAP.csv'),fold=fold,full=False,
                         series_drop=.10,token_drop=.15,jitter=1,no_params=False,output_dir=str(out),run_name=run,workers=24)
    core.train(args)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','smoke','train'])
    parser.add_argument('--cache-root',required=True);parser.add_argument('--teacher',choices=['R','V']);parser.add_argument('--fold',type=int,choices=[3,0])
    parser.add_argument('--output-dir');parser.add_argument('--run-name');parser.add_argument('--steps',type=int,default=50)
    a=parser.parse_args();cache=Path(a.cache_root)
    if a.mode=='prepare': result=prepare(cache)
    elif a.mode=='smoke': result=smoke(cache,a.steps)
    else:
        assert all([a.teacher,a.output_dir,a.run_name]) and a.fold is not None
        train(cache,a.teacher,a.fold,Path(a.output_dir),a.run_name);result=None
    if result is not None:print(json.dumps(result,indent=2),flush=True)
