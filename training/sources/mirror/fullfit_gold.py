"""One fresh preserved Core20 full-fit with explicitly expert-labelled Gold58 in fit."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import math
import os
import time
from importlib.metadata import version

ROOT = Path(__file__).resolve().parent
os.environ['V83_PACKAGE_ROOT'] = str(ROOT)
import numpy as np
import pandas as pd
import torch
import matched_core as matched
import train_core_v2_a as core
import train_v83 as base
import h1_base as h
import models
from token_slot_map import TokenSlotMap
from v83_aug_data import AugConfig, loader

NAMES = json.loads((ROOT/'assets/aux_names.json').read_text())
MAIN = matched.MAIN
IDS = matched.IDS
ROLE = 'in_sample_training_diagnostic; Gold58 explicitly included in fit; not held-out validation'
RUN = 'core_v2a_main_V20_fullfit_gold58'

def dump(name, value):
    base.atomic_json(ROOT/name, value)

def frames(cache):
    r,gold,receipt = matched.frame_for(cache,NAMES,'R')
    v,other_gold,_ = matched.frame_for(cache,NAMES,'V')
    expert = pd.read_csv(ROOT/'assets/GOLD58_EXPERT_MAIN.csv',dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID')
    assert len(expert)==58 and expert.index.is_unique
    assert set(expert.index)==set(gold.StudyInstanceUID.astype(str))
    assert np.array_equal(gold[h.TARGETS],other_gold[h.TARGETS])
    assert np.array_equal(gold[h.TARGETS].to_numpy(),expert.loc[gold.StudyInstanceUID.astype(str),h.TARGETS].to_numpy())
    assert np.isin(gold[h.TARGETS].to_numpy(),[0,1]).all()
    assert gold[['label_mask__'+t for t in h.TARGETS]].eq(1).all().all()
    assert gold[['confidence__'+t for t in h.TARGETS]].eq(1).all().all()
    assert gold[['mask__aux__'+n for n in NAMES]].eq(0).all().all()
    assert gold[['confidence__aux__'+n for n in NAMES]].eq(0).all().all()
    assert set(gold.report_group).isdisjoint(set(v.report_group))
    fit = pd.concat([v,gold],ignore_index=True)
    reference = pd.concat([r,gold],ignore_index=True)
    assert len(fit)==4410 and fit.SampleID.is_unique and fit.ParentStudyInstanceUID.nunique()==4403
    bilateral = v.ParentStudyInstanceUID.astype(str).duplicated(keep=False)
    assert bilateral.sum()==14
    assert np.array_equal(v.loc[bilateral,MAIN],r.loc[bilateral,MAIN],equal_nan=True)
    assert np.array_equal(matched.effective_arrays(v)[1],matched.effective_arrays(r)[1])
    assert np.array_equal(v[['aux__'+n for n in NAMES]],r[['aux__'+n for n in NAMES]],equal_nan=True)
    return fit,reference,gold,receipt

def weights(reference):
    y,w = matched.effective_arrays(reference)
    assert (w.sum(axis=0)>0).all()
    prevalence = np.clip((y*w).sum(axis=0)/w.sum(axis=0),.03,.7)
    main = torch.tensor(np.clip((1-prevalence)/prevalence,1,10),dtype=torch.float32)
    pos = models.positive_weights(reference,NAMES)
    pos[:12] = main
    assert pos.shape==(60,) and torch.isfinite(pos).all()
    return pos

def verify_slots(cache,fit):
    slot = TokenSlotMap(ROOT/'assets/FULLFIT_TOKEN_SLOT_MAP.csv')
    local = pd.read_csv(ROOT/'assets/GOLD_CACHE_MANIFEST.csv',dtype={'StudyInstanceUID':str,'SampleID':str})
    actual = pd.read_csv(cache/'GOLD58_MANIFEST.csv',dtype={'StudyInstanceUID':str,'SampleID':str})
    fields=['StudyInstanceUID','SampleID','side','tokens','bytes','sha256','decoded_source_frames','source_errors']
    assert np.array_equal(local.sort_values('SampleID')[fields].to_numpy(),actual.sort_values('SampleID')[fields].to_numpy())
    gold_hashes=local.set_index('SampleID')['sha256'].to_dict()
    for row in fit.itertuples(index=False):
        if str(row.SampleID) in gold_hashes:
            assert base.sha(row.cache_path)==gold_hashes[str(row.SampleID)], 'Gold NPZ byte identity changed'
        with np.load(row.cache_path,allow_pickle=False) as z:
            assert str(z['sample_id'])==str(row.SampleID)
            slot.slot_ids(str(row.SampleID),int(z['mask'].sum()))
    return {'samples':len(fit),'Gold58':58,'NPZ_identity_and_token_mask_checked':True,
            'Gold58_NPZ_SHA256_checked':True,'slot_boundaries':'source-derived builder/series geometry/count map; not independent pixel plane proof'}

def prepare(cache):
    base.guard()
    path=ROOT/'assets/FULLFIT_TARGET_FREEZE.json'
    assert not path.exists(), 'Never overwrite an existing fullfit freeze'
    fit,reference,gold,_=frames(cache)
    slot_receipt=verify_slots(cache,fit)
    pos=weights(reference)
    fit[IDS+MAIN].to_csv(ROOT/'assets/FULLFIT_MAIN.csv',index=False)
    reference[IDS+MAIN].to_csv(ROOT/'assets/FULLFIT_R_WEIGHT_REFERENCE.csv',index=False)
    dump('assets/POS_WEIGHT_FULLFIT.json',dict(zip(h.TARGETS+NAMES,pos.tolist())))
    record={'status':'PASS_FULLFIT_GOLD_TARGET_FREEZE','samples':len(fit),'parents':fit.ParentStudyInstanceUID.nunique(),
            'nongold_samples':4352,'Gold58_samples':58,'bilateral_rows':14,'Gold58_in_fit':True,
            'Gold58_main_source':'expert binary labels, confidence1, mask1','Gold58_aux':'unknown/masked0; no invented reader/report aux',
            'sampling':'one occurrence per4410SampleID, usual shuffled loader, no Goldoversampling',
            'main_source':'nongold pinned V_MAIN with inherited R support/confidence, mandatorybilateral14 fromR; expertGold58',
            'pos_weight_rule':'same confidence*mask-weighted soft R prevalence clipped[.03,.7], ratio clipped[1,10]; allnongoldR plus expertGold; aux originalformula',
            'gold_metric_role':ROLE,'selected_epochs':[18,19,20],'schedule':24,'stop':20,
            'training_sample_ids_sha256':h.semantic(sorted(fit.SampleID.astype(str))),
            'Gold58_ids_sha256':h.semantic(sorted(gold.SampleID.astype(str))),
            'target_semantic_sha256':matched.semantic(fit[IDS+MAIN].to_dict(orient='records')),
            'parent_target_freeze_sha256':base.sha(ROOT/'assets/PARENT_TARGET_FREEZE.json'),
            'current_cache_target_binding_sha256':base.sha(ROOT/'assets/TARGET_FREEZE.json'),
            'cache_release_sha256':base.sha(cache/'GLOBAL_RELEASE_RECEIPT.json'),
            'code_sha256':base.sha(__file__),'slot_check':slot_receipt,
            'files':{n:base.sha(ROOT/'assets'/n) for n in ['FULLFIT_MAIN.csv','FULLFIT_R_WEIGHT_REFERENCE.csv','POS_WEIGHT_FULLFIT.json','FULLFIT_TOKEN_SLOT_MAP.csv','FULLFIT_TOKEN_SLOT_MAP_RECEIPT.json','GOLD58_EXPERT_MAIN.csv']}}
    dump('assets/FULLFIT_TARGET_FREEZE.json',record)
    return record

def verify(cache):
    freeze=json.loads((ROOT/'assets/FULLFIT_TARGET_FREEZE.json').read_text())
    assert freeze['status']=='PASS_FULLFIT_GOLD_TARGET_FREEZE' and freeze['code_sha256']==base.sha(__file__)
    for name,digest in freeze['files'].items():assert base.sha(ROOT/'assets'/name)==digest,name
    assert base.sha(cache/'GLOBAL_RELEASE_RECEIPT.json')==freeze['cache_release_sha256']
    fit,reference,gold,receipt=frames(cache)
    assert matched.semantic(fit[IDS+MAIN].to_dict(orient='records'))==freeze['target_semantic_sha256']
    values=json.loads((ROOT/'assets/POS_WEIGHT_FULLFIT.json').read_text())
    pos=torch.tensor([values[n] for n in h.TARGETS+NAMES],dtype=torch.float32)
    assert torch.equal(weights(reference),pos)
    return freeze,fit,gold,receipt,pos

def smoke(cache,steps):
    base.guard();freeze,fit,gold,_,pos=verify(cache)
    assert torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported()
    device=torch.device('cuda:0');h.seed_all(base.SEED);torch.set_num_threads(4)
    goldids=set(gold.SampleID.astype(str))
    nongold=fit.loc[~fit.SampleID.isin(goldids)].iloc[:4]
    selected=pd.concat([nongold,fit.loc[fit.SampleID.isin(goldids)].iloc[:4]],ignore_index=True)
    ds=matched.ConfidenceBags(selected,NAMES,train=True,seed=base.SEED,epoch=0,
                              slot_map=TokenSlotMap(ROOT/'assets/FULLFIT_TOKEN_SLOT_MAP.csv'),config=AugConfig())
    assert np.array_equal(ds.y[4:,:12],selected.iloc[4:][h.TARGETS].to_numpy(np.float32))
    assert (ds.label_weight[4:,:12]==1).all() and (ds.label_weight[4:,12:]==0).all()
    model=core.make(ROOT/'assets').to(device)
    opt,scheduler=models.optimizer(model,core.SCHEDULE,math.ceil(len(fit)/core.BATCH),False)
    timings=[]
    for step in range(steps):
        data=loader(ds,8,0,base.SEED+step,False)
        x,mask,meta,y,lw,index,gain,params=next(iter(data))
        batch=[v.to(device) for v in [x,mask,meta,y,lw,gain,params]]
        x,mask,meta,y,lw,gain,params=batch
        t=time.monotonic();model.train();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            z=model(x,mask,meta,gain,params)
            main,aux,losses=models.loss_parts(z,y,lw,pos.to(device),aux_weight=core.AUX_WEIGHT)
            loss=losses.sum()
        assert torch.isfinite(loss) and torch.isfinite(z).all()
        assert torch.equal(aux[4:],torch.zeros_like(aux[4:]))
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None:parameter.grad.div_(len(y))
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),3.0,error_if_nonfinite=True)
        assert torch.isfinite(norm) and float(norm)>0
        opt.step();scheduler.step();torch.cuda.synchronize();timings.append(time.monotonic()-t)
    record={'status':'PASS_CUDA_FULLFIT_GOLD_SMOKE','steps':steps,'code_sha256':base.sha(__file__),
            'target_freeze_sha256':base.sha(ROOT/'assets/FULLFIT_TARGET_FREEZE.json'),
            'Gold_main_labels_confidence_and_aux_zero_checked':True,'finite_forward_backward_updates':True,
            'mixed_batch_samples':8,'Gold_in_mixed_batch':4,'smoke_step_seconds':timings,
            'timing_not_fit_eta':True,'gpu':torch.cuda.get_device_name(0)}
    dump('SMOKE_RESULT.json',record)
    return record

def train(cache,out):
    assert not out.exists(), 'Fresh fullfit only; no implicit resume or overwrite'
    freeze,fit,gold,receipt,pos=verify(cache)
    sm=json.loads((ROOT/'SMOKE_RESULT.json').read_text())
    assert sm['status']=='PASS_CUDA_FULLFIT_GOLD_SMOKE' and sm['code_sha256']==base.sha(__file__)
    assert sm['target_freeze_sha256']==base.sha(ROOT/'assets/FULLFIT_TARGET_FREEZE.json')
    core.ARM='CORE_V2_A_MAIN_V20_FULLFIT_GOLD58'
    core.V83AugBags=matched.ConfidenceBags
    core.load_data=lambda cache,names:(fit,gold,receipt)
    core.positive_weights=lambda frame,names:pos.clone()
    original_hashes=core.code_hashes
    core.code_hashes=lambda:{**original_hashes(),'fullfit_gold.py':base.sha(__file__),'matched_core.py':base.sha(ROOT/'matched_core.py')}
    core.IDENTITY_EXTRA={'recipe_variant':'fullfit_gold58_short20','teacher':'V_nonGold_expert_Gold58',
                         'runtime_versions':{n:version(n) for n in ['torch','torchvision','timm','numpy','pandas']},
                         'gold_in_training':True,'gold_fit_or_selection':True,'gold_checkpoint_selection':False,
                         'gold_metric_role':ROLE,'validation_each_epoch':False,'Gold58_diagnostic_each_epoch':True,
                         'target_freeze_sha256':base.sha(ROOT/'assets/FULLFIT_TARGET_FREEZE.json'),
                         'main_targets_sha256':freeze['files']['FULLFIT_MAIN.csv'],
                         'fixed_pos_weight_sha256':freeze['files']['POS_WEIGHT_FULLFIT.json'],
                         'common_support_sha256':json.loads((ROOT/'assets/TARGET_FREEZE.json').read_text())['common_support_sha256'],
                         'main_confidence_used':True,'Gold58_aux_masked':True,'Gold58_oversampling':False,
                         'parent_recipe':'exact preserved CoreV20 model/E1augmentation/optimizer/loss/24-20-average18-20; new fullfitpool+expertGold58'}
    old_atomic=core.atomic_json
    def role_json(path,value):
        if isinstance(value,dict) and Path(path).name in ['RESULT.json','LIVE.json','EPOCH_AVERAGING.json']:
            value={**value,'gold_in_training':True,'gold_metric_role':ROLE}
            if value.get('phase')=='gold_validation':value['phase']='gold_in_sample_diagnostic'
        elif isinstance(value,list) and Path(path).name=='history.json':
            value=[{**x,'gold_metric_role':ROLE} for x in value]
        return old_atomic(path,value)
    core.atomic_json=role_json
    args=SimpleNamespace(cache_root=str(cache),slot_map=str(ROOT/'assets/FULLFIT_TOKEN_SLOT_MAP.csv'),
                         fold=3,full=True,series_drop=.10,token_drop=.15,jitter=1,no_params=False,
                         output_dir=str(out),run_name=RUN,workers=24)
    core.train(args)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','smoke','train']);a.add_argument('--cache-root',required=True)
    a.add_argument('--output-dir');a.add_argument('--steps',type=int,default=8);args=a.parse_args();cache=Path(args.cache_root)
    if args.mode=='prepare':result=prepare(cache)
    elif args.mode=='smoke':result=smoke(cache,args.steps)
    else:
        assert args.output_dir;train(cache,Path(args.output_dir));result=None
    if result is not None:print(json.dumps(result,indent=2),flush=True)
