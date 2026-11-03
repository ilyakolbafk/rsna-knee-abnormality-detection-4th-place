"""Data/CPU qualification and one discarded CUDA smoke; never optimizer updates."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import fullfit_k24 as entry
import train_core_v2_a as core
from core_v2_a_model import H1AuxAug
from token_slot_map import TokenSlotMap

PACKAGE=Path(__file__).resolve().parent
CACHE=Path('/root/rsna_joint24_fullfit_gold336_r3/cache336')


def cpu():
    freeze=entry.prepare(CACHE)
    winner,_=entry.configure()
    fit,gold,_,pos,_=entry.verify(CACHE)
    gold_ids=set(gold.SampleID.astype(str))
    selected=pd.concat([fit.loc[~fit.SampleID.isin(gold_ids)].iloc[:4],fit.loc[fit.SampleID.isin(gold_ids)].iloc[:4]],ignore_index=True)
    slots=TokenSlotMap(PACKAGE/'assets/FULLFIT_TOKEN_SLOT_MAP.csv')
    ds=core.V83AugBags(selected,entry.canonical.NAMES,train=True,seed=core.SEED,epoch=0,slot_map=slots,config=core.AugConfig.identity())
    if not np.array_equal(ds.y[4:,:12],selected.iloc[4:][core.h.TARGETS].to_numpy(np.float32)) or not (ds.label_weight[4:,:12]==1).all() or not (ds.label_weight[4:,12:]==0).all():
        raise ValueError('Gold fit labels/confidence/aux parity failed')
    torch.set_num_threads(4);torch.manual_seed(core.SEED)
    row=fit.iloc[0]
    with np.load(row.cache_path,allow_pickle=False) as z:
        valid=np.flatnonzero(z['mask']);chosen=valid[:24]
        if len(chosen)!=24:raise ValueError('CPU cache sample has fewer than24 windows')
        x=torch.from_numpy(np.ascontiguousarray(z['bank'][z['refs'][chosen]])).float().div_(255).unsqueeze(0)
        meta=torch.from_numpy(np.ascontiguousarray(z['meta'][chosen])).unsqueeze(0)
    mask=torch.ones((1,24),dtype=torch.bool);gain=torch.ones(1)
    base=H1AuxAug(PACKAGE/'assets/coatnet_imagenet.safetensors',chunk=2,checkpointing=False)
    if winner=='transformer':
        from context_head import ContextMIL
        from slot_features import append_slots
        with torch.random.fork_rng(devices=[]):model=ContextMIL(None,backbone=base.backbone,block='transformer',chunk=2,checkpointing=False)
        model.load_native20(base.state_dict())
        meta_candidate=append_slots(meta[0],torch.from_numpy(slots.slot_ids(str(row.SampleID),len(valid))[:24])).unsqueeze(0)
    else:
        model=H1AuxAug(None,chunk=2,checkpointing=False);model.load_state_dict(base.state_dict(),strict=True);meta_candidate=meta
    base.eval();model.eval()
    with torch.inference_mode():expected=base(x,mask,meta,gain);actual=model(x,mask,meta_candidate,gain)
    delta=float((expected-actual).abs().max())
    if delta!=0 or actual.shape!=(1,60):raise ValueError('Actual encoder CPU parity failed')
    return {'status':'PASS_SELECTED_K24_FULLFIT_CPU','architecture':winner,'max_abs_logit_delta':delta,'real_pixel_tokens':24,
            'fit_samples':len(fit),'Gold_in_fit':58,'Gold_aux_masked':True,'optimizer_updates':0,'CUDA_initialized':torch.cuda.is_initialized(),'freeze':freeze}


def cuda():
    import subprocess
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True)
    if apps.returncode or apps.stdout.strip():raise ValueError('GPU busy before one smoke')
    winner,_=entry.configure();fit,gold,_,pos,freeze=entry.verify(CACHE)
    gold_ids=set(gold.SampleID.astype(str))
    selected=pd.concat([fit.loc[~fit.SampleID.isin(gold_ids)].iloc[:4],fit.loc[fit.SampleID.isin(gold_ids)].iloc[:4]],ignore_index=True)
    slots=TokenSlotMap(PACKAGE/'assets/FULLFIT_TOKEN_SLOT_MAP.csv')
    ds=core.V83AugBags(selected,entry.canonical.NAMES,train=True,seed=core.SEED,epoch=0,slot_map=slots,
        config=core.AugConfig(series_drop=0.,token_drop=0.,jitter=1,params=True))
    batch=next(iter(core.loader(ds,8,0,core.SEED,False)))
    x,mask,meta,y,lw,index,gain,params=batch
    if x.shape[:2]!=(8,24) or int(mask.sum())!=192 or y.shape!=(8,60):raise ValueError('Smoke not worst-case B8×24')
    torch.manual_seed(core.SEED);torch.cuda.manual_seed_all(core.SEED);torch.set_num_threads(4)
    model=core.make(PACKAGE/'assets').to('cuda');model.train()
    x,mask,meta,y,lw,gain,params=[v.to('cuda') for v in (x,mask,meta,y,lw,gain,params)]
    model.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        z=model(x,mask,meta,gain,params);main,aux,losses=core.loss_parts(z,y,lw,pos.to('cuda'),aux_weight=core.AUX_WEIGHT);loss=losses.sum()
    if not torch.isfinite(loss) or not torch.equal(aux[4:],torch.zeros_like(aux[4:])):raise ValueError('Gold/loss smoke failed')
    loss.backward();torch.cuda.synchronize();elapsed=time.monotonic()-started
    if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise ValueError('Nonfinite gradient')
    peak=torch.cuda.max_memory_reserved()/2**30
    if peak>=30.5 or model.clsW.grad is None or float(model.clsW.grad.norm())<=0:raise ValueError('Full-fit memory/gradient gate failed')
    return {'status':'PASS_WORST_CASE_B8_K24_CUDA_SMOKE','architecture':winner,'optimizer_updates':0,'checkpoint_written':False,
            'weights_discarded':True,'batch':8,'max_windows':24,'valid_windows':192,'peak_reserved_gib':peak,
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'elapsed_seconds':elapsed,
            'fit_rows':len(fit),'Gold_in_fit':58,'Gold_aux_masked':True,'GPU':torch.cuda.get_device_name(0)}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['cpu','cuda']);args=parser.parse_args()
    print(json.dumps(cpu() if args.mode=='cpu' else cuda(),allow_nan=False))
