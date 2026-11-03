"""Fresh ordinary336 K24 folds0/1; frozen recipe, no Gold fitting/selection."""
import argparse,json,os,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parent
TASK=ROOT.parent
os.environ['V83_PACKAGE_ROOT']=str(ROOT)
import matched_core as matched
import train_core_v2_a as core
import train_v83 as base
import v83_aug_data as data
from token_slot_map import TokenSlotMap

def dump(name,d):base.atomic_json(TASK/name,d)
def source_check():
    cfg=json.loads((TASK/'QUEUE_CONFIG.json').read_text());assert cfg['instance_id']==55983617
    for name,digest in json.loads((ROOT/'SOURCE_PACKAGE_MANIFEST.json').read_text())['files'].items():
        assert base.sha(ROOT/name)==digest,name
    cache=Path(cfg['cache336']);verified=json.loads((cache/'VERIFIED.json').read_text())
    assert verified['status']=='PASS_ALL4410_CANONICAL336' and verified['instance_id']==55983617
    assert verified['inventory_sha256']==cfg['cache_inventory_sha256'] and verified['npz_count']==4410
    assert verified['source_image_size']==336 and verified['crop_mm']==140
    assert data.WINDOWS==24 and core.BATCH==8 and core.STOP==20 and core.SCHEDULE==24 and core.SWA_WINDOW==[18,19,20]
    return cfg,cache,verified

def frames():
    cfg,cache,verified=source_check()
    names=json.loads((ROOT/'assets/aux_names.json').read_text());assert len(names)==48
    frame,gold,receipt=matched.frame_for(cache,names,'V')
    assert len(frame)==4352 and frame.SampleID.nunique()==4352 and frame.ParentStudyInstanceUID.nunique()==4345
    assert len(gold)==58 and not set(frame.ParentStudyInstanceUID)&set(gold.ParentStudyInstanceUID)
    assert not set(frame.report_group)&set(gold.report_group)
    assert frame.groupby('report_group').fold.nunique().max()==1 and frame.groupby('ParentStudyInstanceUID').fold.nunique().max()==1
    assert not (gold[['mask__aux__'+n for n in names]]>0).any().any()
    return cfg,cache,verified,names,frame,gold,receipt

def configure(fold):
    cfg,cache,verified,names,frame,gold,receipt=frames()
    core.ARM=f'CORE_V20_ORDINARY_K24_CACHE336_F{fold}'
    core.V83AugBags=matched.ConfidenceBags;core.loader=data.loader
    core.load_data=lambda root,names:matched.frame_for(Path(root),names,'V')
    core.evaluate_fold=matched.original_reference_eval
    core.positive_weights=lambda fit,names:matched.fixed_weights(fold,names)
    original=core.code_hashes
    core.code_hashes=lambda:{**original(),'ordinary_k24_fold.py':base.sha(__file__),'matched_core.py':base.sha(ROOT/'matched_core.py')}
    core.IDENTITY_EXTRA={'recipe_variant':f'ordinary_teacherV_K24_cache336_fold{fold}','teacher':'V','source_image_size':336,'crop_mm':140,
       'crop_center':'geometric','architecture':'ordinary Core V20 H1AuxAug without Transformer','gold_in_training':False,'gold_checkpoint_selection':False,
       'gold_metric_role':'held_out_reused_development_diagnostic','main_targets_sha256':base.sha(ROOT/'assets/V_MAIN.csv'),
       'original_reference_sha256':base.sha(ROOT/'assets/FROZEN_REFERENCE.csv'),'fixed_pos_weight_sha256':base.sha(ROOT/'assets'/f'POS_WEIGHT_f{fold}.json'),
       'cache_actual_verification_sha256':base.sha(cache/'VERIFIED.json'),'source_package_manifest_sha256':base.sha(ROOT/'SOURCE_PACKAGE_MANIFEST.json'),
       'orientation_preflight_sha256':cfg['orientation_preflight_sha256']}
    return cfg,cache,verified,names,frame,gold,receipt

def cpu():
    cfg,cache,verified,names,frame,gold,receipt=frames();recipes=json.loads((TASK/'FOLD_RECIPES.json').read_text());slots=TokenSlotMap(ROOT/'assets/TOKEN_SLOT_MAP.csv')
    assert len(set(slots.counts)|slots.unmapped)==4352
    for row in frame.itertuples(index=False):
        with np.load(row.cache_path,allow_pickle=False) as z:
            assert str(z['sample_id'])==row.SampleID and z['bank'].shape[1:]==(336,336)
            slots.slot_ids(row.SampleID,int(z['mask'].sum()))
    reports={}
    for fold in [0,1]:
        fit=frame.loc[frame.fold!=fold].reset_index(drop=True);val=frame.loc[frame.fold==fold].reset_index(drop=True);expected=recipes[str(fold)]
        assert len(fit)==expected['fit'] and len(val)==expected['val']
        assert core.h.semantic(sorted(fit.SampleID.astype(str)))==expected['fit_sample_ids_sha256']
        for key in ['SampleID','ParentStudyInstanceUID','report_group']:assert not set(fit[key])&set(val[key])
        y,w=matched.effective_arrays(frame);assert int(((w==.25)&(w>0)).sum())==69
        ds=matched.ConfidenceBags(fit,names,train=True,seed=base.SEED,epoch=0,slot_map=slots)
        rows=[ds[i] for i in range(8)];batch=data.collate(rows)
        assert batch[0].shape[0]==8 and batch[0].shape[2:]==(3,336,336) and batch[0].shape[1]<=24
        assert all(torch.isfinite(x).all() for x in [batch[0],batch[2],batch[3],batch[4],batch[6],batch[7]])
        pw=matched.fixed_weights(fold,names);assert pw.shape==(60,) and torch.isfinite(pw).all()
        report={'status':'PASS_CPU_DATA_AND_SPLIT','fold':fold,**expected,'source_image_size':336,'crop_mm':140,'Gold58_outside_fit':True,
             'batch_shape':list(batch[0].shape),'confidence025_cells':69,'cache_verified_sha256':base.sha(cache/'VERIFIED.json')}
        reports[str(fold)]=report;dump(f'DATA_QA_f{fold}.json',report)
    dump('CPU_QA.json',{'status':'PASS_BOTH_FOLDS_CPU_QA','folds':reports,'optimizer_updates':0})

def cuda():
    cfg,cache,verified,names,frame,gold,receipt=frames();slots=TokenSlotMap(ROOT/'assets/TOKEN_SLOT_MAP.csv')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
    core.h.seed_all(base.SEED);model=core.make(ROOT/'assets').cuda().train();opt,_=core.optimizer(model,24,436,False)
    for group in opt.param_groups:
        for parameter in group['params']:
            if parameter.requires_grad:opt.state[parameter]={'step':torch.tensor(0.),'exp_avg':torch.zeros_like(parameter),'exp_avg_sq':torch.zeros_like(parameter)}
    torch.cuda.reset_peak_memory_stats();trials=[]
    for fold,n in [(0,8),(0,2),(1,8),(1,5)]:
        fit=frame.loc[frame.fold!=fold].reset_index(drop=True)
        ds=matched.ConfidenceBags(fit,names,train=True,seed=base.SEED,epoch=0,slot_map=slots,config=data.AugConfig(series_drop=.1,token_drop=0,jitter=1,params=True))
        x,mask,meta,y,lw,index,gain,params=data.collate([ds[i] for i in range(n)])
        assert x.shape==(n,24,3,336,336)
        x,mask,meta,y,lw,gain,params=[a.cuda() for a in [x,mask,meta,y,lw,gain,params]]
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(x,mask,meta,gain,params);main,aux,per=core.loss_parts(logits,y,lw,matched.fixed_weights(fold,names).cuda(),aux_weight=.25);loss=per.sum()
        assert torch.isfinite(loss);loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None:parameter.grad.div_(n)
        torch.nn.utils.clip_grad_norm_(model.parameters(),3.)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        torch.cuda.synchronize();trials.append({'fold':fold,'batch':n,'loss':float(loss.detach().cpu())})
        del x,mask,meta,y,lw,gain,params,logits,loss,main,aux,per
    peak=torch.cuda.max_memory_reserved()/2**30;assert peak<30.5
    dump('CUDA_QA.json',{'status':'PASS_B8_K24_AND_FINAL_BATCHES_2_5','optimizer_updates':0,'peak_reserved_gib':peak,'trials':trials})

def train(fold):
    cfg,cache,*_=configure(fold)
    gate=json.loads((TASK/'gates'/f'fold{fold}.json').read_text())
    assert gate['status']=='PASS_SINGLE_FRESH_FOLD_START' and gate['fold']==fold and gate['instance_id']==55983617
    assert gate['package_sha256']==base.sha(ROOT/'SOURCE_PACKAGE_MANIFEST.json') and gate['cache_verified_sha256']==base.sha(cache/'VERIFIED.json')
    assert gate['cpu_qa_sha256']==base.sha(TASK/'CPU_QA.json') and gate['cuda_qa_sha256']==base.sha(TASK/'CUDA_QA.json')
    out=TASK/f'fold{fold}'/'training';assert not out.exists()
    core.train(SimpleNamespace(cache_root=str(cache),slot_map=str(ROOT/'assets/TOKEN_SLOT_MAP.csv'),fold=fold,full=False,
      series_drop=.1,token_drop=.15,jitter=1,no_params=False,output_dir=str(out),run_name=f'core_v20_ordinary_k24_cache336_f{fold}',workers=12))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cpu','cuda','train']);p.add_argument('--fold',type=int,choices=[0,1],default=0);a=p.parse_args()
    {'cpu':cpu,'cuda':cuda,'train':lambda:train(a.fold)}[a.mode]()
