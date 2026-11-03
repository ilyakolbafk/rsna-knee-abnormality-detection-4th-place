"""Fresh VM336 K24 fullfit: the accepted ordinary recipe plus historical Mirror50."""
import argparse, json, os, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
TASK=ROOT.parent
os.environ['V83_PACKAGE_ROOT']=str(ROOT)
import numpy as np
import torch
import fullfit_k24 as parent
import matched_core as matched
import train_core_v2_a as core
import train_v83 as base
import v83_aug_data as data
from token_slot_map import TokenSlotMap
from mirror_tokens import MirrorPlaneMap, mirror_draw, reflect_tokens
from aug336 import IDENTITY

def source_check(cache):
    for n,h in json.loads((ROOT/'SOURCE_PACKAGE_MANIFEST.json').read_text())['files'].items():
        assert base.sha(ROOT/n)==h, 'PACKAGE_DRIFT:'+n
    v=json.loads((cache/'VERIFIED.json').read_text());cfg=json.loads((TASK/'QUEUE_CONFIG.json').read_text())
    assert v['status']=='PASS_ALL4410_CANONICAL336' and v['npz_count']==4410
    assert v['instance_id']==55518941 and v['inventory_sha256']==cfg['cache_inventory_sha256']
    assert v['source_image_size']==336 and v['crop_mm']==140 and v['orientation_state']=='canonical_L_v8_3'
    assert parent.configure()[0]=='ordinary'
    fit,gold,receipt,pos,freeze=parent.verify(cache)
    assert len(fit)==4410 and fit.SampleID.nunique()==4410 and fit.ParentStudyInstanceUID.nunique()==4403
    assert len(gold)==58 and set(gold.SampleID)<=set(fit.SampleID)
    names=json.loads((ROOT/'assets/aux_names.json').read_text())
    assert not (gold[['mask__aux__'+n for n in names]]>0).any().any()
    return fit,gold,receipt,pos,freeze,names

def batches(cache):
    fit,gold,receipt,pos,freeze,names=source_check(cache)
    slots=TokenSlotMap(ROOT/'assets/FULLFIT_TOKEN_SLOT_MAP.csv')
    ds=matched.ConfidenceBags(fit,names,train=True,seed=base.SEED,epoch=0,slot_map=slots,
        config=data.AugConfig(series_drop=0,token_drop=0,jitter=1,params=True))
    batch=data.collate([ds[i] for i in range(8)])
    assert batch[0].shape==(8,24,3,336,336)
    eds=matched.ConfidenceBags(gold.loc[gold.tokens==64].iloc[:1],names,train=False,seed=base.SEED,epoch=0)
    ev=next(iter(data.loader(eds,1,0,base.SEED,False)))
    assert int(ev[1].sum())==64
    return fit,gold,pos,names,ds,batch,ev

def cpu(cache):
    torch.set_num_threads(4)
    fit,gold,pos,names,ds,batch,ev=batches(cache)
    planes=MirrorPlaneMap(ROOT/'assets/FULLFIT_TOKEN_SLOT_MAP.csv')
    for row in fit.itertuples(index=False):
        with np.load(row.cache_path,allow_pickle=False) as z:
            assert str(z['sample_id'])==row.SampleID and z['bank'].shape[1:]==(336,336)
            ds.slot_map.slot_ids(row.SampleID,int(z['mask'].sum()))
            planes.plane_ids(row.SampleID,int(z['mask'].sum()))
    # The confidence overlay must survive the mirror seam; a forced reflection changes only x/meta.
    ds.force_mirror=False;a=ds[0];ds.force_mirror=True;b=ds[0]
    for i in [2,3,4,5,6]:
        assert torch.equal(a[i],b[i]) if torch.is_tensor(a[i]) else a[i]==b[i]
    assert torch.isfinite(batch[3]).all() and torch.isfinite(batch[4]).all() and pos.shape==(60,)
    # Gold58 is part of fit but contributes no report auxiliary supervision.
    gd=matched.ConfidenceBags(fit.loc[fit.SampleID.isin(gold.SampleID)].reset_index(drop=True),names,
        train=True,seed=base.SEED,epoch=0,slot_map=ds.slot_map,config=data.AugConfig.identity())
    assert all(not gd[i][3][12:].any() for i in range(58))
    model=core.make(ROOT/'assets').eval();x=batch[0];gain=batch[6]
    with torch.inference_mode():
        assert torch.equal(model.features(x[0,:2],gain[:1].expand(2)),
            model.features_aug(x[0,:2],gain[:1].expand(2),torch.from_numpy(IDENTITY).repeat(2,1)))
    base.atomic_json(TASK/'job/CPU_QA.json',{'status':'PASS_FULLFIT4410_NATIVE336_MIRROR50_CPU',
        'samples':4410,'Gold58_in_fit':True,'Gold58_aux_zero':True,'all_actual_slot_plane_counts_checked':True,
        'encoder_identity_parity':True,'confidence_overlay_preserved':True,'optimizer_updates':0})

def cuda(cache):
    assert json.loads((TASK/'job/CPU_QA.json').read_text())['status']=='PASS_FULLFIT4410_NATIVE336_MIRROR50_CPU'
    fit,gold,pos,names,ds,batch,ev=batches(cache);torch.set_num_threads(4);core.h.seed_all(base.SEED)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
    model=core.make(ROOT/'assets').cuda();opt,scheduler=core.optimizer(model,24,552,False)
    for group in opt.param_groups:
        for p in group['params']:opt.state[p]={'step':torch.zeros(()),'exp_avg':torch.zeros_like(p),'exp_avg_sq':torch.zeros_like(p)}
    assert scheduler.total_steps==24*552
    pw=pos.cuda();records=[];evals=[];torch.cuda.reset_peak_memory_stats()
    for k,n in [(24,8),(23,8),(18,8),(24,8),(24,2)]:
        model.train();opt.zero_grad(set_to_none=True);x,mask,meta,y,w,index,gain,params=batch
        x,mask,meta=[z[:n,:k].cuda() for z in [x,mask,meta]]
        y,w,gain=[z[:n].cuda() for z in [y,w,gain]];params=params[:n,:k].cuda()
        params[:,:,11]=1.5;params[:,:,12]=.03;started=time.monotonic()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(x,mask,meta,gain,params);_,_,losses=core.loss_parts(logits,y,w,pw,aux_weight=.25);loss=losses.sum()
        assert torch.isfinite(loss);loss.backward()
        for p in model.parameters():
            if p.grad is not None:p.grad.div_(n)
        torch.nn.utils.clip_grad_norm_(model.parameters(),3,error_if_nonfinite=True);torch.cuda.synchronize()
        records.append({'batch':n,'windows':k,'seconds':time.monotonic()-started});opt.zero_grad(set_to_none=True)
        del x,mask,meta,y,w,gain,params,logits,losses,loss
        model.eval();started=time.monotonic()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            x,mask,meta,_,_,_,gain,_=ev;model(x.cuda(),mask.cuda(),meta.cuda(),gain.cuda())
        torch.cuda.synchronize();evals.append(time.monotonic()-started)
    peak=torch.cuda.max_memory_reserved()/2**30
    assert peak<30.5 and scheduler.last_epoch==0 and all(float(v['step'])==0 for v in opt.state.values())
    base.atomic_json(TASK/'job/CUDA_QA.json',{'status':'PASS_NATIVE336_K24_B8_ZEROOPT','peak_reserved_gib':peak,
        'records':records,'worst_full_batch_seconds':max(r['seconds'] for r in records if r['batch']==8),
        'actual64_eval_seconds':max(evals),'optimizer_updates':0,'checkpoint_written':False})

def train(cache,out):
    source_check(cache)
    gate=json.loads((TASK/'job/START_GATE.json').read_text())
    assert gate['status']=='PASS_ONE_FRESH_VM336_K24_MIRROR50_FULLFIT20'
    assert gate['entry_sha256']==base.sha(__file__) and gate['package_sha256']==base.sha(ROOT/'SOURCE_PACKAGE_MANIFEST.json')
    assert gate['cache_verified_sha256']==base.sha(cache/'VERIFIED.json') and time.time()<gate['launch_valid_until_unix']
    original=core.train
    def run(args):
        args.run_name='core_v20_vm336_k24_mirror50_fullfit_gold58'
        core.ARM='CORE_V20_VM336_K24_MIRROR50_FULLFIT_GOLD58'
        extra=dict(core.IDENTITY_EXTRA)
        extra.update(augmentation=json.loads((ROOT/'assets/MIRROR_AUGMENTATION.json').read_text()),recipe_variant='VM336_K24_Mirror50_fullfit_gold58',source_image_size=336,model_image_size=384,crop_mm=140,
            mirror_probability=.5,mirror_scope='one whole-knee draw per SampleID epoch after gather; sagittal triplet reversal and direction signs; coronal/axial width reflection',
            canonical_cache_orientation='canonical_L_v8_3',mirror_eval=False,
            source_package_manifest_sha256=base.sha(ROOT/'SOURCE_PACKAGE_MANIFEST.json'),
            mirror_entry_sha256=base.sha(__file__),mirror_transform_sha256=base.sha(ROOT/'mirror_tokens.py'),
            cache_actual_verification_sha256=base.sha(cache/'VERIFIED.json'),
            baseline_averaged_sha256='9b6eab4a07699ae622f7d210ac7ffcabf592dba016511bd6abaf98eb34b04adb')
        core.IDENTITY_EXTRA=extra
        old=core.code_hashes;core.code_hashes=lambda:{**old(),'mirror_fullfit.py':base.sha(__file__),'mirror_tokens.py':base.sha(ROOT/'mirror_tokens.py')}
        return original(args)
    core.train=run
    parent.train(cache,out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cpu','cuda','train']);p.add_argument('--cache-root',type=Path,required=True);p.add_argument('--output-dir',type=Path)
    a=p.parse_args()
    if a.mode=='train':
        assert a.output_dir is not None;train(a.cache_root,a.output_dir)
    else:globals()[a.mode](a.cache_root)
