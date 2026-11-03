"""Fullfit-only, exactly one requested Core6/VM arm. No fold CLI or 110 path."""
import os,sys
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE','0')
from pathlib import Path
import argparse,fcntl,math,time,random,json,gc
import numpy as np
import torch
import psutil
from common import ROOT,TARGET,GPU_UUIDS,POLICY,read,write,sha,pin_check
from admission import host_identity,inspect_inputs,check_targets
from model import Model,clinical_loss
from main_data import MainDataset
from fm_data import EpochDataset
from fm_engine import backward as fm_backward
from augmentation import Augmentor
from loop import run,atomic_torch,same,RECIPE_SCHEMA
from recovery_state import freeze,rng_state,restore_rng
from ema_state import EMA
from clipping import fm_clip
from run_authorization import require_run_intent


def make_model(arm,device,config):
    torch.manual_seed(1000+17*config['seed_base'])
    model=Model(ROOT/'assets/pretrain.pth',arm,config['chunk']).to(device)
    bases=config['head_bases'][arm];model.initialize_priors(bases['main'],bases.get('oai_vm'))
    torch.manual_seed(1000+17*config['seed_base'])
    if str(device).startswith('cuda'):torch.cuda.manual_seed_all(1000+17*config['seed_base'])
    return model


def make_optimizer(model,config,steps):
    optimizer=torch.optim.AdamW(model.parameter_groups(),weight_decay=config['weight_decay'])
    total=steps*config['epochs'];warm=max(1,int(config['warmup_fraction']*total))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:(s+1)/warm if s<warm else .5*(1+math.cos(math.pi*(s-warm)/max(1,total-warm))))
    return optimizer,scheduler


def objective(config):
    def loss_step(model,batch,fm_batch,epoch,step,device):
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=str(device).startswith('cuda')):value=clinical_loss(model,batch)
        if value is not None:
            if not bool(torch.isfinite(value)):raise ValueError('NONFINITE_MAIN_LOSS')
            value.backward()
        fm=fm_backward(model,fm_batch,config['fm_weight'],config['fm_chunk'],seed_base=config['seed_base'],rng_key=0,epoch=epoch,step=step,device=device)
        return {'clinical_lr':float(value.detach()) if value is not None else 0.,**fm}
    return loss_step


def move(batch,device):return {k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}


def recovery_binding(arm,package,gpu,inputs):
    return {'package_sha256':package,'arm':arm,'GPU_UUID':gpu,'target':TARGET,'config_sha256':sha(ROOT/'CONFIG.json'),'inputs_sha256':sha(ROOT/'runtime/INPUTS.json'),
            'targets_sha256':sha(ROOT/f'assets/{arm}_targets.npz'),'cache_acceptance':inputs['acceptance'],'recipe_schema':RECIPE_SCHEMA,'orientation_policy':POLICY}


def smoke(arm,dataset,inputs,config,device,package):
    from torch.utils.data import default_collate
    batch=lambda ids:move(default_collate([dataset[i] for i in ids]),device)
    model=make_model(arm,device,config);model.train();tests=[];source_grad={}
    for uid,expected in read(ROOT/'assets/TEST3_EXPECTED.json').items():
        i=next(i for i,r in enumerate(dataset.rows) if r['id']==uid);b=batch([i])
        if not torch.equal(b['main_probability'][0].cpu(),torch.tensor(expected,dtype=torch.float32)) or not bool((b['main_weight']==1).all()):raise ValueError('ACTUAL_LOADER_TEST3_VM36')
        b=Augmentor(config['seed_base'])(b,1,1)
        with torch.autocast('cuda',dtype=torch.bfloat16):value=clinical_loss(model,b)
        value.backward();tests.append({'id':uid,'active_cells':12,'actual_tokens':int(b['mask'].sum()),'loss_finite':bool(torch.isfinite(value))});model.zero_grad(set_to_none=True)
    for source in [0,1]:
        key='main_weight' if source==0 or arm=='core6' else 'oai_vm_weight'
        indices=[i for i,r in enumerate(dataset.rows) if r['source_id']==source and np.any(dataset.targets[key][dataset.byid[r['id']]]>0)]
        b=Augmentor(config['seed_base'])(batch([indices[0]]),1,1)
        with torch.autocast('cuda',dtype=torch.bfloat16):value=clinical_loss(model,b)
        value.backward();mass=sum(float(p.grad.abs().sum()) for p in model.encoder.parameters() if p.grad is not None)
        if not math.isfinite(mass) or mass<=0:raise ValueError('SOURCE_ENCODER_NO_GRADIENT')
        if arm=='vm':
            inactive=model.oai_vm if source==0 else model.main
            if any(p.grad is not None for p in inactive.parameters()):raise ValueError('CROSS_DOMAIN_HEAD_GRADIENT')
            del inactive
        source_grad[str(source)]=mass;model.zero_grad(set_to_none=True)
    # B2x176 capacity is a memory/stability smoke; repeated windows here are
    # explicitly QA-only. The real fit sampler remains unique and cover-based.
    b=batch([0,1]);real=b['x'][b['mask']];met=b['meta'][b['mask']]
    force=torch.arange(176,device=device)%len(real);b['x']=real[force][None].repeat(2,1,1,1,1);b['meta']=met[force][None].repeat(2,1,1)
    b['mask']=torch.ones(2,176,dtype=torch.bool,device=device);b=Augmentor(config['seed_base'])(b,1,1)
    fm=EpochDataset(inputs['fastMRI'],1,config['seed_base'],1)[0]
    torch.cuda.reset_peak_memory_stats();opt,sched=make_optimizer(model,config,len(dataset)//2);ema=EMA(model)
    values=objective(config)(model,b,fm,1,1,device);norm=fm_clip(model,config['gradient_clip'])
    if not bool(torch.isfinite(norm)):raise ValueError('SMOKE_NONFINITE_NORM')
    opt.step();sched.step();ema.update(model,sched.last_epoch);model.zero_grad(set_to_none=True)
    for start in range(0,800,config['chunk']):
        n=min(config['chunk'],800-start);image=real[torch.arange(n,device=device)%len(real),:,16:368,16:368]
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=model.fm_dense(image)
        (logits.float().square().mean()/math.ceil(800/config['chunk'])).backward()
    model.zero_grad(set_to_none=True)
    snapshot={'model':freeze(model.state_dict()),'optimizer':freeze(opt.state_dict()),'scheduler':freeze(sched.state_dict()),'ema':ema.state_dict(),'rng':rng_state()}
    file=ROOT/'runtime'/f'gpu_restore_{arm}.pt';atomic_torch(file,snapshot);digest=sha(file)
    del model,opt,sched,ema,value,logits,b,real,met,image;gc.collect();torch.cuda.empty_cache()
    model=make_model(arm,device,config);opt,sched=make_optimizer(model,config,len(dataset)//2);ema=EMA(model)
    if sha(file)!=digest:raise ValueError('GPU_RECOVERY_FULLSHA')
    saved=torch.load(file,map_location='cpu',weights_only=False);model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);sched.load_state_dict(saved['scheduler']);ema.load_state_dict(saved['ema'],model);restore_rng(saved['rng'])
    if not all([same(freeze(model.state_dict()),snapshot['model']),same(freeze(opt.state_dict()),snapshot['optimizer']),same(sched.state_dict(),snapshot['scheduler']),same(ema.state_dict(),snapshot['ema']),same(rng_state(),snapshot['rng'])]):raise ValueError('GPU_RECOVERY_EXACT_RESTORE')
    file.unlink()
    return {'status':'PASS','arm':arm,'package_sha256':package,'GPU_UUID':os.environ['EXPECTED_GPU_UUID'],'boot_id':host_identity(),
            'admission_sha256':sha(ROOT/'runtime/ADMISSION.json'),'test3':tests,'source_encoder_gradients':source_grad,'actual800_fullcache_window_stress':True,
            'max_main_shape':[2,176,3,352,352],'fm_one_forward_windows':96,'exact_fullSHA_save_destroy_restore':True,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'QA_weights_discarded':True,'at_unix':time.time()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['core6','vm'],required=True)
    modes=ap.add_mutually_exclusive_group(required=True);modes.add_argument('--smoke',action='store_true');modes.add_argument('--fresh',action='store_true');modes.add_argument('--resume',action='store_true');a=ap.parse_args()
    package=pin_check();boot=host_identity();gpu=os.environ['EXPECTED_GPU_UUID']
    if gpu not in GPU_UUIDS:raise ValueError('UNAPPROVED_GPU')
    lease_dir=Path('/workspace/.knee_gpu_leases');lease_dir.mkdir(exist_ok=True)
    lease=open(lease_dir/(gpu+'.lock'),'a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(3)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise ValueError('CUDA_BF16_REQUIRED')
    if str(torch.cuda.get_device_properties(0).uuid).replace('GPU-','').lower()!=gpu.replace('GPU-','').lower():raise ValueError('VISIBLE_GPU_IDENTITY')
    device='cuda:0';config=read(ROOT/'CONFIG.json');admission=read(ROOT/'runtime/ADMISSION.json');inputs=read(ROOT/'runtime/INPUTS.json')
    if admission['status']!='PASS_FULL_CACHE_METADATA_AND_LABEL_ADMISSION' or admission['package_sha256']!=package or admission['boot_id']!=boot or admission['inputs_sha256']!=sha(ROOT/'runtime/INPUTS.json'):raise ValueError('FRESH_FULLCACHE_ADMISSION_REQUIRED')
    inspect_inputs(inputs)
    dataset=MainDataset(inputs,ROOT/f'assets/{a.arm}_targets.npz',config['seed_base'],phys_mm=config['phys_mm'])
    if a.smoke:write(ROOT/'runtime'/f'SMOKE_{a.arm}.json',smoke(a.arm,dataset,inputs,config,device,package));return
    sm=read(ROOT/'runtime'/f'SMOKE_{a.arm}.json');intent=read(ROOT/'runtime/DISPATCH_INTENT.json')
    if sm['status']!='PASS' or sm['package_sha256']!=package or sm['GPU_UUID']!=gpu or sm['boot_id']!=boot or sm['admission_sha256']!=sha(ROOT/'runtime/ADMISSION.json'):raise ValueError('CURRENT_GPU_SMOKE_REQUIRED')
    require_run_intent(ROOT,a.arm,package,boot,resume=a.resume,binding=recovery_binding(a.arm,package,gpu,inputs))
    out=ROOT/'outputs'/a.arm
    if (out/'COMPLETE.json').exists() or (a.fresh and out.exists()):raise ValueError('FRESH_REPLAY_OR_COMPLETE')
    if a.fresh:
        from pair_queue import write_exclusive
        # This durable marker PRECEDES output creation and every optimizer step.
        # Its absence proves that a launcher failed before training began.
        write_exclusive(ROOT/'runtime'/f'STARTED_{a.arm}.json',{'package_sha256':package,'arm':a.arm,'boot_id':boot,'intent_sha256':sha(ROOT/'runtime/DISPATCH_INTENT.json'),'at_unix':time.time()})
    out.mkdir(parents=True,exist_ok=True);model=make_model(a.arm,device,config)
    binding=recovery_binding(a.arm,package,gpu,inputs)
    process=psutil.Process();write(out/'IDENTITY.json',{**binding,'pid':process.pid,'process_create_time':process.create_time(),'argv':process.cmdline(),'Gold_fit':False})
    opt,sched=make_optimizer(model,config,len(dataset)//config['main_batch'])
    final=run(model,opt,sched,dataset,lambda epoch,steps:EpochDataset(inputs['fastMRI'],steps,config['seed_base'],epoch),objective(config),out,binding,config,device,resume=a.resume,augmentor=Augmentor(config['seed_base']))
    names=['epoch16_raw.pt','epoch16_ema.pt','swa_ema05_10.pt']
    write(out/'COMPLETE.json',{'status':'FULLFIT_COMPLETE','epochs':16,'global_step':final['cursor']['global_step'],'arm':a.arm,'package_sha256':package,'files':{n:sha(out/n) for n in names},'swa_source':'EMA','swa_epochs':list(range(5,11)),'at_unix':time.time()})


if __name__=='__main__':main()
