"""Fullfit-only16-epoch loop with optimizer-boundary raw+EMA recovery.

The same function is exercised by CPU worker/fault parity tests and GPU training.
"""
from pathlib import Path
import math,time,random,hashlib
import numpy as np
import torch
from recovery_state import (RecoveryStore,SCHEMA,freeze,rng_state,restore_rng,cursor,validate_cursor,
                            IN_EPOCH,BETWEEN_EPOCHS,atomic_json)
from ema_state import EMA,swa_update
from clipping import fm_clip

RECIPE_SCHEMA='h6_full_v4_fullfit_ema_v1'

def atomic_torch(path,state):
    import os
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.partial.'+str(os.getpid()))
    with temporary.open('wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)
    fd=os.open(str(path.parent),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

def same(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b

def output_once(path,state):
    if path.exists():assert same(torch.load(path,map_location='cpu',weights_only=False),state),'EPOCH_OUTPUT_CONFLICT'
    else:atomic_torch(path,state)

def loader(dataset,batch,indices,workers,seed):
    kw={'num_workers':workers,'pin_memory':torch.cuda.is_available(),'generator':torch.Generator().manual_seed(seed)}
    if workers:kw.update(prefetch_factor=2,persistent_workers=False)
    return torch.utils.data.DataLoader(dataset,batch_size=batch,sampler=indices,**kw)

def run(model,optimizer,scheduler,dataset,fm_factory,loss_step,out,binding,config,device,
        *,resume=False,augmentor=None,fault=None,stop_after=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    steps=len(dataset)//config['main_batch'] if config.get('drop_last',True) else math.ceil(len(dataset)/config['main_batch']);epochs=config['epochs'];swa_epochs=config['swa_epochs']
    assert epochs==16 and swa_epochs==list(range(5,11))
    store=RecoveryStore(out/'recovery',binding);ema=EMA(model);swa=None;swa_n=0
    position=cursor(1,0,0,BETWEEN_EPOCHS);seen=[0,0];meter={};elapsed=0.;start=time.monotonic();saved_rng=None
    if resume:
        state,entry=store.load()
        assert state.get('recipe_schema')==RECIPE_SCHEMA and state['binding']==binding
        validate_cursor(state['cursor'],steps,epochs)
        assert state['ema']['updates']==state['cursor']['global_step']
        assert state['swa_n']==sum(e<state['cursor']['epoch'] for e in swa_epochs)
        model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        ema.load_state_dict(state['ema'],model);swa=state['swa'];swa_n=state['swa_n']
        position=state['cursor'];seen=state['seen'];meter=state['meter'];elapsed=state['elapsed_seconds']
        if augmentor is not None:augmentor.load_state_dict(state['augmentation_rng'])
        saved_rng=state['rng'];restore_rng(saved_rng)
    else:assert not store.exists() and not (out/'epoch01_raw.pt').exists(),'FRESH_OUTPUT_EXISTS'

    def hook(stage):
        if fault is not None:fault(stage,position)

    def capture():
        if str(device).startswith('cuda'):torch.cuda.synchronize()
        assert ema.updates==position['global_step']
        assert swa_n==sum(e<position['epoch'] for e in swa_epochs)
        validate_cursor(position,steps,epochs)
        state={'schema':SCHEMA,'recipe_schema':RECIPE_SCHEMA,'binding':binding,'cursor':dict(position),
               'model':freeze(model.state_dict()),'optimizer':freeze(optimizer.state_dict()),'scheduler':freeze(scheduler.state_dict()),
               'ema':ema.state_dict(),'swa':freeze(swa),'swa_n':swa_n,'seen':list(seen),'meter':dict(meter),
               'rng':rng_state(),'augmentation_rng':augmentor.state_dict() if augmentor is not None else None,
               'elapsed_seconds':elapsed+time.monotonic()-start}
        store.save(state)
        return state

    while position['epoch']<=epochs:
        epoch=position['epoch'];done=position['step'];dataset.epoch=epoch
        ids=np.random.default_rng(config['seed_base']+epoch).permutation(len(dataset)).tolist()
        if config.get('drop_last',True):ids=ids[:steps*config['main_batch']]
        order_digest=hashlib.sha256(np.asarray(ids,np.int64).tobytes()).hexdigest()
        fm_dataset=fm_factory(epoch,steps)
        main_loader=loader(dataset,config['main_batch'],[(i,(epoch-1)*len(dataset)+j) for j,i in enumerate(ids)][done*config['main_batch']:],config['workers'],6000+epoch)
        fm_loader=loader(fm_dataset,None,list(range(done,steps)),config['workers'],7000+epoch)
        main_it,fm_it=iter(main_loader),iter(fm_loader)
        if saved_rng is not None:restore_rng(saved_rng);saved_rng=None
        model.train()
        for step,batch in enumerate(main_it,done+1):
            if (out.parent.parent/'STOP_TRAINING').exists():raise RuntimeError('USER_STOP_TRAINING')
            if str(device).startswith('cuda') and __import__('shutil').disk_usage(out).free<config['disk_floor_bytes']:raise RuntimeError('TRAINING_DISK_FLOOR')
            batch={k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
            fm_batch=next(fm_it);optimizer.zero_grad(set_to_none=True)
            if augmentor is not None:batch=augmentor(batch,epoch,step)
            values=loss_step(model,batch,fm_batch,epoch,step,device)
            norm=fm_clip(model,config['gradient_clip'])
            assert bool(torch.isfinite(norm)), 'NONFINITE_MERGED_GRADIENT'
            optimizer.step();hook('after_optimizer_before_EMA');scheduler.step();ema.update(model,scheduler.last_epoch)
            hook('after_EMA_before_commit')
            position=cursor(epoch,step,(epoch-1)*steps+step,IN_EPOCH)
            for k in range(2):seen[k]+=int((batch['source_id']==k).sum())
            for key,value in values.items():meter[key]=meter.get(key,0.)+float(value)
            if position['global_step']==1 or position['global_step']%config['recovery_interval']==0 or step==steps:
                capture();hook('after_step_commit')
            if step%10==0 or step==steps or config.get('qa_progress'):
                atomic_json(out/'PROGRESS.json',{'state':'TRAINING','arm':binding['arm'],'epoch':epoch,'step':step,
                    'steps_per_epoch':steps,'global_step':position['global_step'],'total_steps':steps*epochs,'losses':values,
                    'gradient_norm':float(norm),'main_seen':seen,'order_sha256':order_digest,'fm_specs_sha256':getattr(fm_dataset,'digest',None),
                    'elapsed_seconds':elapsed+time.monotonic()-start,'at_unix':time.time()})
            if stop_after==position['global_step']:return capture()
        assert position['step']==steps and sum(seen)==(steps*config['main_batch'] if config.get('drop_last',True) else len(dataset))
        raw=freeze(model.state_dict());ema_model=ema.full_model_state(model)
        if epoch in swa_epochs:swa,swa_n=swa_update(swa,swa_n,ema_model)
        hook('after_SWA_before_outputs')
        for kind,weights in [('raw',raw),('ema',ema_model)]:
            output_once(out/f'epoch{epoch:02d}_{kind}.pt',{'model':weights,'epoch':epoch,'kind':kind,'arm':binding['arm'],'package_sha256':binding['package_sha256'],'binding':binding})
            hook('after_'+kind+'_output')
        atomic_json(out/f'epoch{epoch:02d}_metrics.json',{'epoch':epoch,'steps':steps,'losses':{k:v/steps for k,v in meter.items()},'seen':seen})
        hook('after_metrics')
        position=cursor(epoch+1,0,epoch*steps,BETWEEN_EPOCHS);seen=[0,0];meter={}
        state=capture();hook('after_between_commit')
    assert swa_n==6 and ema.updates==steps*epochs
    output_once(out/'swa_ema05_10.pt',{'model':swa,'epochs':swa_epochs,'averaging_source':'EMA','arm':binding['arm'],'package_sha256':binding['package_sha256'],'binding':binding})
    final=capture()
    return final
