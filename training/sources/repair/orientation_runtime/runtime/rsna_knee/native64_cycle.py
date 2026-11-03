"""Adaptive Native64 family queue; budget and verified backups before expansion."""
from __future__ import annotations
import argparse
from datetime import datetime
import fcntl
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import pandas as pd
import yaml
from .screen_queue import verify_completion
from .training_state import file_sha256, save_json

ARMS = ('canonical_meta', 'canonical_only', 'native')


def next_action(scores, baseline):
    """Missing screen is work to do, never interpreted as a failed candidate."""
    if not math.isfinite(baseline):
        raise ValueError('Finite frozen baseline required')
    for arm in ARMS[:2]:
        if arm not in scores:
            return {'screen': arm, 'fold': 3}
        if not math.isfinite(scores[arm]):
            raise ValueError('Nonfinite screen score is a failed run, not a quality rejection')
        if scores[arm] >= baseline:
            return {'selected': arm}
    return {'selected': 'native'}


def budget_check(contract, required_seconds, now=None):
    now = time.time() if now is None else now
    start = datetime.fromisoformat(contract['rental_started_at_utc']).timestamp()
    deadline = datetime.fromisoformat(contract['stop_deadline_utc']).timestamp()
    rate, reserve, cap = (float(contract[k]) for k in ('hourly_all_in_usd','fixed_reserve_usd','budget_usd'))
    if not all(math.isfinite(x) for x in (rate,reserve,cap,required_seconds,now)):
        raise ValueError('Nonfinite budget input')
    if not 0 < cap <= 10 or rate <= 0 or reserve < 0 or required_seconds < 0 or not start <= now < deadline:
        raise ValueError('Invalid or expired budget contract')
    if (deadline-start)*rate/3600+reserve > cap:
        raise ValueError('Hard rental bound exceeds total user budget')
    if now + required_seconds + 900 > deadline:
        raise ValueError('Insufficient time/cost reserve for the next complete run and backup')


def run(contract_path):
    path=Path(contract_path);c=json.loads(path.read_text());contract_sha=file_sha256(path)
    root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'cycle.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if c.get('num_gpus') != 1 or not c.get('cuda_preflight_passed'):
        raise ValueError('One verified GPU and measured CUDA preflight required')
    for name,digest in c['input_files'].items():
        if file_sha256(name)!=digest:raise ValueError('Frozen cycle input changed: '+name)
    frame=pd.read_csv(c['manifest'],dtype={'StudyInstanceUID':str})
    if len(frame)!=4407 or frame.StudyInstanceUID.duplicated().any() or set(frame.fold)!={0,1,2,3,4}:
        raise ValueError('Unexpected official fold population')
    scores={};completed=[]
    def status(state,**kwargs):save_json(root/'CYCLE_STATUS.json',dict(state=state,contract_sha256=contract_sha,at_utc=datetime.now().astimezone().isoformat(),**kwargs))
    def verify(folder,fold,identity_sha):
        s=json.loads((folder/'summary.json').read_text())
        n=int(s['completed_epochs'])
        if n not in (14,16) or s['selection_metric']!='pseudo_macro_auc' or s['selected_weights']!='ema':
            raise ValueError('Unexpected fold selection protocol')
        if n==14 and s.get('termination_reason') not in ('predeclared_plateau_stop','user_authorized_early_stop'):
            raise ValueError('Unexplained shortened run')
        marker=verify_completion(folder,fold=fold,epochs=n,expected_ids=set(frame.loc[frame.fold.eq(fold),'StudyInstanceUID']),expected_identity_sha256=identity_sha)
        return s,marker
    def one(arm,fold):
        rid=f'native64-{arm.replace("_","-")}-f{fold}-s20260904';folder=root/rid
        identity_sha=c['identities'][arm][str(fold)]
        cfg=yaml.safe_load(Path(c['configs'][arm]).read_text())
        expected_flags={'canonical_meta':(True,True,True),'canonical_only':(True,False,False),'native':(False,False,False)}[arm]
        flags=cfg['data'].get('anatomy_features',{})
        if tuple(bool(flags.get(k,False)) for k in ('canonical_side','position','fat_suppression'))!=expected_flags:
            raise ValueError('Arm input flags differ from cycle contract')
        if cfg['train']['epochs']!=16 or cfg['seed']!=20260904 or cfg['train']['batch_size']*cfg['train']['accumulate_steps']!=8:
            raise ValueError('Cycle recipe differs from approved training horizon/effective batch')
        if not (folder/'COMPLETED.json').exists():
            budget_check(c,c['forecast_seconds_per_fold'])
            from .train import build_fold_partitions
            from .training_state import make_training_identity
            import torch
            tr,va=build_fold_partitions(frame,fold=fold,n_splits=5,validation_scope='all_holdout')
            if make_training_identity(cfg,tr,va,None,fold=fold,device=torch.device('cuda'))['sha256']!=identity_sha:
                raise ValueError('Current runtime differs from frozen training identity')
            status('training',arm=arm,fold=fold,run_id=rid)
            args=[sys.executable,'-m','rsna_knee.cli','train','--config',c['configs'][arm],
                  '--manifest-csv',c['manifest'],'--fold',str(fold),'--output-dir',str(folder),'--device','cuda']
            if (folder/'last_training.pt').exists():args += ['--resume',str(folder/'last_training.pt')]
            deadline=datetime.fromisoformat(c['stop_deadline_utc']).timestamp()
            folder.mkdir(parents=True,exist_ok=True)
            with (folder/'training.log').open('ab') as log:
                subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=min(c['forecast_seconds_per_fold']*1.5,deadline-time.time()-900))
        s,marker=verify(folder,fold,identity_sha)
        epoch=json.loads((folder/'EPOCH_COMMITTED.json').read_text())
        if epoch['next_epoch']!=s['completed_epochs'] or epoch['training_identity_sha256']!=identity_sha:
            raise ValueError('Completed fold has no matching resumable generation')
        expected=dict(contract_sha256=contract_sha,run_id=rid,training_identity_sha256=identity_sha,completion_sha256=file_sha256(folder/'COMPLETED.json'),
                      recovery_marker_sha256=file_sha256(folder/'EPOCH_COMMITTED.json'),recovery_checkpoint_sha256=epoch['checkpoint']['sha256'],next_epoch=epoch['next_epoch'])
        status('awaiting_verified_backup',**{k:v for k,v in expected.items() if k!='contract_sha256'})
        ack=root/'backup_acks'/(rid+'.json');end=time.time()+900
        while not ack.exists():
            budget_check(c,0)
            if time.time()>end:raise TimeoutError('Local backup acknowledgement missing')
            time.sleep(10)
        receipt=json.loads(ack.read_text())
        if any(receipt.get(k)!=v for k,v in expected.items()):raise ValueError('Wrong backup acknowledgement')
        completed.append(dict(arm=arm,fold=fold,run_id=rid,score=s['best_selection_score']))
        return float(s['best_selection_score'])
    try:
        while True:
            action=next_action(scores,float(c['baseline_score']))
            if 'selected' in action:break
            arm=action['screen'];scores[arm]=one(arm,3)
            save_json(root/'SCREEN_DECISION.json',dict(scores=scores,baseline=c['baseline_score'],next=next_action(scores,c['baseline_score'])))
        selected=action['selected']
        if selected=='native':
            s,_=verify(Path(c['native_fold3']),3,c['native_fold3_identity_sha256'])
            if s['best_selection_score']!=c['baseline_score'] or file_sha256(Path(c['native_fold3'])/'best.pt')!=c['native_fold3_checkpoint_sha256']:
                raise ValueError('Native fallback reference differs')
        for fold in (0,1,2,4):one(selected,fold)
        paths=[(Path(c['native_fold3']) if selected=='native' and fold==3 else root/f'native64-{selected.replace("_","-")}-f{fold}-s20260904') for fold in range(5)]
        oof=pd.concat([pd.read_csv(p/'oof_predictions.csv') for p in paths],ignore_index=True)
        if oof.StudyInstanceUID.duplicated().any() or set(oof.StudyInstanceUID)!=set(frame.StudyInstanceUID):raise ValueError('Incomplete family OOF')
        oof.sort_values('StudyInstanceUID').to_csv(root/'family_oof.csv',index=False)
        save_json(root/'FAMILY_READY.json',dict(selected=selected,contract_sha256=contract_sha,
            folds=[dict(fold=i,path=str(p),best_sha256=file_sha256(p/'best.pt')) for i,p in enumerate(paths)],
            oof_sha256=file_sha256(root/'family_oof.csv'),screens=scores,publication='Await exact verified inference bundle and one private Kaggle runtime before family LB'))
        status('family_saved_ready_for_kaggle',selected=selected,completed=completed)
    except Exception as exc:
        status('failed_stop_required',error_type=type(exc).__name__,error=str(exc));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--contract',required=True);a=p.parse_args();run(a.contract)
