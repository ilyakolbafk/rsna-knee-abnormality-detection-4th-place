"""Bind a current-boot admission/smoke to an immutable original intent and slot."""
import argparse,os,time,fcntl
from pathlib import Path
import psutil
from common import ROOT,TARGET,read,write,sha,pin_check
from admission import host_identity,inspect_inputs
from recovery_state import RecoveryStore
from train import recovery_binding


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['core6','vm'],required=True);a=ap.parse_args()
    package=pin_check();boot=host_identity();inputs=read(ROOT/'runtime/INPUTS.json');inspect_inputs(inputs)
    intent=read(ROOT/'runtime/DISPATCH_INTENT.json');admission=read(ROOT/'runtime/ADMISSION.json');smoke=read(ROOT/'runtime'/f'SMOKE_{a.arm}.json')
    if intent['package_sha256']!=package or intent['target']!=TARGET or intent['arms']!=['core6','vm']:raise ValueError('ORIGINAL_PAIR_INTENT_REQUIRED')
    if admission['package_sha256']!=package or admission['boot_id']!=boot or admission['inputs_sha256']!=sha(ROOT/'runtime/INPUTS.json'):raise ValueError('CURRENT_BOOT_ADMISSION_REQUIRED')
    if smoke['status']!='PASS' or smoke['package_sha256']!=package or smoke['boot_id']!=boot or smoke['admission_sha256']!=sha(ROOT/'runtime/ADMISSION.json'):raise ValueError('CURRENT_BOOT_GPU_SMOKE_REQUIRED')
    out=ROOT/'outputs'/a.arm;identity=read(out/'IDENTITY.json')
    if (out/'COMPLETE.json').exists():raise ValueError('ALREADY_COMPLETE')
    try:
        p=psutil.Process(identity['pid'])
        if abs(p.create_time()-identity['process_create_time'])<.02:raise ValueError('FIT_STILL_ALIVE')
    except psutil.NoSuchProcess:pass
    gpu=identity['GPU_UUID']
    if smoke['GPU_UUID']!=gpu:raise ValueError('RESUME_GPU_BINDING')
    lease=open(Path('/workspace/.knee_gpu_leases')/(gpu+'.lock'),'a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    binding=recovery_binding(a.arm,package,gpu,inputs);state,entry=RecoveryStore(out/'recovery',binding).load()
    path=out/'recovery'/f"slot{entry['slot']}.manifest.json"
    write(ROOT/'runtime'/f'RESUME_AUTHORIZATION_{a.arm}.json',{'status':'PASS_FULLSHA_RESUME_AUTHORIZATION','arm':a.arm,'package_sha256':package,
        'original_intent_sha256':sha(ROOT/'runtime/DISPATCH_INTENT.json'),'new_admission_sha256':sha(ROOT/'runtime/ADMISSION.json'),
        'new_smoke_sha256':sha(ROOT/'runtime'/f'SMOKE_{a.arm}.json'),'boot_id':boot,'recovery_manifest':path.name,
        'recovery_manifest_sha256':sha(path),'slot_sha256':entry['sha256'],'cursor':entry['cursor'],'at_unix':time.time()})


if __name__=='__main__':main()
