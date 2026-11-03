"""Actual accepted-cache test3/loader check on the target, before GPU smoke."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','4')
import time
import torch
from common import ROOT,read,write,sha,pin_check
from admission import host_identity,inspect_inputs
from main_data import MainDataset


def main():
    package=pin_check();boot=host_identity();inputs=read(ROOT/'runtime/INPUTS.json')
    inspect_inputs(inputs);config=read(ROOT/'CONFIG.json');torch.set_num_threads(4)
    expected=read(ROOT/'assets/TEST3_EXPECTED.json');checks=[]
    for arm in ['core6','vm']:
        dataset=MainDataset(inputs,ROOT/f'assets/{arm}_targets.npz',config['seed_base'],phys_mm=config['phys_mm'])
        # Dataset order is independently bound to target IDs, not necessarily the same order.
        indices={i for i,row in enumerate(dataset.rows) if row['id'] in expected}
        indices.update(next(i for i,row in enumerate(dataset.rows) if row['source_id']==s) for s in [0,1])
        for i in sorted(indices):
            b=dataset[i];uid=b['id'];n=int(b['mask'].sum())
            if not 0<n<=176 or b['x'].shape!=(176,3,384,384) or not torch.isfinite(b['meta']).all():raise ValueError('ACTUAL_FULLCACHE_LOADER')
            if uid in expected:
                if not torch.equal(b['main_probability'],torch.tensor(expected[uid],dtype=torch.float32)) or not bool((b['main_weight']==1).all()):raise ValueError('ACTUAL_TEST3_VM36')
            if arm=='vm' and int(b['source_id'])==1 and bool(b['main_weight'].any()):raise ValueError('OAI_IN_RSNA_VM_HEAD')
            checks.append({'arm':arm,'id':uid,'actual_unique_windows':n,'source_id':int(b['source_id'])})
        del dataset
    write(ROOT/'runtime/CPU_QA.json',{'status':'PASS_CURRENT_TARGET_ACTUAL_FULLCACHE_CPU_TEST3','package_sha256':package,'boot_id':boot,
        'admission_sha256':sha(ROOT/'runtime/ADMISSION.json'),'checks':checks,'GPU_QA':False,'at_unix':time.time()})


if __name__=='__main__':main()
