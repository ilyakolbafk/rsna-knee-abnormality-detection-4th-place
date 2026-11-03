"""One fresh adaptive336 fullfit4413; original v3 model/sampler/E1 and fixed SWA18–20."""
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,json,os
ROOT=Path(__file__).resolve().parent;TASK=ROOT.parent
os.environ['V83_PACKAGE_ROOT']=str(ROOT)
import numpy as np,pandas as pd,torch
import noe1_fullfit as inputs
import train_core_v2_a as core
import train_v83 as base
import h1_base as h
from adaptive_inputs import dataset_factory,loader
from adaptive_model import make
from fullfit_extensions import FullfitExtensionIndex
from fullfit_budget import budget_guard
from v83_aug_data import AugConfig
CFG=json.loads((TASK/'QUEUE_CONFIG.json').read_text())
RUN='core_v20_k24_adaptive336_fullfit_test3VM'
def sha(p):return base.sha(Path(p))
def verify_manifest():
    m=json.loads((TASK/'CONTROL_MANIFEST.json').read_text())
    for n,pin in m['files'].items():assert sha(TASK/n)==pin,n
    return m
def frames():return inputs.frames()
def prepare():
    fit,gold,receipt,pos,old=frames()
    index=FullfitExtensionIndex(CFG['extension_root'])
    assert set(index.records)==set(fit.SampleID.astype(str))
    union=TASK/'cache_union';union.mkdir(exist_ok=True)
    membership=fit[['SampleID','StudyInstanceUID','ParentStudyInstanceUID','cache_path']].to_csv(index=False).encode()
    p=union/'FIT_MANIFEST.csv'
    if p.exists():assert p.read_bytes()==membership
    else:p.write_bytes(membership)
    from orientation_policy import orientation_identity,require_cache_orientation
    policy=json.loads((ROOT/'ORIENTATION_POLICY.json').read_text())
    meta={'status':'PASS_ADAPTIVE_FULLFIT_UNION4413','samples':4413,'orientation_policy':orientation_identity(policy),'orientation_policy_id':'knee-laterality-v8.3','orientation_state':'canonical_L_v8_3','source_image_size':336,'crop_mm':140,'base_cache':CFG['base_cache'],'base_release_sha256':sha(Path(CFG['base_cache'])/'GLOBAL_RELEASE_RECEIPT.json'),'test3_manifest_sha256':sha(TASK/'test3_cache/TEST3_MANIFEST.csv'),'canonical_once':True,'old4410_recanonicalized':False,'union_fit_manifest_sha256':sha(p)}
    require_cache_orientation(meta,policy)
    p=union/'GLOBAL_RELEASE_RECEIPT.json'
    if p.exists():assert json.loads(p.read_text())==meta
    else:base.atomic_json(p,meta)
    freeze={'status':'PASS_ADAPTIVE_4413_INPUT_FREEZE','fit_samples':4413,'fit_parents':4406,'main_targets_sha256':sha(inputs.MAINFILE),'fixed_pos_weight_sha256':sha(ROOT/'assets/POS_WEIGHT_FULLFIT.json'),'token_map_sha256':sha(inputs.SLOTS),'test3_manifest_sha256':sha(TASK/'test3_cache/TEST3_MANIFEST.csv'),'extension_serialization_inventory_sha256':sha(TASK/'EXTENSION_SERIALIZATION_INVENTORY.json'),'test3_extension_QA_sha256':sha(TASK/'TEST3_EXTENSION_QA.json'),'entry_sha256':sha(__file__),'package_manifest_sha256':sha(ROOT/'SOURCE_PACKAGE_MANIFEST.json'),'metadata_width':8,'params':True,'series_drop':.1,'token_drop':.15,'jitter':1,'gain':[.9,1.1],'selected_epochs':[18,19,20],'new_VM_main_cells':36,'new_aux_weight':0,'Gold58_in_fit':True,'Gold58_role':inputs.canonical.ROLE,'extension_roster':4413,'fullfit_only':True,'union_release_sha256':sha(p)}
    p=TASK/'INPUT_FREEZE.json'
    if p.exists():assert json.loads(p.read_text())==freeze
    else:base.atomic_json(p,freeze)
    return freeze
def configure(cache=None,ext=None,variant='N0'):
    base.guard=budget_guard;core.guard=budget_guard
    core.ARM='CORE_V20_ADAPTIVE336_FULLFIT_GOLD58_TEST3VM'
    core.V83AugBags=dataset_factory(FullfitExtensionIndex(ext or CFG['extension_root']),variant)
    core.loader=loader;core.AugConfig=AugConfig;core.make=make
    fit,gold,receipt,pos,old=frames()
    core.load_data=lambda cache,names:(fit,gold,receipt)
    core.positive_weights=lambda frame,names:pos.clone()
    # Checkpoint storage only. The same trainer/optimizer and fixed3-state SWA are unchanged.
    core.KEEP_WEIGHTS=[18,19,20]
    original=h.atomic_torch
    def artifacts(path,value):
        if Path(path).name in ['latest.pt','main12.pt']:return
        return original(path,value)
    h.atomic_torch=artifacts
    hashes=core.code_hashes
    core.code_hashes=lambda:{**hashes(),**{n:sha(ROOT/n) for n in ['adaptive_fullfit.py','adaptive_model.py','adaptive_inputs.py','allseries_inputs.py','fullfit_extensions.py','fullfit_budget.py']}}
    core.IDENTITY_EXTRA={'recipe_variant':'V20_teacherV_adaptive336_fullfit_gold58_test3VM','teacher':'V_nonGold_expert_Gold58_plus_finalVM_test3','architecture':'adaptive24plus24','metadata_width':8,'gold_in_training':True,'gold_fit_or_selection':True,'gold_checkpoint_selection':False,'gold_metric_role':inputs.canonical.ROLE,'validation_each_epoch':False,'Gold58_diagnostic_each_epoch':True,'main_targets_sha256':sha(inputs.MAINFILE),'fixed_pos_weight_sha256':sha(ROOT/'assets/POS_WEIGHT_FULLFIT.json'),'input_freeze_sha256':sha(TASK/'INPUT_FREEZE.json'),'public_test3_source_sha256':'452962cd2ff5de179f985222be18e4825eb39f0bac21beee0bff08e8821e8587','test3_fit_only':True,'fit_samples':4413,'fit_parents':4406,'last_batch':5,'Gold58_aux_masked':True,'Gold58_oversampling':False,'windows_per_train_sample':{'base_before_token_drop':24,'extra_fraction':.375,'extra_cap':24,'max_before_drop':48},'extra_jitter':0,'base_and_extra_gain':'one unchanged base getter gain','validation_tokens':'all valid base up to64 + extension up to108','augmentation':{'series_parameters':True,'source_sha256':sha(ROOT/'aug336.py'),'base_draws_unchanged':True,'extra_per_series_params':True,'flips':'none(canonical_L_v8_3)'},'checkpoint_policy':CFG['checkpoint_policy'],'extension_serialization':'ZIP_BZIP2 lossless exact inner .npy bytes; new namespace; original arrays unchanged'}
    original_json=core.atomic_json
    def role(path,value):
        if isinstance(value,dict) and Path(path).name in ['RESULT.json','LIVE.json','EPOCH_AVERAGING.json']:
            value={**value,'gold_in_training':True,'gold_metric_role':inputs.canonical.ROLE,'fit_samples':4413}
            if value.get('phase')=='gold_validation':value['phase']='gold_in_sample_diagnostic'
        elif isinstance(value,list) and Path(path).name=='history.json':value=[{**r,'gold_metric_role':inputs.canonical.ROLE} for r in value]
        return original_json(path,value)
    core.atomic_json=role
    return fit,gold,receipt,pos
def frame_rosters(cache=None):
    fit,gold,_,_,_=frames()
    return inputs.canonical.NAMES,fit,fit.iloc[:0].copy(),gold
def train(out):
    assert not out.exists(),'One fresh attempt only; no resume/retry'
    verify_manifest();freeze=prepare();budget_guard()
    gate=json.loads((TASK/'START_GATE.json').read_text())
    assert gate['status']=='PASS_ADAPTIVE_4413_START' and gate['entry_sha256']==sha(__file__) and gate['input_freeze_sha256']==sha(TASK/'INPUT_FREEZE.json') and gate['max_optimizer_training_attempts']==1
    assert gate['control_manifest_sha256']==sha(TASK/'CONTROL_MANIFEST.json')
    for n,key in [('CPU_QA.json','cpu_qa_sha256'),('CUDA_QA.json','cuda_qa_sha256')]:assert sha(TASK/n)==gate[key]
    configure();assert core.BATCH==8 and core.SCHEDULE==24 and core.STOP==20 and core.SWA_WINDOW==[18,19,20]
    core.train(SimpleNamespace(cache_root=str(TASK/'cache_union'),slot_map=str(inputs.SLOTS),fold=3,full=True,series_drop=.1,token_drop=.15,jitter=1,no_params=False,output_dir=str(out),run_name=RUN,workers=4))
    hist=json.loads((out/'history.json').read_text());assert [r['updates'] for r in hist]==[552*i for i in range(1,21)]
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train']);p.add_argument('--output-dir',type=Path);a=p.parse_args()
    print(json.dumps(prepare(),indent=2)) if a.action=='prepare' else train(a.output_dir)
