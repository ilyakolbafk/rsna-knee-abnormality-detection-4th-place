"""New ordinary fullfit: immutable historical4410 +3VM-only rows; E1imageparams off only."""
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,json,os
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parent;TASK=ROOT.parent;os.environ['V83_PACKAGE_ROOT']=str(ROOT)
import fullfit_gold as canonical
import matched_core as matched
import train_core_v2_a as core
import train_v83 as base
import h1_base as h
import v83_aug_data as data
from token_slot_map import TokenSlotMap
CFG=json.loads((TASK/'QUEUE_CONFIG.json').read_text());ASSETS=ROOT/'assets';MAINFILE=ASSETS/'NOE1_FULLFIT_MAIN_4413.csv';SLOTS=ASSETS/'NOE1_FULLFIT_TOKEN_SLOT_MAP.csv'

def source_gate():
    for name,digest in json.loads((ROOT/'SOURCE_PACKAGE_MANIFEST.json').read_text())['files'].items():assert base.sha(ROOT/name)==digest,name
    assert base.sha(ASSETS/'POS_WEIGHT_FULLFIT.json')=='3d6741a423181312d6fe8aaf7e0d836da944e309e2ee415381da6653e3e17e2f'
    assert len(pd.read_csv(MAINFILE))==4413

def frames():
    source_gate();fit,reference,gold,receipt=canonical.frames(Path(CFG['base_cache']))
    oldfreeze=json.loads((ASSETS/'CANONICAL_FULLFIT_FREEZE.json').read_text())
    assert len(fit)==4410 and matched.semantic(fit[matched.IDS+matched.MAIN].to_dict('records'))==oldfreeze['target_semantic_sha256']
    pos=canonical.weights(reference);pinned=json.loads((ASSETS/'POS_WEIGHT_FULLFIT.json').read_text());names=canonical.NAMES
    assert torch.equal(pos,torch.tensor([pinned[t] for t in h.TARGETS+names],dtype=torch.float32))
    vm=pd.read_csv(ASSETS/'TEST3_VM_MAIN.csv',dtype={'SampleID':str,'StudyInstanceUID':str,'ParentStudyInstanceUID':str,'report_group':str});manifest=pd.read_csv(TASK/'test3_cache/TEST3_MANIFEST.csv',dtype=str).set_index('SampleID');new=[]
    for row in vm.to_dict('records'):
        sid=str(row['SampleID']);assert sid==row['StudyInstanceUID']==row['ParentStudyInstanceUID'] and sid not in set(fit.SampleID.astype(str));record=manifest.loc[sid]
        p=TASK/'test3_cache'/record.cache_file;assert p.stat().st_size==int(record.bytes) and base.sha(p)==record.sha256
        with np.load(p,allow_pickle=False) as z:
            assert str(z['sample_id'])==sid and str(z['split'])=='fullfit_only' and str(z['orientation_state'])=='canonical_L_v8_3'
            assert z['mask'].sum()==64 and z['meta'].shape==(64,6) and z['bank'].shape[1:]==(336,336)
        added={c:np.nan for c in fit.columns};added.update(row);added['cache_path']=str(p);added['fold']=-1
        for n in names:added['aux__'+n]=0.;added['mask__aux__'+n]=0.;added['confidence__aux__'+n]=0.
        for t in h.TARGETS:
            assert added['label_mask__'+t]==added['confidence__'+t]==1
            for prefix in ['reference__']:
                for col in [t,'label_mask__'+t,'confidence__'+t]:
                    if prefix+col in added:added[prefix+col]=np.nan  # No invented R labels for test3.
        new.append(added)
    result=pd.concat([fit,pd.DataFrame(new)],ignore_index=True);assert len(result)==4413 and result.SampleID.is_unique and result.ParentStudyInstanceUID.nunique()==4406
    assert np.array_equal(result.iloc[:4410][matched.IDS].astype(str).to_numpy(),fit[matched.IDS].astype(str).to_numpy())
    assert np.array_equal(result.iloc[:4410][matched.MAIN].to_numpy(np.float32),fit[matched.MAIN].to_numpy(np.float32),equal_nan=True)
    slot=TokenSlotMap(SLOTS)
    for row in result.iloc[-3:].itertuples(index=False):slot.slot_ids(str(row.SampleID),64)
    audit=pd.read_csv(MAINFILE,dtype={'SampleID':str}).set_index('SampleID').loc[result.SampleID.astype(str)]
    assert np.array_equal(audit[matched.MAIN].to_numpy(np.float32),result[matched.MAIN].to_numpy(np.float32),equal_nan=True)
    for p in result.cache_path:assert Path(p).is_file(),p
    return result,gold,receipt,pos,fit

def prepare():
    fit,gold,receipt,pos,old=frames();union=TASK/'cache_union';union.mkdir(exist_ok=True)
    membership=fit[['SampleID','StudyInstanceUID','ParentStudyInstanceUID','cache_path']].copy()
    membership['source_namespace']=['immutable_verified_base4410']*4410+['new_test3_fullfit_only']*3
    membership_path=union/'FIT_MANIFEST.csv';membership_bytes=membership.to_csv(index=False).encode()
    if membership_path.exists():assert membership_path.read_bytes()==membership_bytes
    else:membership_path.write_bytes(membership_bytes)
    from orientation_policy import orientation_identity, require_cache_orientation
    policy=json.loads((ROOT/'ORIENTATION_POLICY.json').read_text())
    meta={'status':'PASS_FULLFIT_UNION_CACHE_4413','samples':4413,'base_samples':4410,'test3_samples':3,'orientation_policy':orientation_identity(policy),'orientation_policy_id':'knee-laterality-v8.3','orientation_state':'canonical_L_v8_3','source_image_size':336,'crop_mm':140,'base_cache':CFG['base_cache'],'base_release_sha256':base.sha(Path(CFG['base_cache'])/'GLOBAL_RELEASE_RECEIPT.json'),'test3_manifest_sha256':base.sha(TASK/'test3_cache/TEST3_MANIFEST.csv'),'test3_build_receipt_sha256':base.sha(TASK/'test3_cache/BUILD_RECEIPT.json'),'canonical_once':True,'old4410_recanonicalized':False}
    meta['union_fit_manifest_sha256']=base.sha(membership_path)
    existing=union/'GLOBAL_RELEASE_RECEIPT.json'
    if existing.exists():assert json.loads(existing.read_text())==meta
    else:base.atomic_json(existing,meta)
    require_cache_orientation(meta,policy)
    freeze={'status':'PASS_NOE1_4413_INPUT_FREEZE','fit_samples':len(fit),'fit_parents':4406,'test3_UIDs':fit.iloc[-3:].SampleID.tolist(),'old4410_semantic_sha256':matched.semantic(old[matched.IDS+matched.MAIN].to_dict('records')),'old4410_main_sha256':base.sha(ASSETS/'FULLFIT_MAIN.csv'),'new_main_sha256':base.sha(MAINFILE),'old_pos_weight_sha256':base.sha(ASSETS/'POS_WEIGHT_FULLFIT.json'),'token_map_sha256':base.sha(SLOTS),'union_cache_release_sha256':base.sha(existing),'test3_build_receipt_sha256':meta['test3_build_receipt_sha256'],'old_label_count_unchanged':4410,'new_VM_main_cells':36,'new_main_weight':1.,'new_aux48_weight':0.,'selected_epochs':[18,19,20],'series_drop':.10,'token_drop':.15,'jitter':1,'params':False,'pos_weight_rule':CFG['pos_weight_rule'],'fullfit_only':True,'entry_sha256':base.sha(__file__)}
    out=TASK/'INPUT_FREEZE.json'
    if out.exists():assert json.loads(out.read_text())==freeze
    else:base.atomic_json(out,freeze)
    return freeze

def train(out):
    assert not out.exists(),'One fresh attempt only, no implicit resume'
    freeze=prepare();gate=json.loads((TASK/'START_GATE.json').read_text())
    assert gate['status']=='PASS_NOE1_4413_START' and gate['entry_sha256']==base.sha(__file__) and gate['input_freeze_sha256']==base.sha(TASK/'INPUT_FREEZE.json') and gate['max_optimizer_training_attempts']==1
    for name,key in [('CPU_QA.json','cpu_qa_sha256'),('CUDA_QA.json','cuda_qa_sha256')]:assert base.sha(TASK/name)==gate[key]
    fit,gold,receipt,pos,old=frames();assert data.WINDOWS==24 and core.BATCH==8 and core.SCHEDULE==24 and core.STOP==20 and core.SWA_WINDOW==[18,19,20]
    core.V83AugBags=matched.ConfidenceBags;core.loader=data.loader;core.AugConfig=data.AugConfig;core.load_data=lambda cache,names:(fit,gold,receipt);core.positive_weights=lambda frame,names:pos.clone();core.ARM='CORE_V20_ORDINARY_K24_FULLFIT_GOLD58_NOE1_TEST3VM'
    prior=core.code_hashes;core.code_hashes=lambda:{**prior(),'noe1_fullfit.py':base.sha(__file__)}
    core.IDENTITY_EXTRA={'recipe_variant':'V20_teacherV_ordinary_K24_fullfit_gold58_noE1_test3VM','teacher':'V_nonGold_expert_Gold58_plus_finalVM_test3','architecture':'ordinary','gold_in_training':True,'gold_fit_or_selection':True,'gold_checkpoint_selection':False,'gold_metric_role':canonical.ROLE,'validation_each_epoch':False,'Gold58_diagnostic_each_epoch':True,'main_targets_sha256':base.sha(MAINFILE),'fixed_pos_weight_sha256':base.sha(ASSETS/'POS_WEIGHT_FULLFIT.json'),'input_freeze_sha256':base.sha(TASK/'INPUT_FREEZE.json'),'public_test3_vm_policy_id':'public-test3-vm-fullfit-v1','public_test3_source_sha256':'452962cd2ff5de179f985222be18e4825eb39f0bac21beee0bff08e8821e8587','test3_fit_only':True,'fit_samples':4413,'fit_parents':4406,'last_batch':5,'training_differences':['E1imageparametersoff','mandatory3VMteststudiesappend'],'Gold58_aux_masked':True,'Gold58_oversampling':False,'augmentation':{'series_parameters':False,'random_resized_crop_area':[1,1],'rotation_deg':[0,0],'shift_fraction':[0,0],'gamma_log_uniform':[1,1],'contrast':[1,1],'bias_field':'off','blur':'off','noise':'off','per_study_gain':'unchangedU(0.9,1.1)insideH1.features','flips':'none(canonical_L_v8_3)'},'pos_weight_rule':CFG['pos_weight_rule']}
    original=core.atomic_json
    def role_json(path,value):
        if isinstance(value,dict) and Path(path).name in ['RESULT.json','LIVE.json','EPOCH_AVERAGING.json']:value={**value,'gold_in_training':True,'gold_metric_role':canonical.ROLE,'fit_samples':4413}
        elif isinstance(value,list) and Path(path).name=='history.json':value=[{**r,'gold_metric_role':canonical.ROLE} for r in value]
        return original(path,value)
    core.atomic_json=role_json
    core.train(SimpleNamespace(cache_root=str(TASK/'cache_union'),slot_map=str(SLOTS),fold=3,full=True,series_drop=.10,token_drop=.15,jitter=1,no_params=True,output_dir=str(out),run_name='core_v20_k24_noe1_test3VM_fullfit',workers=24))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train']);p.add_argument('--output-dir',type=Path);a=p.parse_args()
    print(json.dumps(prepare(),indent=2)) if a.action=='prepare' else train(a.output_dir)
