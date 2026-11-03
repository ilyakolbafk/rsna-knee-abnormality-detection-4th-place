"""One selected K24 full-fit; canonical V20 labels/weights and expert Gold pool."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import fullfit_gold as canonical
import matched_core as matched
import train_core_v2_a as core
import train_v83 as base
from token_slot_map import TokenSlotMap

ROOT = Path(__file__).resolve().parent
TASK = ROOT.parent
ROLE = canonical.ROLE


def configure():
    selection = json.loads((TASK / 'SELECTION.json').read_text())
    winner = selection['winner']
    if winner not in ('ordinary', 'transformer'):
        raise ValueError('Unknown selected K24 architecture')
    if winner == 'transformer':
        import context_v83_aug_data as data
        from context_v20_fold3 import ConfidenceContextBags
        from context_head import ContextMIL
        core.V83AugBags = ConfidenceContextBags
        core.make = lambda assets: ContextMIL(assets/'coatnet_imagenet.safetensors', block='transformer', chunk=96, checkpointing=True)
    else:
        import v83_aug_data as data
        core.V83AugBags = matched.ConfidenceBags
    if data.WINDOWS != 24 or core.BATCH != 8 or core.SCHEDULE != 24 or core.STOP != 20 or core.SWA_WINDOW != [18,19,20]:
        raise ValueError('K24 recipe drift')
    core.loader = data.loader
    core.AugConfig = data.AugConfig
    return winner, selection


def prepare(cache):
    # Data-only preparation deliberately performs no GPU or optimizer action.
    matched.verify_freeze(cache)
    configure()
    fit, reference, gold, _ = canonical.frames(cache)
    slots = canonical.verify_slots(cache, fit)
    pos = canonical.weights(reference)
    baseline = json.loads((ROOT/'assets/CANONICAL_FULLFIT_FREEZE.json').read_text())
    if matched.semantic(fit[matched.IDS+matched.MAIN].to_dict(orient='records')) != baseline['target_semantic_sha256']:
        raise ValueError('Canonical full-fit targets changed')
    expected = json.loads((ROOT/'assets/CANONICAL_FULLFIT_POS_WEIGHT.json').read_text())
    if not torch.equal(pos, torch.tensor([expected[t] for t in core.h.TARGETS+canonical.NAMES])):
        raise ValueError('Canonical full-fit weights changed')
    if base.sha(ROOT/'assets/FULLFIT_TOKEN_SLOT_MAP.csv') != baseline['files']['FULLFIT_TOKEN_SLOT_MAP.csv']:
        raise ValueError('Full-fit slot map changed')
    fit[matched.IDS+matched.MAIN].to_csv(ROOT/'assets/FULLFIT_MAIN.csv',index=False)
    reference[matched.IDS+matched.MAIN].to_csv(ROOT/'assets/FULLFIT_R_WEIGHT_REFERENCE.csv',index=False)
    base.atomic_json(ROOT/'assets/POS_WEIGHT_FULLFIT.json',dict(zip(core.h.TARGETS+canonical.NAMES,pos.tolist())))
    freeze={'status':'PASS_SELECTED_K24_FULLFIT_TARGET_FREEZE','samples':4410,'parents':4403,
            'nongold_samples':4352,'Gold58_samples':58,'Gold58_in_fit':True,'Gold_aux_masked':True,
            'target_semantic_sha256':baseline['target_semantic_sha256'],
            'training_sample_ids_sha256':core.h.semantic(sorted(fit.SampleID.astype(str))),
            'canonical_fullfit_source_sha256':base.sha(ROOT/'fullfit_gold.py'),
            'canonical_freeze_sha256':base.sha(ROOT/'assets/CANONICAL_FULLFIT_FREEZE.json'),
            'cache_release_sha256':base.sha(cache/'GLOBAL_RELEASE_RECEIPT.json'),
            'Gold_metric_role':ROLE,'slot_check':slots,'entry_sha256':base.sha(__file__),
            'files':{n:base.sha(ROOT/'assets'/n) for n in ['FULLFIT_MAIN.csv','FULLFIT_R_WEIGHT_REFERENCE.csv','POS_WEIGHT_FULLFIT.json','FULLFIT_TOKEN_SLOT_MAP.csv','GOLD58_EXPERT_MAIN.csv']}}
    path=ROOT/'assets/FULLFIT_TARGET_FREEZE.json'
    if path.exists() and json.loads(path.read_text()) != freeze:
        raise ValueError('Never overwrite a different full-fit freeze')
    base.atomic_json(path,freeze)
    return freeze


def verify(cache):
    freeze=json.loads((ROOT/'assets/FULLFIT_TARGET_FREEZE.json').read_text())
    if freeze['status']!='PASS_SELECTED_K24_FULLFIT_TARGET_FREEZE' or freeze['entry_sha256']!=base.sha(__file__):
        raise ValueError('Unqualified full-fit freeze')
    if freeze['canonical_fullfit_source_sha256']!=base.sha(ROOT/'fullfit_gold.py') or freeze['canonical_freeze_sha256']!=base.sha(ROOT/'assets/CANONICAL_FULLFIT_FREEZE.json'):
        raise ValueError('Canonical full-fit source/freeze changed')
    for name,digest in freeze['files'].items():
        if base.sha(ROOT/'assets'/name)!=digest:raise ValueError('Full-fit frozen source changed: '+name)
    if freeze['cache_release_sha256']!=base.sha(cache/'GLOBAL_RELEASE_RECEIPT.json'):raise ValueError('Cache release changed')
    fit, reference, gold, receipt=canonical.frames(cache)
    pos=canonical.weights(reference)
    values=json.loads((ROOT/'assets/POS_WEIGHT_FULLFIT.json').read_text())
    if not torch.equal(pos,torch.tensor([values[n] for n in core.h.TARGETS+canonical.NAMES],dtype=torch.float32)):
        raise ValueError('Recomputed full-fit weights differ from frozen vector')
    if matched.semantic(fit[matched.IDS+matched.MAIN].to_dict(orient='records'))!=freeze['target_semantic_sha256']:raise ValueError('Fit targets changed')
    return fit,gold,receipt,pos,freeze


def train(cache,out):
    if out.exists():raise ValueError('Fresh full-fit only; no implicit resume')
    winner,selection=configure()
    gate=json.loads((TASK/'START_GATE.json').read_text())
    if gate['status']!='PASS_START_SELECTED_K24_FULLFIT' or gate['entry_sha256']!=base.sha(__file__) or gate['selection_sha256']!=base.sha(TASK/'SELECTION.json'):
        raise ValueError('Selected full-fit start gate mismatch')
    base.guard()
    fit,gold,receipt,pos,freeze=verify(cache)
    core.ARM='CORE_V20_'+winner.upper()+'_K24_FULLFIT_GOLD58'
    core.load_data=lambda cache,names:(fit,gold,receipt)
    core.positive_weights=lambda frame,names:pos.clone()
    old_hashes=core.code_hashes
    core.code_hashes=lambda:{**old_hashes(),'fullfit_k24.py':base.sha(__file__),'fullfit_gold.py':base.sha(ROOT/'fullfit_gold.py'),**({n:base.sha(ROOT/n) for n in ['context_head.py','context_v83_aug_data.py','slot_features.py']} if winner=='transformer' else {})}
    core.IDENTITY_EXTRA={'recipe_variant':'V20_teacherV_'+winner+'_K24_fullfit_gold58','teacher':'V_nonGold_expert_Gold58',
        'architecture':winner,'gold_in_training':True,'gold_fit_or_selection':True,'gold_checkpoint_selection':False,
        'gold_metric_role':ROLE,'validation_each_epoch':False,'Gold58_diagnostic_each_epoch':True,
        'target_freeze_sha256':base.sha(ROOT/'assets/FULLFIT_TARGET_FREEZE.json'),
        'main_targets_sha256':freeze['files']['FULLFIT_MAIN.csv'],'fixed_pos_weight_sha256':freeze['files']['POS_WEIGHT_FULLFIT.json'],
        'selection_sha256':base.sha(TASK/'SELECTION.json'),'Gold58_aux_masked':True,'Gold58_oversampling':False,
        'windows_per_train_sample':24,'parent_recipe':'Canonical V20 fullfit+expertGold; selected architecture; K24/64; B8 OneCycle24 stop20 SWA18-20'}
    original=core.atomic_json
    def role_json(path,value):
        if isinstance(value,dict) and Path(path).name in ['RESULT.json','LIVE.json','EPOCH_AVERAGING.json']:
            value={**value,'gold_in_training':True,'gold_metric_role':ROLE}
            if value.get('phase')=='gold_validation':value['phase']='gold_in_sample_diagnostic'
        elif isinstance(value,list) and Path(path).name=='history.json':value=[{**r,'gold_metric_role':ROLE} for r in value]
        return original(path,value)
    core.atomic_json=role_json
    core.train(SimpleNamespace(cache_root=str(cache),slot_map=str(ROOT/'assets/FULLFIT_TOKEN_SLOT_MAP.csv'),fold=3,full=True,
        series_drop=.10,token_drop=.15,jitter=1,no_params=False,output_dir=str(out),run_name='core_v20_selected_k24_fullfit_gold58',workers=24))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','train']);parser.add_argument('--cache-root',type=Path,required=True);parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    if args.mode=='prepare':print(json.dumps(prepare(args.cache_root)))
    else:
        if args.output_dir is None:raise ValueError('output-dir required')
        train(args.cache_root,args.output_dir)
