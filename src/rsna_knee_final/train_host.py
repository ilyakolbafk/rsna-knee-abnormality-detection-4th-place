"""Host entry points for the 12 FINAL ens models (called by train.py).

The historical training packages are kept byte for byte in training/sources (SOURCE_TREES.json); they are what produced
the submitted checkpoints. A run rebuilds one package into <runs>/<model>/package/ (verified file by file), imports it and
calls the package's own trainer with exactly the configuration its historical launcher used. Only launcher code that was
tied to the rented machines is replaced; every replacement is listed here and in training/HOST_PORT.md:

  P1  instance/budget guards (BUDGET_GUARD.json, Vast container label, boot id, paid-hour deadline, GPU UUID lease)
      -> host_guard(): same return keys, no limit.
  P2  start gates and QA receipts written by the historical queue (START_GATE.json, CPU_QA/CUDA_QA.json, VERIFIED.json,
      CONTROL_MANIFEST.json, H6 ADMISSION/SMOKE/DISPATCH_INTENT) -> not required; the data assertions those gates
      protected are re-run here (frame counts, fold/report-group disjointness, Gold58 exclusion, fit-sample hashes).
  P3  absolute paths recorded on the rented machines (cache roots, Joint36 cache_path column) -> CLI / relocation by SampleID.
  P4  only with --cache-identity relaxed: byte identity of the cache *receipts* -> semantic identity against
      training/cache336_reference (same samples, sides, token counts, labels, builder version, orientation state).

Model code, sampling, augmentation, labels, class weights, optimizer, schedule, seeds, numerics flags and SWA are not
touched: those lines run from the staged historical files.
"""
import importlib
import json
import math
import os
import random
import shutil
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from . import PACKAGE_ROOT
from . import sources

HOST_CONFIGS = PACKAGE_ROOT / 'training' / 'host_configs'
CACHE_REFERENCE = PACKAGE_ROOT / 'training' / 'cache336_reference'
HOST_PORT = 'rsna-knee-final-ens train.py host port (P1-P3{})'

W = 'rsna-vm336-k24-folds-fullfit-weights-20261016'
MODELS = {
    # Sixway: ten CoAtNet-rmlp-2-384 checkpoints, all trained by the same train_core_v2_a.py recipe
    'fold0': dict(family='k24_fold', alias='fold01', fold=0, workers=12, run='core_v20_ordinary_k24_cache336_f0', checkpoint=f'{W}/fold0.pt'),
    'fold1': dict(family='k24_fold', alias='fold01', fold=1, workers=12, run='core_v20_ordinary_k24_cache336_f1', checkpoint=f'{W}/fold1.pt'),
    'fold2': dict(family='k24_fold', alias='fold2', fold=2, workers=12, run='core_v20_ordinary_k24_cache336_f2', checkpoint=f'{W}/fold2.pt'),
    'fold3': dict(family='k24_fold', alias='fold3', fold=3, workers=24, run='core_v20_ordinary_k24_f3', checkpoint=f'{W}/fold3.pt'),
    'fold4': dict(family='k24_fold', alias='fold4', fold=4, workers=12, run='core_v20_ordinary_k24_cache336_f4', checkpoint=f'{W}/fold4.pt'),
    'core': dict(family='fullfit', alias='core', workers=24, run='core_v20_selected_k24_fullfit_gold58',
                 checkpoint='rsna-v20-ordinary-k24-fullfit-weights-20261013/checkpoint.pt'),
    'repair': dict(family='fullfit', alias='repair', workers=24, run='core_v20_selected_k24_fullfit_gold58',
                   checkpoint='rsna-vm336-k24-vmrepair-fullfit-weights-20261016/checkpoint.pt'),
    'mirror': dict(family='mirror', alias='mirror', workers=24, run='core_v20_vm336_k24_mirror50_fullfit_gold58',
                   checkpoint='rsna-mirror336-k24-r3-raw-weights-20261016/checkpoint.pt'),
    'adaptive': dict(family='adaptive', alias='adaptive', workers=4, run='core_v20_k24_adaptive336_fullfit_test3VM',
                     checkpoint='rsna-adaptive336-fullfit-weights-20261017/checkpoint.pt'),
    'joint36': dict(family='joint36', alias='joint', workers=8, run='joint36_masked336_fullfit',
                    checkpoint='rsna-j36-masked336-weights-20261016/checkpoint.pt'),
    # H6: ConvNeXt-Small dense-window model, two arms (deployed: SWA of EMA epochs 14-16, tools/average_h6.py)
    'h6_core6': dict(family='h6', alias='h6', arm='core6', checkpoint='rsna-h6-swa1416-weights-20261022/core6_swa_ema14_16.pt'),
    'h6_vm': dict(family='h6', alias='h6', arm='vm', checkpoint='rsna-h6-swa1416-weights-20261022/vm_swa_ema14_16.pt'),
}


# ---------------------------------------------------------------- staging, guards

def stage_package(model, runs_root, weights_dir):
    """<runs>/<model>/package = the historical package, rebuilt and verified byte for byte (training/sources)."""
    spec = MODELS[model]
    task = Path(runs_root).resolve() / model
    task.mkdir(parents=True, exist_ok=True)
    pkg = sources.stage(spec['alias'], task / 'package', weights_dir)
    return spec, pkg, task


def import_package(pkg):
    os.environ['V83_PACKAGE_ROOT'] = str(pkg)
    sys.path[:0] = [str(pkg)]
    importlib.invalidate_caches()


def host_guard():
    """P1: stands in for the paid-instance guards; same keys the trainers copy into IDENTITY.json."""
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
        count = torch.cuda.device_count()
    except Exception:  # pragma: no cover
        gpu, count = 'unknown', 0
    return {'instance_id': 'host:' + socket.gethostname(), 'hourly_usd': 0.0, 'gpu_count': count, 'gpu_name': gpu,
            'hard_stop_unix': time.time() + 10 * 365 * 86400}


def patch_guards(*modules):
    for module in modules:
        if hasattr(module, 'guard'):
            module.guard = host_guard


def host_port(relaxed):
    return HOST_PORT.format(', P4 relaxed cache identity' if relaxed else '')


# ---------------------------------------------------------------- P4: relaxed cache identity

def compare_with_reference(cache):
    """Semantic comparison of an assembled cache336 release with the historical one (training/cache336_reference)."""
    import numpy as np
    import pandas as pd
    cache = Path(cache)
    report = {}
    for name in ['TRAIN_MANIFEST.csv', 'GOLD58_MANIFEST.csv']:
        new = pd.read_csv(cache / name, dtype=str).set_index('SampleID')
        ref = pd.read_csv(CACHE_REFERENCE / name, dtype=str).set_index('SampleID')
        if set(new.index) != set(ref.index):
            raise ValueError(f'{name}: sample set differs from the historical cache')
        new = new.loc[ref.index]
        for column in ['StudyInstanceUID', 'side', 'tokens', 'padded_tokens']:
            if column in ref and not (new[column] == ref[column]).all():
                raise ValueError(f'{name}: column {column} differs from the historical cache')
        same = {c: float((new[c] == ref[c]).mean()) for c in ['pixels_sha256', 'mask_sha256', 'meta_sha256'] if c in ref and c in new}
        report[name] = {'samples': len(ref), 'identical_array_fraction': same}
    for name in ['TRAIN_LABELS.csv', 'GOLD58_LABELS.csv']:
        new = pd.read_csv(cache / name).sort_values(list(pd.read_csv(cache / name, nrows=0).columns[:1])).reset_index(drop=True)
        ref = pd.read_csv(CACHE_REFERENCE / name).sort_values(list(pd.read_csv(CACHE_REFERENCE / name, nrows=0).columns[:1])).reset_index(drop=True)
        if list(new.columns) != list(ref.columns) or len(new) != len(ref):
            raise ValueError(f'{name}: layout differs from the historical cache')
        num = ref.select_dtypes('number').columns
        if not np.allclose(new[num].to_numpy(float), ref[num].to_numpy(float), equal_nan=True, atol=1e-6):
            raise ValueError(f'{name}: label values differ from the historical cache')
        other = [c for c in ref.columns if c not in num]
        if not (new[other].astype(str).to_numpy() == ref[other].astype(str).to_numpy()).all():
            raise ValueError(f'{name}: identifiers differ from the historical cache')
        report[name] = {'rows': len(ref), 'equal': True}
    return report


def relax_cache_identity(matched):
    """Replace byte checks of the cache *receipts* (they contain timings and paths) with semantic checks."""
    base = matched.base
    root = Path(matched.ROOT)

    def verify_cache(cache):
        cache = Path(cache)
        for name, digest in json.loads((root / 'SOURCE_PACKAGE_MANIFEST.json').read_text())['files'].items():
            assert matched.sha(root / name) == digest, name
        receipt = json.loads((cache / 'GLOBAL_RELEASE_RECEIPT.json').read_text())
        provenance = json.loads((root / 'PREPARATION_PROVENANCE.json').read_text())
        assert receipt['status'] == 'PASS_GLOBAL_AUX60_V83_RELEASE', receipt.get('status')
        assert receipt['builder_sha256'] == provenance['builder_sha256'], 'cache built by a different builder version'
        roles = {x['role'] for x in receipt['components']}
        assert roles == set(json.loads((root / 'FROZEN_COMPONENT_SHA256.json').read_text())), roles
        for role in roles:
            paths = list((cache / 'downloads' / role).rglob('CACHE_COMPLETE.json'))
            assert len(paths) == 1, role
            assert json.loads(paths[0].read_text())['asset_manifest_sha256'] == provenance['asset_manifest_sha256'], role
        compare_with_reference(cache)
        metadata = {'orientation_policy': provenance['orientation_policy'], 'orientation_state': receipt['orientation_state'],
                    'cache_release_sha256': base.sha(cache / 'GLOBAL_RELEASE_RECEIPT.json')}
        matched.require_cache_orientation(metadata, json.loads((root / 'ORIENTATION_POLICY.json').read_text()))
        return metadata

    def verify_freeze(cache):
        freeze = json.loads((root / 'assets/TARGET_FREEZE.json').read_text())
        assert freeze['status'] == 'PASS_MATCHED_TARGET_FREEZE'
        verify_cache(cache)  # includes the label comparison that replaces the TRAIN_LABELS.csv byte check
        for name, digest in freeze['files'].items():
            assert matched.sha(root / 'assets' / name) == digest, name
        for fold, digest in freeze['weights'].items():
            assert matched.sha(root / 'assets' / f'POS_WEIGHT_f{fold}.json') == digest
        return freeze

    matched.verify_cache = verify_cache
    matched.verify_freeze = verify_freeze


def _check_cache_root(cache):
    cache = Path(cache or '')
    missing = [n for n in ['GLOBAL_RELEASE_RECEIPT.json', 'TRAIN_MANIFEST.csv', 'TRAIN_LABELS.csv', 'GOLD58_MANIFEST.csv', 'GOLD58_LABELS.csv']
               if not (cache / n).is_file()]
    if missing:
        raise FileNotFoundError(f'--cache-root {cache} is not an assembled cache336 release (missing {missing}); see data_prep/cache336/README.md')
    return cache.resolve()


def _k24_args(pkg, cache, out, fold, full, run, workers, slot_map='assets/TOKEN_SLOT_MAP.csv'):
    return SimpleNamespace(cache_root=str(cache), slot_map=str(pkg / slot_map), fold=fold, full=full, series_drop=.10,
                           token_drop=.15, jitter=1, no_params=False, output_dir=str(out), run_name=run, workers=workers)


# ---------------------------------------------------------------- K24 folds (ordinary_k24_fold.py, ordinary_v20_fold3.py)

def fold_frames(pkg, cache, matched, core, data):
    """The data assertions of the historical fold launchers (their frames()/cpu() checks)."""
    names = json.loads((pkg / 'assets/aux_names.json').read_text())
    assert len(names) == 48
    frame, gold, receipt = matched.frame_for(cache, names, 'V')
    assert len(frame) == 4352 and frame.SampleID.nunique() == 4352 and frame.ParentStudyInstanceUID.nunique() == 4345
    assert len(gold) == 58 and not set(frame.ParentStudyInstanceUID) & set(gold.ParentStudyInstanceUID)
    assert not set(frame.report_group) & set(gold.report_group)
    assert frame.groupby('report_group').fold.nunique().max() == 1 and frame.groupby('ParentStudyInstanceUID').fold.nunique().max() == 1
    assert not (gold[['mask__aux__' + n for n in names]] > 0).any().any()
    assert data.WINDOWS == 24 and core.BATCH == 8 and core.STOP == 20 and core.SCHEDULE == 24 and core.SWA_WINDOW == [18, 19, 20]
    return names, frame, gold


def check_fold_recipe(fold, frame, core, matched):
    recipes = json.loads((HOST_CONFIGS / 'fold_recipes.json').read_text())
    if str(fold) not in recipes:
        return None
    expected = recipes[str(fold)]
    fit = frame.loc[frame.fold != fold].reset_index(drop=True)
    val = frame.loc[frame.fold == fold].reset_index(drop=True)
    assert len(fit) == expected['fit'] and len(val) == expected['val'], (len(fit), len(val))
    assert core.h.semantic(sorted(fit.SampleID.astype(str))) == expected['fit_sample_ids_sha256']
    if 'validation_sample_ids_sha256' in expected:
        assert core.h.semantic(sorted(val.SampleID.astype(str))) == expected['validation_sample_ids_sha256']
    for key in ['SampleID', 'ParentStudyInstanceUID', 'report_group']:
        assert not set(fit[key]) & set(val[key])
    return expected


def train_k24_fold(spec, pkg, task, cache, out, workers, relaxed):
    import_package(pkg)
    fold = spec['fold']
    import train_core_v2_a as core
    import train_v83 as base
    import matched_core as matched
    import v83_aug_data as data
    from token_slot_map import TokenSlotMap
    patch_guards(base, core)
    if relaxed:
        relax_cache_identity(matched)
    names, frame, gold = fold_frames(pkg, cache, matched, core, data)
    check_fold_recipe(fold, frame, core, matched)
    slots = TokenSlotMap(pkg / 'assets/TOKEN_SLOT_MAP.csv')
    assert len(set(slots.counts) | slots.unmapped) == 4352
    if fold == 3:
        wrapper = importlib.import_module('ordinary_v20_fold3')
        matched.verify_freeze(cache)
        wrapper.configure_core(Path(cache))  # the historical configuration function, unchanged
    else:  # body of ordinary_k24_fold.configure(fold); receipt hashes of the rented machine replaced by host markers
        wrapper = importlib.import_module('ordinary_k24_fold')
        core.ARM = f'CORE_V20_ORDINARY_K24_CACHE336_F{fold}'
        core.V83AugBags = matched.ConfidenceBags
        core.loader = data.loader
        core.load_data = lambda root, n: matched.frame_for(Path(root), n, 'V')
        core.evaluate_fold = matched.original_reference_eval
        core.positive_weights = lambda fit, n: matched.fixed_weights(fold, n)
        original = core.code_hashes
        core.code_hashes = lambda: {**original(), 'ordinary_k24_fold.py': base.sha(wrapper.__file__), 'matched_core.py': base.sha(pkg / 'matched_core.py')}
        core.IDENTITY_EXTRA = {
            'recipe_variant': f'ordinary_teacherV_K24_cache336_fold{fold}', 'teacher': 'V', 'source_image_size': 336, 'crop_mm': 140,
            'crop_center': 'geometric', 'architecture': 'ordinary Core V20 H1AuxAug without Transformer', 'gold_in_training': False,
            'gold_checkpoint_selection': False, 'gold_metric_role': 'held_out_reused_development_diagnostic',
            'main_targets_sha256': base.sha(pkg / 'assets/V_MAIN.csv'), 'original_reference_sha256': base.sha(pkg / 'assets/FROZEN_REFERENCE.csv'),
            'fixed_pos_weight_sha256': base.sha(pkg / 'assets' / f'POS_WEIGHT_f{fold}.json'),
            'cache_actual_verification_sha256': 'host-port:not-applicable', 'source_package_manifest_sha256': base.sha(pkg / 'SOURCE_PACKAGE_MANIFEST.json'),
            'orientation_preflight_sha256': 'host-port:not-applicable'}
    core.IDENTITY_EXTRA = {**core.IDENTITY_EXTRA, 'host_port': host_port(relaxed)}
    core.train(_k24_args(pkg, cache, out, fold, False, spec['run'], workers))


# ---------------------------------------------------------------- full fits (fullfit_k24.py, mirror_fullfit.py)

def _fullfit_verify(ff, canonical, matched, base, cache, relaxed):
    """fullfit_k24.verify(cache); with --cache-identity relaxed the cache-receipt byte check is the only line skipped."""
    import torch
    if not relaxed:
        return ff.verify(Path(cache))
    root = Path(ff.ROOT)
    freeze = json.loads((root / 'assets/FULLFIT_TARGET_FREEZE.json').read_text())
    if freeze['status'] != 'PASS_SELECTED_K24_FULLFIT_TARGET_FREEZE' or freeze['entry_sha256'] != base.sha(ff.__file__):
        raise ValueError('Unqualified full-fit freeze')
    if freeze['canonical_fullfit_source_sha256'] != base.sha(root / 'fullfit_gold.py') or freeze['canonical_freeze_sha256'] != base.sha(root / 'assets/CANONICAL_FULLFIT_FREEZE.json'):
        raise ValueError('Canonical full-fit source/freeze changed')
    for name, digest in freeze['files'].items():
        if base.sha(root / 'assets' / name) != digest:
            raise ValueError('Full-fit frozen source changed: ' + name)
    fit, reference, gold, receipt = canonical.frames(Path(cache))
    pos = canonical.weights(reference)
    values = json.loads((root / 'assets/POS_WEIGHT_FULLFIT.json').read_text())
    if not torch.equal(pos, torch.tensor([values[n] for n in ff.core.h.TARGETS + canonical.NAMES], dtype=torch.float32)):
        raise ValueError('Recomputed full-fit weights differ from frozen vector')
    if matched.semantic(fit[matched.IDS + matched.MAIN].to_dict(orient='records')) != freeze['target_semantic_sha256']:
        raise ValueError('Fit targets changed')
    return fit, gold, receipt, pos, freeze


def train_fullfit(spec, pkg, task, cache, out, workers, relaxed, before_train=None, check_only=False):
    """Core and VM-repair full fits: all 4352 knees (VM or repaired VM labels) + Gold58 with expert labels."""
    import_package(pkg)
    (task / 'SELECTION.json').write_text(json.dumps({'winner': 'ordinary', 'source': 'host replay of the selected K24 recipe (FINAL ens uses the ordinary architecture)'}, indent=2) + '\n')
    ff = importlib.import_module('fullfit_k24')
    import fullfit_gold as canonical
    import matched_core as matched
    import train_core_v2_a as core
    import train_v83 as base
    patch_guards(base, core)
    if relaxed:
        relax_cache_identity(matched)
    winner, _ = ff.configure()
    assert winner == 'ordinary'
    fit, gold, receipt, pos, freeze = _fullfit_verify(ff, canonical, matched, base, cache, relaxed)
    assert len(fit) == 4410 and fit.SampleID.nunique() == 4410 and fit.ParentStudyInstanceUID.nunique() == 4403 and len(gold) == 58
    if check_only:
        return fit, gold, pos
    # body of fullfit_k24.train() after its start gate, unchanged
    core.ARM = 'CORE_V20_' + winner.upper() + '_K24_FULLFIT_GOLD58'
    core.load_data = lambda c, n: (fit, gold, receipt)
    core.positive_weights = lambda frame, n: pos.clone()
    old_hashes = core.code_hashes
    core.code_hashes = lambda: {**old_hashes(), 'fullfit_k24.py': base.sha(ff.__file__), 'fullfit_gold.py': base.sha(Path(ff.ROOT) / 'fullfit_gold.py')}
    core.IDENTITY_EXTRA = {
        'recipe_variant': 'V20_teacherV_' + winner + '_K24_fullfit_gold58', 'teacher': 'V_nonGold_expert_Gold58',
        'architecture': winner, 'gold_in_training': True, 'gold_fit_or_selection': True, 'gold_checkpoint_selection': False,
        'gold_metric_role': ff.ROLE, 'validation_each_epoch': False, 'Gold58_diagnostic_each_epoch': True,
        'target_freeze_sha256': base.sha(Path(ff.ROOT) / 'assets/FULLFIT_TARGET_FREEZE.json'),
        'main_targets_sha256': freeze['files']['FULLFIT_MAIN.csv'], 'fixed_pos_weight_sha256': freeze['files']['POS_WEIGHT_FULLFIT.json'],
        'selection_sha256': base.sha(task / 'SELECTION.json'), 'Gold58_aux_masked': True, 'Gold58_oversampling': False,
        'windows_per_train_sample': 24,
        'parent_recipe': 'Canonical V20 fullfit+expertGold; selected architecture; K24/64; B8 OneCycle24 stop20 SWA18-20',
        'host_port': host_port(relaxed)}
    original = core.atomic_json

    def role_json(path, value):
        if isinstance(value, dict) and Path(path).name in ['RESULT.json', 'LIVE.json', 'EPOCH_AVERAGING.json']:
            value = {**value, 'gold_in_training': True, 'gold_metric_role': ff.ROLE}
            if value.get('phase') == 'gold_validation':
                value['phase'] = 'gold_in_sample_diagnostic'
        elif isinstance(value, list) and Path(path).name == 'history.json':
            value = [{**r, 'gold_metric_role': ff.ROLE} for r in value]
        return original(path, value)
    core.atomic_json = role_json
    if before_train:
        before_train(core, base, cache)
    core.train(_k24_args(pkg, cache, out, 3, True, spec['run'], workers, slot_map='assets/FULLFIT_TOKEN_SLOT_MAP.csv'))


def train_mirror(spec, pkg, task, cache, out, workers, relaxed, check_only=False):
    """Mirror50 full fit = the full-fit recipe + whole-knee mirror augmentation p=0.5 (mirror-aware v83_aug_data.py)."""
    def before(core, base, cache_root):
        original = core.train

        def run(args):  # identical to the wrapper inside mirror_fullfit.train()
            args.run_name = spec['run']
            core.ARM = 'CORE_V20_VM336_K24_MIRROR50_FULLFIT_GOLD58'
            extra = dict(core.IDENTITY_EXTRA)
            extra.update(augmentation=json.loads((pkg / 'assets/MIRROR_AUGMENTATION.json').read_text()), recipe_variant='VM336_K24_Mirror50_fullfit_gold58',
                         source_image_size=336, model_image_size=384, crop_mm=140, mirror_probability=.5,
                         mirror_scope='one whole-knee draw per SampleID epoch after gather; sagittal triplet reversal and direction signs; coronal/axial width reflection',
                         canonical_cache_orientation='canonical_L_v8_3', mirror_eval=False,
                         source_package_manifest_sha256=base.sha(pkg / 'SOURCE_PACKAGE_MANIFEST.json'),
                         mirror_entry_sha256=base.sha(pkg / 'mirror_fullfit.py'), mirror_transform_sha256=base.sha(pkg / 'mirror_tokens.py'),
                         cache_actual_verification_sha256='host-port:not-applicable',
                         baseline_averaged_sha256='9b6eab4a07699ae622f7d210ac7ffcabf592dba016511bd6abaf98eb34b04adb')
            core.IDENTITY_EXTRA = extra
            old = core.code_hashes
            core.code_hashes = lambda: {**old(), 'mirror_fullfit.py': base.sha(pkg / 'mirror_fullfit.py'), 'mirror_tokens.py': base.sha(pkg / 'mirror_tokens.py')}
            return original(args)
        core.train = run
    return train_fullfit(spec, pkg, task, cache, out, workers, relaxed, before_train=before, check_only=check_only)


# ---------------------------------------------------------------- Adaptive336 (adaptive_fullfit.py)

def write_adaptive_task(task, cache, extension_root, test3_cache):
    config = json.loads((HOST_CONFIGS / 'adaptive_QUEUE_CONFIG.historical.json').read_text())
    config.update(base_cache=str(Path(cache).resolve()), extension_root=str(Path(extension_root).resolve()),
                  host_port='historical QUEUE_CONFIG; base_cache/extension_root re-pointed; budget/binding unused (P1)')
    (task / 'QUEUE_CONFIG.json').write_text(json.dumps(config, indent=2) + '\n')
    for name in ['EXTENSION_SERIALIZATION_INVENTORY.json', 'TEST3_EXTENSION_QA.json']:  # provenance records hashed into INPUT_FREEZE.json
        shutil.copy(HOST_CONFIGS / f'adaptive_{name}', task / name)
    link = task / 'test3_cache'
    if not link.exists():
        os.symlink(Path(test3_cache).resolve(), link, target_is_directory=True)


def train_adaptive(spec, pkg, task, cache, out, workers, relaxed, extension_root, test3_cache, check_only=False):
    """Adaptive336 full fit: 24 base + up to 24 extension windows; Gold58 + the 3 public sample studies (VM labels)."""
    if not extension_root or not test3_cache:
        raise ValueError('adaptive needs --extension-root and --test3-cache')
    write_adaptive_task(task, cache, extension_root, test3_cache)
    import_package(pkg)
    af = importlib.import_module('adaptive_fullfit')
    import train_core_v2_a as core
    import train_v83 as base
    import matched_core as matched
    af.budget_guard = host_guard
    patch_guards(base, core)
    if relaxed:
        relax_cache_identity(matched)
    freeze = af.prepare()  # writes <task>/cache_union and INPUT_FREEZE.json exactly as historically
    if check_only:
        return freeze
    af.configure()
    base.guard = core.guard = host_guard  # configure() re-binds the guards to budget_guard
    core.IDENTITY_EXTRA = {**core.IDENTITY_EXTRA, 'host_port': host_port(relaxed)}
    assert core.BATCH == 8 and core.SCHEDULE == 24 and core.STOP == 20 and core.SWA_WINDOW == [18, 19, 20]
    core.train(SimpleNamespace(cache_root=str(task / 'cache_union'), slot_map=str(af.inputs.SLOTS), fold=3, full=True, series_drop=.1,
                               token_drop=.15, jitter=1, no_params=False, output_dir=str(out), run_name=spec['run'], workers=workers))
    history = json.loads((Path(out) / 'history.json').read_text())
    assert [r['updates'] for r in history] == [552 * i for i in range(1, 21)]


# ---------------------------------------------------------------- Joint36 (fullfit_masked.py)

def joint36_relocation(cache, test3_cache):
    import pandas as pd
    locations = {}
    for manifest in [Path(cache) / 'TRAIN_MANIFEST.csv', Path(cache) / 'GOLD58_MANIFEST.csv']:
        m = pd.read_csv(manifest, dtype=str)
        locations.update({sid: str(Path(cache) / rel) for sid, rel in zip(m.SampleID, m.cache_file)})
    m = pd.read_csv(Path(test3_cache) / 'TEST3_MANIFEST.csv', dtype=str)
    locations.update({sid: str(Path(test3_cache) / rel) for sid, rel in zip(m.SampleID, m.cache_file)})
    return locations


def train_joint36(spec, pkg, task, cache, out, workers, relaxed, test3_cache, check_only=False):
    """Joint36 full fit: three 12-finding heads (VM, Evidence, Core6) with source masks, branch weights .45/.45/.10."""
    import numpy as np
    import torch
    from datetime import datetime, timezone
    if not test3_cache:
        raise ValueError('joint36 needs --test3-cache')
    import_package(pkg)
    j = importlib.import_module('fullfit_masked')
    t, data = j.t, j.data
    j.guard = host_guard
    locations = joint36_relocation(cache, test3_cache)  # P3: frozen frames store absolute paths of the rented machine
    read_frame = t.read_frame

    def relocated(path):
        frame = read_frame(path)
        if 'cache_path' in frame:
            frame['cache_path'] = frame.SampleID.map(locations)
            assert frame.cache_path.notna().all(), 'cache does not cover every Joint36 sample'
        return frame
    t.read_frame = relocated
    j.verify()
    f, v, e, c = j.frames(True)
    if check_only:
        return f
    # main() numerics flags and the body of fullfit_masked.train() after its QA-receipt checks, unchanged
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    t.h.seed_all(t.SEED)
    m = t.network(j.ARM).cuda()
    opt, sched = t.models.optimizer(m, 24, math.ceil(len(f) / 8))
    pw = t.weights(j.ARM, torch.device('cuda'))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    identity = {**j.contract(), 'code_sha256': j.H(j.__file__), 'package_sha256': j.H(j.K / 'PACKAGE_MANIFEST.json'),
                'instance_id': host_guard()['instance_id'], 'quality_role': j.ROLE, 'host_port': host_port(relaxed)}
    j.atomic(out / 'IDENTITY.json', identity)
    f[j.IDS].to_csv(out / 'fit_ids.csv', index=False)
    history = []
    updates = 0
    for ei in range(20):
        ep = ei + 1
        begin = time.monotonic()
        m.train()
        total = ml = al = 0.
        n = 0
        dl = data.loader(j.bags(f, v, e, True, ei), 8, workers, t.SEED + ei, True)
        for step, (x, mask, meta, y, w, index, gain, params) in enumerate(dl):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                z = m(x.cuda(), mask.cuda(), meta.cuda(), gain.cuda(), params.cuda())
                main, aux, ls = j.losses(z, y.cuda(), w.cuda(), pw)
                loss = ls.sum()
            assert torch.isfinite(loss)
            loss.backward()
            for parameter in m.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(len(x))
            torch.nn.utils.clip_grad_norm_(m.parameters(), 3., error_if_nonfinite=True)
            opt.step()
            sched.step()
            updates += 1
            n += len(x)
            total += float(loss.detach())
            ml += float(main.sum().detach())
            al += float(aux.sum().detach())
            if step % 25 == 0:
                j.atomic(out / 'LIVE.json', {'phase': 'train', 'epoch': ep, 'step': step + 1, 'steps': len(dl), 'updates': updates,
                                             'samples': n, 'loss': total / n, 'utc': datetime.now(timezone.utc).isoformat()})
        torch.cuda.synchronize()
        record = {'epoch': ep, 'updates': updates, 'train_loss': total / n, 'main_loss': ml / n, 'aux_loss': al / n, 'seconds': time.monotonic() - begin}
        history.append(record)
        state = {k: val.detach().cpu().clone() for k, val in m.state_dict().items()}
        t.h.atomic_torch(out / f'weights-e{ep:02d}.pt', {'model': state, 'identity': identity, 'epoch': ep})
        t.h.atomic_torch(out / 'latest.pt', {'model': state, 'identity': identity, 'epoch': ep, 'optimizer': opt.state_dict(),
                                             'scheduler': sched.state_dict(), 'history': history, 'updates': updates,
                                             'rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                                             'python_rng': random.getstate(), 'numpy_rng': np.random.get_state()})
        j.atomic(out / 'history.json', history)
        print(json.dumps(record), flush=True)
    states = [t.h.load_checkpoint(out / f'weights-e{ep:02d}.pt')['model'] for ep in [18, 19, 20]]
    averaged = {k: (sum(s[k].double() for s in states) / 3).to(val.dtype) if val.is_floating_point() else val.clone() for k, val in states[0].items()}
    m.load_state_dict(averaged)
    t.h.atomic_torch(out / 'averaged.pt', {'model': averaged, 'identity': identity, 'epochs': [18, 19, 20]})
    metrics = j.evaluate(m, f, v, e, out, 'swa18_20', workers)
    j.atomic(out / 'RESULT.json', {'status': 'COMPLETE', 'epochs': 20, 'updates': updates, 'selected_epochs': [18, 19, 20],
                                   'averaged_sha256': j.H(out / 'averaged.pt'), 'evaluation': metrics, 'role': j.ROLE,
                                   'code_sha256': j.H(j.__file__), 'package_sha256': j.H(j.K / 'PACKAGE_MANIFEST.json')})


# ---------------------------------------------------------------- H6 (train.py of the H6 package)

def train_h6(spec, pkg, task, out, inputs_json, smoke=False, check_only=False):
    """One H6 arm: the original 16-epoch loop with its EMA; SWA of EMA epochs 14-16 afterwards (tools/average_h6.py).
    Replaced: hostname/Vast label/boot-id identity, GPU-UUID lease, admission/smoke/dispatch receipts (P1, P2)."""
    if not inputs_json:
        raise ValueError('H6 needs --h6-inputs (INPUTS.json for the RSNA/OAI/fastMRI v4 caches; see data_prep/h6_v4/README.md)')
    import_package(pkg)
    (pkg / 'runtime').mkdir(exist_ok=True)
    shutil.copy(inputs_json, pkg / 'runtime' / 'INPUTS.json')
    import common
    import torch
    tr = importlib.import_module('train')
    config = common.read(pkg / 'CONFIG.json')
    inputs = common.read(pkg / 'runtime' / 'INPUTS.json')
    arm = spec['arm']
    tr.check_targets(arm)
    dataset = tr.MainDataset(inputs, pkg / f'assets/{arm}_targets.npz', config['seed_base'], phys_mm=config['phys_mm'])
    if check_only:
        return dataset
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('H6 training needs a CUDA GPU with bf16 (historically one RTX 5090 32 GB per arm)')
    torch.set_num_threads(3)
    device = 'cuda:0'
    package = common.sha(pkg / 'PACKAGE_PINS.json')
    binding = {'package_sha256': package, 'arm': arm, 'GPU_UUID': 'host', 'target': common.TARGET, 'config_sha256': common.sha(pkg / 'CONFIG.json'),
               'inputs_sha256': common.sha(pkg / 'runtime/INPUTS.json'), 'targets_sha256': common.sha(pkg / f'assets/{arm}_targets.npz'),
               'cache_acceptance': inputs['acceptance'], 'recipe_schema': tr.RECIPE_SCHEMA, 'orientation_policy': common.POLICY}
    model = tr.make_model(arm, device, config)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    common.write(out / 'IDENTITY.json', {**binding, 'Gold_fit': False, 'host_port': 'rsna-knee-final-ens train.py (P1, P2)'})
    opt, sched = tr.make_optimizer(model, config, len(dataset) // config['main_batch'])
    final = tr.run(model, opt, sched, dataset, lambda epoch, steps: tr.EpochDataset(inputs['fastMRI'], steps, config['seed_base'], epoch),
                   tr.objective(config), out, binding, config, device, augmentor=tr.Augmentor(config['seed_base']),
                   stop_after=2 if smoke else None)
    common.write(out / 'COMPLETE.json', {'status': 'SMOKE_2_STEPS' if smoke else 'FULLFIT_COMPLETE', 'arm': arm,
                                         'global_step': final['cursor']['global_step'] if isinstance(final, dict) and 'cursor' in final else None})


# ---------------------------------------------------------------- dispatcher

def run(model, runs, weights, cache=None, relaxed=False, extension_root=None, test3_cache=None, h6_inputs=None, workers=None,
        smoke=False, check_only=False):
    spec, pkg, task = stage_package(model, runs, weights)
    out = task / 'training'
    if out.exists() and not check_only:
        raise FileExistsError(f'{out} exists: one fresh attempt per run directory')
    workers = workers or spec.get('workers', 8)
    info = {'model': model, 'package': spec['alias'], 'historical_path': sources.manifest()['trees'][spec['alias']]['historical_path'],
            'deployed_checkpoint': spec['checkpoint'], 'cache_identity': 'relaxed' if relaxed else 'original', 'workers': workers,
            'started_unix': time.time(), 'check_only': check_only}
    (task / 'RUN_INFO.json').write_text(json.dumps(info, indent=2) + '\n')
    family = spec['family']
    if family != 'h6':
        cache = _check_cache_root(cache)
    if family == 'k24_fold':
        if check_only:
            import_package(pkg)
            import train_core_v2_a as core
            import train_v83 as base
            import matched_core as matched
            import v83_aug_data as data
            patch_guards(base, core)
            if relaxed:
                relax_cache_identity(matched)
            names, frame, gold = fold_frames(pkg, cache, matched, core, data)
            check_fold_recipe(spec['fold'], frame, core, matched)
        else:
            train_k24_fold(spec, pkg, task, cache, out, workers, relaxed)
    elif family == 'fullfit':
        train_fullfit(spec, pkg, task, cache, out, workers, relaxed, check_only=check_only)
    elif family == 'mirror':
        import_package(pkg)
        importlib.import_module('mirror_fullfit')  # loads the mirror-aware v83_aug_data.py of this package
        train_mirror(spec, pkg, task, cache, out, workers, relaxed, check_only=check_only)
    elif family == 'adaptive':
        train_adaptive(spec, pkg, task, cache, out, workers, relaxed, extension_root, test3_cache, check_only=check_only)
    elif family == 'joint36':
        train_joint36(spec, pkg, task, cache, out, workers, relaxed, test3_cache, check_only=check_only)
    elif family == 'h6':
        train_h6(spec, pkg, task, out, h6_inputs, smoke=smoke, check_only=check_only)
    info.update(finished_unix=time.time(), status='CHECKED' if check_only else 'TRAINED', output=str(out))
    (task / 'RUN_INFO.json').write_text(json.dumps(info, indent=2) + '\n')
    return info
