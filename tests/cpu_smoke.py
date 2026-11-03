#!/usr/bin/env python3
"""CPU smoke test of the training entry points (no GPU, no MRI pixels needed).

1. Every stored historical source file matches SOURCE_TREES.json.
2. A synthetic cache336 release is built from the *historical* cache metadata (training/cache336_reference: the exact
   GLOBAL_RELEASE_RECEIPT.json, manifests, labels and per-shard CACHE_COMPLETE.json) plus placeholder NPZ files with the
   historical token counts, so the strict receipt checks run unchanged (--cache-identity original).
3. For every CoAtNet model: stage the package, run the historical launcher data checks (frame counts, fold/report-group
   disjointness, fit-sample hashes, frozen target and class-weight checks, Adaptive INPUT_FREEZE, Joint36 frames).
4. One real optimizer step of the historical CoAtNet model and loss on CPU (fold0 package, batch 2, K24 windows) and
   one forward pass of the Joint36 network; H6: frozen target checks and model construction from the H6 package.

Usage: python tests/cpu_smoke.py --weights WEIGHTS --test3-cache WEIGHTS/training_inputs/test3_cache [--work DIR]
(WEIGHTS needs pretrained/ for steps 4; steps 1-3 need only the repository and the test3 cache.)
"""
import argparse
import importlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))


def synthetic_cache(dst):
    import numpy as np
    import pandas as pd
    ref = REPO / 'training' / 'cache336_reference'
    if dst.exists():
        return dst
    shutil.copytree(ref, dst)
    rng = np.random.default_rng(0)
    for name in ['TRAIN_MANIFEST.csv', 'GOLD58_MANIFEST.csv']:
        m = pd.read_csv(dst / name, dtype=str)
        for row in m.itertuples(index=False):
            tokens = int(row.tokens)
            bank = rng.integers(0, 255, (4, 336, 336), dtype=np.uint8)
            refs = np.zeros((64, 3), np.int16)
            refs[:tokens] = rng.integers(0, 4, (tokens, 3))
            mask = np.zeros(64, bool)
            mask[:tokens] = True
            meta = np.zeros((64, 6), np.float32)
            meta[:tokens] = rng.normal(size=(tokens, 6)).astype(np.float32) * .1
            path = dst / row.cache_file
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, format=np.asarray('native64-token-bank-v1'), bank=bank, refs=refs, mask=mask, meta=meta,
                     sample_id=np.asarray(row.SampleID), study_uid=np.asarray(row.StudyInstanceUID), side=np.asarray(row.side),
                     split=np.asarray(row.namespace), orientation_policy=np.asarray('knee-laterality-v8.3'),
                     orientation_state=np.asarray('canonical_L_v8_3'))
    return dst


def check_model(model, cache, weights, runs, test3, extension):
    cmd = [sys.executable, str(REPO / 'train.py'), '--model', model, '--check-data', '--cache-root', str(cache), '--weights', str(weights),
           '--runs', str(runs), '--test3-cache', str(test3)]
    if extension:
        cmd += ['--extension-root', str(extension)]
    began = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {'model': model, 'rc': proc.returncode, 'seconds': round(time.time() - began, 1),
            'tail': (proc.stdout + proc.stderr)[-1500:] if proc.returncode else ''}


def optimizer_step(weights, runs):
    """fold0 package: real ConfidenceBags batch from the synthetic cache -> CoAtNet forward/backward/AdamW step on CPU."""
    import torch
    from rsna_knee_final import train_host
    spec, pkg, task = train_host.stage_package('fold0', runs / 'step', weights)
    train_host.import_package(pkg)
    import train_core_v2_a as core
    import train_v83 as base
    import matched_core as matched
    import v83_aug_data as data
    from token_slot_map import TokenSlotMap
    train_host.patch_guards(base, core)
    names = json.loads((pkg / 'assets/aux_names.json').read_text())
    frame, gold, _ = matched.frame_for(runs.parents[1] / 'cache336_synthetic', names, 'V')
    fit = frame.loc[frame.fold != 0].reset_index(drop=True)
    ds = matched.ConfidenceBags(fit, names, train=True, seed=base.SEED, epoch=0, slot_map=TokenSlotMap(pkg / 'assets/TOKEN_SLOT_MAP.csv'),
                                config=data.AugConfig(series_drop=.1, token_drop=.15, jitter=1, params=True))
    x, mask, meta, y, w, index, gain, params = data.collate([ds[i] for i in range(2)])
    # CPU memory: keep one knee and 6 of its windows for the optimizer step (the GPU recipe uses 8 x 24)
    x, mask, meta, params = x[:1, :6], mask[:1, :6], meta[:1, :6], params[:1, :6]
    y, w, gain = y[:1], w[:1], gain[:1]
    core.h.seed_all(base.SEED)
    model = core.make(pkg / 'assets').train()
    opt, sched = core.optimizer(model, core.SCHEDULE, 436, False)
    pw = matched.fixed_weights(0, names)
    began = time.time()
    logits = model(x, mask, meta, gain, params)
    main, aux, per = core.loss_parts(logits, y, w, pw, aux_weight=core.AUX_WEIGHT)
    loss = per.sum()
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            p.grad.div_(len(x))
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0, error_if_nonfinite=True)
    opt.step()
    sched.step()
    return {'status': 'PASS' if bool(torch.isfinite(loss)) else 'FAIL', 'collated_batch': '2 knees x 24 windows (sliced to 1 x 6 for CPU memory)',
            'batch_shape': list(x.shape), 'logits': list(logits.shape),
            'loss': float(loss.detach()), 'grad_norm': float(norm), 'seconds': round(time.time() - began, 1),
            'params_million': round(sum(p.numel() for p in model.parameters()) / 1e6, 1)}


def _h6_targets(tr, pkg, arm):
    """Frozen-target check. The released target files hold the RSNA rows only (OAI rows are not distributed, see
    training/H6_RECIPE.md), so the full-roster check of the historical package stops with FROZEN_FIT_ROSTER; then the
    RSNA rows are checked with the same shape, range and head-routing rules."""
    try:
        return tr.check_targets(arm)
    except ValueError as e:
        if str(e) != 'FROZEN_FIT_ROSTER':
            raise
    import numpy as np
    with np.load(pkg / f'assets/{arm}_targets.npz', allow_pickle=False) as z:
        t = {k: z[k] for k in z.files}
    n = len(t['sample_id'])
    assert n == 4355 and len(set(map(str, t['sample_id']))) == n and (t['source_id'] == 0).all(), 'RSNA_ROWS'
    for key in ['main', 'oai_vm']:
        p, w, lr = [t[key + '_' + k] for k in ['probability', 'weight', 'lr']]
        assert p.shape == (n, 12) and np.isfinite(p).all() and np.isfinite(w).all() and np.isfinite(lr).all(), 'TARGET_SHAPE_FINITE'
        assert not (np.any(p < 0) or np.any(p > 1) or np.any(w < 0) or np.any(lr <= 0)), 'TARGET_RANGE'
    assert not np.any(t['oai_vm_weight']), 'OAI_LABEL_IN_RSNA_HEAD'
    return {'arm': arm, 'rows': n, 'scope': 'RSNA rows only (OAI rows not distributed)'}


def h6_construct(weights, runs):
    """H6 package: frozen target checks for both arms and construction of the H6 network from the bundled ImageNet weights."""
    from rsna_knee_final import train_host
    spec, pkg, task = train_host.stage_package('h6_core6', runs / 'h6', weights)
    train_host.import_package(pkg)
    import common
    tr = importlib.import_module('train')
    config = common.read(pkg / 'CONFIG.json')
    out = {'targets': [_h6_targets(tr, pkg, arm) for arm in ['core6', 'vm']]}
    model = tr.make_model('vm', 'cpu', config)  # loads the bundled ImageNet-22k->1k ConvNeXt-S weights (pretrain.pth)
    out['params_million'] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
    out['status'] = 'PASS'
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--test3-cache', required=True)
    ap.add_argument('--work', default='/tmp/rsna_cpu_smoke')
    ap.add_argument('--skip-step', action='store_true')
    ap.add_argument('--only', choices=['step', 'h6'], help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.only:  # isolated child process (the H6 and CoAtNet packages share module names)
        work = Path(a.work)
        fn = optimizer_step if a.only == 'step' else h6_construct
        print('RESULT ' + json.dumps(fn(Path(a.weights), work / 'runs' / ('child_' + a.only))), flush=True)
        return
    work = Path(a.work)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    report = {}
    from rsna_knee_final import sources
    checked, problems = sources.verify_repository()
    report['sources'] = {'checked_files': checked, 'problems': problems}
    began = time.time()
    cache = synthetic_cache(work / 'cache336_synthetic')
    report['synthetic_cache_seconds'] = round(time.time() - began, 1)
    ext = work / 'extension_synthetic'
    shutil.copytree(REPO / 'data_prep' / 'extension_cache' / 'reference', ext)
    def save():
        (work / 'CPU_SMOKE.partial.json').write_text(json.dumps(report, indent=1) + '\n')
    report['data_checks'] = []
    for m in ['fold0', 'fold1', 'fold2', 'fold3', 'fold4', 'core', 'repair', 'mirror', 'adaptive', 'joint36']:
        report['data_checks'].append(check_model(m, cache, a.weights, work / 'runs', a.test3_cache, ext if m == 'adaptive' else None))
        print(json.dumps(report['data_checks'][-1]), flush=True)
        save()
    if not a.skip_step:
        for key, only in [('h6_cpu', 'h6'), ('coatnet_cpu_step', 'step')]:
            proc = subprocess.run([sys.executable, __file__, '--weights', a.weights, '--test3-cache', a.test3_cache, '--work', str(work), '--only', only],
                                  capture_output=True, text=True)
            lines = [l for l in proc.stdout.splitlines() if l.startswith('RESULT ')]
            report[key] = json.loads(lines[-1][7:]) if lines else {'status': 'FAIL', 'rc': proc.returncode, 'tail': (proc.stdout + proc.stderr)[-2000:]}
            print(key, json.dumps(report[key])[:400], flush=True)
            save()
    ok = not problems and all(r['rc'] == 0 for r in report['data_checks']) and (a.skip_step or (report['coatnet_cpu_step']['status'] == 'PASS' and report['h6_cpu']['status'] == 'PASS'))
    report['verdict'] = 'PASS' if ok else 'FAIL'
    (work / 'CPU_SMOKE.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
