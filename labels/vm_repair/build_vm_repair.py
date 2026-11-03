#!/usr/bin/env python3
"""Consensus repair of VM (teacher-V) labels in the zone "K24 OOF much higher than VM, confirmed by an
independent channel". Written by Claude 2026-10-15 (data research, see DATA_RESEARCH_K24_VM_E_CORE6_20261015.md).

CPU only, numpy + pandas. No training, no Gold, no thresholds tuned on Gold.

Rule (declared before any LB):
  k24c   = per-target monotone quantile map of K24 OOF onto the VM label distribution, fit on the pooled OOF
           of all supplied folds (knee level).
  aux60c = same map for the old AUX60 5-fold OOF (CoAtNet trained on Core5 TEXT labels), fit on all knees.
  core6  = Core6 equal text reader mean (per study).
  zone   = (k24c - vm >= THR) & (aux60c >= .5 | core6 >= .5) & label_mask > 0 & not bilateral knee.
  new    = k24c (mode 'k24', default since 15.10 evening, Ilya's question)  |  (vm + k24c) / 2 (mode 'mean')  |
           max(vm, FIXED) (mode 'fixed').
Masks, confidences, row order and all other columns are copied unchanged.

Usage (all five folds required by default):
  python3 build_vm_repair.py --vmain V_MAIN.csv --oof f0.csv f1.csv f2.csv f3.csv f4.csv \
      --aux60 aux60_oof_long.csv --core6 train4407_core6_long.csv --out OUTDIR [--mode k24] [--thr 0.4]
  add --allow-partial for a dry run with fewer folds (labels of knees without OOF stay unchanged).
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd

T = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
IDS = ['SampleID','StudyInstanceUID','ParentStudyInstanceUID','report_group']

def sha(p):
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()

def qmap(src, ref):
    qs = np.linspace(0, 1, 201)
    s = np.quantile(src, qs); r = np.quantile(ref, qs)
    s = np.maximum.accumulate(s + np.arange(len(s)) * 1e-12)
    return lambda x: np.interp(x, s, r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vmain', required=True); ap.add_argument('--oof', nargs='+', required=True)
    ap.add_argument('--aux60', required=True); ap.add_argument('--core6', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--mode', default='k24', choices=['mean','k24','fixed'])
    ap.add_argument('--fixed', type=float, default=0.75); ap.add_argument('--thr', type=float, default=0.4)
    ap.add_argument('--allow-partial', action='store_true')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    v = pd.read_csv(a.vmain, dtype={c: str for c in IDS})
    n = len(v); vm = v[T].to_numpy(float)
    mask = v[['label_mask__'+t for t in T]].to_numpy(float) > 0
    folds = v.fold.astype(int).to_numpy()
    # K24 OOF
    kk = np.full((n, 12), np.nan); pos = {s: i for i, s in enumerate(v.SampleID)}
    seen = set()
    for p in a.oof:
        f = pd.read_csv(p, dtype={c: str for c in IDS})
        rows = np.array([pos[s] for s in f.SampleID])
        fl = set(folds[rows].tolist())
        if len(fl) != 1: sys.exit(f'{p}: rows span folds {fl}')
        fl = fl.pop()
        if fl in seen: sys.exit(f'fold {fl} supplied twice')
        if len(rows) != (folds == fl).sum(): sys.exit(f'{p}: fold {fl} incomplete ({len(rows)} of {(folds==fl).sum()})')
        seen.add(fl); kk[rows] = f[T].to_numpy(float)
    missing = sorted(set(folds.tolist()) - seen)
    if missing and not a.allow_partial: sys.exit(f'missing OOF folds {missing}; use --allow-partial for a dry run')
    has = ~np.isnan(kk).any(1)
    kc = np.full_like(kk, np.nan)
    for j in range(12):
        f = qmap(kk[has, j], vm[has, j]); kc[has, j] = f(kk[has, j])
    # AUX60 OOF (per study) and Core6 text (per study)
    ax = pd.read_csv(a.aux60).pivot(index='StudyInstanceUID', columns='label', values='aux60_oof').reindex(columns=T)
    AX = ax.reindex(v.StudyInstanceUID.values).to_numpy(float)
    axc = np.full_like(AX, np.nan)
    for j in range(12):
        ok = ~np.isnan(AX[:, j]); f = qmap(AX[ok, j], vm[ok, j]); axc[ok, j] = f(AX[ok, j])
    c6 = pd.read_csv(a.core6).pivot(index='StudyInstanceUID', columns='label', values='core6_equal').reindex(columns=T)
    TX = c6.reindex(v.StudyInstanceUID.values).to_numpy(float)
    bil = v.SampleID.str.contains('::').to_numpy()
    confirm = (np.nan_to_num(axc, nan=0) >= .5) | (np.nan_to_num(TX, nan=0) >= .5)
    zone = has[:, None] & mask & (~bil)[:, None] & ((kc - vm) >= a.thr) & confirm
    new = vm.copy()
    if a.mode == 'mean': new[zone] = (vm[zone] + kc[zone]) / 2
    elif a.mode == 'k24': new[zone] = kc[zone]
    else: new[zone] = np.maximum(vm[zone], a.fixed)
    new = np.round(new, 4)
    r = v.copy(); r[T] = new
    r.to_csv(out / 'V_MAIN_REPAIR.csv', index=False)
    ii, jj = np.where(zone)
    ch = pd.DataFrame({'SampleID': v.SampleID.values[ii], 'fold': folds[ii], 'target': [T[j] for j in jj],
                       'vm_old': vm[ii, jj], 'k24_oof': kk[ii, jj], 'k24c': np.round(kc[ii, jj], 4),
                       'aux60c': np.round(axc[ii, jj], 4), 'core6': np.round(TX[ii, jj], 4), 'new': new[ii, jj]})
    ch.to_csv(out / 'CHANGES.csv', index=False)
    vmpos = (vm >= .5) & mask & has[:, None]
    receipt = {
        'rule': {'thr': a.thr, 'mode': a.mode, 'fixed': a.fixed if a.mode == 'fixed' else None,
                 'confirm': 'aux60c>=.5 or core6>=.5', 'bilateral': 'excluded', 'gold': 'not used'},
        'folds_with_oof': sorted(seen), 'folds_missing': missing, 'knees_with_oof': int(has.sum()), 'knees_total': n,
        'changed_cells': int(zone.sum()), 'changed_knees': int(zone.any(1).sum()),
        'changed_by_target': {t: int(zone[:, j].sum()) for j, t in enumerate(T)},
        'vm_pos_cells_by_target_in_oof_knees': {t: int(vmpos[:, j].sum()) for j, t in enumerate(T)},
        'changed_by_fold': {int(f): int(zone[folds == f].sum()) for f in sorted(seen)},
        'inputs': {'vmain': {'path': a.vmain, 'sha256': sha(a.vmain)}, 'aux60': {'path': a.aux60, 'sha256': sha(a.aux60)},
                   'core6': {'path': a.core6, 'sha256': sha(a.core6)}, 'oof': [{'path': p, 'sha256': sha(p)} for p in a.oof]},
        'outputs': {'V_MAIN_REPAIR.csv': sha(out / 'V_MAIN_REPAIR.csv'), 'CHANGES.csv': sha(out / 'CHANGES.csv')},
        'unchanged_columns_check': bool((r.drop(columns=T).astype(str).values == v.drop(columns=T).astype(str).values).all()),
    }
    (out / 'RECEIPT.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False))
    print(json.dumps({k: receipt[k] for k in ['folds_with_oof', 'folds_missing', 'changed_cells', 'changed_knees', 'changed_by_target']}, ensure_ascii=False))

if __name__ == '__main__':
    main()
