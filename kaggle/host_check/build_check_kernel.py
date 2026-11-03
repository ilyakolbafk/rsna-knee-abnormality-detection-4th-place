#!/usr/bin/env python3
"""Build a private Kaggle verification notebook for this package (no submission).

It embeds predict.py + src/ + configs/ + labels/gold58 and runs, on 2x T4 with the same 9 datasets as the scored notebook:
  A. public sample (3 test studies), mode --input-root: Sixway output, H6 parity and the final submission must equal the scored
     notebook's sample outputs byte for byte (asserted inside predict.py);
  B. Gold58 (58 train studies with our expert labels, as a test-style folder of symlinks), mode --weights/--data through a
     weights folder that contains only the files listed in configs/weights_manifest.json: checks that the reduced weights
     set is sufficient, compares H6 to its accepted-cache predictions, reports AUCs and timing.
Note: most Sixway members and the final blend were fitted with Gold58 in training, so B's AUCs are not held-out
(only H6 excluded Gold58 from its fit).

Usage: python kaggle/host_check/build_check_kernel.py [--id owner/slug]   ->  kaggle/host_check/kernel/
"""
import argparse
import base64
import io
import json
import tarfile
from pathlib import Path

PKG = Path(__file__).resolve().parents[2]
SCORED = PKG / 'kaggle' / 'scored_notebook'

SETUP = r'''# Host check: unpack the package subset (predict.py, src/, configs/, labels/gold58)
import base64, io, json, os, shutil, subprocess, sys, tarfile, time
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path('/tmp/repo'); shutil.rmtree(REPO, ignore_errors=True); REPO.mkdir(parents=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PACKAGE_B64)), mode='r:gz') as t: t.extractall(REPO, filter='data')
WORK = Path('/kaggle/working')
print('REPO', sorted(p.name for p in REPO.iterdir()), flush=True)
def run_predict(name, *args):
    out = WORK / name; t0 = time.perf_counter()
    with (WORK / f'{name}_predict.log').open('w', buffering=1) as f:
        rc = subprocess.run([sys.executable, '-u', str(REPO / 'predict.py'), *args, '--out', str(out)], stdout=f, stderr=subprocess.STDOUT).returncode
    print((WORK / f'{name}_predict.log').read_text()[-4000:], flush=True)
    return rc, time.perf_counter() - t0, out
'''

RUN_A = r'''# A. Public sample through the Kaggle mounts (mode 1). predict.py asserts byte equality with the scored notebook.
rc_a, sec_a, out_a = run_predict('A_test3', '--input-root', '/kaggle/input')
rec_a = json.loads((out_a / 'RUN_RECEIPT.json').read_text()) if (out_a / 'RUN_RECEIPT.json').exists() else None
RESULT = {'A_test3': dict(rc=rc_a, seconds=sec_a, receipt=rec_a)}
print('A_RESULT', json.dumps(RESULT['A_test3'], default=str)[:3000], flush=True)
'''

RUN_B = r'''# B. Gold58 through a reduced weights folder (mode 2).
ROOT = Path('/kaggle/input')
comp = [p for p in [ROOT/'rsna-knee-abnormality-detection', ROOT/'competitions/rsna-knee-abnormality-detection'] if (p/'sample_submission.csv').is_file()][0]
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
RD = dict(dtype={'StudyInstanceUID': str}, float_precision='round_trip')
gold = pd.read_csv(REPO/'labels/gold58/gold58_labels.csv', dtype={'StudyInstanceUID': str})
gser = pd.read_csv(REPO/'labels/gold58/gold58_series.csv', dtype=str)
gids = gold.StudyInstanceUID.tolist(); assert len(gids) == 58
GD = Path('/tmp/gold_data'); shutil.rmtree(GD, ignore_errors=True); (GD/'test_series').mkdir(parents=True)
for u in gids:
    assert (comp/'train_series'/u).is_dir(), u
    os.symlink(comp/'train_series'/u, GD/'test_series'/u)
gser.to_csv(GD/'test_series.csv', index=False)
pd.DataFrame({'StudyInstanceUID': gids}).to_csv(GD/'test.csv', index=False)
pd.DataFrame({'StudyInstanceUID': gids, **{t: 0.5 for t in TARGETS}}).to_csv(GD/'sample_submission.csv', index=False)
# weights folder = only the files of configs/weights_manifest.json (symlinks into the mounted datasets)
man = json.loads((REPO/'configs/weights_manifest.json').read_text())
WV = Path('/tmp/weights_view'); shutil.rmtree(WV, ignore_errors=True)
for row in man['inference_files']:
    slug, name = row['path'].split('/', 1)
    src = [p for p in [ROOT/slug, ROOT/'datasets/ilyakolb'/slug] if p.is_dir()][0] / name
    rel = row['path']
    if not src.exists() and name.endswith('.tar') and (src.parent/Path(name).stem).is_dir():
        src, rel = src.parent/Path(name).stem, rel[:-4]  # Kaggle mounted the archive unpacked
    dst = WV/rel; dst.parent.mkdir(parents=True, exist_ok=True); os.symlink(src, dst, target_is_directory=src.is_dir())
rc_b, sec_b, out_b = run_predict('B_gold58', '--weights', str(WV), '--data', str(GD), '--staging', '/tmp/stage_gold', '--skip-weights-check')
def auc(y, s):
    y = np.asarray(y, float); r = pd.Series(np.asarray(s, float)).rank(method='average').to_numpy(); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
Y = gold.set_index('StudyInstanceUID').loc[gids, TARGETS].to_numpy(float)
res = dict(rc=rc_b, seconds=sec_b)
if (out_b/'submission.csv').exists():
    res['receipt'] = json.loads((out_b/'RUN_RECEIPT.json').read_text())
    for name, file in [('sixway', 'sixway_core.csv'), ('h6_core6', 'h6_core6_raw.csv'), ('h6_vm', 'h6_vm_raw.csv'), ('final_ens', 'submission.csv')]:
        P = pd.read_csv(out_b/file, **RD).set_index('StudyInstanceUID').loc[gids, TARGETS].to_numpy(float)
        per = {t: auc(Y[:, j], P[:, j]) for j, t in enumerate(TARGETS)}
        res[name] = dict(macro=float(np.mean(list(per.values()))), per_target=per)
    ref = json.loads((REPO/'configs/reference/gold58_h6_swa1416_cache_reference.json').read_text())
    for arm in ['core6', 'vm']:
        P = pd.read_csv(out_b/f'h6_{arm}_raw.csv', **RD).set_index('StudyInstanceUID').loc[gids, TARGETS].to_numpy(float)
        R = pd.DataFrame(ref[arm]['pred'], index=ref[arm]['ids'], columns=TARGETS).loc[gids].to_numpy(float)
        d = np.abs(P - R)
        res[f'h6_{arm}_vs_cache_reference'] = dict(max_abs=float(np.nanmax(d)), mean_abs=float(np.nanmean(d)), ref_macro=ref[arm]['macro'])
RESULT['B_gold58'] = res
RESULT['verdict'] = dict(
    A_public_sample_byte_identical=bool(rec_a and rec_a.get('sample_reference_match') and rec_a['sixway']['sha256'] == '9e42cd0844c2dc7dd7bcc22f995e849c368b060abc3565665d252b876fc55e49'),
    B_gold58_complete=bool(rc_b == 0 and 'final_ens' in res),
    B_h6_parity_within_5e3=bool('final_ens' in res and all(res[f'h6_{a}_vs_cache_reference']['max_abs'] <= 5e-3 for a in ['core6', 'vm'])))
(WORK/'HOST_CHECK.json').write_text(json.dumps(RESULT, indent=2, default=str) + '\n')
print('HOST_CHECK', json.dumps(RESULT['verdict']), json.dumps({k: res[k]['macro'] for k in ['sixway', 'h6_core6', 'h6_vm', 'final_ens'] if k in res}), flush=True)
'''


def package_b64():
    """predict.py + src/rsna_knee_final (inference only) + configs + labels/gold58 as base64 tar.gz."""
    buf = io.BytesIO()
    skip = ('__pycache__', 'train_host.py', 'sources.py')
    with tarfile.open(fileobj=buf, mode='w:gz') as t:
        for rel in ['predict.py', 'src', 'configs', 'labels/gold58']:
            t.add(PKG / rel, arcname=rel, filter=lambda ti: None if any(x in ti.name for x in skip) else ti)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--id', default='ilyakolb/rsna-final-ens-host-check-20261023')
    a = ap.parse_args()
    b64 = package_b64()
    scored = json.loads((SCORED / 'inference.ipynb').read_text())
    cells = [dict(cell_type='markdown', metadata={}, source=['# FINAL ens host check: package predict.py on the public sample and Gold58 (no submission)\n'])]
    for i, code in enumerate([f'PACKAGE_B64 = {b64!r}\n' + SETUP, RUN_A, RUN_B]):
        compile(code, f'cell{i}', 'exec')
        cells.append(dict(cell_type='code', id=f'cell-{i}', metadata={}, execution_count=None, outputs=[], source=code.splitlines(keepends=True)))
    nb = dict(scored, cells=cells)
    kdir = Path(__file__).resolve().parent / 'kernel'
    kdir.mkdir(exist_ok=True)
    (kdir / 'inference.ipynb').write_text(json.dumps(nb, indent=1) + '\n')
    meta = json.loads((SCORED / 'kernel-metadata.json').read_text())
    meta.update(id=a.id, title='RSNA Final Ens Host Check 20261023', is_private=True)
    (kdir / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps({'kernel': a.id, 'embedded_b64_chars': len(b64), 'dir': str(kdir)}))


if __name__ == '__main__':
    main()
