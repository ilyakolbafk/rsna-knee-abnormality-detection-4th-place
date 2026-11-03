"""FINAL ens inference: Sixway (stage 1) -> H6 Core6 + VM (stage 2) -> rank blend (stage 3).

Same code and arithmetic as the scored Kaggle notebook ilyakolb/rsna-final-six-h6-ens-20261022 v2 (submission 57532611),
reorganised into a script. Differences from the notebook, all outside the models and the arithmetic:
  * the Sixway stage reads its input/output roots from RSNA_INPUT_ROOT / RSNA_WORK_DIR (defaults: the Kaggle paths);
  * the Sixway and H6 helper scripts are plain files instead of base64 strings inside notebook cells;
  * the H6 time cap is a budget relative to the start of the run (--time-budget-hours, default 9 h as on Kaggle)
    instead of the fixed competition-night wall clock (2026-10-22 23:45 UTC). The scored run finished well before that
    wall clock, so the cap did not fire there; the DEADLINE fallback rule (Sixway rank for unscored studies) is unchanged.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import PACKAGE_ROOT, TARGETS
from .blend import blend, load_weights

HERE = Path(__file__).resolve().parent / 'inference'
CONFIGS = PACKAGE_ROOT / 'configs'
READ = dict(dtype={'StudyInstanceUID': str}, float_precision='round_trip')
COMPETITION = 'rsna-knee-abnormality-detection'
OWNER = 'ilyakolb'
SIXWAY_TMP = Path('/tmp/rsna-sixway-resident-20261017')
PUBLIC_UIDS = ['1.2.826.0.1.3680043.8.498.10047035057544427318018579121635276191',
               '1.2.826.0.1.3680043.8.498.10062861783145312629332250977456991776',
               '1.2.826.0.1.3680043.8.498.10067514707072572280263481548497591402']  # the 3 public sample studies
SIXWAY_REF_SHA = '9e42cd0844c2dc7dd7bcc22f995e849c368b060abc3565665d252b876fc55e49'
SUBMISSION_REF_SHA = '9c9d3132f2ae34a9e437bd2ee92f7ae32a89a13a598b2760f519a84995d55e3b'
H6_SLUG = 'rsna-h6-rank-full336-20261022'
H6_RELEASE = 'f7d79463ab16720b1845c1ddbf07a890ecee10dcd7b08f1c6922c317195a525e'
H6_INFER_SHA = '781f18650de9e5ee62c1c76b928273a69b0df1cd8c358cd2a43218b808e9e720'
H6_SWA_SLUG = 'rsna-h6-swa1416-weights-20261022'
H6_SWA_SHA = {'core6': 'c9d916ffe853af4dcdee3452178ce62c6d558c15b86c58881b6956a4b146b015',
              'vm': '9e0960154610daf25a63a4d3ede3737bf69d06079d649f11a72b54d85390f7d6'}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def log(tag, **kw):
    print(tag, json.dumps(kw, default=str), flush=True)


def valid_frame(frame, uids=None):
    ok = list(frame.columns) == ['StudyInstanceUID', *TARGETS] and len(frame) >= 2 and frame.StudyInstanceUID.is_unique
    if ok:
        v = frame[TARGETS].to_numpy(float)
        ok = bool(np.isfinite(v).all() and ((v >= 0) & (v <= 1)).all())
    return ok and (uids is None or frame.StudyInstanceUID.tolist() == uids)


# ---------------------------------------------------------------- input layout

def find_dataset(root, slug):
    hits = [p for p in (root / slug, root / 'datasets' / OWNER / slug) if p.is_dir()]
    if len(hits) != 1:
        raise FileNotFoundError(f'expected exactly one mount of {OWNER}/{slug} under {root}, found {len(hits)}')
    return hits[0]


def find_competition(root):
    hits = [p for p in (root / COMPETITION, root / 'competitions' / COMPETITION)
            if (p / 'test_series').is_dir() and all((p / n).is_file() for n in ('sample_submission.csv', 'test.csv', 'test_series.csv'))]
    if len(hits) != 1:
        raise FileNotFoundError(f'expected exactly one competition directory under {root}, found {len(hits)}')
    return hits[0]


def stage_input_root(weights_dir, data_dir, staging):
    """Build a Kaggle-style input root from a local weights folder and a competition-style data folder (symlinks only)."""
    weights_dir, data_dir, staging = Path(weights_dir).resolve(), Path(data_dir).resolve(), Path(staging)
    if staging.exists():
        raise FileExistsError(f'{staging} exists; use a fresh staging directory')
    manifest = json.loads((CONFIGS / 'weights_manifest.json').read_text())
    for row in manifest['inference_files']:
        src, rel = weights_dir / row['path'], row['path']
        if not src.is_file() and rel.endswith('.tar') and src.with_suffix('').is_dir():
            src, rel = src.with_suffix(''), rel[:-len('.tar')]  # an archive delivered already unpacked (Kaggle does this on upload)
        if not src.exists():
            raise FileNotFoundError(f'missing weights file {src}')
        dst = staging / 'datasets' / OWNER / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(src, dst, target_is_directory=src.is_dir())
    comp = staging / 'competitions'
    comp.mkdir(parents=True)
    os.symlink(data_dir, comp / COMPETITION, target_is_directory=True)
    return staging


def verify_weights(input_root):
    manifest = json.loads((CONFIGS / 'weights_manifest.json').read_text())
    bad = []
    for row in manifest['inference_files']:
        slug, name = row['path'].split('/', 1)
        p = find_dataset(input_root, slug) / name
        if not p.exists():
            # Kaggle may unpack an uploaded .tar into a folder of the same stem; the stages accept both forms.
            if name.endswith('.tar') and (p.parent / Path(name).stem).is_dir():
                continue
            bad.append((row['path'], 'missing'))
        elif sha256(p) != row['sha256']:
            bad.append((row['path'], 'sha256'))
    if bad:
        raise ValueError(f'weights verification failed: {bad[:5]}')
    return len(manifest['inference_files'])


# ---------------------------------------------------------------- stage 1: Sixway

def run_sixway(input_root, work):
    """The Sixway resident cell of 359192097, unchanged except for the two root paths, as an isolated child process."""
    script = HERE / 'sixway_stage.py'
    (work / 'submission.csv').unlink(missing_ok=True)
    if SIXWAY_TMP.exists():
        raise FileExistsError(f'{SIXWAY_TMP} exists (left from another run); start from a clean /tmp')
    env = dict(os.environ, RSNA_INPUT_ROOT=str(input_root), RSNA_WORK_DIR=str(work))
    t0 = time.perf_counter()
    with (work / 'sixway_stage.log').open('w', buffering=1) as f:
        rc = subprocess.run([sys.executable, '-u', str(script)], cwd=str(work), env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    seconds = time.perf_counter() - t0
    shutil.rmtree(SIXWAY_TMP, ignore_errors=True)  # free packed weights and the raw DICOM bank before H6
    if not (work / 'submission.csv').exists():
        raise RuntimeError(f'Sixway stage wrote no submission (rc={rc}); see {work / "sixway_stage.log"}')
    os.replace(work / 'submission.csv', work / 'sixway_core.csv')  # no submission.csv exists until the blend is written
    base = pd.read_csv(work / 'sixway_core.csv', **READ)
    if not valid_frame(base):
        raise ValueError('Sixway output invalid')
    digest = sha256(work / 'sixway_core.csv')
    sample = base.StudyInstanceUID.tolist() == PUBLIC_UIDS
    if sample:
        assert rc == 0, ('Sixway child failed on the public sample', rc)
        assert digest == SIXWAY_REF_SHA, ('Sixway public-sample output differs from the scored notebook', digest)
    log('SIXWAY_STAGE', rc=rc, seconds=seconds, sha256=digest, sample_run=sample, reference_match=digest == SIXWAY_REF_SHA, studies=len(base))
    return base, dict(rc=rc, seconds=seconds, sha256=digest, sample_run=sample)


# ---------------------------------------------------------------- stage 2: H6

def unpack_h6_release(input_root, root=Path('/tmp/h6/release/full-v4')):
    mount = find_dataset(input_root, H6_SLUG)
    assert sha256(mount / 'RELEASE.json') == H6_RELEASE, 'H6 release dataset not found'
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)  # the producer code uses Path(__file__).parents[2]: keep the path depth >= 3
    if (mount / 'payload.tar').exists():
        with tarfile.open(mount / 'payload.tar') as t:
            t.extractall(root, filter='data')
    else:
        shutil.copytree(mount / 'payload', root, dirs_exist_ok=True)
    assert sha256(root / 'RELEASE.json') == H6_RELEASE
    release = json.loads((root / 'RELEASE.json').read_text())
    for name, digest in release['files'].items():
        if name == 'wheels.tar' and not (root / name).exists():
            continue  # Kaggle unpacks nested archives on upload
        assert sha256(root / name) == digest, name
    assert release['files']['infer_full.py'] == H6_INFER_SHA
    wheels = Path('/tmp/h6-profile-wheels')
    shutil.rmtree(wheels, ignore_errors=True)
    wheels.mkdir()
    if (root / 'wheels.tar').exists():
        with tarfile.open(root / 'wheels.tar') as t:
            t.extractall(wheels, filter='data')
    else:
        for whl in list(root.rglob('*.whl')) + list(mount.rglob('*.whl')):
            if not (wheels / whl.name).exists():
                shutil.copy(whl, wheels / whl.name)
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename
    tags = set(sys_tags())
    compatible = [p for p in wheels.glob('*.whl') if parse_wheel_filename(p.name)[3] & tags]
    assert any(p.name.startswith('timm-1.0.29-') for p in compatible)
    deps = Path('/tmp/h6-pinned-deps')
    shutil.rmtree(deps, ignore_errors=True)
    deps.mkdir()
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '--no-index', '--no-deps',
                    '--target', str(deps), *map(str, compatible)], check=True)
    for name in ('h6_fast.py', 'h6_prep.py'):  # the pipelined scheduler of the scored notebook (same producer, models, windows)
        shutil.copy(HERE / name, root / name)
    return root, deps, sorted(p.name for p in compatible)


def run_h6(input_root, comp, work, uids, sample, deadline_unix, recipe):
    root, deps, wheels = unpack_h6_release(input_root)
    wdir = find_dataset(input_root, H6_SWA_SLUG)
    wm = json.loads((wdir / 'MANIFEST.json').read_text())
    assert {arm: wm[arm]['sha256'] for arm in H6_SWA_SHA} == H6_SWA_SHA
    wman = Path('/tmp/h6_weights_manifest.json')
    wman.write_text(json.dumps({arm: dict(path=str(wdir / wm[arm]['file']), sha256=wm[arm]['sha256'], epochs=wm[arm]['epochs']) for arm in H6_SWA_SHA}))
    out = Path('/tmp/h6_out')
    shutil.rmtree(out, ignore_errors=True)
    shutil.rmtree('/tmp/h6q', ignore_errors=True)  # h6_fast.py's per-study scratch queue
    env = dict(os.environ, PYTHONPATH=str(deps), PYTHONNOUSERSITE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', **recipe['env'])
    deadline = max(time.time() + 1200, deadline_unix)  # same 20-minute floor as the notebook
    budget_seconds = deadline - time.time()
    t0 = time.perf_counter()
    cmd = [sys.executable, '-u', str(root / 'h6_fast.py'), '--data', str(comp), '--out', str(out), '--workers', str(int(recipe['workers'])),
           '--deadline-unix', str(deadline), '--precision', recipe['precision'], '--weights-manifest', str(wman), *recipe.get('args', [])]
    with (work / 'h6_stage.log').open('w', buffering=1) as f:
        try:
            rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=max(1500, int(budget_seconds) + 600)).returncode
        except subprocess.TimeoutExpired:
            rc = 'timeout'
    seconds = time.perf_counter() - t0
    for name in ('H6_FAST_AUDIT.json', 'H6_FAST_PROGRESS.json'):
        if (out / name).exists():
            shutil.copy(out / name, work / ('h6_' + name))
    assert (out / 'H6_FAST_AUDIT.json').exists(), ('H6 stage wrote no audit', rc)
    audit = json.loads((out / 'H6_FAST_AUDIT.json').read_text())
    fail_ids = sorted(x['id'] for x in audit['failures'])
    frames = {}
    for arm in ('core6', 'vm'):
        p = out / f'{arm}_raw_probabilities.csv'
        assert p.exists(), ('H6 stage wrote no raw probabilities', arm, rc)
        f = pd.read_csv(p, **READ)
        assert list(f.columns) == ['StudyInstanceUID', *TARGETS] and sorted(f.StudyInstanceUID) == sorted(uids), ('H6 UIDs differ from Sixway', arm)
        f = f.set_index('StudyInstanceUID').loc[uids].reset_index()
        v = f[TARGETS].to_numpy(float)
        nan_rows = np.isnan(v).any(axis=1)
        assert sorted(f.StudyInstanceUID[nan_rows]) == fail_ids and bool(((v[~nan_rows] >= 0) & (v[~nan_rows] <= 1)).all()), ('H6 values invalid', arm)
        f.to_csv(work / f'h6_{arm}_raw.csv', index=False)
        frames[arm] = f
    max_fail = max(1, int(0.03 * len(uids)))
    deadline_ids = sorted(x['id'] for x in audit['failures'] if x.get('error') == 'DEADLINE')
    info = dict(weights=H6_SWA_SHA, recipe=recipe, wheels=wheels, rc=rc, seconds=seconds, failures=len(fail_ids), max_failures=max_fail,
                deadline_failures=len(deadline_ids), hard_failures=len(fail_ids) - len(deadline_ids), budget_seconds=budget_seconds,
                gpu_seconds=audit['gpu_seconds'], total_seconds=audit['total_seconds'], first_failures=audit['failures'][:3])
    assert info['hard_failures'] <= max_fail, ('too many H6 failures (not deadline)', info)  # DEADLINE studies take the Sixway rank
    if sample:
        assert rc == 0 and not fail_ids, ('H6 child failed on the public sample', info)
        ref = json.loads((CONFIGS / 'reference' / 'test3_h6_reference.json').read_text())
        for arm in ('core6', 'vm'):
            d = np.abs(np.asarray([ref[arm][u] for u in uids], float) - frames[arm][TARGETS].to_numpy(float))
            info[f'{arm}_max_abs_difference'] = float(d.max())
            assert d.max() <= float(recipe['parity_tol']), info
    log('H6_STAGE', **info)
    return frames, info


# ---------------------------------------------------------------- full run

BLENDS = {  # name -> (weights file, public-sample submission sha256, pipeline id, reference submission)
    'final': ('blend_final_ens.json', SUBMISSION_REF_SHA, 'FINAL_SIX_H6SWA_ENS_RANK', 57532611),
    # same members, H6 only on Medial OA / Lateral OA / PF OA / Baker's; sample sha derived from the same member outputs
    'late_p4': ('blend_late_p4.json', '68946c1fe1eb92bea579073bdfdde7de4992db03dd48b38cf87abcf4039e1c50', 'LATE_SIXWAY_P4_LITE', 57543464),
}


def run(input_root, work, time_budget_hours=9.0, check_weights=True, blend_name='final', sixway_only=False):
    """Full pipeline. blend_name selects the stage-3 weights (configs/blend_*.json); sixway_only stops after stage 1
    and writes the Sixway output as the submission (Sixway solo, submission 57291068)."""
    t_start, t_unix = time.perf_counter(), time.time()
    input_root, work = Path(input_root), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    if any(work.iterdir()):
        raise FileExistsError(f'{work} is not empty')
    comp = find_competition(input_root)
    n_files = verify_weights(input_root) if check_weights else None
    log('PREFLIGHT', input_root=str(input_root), competition=str(comp), weights_files_verified=n_files)
    base, s_info = run_sixway(input_root, work)
    uids = base.StudyInstanceUID.tolist()
    if sixway_only:
        shutil.copyfile(work / 'sixway_core.csv', work / 'submission.csv')
        receipt = dict(status='PASS', pipeline='SIXWAY_ONLY', reference_submission=57291068, studies=len(uids), sixway=s_info,
                       submission_sha256=sha256(work / 'submission.csv'), sample_run=s_info['sample_run'],
                       sample_reference_match=(s_info['sha256'] == SIXWAY_REF_SHA) if s_info['sample_run'] else None,
                       wall_seconds=time.perf_counter() - t_start)
        (work / 'RUN_RECEIPT.json').write_text(json.dumps(receipt, indent=2, default=str) + '\n')
        log('SIXWAY_ONLY_COMPLETE', **receipt)
        return receipt
    weights_file, ref_sha, pipeline_id, reference = BLENDS[blend_name]
    recipe = json.loads((CONFIGS / 'h6_inference_recipe.json').read_text())
    # Same reserve as the notebook: the whole run gets time_budget_hours; H6 stops 25 minutes before it ends
    # (blend and write), and every study it has not reached by then takes the Sixway rank (DEADLINE rule).
    h6, h_info = run_h6(input_root, comp, work, uids, s_info['sample_run'], t_unix + time_budget_hours * 3600 - 1500, recipe)
    assert not (work / 'submission.csv').exists()
    final = blend(base, h6['core6'], h6['vm'], load_weights(CONFIGS / weights_file))
    final.to_csv(work / 'submission.csv', index=False)
    check = pd.read_csv(work / 'submission.csv', **READ)
    assert check.StudyInstanceUID.tolist() == uids
    digest = sha256(work / 'submission.csv')
    if s_info['sample_run']:
        assert digest == ref_sha, ('public-sample submission differs from the reference', blend_name, digest)
    receipt = dict(status='PASS', pipeline=pipeline_id, blend=weights_file, reference_submission=reference, studies=len(uids),
                   sixway=s_info, h6=h_info, submission_sha256=digest, sample_run=s_info['sample_run'],
                   sample_reference_match=digest == ref_sha if s_info['sample_run'] else None,
                   wall_seconds=time.perf_counter() - t_start)
    (work / 'RUN_RECEIPT.json').write_text(json.dumps(receipt, indent=2, default=str) + '\n')
    log('FINAL_ENS_COMPLETE', **receipt)
    return receipt
