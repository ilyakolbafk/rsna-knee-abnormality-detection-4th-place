"""22.10 (Claude): pipelined H6 full-v4 EMA10 member (release f7d79463). Same producer, models, FP32 math and windows as infer_full.py;
only the schedule changes: N spawned producer workers prepare upcoming studies while the parent runs both arms (cuda:0 core6, cuda:1 vm).
A study whose producer or model fails gets NaN (recorded); the caller decides the failure policy. Writes core6/vm raw probabilities."""
import os
os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
from pathlib import Path
import argparse, csv, gc, hashlib, json, multiprocessing as mp, sys, threading, time, traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'h6')); sys.path.insert(0, str(HERE))
from model import Model
from public_math import Net
import h6_prep
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def read(p): return json.loads(p.read_text())
def save(p, x):
    t = p.with_suffix('.partial'); t.write_text(json.dumps(x, indent=2) + '\n'); os.replace(t, p)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=2); ap.add_argument('--deadline-unix', type=float, default=0.0); ap.add_argument('--max-studies', type=int, default=0); ap.add_argument('--repeat', type=int, default=1); ap.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32'); ap.add_argument('--prefetch-ahead', type=int, default=0); ap.add_argument('--prefetch-threads', type=int, default=16); ap.add_argument('--weights-manifest', type=Path, default=None)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    release = read(HERE / 'RELEASE.json'); assert release['H6_epoch'] == 10  # release code/producer; weights may be overridden below
    assert torch.cuda.is_available() and torch.cuda.device_count() >= 2
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.75, 0); torch.cuda.set_per_process_memory_fraction(.75, 1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    with (a.data / 'sample_submission.csv').open() as f: r = csv.DictReader(f); columns = r.fieldnames; sample = list(r)
    names = release['targets']; ids = [x['StudyInstanceUID'] for x in sample]
    assert columns == ['StudyInstanceUID', *names] and ids and len(ids) == len(set(ids))
    if a.max_studies: ids = ids[:a.max_studies]
    if a.repeat > 1: ids = ids * a.repeat  # timing benchmark only: the same studies again, outputs keyed by position
    rows = list(csv.DictReader((a.data / 'test_series.csv').open()))
    by = {}
    for x in rows: by.setdefault(x['StudyInstanceUID'], []).append(x)
    assert set(ids) <= set(by)
    t_start = time.perf_counter(); models = {}
    # Weights: the release EMA10 by default; --weights-manifest {arm: {path, sha256, epochs}} swaps in other checkpoints of the same package
    # (e.g. fixed SWA of EMA epochs 14-16). Same architecture, strict load, sha pinned.
    wman = json.loads(a.weights_manifest.read_text()) if a.weights_manifest else None
    weights_used = {}
    for arm, device in [('core6', 'cuda:0'), ('vm', 'cuda:1')]:
        if wman:
            file = Path(wman[arm]['path']); assert sha(file) == wman[arm]['sha256'], ('WEIGHTS_SHA', arm)
            cp = torch.load(file, map_location='cpu', mmap=True, weights_only=False)
            assert cp['arm'] == arm and list(cp['epochs']) == list(wman[arm]['epochs']) and cp['averaging_source'] == 'EMA' and cp['package_sha256'] == release['source_package_sha256']
        else:
            info = release['weights'][arm]; file = HERE / info['file']; assert sha(file) == info['sha256']
            cp = torch.load(file, map_location='cpu', mmap=True, weights_only=False)
            assert cp['arm'] == arm and cp['epoch'] == 10 and cp['kind'] == 'ema' and cp['package_sha256'] == release['source_package_sha256']
        weights_used[arm] = {'file': str(file), 'sha256': sha(file)}
        public = Net('convnext_small.fb_in22k_ft_in1k_384', None, hidden=384, drop=.1, crop=352, spatial=True, syn_decomp=True, fm_aux=11)
        model = Model(None, arm, 8, public=public); model.load_state_dict(cp['model'], strict=True); del cp, public; gc.collect(); model.to(device).eval(); models[arm] = (model, device)
    load_seconds = time.perf_counter() - t_start
    canaries = release['public_test3_cache_canaries']
    fp32_reruns = []
    def predict(arm, images, meta):
        model, device = models[arm]; torch.cuda.set_device(device)
        with torch.inference_mode():
            x = torch.from_numpy(images)[None].to(device); m = torch.from_numpy(meta)[None].to(device); mask = torch.ones((1, len(images)), dtype=torch.bool, device=device)
            if a.precision == 'fp16':  # autocast fp16 on T4 tensor cores; any non-finite output -> exact FP32 rerun of that knee
                with torch.autocast('cuda', dtype=torch.float16):
                    z = model.gold(x, m, mask)
                prob = z.float().sigmoid()[0].cpu().numpy()
                if not (np.isfinite(prob).all() and ((prob >= 0) & (prob <= 1)).all()):
                    fp32_reruns.append(arm); prob = model.gold(x, m, mask).sigmoid()[0].float().cpu().numpy()
            else:
                prob = model.gold(x, m, mask).sigmoid()[0].float().cpu().numpy()
            torch.cuda.synchronize(device)
        assert prob.shape == (12,) and np.isfinite(prob).all() and ((prob >= 0) & (prob <= 1)).all()
        return prob
    qdir = Path('/tmp/h6q'); qdir.mkdir(exist_ok=True)
    ctx = mp.get_context('spawn')
    def new_pool(): return ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx, initializer=h6_prep.init)
    pool = new_pool(); pending = {}; nxt = 0; depth = a.workers + 1
    def fill():
        nonlocal nxt
        while nxt < len(ids) and len(pending) < depth:
            u = ids[nxt]; pending[nxt] = pool.submit(h6_prep.prepare, u, by[u], str(a.data / 'test_series'), str(qdir), f'q{nxt}'); nxt += 1
    # DICOM prefetch (reads only, bytes discarded): warms the page cache of /kaggle/input for the next studies with parallel I/O,
    # so the producer's sequential pydicom reads hit memory. Does not change any computation.
    consumer = {'i': 0, 'stop': False}; pf_stats = {'studies': 0, 'bytes': 0, 'seconds': 0.0}
    def _read(path):
        n = 0
        try:
            with open(path, 'rb', buffering=0) as fh:
                while True:
                    b = fh.read(8 << 20)
                    if not b: break
                    n += len(b)
        except OSError:
            pass
        return n
    def prefetch_loop():
        with ThreadPoolExecutor(max_workers=a.prefetch_threads) as io:
            for j, u in enumerate(ids):
                while not consumer['stop'] and j > consumer['i'] + a.prefetch_ahead: time.sleep(0.2)
                if consumer['stop']: return
                t0 = time.perf_counter()
                files = [os.path.join(d, f) for d, _, fs in os.walk(a.data / 'test_series' / u) for f in fs]
                pf_stats['bytes'] += sum(io.map(_read, files)); pf_stats['studies'] += 1; pf_stats['seconds'] += time.perf_counter() - t0
    if a.prefetch_ahead > 0:
        threading.Thread(target=prefetch_loop, daemon=True).start()
    out = {'core6': np.full((len(ids), 12), np.nan), 'vm': np.full((len(ids), 12), np.nan)}
    records = []; failures = []; gpu_seconds = 0.0; restarts = 0
    with ThreadPoolExecutor(max_workers=2) as threads:
        fill()
        for i, uid in enumerate(ids):
            consumer['i'] = i; tick = time.perf_counter()
            if a.deadline_unix and time.time() > a.deadline_unix:
                failures.append({'id': uid, 'error': 'DEADLINE'}); pending.pop(i, None); continue
            try:
                try:
                    res = pending.pop(i).result()
                except BrokenProcessPool:
                    restarts += 1; pool.shutdown(wait=False, cancel_futures=True); pool = new_pool()
                    for j in list(pending): pending[j] = pool.submit(h6_prep.prepare, ids[j], by[ids[j]], str(a.data / 'test_series'), str(qdir), f'q{j}')
                    res = pool.submit(h6_prep.prepare, uid, by[uid], str(a.data / 'test_series'), str(qdir), f'q{i}').result()
                fill()
                g0 = time.perf_counter(); probs = {'core6': [], 'vm': []}; canary = None
                for kn in res['knees']:
                    if uid in canaries:  # recorded, not asserted: the caller compares predictions with the accepted-cache reference
                        canary = dict(variants=res['variants'], px_exact=kn['px_sha256'] == canaries[uid]['pixel_array_sha256'], meta_exact=kn['meta_sha256'] == canaries[uid]['meta_array_sha256'])
                    images = np.load(kn['images']); meta = np.load(kn['meta'])
                    futs = {arm: threads.submit(predict, arm, images, meta) for arm in ['core6', 'vm']}
                    for arm in futs: probs[arm].append(futs[arm].result())
                    del images, meta
                for arm in probs: out[arm][i] = np.max(np.stack(probs[arm]), axis=0)
                gpu_seconds += time.perf_counter() - g0
                records.append({'id': uid, 'index': i, 't_done': time.perf_counter() - t_start, 'canary': canary, 'bilateral': res['bilateral'], 'prep_seconds': res['prep_seconds'], 'stage_seconds': res.get('stage_seconds'), 'gpu_seconds': time.perf_counter() - g0, 'wait_seconds': g0 - tick,
                                'knees': [{k: v for k, v in kn.items() if k not in ('images', 'meta')} for kn in res['knees']]})
            except Exception as ex:
                fill()
                failures.append({'id': uid, 'error': repr(ex)[:500], 'trace': traceback.format_exc()[-1500:]})
            finally:
                for f in qdir.glob(f'q{i}_*'): f.unlink(missing_ok=True)
            if (i + 1) % 25 == 0 or i + 1 == len(ids):
                save(a.out / 'H6_FAST_PROGRESS.json', {'checked': i + 1, 'total': len(ids), 'failures': len(failures), 'elapsed': time.perf_counter() - t_start, 'at_unix': time.time()})
            print('H6_STUDY_DONE', i + 1, len(ids), uid, round(time.perf_counter() - tick, 3), 'fail' if failures and failures[-1]['id'] == uid else 'ok', flush=True)
    consumer['stop'] = True; pool.shutdown(wait=False, cancel_futures=True)
    for arm, p in out.items():
        with (a.out / (arm + '_raw_probabilities.csv')).open('w', newline='') as f:
            w = csv.writer(f); w.writerow(columns); w.writerows([uid, *p[i]] for i, uid in enumerate(ids))
    canary_pass = [r['id'] for r in records if r.get('canary') and r['canary']['px_exact'] and r['canary']['meta_exact'] and r['canary']['variants'] == 1]
    audit = {'status': 'DONE', 'studies': len(ids), 'failures': failures, 'n_failures': len(failures), 'workers': a.workers, 'pool_restarts': restarts,
             'model_load_seconds': load_seconds, 'total_seconds': time.perf_counter() - t_start, 'gpu_seconds': gpu_seconds, 'records': records,
             'public_test3_canaries_passed': canary_pass, 'release_sha256': sha(HERE / 'RELEASE.json'), 'precision': 'FP32_noTF32_noautocast' if a.precision == 'fp32' else 'FP16_autocast_fp32_fallback', 'fp32_reruns': len(fp32_reruns), 'weights': weights_used, 'prefetch': dict(pf_stats, ahead=a.prefetch_ahead, threads=a.prefetch_threads), 'at_unix': time.time()}
    save(a.out / 'H6_FAST_AUDIT.json', audit); print('H6_FAST_DONE', len(ids), len(failures), round(audit['total_seconds'], 1), flush=True)
if __name__ == '__main__':
    main()
