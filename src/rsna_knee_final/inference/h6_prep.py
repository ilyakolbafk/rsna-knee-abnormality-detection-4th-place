"""22.10 (Claude): H6 full-v4 producer for one study, run in a spawned worker process. The body is infer_full.py's per-study
producer block (release f7d79463) line for line: producer.native -> original anatomy VNet on cuda:0 FP32 -> render_on_demand v4 ->
window_rows(eval, stride 1) -> centre crop 352 + meta projection. Model inference stays in the parent process."""
import os
os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
from pathlib import Path
import gc, sys, tempfile, time
import importlib, importlib.machinery
# Python >= 3.12 removed FileFinder.find_module; the vendored monai 0.6.0 (anatomy VNet) still calls importer.find_module(name).load_module(name)
# right after import_module(name) in load_submodules. Shim: return a loader whose load_module is import_module (module already imported).
if not hasattr(importlib.machinery.FileFinder, 'find_module'):
    class _LegacyLoader:
        def __init__(self, name): self.name = name
        def load_module(self, name): return importlib.import_module(name)
    importlib.machinery.FileFinder.find_module = lambda self, name, path=None: _LegacyLoader(name)
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'h6')); sys.path.insert(0, str(HERE))
from sampling import window_rows
import rebuild_v4 as producer

# Profiling wrappers (same computations, only timers): DICOM read/decode, laterality resolver, orientation (incl. quantization).
PROF = {}
def _timed(key, fn):
    def wrapper(*args, **kwargs):
        t = time.perf_counter()
        try: return fn(*args, **kwargs)
        finally: PROF[key] = PROF.get(key, 0.0) + time.perf_counter() - t
    return wrapper
class _TimedSourceStudy(producer.SourceStudy):
    def __init__(self, *args, **kwargs):
        t = time.perf_counter(); super().__init__(*args, **kwargs); PROF['read'] = PROF.get('read', 0.0) + time.perf_counter() - t
    def side_inputs(self, resolver):
        t = time.perf_counter()
        try: return super().side_inputs(resolver)
        finally: PROF['resolver'] = PROF.get('resolver', 0.0) + time.perf_counter() - t
producer.SourceStudy = _TimedSourceStudy
producer.C.orient = _timed('orient', producer.C.orient)
producer.C.quantize_series = _timed('quantize', producer.C.quantize_series)

def init():
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    dev = os.environ.get('H6_RESOLVER_DEVICE', 'cpu')  # original: LateralityResolver(ORIENT, device='cpu')
    if dev != 'cpu':
        producer.C.resources()
        producer.C.RESOLVER = producer.C.LateralityResolver(producer.C.ORIENT, device=dev)

def prepare(uid, study_rows, series_dir, qdir, tag=None):
    tag = tag or uid
    PROF.clear(); tick = time.perf_counter(); knees = []; tm = {'native': 0.0, 'anatomy': 0.0, 'render': 0.0, 'windows': 0.0, 'save': 0.0}
    with tempfile.TemporaryDirectory(prefix='h6-v4-', dir='/tmp') as scratch:
        t = time.perf_counter()
        variants, decision = producer.native(uid, study_rows, Path(series_dir), Path(scratch) / 'native')
        tm['native'] += time.perf_counter() - t
        for k, (study, geometry, route) in enumerate(variants):
            t = time.perf_counter()
            dev = os.environ.get('H6_ANATOMY_DEVICE', 'cuda:0')  # infer_full: cuda:0; accepted CPU canaries (rebuild_v4.main): cpu
            anatomical, _ = producer.C.resources(); anatomical.to(dev).eval()
            bank = producer.C.MemoryAnchorBank(anatomical, geometry); bank.dev = dev
            with torch.inference_mode(): anchors = bank.record(study['StudyInstanceUID'], study)
            if anchors['joint_lps'] is None: raise ValueError('NO_ANATOMICAL_OR_GEOMETRY_ANCHOR')
            tm['anatomy'] += time.perf_counter() - t; t = time.perf_counter()
            rendered = producer.R.render_on_demand(study, anchors, {'components_root': str(producer.COMPONENTS), 'image_size': 384, 'views': 'v4', 'local_mm': 100., 'intensity_policy': 'series_companded'})
            px, meta = rendered['px'], rendered['meta']; pxhash = producer.array_sha(px); metahash = producer.array_sha(meta)
            tm['render'] += time.perf_counter() - t; t = time.perf_counter()
            blocks = [{**b, 'views': [(0, v, start) for v, start in b['views']]} for b in rendered['blocks']]
            refs = np.asarray(window_rows(blocks, lambda file, row: meta[row], 'eval', 3, eval_stride=1, cover_stride=3, phys_mm=3.5, rng=None, slice_jitter=.25, train_sampling='cover'), np.int64)
            assert len(refs) > 0
            images = np.ascontiguousarray(px[refs[:, 2:]][..., 16:368, 16:368]); projected = meta[refs[:, 1]][:, [1, 3, 4, 5, 6, 7, 0, 2, 10]].copy()
            assert images.shape == (len(refs), 3, 352, 352) and projected.shape == (len(refs), 9)
            tm['windows'] += time.perf_counter() - t; t = time.perf_counter()
            ip = Path(qdir) / f'{tag}_{k}_images.npy'; mp = Path(qdir) / f'{tag}_{k}_meta.npy'
            np.save(ip, images); np.save(mp, projected)
            tm['save'] += time.perf_counter() - t
            knees.append({'images': str(ip), 'meta': str(mp), 'route': route, 'pixel_rows': len(px), 'windows': len(refs), 'px_sha256': pxhash, 'meta_sha256': metahash, 'anatomy_method': anchors['method'], 'anatomy_device': dev})
            del images, projected, px, meta, rendered, refs; gc.collect()
        assert len(variants) == (2 if decision['bilateral'] else 1)
    return {'id': uid, 'bilateral': bool(decision['bilateral']), 'variants': len(variants), 'knees': knees, 'prep_seconds': time.perf_counter() - tick, 'stage_seconds': {**tm, **{'native_' + k: v for k, v in PROF.items()}}, 'pid': os.getpid()}
