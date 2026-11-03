"""Long-lived resources and opt-in timings for exact anatomy construction.

Only immutable file reads and frozen model instances are reused. Model batches,
tensor dtypes, interpolation and quality thresholds remain in the existing
anatomy functions. Profiling synchronizes CUDA at explicitly labelled boundaries;
normal production calls do not add synchronization.
"""
from __future__ import annotations

from collections import OrderedDict,defaultdict
from contextlib import contextmanager,nullcontext
from contextvars import ContextVar
from pathlib import Path
import os
import tempfile
import time
import zipfile

import numpy as np
import torch

from .common import file_hash

_ACTIVE=ContextVar('anatomy_runtime',default=None)


def identity(path):
    path=Path(path).resolve();st=path.stat()
    return (str(path),st.st_size,st.st_mtime_ns,st.st_ctime_ns)


class StageTimings:
    def __init__(self,enabled=False,device='cuda'):
        self.enabled=enabled;self.device=str(device);self.seconds=defaultdict(float);self.calls=defaultdict(int)
        self.trace=[]

    def sync(self):
        if self.device.startswith('cuda') and torch.cuda.is_available():
            start=time.perf_counter();torch.cuda.synchronize(self.device)
            self.seconds['explicit_cuda_synchronize']+=time.perf_counter()-start
            self.calls['explicit_cuda_synchronize']+=1

    @contextmanager
    def stage(self,name,*,gpu=False):
        if not self.enabled:
            yield
            return
        if gpu:self.sync()
        start=time.perf_counter()
        try:yield
        finally:
            if gpu:self.sync()
            elapsed=time.perf_counter()-start
            self.seconds[name]+=elapsed;self.calls[name]+=1
            self.trace.append({'stage':name,'start':start,'seconds':elapsed,'cuda_synchronized':gpu})

    def snapshot(self):
        return {'seconds':dict(self.seconds),'calls':dict(self.calls),
                'timing_scope':'Inclusive stage timers; nested stages and synchronization must not be summed as disjoint wall time',
                'profiling_synchronizes_cuda':self.enabled,'gpu_input_wait':'not measured; no input-stall conclusion from utilization alone'}


class _TimedWriter:
    def __init__(self,stream):self.stream=stream;self.write_seconds=0.;self.bytes_written=0
    def write(self,value):
        start=time.perf_counter()
        try:return self.stream.write(value)
        finally:self.write_seconds+=time.perf_counter()-start;self.bytes_written+=len(value)
    def flush(self):
        start=time.perf_counter()
        try:return self.stream.flush()
        finally:self.write_seconds+=time.perf_counter()-start
    def __getattr__(self,name):return getattr(self.stream,name)


def save_npz(path,payload,*,compression_level=1,profile=None):
    """Identical ndarray payloads, adjustable lossless DEFLATE effort, atomic file."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    handle,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    start=time.perf_counter();written=0;write_seconds=0.
    try:
        with os.fdopen(handle,'w+b',buffering=8*2**20) as raw:
            stream=_TimedWriter(raw)
            with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=int(compression_level),allowZip64=True) as archive:
                for key,value in payload.items():
                    array=np.asanyarray(value)
                    if array.dtype.hasobject:raise ValueError('anatomy outputs cannot contain pickled object arrays')
                    with archive.open(key+'.npy','w',force_zip64=True) as target:
                        np.lib.format.write_array(target,array,allow_pickle=False)
            stream.flush();write_seconds=stream.write_seconds;written=stream.bytes_written
        for attempt in range(7):
            try:
                Path(name).replace(path)
                break
            except PermissionError:
                if attempt==6:raise
                time.sleep(min(.05*2**attempt,.8))
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
    elapsed=time.perf_counter()-start
    if profile is not None and profile.enabled:
        profile.seconds['npz_compress_serialize']+=max(0.,elapsed-write_seconds)
        profile.seconds['npz_buffered_write_calls']+=write_seconds
        profile.calls['npz_compress_serialize']+=1;profile.calls['npz_buffered_write_calls']+=1
    return {'seconds':elapsed,'buffered_write_seconds':write_seconds,'logical_written_bytes':written,'file_bytes':path.stat().st_size,
            'compression_level':int(compression_level),'write_timing_limit':'Buffered writes, not physical device flush latency; no fsync added'}


class AnatomyRuntime:
    """One session across batches, with explicit model/file invalidation."""
    def __init__(self,weights,device='cuda',assets_dir=None,*,profile=False,
                 compression_level=1,volume_cache_bytes=512*2**20):
        self.weights=Path(weights).resolve();self.device=str(device)
        self.assets=Path(assets_dir or (self.weights.parent.parent/'kneexnet' if self.weights.parent.name=='skm_tea' else self.weights.parent/'kneexnet')).resolve()
        self.weight_paths=(self.weights,self.assets/'localizer.pt',self.assets/'best_model_256_noise_notebook_5_0.15.pth')
        self.session_weight_identities=tuple(identity(p) for p in self.weight_paths)
        self.loaded_weight_identities=None;self._models=None;self.hashes={};self.volumes=OrderedDict()
        self.volume_cache_bytes=int(volume_cache_bytes);self.cached_volume_bytes=0
        self.compression_level=int(compression_level);self.profile=StageTimings(profile,self.device)
        self.stats=defaultdict(int)

    def validate(self,weights=None,device=None,assets_dir=None):
        if weights is not None and Path(weights).resolve()!=self.weights:raise ValueError('runtime belongs to different segmentation weights')
        if device is not None and str(device)!=self.device:raise ValueError('runtime belongs to a different device')
        if assets_dir is not None and Path(assets_dir).resolve()!=self.assets:raise ValueError('runtime belongs to different auxiliary weights')
        if tuple(identity(p) for p in self.weight_paths)!=(self.loaded_weight_identities or self.session_weight_identities):
            raise ValueError('frozen anatomy model weights changed; create a fresh runtime before reuse')

    def models(self):
        self.validate()
        if self._models is None:
            from .anatomy import load_segmenter
            from .anatomy_quality import load_localizer,load_independent_segmenter
            state=tuple(identity(p) for p in self.weight_paths)
            with self.profile.stage('model_load.skm',gpu=True):first=load_segmenter(self.weights,self.device)
            with self.profile.stage('model_load.localizer',gpu=True):localizer=load_localizer(self.weight_paths[1])
            with self.profile.stage('model_load.independent',gpu=True):second=load_independent_segmenter(self.weight_paths[2],self.device)
            if tuple(identity(p) for p in self.weight_paths)!=state:raise ValueError('anatomy weights changed during model loading')
            self._models=(first,localizer,second);self.loaded_weight_identities=state;self.stats['model_load_sets']+=1
        else:self.stats['model_set_hits']+=1
        return self._models

    def file_hash(self,path):
        key=identity(path)
        if key not in self.hashes:
            with self.profile.stage('file_hash.read'):value=file_hash(path)
            if identity(path)!=key:raise ValueError('input changed while hashing: '+str(path))
            self.hashes[key]=value;self.stats['hash_misses']+=1;self.stats['hashed_bytes']+=key[1]
        else:self.stats['hash_hits']+=1
        return self.hashes[key]

    def volume(self,path):
        key=identity(path)
        if key in self.volumes:
            self.stats['volume_hits']+=1;self.volumes.move_to_end(key);return self.volumes[key]
        with self.profile.stage('native_mmap_open'):value=np.load(path,mmap_mode='r',allow_pickle=False)
        self.stats['volume_misses']+=1
        if value.nbytes<=self.volume_cache_bytes:
            while self.volumes and self.cached_volume_bytes+value.nbytes>self.volume_cache_bytes:
                _,old=self.volumes.popitem(last=False);self.cached_volume_bytes-=old.nbytes
                # Do not explicitly close a map still referenced by a caller.
            self.volumes[key]=value;self.cached_volume_bytes+=value.nbytes
            self.stats['maximum_cached_volume_bytes']=max(self.stats['maximum_cached_volume_bytes'],self.cached_volume_bytes)
        return value

    @contextmanager
    def activate(self):
        self.validate();token=_ACTIVE.set(self)
        try:yield self
        finally:_ACTIVE.reset(token)

    def save(self,path,payload):
        with self.profile.stage('npz_save_total'):
            result=save_npz(path,payload,compression_level=self.compression_level,profile=self.profile)
        self.stats['npz_file_bytes']+=result['file_bytes'];self.stats['npz_files']+=1
        return result

    def snapshot(self):
        return {'profile':self.profile.snapshot(),'resource_counters':dict(self.stats),
                'volume_cache_budget_bytes':self.volume_cache_bytes,'cached_volume_bytes':self.cached_volume_bytes,
                'compression_level':self.compression_level,'model_batches_or_dtypes_changed':False}

    def close(self):
        """Release this session's own model/map references at a safe boundary."""
        import gc
        self.volumes.clear();self.cached_volume_bytes=0;self._models=None
        self.loaded_weight_identities=None;self.hashes.clear();gc.collect()
        if self.device.startswith('cuda') and torch.cuda.is_available():torch.cuda.empty_cache()


def active_runtime():return _ACTIVE.get()


def stage(name,*,gpu=False):
    runtime=_ACTIVE.get()
    return runtime.profile.stage(name,gpu=gpu) if runtime is not None else nullcontext()


def native_volume(path):
    runtime=_ACTIVE.get()
    return runtime.volume(path) if runtime is not None else np.load(path,mmap_mode='r',allow_pickle=False)
