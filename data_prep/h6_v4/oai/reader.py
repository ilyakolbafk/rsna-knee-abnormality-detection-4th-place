"""Bounded per-worker NPZ reader; public exact window sampling, no training launch."""
from common import *
import ast, math
from collections import OrderedDict
import numpy as np
import torch

path=HANDOFF/'reference/public_v4/compact_train.py'
tree=ast.parse(path.read_text());scope={'np':np,'math':math}
for name in ['covering_centers','phys_step','window_rows']:
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
window_rows=scope['window_rows']

class CaseReader:
    def __init__(self,root,max_bytes=256*2**20):
        self.root=Path(root);self.max_bytes=max_bytes;self._pid=None;self._cache=OrderedDict();self._bytes=0
    def __getstate__(self):return {'root':self.root,'max_bytes':self.max_bytes,'_pid':None,'_cache':OrderedDict(),'_bytes':0}
    def read(self,case):
        if self._pid!=os.getpid():self._cache.clear();self._bytes=0;self._pid=os.getpid()
        if case in self._cache:self._cache.move_to_end(case);return self._cache[case]
        with np.load(self.root/f'{case}.npz',allow_pickle=False) as z:
            px=z['px'];meta=z['meta']
            if str(z['orientation_state'])!='canonical_L_v8_3':raise ValueError('WRONG_ORIENTATION_STATE')
            recipe=str(z['recipe_sha256'])
        if px.dtype!=np.uint8 or px.ndim!=3 or px.shape[1:]!=(384,384) or meta.shape!=(len(px),12) or meta.dtype!=np.float32:
            raise ValueError('INVALID_CACHE_SCHEMA')
        audit=json.loads((self.root/f'{case}.audit.json').read_text())
        if recipe!=audit['recipe_sha256']:raise ValueError('RECIPE_BINDING_CONFLICT')
        if len(px)!=audit['rows']:raise ValueError('CURSOR_ROW_COUNT_CONFLICT')
        n=px.nbytes+meta.nbytes
        while self._cache and self._bytes+n>self.max_bytes:
            _,value=self._cache.popitem(last=False);self._bytes-=value[0].nbytes+value[1].nbytes
        value=(px,meta,audit)
        if n<=self.max_bytes:self._cache[case]=value;self._bytes+=n
        return value
    def windows(self,case,mode='train',seed=20261021,n_win=176,extent_meta=False):
        px,m,a=self.read(case)
        blocks=[{**b,'views':[(0,v,start) for v,start in b['views']]} for b in a['audit']['blocks']]
        rng=np.random.default_rng(seed) if mode=='train' else None
        cand=window_rows(blocks,lambda f,r:m[r],mode,3,eval_stride=1,cover_stride=3,
                         phys_mm=0.,rng=rng,slice_jitter=.25,train_sampling='cover')
        if mode=='train' and len(cand)>n_win:cand=[cand[i] for i in np.sort(rng.choice(len(cand),n_win,replace=False))]
        rows=np.asarray(cand,np.int64)
        images=np.stack([px[rows[:,2+d]] for d in range(3)],axis=1)
        meta=m[rows[:,1]][:,[1,3,4,5,6,7]]
        if extent_meta:meta=np.concatenate([meta,m[rows[:,1]][:,[0,2,10]]],axis=1)
        return images,meta,rows
