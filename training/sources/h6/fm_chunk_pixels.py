"""Lazy lossless FM pixels; immutable destination stat binding on every access.

Historical full-SHA is admitted separately through the explicit sameboot/ctime
continuity policy. This reader never hashes the image corpus during training.
"""
from pathlib import Path
from collections import OrderedDict
from bisect import bisect_right
import os,json
import numpy as np
from fm_delta_codec import decode,CODEC


def require(ok,reason):
    if not ok:raise ValueError('FM_CHUNK:'+reason)


class ChunkArray:
    def __init__(self,descriptor,cache_bytes=128*2**20,max_cache_volumes=2):
        self.d=descriptor;self.root=Path(descriptor['root']).resolve()
        self.spec=json.loads(self.path('pixel_store.json').read_text())
        require(self.spec['format']=='fm384_lossless_volume_chunks_v1' and self.spec['codec']==CODEC and self.spec['dtype']=='uint8','FORMAT')
        self.shape=tuple(self.spec['shape']);self.dtype=np.dtype('uint8');self.ndim=3
        require(len(self.shape)==3 and self.shape[0]>0 and self.shape[1:]==(384,384),'SHAPE')
        self.chunks=self.spec['chunks'];self.starts=[];pos=0;names=set()
        for c in self.chunks:
            require(c['start']==pos and c['shape'][1:]==[384,384] and 0<c['shape'][0]<=1536,'CHUNK_ROW_COVERAGE')
            require(c['path']=='chunks/'+c['stem']+'.dxz' and c['path'] not in names,'CHUNK_PATH_OR_DUPLICATE')
            names.add(c['path']);self.starts.append(pos);pos+=c['shape'][0]
            require(c['path'] in self.d['files'],'UNBOUND_PATH')
            require(self.d['files'][c['path']]['sha256']==c['compressed_sha256'],'ACCEPTED_SHA_BINDING')
            self.path(c['path'])
        require(pos==self.shape[0] and names=={r for r in self.d['files'] if r.startswith('chunks/')},'EXACT_CHUNK_SET')
        require(cache_bytes>0 and max_cache_volumes>0,'CACHE_LIMIT')
        self.limit=int(cache_bytes);self.max_volumes=int(max_cache_volumes);self.cache=OrderedDict();self.used=0;self.pid=os.getpid()

    def path(self,rel):
        p=self.root/rel
        require(not Path(rel).is_absolute() and '..' not in Path(rel).parts and not p.is_symlink() and p.resolve().is_relative_to(self.root),'PATH_ESCAPE_OR_SYMLINK')
        z=self.d['files'].get(rel);require(z is not None,'UNBOUND_PATH')
        s=p.stat(follow_symlinks=False)
        require(p.is_file() and [s.st_size,s.st_mtime_ns,s.st_ctime_ns]==z['stat'] and s.st_dev==z['dev'] and s.st_ino==z['inode'],'DESTINATION_STAT_CHANGED')
        return p

    def __getstate__(self):
        state=self.__dict__.copy();state.update(cache=OrderedDict(),used=0,pid=None);return state

    def block(self,k):
        if self.pid!=os.getpid():self.cache.clear();self.used=0;self.pid=os.getpid()
        c=self.chunks[k];p=self.path(c['path'])  # Includes cachehits: modified source is always refused.
        if k in self.cache:self.cache.move_to_end(k);return self.cache[k]
        with p.open('rb') as f:
            before=os.fstat(f.fileno());z=self.d['files'][c['path']]
            require([before.st_size,before.st_mtime_ns,before.st_ctime_ns]==z['stat'] and (before.st_dev,before.st_ino)==(z['dev'],z['inode']),'OPEN_STAT_CHANGED')
            wire=f.read();after=os.fstat(f.fileno());require(before==after,'MUTATED_DURING_READ')
        self.path(c['path']);require(len(wire)==c['compressed_bytes'],'COMPRESSED_SIZE')
        a=decode(wire,c['shape'])  # Exact pinned producer codec; bounded zlib+checksum, no preprocessing.
        require(a.dtype==np.uint8 and a.shape==tuple(c['shape']),'DECODE_SCHEMA')
        a=np.frombuffer(a.tobytes(),dtype=np.uint8).reshape(a.shape)  # Backing bytes prevent setflags(write=True).
        self.path(c['path'])
        if a.nbytes<=self.limit:
            while self.cache and (self.used+a.nbytes>self.limit or len(self.cache)>=self.max_volumes):
                _,old=self.cache.popitem(last=False);self.used-=old.nbytes
            self.cache[k]=a;self.used+=a.nbytes
        return a

    def __len__(self):return self.shape[0]

    def __getitem__(self,key):
        if isinstance(key,tuple):
            row,*rest=key;a=self[row]
            return a[tuple(rest)] if isinstance(row,(int,np.integer)) else a[(slice(None),*rest)]
        if isinstance(key,(int,np.integer)):
            row=int(key);row=row if row>=0 else row+len(self)
            if not 0<=row<len(self):raise IndexError(row)
            k=bisect_right(self.starts,row)-1;return self.block(k)[row-self.starts[k]]
        if isinstance(key,slice):ids=range(*key.indices(len(self)))
        else:
            ids=np.asarray(key);require(ids.ndim==1 and np.issubdtype(ids.dtype,np.integer),'INDEX_TYPE');ids=ids.tolist()
        rows=[self[int(i)] for i in ids]
        return np.stack(rows) if rows else np.empty((0,384,384),np.uint8)
