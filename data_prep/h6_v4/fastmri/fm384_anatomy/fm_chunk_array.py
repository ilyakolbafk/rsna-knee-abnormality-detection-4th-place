"""Explicit read-only lossless volume chunks; this object is not a NumPy mmap."""
from pathlib import Path
from collections import OrderedDict
from bisect import bisect_right
import os,json,hashlib
import numpy as np
from fm_delta_codec import decode,CODEC
class ChunkArray:
 def __init__(self,root,max_cache_volumes=2):
  self.root=Path(root).resolve();self.spec=json.loads((self.root/'pixel_store.json').read_text());assert self.spec['codec']==CODEC and self.spec['dtype']=='uint8';self.shape=tuple(self.spec['shape']);assert len(self.shape)==3 and self.shape[1:]==(384,384);self.dtype=np.dtype('uint8');self.ndim=3;self.chunks=self.spec['chunks'];self.starts=[];pos=0
  for c in self.chunks:
   assert c['start']==pos and c['shape'][1:]==[384,384] and c['shape'][0]>0;self.starts.append(pos);pos+=c['shape'][0];p=(self.root/c['path']).resolve();assert p.parent==self.root/'chunks' and not Path(c['path']).is_absolute(),'INVALID_PIXEL_CHUNK_PATH'
  assert pos==self.shape[0];self.cache=OrderedDict();self.max_cache_volumes=int(max_cache_volumes);assert self.max_cache_volumes>=1;self.cache_pid=os.getpid()
 def __len__(self):return self.shape[0]
 def block(self,k):
  if self.cache_pid!=os.getpid():self.cache=OrderedDict();self.cache_pid=os.getpid()
  if k in self.cache:a=self.cache.pop(k);self.cache[k]=a;return a
  c=self.chunks[k];wire=(self.root/c['path']).read_bytes();assert hashlib.sha256(wire).hexdigest()==c['compressed_sha256'],'PIXEL_CHUNK_COMPRESSED_SHA_MISMATCH';a=decode(wire,c['shape']);assert hashlib.sha256(a.tobytes()).hexdigest()==c['uncompressed_sha256'],'PIXEL_CHUNK_UNCOMPRESSED_SHA_MISMATCH';a=np.frombuffer(a.tobytes(),dtype=np.uint8).reshape(a.shape);self.cache[k]=a
  while len(self.cache)>self.max_cache_volumes:self.cache.popitem(last=False)
  return a
 def __getitem__(self,key):
  if isinstance(key,tuple):
   row,*rest=key;a=self[row];return a[tuple(rest)] if isinstance(row,(int,np.integer)) else a[(slice(None),*rest)]
  if isinstance(key,(int,np.integer)):
   row=int(key);row=row if row>=0 else row+self.shape[0]
   if not 0<=row<self.shape[0]:raise IndexError(row)
   k=bisect_right(self.starts,row)-1;return self.block(k)[row-self.starts[k]]
  if isinstance(key,slice):ids=range(*key.indices(len(self)))
  else:
   ids=np.asarray(key);assert ids.ndim==1 and np.issubdtype(ids.dtype,np.integer);ids=ids.tolist()
  rows=[self[int(i)] for i in ids];return np.stack(rows) if rows else np.empty((0,384,384),np.uint8)
