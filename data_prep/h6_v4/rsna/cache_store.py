"""Lossless bounded chunks, atomic commits and physical-stride public window sampling."""
from support import *
import numpy as np,math,zipfile
from collections import OrderedDict
scope={'np':np,'math':math}
for name in ['covering_centers','phys_step','window_rows']:
 exact_function(PACKAGE/'reference/compact_train.py',name,scope)
window_rows=scope['window_rows']

class ChunkWriter:
 def __init__(self,folder,binding,max_rows=64):
  self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True);self.binding=binding;self.max_rows=max_rows
  b=self.folder/'BINDING.json'
  if b.exists()and json.loads(b.read_text())!=binding:raise ValueError('EXISTING_BUILD_BINDING_DIFFERS')
  dump(b,binding);self.px=[];self.meta=[];self.chunks=[];self.rows=0
 def add(self,px,meta):
  self.px.append(px);self.meta.append(meta);self.rows+=1
  if len(self.px)>=self.max_rows:self.flush()
 def flush(self):
  if not self.px:return
  px=np.asarray(self.px,np.uint8);meta=np.asarray(self.meta,np.float32)
  ph=hashlib.sha256(px.tobytes()).hexdigest();mh=hashlib.sha256(meta.tobytes()).hexdigest();start=self.rows-len(px)
  name=f'part_{start:07d}_{ph[:12]}_{mh[:12]}.npz';p=self.folder/name
  if p.exists():
   with np.load(p,allow_pickle=False)as z:
    if not np.array_equal(z['px'],px)or not np.array_equal(z['meta'],meta):raise ValueError('EXISTING_PART_ARRAY_MISMATCH')
  else:
   tmp=p.with_suffix('.partial')
   with tmp.open('wb')as f:
    with zipfile.ZipFile(f,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True)as z:
     for k,v in [('px',px),('meta',meta)]:
      with z.open(k+'.npy','w',force_zip64=True)as q:np.lib.format.write_array(q,v,allow_pickle=False)
    f.flush();os.fsync(f.fileno())
   with np.load(tmp,allow_pickle=False)as z:
    if not np.array_equal(z['px'],px)or not np.array_equal(z['meta'],meta):raise ValueError('LOSSLESS_ROUNDTRIP_FAILED')
   os.replace(tmp,p)
  self.chunks.append({'path':name,'start':start,'stop':self.rows,'bytes':p.stat().st_size,'sha256':sha(p),'pixel_sha256':ph,'meta_sha256':mh,'stat':stat_id(p)})
  self.px=[];self.meta=[]
 def finish(self,index):
  self.flush();index.update(format=RECIPE,rows=self.rows,chunks=self.chunks,binding=self.binding,orientation_state='canonical_L_v8_3',phys_mm=3.5,chunk_max_rows=self.max_rows)
  dump(self.folder/'index.json',index)
  fd=os.open(self.folder,os.O_RDONLY)
  try:os.fsync(fd)
  finally:os.close(fd)
  return index

class CacheReader:
 def __init__(self,folder,max_bytes=64*2**20):
  self.folder=Path(folder);self.index=json.loads((self.folder/'index.json').read_text());self.max_bytes=max_bytes;self.cache=OrderedDict();self.used=0
  if self.index['format']!=RECIPE or self.index['orientation_state']!='canonical_L_v8_3':raise ValueError('CACHE_IDENTITY')
  self.meta=np.concatenate([self._chunk(c)[1]for c in self.index['chunks']])
  if self.meta.shape!=(self.index['rows'],12)or not np.isfinite(self.meta).all():raise ValueError('META_SCHEMA')
 def _chunk(self,c):
  path=self.folder/c['path']
  if c['path']in self.cache:self.cache.move_to_end(c['path']);return self.cache[c['path']]
  with np.load(path,allow_pickle=False)as z:px=z['px'];m=z['meta']
  if px.dtype!=np.uint8 or px.shape!=(c['stop']-c['start'],384,384)or m.dtype!=np.float32 or m.shape!=(len(px),12):raise ValueError('CHUNK_SCHEMA')
  n=px.nbytes+m.nbytes
  while self.cache and self.used+n>self.max_bytes:
   _,v=self.cache.popitem(last=False);self.used-=v[0].nbytes+v[1].nbytes
  if n<=self.max_bytes:self.cache[c['path']]=(px,m);self.used+=n
  return px,m
 def row(self,i):
  for c in self.index['chunks']:
   if c['start']<=i<c['stop']:return self._chunk(c)[0][i-c['start']]
  raise IndexError(i)
 def window_rows(self,mode='eval',seed=1,n_win=176):
  rng=np.random.default_rng(seed)if mode=='train'else None
  blocks=[{**b,'views':[(0,v,start)for v,start in b['views']]}for b in self.index['blocks']]
  cand=window_rows(blocks,lambda f,r:self.meta[r],mode,3,eval_stride=1,cover_stride=3,phys_mm=3.5,rng=rng,slice_jitter=.25,train_sampling='cover')
  if mode=='train'and len(cand)>n_win:cand=[cand[i]for i in np.sort(rng.choice(len(cand),n_win,replace=False))]
  return np.asarray(cand,np.int64)
 def iter_windows(self,mode='eval',seed=1,n_win=176,batch=8):
  rows=self.window_rows(mode,seed,n_win)
  for j in range(0,len(rows),batch):
   group=rows[j:j+batch];px=np.stack([np.stack([self.row(int(i))for i in row[2:]])for row in group])
   yield px,self.meta[group[:,1]][:,[1,3,4,5,6,7]],group
 def verify(self):
  seen=0
  for c in self.index['chunks']:
   p=self.folder/c['path']
   if p.stat().st_size!=c['bytes']or sha(p)!=c['sha256']:raise ValueError('CACHE_FULL_SHA_MISMATCH:'+c['path'])
   px,m=self._chunk(c)
   if hashlib.sha256(px.tobytes()).hexdigest()!=c['pixel_sha256']or hashlib.sha256(m.tobytes()).hexdigest()!=c['meta_sha256']:raise ValueError('ARRAY_SHA')
   seen+=len(px)
  if seen!=self.index['rows']:raise ValueError('ROW_COVERAGE')
  for mode in ['train','eval']:
   for px,m,rows in self.iter_windows(mode):
    if px.shape[1:]!=(3,384,384)or m.shape!=(len(px),6):raise ValueError('WINDOW_SCHEMA')
  return {'status':'PASS_CHUNKS_ARRAYS_AND_BOUNDED_LOADER','rows':seen,'chunks':len(self.index['chunks']),'phys_mm':3.5,'loader_cache_budget_bytes':self.max_bytes,'transient_max_chunk_bytes':self.index['chunk_max_rows']*(384*384+12*4)}
