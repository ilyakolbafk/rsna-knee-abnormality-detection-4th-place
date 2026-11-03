"""Strict full-view v4 readers. No legacy bank/refs path or guessed metadata.

Stat bindings originate in accepted destination full-SHA receipts. Images are not
rehashed every epoch; metadata documents remain SHA-checked. Admission of the
whole corpus is separate from per-read schema/identity verification.
"""
from pathlib import Path
from collections import OrderedDict
import os,json,hashlib
import numpy as np
from common import POLICY,ORIENTATION_SPEC
from orientation_policy import require_cache_orientation
from sampling import window_rows


class FullCacheError(ValueError):pass


class Reader:
    def __new__(cls,descriptor=None,cache_bytes=128*2**20,*,diagnostic=False,purpose='training'):
        if cls is Reader and descriptor is not None and descriptor.get('backend')=='rsna_public_npz_collection':
            from rsna_collection import CollectionReader
            return CollectionReader(descriptor,cache_bytes=cache_bytes,diagnostic=diagnostic,purpose=purpose)
        return super().__new__(cls)
    def __init__(self,descriptor,cache_bytes=128*2**20,*,diagnostic=False,purpose='training'):
        if purpose not in ['training','protocol_probe']:raise FullCacheError('READER_PURPOSE')
        self.d=descriptor;self.diagnostic=diagnostic;self.root=Path(descriptor['root']).resolve()
        self.kind=descriptor['backend'];self.limit=cache_bytes;self.pid=None
        self.cache=OrderedDict();self.used=0;self.docs=OrderedDict()
        if self.kind not in ['public_corpus','case_npz','chunked_case']:raise FullCacheError('UNSUPPORTED_FULL_CACHE_SCHEMA')
        if descriptor.get('profile')!='full_v4':raise FullCacheError('FULL_V4_ONLY')
        require_cache_orientation(descriptor,ORIENTATION_SPEC)
        if not descriptor.get('destination_fullsha_proof_sha256'):raise FullCacheError('NO_DESTINATION_FULLSHA_BINDING')

    def __getstate__(self):return {'d':self.d,'diagnostic':self.diagnostic,'root':self.root,'kind':self.kind,'limit':self.limit,'pid':None,'cache':OrderedDict(),'used':0,'docs':OrderedDict()}

    def path(self,rel,metadata=False):
        p=(self.root/rel).resolve()
        if not p.is_relative_to(self.root):raise FullCacheError('CACHE_PATH_ESCAPE')
        expected=self.d['files'].get(rel)
        if expected is None:raise FullCacheError('FILE_NOT_IN_ACCEPTED_MANIFEST:'+rel)
        st=p.stat()
        if [st.st_size,st.st_mtime_ns,st.st_ctime_ns]!=expected['stat']:raise FullCacheError('CACHE_STAT_CHANGED:'+rel)
        if metadata and hashlib.sha256(p.read_bytes()).hexdigest()!=expected['sha256']:raise FullCacheError('CACHE_METADATA_SHA_CHANGED:'+rel)
        return p

    def document(self,rel):
        self._reset();self.path(rel)
        if rel in self.docs:self.docs.move_to_end(rel);return self.docs[rel]
        value=json.loads(self.path(rel,metadata=True).read_text());self.docs[rel]=value
        while len(self.docs)>8:self.docs.popitem(last=False)
        return value

    def _reset(self):
        if self.pid!=os.getpid():self.cache.clear();self.docs.clear();self.used=0;self.pid=os.getpid()

    def arrays(self,rel):
        self._reset();path=self.path(rel)
        if rel in self.cache:self.cache.move_to_end(rel);return self.cache[rel]
        with np.load(path,allow_pickle=False) as z:
            px,meta=z['px'],z['meta']
            if self.kind=='case_npz':
                keys=[k for k,v in self.d['studies'].items() if v['pixels']==rel]
                if len(keys)!=1 or str(z['case_id'])!=keys[0]:raise FullCacheError('NPZ_SAMPLE_IDENTITY')
                if self.d.get('collection'):
                    from oai_collection import config_for_case
                    config=config_for_case(self,keys[0]);recipe=config['semantic_sha256']
                    if str(z['roster_sha256'])!=config['roster_sha256'] or str(z['format'])!=config['format']:raise FullCacheError('NPZ_COLLECTION_ROSTER_FORMAT')
                else:recipe=self.d['recipe_sha256']
                if str(z['orientation_state'])!='canonical_L_v8_3' or str(z['recipe_sha256'])!=recipe:raise FullCacheError('NPZ_IDENTITY')
        if px.dtype!=np.uint8 or px.ndim!=3 or px.shape[1:]!=(384,384) or meta.dtype!=np.float32 or meta.shape!=(len(px),12) or not np.isfinite(meta).all():raise FullCacheError('FULL_CACHE_ARRAY_SCHEMA')
        amount=px.nbytes+meta.nbytes
        while self.cache and self.used+amount>self.limit:
            _,(x,m)=self.cache.popitem(last=False);self.used-=x.nbytes+m.nbytes
        if amount<=self.limit:self.cache[rel]=(px,meta);self.used+=amount
        return px,meta

    @staticmethod
    def validate_blocks(blocks,meta,rows):
        expected={'Sagittal':{0,1,4,5},'Coronal':{0,2,3},'Axial':{0,1,4,5},0:{0,1,4,5},1:{0,2,3},2:{0,1,4,5}}
        spans=[]
        for b in blocks:
            n=b['n_slices'];views=b['views']
            if not isinstance(n,int) or n<1:raise FullCacheError('NO_PHYSICAL_SLICES')
            starts=[int(v[1]) for v in views]
            plane=b.get('plane',int(meta[starts[0],2]));ids=[int(v[0]) for v in views]
            if plane not in expected or set(ids)!=expected[plane] or len(ids)!=len(set(ids)):raise FullCacheError('FULL_ANATOMICAL_VIEWS_REQUIRED')
            for view,start in views:
                if start<0 or start+n>rows:raise FullCacheError('VIEW_ROW_RANGE')
                selected=meta[start:start+n]
                if not np.all(selected[:,3]==view) or not np.all(selected[:,10]==n):raise FullCacheError('VIEW_METADATA_MISMATCH')
                if not np.array_equal(selected[:,9],np.arange(n,dtype=np.float32)):raise FullCacheError('PHYSICAL_SLICE_ORDER')
                if not np.isfinite(selected).all() or np.any(selected[:,5]<=0) or np.any(selected[:,6]<=0):raise FullCacheError('REAL_PHYSICAL_METADATA_REQUIRED')
                spans.append((start,start+n))
        for a,b in zip(sorted(spans),sorted(spans)[1:]):
            if b[0]<a[1]:raise FullCacheError('OVERLAPPING_VIEW_ROWS')

    def study(self,key):
        info=self.d['studies'][key]
        if info.get('diagnostic_Gold58') and not self.diagnostic:raise FullCacheError('GOLD_IN_FIT_CACHE')
        if self.diagnostic:
            from common import ROOT,read
            if info.get('sample_id',key) not in set(read(ROOT/'assets/GOLD_EXCLUDED_IDS.json')):raise FullCacheError('DIAGNOSTIC_REQUIRES_EXACT_GOLD58')
        if self.kind=='public_corpus':
            index=self.document(self.d['index']);blocks=index['blocks'][key]
            px=np.load(self.path(self.d['pixels']),mmap_mode='r',allow_pickle=False)
            meta=np.load(self.path(self.d['meta']),mmap_mode='r',allow_pickle=False)
            if px.dtype!=np.uint8 or px.ndim!=3 or px.shape[1:]!=(384,384) or meta.dtype!=np.float32 or meta.shape!=(len(px),12):raise FullCacheError('CORPUS_SCHEMA')
            self.validate_blocks(blocks,meta,len(px))
            return blocks,meta,lambda ids:np.asarray(px[ids])
        if self.kind=='case_npz':
            px,meta=self.arrays(info['pixels']);audit=self.document(info['audit'])
            if self.d.get('collection'):
                from oai_collection import validate_case
                config=validate_case(self,key,audit);recipe=config['semantic_sha256']
            else:recipe=self.d['recipe_sha256']
            if audit['case_id']!=key or audit['recipe_sha256']!=recipe or audit['rows']!=len(px) or audit['orientation_state']!='canonical_L_v8_3':raise FullCacheError('CASE_AUDIT_BINDING')
            if any(s.get('orientation_applications')!=1 for s in audit['audit']['lineage']):raise FullCacheError('CANONICAL_NOT_ONCE')
            blocks=audit['audit']['blocks']
            if any(b.get('dropped_tail_source_sops') for b in blocks):raise FullCacheError('SOURCE_SLICES_DROPPED')
            self.validate_blocks(blocks,meta,len(px))
            return blocks,meta,lambda ids:px[ids]
        index=self.document(info['index']);qa=self.document(info['qa'])
        if index['sample']['sample_id']!=key or index['format']!='rsna-v4-v83-lps-chunks-v1' or index['orientation_state']!='canonical_L_v8_3' or qa.get('verified_index_sha256')!=hashlib.sha256(self.path(info['index']).read_bytes()).hexdigest():raise FullCacheError('CHUNK_INDEX_QA_BINDING')
        chunks=index['chunks'];allmeta=[];cursor=0
        for c in chunks:
            if c['start']!=cursor or c['stop']<=cursor or c['stop']-cursor>64:raise FullCacheError('CHUNK_ROW_COVERAGE')
            rel=str(Path(info['index']).parent/c['path']);_,m=self.arrays(rel)
            if len(m)!=c['stop']-c['start']:raise FullCacheError('CHUNK_ROW_COUNT')
            allmeta.append(m);cursor=c['stop']
        meta=np.concatenate(allmeta)
        if cursor!=index['rows']:raise FullCacheError('CHUNK_TOTAL_ROWS')
        blocks=index['blocks'];self.validate_blocks(blocks,meta,cursor)
        def take(ids):
            ids=np.asarray(ids,np.int64);result=np.empty((len(ids),384,384),np.uint8)
            for c in chunks:
                selected=np.flatnonzero((ids>=c['start'])&(ids<c['stop']))
                if len(selected):result[selected]=self.arrays(str(Path(info['index']).parent/c['path']))[0][ids[selected]-c['start']]
            if np.any(ids<0) or np.any(ids>=cursor):raise FullCacheError('CHUNK_BAD_ROW')
            return result
        return blocks,meta,take

    def windows(self,key,*,mode,seed,epoch,index,nonce,capacity=176,phys_mm=0.):
        blocks,meta,take=self.study(key)
        rng=np.random.default_rng([int(seed),int(epoch),int(index),int(nonce)]) if mode=='train' else None
        bb=[{**b,'views':[(0,v,start) for v,start in b['views']]} for b in blocks]
        cand=window_rows(bb,lambda f,r:meta[r],mode,3,eval_stride=1,cover_stride=3,phys_mm=phys_mm,rng=rng,slice_jitter=.25,train_sampling='cover')
        if mode=='train' and len(cand)>capacity:cand=[cand[j] for j in np.sort(rng.choice(len(cand),capacity,replace=False))]
        if not cand:raise FullCacheError('EMPTY_STUDY')
        rows=np.asarray(cand,np.int64);pixels=take(rows[:,2:].reshape(-1)).reshape(len(rows),3,384,384)
        return pixels,meta[rows[:,1]][:,[1,3,4,5,6,7,0,2,10]].copy(),rows
