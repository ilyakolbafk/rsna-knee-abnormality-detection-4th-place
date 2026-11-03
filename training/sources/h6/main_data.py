"""One uniform pooled fullfit dataset; RSNA/OAI VM labels stay in separate heads."""
import numpy as np
import torch
from full_cache_loader import Reader

FIELDS=['source_id','main_probability','main_lr','main_weight','oai_vm_probability','oai_vm_lr','oai_vm_weight']

class MainDataset(torch.utils.data.Dataset):
    def __init__(self,manifest,targets,seed,*,mode='train',capacity=176,phys_mm=0.,cache_purpose='training'):
        self.rows=manifest['studies'];self.seed=seed;self.epoch=1;self.mode=mode;self.capacity=capacity;self.phys_mm=phys_mm
        self.readers={k:Reader(d,purpose=cache_purpose) for k,d in manifest['corpora'].items()}
        with np.load(targets,allow_pickle=False) as z:self.targets={k:z[k] for k in z.files}
        self.byid={str(v):i for i,v in enumerate(self.targets['sample_id'])}
        ids=[r['id'] for r in self.rows]
        if len(ids)!=len(set(ids)) or set(ids)!=set(self.byid):raise ValueError('EXACT_TARGET_CACHE_ROSTER')
        gold=set(manifest['Gold_excluded_ids'])
        for r in self.rows:
            if r['corpus']!=('RSNA' if r['source_id']==0 else 'OAI') or r['source_id'] not in [0,1]:raise ValueError('CORPUS_DOMAIN_BINDING')
            canonical=self.readers[r['corpus']].d['studies'][r['cache_key']]
            for field,rowfield in [('sample_id','id'),('parent_id','parent_id'),('report_group','report_group')]:
                if canonical.get(field)!=r.get(rowfield):raise ValueError('PIXEL_LABEL_IDENTITY:'+field)
            if canonical.get('source_id')!=r['source_id']:raise ValueError('PIXEL_SOURCE_IDENTITY')
            if canonical.get('diagnostic_Gold58'):raise ValueError('GOLD_CACHE_ALIAS_IN_FIT')
            if r['id'] in gold or r['parent_id'] in gold or r.get('report_group') in gold:raise ValueError('GOLD_LEAKAGE')
            if r.get('diagnostic_Gold58'):raise ValueError('GOLD_DIAGNOSTIC_IN_FIT')
            if int(self.targets['source_id'][self.byid[r['id']]])!=r['source_id']:raise ValueError('SOURCE_TARGET_BINDING')
    def __len__(self):return len(self.rows)
    def __getitem__(self,item):
        i,nonce=item if isinstance(item,tuple) else (item,item);r=self.rows[i]
        x,m,rows=self.readers[r['corpus']].windows(r['cache_key'],mode=self.mode,seed=self.seed,epoch=self.epoch,index=i,nonce=nonce,capacity=self.capacity,phys_mm=self.phys_mm)
        n=len(x);size=self.capacity if self.mode=='train' else n
        xx=np.zeros((size,3,384,384),np.uint8);mm=np.zeros((size,9),np.float32);mask=np.zeros(size,bool)
        xx[:n]=x;mm[:n]=m;mask[:n]=True
        result={'x':torch.from_numpy(xx),'meta':torch.from_numpy(mm),'mask':torch.from_numpy(mask),'id':r['id']}
        j=self.byid[r['id']]
        result.update({k:torch.as_tensor(self.targets[k][j]) for k in FIELDS})
        return result
