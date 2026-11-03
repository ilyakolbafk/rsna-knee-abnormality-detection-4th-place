from common import *
from reader import CaseReader
import numpy as np,torch,tempfile,shutil

CASES=['OAI0001','OAI0008','OMV00_9444196_R','OMV00_9220603_L','M6B0001','M6000002']
class Probe(torch.utils.data.Dataset):
    def __init__(self):self.reader=CaseReader(RUN/'cache',max_bytes=128*2**20)
    def __len__(self):return len(CASES)
    def __getitem__(self,i):
        x,m,rows=self.reader.windows(CASES[i],seed=20261021)
        return {'case_id':CASES[i],'pid':os.getpid(),'px':hashlib.sha256(x.tobytes()).hexdigest(),'meta':hashlib.sha256(m.tobytes()).hexdigest(),'rows':hashlib.sha256(rows.tobytes()).hexdigest(),'shape':list(x.shape),'cached_bytes':self.reader._bytes}

if __name__=='__main__':
    p=Probe();expected={x['case_id']:x for x in [p[i] for i in range(len(p))]}
    loader=torch.utils.data.DataLoader(Probe(),batch_size=None,num_workers=2,prefetch_factor=1,multiprocessing_context='spawn')
    observed=list(loader);pids=set()
    for o in observed:
        e=expected[o['case_id']]
        for key in ['px','meta','rows']:assert o[key]==e[key]
        assert o['cached_bytes']<=128*2**20;pids.add(o['pid'])
    assert len(pids)==2
    temp=Path(tempfile.mkdtemp(prefix='reader-corruption-',dir=RUN/'qa'))
    try:
        shutil.copyfile(RUN/'cache/OAI0001.audit.json',temp/'OAI0001.audit.json')
        (temp/'OAI0001.npz').write_bytes((RUN/'cache/OAI0001.npz').read_bytes()[:2048])
        rejected=False
        try:CaseReader(temp).read('OAI0001')
        except Exception:rejected=True
        assert rejected
        (RUN/'cache/OAI0001.npz.partial').write_bytes(b'uncommitted interrupted output')
        assert CaseReader(RUN/'cache').read('OAI0001')[0].shape[0]==647
        (RUN/'cache/OAI0001.npz.partial').unlink()
    finally:shutil.rmtree(temp)
    result={'status':'PASS','worker_count':2,'prefetch_factor':1,'worker_local_cache_max_bytes':128*2**20,
      'multiworker_pixels_metadata_window_hash_parity':True,'corrupt_output_rejected':True,
      'interrupted_partial_not_used':True,'cases':CASES,'worker_pids':sorted(pids),'model_training_launched':False}
    dump(RUN/'qa/LOADER_SMOKE.json',result);print(json.dumps(result))
