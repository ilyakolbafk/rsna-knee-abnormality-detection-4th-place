"""Drop-in Studies interface for the frozen H5/H6 window contract; one shared corpus."""
from reader import CaseReader
import torch

class StudiesNPZ(torch.utils.data.Dataset):
    def __init__(self,root,uids,n_win=176,mode='train',seed=0,*,extent_meta=False,cache_bytes=128*2**20):
        if mode not in ('train','eval'):raise ValueError('Unsupported mode')
        self.uids=list(uids);self.n_win=int(n_win);self.mode=mode;self.seed=int(seed);self.epoch=0
        self.extent_meta=extent_meta;self.reader=CaseReader(root,max_bytes=cache_bytes)
    def __len__(self):return len(self.uids)
    def __getitem__(self,index):
        nonce=None
        if isinstance(index,tuple):index,nonce=index
        seed=self.seed+1000003*self.epoch+index if nonce is None else [self.seed,self.epoch,index,int(nonce)]
        x,m,_=self.reader.windows(self.uids[index],mode=self.mode,seed=seed,n_win=self.n_win,extent_meta=self.extent_meta)
        return torch.from_numpy(x),torch.from_numpy(m),index
