"""One deterministic knee-wide reflection of gathered canonical token copies.
No side resolver/canonical adapter is called; evaluation bypasses this module.
"""
import csv
from pathlib import Path
import numpy as np
from v83_data import _rng

class MirrorPlaneMap:
    def __init__(self,path):
        self.rows={}
        with Path(path).open() as f:
            for r in csv.DictReader(f):
                sid=str(r['SampleID'])
                if sid in self.rows:raise ValueError('Duplicate mirror SampleID: '+sid)
                counts=[int(r['count_'+str(i)]) for i in range(5)]
                if min(counts)<0 or not 0<sum(counts)<=64:raise ValueError('Bad mirror counts: '+sid)
                self.rows[sid]=np.repeat(np.asarray([0,0,1,1,2],np.int64),counts)
        if len(self.rows)!=4410:raise ValueError('Mirror coverage must be all4410')
    def plane_ids(self,sample_id,valid_count):
        planes=self.rows[str(sample_id)]
        if len(planes)!=valid_count:raise ValueError('Mirror token-count mismatch: '+str(sample_id))
        return planes

def mirror_draw(seed,epoch,sample_id):
    return _rng(seed,epoch,str(sample_id),'joint24_whole_knee_mirror50_v1').random()<0.5

def reflect_tokens(pixels,meta,planes):
    """planes:0Sag,1Cor,2Ax. Input arrays never mutated; center remains channel1."""
    if pixels.ndim!=4 or pixels.shape[1]!=3 or meta.shape!=(len(pixels),6):raise ValueError('Mirror tensor shape')
    planes=np.asarray(planes)
    if planes.shape!=(len(pixels),) or not np.isin(planes,[0,1,2]).all():raise ValueError('Unknown token plane')
    out=np.array(pixels,copy=True);features=np.array(meta,copy=True)
    sag=planes==0;in_plane=~sag
    out[sag]=pixels[sag,::-1,:,:]
    out[in_plane]=pixels[in_plane,:,:,::-1]
    features[sag,:2]*=-1
    return out,features
