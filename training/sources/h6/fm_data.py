"""Full fm384_v1 three-view data. Candidate pixels never bypass semantics."""
from pathlib import Path
import os,json,hashlib
import numpy as np
import torch
from fm_store import FmStore
from fm_math import FmAuxSampler,fm_epoch_seeds,fm_window_targets,fm_window_image


def bound_files(descriptor):
    root=Path(descriptor['root']).resolve()
    if not descriptor.get('destination_fullsha_proof_sha256'):raise ValueError('FM_NO_DESTINATION_PROOF')
    chunked=descriptor.get('pixel_backend')=='lossless_volume_chunks_v1'
    required={'pixel_store.json','CORE_BINDING.json','PROMOTION_RECEIPT.json'} if chunked else {'px.u8.npy'}
    if not required|{'index.json','boxes.json','manifest.json'}<=set(descriptor['files']):raise ValueError('FM_REQUIRED_STAT_BOUND_FILES')
    for rel,expected in descriptor['files'].items():
        if chunked and (root/rel).is_symlink():raise ValueError('FM_FILE_SYMLINK:'+rel)
        path=(root/rel).resolve()
        if not path.is_relative_to(root):raise ValueError('FM_PATH_ESCAPE')
        st=path.stat()
        if [st.st_size,st.st_mtime_ns,st.st_ctime_ns]!=expected['stat']:raise ValueError('FM_FILE_STAT_CHANGED:'+rel)
        if chunked and (path.is_symlink() or st.st_dev!=expected['dev'] or st.st_ino!=expected['inode']):raise ValueError('FM_FILE_INODE_CHANGED:'+rel)
        if rel.endswith('.json') and hashlib.sha256(path.read_bytes()).hexdigest()!=expected['sha256']:raise ValueError('FM_METADATA_SHA_CHANGED:'+rel)
    if chunked:
        from fm_anatomy_proof import descriptor_guard
        descriptor_guard(descriptor)
    elif descriptor.get('pixel_backend') not in [None,'numpy_mmap']:
        raise ValueError('FM_UNKNOWN_PIXEL_BACKEND')
    return root


class Store(FmStore):
    def __init__(self,descriptor,*,min_volumes=1000):
        root=bound_files(descriptor)
        index=json.loads((root/'index.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
        if index.get('training_ready') is not True or index.get('semantic_gate')!='PASS' or manifest.get('training_ready') is not True:raise ValueError('FM_SEMANTIC_ADMISSION_BLOCKED')
        if descriptor.get('format_profile')!='fm384_v1' or descriptor.get('semantic_qa_verified') is not True:raise ValueError('FM_RECIPE_NOT_ACCEPTED')
        chunked=descriptor.get('pixel_backend')=='lossless_volume_chunks_v1'
        if chunked:
            from fm_chunk_pixels import ChunkArray
            pixels=ChunkArray(descriptor)
        else:pixels=None
        super().__init__(str(root),min_volumes=min_volumes,pixel_array=pixels)
        if any(not np.isfinite(a).all() or np.any(a[:,5]<=0) for a in self.row_boxes.values()):raise ValueError('FM_BOX_FINITE_ORIGINAL_AREA')
        # The public schema establishes boxed/no-boxed. A missing box alone is
        # insufficient evidence that a source volume was reviewed as normal.
        for vi in self.train_neg:
            row=index['volumes'][self.vols[vi]]
            if row.get('reviewed_no_findings') is not True:raise ValueError('FM_UNREVIEWED_NEGATIVE')
        self._px=None;self._pid=None;self.descriptor=descriptor
    def pixels(self):
        if self._pid!=os.getpid() or self._px is None:
            bound_files(self.descriptor)
            if self.descriptor.get('pixel_backend')=='lossless_volume_chunks_v1':
                from fm_chunk_pixels import ChunkArray
                self._px=ChunkArray(self.descriptor)
            else:self._px=np.load(Path(self.dir)/'px.u8.npy',mmap_mode='r',allow_pickle=False)
            self._pid=os.getpid()
        if self.descriptor.get('pixel_backend')!='lossless_volume_chunks_v1':
            st=(Path(self.dir)/'px.u8.npy').stat()
            if [st.st_size,st.st_mtime_ns,st.st_ctime_ns]!=self.descriptor['files']['px.u8.npy']['stat']:raise ValueError('FM_PIXEL_STAT_CHANGED')
        return self._px
    def __getstate__(self):
        state=self.__dict__.copy();state['_px']=None;state['_pid']=None;return state


class EpochDataset(torch.utils.data.Dataset):
    def __init__(self,descriptor,steps,seed,epoch,*,min_volumes=1000):
        self.store=Store(descriptor,min_volumes=min_volumes)
        sampler=FmAuxSampler(self.store,steps,96,352,0,.5,fm_epoch_seeds(seed,0,epoch-1))
        self.specs=np.asarray(list(sampler),dtype=np.float64)
        if self.specs.shape!=(steps,96,10):raise ValueError('FM_EPOCH_SPEC_SHAPE')
        self.digest=hashlib.sha256(self.specs.tobytes()).hexdigest()
    def __len__(self):return len(self.specs)
    def __getitem__(self,index):
        px=self.store.pixels();result=[]
        for spec in self.specs[index]:
            vi,view,center,oy,ox,dy,dx,flip,gamma,contrast=spec
            vi,view,center,oy,ox,dy,dx=map(int,[vi,view,center,oy,ox,dy,dx])
            rows=self.store.window_rows(vi,view,center)
            boxes=self.store.boxes_of(rows)
            dense,presence,_=fm_window_targets(boxes,352,oy,ox,dy,dx,0,bool(flip))
            image=fm_window_image(px,rows,352,oy,ox,dy,dx,0,bool(flip))
            result.append((image,dense,presence,gamma,contrast))
        return tuple(torch.from_numpy(np.stack([row[k] for row in result])) if k<3 else torch.tensor([row[k] for row in result],dtype=torch.float32) for k in range(5))
