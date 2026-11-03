"""Read cache336 extension and construct the revised eight metadata features."""
from pathlib import Path
import numpy as np

META8=('relative_position','offset_mm_over_65','position_valid','fs_unknown',
    'fs_no','fs_yes','is_extra','is_oblique')

def metadata8(meta6,is_extra,is_oblique):
    meta6=np.asarray(meta6,np.float32)
    if meta6.ndim!=2 or meta6.shape[1]!=6:
        raise ValueError('Invalid metadata dimensions')
    return np.column_stack([meta6,
        np.broadcast_to(is_extra,len(meta6)),np.broadcast_to(is_oblique,len(meta6))]).astype(np.float32)

def load_extension(path,sample_id):
    with np.load(Path(path),allow_pickle=False) as z:
        if str(z['format'])!='k24-allseries-ext-v1' or str(z['sample_id'])!=str(sample_id):
            raise ValueError('Extension format/identity mismatch')
        if str(z['orientation_policy'])!='knee-laterality-v8.3' or str(z['orientation_state'])!='canonical_L_v8_3':
            raise ValueError('Extension orientation mismatch')
        if z['native_orientation'].any() or not np.isin(z['orient'],['legacy','v3']).all():
            raise ValueError('Unresolved extension orientation')
        return dict(pixels=z['bank'][z['refs']],mask=z['mask'],meta6=z['meta6'],
            meta8=metadata8(z['meta6'],True,z['is_oblique']),
            series_uid=z['series_uid'],orient=z['orient'])
