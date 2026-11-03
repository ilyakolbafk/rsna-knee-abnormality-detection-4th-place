"""Verify the delivered public slab rule against complete native provenance.

This is source-index/metadata consistency QA, not a claim of pixel rerender or
basis-source-code parity. It permits only the exact public suffix-tail rule.
"""
import numpy as np
from rsna_public_npz import need

def validate(block,native):
    n=int(native['native_shape'][0]);positions=np.asarray(native['positions'],np.float64)
    frames=native['source_frames'];thin=block['thin_stack'];groups=block['source_groups']
    need(n>0 and positions.shape==(n,) and np.isfinite(positions).all() and len(frames)==n,'SLAB_NATIVE_COUNT')
    need(all(f['final_index']==j and f['crc_verified'] is True and f['decode_valid'] is True for j,f in enumerate(frames)),'SLAB_NATIVE_INDEX')
    need(block['native_shape']==native['native_shape'] and block['native_positions_mm']==native['positions'],'SLAB_NATIVE_BINDING')
    need(block['native_affines_col_row_origin_lps_mm']==native['affines_col_row_origin_lps_mm'] and block['basis_audit']==native['basis_audit'],'SLAB_NATIVE_BASIS_BINDING')
    order=np.argsort(positions,kind='stable');ordered=positions[order]
    spacing=float(np.median(np.diff(ordered))) if n>1 else 0.
    need(thin['target_slab_mm']==3.5 and thin['threshold_mm']==1.75,'SLAB_PUBLIC_CONSTANTS')
    need(thin['formula']=='n//g + int(n%g >= max(1,g//2)); only apply if n_new>=2' and thin['averaging']=='rint(mean(float32),axis=0).astype(native_uint16)','SLAB_PUBLIC_RULE')
    need(np.isclose(thin['original_spacing_mm'],spacing,rtol=1e-12,atol=1e-9),'SLAB_ORIGINAL_SPACING')
    g,new,applied=1,n,False
    if 0<spacing<1.75:
        g=max(1,int(round(3.5/spacing)));new=n//g+int(n%g>=max(1,g//2));applied=new>=2
    expected=[order[j*g:min(n,(j+1)*g)].tolist() for j in range(new)] if applied else [[int(j)] for j in order]
    used=min(n,new*g) if applied else n;tail=order[used:].tolist()
    need(thin['g']==g and thin['n_new_rule_result']==new and thin['applied'] is applied,'SLAB_PUBLIC_GROUPING')
    need(block['physical_order_native_indices']==order.tolist() and [q['native_indices'] for q in groups]==expected,'SLAB_EXACT_NATIVE_GROUPS')
    need(block['n_slices']==len(expected) and block['representative_native_indices']==[q[0] for q in expected],'SLAB_REPRESENTATIVES')
    need(thin['tail_drop_native_indices']==tail and thin['tail_drop_source_frames']==[frames[j] for j in tail],'SLAB_EXACT_DECLARED_SUFFIX')
    flat=[j for group in expected for j in group]+tail
    need(len(flat)==len(set(flat))==n and set(flat)==set(range(n)),'SLAB_NO_MISSING_OR_DUPLICATED_SOURCE')
    means=[]
    for record,indices in zip(groups,expected):
        need(record['source_frames']==[frames[j] for j in indices],'SLAB_SOURCE_FRAME_BINDING')
        need(np.allclose(record['uniform_weights'],np.full(len(indices),1/len(indices)),rtol=0,atol=1e-12),'SLAB_UNIFORM_WEIGHTS')
        mean=float(positions[indices].mean());means.append(mean)
        need(np.isclose(record['physical_position_mean_mm'],mean,rtol=1e-12,atol=1e-9),'SLAB_PHYSICAL_MEAN')
    need(np.allclose(block['positions_mm'],means,rtol=1e-12,atol=1e-9),'SLAB_STORED_POSITIONS')
    result_spacing=float(np.median(np.diff(means))) if len(means)>1 else spacing
    need(np.isclose(thin['resulting_spacing_mm'],result_spacing,rtol=1e-12,atol=1e-9),'SLAB_RESULT_SPACING')
    return {'native_frames':n,'stored_slices':len(groups),'tail_frames':len(tail),'source_SOPs':{f['SOPInstanceUID'] for f in frames},'tail':tail}
