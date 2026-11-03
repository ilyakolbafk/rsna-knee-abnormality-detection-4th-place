"""Declared own-frame sagittal reformat used only for anatomical localization."""
from common import *
import numpy as np
import build_case as base

def localize(member,native,geometry,work,registration,runtime,aligned):
    aff=np.asarray(geometry['affines_col_row_origin_lps_mm'],float);n,h,w=native['native_shape']
    corners=np.asarray([aff[z]@np.asarray([x,y,1.]) for z in [0,n-1] for x in [0,w-1] for y in [0,h-1]])
    low,high=corners.min(0),corners.max(0);center=(low+high)/2
    span=min(30.,float((high[0]-low[0])*.3));xs=np.linspace(center[0]+span,center[0]-span,21)
    ys=np.arange(low[1]+1.,high[1]-1.,1.);zs=np.arange(high[2]-1.,low[2]+1.,-1.)
    assert min(len(ys),len(zs))>=64,'INSUFFICIENT_OWN_FRAME_REFORMAT_FIELD'
    yy,zz=np.meshgrid(ys,zs);values=[];masks=[]
    with runtime.sampling_session(runtime.NativeSamplingCache(ram_budget_bytes=0)):
        for x in xs:
            points=np.stack([np.full_like(yy,x),yy,zz],axis=-1).reshape(-1,3)
            v,valid=registration.v1.sample_native(native,geometry,points)
            values.append(v.reshape(yy.shape));masks.append(valid.reshape(yy.shape))
    values=np.stack(values);valid=np.all(np.stack(masks),axis=0)
    top,left,bottom,right=0,0,len(zs),len(ys)
    while not valid[top:bottom,left:right].all():
        assert min(bottom-top,right-left)>64,'NO_COMPLETE_VALID_SAGITTAL_REFORMAT_RECTANGLE'
        roi=~valid[top:bottom,left:right]
        edges=[roi[0].mean(),roi[-1].mean(),roi[:,0].mean(),roi[:,-1].mean()];worst=int(np.argmax(edges))
        if worst==0:top+=1
        elif worst==1:bottom-=1
        elif worst==2:left+=1
        else:right-=1
    assert np.isfinite(values[:,top:bottom,left:right]).all()
    volume=np.rint(np.clip(values[:,top:bottom,left:right],0,65535)).astype(np.uint16)
    derived_aff=np.asarray([np.column_stack([[0.,1.,0.],[0.,0.,-1.],[float(x),float(ys[left]),float(zs[top])]]) for x in xs])
    quality=aligned(derived_aff);assert quality['aligned'],'DERIVED_SAGITTAL_GEOMETRY'
    path=work/f"{member['family']}.own_frame_sagittal_localizer.npy";np.save(path,volume,allow_pickle=False)
    descriptor={**native,'path':str(path),'native_shape':list(volume.shape),'plane':'Sagittal','spacing_mm':[[1.,1.]]*len(volume),
        'representation':'derived_sagittal_localization_only','actual_source_plane':member['plane']}
    geo={**geometry,'affines_col_row_origin_lps_mm':derived_aff.tolist(),'quality':quality}
    anchor=base.MemoryAnchorBank(base.MODEL,{member['sid']:geo}).record(member['source_study_uid'],{'series':[descriptor]})
    assert str(anchor.get('method','')).startswith('skm_tea_joint') and anchor['joint_lps'] is not None,'NO_SKM_JOINT_ON_OWN_FRAME_SAGITTAL_REFORMAT'
    evidence={'policy':'own_frame_sagittal_reformat_for_localization_only','source_family':member['family'],'actual_source_series_uid':member['sid'],
        'actual_source_study_uid':member['source_study_uid'],'actual_frame_uid':member['frames'][0]['frame_uid'],
        'source_native_sha256':sha(native['path']),'derived_native_sha256':sha(path),'derived_shape':list(volume.shape),
        'derived_affines_lps':derived_aff.tolist(),'interpolation':'unchanged original registration sample_native trilinear sampler',
        'spacing_yz_mm':1.,'common_all_slice_valid_rectangle':[left,top,right,bottom],'invalid_or_padded_samples_consumed':False,
        'raw_render_pixels_resampled':False,'cross_frame_anchor_transport':False,'source_ids_modified':False,
        'model_source':'same original SKM-TEA model and AnchorBank candidate gates on declared sagittal representation',
        'clinical_accuracy_claim':False}
    return anchor,evidence
