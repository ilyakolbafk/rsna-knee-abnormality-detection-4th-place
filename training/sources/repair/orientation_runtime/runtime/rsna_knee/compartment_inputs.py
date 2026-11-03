"""Native physical compartment crops; paired labels are prepared separately.

An orthonormal basis in the actual image plane preserves a 90mm square under
oblique acquisition. Both IOP vectors enter the centre and output grid. Sagittal
neighbours are clamped at the compartment boundary, never admitted from the notch.
"""
from pathlib import Path
import hashlib
import cv2
import numpy as np
from .canonical_inputs import transform_spec
from .central_sampling import allocate_centers

COMPARTMENTS = ('medial', 'lateral')

def physical_crop(d, plane, side, compartment, joint_z, notch_x=None, *, size=384, crop_mm=90., limits=None):
    if side not in ('L','R') or compartment not in COMPARTMENTS:
        raise ValueError('A known recorded side and compartment are required')
    if plane not in ('Sagittal','Coronal'):
        raise ValueError('Only tibiofemoral sagittal/coronal crops are supported')
    first,second=np.asarray(d['iop'],float).reshape(2,3)
    rs,cs=d['spacing']; origin=np.asarray(d['ipp'],float)
    normal=np.cross(first,second);normal/=np.linalg.norm(normal)
    up=np.array([0.,0.,1.])-normal*normal[2]
    if np.linalg.norm(up)<.2:raise ValueError('Plane has no usable superior axis')
    up/=np.linalg.norm(up)
    horizontal=np.cross(up,normal); horizontal/=np.linalg.norm(horizontal)
    axis=1 if plane=='Sagittal' else 0
    if horizontal[axis]<0:horizontal*=-1
    if abs(horizontal[axis])<.2:raise ValueError('Degenerate in-plane geometry')
    img=d['pixels'];h,w=img.shape
    if plane=='Sagittal':
        rr,cc=np.nonzero(img>0)
        r=float(np.median(rr)) if len(rr) else (h-1)/2
        c=float(np.median(cc)) if len(cc) else (w-1)/2
        target_axis=float((origin+first*cs*c+second*rs*r)[1])
    else:
        if notch_x is None:
            notch_x=float((origin+first*cs*(w-1)/2+second*rs*(h-1)/2)[0])
        lateral_sign=1 if side=='L' else -1
        comp_sign=lateral_sign if compartment=='lateral' else -lateral_sign
        # 25mm centre offset gives 20mm overlap and70mm outward extent in a
        # non-oblique plane. Preserve isotropic physical scale when tilted.
        target_axis=float(notch_x+comp_sign*25.)
        horizontal*=lateral_sign
        if compartment=='lateral':horizontal*=-1
    A=np.array([[first[axis]*cs,second[axis]*rs],[first[2]*cs,second[2]*rs]])
    if abs(np.linalg.det(A))<1e-8:raise ValueError('Cannot solve physical crop centre')
    centre_col,centre_row=np.linalg.solve(A,[target_axis-origin[axis],joint_z-origin[2]])
    centre=origin+first*cs*centre_col+second*rs*centre_row
    offsets=(np.arange(size,dtype=np.float32)+.5-size/2)*(crop_mm/size)
    xx,yy=np.meshgrid(offsets,offsets)
    delta=horizontal[:,None,None]*xx-up[:,None,None]*yy
    map_x=(centre_col+np.einsum('i,ihw->hw',first,delta)/cs).astype(np.float32)
    map_y=(centre_row+np.einsum('i,ihw->hw',second,delta)/rs).astype(np.float32)
    source=img.astype(np.float32)
    if limits is not None:
        lo,hi=limits
        if not np.isfinite([lo,hi]).all() or hi<=lo:raise ValueError('Invalid frozen series normalization')
        source=np.clip((source-lo)/(hi-lo),0,1)
    crop=cv2.remap(source,map_x,map_y,interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    if limits is not None:crop=np.rint(crop*255).clip(0,255).astype(np.uint8)
    corners=[]
    for x,y in [(-crop_mm/2,-crop_mm/2),(crop_mm/2,-crop_mm/2),(crop_mm/2,crop_mm/2),(-crop_mm/2,crop_mm/2)]:
        delta=horizontal*x-up*y
        corners.append([float(centre_col+first@delta/cs),float(centre_row+second@delta/rs)])
    valid=(map_x>=0)&(map_x<=w-1)&(map_y>=0)&(map_y<=h-1)
    receipt={'plane':plane,'compartment':compartment,'side':side,'centre_patient':centre.tolist(),
             'centre_pixel':[float(centre_col),float(centre_row)],'horizontal_unit':horizontal.tolist(),'up_unit':up.tolist(),
             'corners_native_pixels':corners,'physical_size_mm':crop_mm,'output_pixels':size,
             'mm_per_pixel':crop_mm/size,'source_fov_fraction':float(valid.mean()),
             'compartment_reflection':plane=='Coronal' and compartment=='lateral','normalization_limits':limits}
    assert abs(horizontal@up)<1e-6 and abs(np.linalg.norm(horizontal)-1)<1e-6
    assert abs(centre[2]-joint_z)<1e-6
    return crop,receipt

def build_compartments(series,packet,locator, *, size=384):
    if not locator['crop_eligible']:raise ValueError(locator['crop_exclusion'])
    side=locator['side'];m=packet['metadata'];byfile={p['source_file']:p for p in m['planes']}
    slots=[None]*4
    for sid,s in m['series'].items():
        slot=s.get('legacy_slot')
        if slot is not None and 0<=slot<4 and sid in series:slots[slot]=(sid,s)
    images=np.zeros((2,48,3,size,size),np.uint8);mask=np.zeros((2,48),bool);features=np.zeros((2,48,7),np.float32)
    features[:,:,3]=1
    records=[];sample_rows=[]
    for ci,comp in enumerate(COMPARTMENTS):
        features[ci,:,6]=ci;pool=[];positions=[]
        for item in slots:
            if item is None:pool.append([]);positions.append(None);continue
            sid,s=item;frames=series[sid];normal=np.cross(frames[0]['iop'][:3],frames[0]['iop'][3:])
            projected=np.array([d['ipp']@normal for d in frames]);positions.append(projected)
            if m['legacy_slots'][s['legacy_slot']]['plane']=='Sagittal':
                xs=np.array([(d['ipp']+d['iop'][:3]*d['spacing'][1]*(d['cols']-1)/2+d['iop'][3:]*d['spacing'][0]*(d['rows']-1)/2)[0] for d in frames])
                lat=xs if side=='L' else -xs;span=float(np.ptp(lat))
                lp=(lat-lat.min())/span if span>1e-6 else np.full(len(lat),.5)
                pool.append(np.flatnonzero(lp<=.42 if comp=='medial' else lp>=.58).tolist())
            else:pool.append(list(range(len(frames))))
        counts=allocate_centers([len(p) for p in pool],48,[14,14,10,10]);at=0
        for slot,n in enumerate(counts):
            if not n:continue
            sid,s=slots[slot];frames=series[sid];available=pool[slot]
            chosen=np.asarray(available)[np.rint(np.linspace(0,len(available)-1,n)).astype(int)]
            plane=m['legacy_slots'][slot]['plane'];lo=s.get('normalization_low');hi=s.get('normalization_high')
            if lo is None or hi is None:raise ValueError('Missing frozen AUX60 per-series normalization')
            pos=positions[slot];pmin,pmax=pos.min(),pos.max();cache={}
            for k in chosen:
                d=frames[k];spec=transform_spec(d['iop'],plane,side)
                if spec is None:raise ValueError('Unsupported image orientation')
                indices=[k-1,k,k+1]
                indices=[j if j in available else int(k) for j in indices]
                constrained=indices!=[k-1,k,k+1]
                if spec['reverse_channels']:indices=indices[::-1]
                provenance=[]
                for channel,j in enumerate(indices):
                    if j not in cache:
                        cache[j]=physical_crop(frames[j],plane,side,comp,locator['crop_z_'+comp],locator.get('notch_x'),size=size,limits=[lo,hi])
                    image,receipt=cache[j];images[ci,at,channel]=image
                    name=Path(frames[j]['source_file']).name;p=byfile[name]
                    provenance.append({'source_file':name,'SOPInstanceUID':p['SOPInstanceUID'],'source_sha256':p['source_sha256'],'bank_index':p['bank_index'],'geometry':receipt})
                offset=float(pos[k]-(pmin+pmax)/2)
                if spec['reverse_channels']:offset*=-1
                if pmax>pmin:features[ci,at,:3]=[2*offset/(pmax-pmin),np.clip(offset/65,-4,4),1]
                fs=s.get('source_inventory',{}).get('fat_suppression');features[ci,at,3:6]=0;features[ci,at,4+int(fs) if fs in [0,1] else 3]=1
                mask[ci,at]=True
                records.append({'compartment':comp,'token':at,'slot':slot,'plane':plane,'SeriesInstanceUID':sid,'source_index':int(k),'series_count':len(frames),
                                'channel_boundary_replicated':constrained,'channel_reversed':spec['reverse_channels'],'channels':provenance})
                at+=1
        if not at:raise ValueError('No valid compartment windows')
        sample_rows.append({'uid':packet['uid'],'compartment':comp,'windows':at,'slot_counts':counts.tolist(),'boundary_replicated_windows':sum(x['channel_boundary_replicated'] for x in records if x['compartment']==comp)})
    assert images.shape==(2,48,3,size,size) and mask.any(1).all() and np.isfinite(features).all()
    return {'pixels':images,'mask':mask,'meta':features}, {'samples':sample_rows,'windows':records,'normalization':'frozen AUX60 per-series low/high; no per-window refit','geometry':'orthonormal physical in-plane square; joint centre solved with both IOP vectors'}
