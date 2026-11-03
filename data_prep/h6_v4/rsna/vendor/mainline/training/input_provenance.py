"""Geometry and source identities for the actual native-cache model inputs.

Pixel coordinates are integer *centres*, in (column, row) order. Native means
the canonical full-resolution cache, not the pre-orientation DICOM matrix.
``output_to_native`` is a pull map, valid where ``input_valid_mask`` is true.
The mask excludes letterbox and augmentation padding/border replication.
``source_crop_fov_mm`` is the original chosen crop (height, width); after an
augmentation, the visible footprint is defined by the map and valid mask.
No image interpolation is reversed by these maps: downsampling still loses
information and a mapped point describes its sampling footprint's centre.
``display_window_uint16`` records the rendering window before augmentation;
``intensity_gain`` separately tracks the subsequent gain and clipping operation.
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class ViewGeometry:
    native_shape: tuple
    spacing: tuple
    size: int
    crop_xyxy: tuple
    valid_xyxy: tuple

    @property
    def crop_shape(self):
        x0,y0,x1,y1=self.crop_xyxy
        return y1-y0,x1-x0

    @property
    def resized_shape(self):
        x0,y0,x1,y1=self.valid_xyxy
        return y1-y0,x1-x0

    @property
    def extent(self):
        ch,cw=self.crop_shape;sy,sx=self.spacing
        return max(ch*sy,cw*sx)

    def receipt(self):
        x,y,_,_=self.crop_xyxy;px,py,_,_=self.valid_xyxy
        ch,cw=self.crop_shape;oh,ow=self.resized_shape;sy,sx=self.spacing
        pull=np.array([[cw/ow,0.,x-.5+(.5-px)*cw/ow],
                       [0.,ch/oh,y-.5+(.5-py)*ch/oh],[0.,0.,1.]],np.float64)
        valid=np.zeros((1,self.size,self.size),bool)
        valid[:,py:py+oh,px:px+ow]=True
        return {
            'output_to_native':pull,
            'native_spacing_mm':np.array([sy,sx],np.float64),
            'native_crop_xyxy':np.array(self.crop_xyxy,np.int64),
            'rendered_valid_xyxy':np.array(self.valid_xyxy,np.int64),
            'valid_image_xyxy':np.array(self.valid_xyxy,np.int64),
            'source_crop_fov_mm':np.array([ch*sy,cw*sx],np.float64),
            'output_spacing_mm':np.array([ch*sy/oh,cw*sx/ow],np.float64),
            'input_valid_mask':valid,
        }


def view_geometry(native_shape,spacing,output_size,*,crop_mm=None,center=(.5,.5)):
    """One crop/rounding/letterbox calculation shared by pixels and masks."""
    h,w=native_shape;sy,sx=map(float,spacing);size=int(output_size)
    if min(h,w,size)<1 or not np.isfinite([sy,sx]).all() or min(sy,sx)<=0:
        raise ValueError('invalid image geometry')
    ch,cw=h,w;y=x=0
    if crop_mm is not None:
        if not np.isfinite(crop_mm) or crop_mm<=0:raise ValueError('invalid crop extent')
        extent=min(float(crop_mm),h*sy,w*sx)
        ch=min(h,max(1,int(round(extent/sy))));cw=min(w,max(1,int(round(extent/sx))))
        y=int(np.clip(round(center[0]*(h-1)-(ch-1)/2),0,h-ch))
        x=int(np.clip(round(center[1]*(w-1)-(cw-1)/2),0,w-cw))
    field_y,field_x=ch*sy,cw*sx;extent=max(field_y,field_x)
    oh=max(1,round(size*field_y/extent));ow=max(1,round(size*field_x/extent))
    py=(size-oh)//2;px=(size-ow)//2
    return ViewGeometry((h,w),(sy,sx),size,(x,y,x+cw,y+ch),(px,py,px+ow,py+oh))


def source_record(series):
    """Indices address this record's files and canonical native array directly."""
    files=list(series.get('files',[]));positions=list(series['positions'])
    if files and len(files)!=len(positions):raise ValueError('native file/position mapping is incomplete')
    shape=series.get('native_shape')
    return {'SeriesInstanceUID':series['SeriesInstanceUID'],'cache_path':series['path'],
            'files':files,'file_mapping_known':bool(files),'positions_mm':positions,
            'native_shape':list(shape) if shape is not None else None,
            'pixel_coordinate_space':'canonical_native_cache',
            'position_convention':'unmodified manifest slice-normal projection in mm; not the right-knee ordering sign',
            'plane':series.get('plane'),'side':series.get('side')}


def window_provenance(series,indices,receipt,affines=None):
    indices=np.asarray(indices,np.int64)
    if indices.shape!=(3,) or np.any(indices<0) or np.any(indices>=len(series['positions'])):
        raise ValueError('invalid source-channel indices')
    repeat=np.array([value in indices[:i] for i,value in enumerate(indices)],bool)
    return {**receipt,'channel_native_index':indices.copy(),
            'channel_position_mm':np.asarray(series['positions'],np.float64)[indices],
            'channel_is_repeat':repeat,
            'channel_native_to_lps':np.asarray(affines,np.float64)[indices] if affines is not None else np.zeros((3,3,3),np.float64),
            'channel_geometry_known':np.full(3,affines is not None,bool)}


_SHAPES={
    'channel_native_index':((3,),np.int64), 'channel_position_mm':((3,),np.float64),
    'channel_is_repeat':((3,),bool), 'channel_native_to_lps':((3,3,3),np.float64),
    'channel_geometry_known':((3,),bool), 'output_to_native':((3,3),np.float64),
    'native_spacing_mm':((2,),np.float64), 'native_crop_xyxy':((4,),np.int64),
    'rendered_valid_xyxy':((4,),np.int64), 'valid_image_xyxy':((4,),np.int64),
    'source_crop_fov_mm':((2,),np.float64), 'output_spacing_mm':((2,),np.float64),
    'display_window_uint16':((2,),np.float64),
    'intensity_gain':((),np.float32),
    'intensity_curve_knots':((4,2),np.float64), 'intensity_curve_count':((),np.int64),
    'intensity_mapping_id':((),np.int64),
}
PROVENANCE_KEYS=(*_SHAPES,'input_valid_mask')


def stack_provenance(rows,size):
    shapes={**_SHAPES,'input_valid_mask':((1,size,size),bool)}
    return {key:torch.from_numpy(np.stack([r[key] for r in rows]) if rows else np.zeros((0,*shape),dtype))
            for key,(shape,dtype) in shapes.items()}


def pixel_pull_from_theta(theta,height,width):
    """affine_grid(align_corners=False): new output centre -> old centre.

    Also handles flips and shears; positive image rotation need not be inferred
    from an angle because this composes the exact matrix used by grid_sample.
    """
    theta=torch.as_tensor(theta)
    affine=torch.eye(3,dtype=theta.dtype,device=theta.device);affine[:2]=theta
    pixel_to_normalized=theta.new_tensor([[2/width,0.,1/width-1],
                                        [0.,2/height,1/height-1],[0.,0.,1.]])
    normalized_to_pixel=theta.new_tensor([[width/2,0.,(width-1)/2],
                                        [0.,height/2,(height-1)/2],[0.,0.,1.]])
    return normalized_to_pixel@affine@pixel_to_normalized


def update_augmented_provenance(batch,b,selection,theta):
    """Update metadata with the same pull affine as the image augmentation."""
    if 'output_to_native' not in batch:return
    h,w=batch['images'].shape[-2:]
    matrix=batch['output_to_native'][b,selection]
    pull=pixel_pull_from_theta(theta.to(device=matrix.device,dtype=matrix.dtype),h,w)
    matrix=matrix@pull
    batch['output_to_native'][b,selection]=matrix
    spacing=batch['native_spacing_mm'][b,selection]
    physical=matrix[:,:2,:2]*spacing.flip(-1)[:,:,None]
    batch['output_spacing_mm'][b,selection]=torch.linalg.vector_norm(physical,dim=1).flip(-1)
    support=batch['input_valid_mask'][b,selection].float()
    mask_theta=theta.to(device=support.device,dtype=support.dtype)
    grid=F.affine_grid(mask_theta[None].expand(len(support),-1,-1),support.shape,align_corners=False)
    # A border-replicated value is observable but is not a new native sample.
    valid=F.grid_sample(support,grid,mode='bilinear',padding_mode='zeros',align_corners=False)>=1-1e-6
    batch['input_valid_mask'][b,selection]=valid
    xs=torch.arange(w,device=valid.device)[None,:];ys=torch.arange(h,device=valid.device)[None,:]
    vx=valid[:,0].any(-2);vy=valid[:,0].any(-1)
    x0=torch.where(vx,xs,w).amin(-1);x1=torch.where(vx,xs+1,0).amax(-1)
    y0=torch.where(vy,ys,h).amin(-1);y1=torch.where(vy,ys+1,0).amax(-1)
    bounds=torch.stack([x0,y0,x1,y1],-1)
    bounds[~valid.flatten(1).any(-1)]=0
    batch['valid_image_xyxy'][b,selection]=bounds
