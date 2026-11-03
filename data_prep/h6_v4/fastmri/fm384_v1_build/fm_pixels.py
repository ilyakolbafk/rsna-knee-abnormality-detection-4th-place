"""Declared source-RSS→display→three screen fields; no anatomical M/L guess."""
import numpy as np
import cv2
from fm_geometry import view_geometry,coronal_bilateral_views
cv2.setNumThreads(1)

def edge_affine(g):
 x,y,_,_=g.crop_xyxy;px,py,_,_=g.valid_xyxy;ch,cw=g.crop_shape;oh,ow=g.resized_shape
 return np.array([[ow/cw,0,px-x*ow/cw],[0,oh/ch,py-y*oh/ch],[0,0,1]],np.float64)
def transform_box(source_xywh,c,g):
 x,y,w,h=map(float,source_xywh);assert np.isfinite([x,y,w,h]).all() and w>0 and h>0
 m=edge_affine(g);raw=np.array([m[0,0]*x+m[0,2],m[1,1]*y+m[1,2],m[0,0]*(x+w)+m[0,2],m[1,1]*(y+h)+m[1,2]])
 a0=w*h*m[0,0]*m[1,1];vx0,vy0,vx1,vy1=g.valid_xyxy
 clipped=np.array([max(raw[0],vx0),max(raw[1],vy0),min(raw[2],vx1),min(raw[3],vy1)])
 if clipped[2]<=clipped[0] or clipped[3]<=clipped[1]:return None,{'a0':float(a0),'retained_fraction':0.,'mapped_unclipped_xyxy':raw.tolist()}
 area=float((clipped[2]-clipped[0])*(clipped[3]-clipped[1]));assert area<=a0*(1+1e-12)
 return [int(c),*map(float,clipped),float(a0)],{'a0':float(a0),'retained_fraction':area/a0,'mapped_unclipped_xyxy':raw.tolist()}
def geometries(shape,spacing_xy_mm,size=384):
 dx,dy=spacing_xy_mm;assert min(shape[0]*dy,shape[1]*dx)>=100.,'SOURCE_FIELD_BELOW100MM'
 vs=coronal_bilateral_views(shape,[dy,dx],size,local_mm=100.);assert [x[0] for x in vs]==[0,2,3],'THREE_DISTINCT_SCREEN_FIELDS_REQUIRED'
 return {view:view_geometry(shape,[dy,dx],size,crop_mm=mm,center=center) for view,mm,center in vs}
def geometry_receipt(g):
 r=g.receipt();r.pop('input_valid_mask');r={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in r.items()}
 display_to_view=edge_affine(g);h=g.native_shape[0];flip_edges=np.array([[1,0,0],[0,-1,h],[0,0,1]],np.float64);flip_centers=np.array([[1,0,0],[0,-1,h-1],[0,0,1]],np.float64)
 output_to_display_centers=np.asarray(r.pop('output_to_native'))
 r.update(annotation_display_to_view_xy_edges=display_to_view.tolist(),view_to_annotation_display_xy_edges=np.linalg.inv(display_to_view).tolist(),raw_rss_to_view_xy_edges=(display_to_view@flip_edges).tolist(),view_to_raw_rss_xy_edges=(flip_edges@np.linalg.inv(display_to_view)).tolist(),output_to_display_pixel_centers=output_to_display_centers.tolist(),output_to_raw_rss_pixel_centers=(flip_centers@output_to_display_centers).tolist(),view_extent_mm=g.extent,annotation_coordinate_frame='fastMRI_plus_display_after_one_flipud',raw_coordinate_frame='stored_unmodified_rss.f32.npy',view_coordinate_frame='384_view_edge_coordinates; pixel-center pulls named separately')
 return r
def render_display_slice(display,g):
 x,y,x1,y1=g.crop_xyxy;px,py,_,_=g.valid_xyxy;oh,ow=g.resized_shape
 v=cv2.resize(np.asarray(display[y:y1,x:x1],np.float32),(ow,oh),interpolation=cv2.INTER_AREA)
 out=np.zeros((g.size,g.size),np.uint8);out[py:py+oh,px:px+ow]=np.rint(np.clip(v,0,1)*255).astype(np.uint8)
 return out
def normalize_volume(raw):
 assert raw.dtype==np.float32 and raw.ndim==3 and np.isfinite(raw).all() and raw.min()>=0
 q=float(np.percentile(raw,99.9,method='linear'));scale=q if q>0 else 1.
 return scale,{'policy':'volume_p99.9_linear_before_resize_clip01_round_to_even_uint8','percentile99_9':q,'denominator':scale,'numpy_percentile_method':'linear','resize':'OpenCV INTER_AREA float32','quantization':'numpy rint(clip01*255) to uint8','flipud_applied_here':True,'slice_order':'source unchanged'}
def display_slice(raw,z,scale):
 return np.clip(np.asarray(raw[z],np.float32)[::-1,:]/scale,0,1)
