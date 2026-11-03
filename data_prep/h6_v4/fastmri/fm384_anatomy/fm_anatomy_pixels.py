"""Relative anatomical RSS adaptation. Physical in-plane coordinates only."""
import numpy as np
from fm_geometry import view_geometry
from fm_pixels import edge_affine,normalize_volume,display_slice,render_display_slice

def reflection(width,flip,*,centers=False):
 return np.array([[-1.,0.,width-1 if centers else width],[0.,1.,0.],[0.,0.,1.]]) if flip else np.eye(3)
def geometries(shape,spacing_xy,anchor,size=384):
 h,w=shape;dx,dy=map(float,spacing_xy);r,c=map(float,anchor)
 assert np.isfinite([r,c,dx,dy]).all() and 0<=r<h and 0<=c<w and min(dx,dy)>0
 assert min(h*dy,w*dx)>=100
 defs={0:(None,(.5,.5)),2:(100.,(r/(h-1),max(0.,min(1.,(c-30/dx)/(w-1))))),3:(100.,(r/(h-1),max(0.,min(1.,(c+30/dx)/(w-1)))))}
 gs={v:view_geometry(shape,[dy,dx],size,crop_mm=mm,center=center) for v,(mm,center) in defs.items()}
 assert len({(g.crop_xyxy,g.valid_xyxy) for g in gs.values()})==3,'DISTINCT_ANATOMICAL_LOCALS_REQUIRED'
 return gs

def source_box_to_view(xywh,klass,g,flip):
 x,y,w,h=map(float,xywh);assert min(w,h)>0 and np.isfinite([x,y,w,h]).all()
 m=edge_affine(g)@reflection(g.native_shape[1],flip)
 pts=np.array([[x,y,1],[x+w,y+h,1]])@m.T;lo=pts[:,:2].min(0);hi=pts[:,:2].max(0);raw=np.r_[lo,hi];a0=float(w*h*abs(np.linalg.det(m[:2,:2])))
 vx,vy,vx1,vy1=g.valid_xyxy;clip=np.array([max(lo[0],vx),max(lo[1],vy),min(hi[0],vx1),min(hi[1],vy1)])
 if clip[2]<=clip[0] or clip[3]<=clip[1]:return None,{'a0':a0,'retained_fraction':0,'unclipped_xyxy':raw.tolist()}
 area=(clip[2]-clip[0])*(clip[3]-clip[1]);assert area<=a0*(1+1e-12)
 return [int(klass),*clip.tolist(),a0],{'a0':a0,'retained_fraction':float(area/a0),'unclipped_xyxy':raw.tolist()}

def geometry_receipt(g,flip,anchor,view):
 r=g.receipt();r.pop('input_valid_mask');r={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in r.items()};h,w=g.native_shape
 E=reflection(w,flip);C=reflection(w,flip,centers=True);F=np.array([[1.,0,0],[0,-1.,h],[0,0,1.]]);Fc=np.array([[1.,0,0],[0,-1.,h-1],[0,0,1.]])
 A=edge_affine(g)@E;pull=np.asarray(r.pop('output_to_native'));x,y,x1,y1=g.crop_xyxy;dx,dy=g.spacing[1],g.spacing[0];desired=[anchor[0],anchor[1]+({0:0,2:-30,3:30}[view])/dx];actual=[(y+y1-1)/2,(x+x1-1)/2]
 r.update(annotation_display_to_view_xy_edges=A.tolist(),view_to_annotation_display_xy_edges=np.linalg.inv(A).tolist(),raw_rss_to_view_xy_edges=(A@F).tolist(),view_to_raw_rss_xy_edges=(F@np.linalg.inv(A)).tolist(),output_to_display_pixel_centers=(C@pull).tolist(),output_to_raw_rss_pixel_centers=(Fc@C@pull).tolist(),canonical_display_to_view_xy_edges=edge_affine(g).tolist(),display_to_canonical_xy_edges=E.tolist(),display_to_canonical_pixel_centers=C.tolist(),anchor_canonical_display_row_col=list(anchor),desired_center_canonical_row_col=desired if view else [(h-1)/2,(w-1)/2],actual_crop_center_canonical_row_col=actual,crop_center_drift_row_col_mm=[(actual[0]-desired[0])*dy,(actual[1]-desired[1])*dx] if view else [0.,0.],source_flipud_count=1,hflip_to_lateral_pluscol=bool(flip),orientation_policy='fm-rss-anatomy-v1',orientation_state='relative_lateral_plus_column_head_up',LPS_or_v8_3_canonical_claimed=False,annotation_coordinate_frame='fastMRI_plus_display_after_one_flipud',source_slice_order='unchanged')
 return r

def render(raw,z,scale,g,flip):
 d=display_slice(raw,z,scale)
 if flip:d=d[:,::-1]
 return render_display_slice(d,g)
