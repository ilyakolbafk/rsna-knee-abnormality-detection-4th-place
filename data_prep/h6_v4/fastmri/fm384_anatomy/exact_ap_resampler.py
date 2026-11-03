"""Same nnUNet separable nearest AP interpolation, evaluating only used source planes.
No installed package edits. All unsupported cases invoke the original implementation.
"""
import numpy as np
from scipy.ndimage import map_coordinates
from skimage.transform import resize

def eligible(data,new_shape,is_seg,axis,order,do_separate_z,order_z):
 try:
  ax=np.asarray(axis).reshape(-1);shape=np.asarray(new_shape)
  valid_axis=ax.size==1 and np.isfinite(ax[0]) and ax[0]==2
  valid_shape=shape.shape==(3,) and np.isfinite(shape).all() and (shape>0).all() and np.equal(shape,np.floor(shape)).all()
 except (TypeError,ValueError):return False
 return isinstance(data,np.ndarray) and data.ndim==4 and data.dtype in (np.dtype('float16'),np.dtype('float32')) and np.isfinite(data).all() and not is_seg and valid_axis and do_separate_z and order_z==0 and order in (1,3) and valid_shape

def resample_exact(data,new_shape,is_seg=False,axis=None,order=3,do_separate_z=False,order_z=0,dtype_out=None,*,original):
 if not eligible(data,new_shape,is_seg,axis,order,do_separate_z,order_z):return original(data,new_shape,is_seg,axis,order,do_separate_z,order_z,dtype_out)
 new_shape=np.asarray(new_shape,dtype=int);old=np.asarray(data.shape[1:])
 if np.array_equal(old,new_shape):return data
 dtype=data.dtype if dtype_out is None else dtype_out;out=np.zeros((data.shape[0],*new_shape),dtype=dtype)
 coords=(float(old[2])/int(new_shape[2]))*(np.arange(new_shape[2],dtype=np.float64)+.5)-.5
 indices=map_coordinates(np.arange(old[2],dtype=np.float64),coords[None],order=0,mode='nearest').astype(np.int64)
 for c in range(data.shape[0]):
  for idx in np.unique(indices):
   # Exact samefloat64 conversion, skimage2D operator/options as upstream.
   plane=resize(data[c,:,:,idx].astype(float,copy=False),new_shape[:2],order=order,mode='edge',anti_aliasing=False)
   for z in np.flatnonzero(indices==idx):out[c,:,:,z]=plane
 return out

def install(prove=None):
 import nnunetv2.preprocessing.resampling.default_resampling as module
 original=module.resample_data_or_seg
 def wrapped(data,new_shape,is_seg=False,axis=None,order=3,do_separate_z=False,order_z=0,dtype_out=None):
  if prove is not None and eligible(data,new_shape,is_seg,axis,order,do_separate_z,order_z):return prove(data,new_shape,is_seg,axis,order,do_separate_z,order_z,dtype_out,original)
  return resample_exact(data,new_shape,is_seg,axis,order,do_separate_z,order_z,dtype_out,original=original)
 module.resample_data_or_seg=wrapped
 return original
