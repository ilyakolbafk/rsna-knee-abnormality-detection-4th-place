"""Lossless uint8 horizontal-delta + stdlib zlib1. Explicit custom reader required."""
import zlib,hashlib
import numpy as np
CODEC='uint8-delta-x-zlib1-v1'
def encode(block):
 a=np.asarray(block);assert a.dtype==np.uint8 and a.ndim==3 and a.flags.c_contiguous
 d=np.empty_like(a);d[:,:,0]=a[:,:,0];np.subtract(a[:,:,1:],a[:,:,:-1],out=d[:,:,1:]);wire=zlib.compress(d.tobytes(),level=1)
 return wire,hashlib.sha256(a.tobytes()).hexdigest()
def decode(wire,shape):
 shape=tuple(map(int,shape));assert len(shape)==3 and min(shape)>0 and shape[1:]==(384,384) and shape[0]<=1536;expected=shape[0]*384*384;obj=zlib.decompressobj();data=obj.decompress(wire,expected+1);assert len(data)==expected and obj.eof and not obj.unused_data and not obj.unconsumed_tail,'INVALID_OR_CONCATENATED_PIXEL_CHUNK'
 d=np.frombuffer(data,dtype=np.uint8).reshape(shape);return np.cumsum(d,axis=2,dtype=np.uint32).astype(np.uint8)
