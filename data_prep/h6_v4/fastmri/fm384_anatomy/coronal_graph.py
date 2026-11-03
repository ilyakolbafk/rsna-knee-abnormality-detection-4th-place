"""Literal inference-only Keras/HDF5 graph adapter. Requires separate TF parity."""
import json,math
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class CoronalGraph(nn.Module):
 def __init__(self,path):
  super().__init__();self.weights={}
  with h5py.File(path,'r') as f:
   cfg=f.attrs['model_config'];cfg=cfg.decode() if isinstance(cfg,bytes) else cfg;cfg=json.loads(cfg)['config'];self.layers=cfg['layers'];self.input_name=cfg['input_layers'][0][0];self.output_name=cfg['output_layers'][0][0]
   for layer in self.layers:
    name=layer['name'];base='model_weights/'+name+'/'+name
    if base not in f:continue
    for k,obj in f[base].items():
     assert isinstance(obj,h5py.Dataset) and obj.dtype==np.float32
     key=name+'__'+k.split(':')[0];a=np.asarray(obj)
     if k.startswith('kernel') and layer['class_name'] in ['Conv2D','Conv2DTranspose']:a=a.transpose(3,2,0,1).copy()
     self.register_buffer(key,torch.from_numpy(np.ascontiguousarray(a)));self.weights[(name,k.split(':')[0])]=key
 def w(self,name,k):return getattr(self,self.weights[(name,k)])
 def forward(self,x,capture=None):
  values={self.input_name:x}
  for layer in self.layers:
   kind=layer['class_name'];c=layer['config'];name=layer['name']
   if kind=='InputLayer':continue
   nodes=layer['inbound_nodes'];assert len(nodes)==1 and all(n[1:3]==[0,0] for n in nodes[0]);xs=[values[n[0]] for n in nodes[0]];a=xs[0]
   if kind=='Conv2D':
    assert c['data_format']=='channels_first' and c['activation']=='linear';s=c['strides'];k=c['kernel_size'];d=c['dilation_rate']
    if c['padding']=='same':
     total=[max(0,(math.ceil(a.shape[-2+j]/s[j])-1)*s[j]+d[j]*(k[j]-1)+1-a.shape[-2+j]) for j in range(2)];a=F.pad(a,[total[1]//2,total[1]-total[1]//2,total[0]//2,total[0]-total[0]//2])
    else:assert c['padding']=='valid'
    y=F.conv2d(a,self.w(name,'kernel'),self.w(name,'bias') if c['use_bias'] else None,stride=s,dilation=d,groups=c.get('groups',1))
   elif kind=='Conv2DTranspose':
    assert c['data_format']=='channels_first' and c['activation']=='linear' and c['dilation_rate']==[1,1] and c['output_padding'] is None
    s=c['strides'];k=c['kernel_size'];y=F.conv_transpose2d(a,self.w(name,'kernel'),self.w(name,'bias') if c['use_bias'] else None,stride=s)
    if c['padding']=='same':
     start=[(k[j]-s[j])//2 for j in range(2)];assert min(start)>=0;y=y[:,:,start[0]:start[0]+a.shape[2]*s[0],start[1]:start[1]+a.shape[3]*s[1]]
    else:assert c['padding']=='valid'
   elif kind=='BatchNormalization':
    axis=c['axis'];axis=axis[0] if isinstance(axis,list) else axis;shape=[1]*a.ndim;shape[axis]=len(self.w(name,'moving_mean'))
    mean=self.w(name,'moving_mean').reshape(shape);var=self.w(name,'moving_variance').reshape(shape);gamma=self.w(name,'gamma').reshape(shape) if c['scale'] else 1.;beta=self.w(name,'beta').reshape(shape) if c['center'] else 0.
    y=(a-mean)*torch.rsqrt(var+c['epsilon'])*gamma+beta
   elif kind=='PReLU':y=torch.relu(a)-self.w(name,'alpha')*torch.relu(-a)
   elif kind=='Dropout':y=a
   elif kind=='Concatenate':y=torch.cat(xs,dim=c['axis'])
   elif kind=='Add':
    y=xs[0]
    for v in xs[1:]:y=y+v
   elif kind=='Permute':y=a.permute([0]+c['dims'])
   elif kind=='Reshape':y=a.reshape(a.shape[0],*c['target_shape'])
   elif kind=='Activation':
    assert c['activation']=='softmax';y=torch.softmax(a,dim=-1)
   else:raise RuntimeError('UNSUPPORTED_KERAS_LAYER:'+kind)
   values[name]=y
   if capture is not None and name in capture:capture[name]=y.detach().cpu().numpy()
  return values[self.output_name]
