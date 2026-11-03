"""Fixed H1Aux ResNet family; only the predeclared arm parameters differ."""
import hashlib
from pathlib import Path
import timm
import torch
from torch import nn
from torch.nn import functional as F
from models import H1Aux

RAD_SHA='08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734'
ARMS={
 'R0':dict(size=336,init='radimagenet',schedule=24,stop=2,average=[],wd=.02,dropout=.2,eval=[]),
 'R1':dict(size=224,init='radimagenet',schedule=24,stop=20,average=[18,19,20],wd=.02,dropout=.2,eval=[5,10,15,20]),
 'R2':dict(size=336,init='radimagenet',schedule=24,stop=20,average=[18,19,20],wd=.02,dropout=.2,eval=[5,10,15,20]),
 'R3':dict(size=336,init='imagenet',schedule=24,stop=20,average=[18,19,20],wd=.02,dropout=.2,eval=[5,10,15,20]),
 'R4':dict(size=336,init='radimagenet',schedule=48,stop=40,average=[38,39,40],wd=.05,dropout=.3,eval=[10,20,30,40]),
 'R5':dict(size=224,init='radimagenet',schedule=48,stop=40,average=[38,39,40],wd=.05,dropout=.3,eval=[10,20,30,40]),
}

class ResnetAux(H1Aux):
 def __init__(self,backbone,size,dropout):
  super().__init__(backbone=backbone,chunk=96,checkpointing=False)
  self.feature_size=size
  self.att[2]=nn.Dropout(dropout);self.aux_att[2]=nn.Dropout(dropout)
 def features(self,x,gain):
  assert x.shape[-2:]==(336,336)
  if self.feature_size!=336:
   x=F.interpolate(x,size=(self.feature_size,self.feature_size),mode='bilinear',align_corners=False)
  x=(x*gain[:,None,None,None]).clamp_(0,1)
  return self.backbone((x-.5)/.5)

def make(arm,assets):
 cfg=ARMS[arm];assets=Path(assets)
 model=timm.create_model('resnet50',pretrained=False,num_classes=0)
 path=assets/('ResNet50.pt' if cfg['init']=='radimagenet' else 'ImageNet_resnet50_a1.pth')
 raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest()
 state=torch.load(path,map_location='cpu',weights_only=True)
 if cfg['init']=='radimagenet':
  assert sha==RAD_SHA
  mapping={'0':'conv1','1':'bn1','4':'layer1','5':'layer2','6':'layer3','7':'layer4'};converted={}
  for k,v in state.items():
   parts=k.split('.');assert parts[0]=='backbone' and parts[1] in mapping
   key=mapping[parts[1]]+'.'+'.'.join(parts[2:]);assert key not in converted;converted[key]=v
  state=converted
 else:
  if 'state_dict' in state:state=state['state_dict']
  state={k:v for k,v in state.items() if not k.startswith('fc.')}
 result=model.load_state_dict(state,strict=True)
 assert not result.missing_keys and not result.unexpected_keys
 identity={'backbone':'timm.resnet50','initialization':cfg['init'],'weights_sha256':sha,
  'weights_bytes':len(raw),'source_keys_loaded':len(state),'missing_keys':[],'unexpected_keys':[],
  'conv1_sha256':hashlib.sha256(state['conv1.weight'].numpy().tobytes()).hexdigest(),
  'parameter_count':sum(p.numel() for p in model.parameters()),'source_filename':path.name,
  'normalization':'(x-0.5)/0.5','feature_size':cfg['size'],'batchnorm':'train, 96 windows per forward',
  'checkpointing':False,'ema':False,'backbone_lr':1e-4,'head_lr':1e-3,'layer_decay':None,
  'recipe':cfg,'radimagenet_source':'marwanmath/resnet-50-radimagenet-marwan/ResNet50.pt',
  'upstream_license':'MIT, BMEII-AI/RadImageNet; see LICENSE_RadImageNet.txt'}
 return ResnetAux(model,cfg['size'],cfg['dropout']),identity
