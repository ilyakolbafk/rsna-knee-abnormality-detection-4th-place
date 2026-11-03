"""Original source functions, explicit v8.3/LPS integration; no private hrepro claim."""
from support import *
import numpy as np,torch,cv2
SOURCE=PACKAGE/'vendor/mainline';ORIENT=PACKAGE/'orientation_v8_3/runtime';PUBLIC=PACKAGE/'reference'
sys.path[:0]=[str(SOURCE),str(ORIENT),str(ORIENT/'rsna_knee')]
from training.dicom_geometry import pixels_from_dataset
from training.input_provenance import view_geometry
from training.intensity_mapping import normalize_intensity
from training.common import SLOTS as SOURCE_SLOTS
from canonical_inputs import transform_spec
from h1_adapter import apply_tokens
from resolver import LateralityResolver
from raw_inputs import DecodedStudy,SLOTS
KB=module('cache_original_anchor_v2',PUBLIC/'knee_anchor_bank_v2.py')
quantize_series=exact_function(SOURCE/'training/native_cache.py','quantize_series',{'np':np})
alignment_quality=exact_function(SOURCE/'training/geometry_bank.py','alignment_quality',{'np':np})
MODEL=None;RESOLVER=None
META_COLS=['series_idx','slot','plane','view_id','rel_pos','log_spacing','extent_130','fluid','fatsat','slice_order','n_slices','real_series_idx']

def resources():
 global MODEL,RESOLVER
 if MODEL is None:
  torch.set_num_threads(2);cv2.setNumThreads(1)
  loader=exact_function(SOURCE/'training/anatomy.py','load_segmenter',{'Path':Path,'sys':sys,'torch':torch,'__file__':str(SOURCE/'training/anatomy.py')})
  p=PACKAGE/'weights/anatomy.pt'
  if sha(p)!='5904dd16eb9a4fd7a1caf90ac977f8fe2ef364908598c23c209db3fe7fd1563e':raise ValueError('ANCHOR_WEIGHT_SHA')
  MODEL=loader(p,'cpu');RESOLVER=LateralityResolver(ORIENT,device='cpu')
  import monai
  if not Path(monai.__file__).resolve().is_relative_to(SOURCE/'training/vendor_runtime/monai060'):raise ValueError('WRONG_MONAI_RUNTIME')
 return MODEL,RESOLVER

class MemoryAnchorBank(KB.AnchorBank):
 def __init__(self,model,geometry):super().__init__(model,'cpu');self.entries=geometry
 def geometry(self,uid,sid,geo_dir=None):return self.entries.get(sid)

def orient(s,decision,route):
 vol,encoding=quantize_series(s['volume']);n,h,w=vol.shape;sid=s['sid'];frames=s['frames'];mid=n//2
 sd=next(x for x in decision['series']if x['sid']==sid)
 if decision['bilateral']and sd['assigned_side']!=route:raise ValueError('WRONG_KNEE_ROUTE')
 spec=transform_spec(frames[mid]['tags']['ImageOrientationPatient'],s['plane'],'L')
 if spec is None:raise ValueError('UNSUPPORTED_V83_GEOMETRY')
 def inplane(a):
  if spec['transpose']:a=np.swapaxes(a,0,1)
  if spec['flip_x']:a=a[:,::-1]
  if spec['flip_y']:a=a[::-1]
  pose=sd.get('apply_pose')
  if pose:
   a=np.rot90(a,int(pose['rot90_k']),axes=(0,1))
   if pose.get('flip_x'):a=a[:,::-1]
  if s['plane']in('Axial','Coronal')and sd.get('appearance_side',route)=='R':a=a[:,::-1]
  return a
 depth=bool(spec['reverse_channels'])^bool(s['plane']=='Sagittal'and sd.get('appearance_side',route)=='R')^bool(sd.get('reverse_depth_extra'))
 order=np.arange(n)[::-1]if depth else np.arange(n)
 # Independent original adapter comparison uses a square coordinate marker. Native
 # rectangular images use the identical D4 sequence without resizing or square crop.
 marker=np.arange(n*17*17,dtype=np.float32).reshape(n,17,17)%65536
 meta={'series':{sid:{'offset':0,'source_count':n,'series_index':0,'legacy_slot':0,'geometry_qc':{'within_series_valid':True},'source_inventory':{'fat_suppression':s['fatsat']}}},'planes':[{'bank_index':i,'SeriesInstanceUID':sid,'geometry_tags':f['tags'],'geometry_valid':True}for i,f in enumerate(frames)],'legacy_slots':[{'plane':s['plane']} ]}
 sample={'token_mask':np.array([True]),'center_refs':np.array([mid]),'slot_ids':np.array([0]),'bank_refs':np.arange(n)[None]}
 (test,_,_),audit=apply_tokens(marker[None],{'series_index':np.zeros(n,int)},meta,sample,decision,route)
 reference=np.stack([inplane(x)for x in marker])[order]
 if not np.allclose(test[0].numpy()*255,reference,atol=.005,rtol=0):raise ValueError('ORIGINAL_ADAPTER_D4_MISMATCH')
 oriented=np.stack([inplane(x)for x in vol])[order]
 yy,xx=np.mgrid[:h,:w];coords=inplane(np.stack([xx,yy,np.ones_like(xx)],-1)).astype(float)
 pull=np.stack([coords[0,1]-coords[0,0],coords[1,0]-coords[0,0],coords[0,0]],axis=1)
 s=dict(s,volume=oriented,aff=s['aff'][order]@pull,frames=[frames[i]for i in order],orientation_audit=audit,pull_stored_to_original=pull.tolist(),native_encoding=encoding,window=np.percentile(oriented[:,::4,::4],[1,99]).astype(float).tolist())
 return s

def slab_chunks(n,spacing):
 indices=np.arange(n);chunks=[indices[i:i+1]for i in range(n)]
 if n>1 and 0<spacing<1.75:
  g=max(1,int(round(3.5/spacing)));nn=n//g+int(n%g>=max(1,g//2))
  if nn>=2:chunks=[indices[j*g:min(n,(j+1)*g)]for j in range(nn)]
 return chunks

def views(s,anchor,route):
 joint=np.asarray(anchor['joint_lps'],float);pat=anchor.get('patella_lps');plane=s['plane'];out=[(0,None,None)]
 if plane=='Sagittal':out +=[(1,100.,joint),(4,100.,np.asarray(pat)if pat is not None else joint+[0,-40,25]),(5,100.,joint+[0,35,-5])]
 elif plane=='Axial':out +=[(1,100.,joint),(4,100.,np.asarray(pat)if pat is not None else joint+[0,-40,0]),(5,100.,joint+[0,35,0])]
 else:
  lateral=30 if route=='L'else-30;out +=[(2,100.,joint+[-lateral,0,0]),(3,100.,joint+[lateral,0,0])]
 result=[]
 for vid,crop,point in out:
  if point is None:center=(.5,.5)
  else:
   _,c,r,_=KB.project(s['aff'],np.asarray(point,float));h,w=s['volume'].shape[-2:];center=(float(np.clip(r/max(h-1,1),0,1)),float(np.clip(c/max(w-1,1),0,1)))
  result.append((vid,crop,center,None if point is None else np.asarray(point,float).tolist()))
 return result
