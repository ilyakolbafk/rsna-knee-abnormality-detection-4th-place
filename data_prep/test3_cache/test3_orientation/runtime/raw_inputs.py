"""Read test DICOM once per study; separate130mm side and140mm H1 normalization.
Requires the frozen rsna_knee runtime alongside this release. No disease labels.
"""
import time
from pathlib import Path
import numpy as np,pandas as pd,pydicom,torch
from rsna_knee.grayscale_cache import select_series_for_cache,sample_window_indices
from rsna_knee.native_neighbor_bank import read_ordered_headers,normalization_bounds,quantize_with_bounds
from rsna_knee.dicom import dicom_pixels,pixel_spacing,center_crop_mm,resize_image
from rsna_knee.cache import resolve_series_dir
from rsna_knee.central_sampling import select_centers,source_triplets
from rsna_knee.compartment_locator_v2 import analyse_series
from canonical_inputs import side_record
from h1_adapter import apply_tokens
SLOTS=[{'name':'sagittal_fluid','plane':'Sagittal','fluid_sensitive':1,'slices':12,'window':[.12,.88]},
 {'name':'sagittal_structural','plane':'Sagittal','fluid_sensitive':0,'slices':12,'window':[.12,.88]},
 {'name':'coronal_fluid','plane':'Coronal','fluid_sensitive':1,'slices':8,'window':[.2,.8]},
 {'name':'coronal_structural','plane':'Coronal','fluid_sensitive':0,'slices':6,'window':[.2,.8]},
 {'name':'axial_fluid','plane':'Axial','fluid_sensitive':1,'slices':8,'window':[.12,.88]}]

def geometry(h):
 return {k:list(getattr(h,k)) if k in ('ImageOrientationPatient','ImagePositionPatient','PixelSpacing') else int(getattr(h,k)) for k in ('ImageOrientationPatient','ImagePositionPatient','PixelSpacing','Rows','Columns') if getattr(h,k,None) is not None}
def norm(x,v,ids):
 if not v[ids].any():raise ValueError('No valid normalization frames')
 lo,hi=normalization_bounds(x[ids],v[ids],(.5,99.5));q=quantize_with_bounds(x,lo,hi);q[~v]=0;return q,float(lo),float(hi)
class DecodedStudy:
 def __init__(self,uid,rows,images_root):
  self.uid=uid;self.rows=rows;self.images_root=images_root;self.series={};self.tag_values=[];self.errors=[];self.decoded=0;self.started=time.monotonic();self.locator={}
  frame=pd.DataFrame(rows)
  if any(str(r['StudyInstanceUID'])!=uid for r in rows):raise ValueError('Study UID mismatch')
  chosen=select_series_for_cache(frame,images_root,SLOTS);slots={c['series_uid']:i for i,c in enumerate(chosen) if c}
  for row in rows:
   sid=str(row['SeriesInstanceUID']);pl=row['Anatomical_Plane']
   if pl not in ('Axial','Coronal','Sagittal'):continue
   try:
    entries,qc=read_ordered_headers(resolve_series_dir(images_root,uid,sid));n=len(entries)
    if not n:continue
   except Exception as e:self.errors.append({'sid':sid,'stage':'headers','type':type(e).__name__});continue
   self.tag_values.extend(str(getattr(e['header'],'Laterality','')).strip() for e in entries[:1])
   raw130=np.zeros((n,336,336),np.float32);raw140=np.zeros_like(raw130);valid=np.zeros(n,bool);loc=[]
   for i,e in enumerate(entries):
    try:
     ds=pydicom.dcmread(e['path'],force=True)
     if str(ds.StudyInstanceUID)!=uid or str(ds.SeriesInstanceUID)!=sid:raise ValueError('DICOM UID mismatch')
     if i==0:self.tag_values.append(str(getattr(ds,'Laterality','')).strip())
     pix=dicom_pixels(ds);sp=pixel_spacing(ds);raw130[i]=resize_image(center_crop_mm(pix,sp,130.),336);raw140[i]=resize_image(center_crop_mm(pix,sp,140.),336);valid[i]=True;self.decoded+=1
     a=ds.pixel_array.astype(np.float32)*float(getattr(ds,'RescaleSlope',1) or 1)+float(getattr(ds,'RescaleIntercept',0) or 0)
     loc.append({'pixels':a,'iop':np.array(ds.ImageOrientationPatient,float),'ipp':np.array(ds.ImagePositionPatient,float),'spacing':np.array(ds.PixelSpacing,float),'rows':a.shape[0],'cols':a.shape[1],'desc':str(getattr(ds,'SeriesDescription','')),'frame':str(getattr(ds,'FrameOfReferenceUID',''))})
    except Exception as e:self.errors.append({'sid':sid,'index':i,'stage':'decode','type':type(e).__name__})
   slot=slots.get(sid);spec=SLOTS[slot] if slot is not None else None;ids=sample_window_indices(n,spec['slices'],*spec['window']) if spec else np.arange(n)
   try:q130,lo,hi=norm(raw130,valid,ids)
   except ValueError:self.errors.append({'sid':sid,'stage':'normalization','type':'no_valid_pixels'});continue
   self.series[sid]={'images130':q130,'float140':raw140,'valid':valid,'geometry_valid':np.array([e['geometry']['valid'] and qc['within_series_valid'] for e in entries],bool),'tags':[geometry(e['header']) for e in entries],'sops':[str(getattr(e['header'],'SOPInstanceUID','')) for e in entries],'qc':qc,'row':row,'slot':slot,'low130':lo,'high130':hi}
   if loc:self.locator[sid]=loc
  if not self.series:raise ValueError('No decodable study series')
  # Same locator as the development population; no side required for joint height.
  try:
   _,_,_,rr=analyse_series(uid,self.locator,frame.set_index('SeriesInstanceUID'),None);self.joint_z=rr.get('joint_z');self.locator_audit={k:rr.get(k) for k in ['joint_z','accept','sag_cor_gap','sag_series','cor_series']}
  except Exception as e:self.joint_z=None;self.locator_audit={'joint_z':None,'error':type(e).__name__}
  self.locator.clear()
 def side_inputs(self,resolver):
  sr=[];records={};planes=[];off=0
  for sid,s in self.series.items():
   row=s['row'];r=resolver.score_source_series(s['images130'],s['tags'],row['Anatomical_Plane'],bool(pd.to_numeric(row.get('Fluid_Sensitive'),errors='coerce')==1),valid=s['valid']&s['geometry_valid'],sop_ids=s['sops'],uid=self.uid,sid=sid);sr.append(r)
   records[sid]={'legacy_slot':s['slot'],'geometry_qc':s['qc']}
   planes.extend({'bank_index':off+i,'SeriesInstanceUID':sid,'geometry_tags':g,'geometry_valid':bool(s['geometry_valid'][i])} for i,g in enumerate(s['tags']));off+=len(s['tags'])
  meta={'series':records,'planes':planes};old=side_record(meta,self.tag_values);decision=resolver.resolve_scored(sr,old,self.joint_z);decision['old_metadata']=old;decision['locator']=self.locator_audit
  return decision
 def h1_bank(self,decision=None,route=None):
  rows=self.rows
  if decision is not None and decision['bilateral']:
   if route not in ('L','R'):raise ValueError('Bilateral route required')
   keep={s['sid'] for s in decision['series'] if s['assigned_side']==route};rows=[r for r in rows if r['SeriesInstanceUID'] in keep]
  choices=select_series_for_cache(pd.DataFrame(rows),self.images_root,SLOTS);images=[];decode=[];geo=[];si=[];sources=[];records={};planes=[];off=0
  for slot,c in enumerate(choices):
   if c is None or c['series_uid'] not in self.series:continue
   sid=c['series_uid'];s=self.series[sid];spec=SLOTS[slot];n=len(s['valid']);ids=sample_window_indices(n,spec['slices'],*spec['window'])
   if not s['valid'][ids].any():continue
   q,lo,hi=norm(s['float140'],s['valid'],ids);ix=len(records);fs=pd.to_numeric(s['row'].get('Fat_Suppression'),errors='coerce');fs=int(fs) if fs in (0,1) else None
   records[sid]={'offset':off,'source_count':n,'series_index':ix,'legacy_slot':slot,'geometry_qc':s['qc'],'source_inventory':{'fat_suppression':fs},'normalization_low':lo,'normalization_high':hi}
   planes.extend({'bank_index':off+i,'SeriesInstanceUID':sid,'geometry_tags':g,'geometry_valid':bool(s['geometry_valid'][i])} for i,g in enumerate(s['tags']))
   images.append(q);decode.extend(s['valid']);geo.extend(s['geometry_valid']);si.extend([ix]*n);sources.extend(range(n));off+=n
  if not images:raise ValueError('No usable H1 semantic slots for knee')
  arrays={'images':np.concatenate(images),'decode_valid':np.array(decode,bool),'geometry_valid':np.array(geo,bool),'series_index':np.array(si,np.int32),'source_index':np.array(sources,np.int32)};meta={'legacy_slots':SLOTS,'series':records,'uid':self.uid,'planes':planes};meta['side_record']=side_record(meta,self.tag_values);return arrays,meta
 def h1_inputs(self,decision,route=None):
  arrays,meta=self.h1_bank(decision,route);centers,slots,sel=select_centers(arrays,meta,budget=64,trim_each_side=[.1,.1,.1,.1,0],slot_weights=[1]*5);sample=source_triplets(arrays,meta,centers,slots);sample['center_refs']=centers
  refs=sample['bank_refs'];pixels=np.zeros((64,3,336,336),np.uint8);ok=refs>=0;pixels[ok]=arrays['images'][refs[ok]];inputs,audit=apply_tokens(pixels,arrays,meta,sample,decision,route)
  return inputs,{'uid':self.uid,'route':route or decision['side'],'tier':decision['confidence_tier'],'selection':sel,'transforms':audit,'decoded_source_frames':self.decoded,'source_errors':self.errors,'preparation_seconds':time.monotonic()-self.started}

def prepare_variants(uid,rows,images_root,resolver):
 study=DecodedStudy(uid,rows,images_root);decision=study.side_inputs(resolver)
 return [study.h1_inputs(decision,route) for route in decision['routes']],decision

def aggregate_knees(probabilities,decision,weights=None):
 """Binary study target means abnormality present in at least one imaged knee.
For bilateral studies use max across knee predictions (explicit unvalidated fusion);
side selection/physical correction is independent of disease scores.
"""
 p=np.asarray(probabilities)
 if p.ndim!=2 or not np.isfinite(p).all():raise ValueError('Invalid knee predictions')
 if decision['bilateral']:return p.max(0)
 if len(p)!=1:raise ValueError('Unexpected number of unilateral predictions')
 return p[0]
