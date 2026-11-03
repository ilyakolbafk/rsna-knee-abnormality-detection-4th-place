"""Server-only medical montage and numeric canary, before candidate corpus write."""
import json,math,time,collections
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from build_candidate import ROOT,sha,write
from fm_pixels import *
from fm_loader_reference import fm_window_targets,FM_CLASSES

def main():
 binding={'source_manifest_sha256':sha(ROOT/'receipts/SOURCE_FULLSHA_MANIFEST.json'),'adaptation_spec_sha256':sha(ROOT/'inputs/FM_ADAPTATION_SPEC.json'),'code_sha256':{n:sha(ROOT/'code'/n) for n in ['qa_real_canary.py','fm_geometry.py','fm_pixels.py','fm_loader_reference.py']},'runtime':{'numpy':np.__version__,'opencv':__import__('cv2').__version__,'Pillow':__import__('PIL').__version__}}
 src=json.loads((ROOT/'receipts/SOURCE_FULLSHA_MANIFEST.json').read_text());spec=json.loads((ROOT/'inputs/FM_ADAPTATION_SPEC.json').read_text());mapping=spec['class_mapping']['mapping'];classes=spec['class_mapping']['classes'];selected={};chosen_categories=set();split_modes=set();outside=None
 for r in src['volumes']:
  p=ROOT/'source_rss'/r['stem'];b=json.loads((p/'boxes.source.json').read_text());meta=json.loads((p/'metadata.json').read_text());sp=b['spatial'];mode='boxed' if sp else 'study_level_only' if b['study_level'] else 'reviewed_no_findings'
  key=(r['split'],mode)
  if mode in ['boxed','reviewed_no_findings'] and key not in split_modes:
   selected[(r['stem'],sp[0]['slice'] if sp else r['n_slices']//2)]=[key[0],key[1]];split_modes.add(key)
  for box in sp:
   label=box['label_raw'];assert label in mapping
   if label not in chosen_categories:selected.setdefault((r['stem'],box['slice']),[]).append('raw_category:'+label);chosen_categories.add(label)
   if outside is None:
    dx,dy=meta['spacing_xy_mm'];cw=round(110/dx);ch=round(110/dy);x0=(320-cw)//2;y0=(320-ch)//2;x,y,w,h=box['source_xywh']
    if x<x0 or y<y0 or x+w>x0+cw or y+h>y0+ch:
     selected.setdefault((r['stem'],box['slice']),[]).append('outside_old110');outside=(r['stem'],box['slice'])
  if r['stem']=='file1002097':selected.setdefault((r['stem'],27),[]).append('zero_height_quarantine_original_line11190')
 assert chosen_categories==set(mapping) and split_modes=={('train','boxed'),('val','boxed'),('train','reviewed_no_findings'),('val','reviewed_no_findings')} and outside is not None
 assert any(s=='file1002097' and k==27 for s,k in selected)
 qa=ROOT/'qa/real_canary';qa.mkdir(exist_ok=True);rows=[];positive_classes=set();montage_rows=[];scales={};receipts=[]
 for (stem,z),reasons in selected.items():
  p=ROOT/'source_rss'/stem;meta=json.loads((p/'metadata.json').read_text());bs=json.loads((p/'boxes.source.json').read_text())['spatial'];raw=np.load(p/'rss.f32.npy',mmap_mode='r');assert 0<=z<len(raw)
  scale,norm=normalize_volume(raw);display=display_slice(raw,z,scale);gs=geometries(raw.shape[-2:],meta['spacing_xy_mm']);images={v:render_display_slice(display,g) for v,g in gs.items()};boxes={v:[] for v in gs}
  for b in bs:
   if b['slice']!=z:continue
   for v,g in gs.items():
    transformed,info=transform_box(b['source_xywh'],classes.index(mapping[b['label_raw']]),g)
    if transformed is not None:boxes[v].append(transformed)
  for v,g in gs.items():
   a=np.asarray(boxes[v],float).reshape(-1,6);d,pr,k=fm_window_targets(a,384,0,0,0,0,0,False)
   positive_classes.update(c for c in range(11) if d[c].sum()>0)
   assert d.shape==(11,12,12) and np.isfinite(d).all()
   for b in a:assert b[5]>0 and b[1]<b[3] and b[2]<b[4]
  raw_display=np.clip(np.asarray(raw[z])/scale,0,1);raw_image=render_display_slice(raw_display,gs[0]);source_boxes=[]
  for b in bs:
   if b['slice']!=z:continue
   x,y,w,h=b['source_xywh'];v,_=transform_box([x,320-y-h,w,h],classes.index(mapping[b['label_raw']]),gs[0])
   if v is not None:source_boxes.append(v)
  panels=[('Raw RSS; inverse-mapped boxes',raw_image,source_boxes),('One flipud; display bbox',images[0],boxes[0]),('Fullfield384',images[0],boxes[0]),('Screen left100mm (not M/L)',images[2],boxes[2]),('Screen right100mm (not M/L)',images[3],boxes[3])]
  canvas=Image.new('RGB',(5*384,430),'#101010');draw=ImageDraw.Draw(canvas)
  for j,(name,a,bb) in enumerate(panels):
   im=Image.fromarray(a).convert('RGB');di=ImageDraw.Draw(im)
   for box in bb:di.rectangle(tuple(box[1:5]),outline='#66ff77',width=2)
   canvas.paste(im,(j*384,35));draw.text((j*384+5,4),name,fill='white')
  draw.text((5,415),stem+' slice '+str(z)+'; '+', '.join(reasons)[:200],fill='white');montage_rows.append(canvas)
  rows.append({'stem':stem,'slice':z,'reasons':reasons,'view_boxes':boxes,'geometries':{str(v):geometry_receipt(g) for v,g in gs.items()},'normalization':norm,'source_receipt_sha256':sha(p/'receipt.json')})
  if len(montage_rows)==4:
   montage=Image.new('RGB',(1920,430*len(montage_rows)))
   for j,im in enumerate(montage_rows):montage.paste(im,(0,j*430))
   target=qa/('montage_%02d.png'%len(receipts));montage.save(target);receipts.append({'path':str(target),'sha256':sha(target),'bytes':target.stat().st_size,'medical_image_stays_on_server':True});montage_rows=[]
 if montage_rows:
  montage=Image.new('RGB',(1920,430*len(montage_rows)))
  for j,im in enumerate(montage_rows):montage.paste(im,(0,j*430))
  target=qa/('montage_%02d.png'%len(receipts));montage.save(target);receipts.append({'path':str(target),'sha256':sha(target),'bytes':target.stat().st_size,'medical_image_stays_on_server':True})
 assert positive_classes==set(range(11))
 write(qa/'records.json',{'samples':rows,'montages':receipts})
 result={'state':'PASS_NUMERIC_REAL_CANARY_MONTAGE_GENERATED','binding':binding,'samples':len(rows),'both_source_splits_and_boxed_no_findings':True,'all_three_fields':True,'all21_raw_categories_covered':True,'all11_positive_area_dense_targets':True,'outside_old110_case':list(outside),'zero_height_case':['file1002097',27],'source_flipud_display_and_views_montages':receipts,'visual_review_completed':False,'anatomical_ML_validated':False,'medical_images_exported_to_Mac_or_chat':False,'records_sha256':sha(qa/'records.json'),'verified_unix':time.time()};write(ROOT/'qa/REAL_CANARY_QA.json',result);print(json.dumps({k:v for k,v in result.items() if k!='source_flipud_display_and_views_montages'},indent=2));return result
if __name__=='__main__':main()
