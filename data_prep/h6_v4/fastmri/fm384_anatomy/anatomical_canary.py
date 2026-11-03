"""Bound source-to-cache numeric/visual canary. PASS requires separate recorded internal source review."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import json,hashlib,collections,argparse,time
import numpy as np
import cv2
from PIL import Image,ImageDraw
from anatomy_job import ROOT,BASE,OwnedJob,write
from fm_proof_chain import verify_decisions,sha,load,require
from fm_anatomy_pixels import geometries,geometry_receipt,source_box_to_view,render,normalize_volume
from fm_delta_codec import encode,decode
from build_anatomical_chunks import source_unchanged
Q=ROOT/'qa';P=Q/'anatomical_panels'

def select_cases(rows,D):
 available=[r for r in rows if D[r['stem']]['accepted']];labels={r['stem']:{x['label_raw'] for x in load(BASE/'source_rss'/r['stem']/'boxes.source.json')['spatial']} for r in available};target=set.union(*labels.values());chosen=[];left=set(target)
 while left:
  best=min(available,key=lambda r:(-len(labels[r['stem']]&left),hashlib.sha256(r['stem'].encode()).hexdigest()))['stem'];require(labels[best]&left,'RAW_CATEGORY_COVERAGE');chosen.append(best);left-=labels[best]
 for split in ['train','val']:
  for kind in ['boxed','negative']:
   choices=[r for r in available if r['split']==split and bool(labels[r['stem']])==(kind=='boxed')];require(bool(choices),'MISSING_CANARY_STRATUM')
   chosen+= [r['stem'] for r in sorted(choices,key=lambda r:hashlib.sha256((r['stem']+'canary').encode()).hexdigest())[:2]]
 chosen += [s for s in sorted(D) if D[s]['accepted'] and D[s]['manual_challenge_alias']][:4]
 chosen=list(dict.fromkeys(chosen))
 for r in sorted(available,key=lambda r:hashlib.sha256((r['stem']+'independent_canary').encode()).hexdigest()):
  if len(chosen)>=24:break
  if r['stem'] not in chosen:chosen.append(r['stem'])
 require(len(chosen)>=24,'CANARY_SMALL');return chosen,target

def main(finalize=False):
 d=verify_decisions(ROOT,BASE);D=d['volumes'];rows=load(BASE/'receipts/SOURCE_FULLSHA_MANIFEST.json')['volumes'];rs={r['stem']:r for r in rows};selected,all_labels=select_cases(rows,D);P.mkdir(exist_ok=True);numeric=collections.Counter();case_records=[];tiles=[]
 for r in rows:
  s=r['stem'];a=D[s]
  if not a['accepted']:continue
  p=source_unchanged(r);m=load(p/'metadata.json');dx,dy=m['spacing_xy_mm'];flip=a['hflip_to_lateral_pluscol'];gs=geometries((320,320),[dx,dy],a['joint_anchor_canonical_display_row_col'])
  for v,g in gs.items():
   rec=geometry_receipt(g,flip,a['joint_anchor_canonical_display_row_col'],v);A=np.array(rec['annotation_display_to_view_xy_edges']);B=np.array(rec['view_to_annotation_display_xy_edges']);require(np.allclose(A@B,np.eye(3),atol=1e-10),'INVERSE_AFFINE')
   if v:require(max(abs(g.crop_shape[0]*dy-100),abs(g.crop_shape[1]*dx-100))<=max(dx,dy)/2+1e-6,'LOCAL_PHYSICAL_FOV')
   numeric['geometry_inverse_and_physical_fov']+=1
  bs=load(p/'boxes.source.json')['spatial']
  for b in bs:
   x,y,w,h=b['source_xywh'];xx=320-x-w if flip else x
   for v,g in gs.items():
    cx,cy,_,_=g.crop_xyxy;vx,vy,vx1,vy1=g.valid_xyxy;sx=(vx1-vx)/g.crop_shape[1];sy=(vy1-vy)/g.crop_shape[0];raw=[(xx-cx)*sx+vx,(y-cy)*sy+vy,(xx+w-cx)*sx+vx,(y+h-cy)*sy+vy];expected=[max(raw[0],vx),max(raw[1],vy),min(raw[2],vx1),min(raw[3],vy1)];got,info=source_box_to_view(b['source_xywh'],0,g,flip)
    require(abs(info['a0']-w*h*sx*sy)<1e-6,'ORIGINAL_AREA_SCALE')
    if expected[2]>expected[0] and expected[3]>expected[1]:require(got is not None and np.allclose(got[1:5],expected),'INDEPENDENT_BOX_COORDINATES')
    else:require(got is None,'OUTSIDE_BOX_NOT_REMOVED')
    numeric['independent_box_area_clip']+=1
  if s not in selected:continue
  raw=np.load(p/'rss.f32.npy',mmap_mode='r');scale,norm=normalize_volume(raw);z=round((r['n_slices']-1)*.6);display=np.clip(raw[z][::-1,:]/scale,0,1);views=[]
  for v,g in gs.items():
   got=render(raw,z,scale,g,flip);canonical=display[:,::-1] if flip else display;cx,cy,cx1,cy1=g.crop_xyxy;vx,vy,vx1,vy1=g.valid_xyxy;expected=np.zeros((384,384),np.uint8);expected[vy:vy1,vx:vx1]=np.rint(np.clip(cv2.resize(canonical[cy:cy1,cx:cx1],(vx1-vx,vy1-vy),interpolation=cv2.INTER_AREA),0,1)*255).astype(np.uint8);require(np.array_equal(got,expected),'INDEPENDENT_PIXEL_RENDER');numeric['independent_pixels']+=1;views.append((v,got))
  block=np.stack([x[1] for x in views]);wire,digest=encode(block);require(np.array_equal(decode(wire,block.shape),block),'CANARY_LOSSLESS');numeric['all_pixel_lossless_pairs']+=1
  source=np.rint(display*255).astype(np.uint8);images=[Image.fromarray(source).convert('RGB')]+[Image.fromarray(x[1]).convert('RGB') for x in views];labels=['source one flipud; lateral='+a['source_lateral_image_side'],'full0 canonical: lateral right','medial2 local100mm','lateral3 local100mm'];tile=Image.new('RGB',(4*272,314),(18,18,18));draw=ImageDraw.Draw(tile)
  for k,im in enumerate(images):
   im=im.resize((256,256),Image.Resampling.NEAREST);tile.paste(im,(k*272,40));draw.text((k*272+2,24),labels[k],fill='white')
   if k==0:ar,ac=a['joint_anchor_source_display_row_col'];cx=ac*256/320;cy=ar*256/320
   else:
    v=[0,2,3][k-1];g=gs[v];rr=geometry_receipt(g,flip,a['joint_anchor_canonical_display_row_col'],v);ar,ac=a['joint_anchor_source_display_row_col'];pt=np.array(rr['annotation_display_to_view_xy_edges'])@np.array([ac+.5,ar+.5,1]);cx=pt[0]*256/384;cy=pt[1]*256/384
   if 0<=cx<256 and 0<=cy<256:draw.line((k*272+cx-5,40+cy,k*272+cx+5,40+cy),fill='red',width=2);draw.line((k*272+cx,40+cy-5,k*272+cx,40+cy+5),fill='red',width=2)
  draw.text((3,4),s+' '+r['split']+' source_slice='+str(z)+' joint red; display resizes for QA only',fill='white');tiles.append(tile);case_records.append({'stem':s,'split':r['split'],'source_slice':z,'source_receipt_sha256':r['receipt_sha256'],'source_lateral_image_side':a['source_lateral_image_side'],'joint_anchor_source_display_row_col':a['joint_anchor_source_display_row_col'],'raw_labels':sorted({b['label_raw'] for b in bs}),'source_has_pathology_overlay':False,'geometry':{str(v):geometry_receipt(g,flip,a['joint_anchor_canonical_display_row_col'],v) for v,g in gs.items()}})
 # Source boxes and their final reflected/full/local coordinates, one visible example for every output class.
 spec=load(BASE/'inputs/FM_ADAPTATION_SPEC.json');classes=spec['class_mapping']['classes'];mapping=spec['class_mapping']['mapping'];examples={}
 for r in rows:
  if not D[r['stem']]['accepted']:continue
  for b in load(BASE/'source_rss'/r['stem']/'boxes.source.json')['spatial']:
   c=classes.index(mapping[b['label_raw']]);area=b['source_xywh'][2]*b['source_xywh'][3]
   if c not in examples or area>examples[c][0]:examples[c]=(area,r,b)
 require(set(examples)==set(range(11)),'CANARY_CLASS_EXAMPLE_MISSING');box_examples=[];box_tiles=[]
 for c,(area,r,chosen_box) in sorted(examples.items()):
  stem=r['stem'];a=D[stem];p=source_unchanged(r);raw=np.load(p/'rss.f32.npy',mmap_mode='r');scale,norm=normalize_volume(raw);z=chosen_box['slice'];dx,dy=load(p/'metadata.json')['spacing_xy_mm'];gs=geometries((320,320),[dx,dy],a['joint_anchor_canonical_display_row_col']);flip=a['hflip_to_lateral_pluscol'];display=np.rint(np.clip(raw[z][::-1,:]/scale,0,1)*255).astype(np.uint8);images=[Image.fromarray(display).convert('RGB')]+[Image.fromarray(render(raw,z,scale,gs[v],flip)).convert('RGB') for v in [0,2,3]];all_boxes=[b for b in load(p/'boxes.source.json')['spatial'] if b['slice']==z];tile=Image.new('RGB',(1088,314),(18,18,18));caption=ImageDraw.Draw(tile);caption.text((3,4),str(c)+' '+classes[c]+' '+stem+' slice='+str(z)+' yellow=chosen class; other boxes=cyan',fill='white')
  for k,im in enumerate(images):
   im=im.resize((256,256),Image.Resampling.NEAREST);draw=ImageDraw.Draw(im)
   for b in all_boxes:
    klass=classes.index(mapping[b['label_raw']]);color='yellow' if klass==c else 'cyan'
    if k==0:
     x,y,w,h=b['source_xywh'];xy=[x,y,x+w,y+h];den=320
    else:
     box,info=source_box_to_view(b['source_xywh'],klass,gs[[0,2,3][k-1]],flip)
     if box is None:continue
     xy=box[1:5];den=384
    draw.rectangle(tuple(v*256/den for v in xy),outline=color,width=2)
   tile.paste(im,(k*272,40));caption.text((k*272+2,24),['source boxes','full0 canonical boxes','medial2 boxes','lateral3 boxes'][k],fill='white')
  box_tiles.append(tile);box_examples.append({'class_index':c,'class_name':classes[c],'stem':stem,'slice':z,'source_box':chosen_box,'source_receipt_sha256':r['receipt_sha256']})

 sheets={}
 for k in range(0,len(tiles),4):
  sheet=Image.new('RGB',(1088,314*len(tiles[k:k+4])),(0,0,0))
  for j,t in enumerate(tiles[k:k+4]):sheet.paste(t,(0,j*314))
  name='anatomical-%02d.png'%(k//4+1);sheet.save(P/name);sheets[name]=sha(P/name)
 for k in range(0,len(box_tiles),4):
  sheet=Image.new('RGB',(1088,314*len(box_tiles[k:k+4])),(0,0,0))
  for j,t in enumerate(box_tiles[k:k+4]):sheet.paste(t,(0,j*314))
  name='boxes-%02d.png'%(k//4+1);sheet.save(P/name);sheets[name]=sha(P/name)
 packet={'box_examples':box_examples,'inputs':d['inputs'],'anatomy_decisions_sha256':sha(ROOT/'ANATOMY_DECISIONS.json'),'code_sha256':sha(__file__),'cases':case_records,'panels_sha256':sheets,'numeric_counts':dict(numeric),'all_accepted_raw_spatial_categories':sorted(all_labels),'visual_review_state':'PENDING','medical_images_server_only':True};write(Q/'ANATOMICAL_CANARY_PACKET.json',packet)
 if finalize:
  review=load(Q/'ANATOMICAL_CANARY_VISUAL_REVIEW.json');require(review['state']=='PASS' and review['packet_sha256']==sha(Q/'ANATOMICAL_CANARY_PACKET.json') and review['panels_sha256']==sheets,'UNBOUND_VISUAL_REVIEW');require(set(review['reviewed_stems'])==set(selected),'INCOMPLETE_VISUAL_REVIEW');require(set(review['reviewed_box_class_indices'])==set(range(11)),'INCOMPLETE_CLASS_BOX_VISUAL_REVIEW')
  write(Q/'ANATOMICAL_CANARY.json',{'state':'PASS','inputs':d['inputs'],'anatomy_decisions_sha256':sha(ROOT/'ANATOMY_DECISIONS.json'),'code_sha256':sha(__file__),'numeric_checks_pass':True,'numeric_counts':dict(numeric),'visual_review_state':'PASS','visual_review_sha256':sha(Q/'ANATOMICAL_CANARY_VISUAL_REVIEW.json'),'packet_sha256':sha(Q/'ANATOMICAL_CANARY_PACKET.json'),'no_private_pixel_or_population_accuracy_claim':True,'training_started':False})
 return {'state':'PASS' if finalize else 'NUMERIC_PASS_VISUAL_PENDING','cases':len(selected),'numeric_counts':dict(numeric),'panels':list(sheets)}
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--finalize',action='store_true');a=ap.parse_args()
 with OwnedJob('anatomical_canary_numeric_and_panels',cpu=2,ram_gib=3,disk_gib=.15) as job:print(json.dumps(main(a.finalize)),flush=True)
