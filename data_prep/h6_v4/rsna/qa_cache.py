"""Verify bounded loader and physical/view directions; produce anatomy montages."""
from support import *
from cache_store import CacheReader
import argparse,math,numpy as np
from PIL import Image,ImageDraw

def qa(folder):
 reader=CacheReader(folder);idx=reader.index;result=reader.verify();tiles=[];dirs=[]
 for b in idx['blocks']:
  expected=[0,2,3]if b['plane']=='Coronal'else[0,1,4,5]
  if [v for v,start in b['views']]!=expected:raise ValueError('VIEW_SET')
  for vid,start in b['views']:
   k=b['n_slices']//2;tile=Image.fromarray(reader.row(start+k)).convert('RGB');draw=ImageDraw.Draw(tile)
   g=b['view_geometry'][str(vid)];aff=np.asarray(b['slabs'][k]['affine_lps'])@np.asarray(g['output_to_native'])
   def xy(point):return np.linalg.lstsq(aff[:,:2],np.asarray(point)-aff[:,2],rcond=None)[0]
   for name,color in [('joint_lps','lime'),('patella_lps','orange')]:
    point=idx['anchor'].get(name)
    if point is not None:
     x,y=xy(point);draw.ellipse((x-5,y-5,x+5,y+5),outline=color,width=2)
   center=aff@np.array([191.5,191.5,1.])
   for label,offset in [('A',[0,-15,0]),('P',[0,15,0]),('S',[0,0,15]),('L',[15,0,0])]:
    delta=xy(center+offset)-xy(center)
    if np.linalg.norm(delta)>5:
     base=np.array([40.,75.]);end=base+delta;draw.line([tuple(base),tuple(end)],fill='cyan',width=2);draw.text(tuple(end),label,fill='cyan')
   point=g['anchor_lps'];joint=np.asarray(idx['anchor']['joint_lps'])
   if point is not None:
    diff=np.asarray(point)-joint
    if vid==5 and diff[1]<=0:raise ValueError('POSTERIOR_LPS_SIGN')
    if vid==4 and diff[1]>=0:raise ValueError('PF_LPS_SIGN')
    if b['plane']=='Coronal':
     sign=(-1 if idx['resolved_route']=='L'else 1)*(1 if vid==2 else-1)
     if sign*diff[0]<=0:raise ValueError('MEDIAL_LATERAL_SIGN')
    dirs.append({'sid':b['series'],'view':vid,'offset_lps':diff.tolist()})
   draw.rectangle((0,0,383,38),fill='black');draw.text((3,3),f"{b['plane']} view{vid} route{idx['resolved_route']}",fill='white');draw.text((3,19),f"FOV {g['fov_mm'][0]:.1f}x{g['fov_mm'][1]:.1f} mm",fill='white');tiles.append(tile.resize((256,256)))
 montage=Image.new('RGB',(4*256,math.ceil(len(tiles)/4)*256))
 for i,tile in enumerate(tiles):montage.paste(tile,((i%4)*256,(i//4)*256))
 montage.save(Path(folder)/'MONTAGE.jpg',quality=90)
 result['verified_index_sha256']=sha(Path(folder)/'index.json')
 result.update(sample_id=idx['sample']['sample_id'],anchor_method=idx['anchor'].get('method'),directions=dirs,visual_anatomy_review='PENDING_HUMAN_OR_IMAGE_TOOL_REVIEW',private_hrepro_parity=False)
 dump(Path(folder)/'QA.json',result);return result
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('root');args=ap.parse_args()
 for p in sorted((Path(args.root)/'samples').glob('*/index.json')):
  r=qa(p.parent);print(json.dumps({k:v for k,v in r.items()if k!='directions'}),flush=True)
