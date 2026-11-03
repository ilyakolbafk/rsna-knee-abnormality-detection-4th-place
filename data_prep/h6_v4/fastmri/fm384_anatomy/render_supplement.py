"""Additional unselected posterior source frames after blinded abstentions."""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from anatomy_job import ROOT,BASE,write
from render_validation_panels import register

def main():
 register(True)
 try:
  cases={c['alias']:c for c in json.loads((ROOT/'VALIDATION_SELECTION.json').read_text())['cases']};out=ROOT/'review_supplement';out.mkdir(exist_ok=True);aliases=json.loads((ROOT/'SUPPLEMENT_REQUEST.json').read_text())['aliases'];pack=[]
  for a in aliases:
   c=cases[a];p=BASE/'source_rss'/c['stem'];assert hashlib.sha256((p/'receipt.json').read_bytes()).hexdigest()==c['source_receipt_sha256'];raw=np.load(p/'rss.f32.npy',mmap_mode='r');zs=list(range(round(.4*(len(raw)-1)),len(raw)));scale=max(float(np.percentile(raw[::2,::2,::2],99.5)),1e-12)
   for page in range(0,len(zs),16):
    ix=zs[page:page+16];H=348*((len(ix)+3)//4);panel=Image.new('RGB',(1280,H),(0,0,0));draw=ImageDraw.Draw(panel)
    for j,z in enumerate(ix):
     im=np.rint(np.clip(np.asarray(raw[z],np.float32)[::-1,:]/scale,0,1)*255).astype(np.uint8);x=j%4*320;y=j//4*348;panel.paste(Image.fromarray(im),(x,y+28));draw.text((x+5,y+5),a+' source slice '+str(z),fill='white')
    file=out/(a+'_p'+str(page//16)+'.png');panel.save(file);pack.append({'relative_path':'review_supplement/'+file.name,'PNG_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'cases':[{'alias':a,'source_slice_indices':ix,'source_receipt_sha256':c['source_receipt_sha256'],'display_operation':'one flipud only; no hflip/overlays/predictions'}]})
  for group,vis in [('A',lambda a:a.startswith('C') or int(a[1:])<49),('B',lambda a:a.startswith('C') or int(a[1:])>=49)]:
   write(out/('SUPPLEMENT_'+group+'.json'),{'reviewer_group':group,'purpose':'additional clean source following initial blinded needs_more_frames; no model information disclosed','panels':[p for p in pack if vis(p['cases'][0]['alias'])]})
  write(out/'COMPLETE.json',{'panels':len(pack),'aliases':aliases})
 finally:register(False)
if __name__=='__main__':main()
