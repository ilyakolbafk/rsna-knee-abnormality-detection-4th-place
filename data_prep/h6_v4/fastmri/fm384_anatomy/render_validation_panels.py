"""Render source-bound blinded anatomy panels. No model scores or lesion labels."""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import json,time,fcntl,socket,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from anatomy_job import ROOT,BASE,SHARED,ticks,live,write
def register(active):
 with (SHARED/'admission.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);p=SHARED/'admission.json';d=json.loads(p.read_text());key='fastmri_anatomy_validation_render';old=d['tasks'].get(key,{})
  if old.get('PID') not in [None,os.getpid()] and live(old['PID'],old.get('ctime')):raise RuntimeError('LIVE_RENDERER_EXISTS')
  if active:
   peers=[v for k,v in d['tasks'].items() if k!=key];assert sum(v.get('cpu_threads',0) for v in peers)+1<=d['limits']['max_total_cpu_threads']
  d['tasks'][key]={'run_id':'fm-anatomy-blinded-validation-20261021','PID':os.getpid(),'ctime':ticks(os.getpid()),'active':active,'cpu_threads':1 if active else 0,'download_workers':0,'convert_workers':0,'reserved_ram_bytes':536870912 if active else 0,'reserved_disk_bytes':536870912 if active else 0,'heartbeat_unix':time.time(),'phase':'rendering_clean_blinded_validation' if active else 'validation_panels_complete','scratch_paths':[str(ROOT/'review_validation')],'argv':Path('/proc/self/cmdline').read_bytes().decode().strip('\0').split('\0')};write(p,d)
def main():
 t=json.loads((BASE/'TARGET_VERIFIED.json').read_text());assert socket.gethostname()==t['hostname'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==t['boot_id'];register(True)
 try:
  sel=json.loads((ROOT/'VALIDATION_SELECTION.json').read_text());cases={x['alias']:x for x in sel['cases']};out=ROOT/'review_validation';out.mkdir(exist_ok=True);panels={}
  sets={'A':['V%03d'%i for i in range(1,49)]+['C%03d'%i for i in range(1,17)],'B':['V%03d'%i for i in range(49,97)]+['C%03d'%i for i in range(1,17)]}
  for group,aliases in sets.items():
   packet=[]
   for i in range(0,len(aliases),2):
    pair=aliases[i:i+2];name='_'.join(pair)+'.png'
    if name not in panels:
     panel=Image.new('RGB',(1280,1392),(0,0,0));draw=ImageDraw.Draw(panel);records=[]
     for ci,alias in enumerate(pair):
      c=cases[alias];p=BASE/'source_rss'/c['stem'];assert hashlib.sha256((p/'receipt.json').read_bytes()).hexdigest()==c['source_receipt_sha256']
      raw=np.load(p/'rss.f32.npy',mmap_mode='r');zs=sorted(set(round(float(f)*(len(raw)-1)) for f in [.08,.18,.28,.40,.52,.64,.76,.88]));scale=max(float(np.percentile(raw[::2,::2,::2],99.5)),1e-12)
      for j,z in enumerate(zs):
       image=np.rint(np.clip(np.asarray(raw[z],np.float32)[::-1,:]/scale,0,1)*255).astype(np.uint8);x=(j%4)*320;y=(ci*2+j//4)*348;panel.paste(Image.fromarray(image),(x,y+28));draw.text((x+5,y+5),alias+' source slice '+str(z),fill=(255,255,255))
      records.append({'alias':alias,'source_slice_indices':zs,'source_receipt_sha256':c['source_receipt_sha256'],'display_operation':'one flipud, no horizontal reflection, no prediction/lesion overlay'})
     file=out/name;panel.save(file);panels[name]={'relative_path':'review_validation/'+name,'PNG_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'cases':records}
    packet.append(panels[name]);register(True)
   write(out/('PACKET_'+group+'.json'),{'reviewer_group':group,'protocol_semantic_sha256':sel['protocol_sha256'],'panels':packet,'blinding':'No model predictions, disease boxes/labels or VQA; aliases only','source_image_files_exported_to_Mac':False})
  write(out/'PANELS_COMPLETE.json',{'panels':len(panels),'cases':len(cases),'source_selection_sha256':hashlib.sha256((ROOT/'VALIDATION_SELECTION.json').read_bytes()).hexdigest(),'protocol_semantic_sha256':sel['protocol_sha256'],'completed_unix':time.time()})
 finally:register(False)
if __name__=='__main__':main()
