"""Receive author's MRAnnotator package; extract Task002 only for a research pilot."""
import os
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import json,hashlib,urllib.request,urllib.parse,time,zipfile,sys,subprocess
from pathlib import Path
from anatomy_job import ROOT,OwnedJob,write
def main():
 with OwnedJob('receiving_MRAnnotator_research_pilot_model',downloads=1) as job:
  d=ROOT/'weights';d.mkdir(exist_ok=True);dst=d/'mrannotator_weights.zip';meta=json.loads((ROOT/'private/mrannotator_download.json').read_text());u=meta['url'];assert urllib.parse.urlsplit(u).hostname=='drive.usercontent.google.com'
  if not dst.exists():
   tmp=dst.with_suffix('.zip.partial');h=hashlib.sha256();n=0
   with urllib.request.urlopen(u,timeout=45) as response,tmp.open('wb') as f:
    assert response.status==200 and int(response.headers['Content-Length'])==2033840887
    for b in iter(lambda:response.read(4*1024**2),b''):
     job.check();f.write(b);h.update(b);n+=len(b);job.progress['downloaded_bytes']=n
    f.flush();os.fsync(f.fileno())
   assert n==2033840887;os.replace(tmp,dst)
   write(d/'MRANNOTATOR_ARCHIVE_RECEIPT.json',{'bytes':n,'receiver_sha256':h.hexdigest(),'official_sha256':None,'source':'MRAnnotator pinned author README4cd0d5b5cd120f17bc9e17462534872c4ad2b12a actual Drive href15gQuvmaDlbs7HlGpISWNqnnWYGp9-Xc4','license':'CC-BY-NC; research pilot only; no competition/commercial eligibility claim','received_unix':time.time()})
  inventory=[];out=d/'mrannotator';out.mkdir(exist_ok=True)
  with zipfile.ZipFile(dst) as z:
   for info in z.infolist():
    name=Path(info.filename);assert not name.is_absolute() and '..' not in name.parts
    inventory.append({'path':info.filename,'bytes':info.file_size,'compressed_bytes':info.compress_size,'CRC32':info.CRC})
    if not info.is_dir() and any('Dataset002' in s for s in name.parts):
     target=out/name;target.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();n=0
     with z.open(info) as f,target.open('wb') as w:
      for b in iter(lambda:f.read(4*1024**2),b''):job.check();w.write(b);h.update(b);n+=len(b)
     assert n==info.file_size;inventory[-1]['extracted_sha256']=h.hexdigest()
  write(d/'MRANNOTATOR_ZIP_INVENTORY.json',inventory);assert any('extracted_sha256' in x for x in inventory),'TASK002_NOT_FOUND'
  subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','--retries','2','--timeout','40','nnunetv2==2.6.2'],check=True)
  write(ROOT/'FIBULA_MODEL_PREPARED.json',{'state':'MODEL_ONLY_RESEARCH_PILOT_PREPARED_NOT_VALIDATED_ON_CORONAL_RSS','training_ready':False,'extracted_files':[{k:v for k,v in x.items() if k in ['path','bytes','extracted_sha256']} for x in inventory if 'extracted_sha256' in x],'prepared_unix':time.time()})
  job.progress.update(phase='fibula_model_research_pilot_prepared')
if __name__=='__main__':main()
