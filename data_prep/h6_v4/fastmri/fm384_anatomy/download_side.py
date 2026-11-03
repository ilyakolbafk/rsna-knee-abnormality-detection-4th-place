"""Receive pinned public R6 side-only checkpoint; signed transport stays private."""
import os
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import json,hashlib,urllib.request,urllib.parse,time
from pathlib import Path
from anatomy_job import ROOT,OwnedJob,write
EXPECTED='ab623e2b619db4917639220fed15202e8c8aedd07c8e298a66a2afc0f153650a'
def main():
 with OwnedJob('receiving_pinned_R6_side_only_weight',downloads=1) as job:
  dst=ROOT/'weights/side_cls.pt';dst.parent.mkdir(exist_ok=True)
  url=json.loads((ROOT/'private/r6_weight_download.json').read_text())['url'];u=urllib.parse.urlsplit(url)
  assert u.scheme=='https' and u.hostname=='storage.googleapis.com'
  tmp=dst.with_suffix('.pt.partial');h=hashlib.sha256();n=0
  with urllib.request.urlopen(url,timeout=45) as response,tmp.open('wb') as f:
   assert response.status==200 and urllib.parse.urlsplit(response.url).hostname=='storage.googleapis.com'
   assert int(response.headers['Content-Length'])==85255211
   for b in iter(lambda:response.read(1024**2),b''):
    job.check();f.write(b);h.update(b);n+=len(b)
   f.flush();os.fsync(f.fileno())
  assert n==85255211 and h.hexdigest()==EXPECTED,'SIDE_WEIGHT_RECEIVER_SHA_MISMATCH';os.replace(tmp,dst)
  import torch,torchvision
  obj=torch.load(dst,map_location='cpu',weights_only=True)
  keys=list(obj)
  write(ROOT/'SIDE_WEIGHT_RECEIPT.json',{'state':'PASS_WEIGHT_RECEIVER_SHA_ONLY','source':'heliosli/rsna-knee-r6-frozen-fe055406-w1/1/side_cls.pt','bytes':n,'sha256':h.hexdigest(),'type':type(obj).__name__,'keys_first_last':keys[:8]+keys[-8:],'torch':torch.__version__,'torchvision':torchvision.__version__,'loaded_weights_only':True,'received_unix':time.time()})
  job.progress.update(phase='R6_side_weight_received_and_safely_inspected')
if __name__=='__main__':main()
