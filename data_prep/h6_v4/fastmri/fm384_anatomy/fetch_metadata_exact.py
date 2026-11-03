"""Copy whitelisted non-image metadata with receiver SHA, never image/weights/secrets."""
from pathlib import Path
import json,sys,base64,zlib,hashlib
T=Path(__file__).resolve().parent
sys.path.insert(0,str(T.parent/'fm384-v1-build-56967269-20261021'))
from remote import run

def fetch(paths):
 allowed_csv={'training_pixel_store/provenance/knee.original.csv','training_pixel_store/provenance/knee_file_list.csv'}
 assert all(not Path(p).is_absolute() and '..' not in Path(p).parts and not p.startswith('private') and (Path(p).suffix in ['.json','.py','.md','.yaml','.txt'] or p in allowed_csv) for p in paths)
 code="""from pathlib import Path
import json,hashlib,zlib,base64
r=Path('/workspace/knee_v4_cache_20261021/fastmri_v1/anatomy_v1');paths=PATHS
out={}
for n in paths:
 p=r/n;b=p.read_bytes();out[n]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'base64':base64.b64encode(b).decode()}
b=json.dumps(out,separators=(',',':')).encode();print(base64.b64encode(zlib.compress(b,6)).decode())
""".replace('PATHS',repr(paths));payload=run(code,timeout=120);data=json.loads(zlib.decompress(base64.b64decode(payload)));out=T/'remote_metadata';receipt={}
 for name,v in data.items():
  b=base64.b64decode(v['base64']);assert len(b)==v['bytes'] and hashlib.sha256(b).hexdigest()==v['sha256'];p=out/name;p.parent.mkdir(exist_ok=True,parents=True);p.write_bytes(b);assert hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256'];receipt[name]={'sha256':v['sha256'],'bytes':len(b)}
 (out/'RECEIVER_LAST_BATCH.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt
if __name__=='__main__':
 paths=sys.argv[1:];print(json.dumps({'copied_files':len(fetch(paths)),'medical_images_exported':False}))
