import base64,json
from remote import ROOT,run
names=['fm_xz_blocks.py','fm_indexed_source.py','source_indexed.py','test_indexed_transport.py'];payload={n:base64.b64encode((ROOT/n).read_bytes()).decode() for n in names};audit=base64.b64encode((ROOT/'PREFIX_CRC_IMPLEMENTATION_AUDIT.json').read_bytes()).decode()
code=r'''import json,base64,time,subprocess,os,fcntl,threading,hashlib,tarfile
from pathlib import Path
r=Path('/workspace/knee_v4_cache_20261021/fastmri_v1')
for n,v in PAYLOAD.items():(r/'code'/n).write_bytes(base64.b64decode(v))
(r/'qa/PREFIX_CRC_IMPLEMENTATION_AUDIT.json').write_bytes(base64.b64decode(__AUDIT_PAYLOAD__))
env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'};q=subprocess.run([str(r/'venv/bin/python'),str(r/'code/test_indexed_transport.py')],cwd=r/'code',env=env,text=True,capture_output=True);report={'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr};(r/'qa/INDEXED_TRANSPORT_TESTS.json').write_text(json.dumps(report,indent=2));assert q.returncode==0,report
import sys;sys.path.insert(0,str(r/'code'))
import source_collect as sc
from source_indexed import train_receipts,anchor_plan
from fm_indexed_source import Source
from fm_xz_blocks import IndexedDecoded
identity=json.loads((r/'receipts/SOURCE_train.json').read_text());src=Source(json.loads((r/'private/links.json').read_text())['train'],identity);deadline=threading.Timer(50,src.close);deadline.start();decoded=None
try:
 meta,parts=src.index();sc.write(r/'receipts/TRAIN_XZ_INDEX.json',meta)
 for n,v in parts.items():
  p=r/'receipts'/('TRAIN_XZ_'+n.upper()+'.bin');p.write_bytes(v)
 items,end=train_receipts();plan=anchor_plan(meta,items,end);decoded=IndexedDecoded(src,meta,plan['anchor_tar_header'],workers=1);header=sc.exact(decoded,512);member=tarfile.TarInfo.frombuf(header,'utf-8','strict');assert Path(member.name).stem==plan['anchor_stem'];remaining=member.size;h=hashlib.sha256()
 while remaining:
  b=sc.exact(decoded,min(1024**2,remaining));h.update(b);remaining-=len(b)
 assert h.hexdigest()==plan['anchor_raw_sha256'],'REAL_ANCHOR_RAW_SHA_MISMATCH'
 result={'state':'PASS_FRESH_INDEX_AND_REAL_ANCHOR_RAW_SHA','plan':plan,'block_count':meta['block_count'],'source_identity':identity,'xz_index_file_sha256':sc.sha(r/'receipts/TRAIN_XZ_INDEX.json'),'anchor_raw_sha256':h.hexdigest(),'diagnostic_wire_bytes':src.wire_bytes,'archive_full_sha256':None,'transport_tests':report,'completed_unix':time.time()};sc.write(r/'qa/INDEXED_PREP_CANARY.json',result);print(json.dumps(result))
finally:
 deadline.cancel();src.close()
 if decoded:decoded.close()
'''.replace('PAYLOAD.items()',repr(payload)+'.items()').replace('__AUDIT_PAYLOAD__',repr(audit))
encoded=base64.b64encode(code.encode()).decode()
bootstrap="import subprocess,base64;from pathlib import Path;r=Path('/workspace/knee_v4_cache_20261021/fastmri_v1');p=r/'code/prepare_indexed_stage.py';p.write_bytes(base64.b64decode("+repr(encoded)+"));q=subprocess.run([str(r/'venv/bin/python'),str(p)],text=True,capture_output=True);print(q.stdout);print(q.stderr);assert q.returncode==0"
out=run(bootstrap,90);(ROOT/'INDEXED_PREP_CANARY_READBACK.json').write_text(out);print(out)
