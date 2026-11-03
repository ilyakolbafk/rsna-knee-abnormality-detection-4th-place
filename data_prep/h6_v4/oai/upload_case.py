from common import *
import shlex
import uuid
import threading
from recipe_registry import validate_audit

PACK_LIMIT=threading.Semaphore(2)

def upload(case):
    enabled=json.loads((RUN/'RESOURCE_PLAN.json').read_text()).get('transport_codec')=='delta_zstd'
    try:return _upload(case,enabled)
    except Exception as e:
        if enabled and any(x in str(e) for x in ['TRANSPORT_','RECONSTRUCTED_ORIGINAL_NPZ_SHA_MISMATCH','RAW_ENTRY_SHA_MISMATCH','Bad magic number','Bad CRC']):
            dump(RUN/'transport_fallbacks'/f'{case}.json',{'case_id':case,'reason':str(e),'unix':time.time(),'action':'FALLBACK_ORIGINAL_NPZ_NO_PIXEL_CHANGE'})
            return _upload(case,False)
        raise

def _upload(case,use_transport):
    timings={'case_id':case,'start_unix':time.time()};started=time.monotonic()
    b=binding();path=RUN/'cache'/f'{case}.npz';ap=RUN/'cache'/f'{case}.audit.json'
    audit=json.loads(ap.read_text());config=validate_audit(case,audit)
    assert audit['recipe_sha256']==config['semantic_sha256'] and sha(path)==audit['sha256']
    nonce=uuid.uuid4().hex;expected_audit_sha=sha(ap)
    transport=None;wire_path=path
    if use_transport:
        try:
            from transport_codec import pack_zstd,FORMAT
            wire_path=RUN/'transport_cache'/f'{case}.transport.zst';wire_path.parent.mkdir(exist_ok=True)
            with PACK_LIMIT:
                import psutil
                needed=int(audit['rows'])*384*384*6+64*2**20
                if psutil.virtual_memory().available<needed+2**30:
                    raise RuntimeError('TRANSPORT_LOCAL_MEMORY_ADMISSION')
                packed=pack_zstd(path,wire_path)
        except Exception as error:
            raise RuntimeError('TRANSPORT_LOCAL_ENCODE_FAILED:'+type(error).__name__+':'+str(error)) from error
        if packed['original_sha256']!=audit['sha256']:raise ValueError('ORIGINAL_NPZ_CHANGED_DURING_TRANSPORT_PACK')
        if packed['bytes']<audit['bytes']:
            transport={'format':FORMAT,'file':f'{case}.{nonce}.transport.zst','bytes':packed['bytes'],'sha256':packed['sha256'],
                       'codec_sha256':sha(RUN/'code/transport_codec.py')}
            timings.update(pack_seconds=packed['seconds'],wire_bytes=packed['bytes'],transport_format=FORMAT)
        else:wire_path=path
    mkdir=f"from pathlib import Path\nimport json\np=Path({REMOTE!r})\nassert Path('/root/.vast_containerlabel').read_text().strip()=='C.56967269'\n(p/'incoming'/{case!r}).mkdir(parents=True,exist_ok=True)\nprint(json.dumps({{'status':'READY_FOR_UPLOAD'}}))"
    remote_python(mkdir)
    timings['prepare_seconds']=time.monotonic()-started
    transfer_start=time.monotonic()
    wire_alias=None
    if transport:
        wire_alias=wire_path.with_name(transport['file']);os.link(wire_path,wire_alias)
        cmd=['rsync','-a','--partial','--timeout=120','-e',shlex.join(SSH[:-1]),str(wire_alias),str(ap),f'{SSH_HOST}:{REMOTE}/incoming/{case}/']
    else:
        cmd=['rsync','-a','--partial','--timeout=120','-e',shlex.join(SSH[:-1]),str(path),str(ap),f'{SSH_HOST}:{REMOTE}/incoming/{case}/']
    try:p=subprocess.run(cmd,capture_output=True,text=True,timeout=900)
    finally:
        if wire_alias:wire_alias.unlink(missing_ok=True)
    if p.returncode:raise RuntimeError(f'RSYNC_EXIT_{p.returncode}: '+p.stderr[-500:])
    timings['transfer_seconds']=time.monotonic()-transfer_start
    start=time.monotonic()
    code=f'''
from pathlib import Path
import json,os,time,fcntl
p=Path({REMOTE!r})
assert Path('/root/.vast_containerlabel').read_text().strip()=='C.56967269'
assert os.uname().nodename=={b['hostname']!r}
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()=={b['boot_id']!r}
request={{'case_id':{case!r},'sha256':{audit['sha256']!r},'audit_sha256':{expected_audit_sha!r},'request_nonce':{nonce!r},'unix':time.time(),'transport':{transport!r}}}
t=p/'incoming'/{(case+'.ready.writing')!r}
with t.open('w') as out:
 json.dump(request,out);out.flush();os.fsync(out.fileno())
with (p/'incoming'/{(case+'.request.lock')!r}).open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 os.replace(t,p/'incoming'/{(case+'.ready.json')!r})
deadline=time.monotonic()+180
result={{'receipt':None,'error':None,'status':'RECEIVER_READBACK_TIMEOUT'}}
while time.monotonic()<deadline:
 r=p/'cache_receipts'/{(case+'.json')!r};e=p/'rejections'/{(case+'.json')!r}
 v=json.loads(r.read_text()) if r.exists() else None
 error=json.loads(e.read_text()) if e.exists() else None
 if error and error.get('request_nonce')=={nonce!r}:
  result={{'receipt':None,'error':error}};break
 f=p/'cache'/{(case+'.npz')!r};a=p/'cache'/{(case+'.audit.json')!r}
 if v and v.get('request_nonce')=={nonce!r} and f.exists() and a.exists():
  s=f.stat();t=a.stat()
  if (s.st_size,s.st_ino,s.st_mtime_ns,s.st_ctime_ns,t.st_ino,t.st_mtime_ns)==(v['bytes'],v['inode'],v['mtime_ns'],v['ctime_ns'],v['audit_inode'],v['audit_mtime_ns']):
   result={{'receipt':v,'error':None}};break
 time.sleep(.5)
print(json.dumps(result))
'''
    r=remote_python(code,timeout=205)
    if r['error']:raise RuntimeError('RECEIVER_REJECTION:'+r['error']['reason'])
    if not r['receipt']:raise TimeoutError('RECEIVER_READBACK_TIMEOUT')
    receipt=r['receipt']
    if receipt['request_nonce']!=nonce or receipt['sha256']!=audit['sha256'] or receipt['recipe_sha256']!=audit['recipe_sha256'] or receipt['audit_sha256']!=expected_audit_sha:raise ValueError('RECEIPT_SHA_CONFLICT')
    timings.update(total_seconds=time.monotonic()-started,receiver_wait_seconds=time.monotonic()-start,
                   bytes=receipt['bytes'],request_nonce=nonce,end_unix=time.time())
    dump(RUN/'upload_timings'/f'{case}.json',timings)
    dump(RUN/'receiver_receipts'/f'{case}.json',receipt);return receipt
