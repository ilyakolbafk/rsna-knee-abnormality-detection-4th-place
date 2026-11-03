"""Official RSS acquisition stage. This is NOT an fm384 producer or training cache.
Only its owned namespace is writable. URLs are read from private input, never logged.
"""
import argparse,csv,fcntl,hashlib,http.client,io,json,lzma,os,re,shutil,socket,ssl,sys,threading,time,tempfile,subprocess,urllib.error,urllib.parse,urllib.request,xml.etree.ElementTree as ET
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import h5py
import numpy as np
ROOT=Path('/workspace/knee_v4_cache_20261021/fastmri_v1');SHARED=ROOT.parent/'shared';G=1024**3
RUN='fm384-v1-56967269-20261021';POLICY='official-rss-lossless-source-stage-v4'
STOP=threading.Event()
class Pressure(RuntimeError):pass

def write(path,value):
 path=Path(path)
 fd,tmpname=tempfile.mkstemp(prefix=path.name+'.tmp.',dir=path.parent);tmp=Path(tmpname)
 with os.fdopen(fd,'w') as f:json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path);syncdir(path.parent)
def syncdir(p):
 fd=os.open(p,os.O_RDONLY);os.fsync(fd);os.close(fd)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def stat(p):
 x=Path(p).stat();return {'bytes':x.st_size,'inode':x.st_ino,'mtime_ns':x.st_mtime_ns,'ctime_ns':x.st_ctime_ns}
def ticks(pid):return Path('/proc/%s/stat'%pid).read_text().rsplit(')',1)[1].split()[19]
def annotation_inputs():
 p=ROOT/'inputs';rows=list(csv.reader((p/'knee_file_list.csv').open()));stems=[r[0] for r in rows if r and re.fullmatch(r'file\d+',r[0])];extra=[r for r in rows if r and not re.fullmatch(r'file\d+',r[0])]
 assert len(stems)==len(set(stems))==1172 and extra==[['Knee data']] and stems[0]=='file1000000'
 anns={s:[] for s in stems}
 with (p/'knee.csv').open() as f:
  for r in csv.DictReader(f):assert r['file'] in anns;anns[r['file']].append(r)
 assert sum(map(len,anns.values()))==16166
 return set(stems),anns

def admission(active=True):
 with (SHARED/'admission.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);d=json.loads((SHARED/'admission.json').read_text());assert d['version']==1 and int(d['target']['instance_id'])==56967269
  old=d['tasks'].get('fastmri_v1')
  if old and old.get('PID') not in [None,os.getpid()]:
   pid=old['PID']
   if Path('/proc/%s'%pid).exists() and str(old.get('ctime'))==ticks(pid):raise RuntimeError('LIVE_OWN_TASK_CONFLICT')
  own={'run_id':RUN,'PID':os.getpid(),'ctime':ticks(os.getpid()),'argv':sys.argv,'heartbeat_unix':time.time(),'reserved_disk_bytes':70*G if active else 0,'reserved_ram_bytes':4*G if active else 0,'cpu_threads':6 if active else 0,'download_workers':1 if active else 0,'convert_workers':1 if active else 0,'xz_threads':4 if active else 0,'scratch_paths':[str(ROOT/'scratch')],'active':active}
  peers=[x for k,x in d['tasks'].items() if k!='fastmri_v1'];l=d['limits']
  if not active:
   d['tasks']['fastmri_v1']=own;write(SHARED/'admission.json',d);return
  assert sum(x.get('cpu_threads',0) for x in peers)+own['cpu_threads']<=l['max_total_cpu_threads']
  assert sum(x.get('download_workers',0) for x in peers)+own['download_workers']<=l['max_total_download_workers']
  mi={a:int(b.split()[0])*1024 for a,b in (x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines())}
  cg=Path('/sys/fs/cgroup/memory');usage=int((cg/'memory.usage_in_bytes').read_text());memstat=dict(x.split() for x in (cg/'memory.stat').read_text().splitlines());head=int((cg/'memory.limit_in_bytes').read_text())-usage+int(memstat.get('total_inactive_file',0))
  promised=sum(x.get('reserved_ram_bytes',0) for x in peers)+own['reserved_ram_bytes']
  pressure=[]
  if min(mi['MemAvailable'],head)<l['memory_reserve_bytes']+promised:pressure.append('RAM_ADMISSION')
  if shutil.disk_usage(ROOT).free<l['disk_reserve_bytes']+sum(x.get('reserved_disk_bytes',0) for x in peers)+own['reserved_disk_bytes']:pressure.append('DISK_ADMISSION')
  own['admission_pressure']=pressure
  d['tasks']['fastmri_v1']=own;write(SHARED/'admission.json',d)
  if pressure:raise Pressure(','.join(pressure))

def wait_admission(state):
 while True:
  if STOP.is_set():raise RuntimeError('STOPPED_AFTER_HEARTBEAT_FAILURE')
  try:admission();state.pop('resource_backoff',None);return
  except Pressure as e:
   state['resource_backoff']=str(e);write(ROOT/'STATE.json',dict(state,updated_unix=time.time(),PID=os.getpid(),ctime=ticks(os.getpid())))
   STOP.wait(20)

class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,req,fp,code,msg,headers,newurl):raise RuntimeError('HTTP_REDIRECT_REFUSED_URL_REDACTED')

class Reader(io.RawIOBase):
 """One HTTP stream, exact-offset reconnect; content length and ETag pinned."""
 def __init__(self,url,split,state):
  u=urllib.parse.urlsplit(url);assert u.scheme=='https' and u.hostname=='fastmri-dataset.s3.amazonaws.com' and not u.fragment
  self.url=url;self.split=split;self.state=state;self.offset=0;self.hash=hashlib.sha256();self.response=None;self.total=None;self.etag=None;self.cancel=threading.Event()
  self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect());self.connect()
  self.identity={'host':u.hostname,'path':u.path,'split':split,'bytes':self.total,'etag':self.etag,'version_id':self.version,'official_sha256':None}
  p=ROOT/'receipts'/('SOURCE_'+split+'.json')
  if p.exists():assert json.loads(p.read_text())==self.identity,'SOURCE_VERSION_CHANGED'
  else:write(p,self.identity)
 def readable(self):return True
 def readinto(self,b):
  d=self.read(len(b));b[:len(d)]=d;return len(d)
 def connect(self):
  if self.cancel.is_set() or self.closed or STOP.is_set():raise RuntimeError('HTTP_READER_CANCELLED')
  headers={'Range':'bytes=%d-'%self.offset,'Accept-Encoding':'identity'}
  if self.etag:headers['If-Match']=self.etag
  req=urllib.request.Request(self.url,headers=headers)
  try:r=self.opener.open(req,timeout=30)
  except urllib.error.HTTPError as e:raise RuntimeError('HTTP_STATUS_%d_URL_REDACTED'%e.code) from None
  if self.cancel.is_set() or self.closed:r.close();raise RuntimeError('HTTP_READER_CANCELLED')
  m=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',r.headers.get('Content-Range',''));assert r.status==206 and m and int(m[1])==self.offset and int(m[2])+1==int(m[3]),'HTTP_RANGE_BINDING'
  total=int(m[3]);etag=r.headers.get('ETag');assert etag,'MISSING_ETAG'
  if self.total is not None:assert (total,etag,r.headers.get('x-amz-version-id'))==(self.total,self.etag,self.version),'SOURCE_VERSION_CHANGED'
  self.total=total;self.etag=etag;self.version=r.headers.get('x-amz-version-id');self.response=r
 def read(self,size=-1):
  if STOP.is_set():raise RuntimeError('STOPPED_AFTER_HEARTBEAT_FAILURE')
  if self.cancel.is_set() or self.closed:raise RuntimeError('HTTP_READER_CANCELLED')
  if size==0:return b''
  if self.offset==self.total:return b''
  size=min(size if size>=0 else 1024**2,self.total-self.offset)
  for retry in range(4):
   if self.cancel.is_set() or self.closed or STOP.is_set():raise RuntimeError('HTTP_READER_CANCELLED')
   try:
    if self.response is None:self.connect()
    b=self.response.read(size)
    if not b:raise OSError('PREMATURE_HTTP_EOF')
    break
   except urllib.error.HTTPError as e:raise RuntimeError('HTTP_STATUS_%d_URL_REDACTED'%e.code) from None
   except (OSError,http.client.IncompleteRead,urllib.error.URLError) as err:
    if self.cancel.is_set() or self.closed or STOP.is_set():raise RuntimeError('HTTP_READER_CANCELLED') from None
    if isinstance(err,http.client.IncompleteRead):
     self.state['bytes_transferred_discarded_partial']+=len(err.partial);self.state['bytes_transferred_this_run']+=len(err.partial)
    if self.response:self.response.close()
    self.response=None
    if retry==3:raise RuntimeError('TRANSIENT_HTTP_RETRY_EXHAUSTED_URL_REDACTED') from None
    self.state['network_reconnects']+=1;self.cancel.wait(min(8,2**retry))
  self.offset+=len(b);self.hash.update(b);self.state['stream_bytes']=self.offset;self.state['stream_total']=self.total;self.state['bytes_transferred_this_run']+=len(b)
  return b
 def close(self):
  self.cancel.set()
  if self.response:
   sock=getattr(getattr(getattr(self.response,'fp',None),'raw',None),'_sock',None)
   if sock:
    try:sock.shutdown(socket.SHUT_RDWR)
    except OSError:pass
   self.response.close()
  super().close()

class StrictXZ:
 """Validate one complete XZ stream and only legal 4-byte zero stream padding.
 Concatenated streams are refused explicitly rather than silently omitted.
 """
 def __init__(self,reader):self.reader=reader;self.decoder=lzma.LZMADecompressor(format=lzma.FORMAT_XZ);self.finished=False;self.padding_bytes=0
 def read(self,n):
  if self.finished:return b''
  out=bytearray()
  while len(out)<n:
   if self.decoder.eof:
    tail=self.decoder.unused_data
    if any(tail):raise RuntimeError('XZ_TRAILING_NONZERO_OR_CONCAT_STREAM')
    self.padding_bytes+=len(tail)
    while True:
     b=self.reader.read(1024**2)
     if not b:break
     if any(b):raise RuntimeError('XZ_TRAILING_NONZERO_OR_CONCAT_STREAM')
     self.padding_bytes+=len(b)
    assert self.padding_bytes%4==0,'XZ_INVALID_STREAM_PADDING'
    self.finished=True;break
   b=self.reader.read(1024**2) if self.decoder.needs_input else b''
   if not b and self.decoder.needs_input:raise RuntimeError('XZ_TRUNCATED_BEFORE_EOF')
   out.extend(self.decoder.decompress(b,max_length=n-len(out)))
  return bytes(out)
 def close(self):pass

class PipeXZ:
 """Bounded one-worker HTTP feed into liblzma's official multithread decoder.
 xz exit0 is required after stdout EOF; CRC/footer/trailing warnings fail closed.
 """
 def __init__(self,reader):
  self.reader=reader;self.finished=False;self.cancel=threading.Event();self.error=None
  executable=shutil.which('xz');assert executable,'XZ_EXECUTABLE_MISSING'
  self.command=[executable,'--format=xz','--decompress','--stdout','--threads=4','--memlimit-decompress=1GiB','--memlimit-mt-decompress=1GiB']
  env={k:v for k,v in os.environ.items() if k not in ['XZ_OPT','XZ_DEFAULTS']}
  self.runtime={'executable':executable,'sha256':sha(executable),'version':subprocess.check_output([executable,'--version'],env=env,text=True).strip(),'XZ_OPT_and_XZ_DEFAULTS_removed':True}
  self.proc=subprocess.Popen(self.command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env)
  def feed():
   try:
    while not self.cancel.is_set():
     b=self.reader.read(1024**2)
     if not b:break
     self.proc.stdin.write(b)
    self.proc.stdin.close()
   except Exception as e:
    if not self.cancel.is_set():self.error=e
    try:self.proc.stdin.close()
    except Exception:pass
  self.thread=threading.Thread(target=feed,daemon=True);self.thread.start()
 def read(self,n):
  b=self.proc.stdout.read(n)
  if b:return b
  self.thread.join(35)
  if self.thread.is_alive():raise RuntimeError('XZ_FEED_THREAD_DID_NOT_FINISH')
  rc=self.proc.wait(timeout=30);diagnostic=self.proc.stderr.read()
  if rc!=0 or diagnostic:raise RuntimeError('XZ_DECODER_INTEGRITY_OR_WARNING_FAILURE_RC_'+str(rc))
  if self.error:raise self.error
  self.finished=True;return b''
 def close(self):
  if not self.finished:
   self.cancel.set()
   if self.proc.poll() is None:self.proc.terminate()
   self.reader.close()
   try:self.proc.wait(timeout=5)
   except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait(timeout=5)
   self.thread.join(35)
   if self.thread.is_alive():raise RuntimeError('XZ_CANCEL_FEED_THREAD_NOT_FINISHED')
  for p in [self.proc.stdin,self.proc.stdout,self.proc.stderr]:
   if p and not p.closed:p.close()

def scalar(x):
 if isinstance(x,bytes):return x.decode(errors='replace')
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 return x

def commit(raw,stem,split,anns,rawsha,source,tar_meta,state):
 final=ROOT/'source_rss'/stem
 binding={'policy':POLICY,'source_identity':source,'raw_sha256':rawsha,'raw_bytes':raw.stat().st_size,'annotation_sha256':hashlib.sha256(json.dumps(anns,sort_keys=True).encode()).hexdigest(),'code_sha256':sha(__file__)}
 if final.exists():
  rec=json.loads((final/'receipt.json').read_text());assert rec['binding']==binding,'RESUME_CONFIG_SOURCE_MISMATCH'
  for name,r in rec['files'].items():assert sha(final/name)==r['sha256'],'RESUME_OUTPUT_SHA_MISMATCH'
  state['reused_source_rss_bytes']+=sum(r['stat']['bytes'] for r in rec['files'].values());return rec
 temp=ROOT/'scratch'/(stem+'.txn')
 if temp.exists():
  journal=temp/'transaction.json'
  if not journal.exists():
   # A crash before the journal was durable: retain all bytes for diagnosis,
   # never reuse or delete unbound output. Re-read this member from source.
   os.rename(temp,temp.with_name(temp.name+'.unbound.'+str(time.time_ns())));syncdir(temp.parent)
  else:
   assert json.loads(journal.read_text())==binding,'INTERRUPTED_TRANSACTION_BINDING_MISMATCH'
   # Confirmed incomplete owned stage; never promote partial files or touch source.
   shutil.rmtree(temp);syncdir(temp.parent)
 temp.mkdir()
 write(temp/'transaction.json',binding)
 with h5py.File(raw,'r') as h:
  assert 'reconstruction_rss' in h,'RSS_MISSING_NO_ESC_FALLBACK';data=h['reconstruction_rss'];assert data.ndim==3 and min(data.shape)>0
  n,ny,nx=data.shape;assert (ny,nx)==(320,320),'NON_320_RSS_GEOMETRY_REVIEW'
  assert data.dtype==np.dtype('float32'),'RSS_DTYPE_REQUIRES_LOSSLESS_POLICY_REVIEW'
  xml=h['ismrmrd_header'][()];xml=xml.encode() if isinstance(xml,str) else bytes(xml);tree=ET.fromstring(xml)
  spaces={}
  for name in ['encodedSpace','reconSpace']:
   sp=tree.find('.//{*}'+name);assert sp is not None
   spaces[name]={sect:{axis:float(sp.find('{*}'+sect+'/{*}'+axis).text) for axis in ['x','y','z']} for sect in ['matrixSize','fieldOfView_mm']}
  rs=spaces['reconSpace'];recon_rows,recon_cols=rs['matrixSize']['x'],rs['matrixSize']['y'];assert int(recon_rows)==recon_rows and int(recon_cols)==recon_cols and recon_rows>=ny and recon_cols>=nx,'RSS_RECON_MATRIX_GEOMETRY'
  spacing=[rs['fieldOfView_mm']['y']/recon_cols,rs['fieldOfView_mm']['x']/recon_rows];assert np.isfinite(spacing).all() and min(spacing)>0
  attrs={k:scalar(v) for k,v in h.attrs.items()}
  # Preserve authentic identity candidates; no synthetic stem-based patient_id.
  candidates={}
  for k,v in attrs.items():
   if any(w in k.lower() for w in ['patient','subject']):candidates['attr:'+k]=v
  for e in tree.iter():
   local=e.tag.split('}')[-1]
   if any(w in local.lower() for w in ['patient','subject']) and e.text and e.text.strip():candidates['xml:'+local]=e.text.strip()
  rawarr=np.lib.format.open_memmap(temp/'rss.f32.npy',mode='w+',dtype=np.float32,shape=data.shape)
  native=np.lib.format.open_memmap(temp/'native320.u8.npy',mode='w+',dtype=np.uint8,shape=data.shape)
  maxima=[]
  for z in range(n):
   a=np.asarray(data[z],np.float32);assert np.isfinite(a).all() and a.min()>=0,'INVALID_RSS_VALUES'
   rawarr[z]=a;display=a[::-1,:];mx=float(display.max());maxima.append(mx);native[z]=(display/mx*255).astype(np.uint8) if mx else np.zeros_like(display,dtype=np.uint8)
  rawarr.flush();native.flush();del rawarr,native
  spatial=[];study=[]
  for row in anns:
   if row['study_level']=='Yes':assert all(not row[k] for k in ['x','y','width','height']);study.append(row);continue
   assert row['study_level']=='No';z=int(row['slice']);xywh=list(map(float,[row[k] for k in ['x','y','width','height']]));assert 0<=z<n and np.isfinite(xywh).all() and min(xywh[2:])>0
   spatial.append({'slice':z,'label_raw':row['label'],'source_xywh':xywh,'coordinate_frame':'fastMRI_plus_flipud_display_frame'})
  write(temp/'boxes.source.json',{'volume_id':stem,'spatial':spatial,'study_level':study,'mapping_applied':False})
  (temp/'ismrmrd_header.xml').write_bytes(xml)
  patient_id=attrs.get('patient_id');patient_id=patient_id if isinstance(patient_id,str) and patient_id.strip() else None
  meta={'volume_id':stem,'split':split,'rss_shape':list(data.shape),'h5_attrs':attrs,'identity_candidates':candidates,'patient_id':patient_id,'patient_id_status':'authentic_H5_patient_id_exam_group' if patient_id else 'not_resolved_from_H5_patient_id_attribute','patient_identity_scope':'Official fastMRI patient_id identifies anonymized examination; stable person identity across repeated examinations is not asserted','spaces':spaces,'spacing_xy_mm':spacing,'header_axis_mapping':'header x=array row, header y=array col','rss_f32_orientation':'unmodified_source_RSS','native320_orientation':'flipud_applied_exactly_once','normalization_native320':'per-slice nonnegative RSS maximum then truncate to uint8; no ImageNet transform','slice_maxima':maxima,'datasets':{k:{'shape':list(h[k].shape),'dtype':str(h[k].dtype)} for k in h if isinstance(h[k],h5py.Dataset)},'private_pixel_parity':False}
  meta['rss_center_crop_origin_in_recon_xy']=[(int(recon_cols)-nx)//2,(int(recon_rows)-ny)//2]
  meta['rss_physical_fov_xy_mm']=[nx*spacing[0],ny*spacing[1]]
  write(temp/'metadata.json',meta)
 files={}
 for p in sorted(temp.iterdir()):
  with p.open('rb') as f:os.fsync(f.fileno())
  files[p.name]={'sha256':sha(p),'stat':stat(p)}
 rec={'volume_id':stem,'split':split,'n_slices':n,'binding':binding,'tar':tar_meta,'files':files,'archive_proof':'per-member complete SHA; full archive completion receipt still required','training_ready':False,'created_unix':time.time()}
 write(temp/'receipt.json',rec);os.rename(temp,final);syncdir(final.parent)
 # Receiver SHA after atomic commit; record actual final inode/stat.
 for name,v in files.items():assert sha(final/name)==v['sha256'] and stat(final/name)==v['stat'],'POST_COMMIT_SHA_STAT'
 return rec

def exact(f,n):
 out=bytearray()
 while len(out)<n:
  b=f.read(min(1024**2,n-len(out)))
  if not b:raise RuntimeError('TRUNCATED_TAR_STREAM')
  out.extend(b)
 return bytes(out)

def collect(split,url,reviewed,anns,state,limit):
 import tarfile
 state.update(split=split,stream_bytes=0,stream_total=0,current_volume=None,phase='streaming_official_RSS')
 reader=Reader(url,split,state);dec=PipeXZ(reader);seen=set();total=0;tarpos=0
 try:
  while True:
   header=exact(dec,512);start=tarpos;tarpos+=512
   if header==bytes(512):
    assert exact(dec,512)==bytes(512),'TAR_SECOND_END_BLOCK';tarpos+=512
    while True:
     b=dec.read(1024**2)
     if not b:break
     assert not any(b),'TAR_TRAILING_NONZERO';tarpos+=len(b)
    break
   member=tarfile.TarInfo.frombuf(header,'utf-8','strict')
   if member.isdir():assert member.size==0;continue
   assert member.isfile() and member.name.endswith('.h5') and '..' not in Path(member.name).parts,'UNEXPECTED_TAR_MEMBER'
   stem=Path(member.name).stem;assert re.fullmatch(r'file\d+',stem) and stem in reviewed and stem not in seen,'UNEXPECTED_DUPLICATE_STEM';seen.add(stem);assert 0<member.size<=4*G
   if STOP.is_set():raise RuntimeError('STOPPED_AFTER_HEARTBEAT_FAILURE')
   wait_admission(state);state.update(current_volume=stem,raw_member_bytes=member.size)
   raw=ROOT/'scratch'/(stem+'.h5.partial');h=hashlib.sha256();remaining=member.size
   with raw.open('wb') as f:
    while remaining:
     b=exact(dec,min(4*1024**2,remaining));f.write(b);h.update(b);remaining-=len(b)
    f.flush();os.fsync(f.fileno())
   padding=(-member.size)%512
   if padding:assert exact(dec,padding)==bytes(padding),'NONZERO_TAR_MEMBER_PADDING'
   tarpos+=member.size+padding
   rec=commit(raw,stem,split,anns[stem],h.hexdigest(),reader.identity,{'member_name':member.name,'header_offset':start,'data_offset':start+512,'bytes':member.size},state)
   # All full pixels, metadata, source boxes and source receipt durable + SHA verified.
   raw.unlink();syncdir(raw.parent);total+=1;state['committed']=len(list((ROOT/'source_rss').glob('file*/receipt.json')));state['slices_committed']=sum(json.loads(p.read_text())['n_slices'] for p in (ROOT/'source_rss').glob('file*/receipt.json'));state['verified_this_run']+=1
   write(ROOT/'STATE.json',dict(state,updated_unix=time.time(),PID=os.getpid(),ctime=ticks(os.getpid()),ready=False))
   print(json.dumps({'event':'source_RSS_commit','split':split,'stem':stem,'n_slices':rec['n_slices'],'split_completed':total,'received_archive_bytes':reader.offset}),flush=True)
   if limit and total>=limit:
    write(ROOT/'qa'/('CANARY_SOURCE_'+split+'.json'),{'split':split,'volumes':sorted(seen),'archive_partial':True,'archive_bytes_received':reader.offset,'archive_integrity_complete':False});return
  assert reader.offset==reader.total,'INCOMPLETE_HTTP_ARCHIVE'
  expected={'train':973,'val':199}[split];assert len(seen)==expected,'SPLIT_EXPECTED_COUNT'
  assert dec.finished,'XZ_NOT_FULLY_VALIDATED'
  proof={'split':split,'source_identity':reader.identity,'local_full_archive_sha256':reader.hash.hexdigest(),'official_full_archive_sha256':None,'full_compressed_bytes_read':reader.offset,'xz_footer_crc':'xz stdout fully drained; complete feed; exit0 and empty stderr required (CRC/footer/concatenation/padding checked by liblzma5.4.5)','xz_command':dec.command,'xz_runtime':dec.runtime,'tar_header_checksum_member_padding_two_zero_end_blocks_trailing_zeros':'verified','volume_count':len(seen),'stems':sorted(seen),'completed_unix':time.time()}
  write(ROOT/'receipts'/('ARCHIVE_COMPLETE_'+split+'.json'),proof)
 finally:dec.close();reader.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--canary-per-split',type=int,default=0);args=p.parse_args()
 b=json.loads((ROOT/'TARGET_VERIFIED.json').read_text());assert socket.gethostname()==b['hostname'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==b['boot_id'] and Path('/root/.vast_containerlabel').read_text().strip()=='C.56967269'
 os.umask(0o077)
 lock=(ROOT/'singleton.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state={'run_id':RUN,'phase':'starting','started_unix':time.time(),'committed':len(list((ROOT/'source_rss').glob('file*/receipt.json'))),'slices_committed':0,'bytes_transferred_this_run':0,'bytes_transferred_discarded_partial':0,'reused_source_rss_bytes':0,'network_reconnects':0,'verified_this_run':0,'blockers':['private_FM_producer_anatomical_anchor_and_mapping_absent','patient_mapping_pending_raw_header_check'],'ready':False}
 stop=STOP;wait_admission(state)
 def heartbeat():
  while not stop.wait(40):
   try:
    admission();elapsed=max(1,time.time()-state['started_unix']);v=dict(state,PID=os.getpid(),ctime=ticks(os.getpid()),argv=sys.argv,updated_unix=time.time(),free_disk_bytes=shutil.disk_usage(ROOT).free,transfer_Bps=state['bytes_transferred_this_run']/elapsed,ETA_seconds=None);write(ROOT/'STATE.json',v)
   except Pressure as e:state['resource_backoff']=str(e)
   except Exception as e:state['heartbeat_error']=type(e).__name__;stop.set()
 thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
 try:
  reviewed,anns=annotation_inputs();urls=json.loads((ROOT/'private/links.json').read_text());assert set(urls)=={'train','val'}
  for split in ['val','train']:collect(split,urls[split],reviewed,anns,state,args.canary_per_split)
  state['phase']='canary_source_complete' if args.canary_per_split else 'source_corpus_complete_geometry_identity_pending'
  if not args.canary_per_split:
   actual={p.parent.name for p in (ROOT/'source_rss').glob('file*/receipt.json')};assert actual==reviewed,'FINAL_REVIEWED_SET_MISMATCH'
  stop.set();thread.join();write(ROOT/'STATE.json',dict(state,updated_unix=time.time(),PID=os.getpid(),ctime=ticks(os.getpid()),process_terminal=True))
 except Exception as e:
  state.update(phase='failed',error_type=type(e).__name__,error=str(e) if 'http' not in str(e).lower() else 'HTTP_failure_URL_redacted',updated_unix=time.time(),process_terminal=True)
  stop.set();thread.join();write(ROOT/'STATE.json',state);print(json.dumps({'event':'failed','error_type':type(e).__name__,'error':state['error']}),flush=True);raise SystemExit(1)
 finally:stop.set();thread.join();admission(active=False)
if __name__=='__main__':main()
