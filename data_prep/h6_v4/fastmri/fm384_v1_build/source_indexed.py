"""Current-source XZ block resume with unchanged v4 RSS/metadata commit."""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import bisect,fcntl,hashlib,json,signal,socket,tarfile,threading,time,shutil
from pathlib import Path
import source_collect as sc
from fm_indexed_source import Source
from fm_xz_blocks import IndexedDecoded
R=sc.ROOT;G=1024**3;S=R.parent/'shared'
def sha(p):return sc.sha(Path(p))
def live(pid,ctime):
 try:return sc.ticks(pid)==str(ctime) and '\nState:\tZ' not in Path('/proc/%d/status'%pid).read_text()
 except FileNotFoundError:return False
def train_receipts():
 out=[]
 for p in (R/'source_rss').glob('file*/receipt.json'):
  v=json.loads(p.read_text())
  if v['split']=='train':out.append(dict(stem=v['volume_id'],receipt_sha256=sha(p),raw_sha256=v['binding']['raw_sha256'],tar=v['tar'],source_identity=v['binding']['source_identity'],extractor_code_sha256=v['binding']['code_sha256']))
 out.sort(key=lambda x:x['tar']['header_offset']);end=512
 for v in out:
  t=v['tar'];assert t['header_offset']==end and t['data_offset']==end+512 and t['bytes']>0,'NONCONTIGUOUS_CURRENT_PREFIX'
  end+=512+t['bytes']+(-t['bytes'])%512
 return out,end
def anchor_plan(meta,items,end):
 offsets=[b['uncompressed_offset'] for b in meta['blocks']];uncertain=max(0,bisect.bisect_left(offsets,end)-1)
 if end not in offsets:uncertain=bisect.bisect_right(offsets,end)-1
 boundary=offsets[uncertain];overlap=[v for v in items if v['tar']['header_offset']+512+v['tar']['bytes']+(-v['tar']['bytes'])%512>boundary]
 assert overlap,'NO_COMPLETE_H5_OVERLAPS_BOUNDARY'
 anchor=overlap[0];first=bisect.bisect_right(offsets,anchor['tar']['header_offset'])-1
 assert 0<=first<=uncertain
 return {'durable_tar_cursor':end,'uncertain_block':uncertain,'verified_sequential_prefix_blocks_exclusive':uncertain,'anchor_stem':anchor['stem'],'anchor_tar_header':anchor['tar']['header_offset'],'anchor_raw_sha256':anchor['raw_sha256'],'resume_block':first,'resume_compressed_offset':meta['blocks'][first]['compressed_offset'],'overlap_stems':[v['stem'] for v in overlap]}
def admission(state,active=True):
 with (S/'admission.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);p=S/'admission.json';d=json.loads(p.read_text());old=d['tasks'].get('fastmri_v1',{})
  if old.get('PID') not in [None,os.getpid()] and live(old['PID'],old.get('ctime')):raise RuntimeError('OTHER_LIVE_FASTMRI_SOURCE')
  peers=[v for k,v in d['tasks'].items() if k!='fastmri_v1'];l=d['limits']
  own={'run_id':'fm384-v1-56967269-20261021','PID':os.getpid(),'ctime':sc.ticks(os.getpid()),'argv':[str(R/'venv/bin/python'),str(R/'code/source_indexed.py')],'active':active,'phase':state['phase'],'cpu_threads':4 if active else 0,'download_workers':2 if active else 0,'convert_workers':1 if active else 0,'reserved_ram_bytes':4*G if active else 0,'reserved_disk_bytes':70*G if active else 0,'heartbeat_unix':time.time(),'scratch_paths':[str(R/'scratch')],'TCP_MAXSEG':1200}
  if active:
   assert sum(v.get('cpu_threads',0) for v in peers)+4<=l['max_total_cpu_threads'],'CPU_ADMISSION'
   assert sum(v.get('download_workers',0) for v in peers)+2<=l['max_total_download_workers'],'HTTP_ADMISSION'
   mi={a:int(b.split()[0])*1024 for a,b in (x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines())};cg=Path('/sys/fs/cgroup/memory');ms=dict(x.split() for x in (cg/'memory.stat').read_text().splitlines());head=int((cg/'memory.limit_in_bytes').read_text())-int((cg/'memory.usage_in_bytes').read_text())+int(ms.get('total_inactive_file',0))
   assert min(head,mi['MemAvailable'])>l['memory_reserve_bytes']+sum(v.get('reserved_ram_bytes',0) for v in peers)+4*G,'RAM_ADMISSION'
   assert shutil.disk_usage(R).free>l['disk_reserve_bytes']+sum(v.get('reserved_disk_bytes',0) for v in peers)+70*G,'DISK_ADMISSION'
  d['tasks']['fastmri_v1']=own;sc.write(p,d)
def main():
 raise RuntimeError('INACTIVE_PROTOTYPE: frozen prefix receipt/cursor/runtime binding and continuous CRC coverage must be implemented and reviewed before any corpus launch')
 target=json.loads((R/'TARGET_VERIFIED.json').read_text());assert socket.gethostname()==target['hostname'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==target['boot_id'] and Path('/root/.vast_containerlabel').read_text().strip()=='C.56967269'
 lock=(R/'singleton.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 reviewed,anns=sc.annotation_inputs();identity=json.loads((R/'receipts/SOURCE_train.json').read_text());meta=json.loads((R/'receipts/TRAIN_XZ_INDEX.json').read_text());checkpoint=json.loads((R/'receipts/PREFIX_STREAM_CHECKPOINT.json').read_text());assert checkpoint['source_identity']==identity and checkpoint['xz_index_sha256']==sha(R/'receipts/TRAIN_XZ_INDEX.json')
 items,end=train_receipts();plan=anchor_plan(meta,items,end);assert plan['resume_block']<=checkpoint['verified_sequential_prefix_blocks_exclusive'] or all((R/'receipts/train_indexed_blocks'/('%05d.json'%i)).exists() for i in range(checkpoint['verified_sequential_prefix_blocks_exclusive'],plan['resume_block']))
 prefix=[v for v in items if v['tar']['header_offset']<plan['anchor_tar_header']];seen=set(v['stem'] for v in prefix);anchor_passed=False
 state={'run_id':'fm384-v1-56967269-20261021','phase':'streaming_official_RSS_indexed_v5','split':'train','started_unix':time.time(),'bytes_transferred_this_run':0,'bytes_transferred_discarded_partial':0,'reused_source_rss_bytes':0,'reused_prior_verified_archive_bytes':16050975296,'stream_bytes':plan['resume_compressed_offset'],'stream_total':identity['bytes'],'resume_compressed_offset':plan['resume_compressed_offset'],'committed':len(list((R/'source_rss').glob('file*/receipt.json'))),'slices_committed':0,'ready':False,'HTTP_workers':2,'TCP_MAXSEG':1200,'archive_full_sha256':None,'archive_proof_mode':'sequential_prefix_plus_indexed_CRC64_suffix'}
 stop=threading.Event();source=None;decoded=None;error=[]
 def heartbeat():
  while not stop.wait(30):
   try:admission(state);sc.write(R/'STATE.json',dict(state,PID=os.getpid(),ctime=sc.ticks(os.getpid()),updated_unix=time.time()))
   except Exception as e:error.append(type(e).__name__);stop.set()
 admission(state);thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
 try:
  def wire(s):
   if stop.is_set():raise RuntimeError('STOP_AFTER_ADMISSION_FAILURE')
   state['bytes_transferred_this_run']=s.wire_bytes
  source=Source(json.loads((R/'private/links.json').read_text())['train'],identity,wire)
  fresh,parts=source.index();assert fresh==meta,'CURRENT_XZ_INDEX_CHANGED'
  blockdir=R/'receipts/train_indexed_blocks';blockdir.mkdir(exist_ok=True)
  def block_done():
   record=decoded.records[-1];i=record['block_index'];record=dict(record,source_identity=identity,xz_index_sha256=meta['index_sha256'],transport_code_sha256=sha(__file__))
   p=blockdir/('%05d.json'%i)
   if p.exists():assert json.loads(p.read_text())==record,'BLOCK_RESUME_PROOF_CHANGED'
   else:sc.write(p,record)
   state.update(stream_bytes=record['compressed_offset']+record['padded_size'],verified_suffix_blocks=len(decoded.records))
  decoded=IndexedDecoded(source,meta,plan['anchor_tar_header'],workers=2,callback=block_done);tarpos=plan['anchor_tar_header']
  while True:
   if stop.is_set():raise RuntimeError('STOP_AFTER_ADMISSION_FAILURE')
   header=sc.exact(decoded,512);start=tarpos;tarpos+=512
   if header==bytes(512):
    assert sc.exact(decoded,512)==bytes(512),'TAR_SECOND_END_BLOCK';tarpos+=512
    while True:
     b=decoded.read(1024**2)
     if not b:break
     assert not any(b),'TAR_TRAILING_NONZERO';tarpos+=len(b)
    break
   m=tarfile.TarInfo.frombuf(header,'utf-8','strict');assert m.isfile() and m.name.endswith('.h5') and '..' not in Path(m.name).parts,'UNEXPECTED_TAR_MEMBER'
   stem=Path(m.name).stem;assert stem in reviewed and stem not in seen and 0<m.size<=4*G,'UNEXPECTED_DUPLICATE_STEM';seen.add(stem);state.update(current_volume=stem,raw_member_bytes=m.size)
   raw=R/'scratch'/(stem+'.h5.partial');h=hashlib.sha256();remaining=m.size
   with raw.open('wb') as f:
    while remaining:
     b=sc.exact(decoded,min(4*1024**2,remaining));f.write(b);h.update(b);remaining-=len(b)
    f.flush();os.fsync(f.fileno())
   pad=(-m.size)%512
   if pad:assert sc.exact(decoded,pad)==bytes(pad),'NONZERO_TAR_MEMBER_PADDING'
   tarpos+=m.size+pad
   if not anchor_passed:
    assert stem==plan['anchor_stem'] and start==plan['anchor_tar_header'] and h.hexdigest()==plan['anchor_raw_sha256'],'ANCHOR_H5_RAW_SHA_MISMATCH';anchor_passed=True
   sc.commit(raw,stem,'train',anns[stem],h.hexdigest(),identity,{'member_name':m.name,'header_offset':start,'data_offset':start+512,'bytes':m.size},state)
   raw.unlink();sc.syncdir(raw.parent);state.update(committed=len(list((R/'source_rss').glob('file*/receipt.json'))),slices_committed=sum(json.loads(p.read_text())['n_slices'] for p in (R/'source_rss').glob('file*/receipt.json')),last_complete_tar_cursor=tarpos)
   sc.write(R/'STATE.json',dict(state,PID=os.getpid(),ctime=sc.ticks(os.getpid()),updated_unix=time.time()))
   print(json.dumps({'event':'indexed_RSS_commit','stem':stem,'committed':state['committed'],'new_wire_bytes':source.wire_bytes}),flush=True)
  assert anchor_passed and len(seen)==973 and tarpos==meta['uncompressed_bytes'],'INDEXED_TAR_COVERAGE_INCOMPLETE'
  assert len(decoded.records)==meta['block_count']-decoded.first,'SUFFIX_BLOCK_COVERAGE_INCOMPLETE'
  records=[json.loads((blockdir/('%05d.json'%i)).read_text()) for i in range(decoded.first,meta['block_count'])];sc.write(R/'receipts/TRAIN_SUFFIX_BLOCKS.json',records)
  proof={'split':'train','source_identity':identity,'verification_mode':'sequential_prefix_plus_indexed_CRC64_suffix_v5','local_full_archive_sha256':None,'official_full_archive_sha256':None,'full_archive_received_this_session':False,'full_compressed_bytes_read':None,'received_compressed_bytes_this_run':source.wire_bytes,'xz_footer_crc':'fresh header/index/footer CRC; original CRC64 of each suffix block; prefix ordered-output proof checkpoint','xz_index_sha256':sha(R/'receipts/TRAIN_XZ_INDEX.json'),'prefix_checkpoint_sha256':sha(R/'receipts/PREFIX_STREAM_CHECKPOINT.json'),'suffix_manifest_sha256':sha(R/'receipts/TRAIN_SUFFIX_BLOCKS.json'),'suffix_first_block':decoded.first,'suffix_block_count':len(records),'anchor':plan,'anchor_raw_SHA_rechecked':True,'tar_header_checksum_member_padding_two_zero_end_blocks_trailing_zeros':'verified; prefix contiguous receipts + anchor/suffix parser','volume_count':973,'stems':sorted(seen),'transport_code_sha256':{n:sha(R/'code'/n) for n in ['source_indexed.py','fm_indexed_source.py','fm_xz_blocks.py']},'completed_unix':time.time()}
  sc.write(R/'receipts/ARCHIVE_COMPLETE_train.json',proof);assert {p.parent.name for p in (R/'source_rss').glob('file*/receipt.json')}==reviewed
  state.update(phase='source_corpus_complete_geometry_identity_pending',stream_bytes=identity['bytes'],archive_proof_mode=proof['verification_mode'])
 except Exception as e:
  state.update(phase='failed',error_type=type(e).__name__,error=str(e) if 'http' not in str(e).lower() else 'HTTP_FAILURE_URL_REDACTED');raise
 finally:
  stop.set()
  if source:source.close()
  if decoded:decoded.close()
  thread.join();sc.write(R/'STATE.json',dict(state,PID=os.getpid(),ctime=sc.ticks(os.getpid()),updated_unix=time.time(),process_terminal=True));admission(state,False)
if __name__=='__main__':main()
