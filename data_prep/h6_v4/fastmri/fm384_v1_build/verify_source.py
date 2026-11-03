"""Destination-only complete source-stage verification; no private producer claim."""
import csv,hashlib,json,os,re,time,collections,sys
from pathlib import Path
import numpy as np
ROOT=Path('/workspace/knee_v4_cache_20261021/fastmri_v1')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def main(progress=None):
 rows=list(csv.reader((ROOT/'inputs/knee_file_list.csv').open()));expected={r[0] for r in rows if r and re.fullmatch(r'file\d+',r[0])}
 assert len(expected)==1172
 anns={s:[] for s in expected}
 with (ROOT/'inputs/knee.csv').open() as f:
  for r in csv.DictReader(f):anns[r['file']].append(r)
 paths={p.parent.name:p.parent for p in (ROOT/'source_rss').glob('file*/receipt.json')};assert set(paths)==expected,'SOURCE1172_NOT_COMPLETE'
 proofs={s:json.loads((ROOT/'receipts'/('ARCHIVE_COMPLETE_'+s+'.json')).read_text()) for s in ['train','val']}
 for split,proof in proofs.items():
  identity=proof['source_identity'];assert proof['split']==identity['split']==split
  assert proof['volume_count']==len(proof['stems'])==len(set(proof['stems']))=={'train':973,'val':199}[split]
  assert proof['full_compressed_bytes_read']==identity['bytes'] and identity['bytes']>0 and isinstance(identity['etag'],str) and identity['etag']
  assert identity['host']=='fastmri-dataset.s3.amazonaws.com' and re.fullmatch(r'[0-9a-f]{64}',proof['local_full_archive_sha256'])
  assert proof['official_full_archive_sha256'] is None and proof['xz_runtime']['XZ_OPT_and_XZ_DEFAULTS_removed'] is True
  assert re.fullmatch(r'[0-9a-f]{64}',proof['xz_runtime']['sha256']) and '--format=xz' in proof['xz_command'] and proof['xz_footer_crc']
 assert set(proofs['train']['stems']).isdisjoint(proofs['val']['stems']) and set(proofs['train']['stems'])|set(proofs['val']['stems'])==expected
 code=sha(ROOT/'code/source_collect.py');items=[];splits=collections.Counter();slices=collections.Counter();category=collections.Counter();groups=collections.defaultdict(list);total=0;spatial=study=0
 for i,stem in enumerate(sorted(expected)):
  if progress is not None and progress.get('abort_reason'):raise RuntimeError('SOURCE_VERIFICATION_ABORT_REQUESTED')
  p=paths[stem];rec=json.loads((p/'receipt.json').read_text());meta=json.loads((p/'metadata.json').read_text());boxes=json.loads((p/'boxes.source.json').read_text())
  assert rec['volume_id']==meta['volume_id']==boxes['volume_id']==stem and rec['binding']['code_sha256']==code
  assert rec['binding']['policy']=='official-rss-lossless-source-stage-v4'
  assert rec['split']==meta['split'] and stem in proofs[rec['split']]['stems']
  assert rec['binding']['source_identity']==proofs[rec['split']]['source_identity'],'MEMBER_ARCHIVE_SOURCE_VERSION_MISMATCH'
  assert rec['binding']['annotation_sha256']==hashlib.sha256(json.dumps(anns[stem],sort_keys=True).encode()).hexdigest()
  for name,v in rec['files'].items():
   f=p/name;st=f.stat();before=(st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns);assert sha(f)==v['sha256'],'SOURCE_FILE_SHA_MISMATCH';st=f.stat();assert before==(st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns),'FILE_CHANGED_DURING_SHA'
   assert st.st_ino==v['stat']['inode'] and st.st_size==v['stat']['bytes'] and st.st_mtime_ns==v['stat']['mtime_ns'] and st.st_ctime_ns==v['stat']['ctime_ns'],'RECEIPT_INODE_STAT_MISMATCH';total+=st.st_size
  rss=np.load(p/'rss.f32.npy',mmap_mode='r');u8=np.load(p/'native320.u8.npy',mmap_mode='r');assert rss.dtype==np.float32 and u8.dtype==np.uint8 and rss.shape==u8.shape==tuple(meta['rss_shape'])
  assert rec['n_slices']==len(rss) and meta['rss_f32_orientation']=='unmodified_source_RSS' and meta['native320_orientation']=='flipud_applied_exactly_once'
  # Full per-slice equality to untouched RSS, not only selected samples.
  for z in range(len(rss)):
   a=np.asarray(rss[z]);assert np.isfinite(a).all() and a.min()>=0;mx=float(a.max());assert mx==meta['slice_maxima'][z]
   target=(a[::-1,:]/mx*255).astype(np.uint8) if mx else np.zeros_like(a,np.uint8);assert np.array_equal(target,u8[z]),'RSS_FLIP_NORMALIZATION_PARITY'
  pid=meta['patient_id'];assert isinstance(pid,str) and pid and pid==meta['h5_attrs']['patient_id'],'AUTHENTIC_PATIENT_ID_REQUIRED'
  flag=int(hashlib.sha1(stem.encode()).hexdigest(),16)%10==0;groups[pid].append((stem,flag));splits[rec['split']]+=1;slices[rec['split']]+=rec['n_slices'];spatial+=len(boxes['spatial']);study+=len(boxes['study_level'])
  source_sp=[r for r in anns[stem] if r['study_level']=='No'];assert len(source_sp)==len(boxes['spatial'])
  for r,b in zip(source_sp,boxes['spatial']):
   assert b['label_raw']==r['label'] and b['slice']==int(r['slice']) and b['source_xywh']==[float(r[k]) for k in ['x','y','width','height']];category[b['label_raw']]+=1
  assert boxes['study_level']==[r for r in anns[stem] if r['study_level']=='Yes']
  items.append({'stem':stem,'split':rec['split'],'n_slices':rec['n_slices'],'patient_id':pid,'stem_holdout':flag,'raw_h5_sha256':rec['binding']['raw_sha256'],'raw_h5_bytes':rec['binding']['raw_bytes'],'receipt_sha256':sha(p/'receipt.json'),'files':rec['files']})
  if progress is not None:progress.update(phase='source_fullSHA_verification',verified_volumes=i+1,expected_volumes=1172,committed=1172,slices_committed=sum(slices.values()),ready=False)
  if (i+1)%50==0:print(json.dumps({'verification_progress':i+1,'expected':1172}),flush=True)
 conflicts={pid:[s for s,f in members] for pid,members in groups.items() if len({f for s,f in members})>1}
 # A possible adaptation is group closure; never assign a seed-holdout exam to train.
 closure={s:any(f for _,f in members) for members in groups.values() for s,_ in members}
 for x in items:x['patient_safe_closure_holdout']=closure[x['stem']]
 assert splits=={'train':973,'val':199} and spatial==16153 and study==13
 report={'state':'PASS_SOURCE_STAGE_ONLY','final_fm384_ready':False,'source':'fresh official reconstruction_rss from user NYU email','expected_volumes':1172,'actual_volumes':len(items),'source_split_volumes':dict(splits),'physical_slices':sum(slices.values()),'source_split_slices':dict(slices),'source_spatial_rows':spatial,'study_level_rows':study,'source_category_counts':dict(category),'lossless_rss_and_native_bytes':total,'source_files_bytes_scope':'files in per-volume source receipts, including rss float32, native uint8, metadata, XML, boxes and transaction binding; excludes receipt files themselves','official_H5_patient_ids':len(groups),'duplicate_exam_groups':sum(len(x)>1 for x in groups.values()),'stem_rule_holdout':sum(x['stem_holdout'] for x in items),'stem_rule_patient_conflict_groups':len(conflicts),'patient_safe_closure_holdout':sum(closure.values()),'patient_identity_scope':'H5 patient_id examination groups; no unprovided stable biological-person mapping across examinations asserted','archive_proofs':{s:{k:v for k,v in p.items() if k!='stems'} for s,p in proofs.items()},'verified_unix':time.time(),'source_code_sha256':code,'all_receiver_SHA_stat_checked':True,'all_rss_native_flip_normalization_checked':True,'private_pixel_parity':False}
 report['reference_reconciliation']={'author_physical_slices':41877,'actual_minus_reference_physical_slices':sum(slices.values())-41877,'author_three_view_rows':125631,'actual_derived_three_view_rows':3*sum(slices.values()),'author_counts_are_reconciliation_reference_only':True}
 manifest={'format':'fastmri_official_rss_source_v4','report':report,'volumes':items,'patient_holdout_conflicts':conflicts}
 for name,v in [('SOURCE_FULLSHA_MANIFEST.json',manifest),('SOURCE_VERIFICATION.json',report)]:
  p=ROOT/'receipts'/name;t=p.with_suffix('.tmp')
  with t.open('w') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(t,p)
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
