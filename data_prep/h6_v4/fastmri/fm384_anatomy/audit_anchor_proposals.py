"""Fail-closed post-run binding/source/segmentation audit; no anatomical accuracy claim."""
import json,hashlib,time,sys
from pathlib import Path
from anatomy_job import ROOT,BASE,write

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def audit(phase):
 import numpy as np
 full=BASE/'receipts/SOURCE_FULLSHA_MANIFEST.json';src=json.loads(full.read_text());bind=json.loads((ROOT/'ANCHOR_BUILD_BINDING.json').read_text());aux=json.loads((ROOT/'ANCHOR_AUXILIARY_BINDING.json').read_text());policy=json.loads((ROOT/'JOINT_ANCHOR_POLICY.v1.json').read_text());sel=ROOT/'VALIDATION_SELECTION.json';cases=json.loads(sel.read_text())['cases'];selected={c['stem'] for c in cases};rows=[r for r in src['volumes'] if phase=='corpus' or r['stem'] in selected]
 assert sha(full)==bind['source_manifest_sha256']==policy['source_manifest_sha256'];assert sha(ROOT/'JOINT_ANCHOR_POLICY.v1.json')==bind['policy_sha256'];assert sha(ROOT/'SIDE_HYPOTHESES.json')==bind['side_hypotheses_sha256'];assert sha(sel)==aux['selection_sha256']
 for bucket in ['model_sha256','code_sha256']:
  for n,d in bind[bucket].items():assert sha(ROOT/n)==d,'ANCHOR_BOUND_FILE_CHANGED:'+n
 for n,v in aux['config_files'].items():
  p=ROOT/n;s=p.stat();assert sha(p)==v['sha256'];assert s.st_size==v['bytes'] and s.st_mtime_ns==v['mtime_ns'] and s.st_ctime_ns==v['ctime_ns'],'CONFIG_STAT_CHANGED:'+n
 model=next((ROOT/'weights/mrannotator').rglob('plans.json')).parent
 for name in ['plans.json','dataset.json']:assert (model/name).read_bytes()==(ROOT/'safe_task002'/name).read_bytes(),'CURATED_CONFIG_DIFFERENT_FROM_ORIGINAL'
 parity=json.loads((ROOT/'qa/CORONAL_GRAPH_PARITY.json').read_text());assert parity['state']=='PASS_NUMERIC_GRAPH_PARITY_ONLY';assert parity['binding']['graph_code_sha256']==sha(ROOT/'coronal_graph.py')
 assert len(cases)==112 and len(selected)==112;primary=[c for c in cases if c['alias'].startswith('V')];assert len(primary)==96 and len({c['patient_id'] for c in primary})==96
 hyp=json.loads((ROOT/'SIDE_HYPOTHESES.json').read_text())['volumes'];issues={};files={};state_disagreements=[]
 for r in rows:
  stem=r['stem'];p=BASE/'source_rss'/stem;f=ROOT/'anchor_proposals'/(stem+'.json');d=json.loads(f.read_text());assert d['row_binding']=={'binding_sha256':sha(ROOT/'ANCHOR_BUILD_BINDING.json'),'source_receipt_sha256':r['receipt_sha256']}
  assert sha(p/'receipt.json')==r['receipt_sha256']
  for name,v in r['files'].items():
   s=(p/name).stat();assert all(getattr(s,k)==v['stat'][n] for k,n in [('st_ino','inode'),('st_size','bytes'),('st_mtime_ns','mtime_ns'),('st_ctime_ns','ctime_ns')]),'SOURCE_STAT_CHANGED'
  flip=hyp[stem]['source_mean_logit']<0;assert bool(d['hflip_to_lateral_pluscol'])==flip
  prop=d['proposal'];a=prop['joint_anchor_canonical_display_row_col'];sa=d['source_anchor_display_row_col']
  if a is not None:assert np.isfinite(a).all() and np.allclose(sa,[a[0],319-a[1] if flip else a[1]])
  issues[stem]=prop['geometry_issues']
  if prop['geometry_issues'] and prop['state'].startswith('PROPOSAL_GEOMETRY_PASS'):state_disagreements.append(stem)
  if stem in selected:
   seg=ROOT/'anchor_validation_segmentation'/(stem+'.seg.npy');assert sha(seg)==d['validation_segmentation_sha256'];v=np.load(seg,mmap_mode='r');assert v.shape==(r['n_slices'],320,320) and v.dtype==np.uint8
  files[stem]=sha(f)
 report={'state':'PASS_TECHNICAL_BINDING_ONLY','phase':phase,'checked_volumes':len(rows),'binding_sha256':sha(ROOT/'ANCHOR_BUILD_BINDING.json'),'auxiliary_binding_sha256':sha(ROOT/'ANCHOR_AUXILIARY_BINDING.json'),'selection_sha256':sha(sel),'policy_sha256':sha(ROOT/'JOINT_ANCHOR_POLICY.v1.json'),'source_manifest_sha256':sha(full),'proposal_sha256':files,'geometry_uncertain_volumes':sum(bool(x) for x in issues.values()),'nonempty_issues_despite_legacy_positive_state':state_disagreements,'acceptance_must_use_geometry_issues_not_state':True,'source_receipt_stat_and_saved_segmentation_fullsha_checked':True,'anatomical_accuracy_claimed':False,'training_ready':False,'verified_unix':time.time()};write(ROOT/('ANCHOR_'+phase.upper()+'_BINDING_AUDIT.json'),report);return report
if __name__=='__main__':
 r=audit(sys.argv[1] if len(sys.argv)>1 else 'validation');print(json.dumps({k:v for k,v in r.items() if k!='proposal_sha256'},indent=2))
