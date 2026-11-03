"""Verify frozen clean-source review documents before any numerical gate."""
from pathlib import Path
import json,hashlib,datetime
T=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify_document(name):
 p=T/name;d=json.loads(p.read_text());group='A' if name.endswith('_A.json') else 'B';initial=name.startswith('BLIND_VALIDATION_');is_c011='_C011_' in name
 if group=='A':
  f=json.loads(p.with_suffix('.FREEZE.json').read_text());assert Path(f['artifact']).name==name and f['sha256']==sha(p) and f['blinding_preserved_at_freeze'] is True
  assert d['individual_model_hypotheses_or_segmentations_or_other_reviewers_read'] is False
  records=[(v['observation'],v['provenance']) for v in d['cases']]
 else:
  parts=p.with_suffix('.sha256').read_text().split();assert parts==[sha(p),name],'FROZEN_B_REVIEW_SHA_MISMATCH'
  assert (all(v is False for v in d['blinding'].values()) if initial else d['individual_model_or_other_reviewer_results_read'] is False)
  assert d['PNG_SHA_verified_before_transient_input'] is True;records=[(v,v) for v in d['observations']]
 protocol=json.loads((T/'ANATOMICAL_QA_PROTOCOL.v1.json').read_text());selection=json.loads((T/'VALIDATION_SELECTION.json').read_text());assert selection['protocol_sha256']==protocol['semantic_sha256'];cases={c['alias']:c for c in selection['cases']}
 packet_name=('PACKET_'+group+'.json') if initial else ('SUPPLEMENT_C011_'+group+'.json' if is_c011 else 'SUPPLEMENT_'+group+'.json');packet=json.loads((T/packet_name).read_text());assert d['packet_sha256' if initial else 'supplement_packet_sha256']==sha(T/packet_name),'REVIEW_PACKET_SHA_MISMATCH'
 if initial:
  assert d['protocol_semantic_sha256']==protocol['semantic_sha256'];assert d['protocol_sha256' if group=='A' else 'protocol_file_sha256']==sha(T/'ANATOMICAL_QA_PROTOCOL.v1.json')
 else:
  assert d['initial_blind_artifact_sha256' if group=='A' else 'original_frozen_observation_sha256']==sha(T/('BLIND_VALIDATION_'+group+'.json')),'SUPPLEMENT_INITIAL_SHA_MISMATCH'
 panelmap={};expected={}
 for panel in packet['panels']:
  panelmap[(panel['relative_path'],panel['PNG_sha256'])]=panel
  for c in panel['cases']:
   assert c['alias'] in cases and c['source_receipt_sha256']==cases[c['alias']]['source_receipt_sha256'];assert all(isinstance(z,int) and 0<=z<cases[c['alias']]['shape'][0] for z in c['source_slice_indices'])
   expected.setdefault(c['alias'],set()).update(c['source_slice_indices'])
 seen=set()
 for c,prov in records:
  a=c['alias'];assert a in expected and a not in seen;seen.add(a);case=cases[a]
  if group=='A' and not initial:
   panels=prov['panels'];assert panels and all((v['relative_path'],v['PNG_sha256']) in panelmap for v in panels)
   assert set(prov['all_presented_source_slice_indices'])==expected[a]
   assert all(any(cc['alias']==a and cc['source_receipt_sha256']==case['source_receipt_sha256'] for cc in v['cases']) for v in panels)
  elif group=='B' and not initial:
   assert prov['source_receipt_sha256']==case['source_receipt_sha256'];assert all((v['relative_path'],v['PNG_sha256']) in panelmap for v in prov['PNG_bindings']);assert set(prov['source_frames_reviewed'])==expected[a]
  else:
   assert prov['source_receipt_sha256']==case['source_receipt_sha256'];assert (prov['panel_relative_path'],prov['PNG_sha256']) in panelmap
   presented=prov['all_presented_source_slice_indices'] if group=='A' else prov['source_slice_indices'];assert set(presented)==expected[a]
  witnesses=c.get('source_slice_witnesses',[v['source_slice'] for v in c.get('witnesses',[])]);assert set(witnesses)<=expected[a],'SIDE_WITNESS_NOT_IN_SOURCE_PACKET'
  side=c.get('side',c.get('fibula_image_side'))
  if side in ['left','right'] and not c.get('needs_more_frames',False):assert len(set(witnesses))>=2,'RESOLVED_SIDE_LACKS_TWO_WITNESSES'
  j=c.get('approx_joint_line',c.get('joint_center'))
  if initial:assert j is not None,'INITIAL_REVIEW_MISSING_JOINT_OBSERVATION'
  if j is not None:
   w=j.get('source_indices',j.get('source_slice_witnesses'));assert len(set(w))>=2 and set(w)<=expected[a],'JOINT_WITNESS_NOT_IN_SOURCE_PACKET'
 assert seen==set(expected),'REVIEW_COMPLETENESS_MISMATCH'
 return {'sha256':sha(p),'packet_sha256':sha(T/packet_name),'cases':len(seen),'frozen_SHA_source_receipt_packet_and_witness_membership':'PASS'}
def main():
 names=['BLIND_VALIDATION_A.json','BLIND_VALIDATION_B.json','BLIND_SUPPLEMENT_A.json','BLIND_SUPPLEMENT_B.json','BLIND_SUPPLEMENT_C011_A.json','BLIND_SUPPLEMENT_C011_B.json'];files={n:verify_document(n) for n in names};report={'state':'PASS','files':files,'selection_sha256':sha(T/'VALIDATION_SELECTION.json'),'protocol_sha256':sha(T/'ANATOMICAL_QA_PROTOCOL.v1.json'),'code_sha256':sha(__file__),'verified_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()};(T/'SOURCE_REVIEW_BINDING_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n');return report
if __name__=='__main__':print(json.dumps(main(),indent=2))
