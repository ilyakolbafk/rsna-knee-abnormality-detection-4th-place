"""Pinned output-based actual-producer acceptance, no nominal cache promotion."""
from pathlib import Path
import hashlib,json,os
from common import ROOT,TARGET,read,sha
STATE='canonical_L_v8_3_with_author_v4_basis_adapter'
_cache=None

def require(v,code):
 if not v:raise ValueError('ACTUAL_ACCEPTANCE:'+code)
def semantic(d):return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()

def dossier():
 global _cache
 p=ROOT/'assets/ACCEPTANCE_DOSSIER.json';policy=read(ROOT/'assets/ACTUAL_RELEASE_POLICY.json');st=p.stat();identity=(st.st_size,st.st_mtime_ns,st.st_ctime_ns,st.st_ino)
 if _cache and _cache[0]==identity:return _cache[1]
 require(not p.is_symlink() and sha(p)==policy['dossier_sha256'],'DOSSIER_SHA');d=read(p)
 require(d['schema']=='h6_actual_full_cache_output_acceptance_v1' and d['target']==TARGET and d['boot_id']=='d7c9a413-650b-4f6e-b872-2fa572010446','TARGET_SCHEMA')
 require(d['legacy110_fallback'] is False and d['Gold_fit'] is False,'SCOPE')
 for name,record in d['evidence'].items():
  q=ROOT/record['path'];require(q.is_file() and not q.is_symlink() and q.resolve().is_relative_to(ROOT.resolve()) and sha(q)==record['sha256'],'EVIDENCE_SHA:'+name)
  document=read(q)
  if 'status' in record:require(document.get('status')==record['status'],'EVIDENCE_STATUS:'+name)
 require(read(ROOT/d['evidence']['RSNA_RAW_REFERENCE_QA']['path'])['all_views_exact'] is True,'RAW_REFERENCE')
 geometry=read(ROOT/d['evidence']['RSNA_GEOMETRY_QA']['path'])
 require(sum(x['samples'] for x in geometry['summaries'])==4413 and sum(x['error_count'] for x in geometry['summaries'])==0 and geometry['full_D4_signed_permutation_and_complete_RAW_corner_coverage'] is True,'GEOMETRY_ALL4413')
 _cache=(identity,d);return d

def require_rsna_release(descriptor):
 d=dossier();r=d['RSNA']
 require(descriptor['backend']=='rsna_public_npz_collection' and descriptor['orientation_state']==STATE,'RSNA_ACTUAL_BASIS')
 require(descriptor['registry_sha256']==r['registry_sha256'] and descriptor['destination_fullsha_proof_sha256']==r['destination_fullsha_proof_sha256'],'RSNA_REGISTRY_PROOF')
 require(str(Path(descriptor['root']))==r['root'],'RSNA_ROOT')
 require(os.uname().nodename==TARGET['hostname'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id'],'CURRENT_TARGET_BOOT')
 return r

def require_unit_release(assessment):
 r=dossier()['RSNA'];require(assessment['orientation_state']==STATE and assessment['native_adapter_sha256']==r['native_adapter_provenance_sha256'] and assessment['source_code_identity_sha256']==r['binding']['code_identity_sha256'],'RSNA_UNIT_SOURCE_BINDING')
 require(assessment['sample_id'] in r['accepted_units'] and [assessment[k] for k in ['parent_id','report_group','rows']]==r['accepted_units'][assessment['sample_id']],'RSNA_UNIT_ROSTER')
 return True

def verify_bundle(inputs):
 d=dossier();require(inputs.get('schema')=='h6_full_v4_inputs_v1' and inputs['target']==TARGET,'INPUT_SCHEMA')
 require(inputs['acceptance']==d['acceptance'],'IMMUTABLE_ACCEPTANCE_CONTENT')
 for name,desc in [('RSNA',inputs['corpora']['RSNA']),('OAI',inputs['corpora']['OAI']),('fastMRI',inputs['fastMRI'])]:require(semantic(desc)==d['descriptor_semantic_sha256'][name],'DESCRIPTOR_SHA:'+name)
 require_rsna_release(inputs['corpora']['RSNA'])
 for name in ['RSNA','OAI','fastMRI']:require(d['acceptance']['datasets'][name]['legacy_110_fallback'] is False,'NO_FALLBACK')
 return True
