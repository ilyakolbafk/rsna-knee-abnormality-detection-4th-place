"""Metadata/proof admission; never repairs or blesses incomplete cache producers."""
from pathlib import Path
import argparse,socket,os,time,json,subprocess,hashlib
import numpy as np
from common import ROOT,TARGET,POLICY,read,write,sha,pin_check
from full_cache_gate import require_bundle
from full_cache_loader import Reader
from main_data import MainDataset
from fm_data import Store


def check_fm_roster(data,manifest,descriptor):
    expected=set(read(ROOT/'assets/FM_EXPECTED_IDS.json')['ids'])
    if len(expected)!=1172 or set(data['expected_ids'])!=expected or set(data['verified_ids'])!=expected:raise ValueError('FROZEN_FM1172_SOURCE_ROSTER')
    if manifest.get('volumes') is None or set(manifest['volumes'])!=expected:raise ValueError('FM_MANIFEST_ROSTER_BINDING')
    index=read(Path(descriptor['root'])/'index.json')
    if index.get('volumes')!=manifest['volumes']:raise ValueError('FM_INDEX_MANIFEST_IDENTITY_BINDING')
    held_patients={row.get('patient_id') for uid,row in index['volumes'].items() if int(hashlib.sha1(uid.encode()).hexdigest(),16)%10==0}
    for uid,row in index['volumes'].items():
        if not row.get('patient_id') or row.get('aux_holdout') is not (row['patient_id'] in held_patients):raise ValueError('FM_PUBLIC_STEM_RULE_PATIENT_CLOSURE:'+uid)
        if row.get('excluded') and not (row.get('exclusion_reason')=='artifact_only' and row.get('exclusion_evidence_sha256') in data.get('reviewed_exclusion_proofs',{})):raise ValueError('FM_UNREVIEWED_EXCLUSION:'+uid)
    return True


def check_targets(arm,path=None):
    path=path or ROOT/f'assets/{arm}_targets.npz'
    with np.load(path,allow_pickle=False) as z:target={k:z[k] for k in z.files}
    ids=list(map(str,target['sample_id']));source=target['source_id'];expected=read(ROOT/'assets/TEST3_EXPECTED.json')
    if len(ids)!=7609 or len(set(ids))!=7609 or int((source==0).sum())!=4355 or int((source==1).sum())!=3254:raise ValueError('FROZEN_FIT_ROSTER')
    for key in ['main','oai_vm']:
        p,w,lr=[target[key+'_'+k] for k in ['probability','weight','lr']]
        if p.shape!=(7609,12) or not np.isfinite(p).all() or not np.isfinite(w).all() or not np.isfinite(lr).all():raise ValueError('TARGET_SHAPE_FINITE')
        if np.any(p<0) or np.any(p>1) or np.any(w<0) or np.any(lr<=0):raise ValueError('TARGET_RANGE')
    if np.any(target['oai_vm_weight'][source==0]):raise ValueError('OAI_LABEL_IN_RSNA_HEAD')
    if arm=='vm' and np.any(target['main_weight'][source==1]):raise ValueError('MOAKS_OR_OAI_IN_VM_RSNA_HEAD')
    if arm=='core6':
        if np.any(target['oai_vm_weight']):raise ValueError('CORE6_EXTRA_VM_HEAD')
        outside=np.ones(12,bool);outside[[5,6,8]]=False
        if np.any(target['main_weight'][source==1][:,outside]):raise ValueError('OAI_MOAKS_ONLY_THREE_CLASSES')
    for uid,p in expected.items():
        j=ids.index(uid)
        if source[j]!=0 or not np.array_equal(target['main_probability'][j],np.asarray(p,np.float32)) or not np.all(target['main_weight'][j]==1) or np.any(target['oai_vm_weight'][j]):raise ValueError('EXACT_TEST3_VM36')
    return {'arm':arm,'rows':7609,'test3_cells':36,'target_sha256':sha(path),'independent_VM_domains':arm=='vm'}


def verify_descriptor_proofs(inputs):
    from actual_acceptance import verify_bundle
    return verify_bundle(inputs)


def verify_actual_current_files(inputs):
    # Reuse accepted destination fullSHA ONLY with unchanged exact stat/inode.
    from oai_collection import collection,validate_case,config_for_case
    from full_cache_gate import require_v4_audit
    from independent_anatomy_rules import validate as anatomy_rules
    rsna=Reader(inputs['corpora']['RSNA'])
    for rel in rsna.files:rsna._path(rel)
    native_count=0
    for key,entry in rsna.registry['studies'].items():
        docs={Path(rel).name:rsna._json(rel) for rel in entry['metadata_files'] if Path(rel).name in ['NATIVE_PROVENANCE.json','ANCHORS.json']}
        anatomy_rules(docs['NATIVE_PROVENANCE.json'],docs['ANCHORS.json']);native_count+=1
    oai=Reader(inputs['corpora']['OAI']);collection(oai);nonces=set()
    for rel,expected in oai.d['files'].items():
        path=oai.path(rel);st=path.stat()
        if path.is_symlink() or st.st_ino!=expected['inode'] or st.st_dev!=expected['dev']:raise ValueError('OAI_CURRENT_INODE:'+rel)
    for key,info in oai.d['studies'].items():
        audit=oai.document(info['audit']);config=validate_case(oai,key,audit);require_v4_audit(audit,recipe_sha256=config['semantic_sha256'])
        if info['request_nonce'] in nonces:raise ValueError('OAI_DUPLICATE_NONCE')
        nonces.add(info['request_nonce'])
    return {'RSNA_all_file_stats':len(rsna.files),'RSNA_public_prior_and_patella_rules':native_count,'OAI_current_files':len(oai.d['files']),'OAI_case_audit_receiver_semantics':len(nonces),'source_image_bytes_rehashed':0}


def inspect_inputs(inputs):
    if inputs.get('schema')!='h6_full_v4_inputs_v1' or inputs.get('target')!=TARGET:raise ValueError('INPUT_TARGET_SCHEMA')
    verify_descriptor_proofs(inputs)
    if set(inputs['corpora'])!={'RSNA','OAI'}:raise ValueError('RSNA_AND_OAI_FULL_CACHES_REQUIRED')
    for name,desc in inputs['corpora'].items():
        if desc['manifest_sha256']!=inputs['acceptance']['datasets'][name]['manifest_sha256']:raise ValueError('MANIFEST_PROOF_BINDING')
        Reader(desc)
    for arm in ['core6','vm']:
        check_targets(arm);MainDataset(inputs,ROOT/f'assets/{arm}_targets.npz',1)
    fm=Store(inputs['fastMRI'],min_volumes=1000)
    if inputs['fastMRI']['manifest_sha256']!=inputs['acceptance']['datasets']['fastMRI']['manifest_sha256']:raise ValueError('FM_MANIFEST_BINDING')
    expected_gold=set(read(ROOT/'assets/GOLD_EXCLUDED_IDS.json'))
    if len(inputs['Gold_excluded_ids'])!=58 or set(inputs['Gold_excluded_ids'])!=expected_gold:raise ValueError('EXACT_GOLD58_EXCLUSION')
    return {'status':'PASS_FULL_CACHE_METADATA_AND_LABEL_ADMISSION','targets':[check_targets(a) for a in ['core6','vm']],
            'source_image_bytes_rehashed':0,'FM_usable_volumes':len(fm.vols),'FM_holdout':len(fm.holdout),'orientation_policy':POLICY}


def host_identity():
    label=Path('/root/.vast_containerlabel')
    if socket.gethostname()!=TARGET['hostname'] or not label.exists() or label.read_text().strip()!='C.'+str(TARGET['instance']):raise ValueError('EXACT_NEW_INSTANCE_REQUIRED')
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    return boot


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',required=True);a=ap.parse_args()
    package=pin_check();boot=host_identity();inputs=read(a.inputs);result=inspect_inputs(inputs);result['actual_all_current_file_and_rule_checks']=verify_actual_current_files(inputs)
    result.update(package_sha256=package,inputs_source_sha256=sha(a.inputs),target=TARGET,boot_id=boot,at_unix=time.time())
    # A new admission is metadata-ready; it does NOT imply either GPU smoke PASS.
    write(ROOT/'runtime/INPUTS.json',inputs);result['inputs_sha256']=sha(ROOT/'runtime/INPUTS.json');write(ROOT/'runtime/ADMISSION.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
