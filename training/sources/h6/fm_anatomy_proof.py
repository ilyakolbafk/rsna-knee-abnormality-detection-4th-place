"""Exact promoted FM semantics + explicit sameboot historical-SHA continuity.

This is new admission evidence, not a claim that the producer saved per-file
stats. Fail closed on every chronology, identity or pinned-metadata mismatch.
"""
from pathlib import Path
import hashlib,json,os,sys,importlib.util
from common import ROOT


def require(ok,reason):
    if not ok:raise ValueError('FM_ANATOMY:'+reason)


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def descriptor_guard(descriptor):
    policy=json.loads((ROOT/'FM_ANATOMY_POLICY.frozen.json').read_text())
    path=ROOT/'assets/FM_DESTINATION_STAT_ADMISSION.json'
    require(digest(path)==policy['destination_stat_admission_sha256']==descriptor.get('stat_admission_sha256'),'STAT_ADMISSION_SHA')
    proof=json.loads(path.read_text());root=Path(descriptor['root']).resolve()
    require(str(root)==proof['root'] and not Path(descriptor['root']).is_symlink(),'EXACT_ACCEPTED_ROOT')
    require(proof['target']==policy['target'] and os.uname().nodename==proof['target']['hostname'],'TARGET')
    require(Path('/root/.vast_containerlabel').read_text().strip()=='C.'+str(proof['target']['instance']),'INSTANCE')
    require(Path('/proc/sys/kernel/random/boot_id').read_text().strip()==proof['boot_id'],'CURRENTBOOT')
    prebuild=json.loads((root.parent.parent/'TARGET_VERIFIED.json').read_text())
    require(hashlib.sha256(json.dumps(prebuild,sort_keys=True,separators=(',',':')).encode()).hexdigest()==proof['prebuild_target_semantic_sha256'],'PREBUILD_TARGET_PROOF')
    require(prebuild['boot_id']==proof['boot_id'] and prebuild['instance_id']==proof['target']['instance'] and prebuild['machine_id']==proof['target']['machine'],'PREBUILD_IDENTITY')
    require(proof['admission_mode']=='SAME_BOOT_PRE_FULLSHA_CTIME_CONTINUITY','EXPLICIT_CONTINUITY_MODE')
    require(digest(root/'CORE_BINDING.json')==proof['core_sha256']==policy['core_sha256'],'CORE_SHA')
    require(digest(root/'PROMOTION_RECEIPT.json')==proof['promotion_sha256']==policy['promotion_sha256']==descriptor['destination_fullsha_proof_sha256'],'PROMOTION_SHA')
    core=json.loads((root/'CORE_BINDING.json').read_text());promotion=json.loads((root/'PROMOTION_RECEIPT.json').read_text())
    require(prebuild['verified_unix']<core['created_unix'],'PREBUILD_BEFORE_FROZEN_CORE')
    require(promotion.get('state')=='PASS' and promotion.get('training_ready') is True and promotion['core_binding_sha256']==proof['core_sha256'],'PROMOTED')
    expected=set(promotion['files_sha256'])|{'PROMOTION_RECEIPT.json'}
    require(set(descriptor['files'])==expected,'EXACT_PROMOTED_FILESET')
    for rel,d in promotion['files_sha256'].items():require(descriptor['files'][rel]['sha256']==d,'PROMOTED_MEMBER_SHA')
    chunks={r for r in expected if r.startswith('chunks/')}
    require(chunks==set(proof['files']) and proof['cutoff_ns']==int(core['created_unix'])*10**9,'EXACT_CONTINUITY_MEMBER_SET_OR_CUTOFF')
    for rel,z in proof['files'].items():
        require(descriptor['files'][rel]==z and z['sha256']==core['files_sha256'][rel] and z['stat'][2]<=proof['cutoff_ns'],'CHUNK_CONTINUITY')
    for name in ['FULLSHA_QA.json','LOADER_TARGET_QA.json']:
        qpath=root/'provenance'/name;q=json.loads(qpath.read_text())
        require(q.get('state')=='PASS' and q['core_binding_sha256']==proof['core_sha256'] and q['verified_unix']>core['created_unix'],'BOUND_RECEIVER_QA')
        require(digest(qpath)==promotion['files_sha256']['provenance/'+name],'QA_SHA')
    q=json.loads((root/'provenance/FULLSHA_QA.json').read_text())
    require(digest(root/'provenance/FULLSHA_QA.json')==policy['fullsha_qa_sha256']==proof['fullsha_qa_sha256'],'FULLSHA_QA_SHA')
    require(q['every_pixel_chunk_compressed_and_decoded_sha_checked'] is True and q['checked_pixel_chunks']==len(chunks) and q['compressed_pixel_bytes']==sum(z['stat'][0] for z in proof['files'].values()),'FULLSHA_COVERAGE')
    loader=json.loads((root/'provenance/LOADER_TARGET_QA.json').read_text())
    require(loader['code_sha256']==policy['producer_code_sha256']['qa_fm_consumer.py']==core['files_sha256']['code/qa_fm_consumer.py'],'EXECUTED_QA_CODE')
    # Run the byte-identical producer semantic verifier without its pixelhash pass.
    code=ROOT/'fm_producer_code'
    for name,d in policy['producer_code_sha256'].items():require(digest(code/name)==d==core['files_sha256']['code/'+name],'EXECUTING_PRODUCER_CODE')
    name='_h6_fm_producer_guard'
    if name not in sys.modules:
        before=list(sys.path);sys.path.insert(0,str(code))
        try:
            spec=importlib.util.spec_from_file_location(name,code/'fm_training_reader.py');module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        finally:sys.path[:]=before
    sys.modules[name].verify_core(root,verify_pixels=False)
    require(digest(ROOT/'fm_delta_codec.py')==policy['producer_code_sha256']['fm_delta_codec.py'],'EXECUTING_CODEC')
    return proof
