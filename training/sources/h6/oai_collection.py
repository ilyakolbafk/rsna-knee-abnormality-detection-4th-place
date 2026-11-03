"""Exact accepted OAI per-case recipe registry. Never follows a mutable pointer."""
from pathlib import Path
import json,hashlib

POLICY=json.loads(Path(__file__).with_name('OAI_COLLECTION_POLICY.frozen.json').read_text())


def require(value,code):
    if not value:raise ValueError('OAI_COLLECTION:'+code)


def collection(reader):
    require(reader.kind=='case_npz','CASE_BACKEND_REQUIRED')
    ref=reader.d['collection'];path=ref['path']
    require(reader.d['files'][path]['sha256']==POLICY['collection_file_sha256'],'FILE_SHA')
    data=reader.document(path)
    require(data['format']=='oai-v4-collection-v1','FORMAT')
    require(data['semantic_sha256']==POLICY['collection_semantic_sha256'],'SEMANTIC_SHA')
    semantic=hashlib.sha256(json.dumps({k:v for k,v in data.items() if k!='semantic_sha256'},sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    require(semantic==data['semantic_sha256'],'SEMANTIC_CONTENT')
    require(data['source_roster_sha256']==POLICY['source_roster_sha256'],'SOURCE_ROSTER')
    wanted=set(POLICY['expected_case_ids'])
    require(len(wanted)==3254 and set(data['case_recipe'])==wanted and set(reader.d['studies'])==wanted,'EXACT_CASE_SET')
    require(all(v.get('source_id')==1 and v.get('sample_id')==k for k,v in reader.d['studies'].items()),'OAI_SOURCE_IDENTITY')
    require({r:v['config_sha256'] for r,v in data['recipes'].items()}==POLICY['expected_recipe_config_sha256'],'EXACT_CONFIG_SET')
    return data


def require_manifest_binding(kind,descriptor,manifest):
    require(descriptor.get('collection')==manifest.get('collection'),'MANIFEST_BINDING')
    if kind=='OAI':require(bool(descriptor.get('collection')) and descriptor.get('backend')=='case_npz','MANIFEST_COLLECTION_REQUIRED')
    else:require(not descriptor.get('collection'),'NON_OAI_COLLECTION')


def config_for_case(reader,key):
    data=collection(reader);recipe=data['case_recipe'][key];rec=data['recipes'][recipe]
    rel=str(Path(reader.d['collection']['provenance_root'])/rec['config'])
    require(reader.d['files'][rel]['sha256']==rec['config_sha256'],'CONFIG_SHA')
    config=reader.document(rel)
    require(config['semantic_sha256']==recipe and config['roster_sha256']==POLICY['source_roster_sha256'],'CONFIG_RECIPE_OR_ROSTER')
    return config


def validate_case(reader,key,audit):
    config=config_for_case(reader,key);data=collection(reader);info=reader.d['studies'][key]
    require(audit['case_id']==key and audit['roster_sha256']==POLICY['source_roster_sha256'],'AUDIT_CASE_OR_ROSTER')
    require(audit['recipe_sha256']==config['semantic_sha256'],'CASE_RECIPE')
    rec=data['recipes'][config['semantic_sha256']]
    require(audit['config_sha256']==rec['config_sha256'],'AUDIT_CONFIG')
    code=config.get('adaptation_component_sha256',config.get('component_sha256',{}))
    builder='code/build_case.py' if config['semantic_sha256']==POLICY['base_recipe_sha256'] else 'code/geometry_build.py'
    require(audit['code_sha256']==code[builder],'AUDIT_BUILDER')
    px=reader.d['files'][info['pixels']];ad=reader.d['files'][info['audit']]
    receipt=reader.document(info['receiver_receipt'])
    require(receipt['status']=='RECEIVER_FULLSHA_AND_ARRAYS_VERIFIED' and receipt['case_id']==key,'RECEIVER_STATUS_CASE')
    require(receipt['request_nonce']==info['request_nonce'] and len(info['request_nonce'])==32,'RECEIVER_NONCE')
    require(receipt['recipe_sha256']==config['semantic_sha256'] and receipt['sha256']==px['sha256']==audit['sha256'] and receipt['audit_sha256']==ad['sha256'],'RECEIVER_FULLSHA_BINDING')
    require([receipt['bytes'],receipt['mtime_ns'],receipt['ctime_ns']]==px['stat'],'RECEIVER_STAT')
    require(reader.path(info['pixels']).stat().st_ino==receipt['inode'],'RECEIVER_INODE')
    if config['semantic_sha256']!=POLICY['base_recipe_sha256']:
        ref=data['case_qa'][key];rel=str(Path(reader.d['collection']['provenance_root'])/ref['path'])
        require(reader.d['files'][rel]['sha256']==ref['sha256'],'GEOMETRY_QA_SHA')
        qa=reader.document(rel)
        require(qa['case_id']==key and qa['status']=='PASS_NUMERIC_READER_AND_LPS_DIRECTIONS' and qa['visual_anatomy_review']=='PASS_ASSISTANT_INSPECTED_ALL_18_VIEWS','GEOMETRY_QA_STATUS_CASE')
        require(qa['cache_sha256']==px['sha256'] and qa['recipe_sha256']==config['semantic_sha256'] and audit['geometry_qa']['sha256']==ref['sha256'],'GEOMETRY_QA_PAYLOAD')
    return config
