"""Immutable collection versions map every frozen case to its exact recipe."""
from common import *
import fcntl,copy

BASE='93d747c1a2c00bfefbd484fb7a553dd346e2d4c2eb622ddcd8d17eebaec88465'
_CACHE={}

def known_cases():
    if 'cases' not in _CACHE:_CACHE['cases']={r['case_id'] for r in roster()}
    return _CACHE['cases']

def roster_digest():
    st=ROSTER.stat();identity=(st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    if _CACHE.get('roster_identity')!=identity:
        digest=sha(ROSTER);assert digest=='0862f5a4ecd077422db8f7d4cb6dcf2b8e593c8eee5b1bbf8888cc15e4d86961'
        _CACHE.update(roster_identity=identity,roster_digest=digest)
    return _CACHE['roster_digest']

def canonical_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()

def collection():
    pointer=RUN/'COLLECTION_ACTIVE.json'
    if not pointer.exists():return None
    ref=json.loads(pointer.read_text());path=RUN/ref['path']
    assert path.resolve().is_relative_to((RUN/'collections').resolve())
    stamp=(ref['sha256'],path.stat().st_mtime_ns,path.stat().st_ctime_ns)
    if _CACHE.get('stamp')==stamp:return _CACHE['value']
    assert sha(path)==ref['sha256'],'COLLECTION_FILE_SHA_CONFLICT'
    data=json.loads(path.read_text());assert canonical_hash({k:v for k,v in data.items() if k!='semantic_sha256'})==data['semantic_sha256'],'COLLECTION_SEMANTIC_SHA_CONFLICT'
    assert data['source_roster_sha256']==roster_digest() and set(data['case_recipe'])==known_cases(),'COLLECTION_EXACT_CASE_SET_CONFLICT'
    _CACHE.update(stamp=stamp,value=data);return data

def config_for_case(case):
    assert case in known_cases(),'UNKNOWN_COLLECTION_CASE'
    data=collection();recipe=BASE if data is None else data['case_recipe'][case]
    record={'config':'BUILD_CONFIG.frozen.json'} if data is None else data['recipes'][recipe]
    path=RUN/record['config'];config=json.loads(path.read_text())
    assert config['semantic_sha256']==recipe,'EXPECTED_RECIPE_CONFIG_CONFLICT'
    if data is not None:assert sha(path)==record['config_sha256'],'REGISTERED_CONFIG_SHA_CONFLICT'
    if recipe!=BASE:
        for rel,digest in config['adaptation_component_sha256'].items():
            component=path.parent/'components'/rel;st=component.stat();key=('component',str(component))
            identity=(st.st_size,st.st_mtime_ns,st.st_ctime_ns,digest)
            if _CACHE.get(key)!=identity:assert sha(component)==digest,'FROZEN_ADAPTATION_COMPONENT_CONFLICT:'+rel;_CACHE[key]=identity
    return {**config,'_config_path':record['config']}

def validate_audit(case,audit):
    config=config_for_case(case)
    assert audit['case_id']==case and audit['roster_sha256']==roster_digest(),'AUDIT_CASE_OR_ROSTER_CONFLICT'
    assert audit['recipe_sha256']==config['semantic_sha256'],'CASE_RECIPE_CONFLICT'
    assert audit['config_sha256']==sha(RUN/config['_config_path']),'AUDIT_CONFIG_SHA_CONFLICT'
    code=config.get('adaptation_component_sha256',config.get('component_sha256',{}))
    assert audit['code_sha256']==code['code/geometry_build.py' if config['semantic_sha256']!=BASE else 'code/build_case.py'],'AUDIT_BUILDER_SHA_CONFLICT'
    if config['semantic_sha256']!=BASE:
        data=collection();qref=data['case_qa'][case];qp=RUN/qref['path'];assert sha(qp)==qref['sha256'],'GEOMETRY_QA_FILE_SHA_CONFLICT'
        q=json.loads(qp.read_text())
        assert q['case_id']==case,'GEOMETRY_QA_CASE_CONFLICT'
        assert q['status']=='PASS_NUMERIC_READER_AND_LPS_DIRECTIONS' and q['visual_anatomy_review']=='PASS_ASSISTANT_INSPECTED_ALL_18_VIEWS','GEOMETRY_QA_INCOMPLETE'
        assert q['cache_sha256']==audit['sha256'] and q['recipe_sha256']==audit['recipe_sha256'],'GEOMETRY_QA_CACHE_BINDING_CONFLICT'
        assert audit['geometry_qa']['sha256']==qref['sha256'],'AUDIT_GEOMETRY_QA_BINDING_CONFLICT'
    return config

def activate_geometry_cases(cases):
    with (RUN/'collection.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        data=collection()
        if data is None:
            data={'version':0,'format':'oai-v4-collection-v1','source_roster_sha256':sha(ROSTER),
                'case_recipe':{r['case_id']:BASE for r in roster()},'recipes':{BASE:{'config':'BUILD_CONFIG.frozen.json','config_sha256':sha(RUN/'BUILD_CONFIG.frozen.json')}},'case_qa':{}}
        else:data=copy.deepcopy(data)
        previous=data.pop('semantic_sha256',None)
        for case in cases:
            assert case in known_cases(),'UNKNOWN_COLLECTION_CASE'
            qa=RUN/'geometry_v2/qa'/f'{case}.json';q=json.loads(qa.read_text())
            assert q['case_id']==case,'ACTIVATION_QA_CASE_CONFLICT'
            recipe=q['recipe_sha256'];cp=RUN/'geometry_v2/recipes'/recipe/'CONFIG.frozen.json';config=json.loads(cp.read_text());assert config['semantic_sha256']==recipe
            data['recipes'][recipe]={'config':str(cp.relative_to(RUN)),'config_sha256':sha(cp),'previous_recipe_sha256':config['previous_semantic_sha256']}
            assert q['recipe_sha256']==recipe and q['visual_anatomy_review']=='PASS_ASSISTANT_INSPECTED_ALL_18_VIEWS' and q['status']=='PASS_NUMERIC_READER_AND_LPS_DIRECTIONS'
            prior=RUN/'receiver_receipts'/f'{case}.json'
            assert not prior.exists() or json.loads(prior.read_text())['recipe_sha256']==recipe,'CANNOT_RELABEL_DELIVERED_RECIPE'
            data['case_recipe'][case]=recipe;data['case_qa'][case]={'path':str(qa.relative_to(RUN)),'sha256':sha(qa)}
        data.update(version=data['version']+1,previous_collection_sha256=previous,unix=time.time());data['semantic_sha256']=canonical_hash(data)
        path=RUN/'collections'/f"{data['semantic_sha256']}.json";dump(path,data)
        dump(RUN/'COLLECTION_ACTIVE.json',{'path':str(path.relative_to(RUN)),'sha256':sha(path),'semantic_sha256':data['semantic_sha256']})
        return data

def recipe_counts():
    data=collection()
    if data is None:return {BASE:3254}
    return {r:list(data['case_recipe'].values()).count(r) for r in data['recipes']}
