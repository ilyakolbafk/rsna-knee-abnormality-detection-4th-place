"""Restore the exact registered source tree; never rebuild v2 with the latest alias."""
from common import *
import shutil,fcntl,signal
from recipe_registry import config_for_case,validate_audit,BASE

def linked(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():return
    try:os.link(source,target)
    except OSError:shutil.copy2(source,target)

def build(case):
    config=config_for_case(case);assert config['semantic_sha256']!=BASE
    cp=RUN/config['_config_path'];root=RUN/'geometry_v2/restores'/case
    root.mkdir(parents=True,exist_ok=True)
    with (root/'dispatch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        base=json.loads((RUN/'BUILD_CONFIG.frozen.json').read_text())
        for rel,digest in base['component_sha256'].items():
            assert sha(RUN/rel)==digest;linked(RUN/rel,root/rel)
        for source in (RUN/'handoff').rglob('*'):
            if source.is_file():linked(source,root/source.relative_to(RUN))
        for rel in ['code/common.py','BUILD_CONFIG.frozen.json','LOCAL_RAW_LOOKUP.json','TARGET_VERIFIED.json','registration_recovery/RECOVERY_STATE.json']:
            linked(RUN/rel,root/rel)
        for rel,digest in config['adaptation_component_sha256'].items():
            source=cp.parent/'components'/rel;assert sha(source)==digest
            target=root/rel;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():assert sha(target)==digest
            else:linked(source,target)
        shutil.copy2(cp,root/'GEOMETRY_BUILD_CONFIG.frozen.json')
        if not (root/'scratch').exists():(root/'scratch').symlink_to(RUN/'scratch',target_is_directory=True)
        log=(root/'rebuild.log').open('a')
        proc=subprocess.Popen(['python3',str(root/'code/geometry_build.py'),case],stdin=subprocess.DEVNULL,stdout=log,stderr=log,
            start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        try:code=proc.wait(timeout=240)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=10);raise
        finally:log.close()
        if code:raise RuntimeError('FROZEN_GEOMETRY_REBUILD_EXIT:'+str(code))
        qa_path=RUN/'geometry_v2/qa'/f'{case}.json';qa=json.loads(qa_path.read_text())
        cache=root/'cache'/f'{case}.npz';ap=root/'cache'/f'{case}.audit.json';audit=json.loads(ap.read_text())
        assert sha(cache)==qa['cache_sha256']==audit['sha256'],'REGISTERED_FROZEN_REBUILD_NOT_BYTE_EXACT'
        audit['geometry_qa']={'path':str(qa_path.relative_to(RUN)),'sha256':sha(qa_path),'status':'PASS_NUMERIC_AND_VISUAL','visual_review':qa['visual_anatomy_review']}
        validate_audit(case,audit)
        os.replace(cache,RUN/'cache'/f'{case}.npz');dump(RUN/'cache'/f'{case}.audit.json',audit)
        dump(root/'DISPATCH_QA.json',{'status':'PASS_REGISTERED_SOURCE_SNAPSHOT_AND_ORIGINAL_NPZ_SHA','case_id':case,'recipe_sha256':config['semantic_sha256'],'cache_sha256':audit['sha256'],'source_components_unchanged':True,'unix':time.time()})
        return audit

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('case');args=p.parse_args();print(json.dumps({'case_id':args.case,'sha256':build(args.case)['sha256']}))
