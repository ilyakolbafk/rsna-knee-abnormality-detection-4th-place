"""Persistent case-isolated candidate worker; visual QA remains with heartbeat agent."""
from common import *
import fcntl,psutil,signal,shutil
STOP=False

if __name__=='__main__':
    lock=(RUN/'geometry_watch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,lambda *args:globals().__setitem__('STOP',True))
    owner={'PID':os.getpid(),'ctime':psutil.Process().create_time(),'argv':['python3',str(Path(__file__))],'unix':time.time()}
    dump(RUN/'geometry_v2/WATCH_OWNER.json',owner)
    while not STOP:
        config=json.loads((RUN/'GEOMETRY_BUILD_CONFIG.frozen.json').read_text());recipe=config['semantic_sha256'];pending=[]
        for p in (RUN/'rejections').glob('*.json'):
            rejection=json.loads(p.read_text());case=p.stem
            if 'CROSS_SERIES_FRAME_OF_REFERENCE' not in rejection.get('reason',''):continue
            qa=RUN/'geometry_v2/qa'/f'{case}.json'
            if qa.exists() and json.loads(qa.read_text()).get('recipe_sha256')==recipe:continue
            failure=RUN/'geometry_v2/watch_failures'/f'{case}.json'
            if failure.exists() and json.loads(failure.read_text()).get('recipe_sha256')==recipe:continue
            pending.append(case)
        if pending and psutil.virtual_memory().available>int(4.5*2**30) and shutil.disk_usage(RUN).free>35*2**30:
            case=pending[0];logpath=RUN/'geometry_v2'/f'{case}.watch.log'
            code="import os;os.nice(10);from geometry_build import build;from geometry_qa import qa;import sys;build(sys.argv[1]);qa(sys.argv[1])"
            with logpath.open('a') as log:
                child=subprocess.Popen(['python3','-c',code,case],cwd=str(RUN/'code'),stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,
                    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
                dump(RUN/'geometry_v2/WATCH_STATE.json',{'status':'BUILDING_NEW_GEOMETRY_CASE','case_id':case,'child_PID':child.pid,'child_ctime':psutil.Process(child.pid).create_time(),'unix':time.time()})
                try:status=child.wait(timeout=240)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=10);status=-15
            if status:dump(failure,{'case_id':case,'status':'ISOLATED_RECONCILIATION_REQUIRED','recipe_sha256':recipe,'exit_code':status,'log':str(logpath),'unix':time.time()})
        dump(RUN/'geometry_v2/WATCH_STATE.json',{'status':'PERSISTENT_WATCH','pending_candidates':pending,'source_stop_preserved':(RUN/'SOURCE_STOP.json').exists(),'unix':time.time()})
        for _ in range(15):
            if STOP:break
            time.sleep(2)
