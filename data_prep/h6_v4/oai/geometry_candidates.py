"""Single admitted repair worker alongside bulk; failures remain per-case evidence."""
from common import *
import fcntl,signal,psutil,gc,traceback

if __name__=='__main__':
    lock=(RUN/'geometry_worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.nice(10)
    if psutil.virtual_memory().available<4*2**30:raise RuntimeError('GEOMETRY_WORKER_RAM_ADMISSION')
    import geometry_build as builder
    import geometry_qa
    requested=json.loads((RUN/'geometry_v2/REQUESTED_CASES.json').read_text())['cases']
    results=[]
    for case in requested:
        marker=RUN/'geometry_v2/WORKER_STATE.json'
        dump(marker,{'status':'BUILDING_GEOMETRY_CANDIDATE','case_id':case,'PID':os.getpid(),'ctime':psutil.Process().create_time(),'unix':time.time(),'results':results})
        try:
            signal.signal(signal.SIGALRM,lambda *args:(_ for _ in ()).throw(TimeoutError('GEOMETRY_CASE_DEADLINE_180S')));signal.alarm(180)
            builder.build(case);geometry_qa.qa(case)
            results.append({'case_id':case,'status':'NUMERIC_QA_PASS_VISUAL_PENDING'})
        except Exception as error:
            result={'case_id':case,'status':'GEOMETRY_REPAIR_REQUIRES_RECONCILIATION','reason':type(error).__name__+':'+str(error),'traceback':traceback.format_exc(),'unix':time.time()}
            dump(RUN/'geometry_v2/errors'/f'{case}.json',result);results.append({k:v for k,v in result.items() if k!='traceback'});print(json.dumps(result),flush=True)
        finally:signal.alarm(0);gc.collect()
    dump(RUN/'geometry_v2/WORKER_STATE.json',{'status':'CANDIDATE_BATCH_FINISHED','PID':os.getpid(),'results':results,'unix':time.time()})
