"""Restart only this local pipeline after bounded technical controller exits."""
from common import *
import fcntl,signal

STOP=False
if __name__=='__main__':
    lock=(RUN/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,lambda *x:globals().__setitem__('STOP',True))
    signal.signal(signal.SIGINT,lambda *x:globals().__setitem__('STOP',True))
    restarts=0;consecutive=0
    while not STOP:
        started=time.time()
        log=(RUN/'pipeline.log').open('a')
        p=subprocess.Popen(['python3',str(RUN/'code/pipeline.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                           start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
        dump(RUN/'SUPERVISOR_STATE.json',{'PID':os.getpid(),'child_PID':p.pid,'restarts':restarts,'unix':time.time(),'status':'RUNNING'})
        while p.poll() is None:
            if STOP:
                p.terminate();p.wait(timeout=1000);break
            state_path=RUN/'STATE.local.json'
            if state_path.exists():
                state=json.loads(state_path.read_text())
                if state.get('PID')==p.pid and time.time()-state.get('unix',time.time())>300:
                    dump(RUN/'supervisor_events'/f'{int(time.time())}.json',{'action':'OWN_CONTROLLER_STALE_HEARTBEAT_GRACEFUL_RESTART','PID':p.pid,'unix':time.time()})
                    p.terminate()
            time.sleep(2)
        log.close()
        if STOP:break
        terminal=json.loads((RUN/'TERMINAL.json').read_text()) if (RUN/'TERMINAL.json').exists() else {}
        state=json.loads((RUN/'STATE.local.json').read_text()) if (RUN/'STATE.local.json').exists() else {}
        if state.get('verified_on_target')==3254:
            try:
                result=subprocess.run(['python3',str(RUN/'code/finalize.py')],capture_output=True,text=True,timeout=180)
            except (subprocess.TimeoutExpired,OSError) as error:
                dump(RUN/'SUPERVISOR_STATE.json',{'PID':os.getpid(),'status':'FINALIZATION_TECHNICAL_RETRY','unix':time.time(),'reason':type(error).__name__})
                for _ in range(6):
                    if STOP:break
                    time.sleep(10)
                continue
            dump(RUN/'SUPERVISOR_STATE.json',{'PID':os.getpid(),'status':'COMPLETE_ADAPTED_PUBLIC_V4' if result.returncode==0 else 'FINALIZATION_REQUIRES_RECONCILIATION','unix':time.time(),'finalizer_exit_code':result.returncode})
            if result.returncode==0:
                state['status']='COMPLETE_ADAPTED_PUBLIC_V4';dump(RUN/'STATE.local.json',state);break
            time.sleep(60);continue
        if p.returncode==0 or (RUN/'SOURCE_STOP.json').exists():
            dump(RUN/'SUPERVISOR_STATE.json',{'PID':os.getpid(),'status':'INCOMPLETE_REQUIRES_RECONCILIATION','unix':time.time(),'terminal':terminal,'source_stop':(RUN/'SOURCE_STOP.json').exists()});time.sleep(30);continue
        restarts+=1;consecutive=0 if time.time()-started>=300 else consecutive+1
        dump(RUN/'SUPERVISOR_STATE.json',{'PID':os.getpid(),'status':'BOUNDED_TECHNICAL_RESTART','exit_code':p.returncode,'restarts':restarts,'unix':time.time()})
        if consecutive>=3:
            dump(RUN/'SUPERVISOR_STATE.json',{'PID':os.getpid(),'status':'PERSISTENT_RECOVERY_COOLDOWN_REQUIRES_HEARTBEAT_DIAGNOSIS','unix':time.time(),'restarts':restarts,'consecutive':consecutive})
            for _ in range(30):
                if STOP:break
                time.sleep(10)
            consecutive=0
        else:time.sleep([5,15,30][min(consecutive,2)])
