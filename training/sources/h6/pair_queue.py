"""New full-cache pair only. Preparation is inert until a verified release exists.

No old110 controller, predecessor intent, cache or recovery path is referenced.
Unexpected failure is a repair event for the monitor, never an automatic fresh replay.
"""
from pathlib import Path
import argparse,fcntl,json,os,subprocess,sys,time,shutil
import psutil
from common import ROOT,TARGET,GPU_UUIDS,read,write,sha,pin_check
from admission import host_identity

ARMS=['core6','vm']
_complete_cache={}


def complete_arm(root,arm,package):
    out=root/'outputs'/arm;path=out/'COMPLETE.json'
    if not path.exists():return False
    rec=read(path);names={'epoch16_raw.pt','epoch16_ema.pt','swa_ema05_10.pt'}
    if rec.get('status')!='FULLFIT_COMPLETE' or rec.get('arm')!=arm or rec.get('package_sha256')!=package or rec.get('epochs')!=16 or rec.get('global_step')!=60864 or set(rec.get('files',{}))!=names:raise ValueError('INVALID_COMPLETE_MARKER:'+arm)
    stat=lambda p:(p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ctime_ns)
    key=(str(path),package,stat(path),tuple((name,stat(out/name)) for name in sorted(names)))
    if key not in _complete_cache:
        for name,digest in rec['files'].items():
            if sha(out/name)!=digest:raise ValueError('COMPLETE_WEIGHT_FULLSHA:'+arm+':'+name)
        _complete_cache[key]=True
    return True


class Deferred(RuntimeError):pass


def release_gate(root,package):
    path=root/'runtime/RELEASE.json'
    if not path.exists():raise Deferred('WAITING_ACCEPTED_FULL_CACHE_RELEASE')
    r=read(path)
    if r.get('automatic_gpu_dispatch_enabled') is not True:raise Deferred('DISPATCH_DISABLED')
    if r.get('target')!=TARGET or r.get('arms')!=ARMS or r.get('package_sha256')!=package:raise ValueError('RELEASE_TARGET_PACKAGE_PAIR')
    inputs=Path(r['inputs'])
    if not inputs.is_absolute() or sha(inputs)!=r['inputs_sha256']:raise ValueError('RELEASE_INPUT_SHA')
    return r


def write_exclusive(path,value):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def matching_processes(root,arm):
    script=str((root/'train.py').resolve());found=[]
    for p in psutil.process_iter(['pid','create_time','cmdline','status']):
        cmd=p.info['cmdline'] or []
        if script in cmd and '--arm' in cmd and cmd[cmd.index('--arm')+1]==arm and p.info['status']!=psutil.STATUS_ZOMBIE:found.append(p.info)
    if len(found)>1:raise ValueError('DUPLICATE_FIT_PROCESSES:'+arm)
    return found


def gpu_idle(gpu):
    p=subprocess.run(['nvidia-smi','--id='+gpu,'--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    if p.stdout.strip():raise Deferred('GPU_BUSY:'+gpu)
    p=subprocess.run(['nvidia-smi','--id='+gpu,'--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    actual,mem,util=[x.strip() for x in p.stdout.strip().split(',')]
    if actual!=gpu:raise ValueError('GPU_QUERY_IDENTITY')
    if int(mem)>256 or int(util)>5:raise Deferred('GPU_NOT_IDLE:'+gpu)
    folder=Path('/workspace/.knee_gpu_leases');folder.mkdir(exist_ok=True)
    lease=(folder/(gpu+'.lock')).open('a+')
    try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise Deferred('GPU_LEASE_BUSY:'+gpu)
    finally:lease.close()


def run_checked(script,*args,gpu=None):
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OPENCV_FOR_THREADS_NUM='1',NUMPY_MADVISE_HUGEPAGE='0')
    if gpu:env.update(CUDA_VISIBLE_DEVICES=gpu,EXPECTED_GPU_UUID=gpu)
    subprocess.run([sys.executable,str(ROOT/script),*args],env=env,check=True)


def ensure_smoke(arm,gpu,package,boot):
    path=ROOT/'runtime'/f'SMOKE_{arm}.json';sm=read(path) if path.exists() else {}
    if not (sm.get('status')=='PASS' and sm.get('package_sha256')==package and sm.get('GPU_UUID')==gpu and sm.get('boot_id')==boot and sm.get('admission_sha256')==sha(ROOT/'runtime/ADMISSION.json')):
        run_checked('train.py','--arm',arm,'--smoke',gpu=gpu)


def prepare_current_boot(release,package,boot):
    path=ROOT/'runtime/ADMISSION.json'
    current=read(path) if path.exists() else {}
    if current.get('boot_id')!=boot or current.get('package_sha256')!=package:
        run_checked('admission.py','--inputs',release['inputs'])
    elif current.get('inputs_source_sha256')!=release['inputs_sha256']:raise ValueError('INPUT_CHANGE_REQUIRES_NEW_CONTRACT')
    cpu=ROOT/'runtime/CPU_QA.json'
    if not cpu.exists() or read(cpu).get('admission_sha256')!=sha(path):run_checked('cpu_admission.py')
    c=read(cpu)
    if c['status']!='PASS_CURRENT_TARGET_ACTUAL_FULLCACHE_CPU_TEST3' or c['boot_id']!=boot or c['package_sha256']!=package:raise ValueError('CPU_QA_REQUIRED')


def launch(arm,gpu,resume):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,EXPECTED_GPU_UUID=gpu,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OPENCV_FOR_THREADS_NUM='1',NUMPY_MADVISE_HUGEPAGE='0')
    log=(ROOT/'runtime'/f'fit_{arm}.log').open('ab',buffering=0)
    p=subprocess.Popen([sys.executable,str(ROOT/'train.py'),'--arm',arm,'--resume' if resume else '--fresh'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    log.close();proc=psutil.Process(p.pid)
    write(ROOT/'runtime'/f'LAUNCH_{arm}.json',{'pid':p.pid,'process_create_time':proc.create_time(),'argv':proc.cmdline(),'resume':resume,'GPU_UUID':gpu,'at_unix':time.time()})


def tick():
    package=pin_check();boot=host_identity();release=release_gate(ROOT,package)
    live={arm:matching_processes(ROOT,arm) for arm in ARMS}
    complete={arm:complete_arm(ROOT,arm,package) for arm in ARMS}
    if all(complete.values()):return {'state':'BOTH_FULLFITS_COMPLETE'}
    intent=ROOT/'runtime/DISPATCH_INTENT.json'
    if not intent.exists():
        if any(live.values()) or any((ROOT/'outputs'/a).exists() for a in ARMS):raise ValueError('OUTPUT_OR_PROCESS_WITHOUT_INTENT')
        if shutil.disk_usage(ROOT).free<read(ROOT/'CONFIG.json')['pair_dispatch_free_bytes']:raise Deferred('PAIR_DISK_RESERVE40GIB')
        for gpu in GPU_UUIDS:gpu_idle(gpu)
        prepare_current_boot(release,package,boot)
        for arm,gpu in zip(ARMS,GPU_UUIDS):
            gpu_idle(gpu);ensure_smoke(arm,gpu,package,boot)
        write_exclusive(intent,{'target':TARGET,'arms':ARMS,'package_sha256':package,'admission_sha256':sha(ROOT/'runtime/ADMISSION.json'),
            'smokes':{a:sha(ROOT/'runtime'/f'SMOKE_{a}.json') for a in ARMS},'boot_id':boot,'release_sha256':sha(ROOT/'runtime/RELEASE.json'),'at_unix':time.time()})
    else:
        record=read(intent)
        if record.get('package_sha256')!=package or record.get('target')!=TARGET or record.get('release_sha256')!=sha(ROOT/'runtime/RELEASE.json'):raise ValueError('IMMUTABLE_INTENT_BINDING')
    for arm,gpu in zip(ARMS,GPU_UUIDS):
        if complete[arm] or live[arm]:continue
        out=ROOT/'outputs'/arm;gpu_idle(gpu)
        if out.exists():
            # Never retry unknown fit failures. The monitor first diagnoses and
            # repairs, then writes a specific immutable repair authorization.
            repair=ROOT/'runtime'/f'REPAIR_{arm}.json'
            if not repair.exists():raise ValueError('CONFIRMED_MISSING_FIT_REQUIRES_TECHNICAL_DIAGNOSIS:'+arm)
            r=read(repair);identity=read(out/'IDENTITY.json')
            if r.get('failed_pid')!=identity['pid'] or r.get('failed_ctime')!=identity['process_create_time'] or r.get('package_sha256')!=package or not r.get('diagnosis') or r.get('status')!='CAUSE_FIXED_AND_QA_PASS':raise ValueError('REPAIR_IDENTITY_BINDING')
            prepare_current_boot(release,package,boot);run_checked('train.py','--arm',arm,'--smoke',gpu=gpu)
            run_checked('authorize_resume.py','--arm',arm);launch(arm,gpu,True)
        else:
            if (ROOT/'runtime'/f'STARTED_{arm}.json').exists():raise ValueError('LOST_WEIGHTS_NO_SILENT_FRESH:'+arm)
            # Even after a reboot, reconcile only an arm that provably never
            # reached its durable pre-training marker. No trained weights reset.
            prepare_current_boot(release,package,boot);ensure_smoke(arm,gpu,package,boot)
            if read(intent)['admission_sha256']!=sha(ROOT/'runtime/ADMISSION.json'):
                write(ROOT/'runtime'/f'STARTUP_AUTHORIZATION_{arm}.json',{'status':'PASS_UNSTARTED_ARM_CURRENT_BOOT_RECONCILIATION','arm':arm,
                    'package_sha256':package,'original_intent_sha256':sha(intent),'admission_sha256':sha(ROOT/'runtime/ADMISSION.json'),'boot_id':boot,
                    'smoke_sha256':sha(ROOT/'runtime'/f'SMOKE_{arm}.json'),'at_unix':time.time()})
            launch(arm,gpu,False)
    return {'state':'PAIR_DISPATCHED_OR_RUNNING','processes':{a:matching_processes(ROOT,a) for a in ARMS}}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--once',action='store_true');a=ap.parse_args()
    (ROOT/'runtime').mkdir(exist_ok=True);lock=(ROOT/'runtime/queue.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=psutil.Process();write(ROOT/'runtime/CONTROLLER_LAUNCH.json',{'pid':p.pid,'process_create_time':p.create_time(),'argv':p.cmdline()})
    while True:
        try:state=tick()
        except Deferred as exc:state={'state':str(exc),'actionable_failure':False}
        except Exception as exc:
            write(ROOT/'runtime/QUEUE_STATE.json',{'state':'TECHNICAL_REPAIR_REQUIRED','error':type(exc).__name__+':'+str(exc),'at_unix':time.time()})
            raise
        write(ROOT/'runtime/QUEUE_STATE.json',{**state,'at_unix':time.time()})
        if a.once or state['state']=='BOTH_FULLFITS_COMPLETE':return
        time.sleep(45)


if __name__=='__main__':main()
