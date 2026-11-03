"""Target-only singleton: shared admission, receiver SHA, explicit incomplete manifest."""
from common import *
import fcntl
import shutil
import signal
import sys

BASE=RUN.parent
SHARED=BASE/'shared'
ROWS=roster()
BYID={r['case_id']:r for r in ROWS}
STOP=False

def identity():
    b=json.loads((RUN/'TARGET_VERIFIED.json').read_text())
    assert Path('/root/.vast_containerlabel').read_text().strip()=='C.56967269'
    assert os.uname().nodename==b['hostname']
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==b['boot_id']
    return b

def current_limits():
    quota=int(Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read_text())
    period=int(Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read_text())
    cpu=min(len(os.sched_getaffinity(0)),quota/period if quota>0 else len(os.sched_getaffinity(0)))
    mem=int(Path('/sys/fs/cgroup/memory/memory.limit_in_bytes').read_text())
    disk=shutil.disk_usage(RUN)
    return {'cpu_quota_threads':cpu,'cpu_threads_max':cpu/2,
            'memory_limit_bytes':mem,'ram_floor_bytes':max(8*2**30,int(mem*.15)),
            'disk_total_bytes':disk.total,'disk_floor_bytes':max(30*2**30,int(disk.total*.1))}

def admit(first=False):
    b=identity();SHARED.mkdir(exist_ok=True)
    with (SHARED/'admission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        p=SHARED/'admission.json'
        if p.exists(): reg=json.loads(p.read_text())
        else: reg={'version':1,'target':{'instance_id':56967269,'machine_id':1569,'host_id':b['provider']['host_id']},'limits':current_limits(),'tasks':{}}
        assert reg['version']==1 and reg['target']['instance_id']==56967269
        measured=current_limits()
        # Preserve registry fields; stricter current limits govern admission.
        lim=reg['limits']
        for k,v in measured.items(): lim.setdefault(k,v)
        cpu_cap=min(lim['cpu_threads_max'],measured['cpu_threads_max'])
        mem_cap=min(lim['memory_limit_bytes'],measured['memory_limit_bytes'])
        ram_floor=max(lim['ram_floor_bytes'],measured['ram_floor_bytes'])
        disk_floor=max(lim['disk_floor_bytes'],measured['disk_floor_bytes'])
        tasks=reg['tasks'];other=[x for k,x in tasks.items() if k!='oai_v4']
        own=tasks.get('oai_v4',{})
        if first and own and own.get('PID')!=os.getpid():
            pid=own.get('PID',own.get('pid'))
            if pid and Path(f'/proc/{pid}').exists(): raise RuntimeError('EXISTING_OAI_OWNER_REQUIRES_RECONCILIATION')
            # No automatic reservation release; caller must explicitly reconcile old owner.
            raise RuntimeError('STALE_OAI_REGISTRY_REQUIRES_PID_CTIME_LOCK_RECONCILIATION')
        reservation=sum(r['source_archive_bytes'] for r in ROWS)+4*2**30
        if 2+sum(x.get('cpu_threads',0) for x in other)>cpu_cap:raise RuntimeError('SHARED_CPU_ADMISSION')
        if 8*2**30+sum(x.get('reserved_ram_bytes',0) for x in other)>mem_cap-ram_floor:raise RuntimeError('SHARED_RAM_RESERVATION')
        disk=shutil.disk_usage(RUN)
        allocated=sum(p.stat().st_size for p in (RUN/'raw').rglob('*.tar.gz')) if (RUN/'raw').exists() else 0
        remaining=max(0,reservation-allocated)
        # Neighbor reservations are conservatively subtracted in full.
        if remaining+sum(x.get('reserved_disk_bytes',0) for x in other)>disk.free-disk_floor:raise RuntimeError('SHARED_DISK_RESERVATION')
        mem_available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
        stat=dict(x.split() for x in Path('/sys/fs/cgroup/memory/memory.stat').read_text().splitlines())
        used=int(Path('/sys/fs/cgroup/memory/memory.usage_in_bytes').read_text())
        cg_headroom=mem_cap-used+int(stat.get('total_inactive_file',stat.get('inactive_file',0)))
        if min(mem_available,cg_headroom)<ram_floor+8*2**30:raise RuntimeError('LIVE_RAM_HEADROOM')
        reg['tasks']['oai_v4']={**own,'run_id':'oai-v4-56967269-20261021','PID':os.getpid(),
            'ctime':Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
            'argv':['python3',str(Path(__file__))],'heartbeat_unix':time.time(),
            'reserved_disk_bytes':reservation,'reserved_ram_bytes':8*2**30,'cpu_threads':2,
            'download_workers':1,'convert_workers':1,'scratch_paths':[str(RUN/'incoming')],
            'phase':'raw_acquisition','final_cache_reservation':'PENDING_MEASURED_V4_CANARY',
            'raw_allocated_bytes':allocated,'observed_disk_free_bytes':disk.free}
        dump(p,reg)

def receive(case):
    row=BYID[case];receipt_path=RUN/'source_receipts'/f'{case}.json'
    objects=[]
    for item in row['objects']:
        expected=item['archive_sha256'];n=int(item['object']['archive_bytes'])
        target=RUN/'raw'/case/f"{item['object']['family']}.tar.gz"
        incoming=RUN/'incoming'/case/f"{item['object']['family']}.tar.gz"
        path=target if target.exists() else incoming
        if not path.is_file() or path.stat().st_size!=n or sha(path)!=expected:
            raise RuntimeError('RECEIVER_SOURCE_SHA_OR_SIZE_MISMATCH')
        target.parent.mkdir(parents=True,exist_ok=True)
        if path!=target:
            with path.open('rb') as f:os.fsync(f.fileno())
            os.replace(path,target)
            fd=os.open(target.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
        st=target.stat()
        objects.append({'family':item['object']['family'],'path':str(target),
                        'bytes':st.st_size,'sha256':expected,'inode':st.st_ino,
                        'mtime_ns':st.st_mtime_ns,'ctime_ns':st.st_ctime_ns,
                        'source_study_uid':item['source_study_uid'],
                        'source_series_uid':item['source_series_uid']})
    receipt={'schema':'oai-v4-raw-receiver-v1','case_id':case,'status':'RECEIVED_FULLSHA_VERIFIED',
             'roster_sha256':sha(ROSTER),'objects':objects,'unix':time.time(),
             'code_sha256':sha(__file__),'built':False,'cache_verified':False}
    dump(receipt_path,receipt)
    ready=RUN/'incoming'/f'{case}.ready.json'
    if ready.exists(): ready.unlink()

def snapshot():
    receipts={}
    for p in (RUN/'source_receipts').glob('*.json'):
        r=json.loads(p.read_text()); valid=r['roster_sha256']==sha(ROSTER)
        for o in r['objects']:
            q=Path(o['path'])
            if not q.is_file():valid=False;continue
            st=q.stat()
            if (st.st_size,st.st_ino,st.st_mtime_ns,st.st_ctime_ns)!=(o['bytes'],o['inode'],o['mtime_ns'],o['ctime_ns']): valid=False
        if valid:receipts[r['case_id']]=r
    states=[{'case_id':r['case_id'],'expected':True,'received':r['case_id'] in receipts,
             'built':False,'verified':False,'rejected':(RUN/'rejections'/f"{r['case_id']}.json").exists(),
             'cache_status':'WAITING_V4_COMPONENT_AND_CANARY_QA'} for r in ROWS]
    status={'schema':'oai-v4-build-state-v1','phase':'SOURCE_ACQUISITION','status':'INCOMPLETE',
            'unix':time.time(),'PID':os.getpid(),'expected':3254,'received':len(receipts),
            'built':0,'verified':0,'rejected':sum(x['rejected'] for x in states),
            'received_bytes':sum(o['bytes'] for r in receipts.values() for o in r['objects']),
            'blocker':'V4_RENDERER_CANARY_AND_READER_QA_PENDING',
            'orientation_state':'RAW_NOT_YET_CANONICALIZED','training_launched':False}
    dump(RUN/'STATE.json',status)
    dump(RUN/'EXPECTED_RECEIVED_BUILT_VERIFIED.json',{'summary':status,'rows':states})
    dump(RUN/'MISSING_REJECTED.json',{'missing':[x['case_id'] for x in states if not x['received']],
                                    'rejected':[x['case_id'] for x in states if x['rejected']]})
    return status

def main():
    global STOP
    identity()
    lock=(RUN/'singleton.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name in ['incoming','raw','source_receipts','rejections']: (RUN/name).mkdir(exist_ok=True)
    admit(first=True)
    signal.signal(signal.SIGTERM,lambda *x:globals().__setitem__('STOP',True))
    signal.signal(signal.SIGINT,lambda *x:globals().__setitem__('STOP',True))
    last=0
    while not STOP:
        if time.monotonic()-last>40:
            admit();s=snapshot();last=time.monotonic();print(json.dumps(s),flush=True)
        for p in sorted((RUN/'incoming').glob('*.ready.json')):
            case=p.name.removesuffix('.ready.json')
            if case not in BYID:raise RuntimeError('UNKNOWN_READY_CASE')
            try:receive(case)
            except Exception as e:
                dump(RUN/'rejections'/f'{case}.json',{'case_id':case,'reason':str(e),'unix':time.time()})
                raise
        time.sleep(2)
    snapshot()

if __name__=='__main__':main()
