from common import *
from source_service import identity,current_limits
from verify_received import verify
from concurrent.futures import ThreadPoolExecutor
import fcntl,shutil,signal
from recipe_registry import config_for_case,recipe_counts,validate_audit,BASE

SHARED=RUN.parent/'shared';STOP=False

def admit():
    limits=current_limits();b=identity()
    with (SHARED/'admission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);p=SHARED/'admission.json';reg=json.loads(p.read_text())
        assert reg['target']['instance_id']==56967269
        own=reg['tasks'].get('oai_v4',{});others=[x for k,x in reg['tasks'].items() if k!='oai_v4']
        lim=reg['limits'];cpu_cap=min(lim.get('max_total_cpu_threads',15),lim.get('cpu_threads_max',15),limits['cpu_threads_max'])
        plan=json.loads((RUN/'RESOURCE_PLAN.json').read_text());cpu_threads=plan.get('remote_cpu_threads',4)
        assert cpu_threads+sum(x.get('cpu_threads',0) for x in others)<=cpu_cap,'SHARED_CPU_ADMISSION'
        reserve=int(json.loads((RUN/'RESOURCE_PLAN.json').read_text())['reserved_disk_bytes'])
        allocated=sum(p.stat().st_size for p in (RUN/'cache').glob('*') if p.is_file())
        floor=max(lim.get('disk_reserve_bytes',0),lim.get('disk_floor_bytes',0),limits['disk_floor_bytes'])
        assert max(0,reserve-allocated)+sum(x.get('reserved_disk_bytes',0) for x in others)<shutil.disk_usage(RUN).free-floor,'SHARED_DISK_ADMISSION'
        mem_floor=max(lim.get('memory_reserve_bytes',0),lim.get('ram_floor_bytes',0),limits['ram_floor_bytes'])
        assert 6*2**30+sum(x.get('reserved_ram_bytes',0) for x in others)<limits['memory_limit_bytes']-mem_floor,'SHARED_RAM_ADMISSION'
        available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
        ms=dict(x.split() for x in Path('/sys/fs/cgroup/memory/memory.stat').read_text().splitlines())
        usage=int(Path('/sys/fs/cgroup/memory/memory.usage_in_bytes').read_text())
        headroom=limits['memory_limit_bytes']-usage+int(ms.get('total_inactive_file',ms.get('inactive_file',0)))
        assert min(available,headroom)>mem_floor+6*2**30,'LIVE_SHARED_RAM_HEADROOM'
        reg['tasks']['oai_v4']={**own,'PID':os.getpid(),'ctime':Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
            'argv':['python3',str(Path(__file__))],'heartbeat_unix':time.time(),'active':True,
            'reserved_disk_bytes':reserve,'reserved_ram_bytes':6*2**30,'cpu_threads':cpu_threads,'download_workers':0,'convert_workers':0,
            'upload_receivers':4,'local_mac_download_workers':plan['local_download_workers'],'local_mac_convert_workers':plan['concurrent_process_cap'],
            'local_mac_upload_workers':plan['upload_workers'],
            'phase':'LOCAL_BUILD_PARALLEL_CACHE_RECEIVE','scratch_paths':[str(RUN/'incoming')],'allocated_cache_bytes':allocated}
        reg['tasks']['oai_v4'].update(final_cache_reservation='MEASURED_CANARY_WITH_MARGIN',observed_disk_free_bytes=shutil.disk_usage(RUN).free)
        dump(p,reg)

def snapshot(publish=True):
    receipts={};stale=[]
    recipe=json.loads((RUN/'BUILD_CONFIG.frozen.json').read_text())['semantic_sha256'];expected={r['case_id'] for r in roster()}
    for p in (RUN/'cache_receipts').glob('*.json'):
        r=json.loads(p.read_text());q=RUN/'cache'/f"{r['case_id']}.npz";ap=RUN/'cache'/f"{r['case_id']}.audit.json"
        if r['case_id'] not in expected:stale.append(r['case_id']);continue
        expected_recipe=config_for_case(r['case_id'])['semantic_sha256']
        if not q.exists() or not ap.exists():stale.append(r['case_id']);continue
        st=q.stat();ast=ap.stat()
        if (st.st_size,st.st_ino,st.st_mtime_ns,st.st_ctime_ns,r['recipe_sha256'],ast.st_ino,ast.st_mtime_ns)!=(r['bytes'],r['inode'],r['mtime_ns'],r['ctime_ns'],expected_recipe,r['audit_inode'],r['audit_mtime_ns']):stale.append(r['case_id']);continue
        if expected_recipe!=BASE:
            try:validate_audit(r['case_id'],json.loads(ap.read_text()))
            except Exception:stale.append(r['case_id']);continue
        receipts[r['case_id']]=r
    qa=json.loads((RUN/'CANARY_QA.frozen.json').read_text()) if (RUN/'CANARY_QA.frozen.json').exists() else {}
    ready=qa.get('ready_for_adapted_public_v4_build') is True and qa.get('recipe_sha256')==recipe
    rows=[{'case_id':r['case_id'],'expected':True,'received':r['case_id'] in receipts,
           'built':r['case_id'] in receipts,'verified':r['case_id'] in receipts and ready,
           'rejected':(RUN/'rejections'/f"{r['case_id']}.json").exists()} for r in roster()]
    reader_ready=True
    if len(recipe_counts())>1:
        qr=RUN/'COLLECTION_READER_QA.json'
        reader_ready=qr.exists()
        if reader_ready:
            reader=json.loads(qr.read_text());reader_ready=reader.get('status')=='PASS_MIXED_RECIPES_TWO_WORKER_READER_AND_IDENTITY_GUARDS' and set(recipe_counts()).issubset(reader.get('covered_recipes',[])) and reader.get('collection_reader_sha256')==sha(RUN/'code/collection_reader.py') and reader.get('recipe_registry_sha256')==sha(RUN/'code/recipe_registry.py')
    complete=set(receipts)==expected and not stale and ready and reader_ready and not any(x['rejected'] for x in rows)
    n=len(receipts);state={'status':'COMPLETE_ADAPTED_PUBLIC_V4' if complete else 'INCOMPLETE',
          'phase':'LOCAL_BUILD_PARALLEL_CACHE_RECEIVE','unix':time.time(),'PID':os.getpid(),'expected':3254,
          'received':n,'built':n,'verified':n if ready else 0,'rejected':sum(x['rejected'] for x in rows),
          'bytes':sum(x['bytes'] for x in receipts.values()),'stale_receipts':stale,
          'blocker':None if ready and reader_ready else 'CANARY_OR_COLLECTION_READER_QA_PENDING','private_hrepro_parity':False,
          'format':'oai-v4-v83-lps-adapt-v1','orientation_state':'canonical_L_v8_3','training_launched':False}
    state['expected_recipe_counts']=recipe_counts();state['received_recipe_counts']={r:sum(v['recipe_sha256']==r for v in receipts.values()) for r in state['expected_recipe_counts']}
    if len(state['expected_recipe_counts'])>1:state['format']='oai-v4-collection-v1'
    if publish:
        dump(RUN/'STATE.json',state);dump(RUN/'EXPECTED_RECEIVED_BUILT_VERIFIED.json',{'summary':state,'rows':rows})
        dump(RUN/'MISSING_REJECTED.json',{'missing':[x['case_id'] for x in rows if not x['received']],
                                       'rejected':[x['case_id'] for x in rows if x['rejected']]})
    return state

def main():
    global STOP
    identity();lock=(RUN/'singleton.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for d in ['cache','cache_receipts','incoming','rejections']:(RUN/d).mkdir(exist_ok=True)
    for claimed in (RUN/'incoming').glob('*.processing.json'):
        req=json.loads(claimed.read_text());case=req['case_id'];nonce=req['request_nonce']
        assert claimed.name==f'{case}.{nonce}.processing.json'
        rp=RUN/'cache_receipts'/f'{case}.json';cached=RUN/'cache'/f'{case}.npz';audit=RUN/'cache'/f'{case}.audit.json'
        receipt=json.loads(rp.read_text()) if rp.exists() else {}
        if receipt.get('request_nonce')==nonce and cached.exists() and audit.exists() and sha(cached)==req['sha256'] and sha(audit)==req['audit_sha256']:
            claimed.unlink();continue
        with (RUN/'incoming'/f'{case}.request.lock').open('a') as request_lock:
            fcntl.flock(request_lock,fcntl.LOCK_EX);ready=RUN/'incoming'/f'{case}.ready.json'
            claimed.replace(ready if not ready.exists() else claimed.with_suffix('.interrupted.json'))
    signal.signal(signal.SIGTERM,lambda *x:globals().__setitem__('STOP',True))
    jobs={};last=0
    with ThreadPoolExecutor(max_workers=4) as pool:
        while not STOP:
            if time.monotonic()-last>40:admit();print(json.dumps(snapshot()),flush=True);last=time.monotonic()
            for case,job in list(jobs.items()):
                f,req,claimed=job
                if not f.done():continue
                try:
                    f.result();claimed.unlink(missing_ok=True)
                    (RUN/'rejections'/f'{case}.json').unlink(missing_ok=True)
                except Exception as e:
                    dump(RUN/'rejections'/f'{case}.json',{'case_id':case,'reason':str(e),'request_nonce':req.get('request_nonce'),'unix':time.time()})
                    claimed.replace(RUN/'incoming'/f"{case}.{req['request_nonce']}.failed.json")
                del jobs[case]
            for p in (RUN/'incoming').glob('*.ready.json'):
                case=p.name.removesuffix('.ready.json')
                if case not in jobs and len(jobs)<4:
                    with (RUN/'incoming'/f'{case}.request.lock').open('a') as request_lock:
                        fcntl.flock(request_lock,fcntl.LOCK_EX)
                        if not p.exists():continue
                        req=json.loads(p.read_text());nonce=req['request_nonce']
                        assert len(nonce)==32 and all(c in '0123456789abcdef' for c in nonce)
                        claimed=p.with_name(f'{case}.{nonce}.processing.json');os.replace(p,claimed)
                    jobs[case]=(pool.submit(verify,case,req),req,claimed)
            time.sleep(1)
    snapshot()

if __name__=='__main__':main()
