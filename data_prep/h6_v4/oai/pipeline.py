"""Bounded local download -> parallel render -> parallel receiver SHA -> own scratch cleanup."""
from common import *
from fetch_case import fetch
from upload_case import upload
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
from build_case import build
import multiprocessing,fcntl,shutil,psutil,signal,traceback,sys
from recipe_registry import config_for_case,validate_audit,recipe_counts,collection,BASE

PLAN=json.loads((RUN/'RESOURCE_PLAN.json').read_text())
DOWNLOAD_WORKERS=PLAN['local_download_workers']
PROCESS_WORKERS=PLAN['local_process_workers']
UPLOAD_WORKERS=PLAN.get('upload_pool_max_workers',PLAN['upload_workers'])
CONFIG=json.loads((RUN/'BUILD_CONFIG.frozen.json').read_text())
ROWS=roster();BYID={r['case_id']:r for r in ROWS}
STOP=False

def size_tree(p):
    total=0
    if p.exists():
        for x in p.rglob('*'):
            try:
                st=x.stat()
                if x.is_file():total+=st.st_size
            except FileNotFoundError:pass
    return total
def buffer_bytes():return sum(size_tree(RUN/x) for x in ['scratch','build_scratch','cache','transport_cache','geometry_v2/work','geometry_v2/restores'])

def check_cache(case):
    p=RUN/'cache'/f'{case}.npz';ap=RUN/'cache'/f'{case}.audit.json'
    if not p.exists() or not ap.exists():return False
    a=json.loads(ap.read_text())
    return a.get('recipe_sha256')==config_for_case(case)['semantic_sha256'] and p.stat().st_size==a['bytes'] and sha(p)==a['sha256']

def available_raw(row,lookup):
    return all(((RUN/'scratch'/row['case_id']/'downloads'/o['object']['download_alias']).is_file()
                and (RUN/'scratch'/row['case_id']/'downloads'/o['object']['download_alias']).stat().st_size==int(o['object']['archive_bytes']))
               or o['archive_sha256'] in lookup for o in row['objects'])

def process_unit(case):
    marker=RUN/'active_builds'/f'{case}.json';dump(marker,{'case_id':case,'PID':os.getpid(),'unix':time.time()})
    try:
        if config_for_case(case)['semantic_sha256']!=BASE:
            deadline=time.monotonic()+300
            while psutil.virtual_memory().available<4*2**30:
                if time.monotonic()>deadline:raise TimeoutError('FROZEN_REBUILD_RAM_ADMISSION')
                time.sleep(2)
            from frozen_geometry_dispatch import build as rebuild
            return rebuild(case)
        return build(case)
    finally:marker.unlink(missing_ok=True)

def process_admission_headroom(cases,pool,add_jobs=1):
    # Reconcile with the measured bulk peak; do not apply the initial canary
    # extrapolation margin again after replacing its peak with a bulk maximum.
    admission=json.loads((RUN/'RESOURCE_PLAN.json').read_text())
    if admission.get('process_admission_mode')=='MEASURED_CONCURRENCY_LIVE_FLOOR':
        # A fixed concurrency was tested on the same complete worst-case cases.
        # Live available RAM and critical-pressure backpressure gate each staggered
        # start; do not reserve another hypothetical peak for every active worker.
        return 0
    total=0;peak=int(admission['peak_rss_bytes']*admission.get('process_peak_admission_margin',1.2));active_pids=set()
    for case in cases:
        marker=RUN/'active_builds'/f'{case}.json';rss=0
        if marker.exists():
            try:
                pid=json.loads(marker.read_text())['PID'];rss=psutil.Process(pid).memory_info().rss;active_pids.add(pid)
            except (psutil.NoSuchProcess,FileNotFoundError):pass
        if rss and admission.get('process_admission_mode')!='LIVE_RSS_STAGGERED':
            total+=max(0,peak-rss)
    idle=[]
    for pid in pool._processes:
        if pid not in active_pids:
            try:idle.append(psutil.Process(pid).memory_info().rss)
            except psutil.NoSuchProcess:pass
    needed=len(cases)-len(active_pids)+add_jobs
    # Existing immutable models/allocator pages are already included in available RAM.
    # Credit the smallest idle RSS values, independent of which worker takes each task.
    for i in range(needed):total+=max(0,peak-(sorted(idle)[i] if i<len(idle) else 0))
    if not pool._processes:total+=max(0,PROCESS_WORKERS-needed)*384*2**20
    return total

def cleanup(case,receipt):
    assert receipt['recipe_sha256']==config_for_case(case)['semantic_sha256']
    rejection=RUN/'rejections'/f'{case}.json'
    if rejection.exists():
        dump(RUN/'repairs/reconciled'/f"{case}.{receipt['request_nonce']}.json",{'prior_rejection':json.loads(rejection.read_text()),'confirmed_receipt':receipt,'unix':time.time()})
        rejection.unlink()
    ap=RUN/'cache'/f'{case}.audit.json'
    if ap.exists():
        a=json.loads(ap.read_text());assert a['sha256']==receipt['sha256'] and a['recipe_sha256']==receipt['recipe_sha256']
        assert sha(ap)==receipt['audit_sha256'],'CURRENT_AUDIT_DIFFERS_FROM_RECEIVER_VERIFIED_ARTIFACT'
        dump(RUN/'build_receipts'/f'{case}.json',a)
    removed=0
    for q in [RUN/'scratch'/case,RUN/'build_scratch'/case,RUN/'geometry_v2/work'/case,RUN/'geometry_v2/restores'/case]:
        assert q.resolve().is_relative_to(RUN.resolve()) and not q.is_symlink()
        if q.exists():removed+=size_tree(q);shutil.rmtree(q)
    p=RUN/'cache'/f'{case}.npz'
    if p.exists():
        assert sha(p)==receipt['sha256'];removed+=p.stat().st_size;p.unlink()
    if ap.exists():ap.unlink()
    packet=RUN/'transport_cache'/f'{case}.transport.zst'
    if packet.exists():removed+=packet.stat().st_size;packet.unlink()
    cleanup_path=RUN/'cleanup_receipts'/f'{case}.json'
    prior=json.loads(cleanup_path.read_text()) if cleanup_path.exists() else {}
    if removed==0 and prior:return
    dump(cleanup_path,{'case_id':case,'removed_own_scratch_bytes':removed+prior.get('removed_own_scratch_bytes',0),
          'receiver_sha256':receipt['sha256'],'receiver_verified_unix':receipt['unix'],'unix':time.time(),
          'existing_sources_deleted':False})

def remote_snapshot(cases=None):
    b=binding()
    return remote_python(f'''
from pathlib import Path
import json,os,sys
p=Path({REMOTE!r})
sys.path.insert(0,str(p/'code'));from recipe_registry import config_for_case,validate_audit,BASE
assert Path('/root/.vast_containerlabel').read_text().strip()=='C.56967269'
assert os.uname().nodename=={b['hostname']!r}
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()=={b['boot_id']!r}
receipts={{}}
for q in (p/'cache_receipts').glob('*.json'):
 r=json.loads(q.read_text());f=p/'cache'/f"{{r['case_id']}}.npz";a=p/'cache'/f"{{r['case_id']}}.audit.json"
 if {cases!r} is not None and r['case_id'] not in {cases!r}:continue
 if not f.exists() or not a.exists():continue
 st=f.stat();ast=a.stat()
 if (st.st_size,st.st_ino,st.st_mtime_ns,st.st_ctime_ns,ast.st_ino,ast.st_mtime_ns)!=(r['bytes'],r['inode'],r['mtime_ns'],r['ctime_ns'],r['audit_inode'],r['audit_mtime_ns']):continue
 if r['recipe_sha256']!=config_for_case(r['case_id'])['semantic_sha256']:continue
 if r['recipe_sha256']!=BASE:
  try:validate_audit(r['case_id'],json.loads(a.read_text()))
  except Exception:continue
 receipts[r['case_id']]=r
print(json.dumps({{'receipts':receipts,'state':json.loads((p/'STATE.json').read_text())}}))
''',timeout=60)

def main():
    global STOP
    lock=(RUN/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    qa=json.loads((RUN/'CANARY_QA.frozen.json').read_text());assert qa['ready_for_adapted_public_v4_build'] and qa['recipe_sha256']==CONFIG['semantic_sha256']
    for d in ['receiver_receipts','build_receipts','cleanup_receipts','rejections','active_builds']:(RUN/d).mkdir(exist_ok=True)
    signal.signal(signal.SIGTERM,lambda *x:globals().__setitem__('STOP',True))
    signal.signal(signal.SIGINT,lambda *x:globals().__setitem__('STOP',True))
    lookup=json.loads((RUN/'LOCAL_RAW_LOOKUP.json').read_text())
    prior=remote_snapshot();done=prior['receipts'];received_bytes=sum(x['bytes'] for x in done.values());received_wire_bytes=sum(x.get('wire_bytes',x['bytes']) for x in done.values())
    for case,r in done.items():dump(RUN/'receiver_receipts'/f'{case}.json',r);cleanup(case,r)
    fetched={r['case_id'] for r in ROWS if available_raw(r,lookup)}|set(done)
    built={r['case_id'] for r in ROWS if check_cache(r['case_id'])}|set(done)
    rejected={p.stem:json.loads(p.read_text()) for p in (RUN/'rejections').glob('*.json')}
    downloading={};processing={};uploading={};attempts={};last=0;last_process_start=0;start=time.time();start_n=len(done);start_bytes=received_bytes
    owner={'PID':os.getpid(),'ctime':subprocess.check_output(['ps','-o','lstart=','-p',str(os.getpid())],text=True).strip(),
           'argv':['python3',str(Path(__file__))],'run_id':'oai-v4-56967269-20261021','recipe_sha256':CONFIG['semantic_sha256'],'unix':start}
    dump(RUN/'PIPELINE_OWNER.json',owner)
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as downloads,ThreadPoolExecutor(max_workers=UPLOAD_WORKERS) as uploads,ProcessPoolExecutor(max_workers=PROCESS_WORKERS,mp_context=multiprocessing.get_context('spawn')) as processes:
        while not STOP:
            live_plan=json.loads((RUN/'RESOURCE_PLAN.json').read_text())
            upload_cap=min(UPLOAD_WORKERS,live_plan['upload_workers'])
            PLAN['local_buffer_max_bytes']=live_plan['local_buffer_max_bytes']
            for futures,kind in [(downloading,'download'),(processing,'process'),(uploading,'upload')]:
                for case,f in list(futures.items()):
                    if not f.done():continue
                    del futures[case]
                    try:
                        result=f.result()
                        if kind=='download':fetched.add(case)
                        elif kind=='process':
                            built.add(case)
                            admission=json.loads((RUN/'RESOURCE_PLAN.json').read_text())
                            if result['peak_rss_bytes']>admission['peak_rss_bytes']:
                                admission['peak_rss_bytes']=result['peak_rss_bytes']
                                admission['peak_measured_case_id']=case
                                admission['peak_updated_unix']=time.time()
                                dump(RUN/'RESOURCE_PLAN.json',admission)
                        else:
                            done[case]=result;received_bytes+=result['bytes'];received_wire_bytes+=result.get('wire_bytes',result['bytes']);cleanup(case,result)
                        print(json.dumps({'case_id':case,'stage':kind,'status':'OK','unix':time.time()}),flush=True)
                    except Exception as e:
                        key=(case,kind);attempts[key]=attempts.get(key,0)+1
                        message=str(e)
                        retryable=(kind=='upload' and (isinstance(e,subprocess.TimeoutExpired) or any(x in message for x in ['RSYNC_EXIT_','SSH_PYTHON_EXIT_255','RECEIVER_READBACK_TIMEOUT']))) or (kind=='download' and not (RUN/'SOURCE_STOP.json').exists() and any(x in message for x in ['SOURCE_TRANSPORT_DEADLINE','SOURCE_TRANSPORT_INCOMPLETE','timeout','timed out','ConnectionError']))
                        if retryable and attempts[key]<3:
                            print(json.dumps({'case_id':case,'stage':kind,'status':'RETRY_UNCONFIRMED_UNIT','attempt':attempts[key],'unix':time.time()}),flush=True)
                            continue
                        record={'case_id':case,'stage':kind,'reason':message,'unix':time.time(),'attempts':attempts[key]}
                        rejected[case]=record;dump(RUN/'rejections'/f'{case}.json',record)
                        print(json.dumps({'case_id':case,'stage':kind,'status':'REJECTED','reason':message[:300]}),flush=True)
            # Drain outputs first. No local pixel deletion until a fresh recipient receipt returns.
            for row in ROWS:
                case=row['case_id']
                if len(uploading)>=upload_cap:break
                if case in built and case not in done and case not in uploading and case not in rejected:
                    uploading[case]=uploads.submit(upload,case)
            current=buffer_bytes();free=shutil.disk_usage(RUN).free;available=psutil.virtual_memory().available
            admission=json.loads((RUN/'RESOURCE_PLAN.json').read_text())
            process_cap=min(PROCESS_WORKERS,admission.get('concurrent_process_cap',PROCESS_WORKERS))
            pressure_level=0
            if sys.platform=='darwin':
                pressure_level=int(subprocess.check_output(['sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True))
            busy=set(downloading)|set(processing)|set(uploading)|set(done)|set(rejected)
            for row in ROWS:
                case=row['case_id']
                if len(processing)>=process_cap or pressure_level>=4:break
                if time.monotonic()-last_process_start<admission.get('process_start_stagger_seconds',0):break
                if case in fetched and case not in built and case not in busy:
                    anticipated=current+sum(BYID[c]['source_archive_bytes']*3 for c in processing)+row['source_archive_bytes']*3
                    if anticipated>PLAN['local_buffer_max_bytes'] or free<30*2**30 or available-process_admission_headroom(processing,processes)<admission.get('local_ram_floor_bytes',2*2**30):break
                    processing[case]=processes.submit(process_unit,case);busy.add(case);last_process_start=time.monotonic()
            if not (RUN/'SOURCE_STOP.json').exists():
                raw_buffer=size_tree(RUN/'scratch')
                for row in ROWS:
                    case=row['case_id']
                    if len(downloading)>=DOWNLOAD_WORKERS:break
                    if case in fetched or case in busy:continue
                    anticipated=current+sum(BYID[c]['source_archive_bytes']*1.2 for c in downloading)+sum(BYID[c]['source_archive_bytes']*3 for c in processing)+row['source_archive_bytes']*1.2
                    anticipated_raw=raw_buffer+sum(BYID[c]['source_archive_bytes']*1.2 for c in downloading)+row['source_archive_bytes']*1.2
                    if anticipated>PLAN['local_buffer_max_bytes'] or anticipated_raw>live_plan.get('raw_prefetch_max_bytes',1_200_000_000) or free<33*2**30:break
                    downloading[case]=downloads.submit(fetch,case);busy.add(case)
            if time.monotonic()-last>40 or len(done)==3254:
                for case in list(rejected):
                    if not (RUN/'rejections'/f'{case}.json').exists():rejected.pop(case)
                registry=collection();sync_path=RUN/'geometry_v2/LAST_COLLECTION_SYNC.json'
                if registry is not None and sync_path.exists() and json.loads(sync_path.read_text()).get('collection_sha256')==registry['semantic_sha256']:
                    for case,recipe in registry['case_recipe'].items():
                        if recipe==BASE or case in done or case in uploading or case in processing or case in downloading:continue
                        if check_cache(case):
                            validate_audit(case,json.loads((RUN/'cache'/f'{case}.audit.json').read_text()));fetched.add(case);built.add(case);rejected.pop(case,None)
                external=[p.stem for p in (RUN/'receiver_receipts').glob('*.json') if p.stem not in done]
                if external:
                    fresh=remote_snapshot(external)['receipts']
                    for case,r in fresh.items():
                        done[case]=r;received_bytes+=r['bytes'];received_wire_bytes+=r.get('wire_bytes',r['bytes']);fetched.add(case);built.add(case)
                        rejected.pop(case,None);cleanup(case,r)
                elapsed=max(1,time.time()-start);speed=(received_bytes-start_bytes)/elapsed
                no_active=not (downloading or processing or uploading)
                state={'status':('BACKPRESSURE_WAIT' if no_active else 'RUNNING') if len(done)<3254 else 'READY_FOR_FINAL_RECONCILIATION','unix':time.time(),
                    'PID':os.getpid(),'ctime':owner['ctime'],'expected':3254,'source_received':len(fetched),'built':len(built),'verified_on_target':len(done),
                    'rejected':len(rejected),'active':{'download':list(downloading),'process':list(processing),'upload':list(uploading)},
                    'workers':{'download':DOWNLOAD_WORKERS,'nda_threads_each':PLAN['NDA_threads_per_worker'],'process':PROCESS_WORKERS,'upload':upload_cap},'upload_pool_max_workers':UPLOAD_WORKERS,'buffer_bytes':buffer_bytes(),'buffer_limit_bytes':PLAN['local_buffer_max_bytes'],
                    'process_cap':process_cap,'memory_pressure_level':pressure_level,'memory_admission_mode':admission.get('process_admission_mode','PEAK_RESERVATION'),
                    'received_bytes':received_bytes,'received_wire_bytes':received_wire_bytes,'mean_verified_bytes_per_second':speed,'estimated_seconds_remaining':None if len(done)==start_n else (3254-len(done))*elapsed/(len(done)-start_n),
                    'local_disk_free_bytes':free,'local_memory_available_bytes':available,'recipe_sha256':CONFIG['semantic_sha256'],
                    'expected_recipe_counts':recipe_counts(),
                    'buffer_by_stage':{d:size_tree(RUN/d) for d in ['scratch','build_scratch','cache','transport_cache']},
                    'backpressure_reason':'BUFFER_OR_RAM_OR_DISK_ADMISSION' if no_active else None,
                    'source_stop':json.loads((RUN/'SOURCE_STOP.json').read_text()) if (RUN/'SOURCE_STOP.json').exists() else None,'private_hrepro_parity':False}
                dump(RUN/'STATE.local.json',state)
                manifest=[{'case_id':r['case_id'],'expected':True,'source_received':r['case_id'] in fetched,'built':r['case_id'] in built,'received':r['case_id'] in done,'verified':r['case_id'] in done,'rejected':r['case_id'] in rejected} for r in ROWS]
                dump(RUN/'EXPECTED_RECEIVED_BUILT_VERIFIED.local.json',{'summary':state,'rows':manifest})
                (RUN/'ledger.md').write_text('# OAI v4 ledger\n\n- Target: 56967269 / machine1569 / host18, identity and hostkey pinned.\n- Processing: Mac; download'+str(DOWNLOAD_WORKERS)+'x2 NDA, process'+str(PROCESS_WORKERS)+'x2 CPU, upload'+str(UPLOAD_WORKERS)+'; own scratch deletion follows receiver fullSHA.\n- Recipe: '+CONFIG['semantic_sha256']+'; public v4 with explicit v8.3/LPS/NPZ and MONAI runtime adaptations; private hrepro parity not claimed.\n- Latest state: '+json.dumps(state,ensure_ascii=False)+'\n- Original sources, labels, training queues and oai15 unchanged.\n- Resume: python3 '+str(Path(__file__))+'\n')
                print(json.dumps({'progress':{k:state[k] for k in ['source_received','built','verified_on_target','rejected','buffer_bytes','received_bytes','mean_verified_bytes_per_second','estimated_seconds_remaining']}}),flush=True)
                last=time.monotonic()
            if len(done)==3254:break
            active=downloading or processing or uploading
            if not active and ((RUN/'SOURCE_STOP.json').exists() or len(done)+len(rejected)==3254):
                dump(RUN/'TERMINAL.json',{'status':'INCOMPLETE_REQUIRES_RECONCILIATION','done':len(done),'rejected':len(rejected),'source_stop':(RUN/'SOURCE_STOP.json').exists(),'unix':time.time()});break
            if not active and current>PLAN['local_buffer_max_bytes']:
                dump(RUN/'TERMINAL.json',{'status':'BUFFER_BACKPRESSURE_REQUIRES_RECONCILIATION','buffer_bytes':current,'unix':time.time()});break
            time.sleep(.5)
    if STOP:dump(RUN/'TERMINAL.json',{'status':'OWN_PIPELINE_DRAINED','unix':time.time(),'verified':len(done)})

if __name__=='__main__':main()
