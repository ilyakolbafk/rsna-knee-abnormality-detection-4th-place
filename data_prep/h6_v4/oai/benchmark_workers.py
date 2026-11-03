"""Same 12 full cases at 1/2/3/4 processes, isolated from delivered cache."""
from common import *
import concurrent.futures as cf
import multiprocessing, fcntl, psutil, re

BENCH=RUN/'worker_benchmark'

def unit(cap,case):
    import build_case as original
    job=BENCH/f'p{cap}'/case
    job.mkdir(parents=True,exist_ok=True)
    for name in ['component_discovery','handoff','code','BUILD_CONFIG.frozen.json']:
        q=job/name
        if not q.exists():q.symlink_to(RUN/name,target_is_directory=(RUN/name).is_dir())
    sources=json.loads((BENCH/'SOURCE_LOOKUP.frozen.json').read_text())
    dump(job/'LOCAL_RAW_LOOKUP.json',sources)
    # Only the output root changes. Original builder body, source components,
    # model, recipe, orientation and pixels remain frozen and independently pinned.
    original.RUN=job
    result=original.build(case)
    return {k:result[k] for k in ['case_id','bytes','rows','pixel_sha256','metadata_sha256','seconds','peak_rss_bytes','rss_bytes','recipe_sha256']}

def vm_sample():
    output=subprocess.check_output(['vm_stat'],text=True)
    page=int(re.search(r'page size of (\d+)',output).group(1))
    counts={k:int(re.search(r'^'+k+r':\s+(\d+)',output,re.M).group(1))*page for k in ['Swapins','Swapouts']}
    pressure=int(subprocess.check_output(['sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True))
    return {'unix':time.time(),'available_bytes':psutil.virtual_memory().available,'pressure_level':pressure,
            'swap_in_bytes_total':counts['Swapins'],'swap_out_bytes_total':counts['Swapouts']}

def main():
    BENCH.mkdir(exist_ok=True)
    lock=(BENCH/'singleton.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    parent_lock=(RUN/'pipeline.lock').open('a');fcntl.flock(parent_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    available=json.loads((RUN/'BENCH_AVAILABLE.frozen.json').read_text())
    ids=['OAI0001','OAI0008','OAI0177','OAI0178','OAI0179','OAI0180','OAI0181','OAI0182','M6B0001','M6000002','OMV00_9148828_R','OMV00_9220603_L']
    byid={x['case_id']:x for x in available};assert len(ids)==12 and all(c in byid for c in ids)
    lookup={h:[{'path':p}] for case in ids for h,p in byid[case]['sources'].items()}
    for h,paths in lookup.items():assert sha(paths[0]['path'])==h
    dump(BENCH/'SOURCE_LOOKUP.frozen.json',lookup)
    config=json.loads((RUN/'BUILD_CONFIG.frozen.json').read_text())
    dump(BENCH/'CONTRACT.frozen.json',{'cases':ids,'concurrency':[1,2,3,4],'recipe_sha256':config['semantic_sha256'],
        'same_input_each_phase':True,'includes_cold_start':True,'threads_per_process':2,'phase_order':[1,2,3,4],
        'memory_throttle':'pause new work if pressure critical and available below1GiB; in-flight units drain',
        'outputs':'isolated benchmark namespaces; no corpus receipt overwrite or source cleanup','unix':time.time()})
    summary=[];reference={}
    for cap in [1,2,3,4]:
        prior=BENCH/f'RESULT.p{cap}.json'
        if prior.exists():
            old=json.loads(prior.read_text());summary.append(old)
            if cap==1:reference={r['case_id']:(r['pixel_sha256'],r['metadata_sha256']) for r in old['cases']}
            continue
        start=time.monotonic();initial=vm_sample();samples=[];results=[];pending={};todo=list(ids);last_sample=0
        with cf.ProcessPoolExecutor(max_workers=cap,mp_context=multiprocessing.get_context('spawn')) as pool:
            while todo or pending:
                sample=vm_sample();children=[]
                for p in pool._processes.values():
                    try:children.append({'PID':p.pid,'rss':psutil.Process(p.pid).memory_info().rss})
                    except psutil.NoSuchProcess:pass
                sample['workers']=children;sample['active_cases']=list(pending.values());sample['cap']=cap
                if time.monotonic()-last_sample>2:
                    samples.append(sample);dump(BENCH/'STATE.json',{'phase':cap,'completed':len(results),'total':12,'elapsed_seconds':time.monotonic()-start,**sample});last_sample=time.monotonic()
                for future,case in list(pending.items()):
                    if future.done():
                        result=future.result();results.append(result);del pending[future]
                        if cap==1:reference[case]=(result['pixel_sha256'],result['metadata_sha256'])
                        assert (result['pixel_sha256'],result['metadata_sha256'])==reference[case], 'FROZEN_PIXEL_PARITY_FAILED'
                throttle=sample['pressure_level']>=4 and sample['available_bytes']<2**30
                while todo and len(pending)<cap and not throttle:
                    case=todo.pop(0);pending[pool.submit(unit,cap,case)]=case
                if time.monotonic()-start>600:raise TimeoutError('BENCH_PHASE_DEADLINE')
                time.sleep(.5)
        final=vm_sample();elapsed=time.monotonic()-start
        result={'processes':cap,'elapsed_seconds':elapsed,'cases_per_minute':12*60/elapsed,
            'output_MB_per_second':sum(x['bytes'] for x in results)/elapsed/1e6,
            'min_available_bytes':min(x['available_bytes'] for x in samples),'max_pressure_level':max(x['pressure_level'] for x in samples),
            'max_group_rss_bytes':max(sum(p['rss'] for p in x['workers']) for x in samples),
            'swap_out_bytes':final['swap_out_bytes_total']-initial['swap_out_bytes_total'],
            'swap_in_bytes':final['swap_in_bytes_total']-initial['swap_in_bytes_total'],
            'all_pixel_metadata_parity':True,'samples':samples,'cases':results,'unix':time.time()}
        dump(prior,result);summary.append(result)
        print(json.dumps({k:v for k,v in result.items() if k not in ['samples','cases']}),flush=True)
    ordered=sorted(summary,key=lambda x:x['elapsed_seconds'])
    report={'status':'PASS_BENCHMARK','fastest_processes':ordered[0]['processes'],'rows':[{k:v for k,v in r.items() if k not in ['samples','cases']} for r in summary],
            'all_pixel_metadata_parity':True,'recipe_sha256':config['semantic_sha256'],'unix':time.time()}
    dump(BENCH/'SUMMARY.json',report);dump(BENCH/'STATE.json',report)
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
