"""New task namespace; existing Mac NDA access only, no credential export."""
from common import *
import argparse
import getpass
import shutil
import signal
import sys

def child(entry, links, target):
    import requests
    # macOS keyring updates delete then add. Protect NDA reads and writes
    # across children so another login never observes that temporary gap.
    import keyring,fcntl,threading
    from contextlib import contextmanager
    guard=threading.RLock();state=threading.local()
    @contextmanager
    def state_lock():
        with guard:
            if getattr(state,'held',False):
                yield;return
            with (RUN/'nda_keyring.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX);state.held=True
                try:yield
                finally:state.held=False
    for method in ['get_password','set_password']:
        original_keyring=getattr(keyring,method)
        def serialized(service,*args,_original=original_keyring,**kwargs):
            if service!='nda-tools':return _original(service,*args,**kwargs)
            with state_lock():return _original(service,*args,**kwargs)
        setattr(keyring,method,serialized)
    from NDATools.Configuration import ClientConfiguration
    for method in ['__init__','_save_username','_check_and_fix_missing_options']:
        if not hasattr(ClientConfiguration,method):continue
        original_config=getattr(ClientConfiguration,method)
        def serialized_config(*args,_original=original_config,**kwargs):
            with state_lock():return _original(*args,**kwargs)
        setattr(ClientConfiguration,method,serialized_config)
    original=requests.sessions.Session.request
    def request(session, method, url, **kwargs):
        if (RUN/'SOURCE_STOP.json').exists(): raise RuntimeError('LATCHED_SOURCE_STOP')
        session.trust_env=False
        kwargs['proxies']={'http':'','https':'','all':''}
        if kwargs.get('timeout') is None: kwargs['timeout']=(15,90)
        response=original(session,method,url,**kwargs)
        if response.status_code in (400,401,403,429):
            latch_source_stop({'case_id':entry['case_id'],'http_status':response.status_code,
                 'unix':time.time(),'action':'RECONCILE_AUTH_QUOTA_OR_SOURCE_REJECTION_BEFORE_RETRY'})
            raise RuntimeError(f'SOURCE_HTTP_{response.status_code}_STOP')
        return response
    requests.sessions.Session.request=request
    def no_password(*args,**kwargs):
        latch_source_stop({'case_id':entry['case_id'],'reason':'NDA_CACHED_CREDENTIAL_UNAVAILABLE',
                                  'unix':time.time(),'action':'RECONCILE_SERIALIZED_KEYRING_AND_EXISTING_ACCESS_BEFORE_RETRY'})
        raise SystemExit('NDA_CACHED_CREDENTIAL_UNAVAILABLE')
    getpass.getpass=no_password
    from NDATools.clientscripts.downloadcmd import main
    # Your own NDA account with approved OAI access and your own NDA data package id.
    user=os.environ.get('NDA_USERNAME');package=os.environ.get('NDA_PACKAGE_ID')
    if not user or not package: raise SystemExit('set NDA_USERNAME and NDA_PACKAGE_ID (see data_prep/h6_v4/oai/README.md)')
    sys.argv=['downloadcmd','-dp',package,'-u',user,'-t',str(links),'-d',str(target),'-wt','2']
    main()

def _fetch(case):
    import psutil
    os.umask(0o077)
    entry=next(x for x in roster() if x['case_id']==case)
    target=RUN/'scratch'/case/'downloads'; target.mkdir(parents=True,exist_ok=True)
    missing=[]
    for x in entry['objects']:
        p=target/x['object']['download_alias']
        if p.is_file():
            if p.stat().st_size<int(x['object']['archive_bytes']):
                isolated=RUN/'repairs/source-incomplete'/case/f'{uuid.uuid4().hex}.{p.name}'
                isolated.parent.mkdir(parents=True,exist_ok=True)
                record={'case_id':case,'status':'SHORT_FINAL_PRESERVED_FOR_FRESH_REQUEST','family':x['object']['family'],'bytes':p.stat().st_size,'expected_bytes':int(x['object']['archive_bytes']),'sha256':sha(p),'isolated_path':str(isolated.relative_to(RUN)),'unix':time.time()}
                os.replace(p,isolated);dump(isolated.with_name(isolated.name+'.json'),record);missing.append(x)
            elif p.stat().st_size!=int(x['object']['archive_bytes']) or sha(p)!=x['archive_sha256']:
                latch_source_stop({'case_id':case,'reason':'EXISTING_COMPLETE_SOURCE_SHA_OR_SIZE_MISMATCH','unix':time.time()})
                raise RuntimeError('EXISTING_SOURCE_SHA_CONFLICT')
        else: missing.append(x)
    if not missing: return target
    if (RUN/'SOURCE_STOP.json').exists(): raise RuntimeError('LATCHED_SOURCE_STOP')
    if shutil.disk_usage(RUN).free<30*2**30+3_000_000_000:raise RuntimeError('MAC_DISK_FLOOR')
    # NDA0.7.0 matches saved jobs by destination and input-list basename only,
    # and subtracts all prior completed IDs from a changed request. A fresh
    # basename prevents that stale tracking from skipping missing objects.
    links=RUN/'scratch'/case/f'source_paths.{uuid.uuid4().hex}.txt'
    links.write_text('\n'.join(x['object']['image_file'] for x in missing)+'\n')
    logs=RUN/'private_logs';logs.mkdir(exist_ok=True);os.chmod(logs,0o700)
    logpath=logs/f'{case}.nda.log'
    cmd=['uv','tool','run','--offline','--from','nda-tools==0.7.0','--with','psutil','python',str(Path(__file__)),case,'--child','--links',str(links)]
    start=last=time.monotonic(); previous=None;reason=None
    with logpath.open('a') as log:
        proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while proc.poll() is None:
                sig=[]
                for p in target.rglob('*'):
                    try:
                        if p.is_file():
                            st=p.stat();sig.append((str(p),st.st_size,st.st_mtime_ns))
                    except FileNotFoundError: pass
                sig=tuple(sorted(sig));now=time.monotonic()
                if sig!=previous: last=now;previous=sig
                if sum(x[1] for x in sig)>3_000_000_000 or shutil.disk_usage(RUN).free<30*2**30:
                    reason='MAC_SCRATCH_DISK_FLOOR';break
                if now-last>90 or now-start>300: reason='SOURCE_TRANSPORT_DEADLINE';break
                time.sleep(1)
        finally:
            if proc.poll() is None:
                try: os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError: pass
                try: proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)
    if reason or proc.returncode:
        content=logpath.read_text(errors='replace').lower()
        markers=[s for s in ['credential','unauthorized','forbidden','timeout','timed out','proxyerror','connectionerror'] if s in content]
        result={'case_id':case,'status':'SOURCE_FETCH_FAILED','reason':reason or 'NDA_CHILD_EXIT',
                'exit_code':proc.returncode,'sanitized_markers':markers,'unix':time.time()}
        dump(RUN/'download_receipts'/f'{case}.json',result)
        raise RuntimeError(json.dumps(result))
    if (RUN/'SOURCE_STOP.json').exists():raise RuntimeError('LATCHED_SOURCE_STOP')
    incomplete=[]
    for x in entry['objects']:
        p=target/x['object']['download_alias']
        if not p.is_file() or p.stat().st_size<int(x['object']['archive_bytes']):
            incomplete.append(x['object']['family']);continue
        if p.stat().st_size!=int(x['object']['archive_bytes']) or sha(p)!=x['archive_sha256']:
            latch_source_stop({'case_id':case,'reason':'COMPLETE_SOURCE_SHA_OR_SIZE_MISMATCH','unix':time.time()})
            raise RuntimeError('COMPLETE_SOURCE_SHA_OR_SIZE_MISMATCH')
    if incomplete:
        dump(RUN/'download_receipts'/f'{case}.json',{'case_id':case,'status':'SOURCE_TRANSPORT_INCOMPLETE','missing_families':incomplete,'unix':time.time(),'complete_sha_conflict':False})
        raise RuntimeError('SOURCE_TRANSPORT_INCOMPLETE:'+','.join(incomplete))
    dump(RUN/'download_receipts'/f'{case}.json',{'case_id':case,'status':'FROZEN_SOURCE_SHA_VERIFIED',
         'objects':5,'bytes':entry['source_archive_bytes'],'seconds':time.monotonic()-start,
         'credentials_exported':False,'unix':time.time()})
    return target

def fetch(case):
    import fcntl
    assert '/' not in case and '..' not in case
    folder=RUN/'source_fetch_locks';folder.mkdir(exist_ok=True)
    with (folder/f'{case}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _fetch(case)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('case');ap.add_argument('--child',action='store_true');ap.add_argument('--links',type=Path);a=ap.parse_args()
    if a.child:
        entry=next(x for x in roster() if x['case_id']==a.case)
        child(entry,a.links or RUN/'scratch'/a.case/'source_paths.txt',RUN/'scratch'/a.case/'downloads')
    else:
        target=fetch(a.case);print(json.dumps({'status':'RAW_SHA_VERIFIED','case_id':a.case}))
