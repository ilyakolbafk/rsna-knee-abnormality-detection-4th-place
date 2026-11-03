from common import *
from fetch_case import fetch
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl

lock=(RUN/'prefetch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
lookup=json.loads((RUN/'LOCAL_RAW_LOOKUP.json').read_text())
rows=[r for r in roster() if not all(x['archive_sha256'] in lookup for x in r['objects'])][:24]
dump(RUN/'PREFETCH_OWNER.json',{'PID':os.getpid(),'unix':time.time(),'workers':4,'cases':len(rows),'stage':'canary_input_buffer','max_declared_bytes':sum(r['source_archive_bytes'] for r in rows)})
assert sum(r['source_archive_bytes'] for r in rows)<2_000_000_000
with ThreadPoolExecutor(max_workers=4) as pool:
    jobs={pool.submit(fetch,r['case_id']):r['case_id'] for r in rows}
    for f in as_completed(jobs):
        try:
            f.result();print(json.dumps({'case_id':jobs[f],'status':'RAW_SHA_VERIFIED','unix':time.time()}),flush=True)
        except Exception as e:
            dump(RUN/'PREFETCH_ERROR.json',{'case_id':jobs[f],'error':str(e),'unix':time.time()})
            for j in jobs:j.cancel()
            raise
