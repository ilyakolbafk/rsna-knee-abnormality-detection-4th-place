from common import *
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing
from build_case import build

if __name__=='__main__':
    cases=['OAI0001','OAI0008','OMV00_9444196_R','OMV00_9220603_L','M6B0001','M6000002']
    with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(build,c):c for c in cases}
        for f in as_completed(futures):
            try:f.result()
            except Exception as e:
                dump(RUN/'qa'/f'{futures[f]}.build_error.json',{'case_id':futures[f],'error':str(e),'unix':time.time()})
                print(json.dumps({'case_id':futures[f],'error':str(e)}),flush=True)
