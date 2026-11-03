"""Separate CPU-only Gold58 and fastMRI holdout diagnostics; never called by fit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
from pathlib import Path
import argparse,math,time,hashlib,json,fcntl
import numpy as np
import torch
import psutil
from common import ROOT,read,write,sha,pin_check
from full_cache_loader import Reader
from fm_data import Store
from fm_math import fm_window_image,fm_window_targets
from train import make_model
from gold_metrics import metrics
from fm_auc import fm_auc
from diagnostic_receipts import expected_binding


class ResourceDeferred(RuntimeError):pass


def resource_guard():
    available=psutil.virtual_memory().available
    cg=Path('/sys/fs/cgroup/memory')
    if (cg/'memory.limit_in_bytes').exists():
        limit=int((cg/'memory.limit_in_bytes').read_text());used=int((cg/'memory.usage_in_bytes').read_text())
        st=dict(line.split() for line in (cg/'memory.stat').read_text().splitlines())
        available=min(available,limit-used+int(st.get('inactive_file',0)))
    elif Path('/sys/fs/cgroup/memory.max').exists():
        root=Path('/sys/fs/cgroup');text=(root/'memory.max').read_text().strip()
        if text!='max':
            st=dict(line.split() for line in (root/'memory.stat').read_text().splitlines())
            available=min(available,int(text)-int((root/'memory.current').read_text())+int(st.get('inactive_file',0)))
    if available<24*2**30:raise ResourceDeferred('WORKING_RESERVE24GIB')
    psi=Path('/proc/pressure/memory')
    if psi.exists():
        full=next(line for line in psi.read_text().splitlines() if line.startswith('full '))
        values=dict(word.split('=') for word in full.split()[1:])
        if float(values['avg10'])>=1:raise ResourceDeferred('MEMORY_PSI_FULL_AVG10')


def cpu_token():
    root=Path('/workspace/knee_v4_cache_20261021/shared/cpu_eval_slots');root.mkdir(parents=True,exist_ok=True)
    quota=os.cpu_count() or 1
    base=Path('/sys/fs/cgroup/cpu')
    if (base/'cpu.cfs_quota_us').exists():
        q=int((base/'cpu.cfs_quota_us').read_text());p=int((base/'cpu.cfs_period_us').read_text())
        if q>0:quota=min(quota,q/p)
    elif Path('/sys/fs/cgroup/cpu.max').exists():
        q,p=Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if q!='max':quota=min(quota,int(q)/int(p))
    count=int(quota*.55//4)
    for i in range(count):
        f=(root/f'slot{i}.lock').open('a+')
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return f
        except BlockingIOError:f.close()
    raise ResourceDeferred('GLOBAL_CPU_TOKEN_CAP')


@torch.inference_mode()
def fm_holdout(model,store,*,chunk=8,guard=resource_guard):
    ps=[];ys=[];volume_ids=[];pixels=store.pixels()
    specs=[(vi,view,k) for vi in store.holdout for view in [0,2,3] for k in range(int(store.n[vi]))]
    for start in range(0,len(specs),chunk):
        guard();images=[];target=[]
        for vi,view,k in specs[start:start+chunk]:
            rows=store.window_rows(vi,view,k)
            images.append(fm_window_image(pixels,rows,352,16,16,0,0,0,False))
            _,present,_=fm_window_targets(store.boxes_of(rows),352,16,16,0,0,0,False)
            target.append(present);volume_ids.append(vi)
        dense=model.fm_dense(torch.from_numpy(np.stack(images)))
        p=torch.sigmoid(dense.float().flatten(2).logsumexp(-1)-math.log(121)).numpy()
        ps.append(p);ys.append(np.asarray(target)>0.5)
    p=np.concatenate(ps);y=np.concatenate(ys);vid=np.asarray(volume_ids);volumes=sorted(set(volume_ids))
    pv=np.stack([p[vid==v].max(0) for v in volumes]);yv=np.stack([y[vid==v].any(0) for v in volumes])
    safe=lambda v:float(v) if np.isfinite(v) else None
    return {'window_auc':{str(c):safe(fm_auc(y[:,c],p[:,c])) for c in range(11)},'volume_auc':{str(c):safe(fm_auc(yv[:,c],pv[:,c])) for c in range(11)},
            'windows':len(specs),'volumes':len(volumes),'precision':'CPU_FP32','sampling':'all slices in views0/2/3, central352, no augmentation','selection':False}


@torch.inference_mode()
def gold58(model,inputs,labels,folder,binding,*,phys_mm):
    reader=Reader(inputs['corpora']['RSNA'],diagnostic=True)
    rows=labels['rows'];pred=[]
    if len(rows)!=58 or {r['id'] for r in rows}!=set(read(ROOT/'assets/GOLD_EXCLUDED_IDS.json')):raise ValueError('EXACT_GOLD58_REQUIRED')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    for r in rows:
        uid=r['id'];file=folder/(hashlib.sha256(uid.encode()).hexdigest()+'.json')
        if file.exists():
            rec=read(file)
            if rec['binding']!=binding or rec['id']!=uid:raise ValueError('GOLD_RESUME_BINDING')
            values=rec['prediction']
        else:
            resource_guard();key=next(k for k,v in reader.d['studies'].items() if v['sample_id']==uid)
            x,m,_=reader.windows(key,mode='eval',seed=1,epoch=0,index=0,nonce=0,phys_mm=phys_mm)
            image=torch.from_numpy(x[...,16:368,16:368]);meta=torch.from_numpy(m)[None];mask=torch.ones(1,len(x),dtype=torch.bool)
            values=model.gold(image[None],meta,mask).float().sigmoid()[0].numpy().astype(np.float32).tolist()
            write(file,{'id':uid,'prediction':values,'binding':binding})
        pred.append(values)
        write(folder/'PROGRESS.json',{'state':'CPU_GOLD58_DIAGNOSTICS','checked':len(pred),'total':58,'at_unix':time.time()})
    y=np.asarray([r['y'] for r in rows]);p=np.asarray(pred,np.float32)
    result=metrics(y,p,labels['targets']);result.update(ids=[r['id'] for r in rows],predictions=p.tolist(),Gold_fit=False,
        scope='REUSED_GOLD58_DESCRIPTIVE_NO_SELECTION',precision='CPU_FP32',binding=binding)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['core6','vm'],required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--gold-labels',required=True);ap.add_argument('--fm-holdout',action='store_true');a=ap.parse_args()
    package=pin_check();resource_guard();token=cpu_token();torch.set_num_threads(4)
    if hasattr(os,'nice'):os.nice(10)
    if os.name=='posix' and Path('/proc/self').exists():
        import ctypes,subprocess
        ctypes.CDLL(None).prctl(41,1,0,0,0)
        subprocess.run(['ionice','-c','2','-n','7','-p',str(os.getpid())],check=True,capture_output=True)
    checkpoint=Path(a.checkpoint).resolve();out=(ROOT/'outputs'/a.arm).resolve()
    if not checkpoint.is_relative_to(out):raise ValueError('CHECKPOINT_OUTSIDE_NEW_ARM')
    cfg=read(ROOT/'CONFIG.json');reference=read(ROOT/'assets/GOLD_DIAGNOSTIC_SOURCE.json')
    if sha(a.gold_labels)!=reference['sha256']:raise ValueError('GOLD_LABELS_REFERENCE_SHA')
    binding=expected_binding(ROOT,a.arm,checkpoint,package)
    before=checkpoint.stat();state=torch.load(checkpoint,map_location='cpu',weights_only=False);after=checkpoint.stat()
    if (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('CHECKPOINT_CHANGED_DURING_LOAD')
    if state['arm']!=a.arm or state['package_sha256']!=package or (state.get('kind')!='ema' and state.get('averaging_source')!='EMA'):raise ValueError('EMA_OR_FIXED_EMA_SWA_ONLY')
    model=make_model(a.arm,'cpu',{**cfg,'chunk':8});model.load_state_dict(state['model'],strict=True);model.eval()
    inputs=read(ROOT/'runtime/INPUTS.json')
    folder=ROOT/'runtime/diagnostics'/a.arm/checkpoint.stem
    result=gold58(model,inputs,read(a.gold_labels),folder/'studies',binding,phys_mm=cfg['phys_mm'])
    write(folder/'GOLD58.json',result)
    if a.fm_holdout:write(folder/'FM_HOLDOUT.json',{**fm_holdout(model,Store(inputs['fastMRI'])),'binding':binding})
    print(json.dumps({'status':'PASS_CPU_DIAGNOSTICS','Gold_checked':58,'arm':a.arm,'checkpoint':checkpoint.name}))


if __name__=='__main__':
    try:main()
    except ResourceDeferred as exc:
        print(json.dumps({'status':'RESOURCE_DEFERRED','reason':str(exc)}))
        raise SystemExit(75)
