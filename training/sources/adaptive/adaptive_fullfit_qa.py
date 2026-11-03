"""Fullfit4413 real-cache CPU and preserved v3 zero-update B8x48/final5/RAM CUDA QA."""
from pathlib import Path
import argparse,gc,json,os,resource,time,threading,hashlib,csv
ROOT=Path(__file__).resolve().parent;TASK=ROOT.parent
os.environ['V83_PACKAGE_ROOT']=str(ROOT)
import numpy as np,pandas as pd,torch
import h1_base as h
import matched_core as matched
import train_v83 as base
from adaptive_inputs import AdaptiveBags,checked_extension,collate,loader
from fullfit_extensions import FullfitExtensionIndex as ExtensionIndex
from adaptive_model import make
from adaptive_fullfit import configure,frame_rosters,frames,sha,verify_manifest,prepare
from fullfit_budget import budget_guard
from models import loss_parts,optimizer
from token_slot_map import TokenSlotMap
from v83_aug_data import AugConfig
CFG=json.loads((TASK/'QUEUE_CONFIG.json').read_text())
def ram():
    root = Path("/sys/fs/cgroup")
    result = {"self_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "children_maxrss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}
    for name in ["memory.current", "memory.peak", "memory.max"]:
        p = root / name
        if p.exists():
            result[name] = p.read_text().strip()
    for name in ["memory.stat", "memory.events"]:
        p = root / name
        if p.exists():
            result[name] = {k: int(v) for k, v in (line.split() for line in p.read_text().splitlines())}
    stat = result.get("memory.stat", {})
    if result.get("memory.max", "max") != "max":
        # Page cache is reclaimable. Do not mistake memory.current's cached45GB for anonymous RAM.
        result["unreclaimable_upper_bytes"] = sum(stat.get(k, 0) for k in ["anon", "kernel", "shmem", "unevictable"])
        result["effective_reserve_bytes"] = int(result["memory.max"]) - result["unreclaimable_upper_bytes"]
    return result


def cpu(cache,ext):
    verify_manifest();budget_guard();prepare();configure(cache,ext)
    inv=json.loads((TASK/'EXTENSION_SERIALIZATION_INVENTORY.json').read_text());counts={}
    for entry in inv['files']:
        p=ext/entry['relative'];assert not p.is_symlink() and p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256']
        _bank,_refs,valid,_meta,_series,_oblique=checked_extension(p,entry['sample_id'])
        assert len(valid)==entry['windows'];counts[entry['sample_id']]=len(valid)
    for entry in inv['metadata']:
        p=ext/entry['relative'];assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256']
    names,fit,val,gold=frame_rosters(cache);assert len(fit)==4413 and len(val)==0 and len(gold)==58
    ix=ExtensionIndex(ext);assert set(ix.records)==set(fit.SampleID.astype(str))
    base_counts={}
    # Actual existing base cache is read-only and every4413 identity/schema is checked.
    for row in fit.itertuples(index=False):
        with np.load(row.cache_path,allow_pickle=False) as z:
            assert str(z['sample_id'])==str(row.SampleID) and str(z['orientation_state'])=='canonical_L_v8_3'
            assert z['bank'].dtype==np.uint8 and z['bank'].shape[1:]==(336,336) and z['meta'].shape==(64,6)
            base_counts[str(row.SampleID)]=int(z['mask'].sum())
    slot=TokenSlotMap(ROOT/'assets/NOE1_FULLFIT_TOKEN_SLOT_MAP.csv')
    selected=list(fit.index[fit.SampleID.astype(str).str.contains('::',regex=False)])+list(fit.index[-3:])+list(fit.index[:20])
    class Empty:
        def path(self,sid):assert sid in ix.records;return None
    parity=0
    for epoch in [0,19]:
        kw=dict(train=True,epoch=epoch,seed=20260908,slot_map=slot,config=AugConfig())
        frame=fit.iloc[selected].reset_index(drop=True);a=matched.ConfidenceBags(frame,names,**kw);b=AdaptiveBags(frame,names,ext_index=Empty(),**kw)
        for i in range(len(frame)):
            left,right=a[i],b[i]
            for j in [0,2,3,6]:assert torch.equal(left[j],right[j])
            assert torch.equal(left[1],right[1][:,:6]) and not right[1][:,6:].any() and left[4:6]==right[4:6];parity+=1
    test=AdaptiveBags(fit.iloc[-3:].reset_index(drop=True),names,train=True,epoch=0,seed=20260908,slot_map=slot,config=AugConfig(),ext_index=ix)
    y=torch.from_numpy(test.y);w=torch.from_numpy(test.label_weight)
    assert (w[:,:12]==1).all() and (w[:,12:]==0).all() and torch.isfinite(y).all()
    # Actual36main cells contribute nonzero finite gradients; all48aux cells are masked.
    logits=torch.zeros_like(y,requires_grad=True);main,aux,parts=loss_parts(logits,y,w,frames()[3],aux_weight=.25)
    parts.sum().backward();assert torch.isfinite(logits.grad).all() and float(logits.grad[:,:12].abs().sum())>0 and not logits.grad[:,12:].any() and not aux.any()
    batch=collate([test[i] for i in range(3)]);assert batch[0].shape[0]==3 and batch[2].shape[-1]==8
    heavy=sorted(sid for sid in fit.SampleID.astype(str) if counts.get(sid,0)>=64);assert len(heavy)>=8
    max_sid=max(fit.SampleID.astype(str),key=lambda sid:base_counts[sid]+counts.get(sid,0))
    d={'status':'PASS_ADAPTIVE_4413_CPU','manifest_sha256':sha(TASK/'CONTROL_MANIFEST.json'),'fit':4413,'parents':4406,'base_npz':4413,'extension_npz':4399,'test3_main_cells':36,'test3_aux_zero':True,'test3_loader_meta8':True,'actual_base_empty_parity_cases':parity,'heavy_fit_ids':heavy[:8],'max_eval_sample_id':max_sid,'max_actual_eval_valid':base_counts[max_sid]+counts.get(max_sid,0),'optimizer_updates':0,'scheduler_updates':0,'gold_fit_or_selection':True,'gold_metric_role':'in_sample_training_diagnostic','serialization_manifest_sha256':sha(TASK/'EXTENSION_SERIALIZATION_INVENTORY.json'),'ram':ram()}
    base.atomic_json(TASK/'CPU_QA.json',d)

def cuda(cache, ext):
    verify_manifest(); budget_guard(); configure(cache, ext)
    if torch.are_deterministic_algorithms_enabled() or torch.backends.cudnn.deterministic or os.environ.get('CUBLAS_WORKSPACE_CONFIG'):
        raise ValueError('Parity-only deterministic controls leaked into production resource QA')
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") != "expandable_segments:True":
        raise ValueError("Allocator must be set from process start")
    cpu_result = json.loads((TASK / "CPU_QA.json").read_text())
    if cpu_result["status"] != "PASS_ADAPTIVE_4413_CPU" or cpu_result["manifest_sha256"] != sha(TASK / "CONTROL_MANIFEST.json"):
        raise ValueError("Missing/current CPU QA")
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise ValueError("Exact single BF16 GPU required")
    torch.cuda.set_device(0); torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    h.seed_all(20260908)
    before_ram = ram()
    ram_stop = threading.Event()
    ram_min = {'sample': before_ram, 'count': 0}
    def monitor_ram():
        while not ram_stop.wait(.05):
            sample = ram(); ram_min['count'] += 1
            if sample.get('effective_reserve_bytes', 0) < ram_min['sample'].get('effective_reserve_bytes', 0):
                ram_min['sample'] = sample
    ram_thread = threading.Thread(target=monitor_ram, daemon=True)
    ram_thread.start()
    names, fit, val, gold = frame_rosters(cache)
    index = ExtensionIndex(ext); slot = TokenSlotMap(ROOT / "assets/NOE1_FULLFIT_TOKEN_SLOT_MAP.csv")
    normal = AdaptiveBags(fit.iloc[:8], names, train=True, seed=20260908, epoch=0, slot_map=slot, config=AugConfig(), ext_index=index)
    final = AdaptiveBags(fit.iloc[-5:], names, train=True, seed=20260908, epoch=0, slot_map=slot, config=AugConfig(), ext_index=index)
    heavy_frame = fit.set_index("SampleID").loc[cpu_result["heavy_fit_ids"]].reset_index()
    heavy = AdaptiveBags(heavy_frame, names, train=True, seed=20260908, epoch=0, slot_map=slot,
                         config=AugConfig(series_drop=0, token_drop=0, jitter=1, params=True), ext_index=index)
    model = make(ROOT / "assets").cuda(); model.train()
    opt, sched = optimizer(model, 24, 552, False)
    scheduler_initial = sched.last_epoch
    versions = {id(p): p._version for p in model.parameters()}
    for group in opt.param_groups:
        for p in group["params"]:
            if p.requires_grad:
                opt.state[p]["step"] = torch.tensor(0.)
                opt.state[p]["exp_avg"] = torch.zeros_like(p)
                opt.state[p]["exp_avg_sq"] = torch.zeros_like(p)
    def forbidden(*args, **kwargs):
        raise AssertionError("QA may never step optimizer/scheduler")
    opt.step = forbidden; sched.step = forbidden
    pw = frames()[3].cuda()
    records = []
    def progress(case, phase, **details):
        value = {"case": case, "phase": phase, "utc_unix": time.time(),
                 "optimizer_updates": 0, "scheduler_updates": 0,
                 "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                 "peak_reserved_bytes": torch.cuda.max_memory_reserved(), **details}
        base.atomic_json(TASK / "CUDA_PROGRESS.json", value)
        print(json.dumps(value), flush=True)
    for name, dataset in [("normal_B8", normal), ("final_B5", final), ("heavy_B8x48", heavy)]:
        opt.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        progress(name, "CASE_START")
        batch = collate([dataset[i] for i in range(len(dataset))])
        x, mask, meta, y, weight, _idx, gain, params = batch
        if name == "heavy_B8x48":
            if tuple(mask.shape) != (8, 48) or int(mask.sum()) != 384:
                raise ValueError("Heavy QA is not384 real valid tokens")
            params[:, :, 11] = 1.5; params[:, :, 12] = .03
        x, mask, meta, y, weight, gain, params = [t.cuda(non_blocking=True) for t in [x, mask, meta, y, weight, gain, params]]
        start = time.monotonic()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x, mask, meta, gain, params)
            _, _, losses = loss_parts(logits, y, weight, pw, aux_weight=.25)
            loss = losses.sum()
        if not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise ValueError("Nonfinite zero-update forward")
        progress(name, "FORWARD_FINITE", valid_tokens=int(mask.sum()))
        loss.backward()
        progress(name, "BACKWARD_COMPLETE")
        for p in model.parameters():
            if p.grad is not None:
                p.grad.div_(len(x))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3, error_if_nonfinite=True)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_reserved(); total = torch.cuda.get_device_properties(0).total_memory
        records.append({"case": name, "batch": int(len(x)), "valid_tokens": int(mask.sum()),
                        "seconds": time.monotonic() - start, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes": peak, "total_vram_bytes": total, "reserve_bytes": total - peak, "ram": ram()})
        progress(name, "CASE_COMPLETE", reserve_bytes=total-peak)
        if total - peak < max(2 * 2**30, total * .05):
            raise ValueError("CUDA reserve below2GiB/5percent")
        del batch, x, mask, meta, y, weight, gain, params, logits, losses, loss
    # Measure production workers4/prefetch1 with real heavy bags, not an arithmetic RAM estimate.
    repeated = pd.concat([heavy_frame] * 12, ignore_index=True)
    ds = AdaptiveBags(repeated, names, train=True, seed=20260908, epoch=0, slot_map=slot,
                      config=AugConfig(series_drop=0, token_drop=0, jitter=1, params=True), ext_index=index)
    production = loader(ds, 8, 4, 20260908, False)
    iterator = iter(production); prefetched = next(iterator)
    if prefetched[0].shape[:2] != (8, 48):
        raise ValueError("Production heavy loader mismatch")
    worker_pids = [worker.pid for worker in iterator._workers]
    for _ in range(6):
        prefetched = next(iterator)
        if prefetched[0].shape[:2] != (8, 48):
            raise ValueError('Warm heavy prefetch batch shape drift')
        time.sleep(.1)
    prefetch_samples = []
    for _ in range(10):
        time.sleep(.2)
        sample = ram()
        sample['worker_pids'] = [worker.pid for worker in iterator._workers]
        sample['workers_alive'] = all(worker.is_alive() for worker in iterator._workers)
        if not sample['workers_alive'] or len(sample['worker_pids']) != 4:
            raise ValueError('Production prefetch worker exited')
        sample['worker_status'] = {}
        for worker_pid in sample['worker_pids']:
            status = Path(f'/proc/{worker_pid}/status').read_text()
            sample['worker_status'][str(worker_pid)] = {
                line.split(':', 1)[0]: line.split(':', 1)[1].strip()
                for line in status.splitlines() if line.startswith(('VmRSS:', 'VmHWM:', 'VmLck:'))}
        prefetch_samples.append(sample)
    prefetch_ram = min(prefetch_samples, key=lambda sample: sample.get('effective_reserve_bytes', 0))
    iterator._shutdown_workers()
    if any(Path(f'/proc/{pid}').exists() for pid in worker_pids):
        raise ValueError('QA worker cleanup incomplete')
    del prefetched, iterator, production, ds
    opt.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
    combined = pd.concat([fit, val, gold], ignore_index=True)
    eval_frame = combined.loc[combined.SampleID.astype(str) == cpu_result["max_eval_sample_id"]].reset_index(drop=True)
    row = AdaptiveBags(eval_frame, names, train=False, ext_index=index)[0]
    bx, bm, bmeta, _y, _w, _i, bg, _bp = collate([row])
    actual_eval = int(bm.sum())
    if actual_eval > 172:
        raise ValueError("Eval bound exceeded")
    if actual_eval < 172:
        take = torch.arange(172) % actual_eval
        bx, bmeta = bx[:, take], bmeta[:, take]
        bm = torch.ones((1, 172), dtype=torch.bool)
    model.eval(); torch.cuda.reset_peak_memory_stats()
    progress("eval172", "CASE_START")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = model(bx.cuda(), bm.cuda(), bmeta.cuda(), bg.cuda())
    torch.cuda.synchronize()
    if not torch.isfinite(prediction).all():
        raise ValueError("Nonfinite full eval172")
    total = torch.cuda.get_device_properties(0).total_memory; peak = torch.cuda.max_memory_reserved()
    if total - peak < max(2 * 2**30, total * .05):
        raise ValueError("Full eval VRAM reserve failed")
    if sched.last_epoch != scheduler_initial or any(p._version != versions[id(p)] for p in model.parameters()):
        raise ValueError("QA parameter/scheduler mutation")
    if any(float(state["step"]) != 0 for state in opt.state.values()):
        raise ValueError("Optimizer advanced during QA")
    after_ram = ram()
    ram_stop.set(); ram_thread.join(timeout=2)
    for key in ["oom", "oom_kill"]:
        if after_ram.get("memory.events", {}).get(key, 0) != before_ram.get("memory.events", {}).get(key, 0):
            raise ValueError("New cgroup OOM during QA")
    if min(prefetch_ram.get("effective_reserve_bytes", 0), after_ram.get("effective_reserve_bytes", 0), ram_min['sample'].get('effective_reserve_bytes', 0)) < 4 * 2**30:
        raise ValueError("Host anonymous/kernel RAM reserve below4GiB")
    guard=budget_guard()
    forecast_hours_upper=CFG['fullfit_upper_seconds']/3600+CFG['retention_upper_seconds']/3600+1
    forecast=guard['budget_upper_usd']+forecast_hours_upper*CFG['budget']['hourly_usd']
    assert forecast<CFG['budget']['stop_usd'] and time.time()+forecast_hours_upper*3600<CFG['budget']['hard_stop_unix']-120
    forecast_hours=forecast_hours_upper
    result = {"status": "PASS_ADAPTIVE_4413_ZERO_UPDATE_CUDA", "manifest_sha256": sha(TASK / "CONTROL_MANIFEST.json"),
              "optimizer_updates": 0, "scheduler_updates": 0, "AdamW_moments_allocated": True,
              "cases": records, "eval_valid_tokens": 172, "eval_actual_source_valid": actual_eval,
              "eval_stress_repeat_only_if_actual_shorter": actual_eval < 172,
              "eval_peak_reserved_bytes": peak, "production_workers": 4, "prefetch_factor": 1,
              "before_ram": before_ram, "prefetch_ram": prefetch_ram, "prefetch_samples": prefetch_samples, "after_ram": after_ram,
              "continuous_ram_minimum": ram_min, "QA_worker_groups_absent": True,
              "forecast_hours_upper": forecast_hours, "forecast_upper_usd": forecast,
              "QA_model_discarded_fresh_train_required": True}
    base.atomic_json(TASK / "CUDA_QA.json", result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cpu','cuda']);a=p.parse_args()
    (cpu if a.mode=='cpu' else cuda)(TASK/'cache_union',Path(CFG['extension_root']))
