from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,importlib.util,json,os,shutil,signal,subprocess,sys,tarfile,threading,time
START=time.perf_counter();ENTRY=datetime.now(timezone.utc).isoformat()
def phase(name,**kw):print('RESIDENT_PHASE',json.dumps(dict(phase=name,seconds=time.perf_counter()-START,utc=datetime.now(timezone.utc).isoformat(),**kw)),flush=True)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
phase('ENTRY')
ROOT=Path(os.environ.get('RSNA_INPUT_ROOT','/kaggle/input'));WORK=Path(os.environ.get('RSNA_WORK_DIR','/kaggle/working'));TMP=Path('/tmp/rsna-sixway-resident-20261017');TMP.mkdir()
SLUG='rsna-sixway-resident-runtime-20261017';EXPECTED='089c50271dcb765ee339eb0cff9246c34409304f1a09dbe73c58e8c05fdd8b28'
matches=[p for p in [ROOT/SLUG,ROOT/'datasets/ilyakolb'/SLUG] if (p/'RELEASE_PACKET.json').is_file() and sha(p/'RELEASE_PACKET.json')==EXPECTED]
assert len(matches)==1;DS=matches[0];release=read(DS/'RELEASE_PACKET.json')
for name,pin in release['files'].items():assert sha(DS/name)==pin,name
models=release['models'];assert len(models)==10
for item in models:
 candidates=[p for p in [ROOT/item['dataset_slug'],ROOT/'datasets/ilyakolb'/item['dataset_slug']] if p.is_dir()];assert len(candidates)==1
 item['mount']=str(candidates[0]);r=read(candidates[0]/item['release_file']);assert sha(candidates[0]/item['release_file'])==item['release_sha256']
 item['training_identity']=r.get('training_identity') if item['key']=='mirror' else r['models'][0]['training_identity']
 item['checkpoint']=str(candidates[0]/(item['key']+'.pt' if item['binding'] else 'checkpoint.pt'))
 item['original_release']=r
A=next(m for m in models if m['key']=='adaptive');M=next(m for m in models if m['key']=='mirror')
sys.path.insert(0,A['mount']);from kaggle_mounts import competition_parent
COMP=competition_parent(ROOT,release['competition']);sample_rows=list(csv.DictReader((COMP/'sample_submission.csv').open()));uids=[r['StudyInstanceUID'] for r in sample_rows]
ref_uids=[r['StudyInstanceUID'] for r in csv.DictReader((DS/'ref_adaptive.csv').open())];SAMPLE_RUN=uids==ref_uids
assert len(uids)>=2 and len(uids)==len(set(uids))
TOTAL_LIMIT=1800 if SAMPLE_RUN else 13400
signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(TimeoutError('Resident totalruntimecap')));signal.alarm(TOTAL_LIMIT)
def unpack(item,name,dest):
 mount=Path(item['mount']);archive=mount/name;dest.mkdir()
 if archive.is_file():
  assert sha(archive)==item['original_release']['files'][name]
  with tarfile.open(archive) as tar:tar.extractall(dest,filter='data')
 else:
  dest.rmdir();shutil.copytree(mount/Path(name).stem,dest)
 pins=item['original_release']['archive_files'][name]
 assert {str(p.relative_to(dest)) for p in dest.rglob('*') if p.is_file()}==set(pins)
 for name,pin in pins.items():assert sha(dest/name)==pin,name
 return dest
SOURCE=unpack(A,'source.tar',TMP/'canonical_source');RUNTIME=unpack(A,'r3_runtime.tar',TMP/'canonical_runtime');WHEELS=unpack(A,'wheels.tar',TMP/'wheels')
RAW_SOURCE=unpack(M,'source.tar',TMP/'raw_source');RAW_RUNTIME=unpack(M,'runtime.tar',TMP/'raw_runtime')
for model in models:
 if model['key']=='mirror':continue
 for name,pin in model['original_release']['archive_files']['source.tar'].items():assert sha(SOURCE/name)==pin,(model['key'],name)
assert not (RAW_SOURCE/'laterality').exists() and not list(RAW_SOURCE.rglob('*.pt'))
DEPS=TMP/'deps';DEPS.mkdir()
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename
tags=set(sys_tags());wheels=[p for p in sorted(WHEELS.glob('*.whl')) if parse_wheel_filename(p.name)[3]&tags]
assert any(p.name.startswith('timm-') for p in wheels) and any(p.name.startswith('pydicom-') for p in wheels)
subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','--no-index','--no-deps','--target',str(DEPS),*map(str,wheels)],check=True)
for folder,name,new in [(RUNTIME,'process_controller.py','resident_controller.py'),(RAW_RUNTIME,'isolated_worker.py','mirror_packed_worker.py')]:
 shutil.copyfile(DS/new,folder/name);manifest=read(folder/'SOURCE_MANIFEST.json');manifest['runtime'][name]=sha(folder/name);save(folder/'SOURCE_MANIFEST.json',manifest)
phase('BOOTSTRAP_COMPLETE',dependency_installs=1,canonical_source_extractions=1,raw_source_extractions=1)
inp=TMP/'official_input';inp.mkdir()
for name in ['sample_submission.csv','test_series.csv']:shutil.copyfile(COMP/name,inp/name)
(inp/'test_series').symlink_to(COMP/'test_series',target_is_directory=True)
series_rows=list(csv.DictReader((COMP/'test_series.csv').open()));sizes={u:0 for u in uids}
for r in series_rows:sizes[r['StudyInstanceUID']]+=1
allocation=[[],[]];cost=[0,0]
for uid in sorted(uids,key=lambda u:(-sizes[u],u)):
 owner=min(range(2),key=lambda i:(cost[i],len(allocation[i]),i));allocation[owner].append(uid);cost[owner]+=sizes[uid]
assert all(allocation)
RAW_BANK=TMP/'raw_bank';RAW_BANK.mkdir()
env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(DS),str(DEPS)]),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',RSNA_SHARED_RAW_BANK=str(RAW_BANK))
env.pop('RSNA_SHARED_INPUT_BANK',None)
guard=RAW_RUNTIME/'process_guard.py';assert sha(guard)=='02c102e4b4f25fbc16e854c6a9c8e4efde895145be4ce37591b65917612071b7'
def run(args,child_env,log):
 spec=importlib.util.spec_from_file_location('rsna_resident_guard',guard);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
 with Path(log).open('w',buffering=1) as f:
  process=subprocess.Popen(args,env=child_env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   deadline=time.monotonic()+TOTAL_LIMIT-(time.perf_counter()-START)
   while process.poll() is None:
    try:process.wait(timeout=15)
    except subprocess.TimeoutExpired:
     phase('CONTROLLER_RUNNING',controller=Path(log).stem)
     assert time.monotonic()<deadline
   assert process.returncode==0,(str(log),process.returncode)
  finally:
   g.cleanup(process.pid)
   try:process.wait(timeout=3)
   except subprocess.TimeoutExpired:process.kill();process.wait(timeout=3)
resource=read(DS/'ADAPTIVE_WORSTCASE_RESOURCE_REFERENCE.json')
cfg=read(RUNTIME/'RECIPE.json');cfg.update(source=str(SOURCE),runtime=str(RUNTIME),dataset=A['mount'],checkpoint=A['checkpoint'],source_manifest=str(RUNTIME/'SOURCE_MANIFEST.json'),input=str(inp),output=str(WORK/'canonical'),allocation=allocation,training_identity=A['training_identity'],models=[{k:v for k,v in m.items() if k!='original_release'} for m in models],packed_root=str(TMP/'packed_weights'),packed_models_manifest=str(TMP/'PACKED_MODELS.json'),adaptive_resource_reference_peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in resource['records']),adaptive_resource_reference_sha256=sha(DS/'ADAPTIVE_WORSTCASE_RESOURCE_REFERENCE.json'))
config=TMP/'canonical_config.json';save(config,cfg)
phase('CANONICAL_START',resident_checkpoints_per_gpu=9)
run([sys.executable,'-B','-u',str(RUNTIME/'process_controller.py'),'--config',str(config)],dict(env,RSNA_COMPONENT_KEY='adaptive'),WORK/'CANONICAL_CONTROLLER.log')
phase('CANONICAL_COMPLETE')
packed=read(TMP/'PACKED_MODELS.json');assert len(packed)==10
pack_qa={m['key']:read(Path(m['packed_weights'])/'HEADER.json') for m in packed}
save(WORK/'MODEL_PACK_QA.json',{k:{a:v for a,v in r.items() if a!='tensors' and a!='identity'} for k,r in pack_qa.items()})
mirror=next(m for m in packed if m['key']=='mirror');raw_cfg=read(RAW_RUNTIME/'RECIPE.json')
raw_cfg.update(source=str(RAW_SOURCE),dataset=M['mount'],checkpoint=M['checkpoint'],source_manifest=str(RAW_RUNTIME/'SOURCE_MANIFEST.json'),input=str(inp),output=str(WORK/'mirror_run'),allocation=allocation,training_identity=M['training_identity'],packed_weights=mirror['packed_weights'],packed_checkpoint_sha256=M['checkpoint_sha256'])
raw_config=TMP/'raw_config.json';save(raw_config,raw_cfg)
phase('MIRROR_RAW_START')
run([sys.executable,'-B','-u',str(RAW_RUNTIME/'process_controller.py'),'--config',str(raw_config)],dict(env,RSNA_COMPONENT_KEY='mirror'),WORK/'MIRROR_CONTROLLER.log')
phase('MIRROR_RAW_COMPLETE')
sys.path[:0]=[str(DEPS),str(DS)]
import numpy as np,pandas as pd
from joint36_ranked import rank_frame,TARGETS
sample=pd.read_csv(COMP/'sample_submission.csv',dtype={'StudyInstanceUID':str});model_frames={};parity={};model_dir=WORK/'models';model_dir.mkdir()
for item in models:
 key=item['key'];path=WORK/'mirror_run/core_probabilities.csv' if key=='mirror' else WORK/'canonical'/(key+'.csv')
 frame=pd.read_csv(path,dtype={'StudyInstanceUID':str})
 if key=='joint':frame=rank_frame(frame)
 assert frame.StudyInstanceUID.tolist()==uids and list(frame.columns)==list(sample.columns)
 frame.to_csv(model_dir/(key+'.csv'),index=False)
 parsed=pd.read_csv(model_dir/(key+'.csv'),dtype={'StudyInstanceUID':str},float_precision='round_trip');model_frames[key]=parsed
 if SAMPLE_RUN:
  reference=pd.read_csv(DS/item['reference_file'],dtype={'StudyInstanceUID':str},float_precision='round_trip')
  delta=np.abs(parsed[TARGETS].to_numpy(float)-reference[TARGETS].to_numpy(float));assert parsed.equals(reference),(key,float(delta.max()))
  parity[key]={'status':'PASS','max_absolute_difference':float(delta.max()),'csv_bytes_equal':sha(model_dir/(key+'.csv'))==item['reference_sha256']}
fold=np.mean(np.stack([model_frames['fold'+str(i)][TARGETS].to_numpy(float) for i in range(5)]),axis=0)
components={'fivefold':fold,**{k:model_frames[k][TARGETS].to_numpy(float) for k in ['adaptive','repair','core','mirror','joint']}}
if SAMPLE_RUN:assert np.array_equal(fold,pd.read_csv(DS/'ref_fivefold.csv',float_precision='round_trip')[TARGETS].to_numpy(float))
final_values=np.mean(np.stack(list(components.values())),axis=0);assert np.isfinite(final_values).all() and ((final_values>=0)&(final_values<=1)).all()
final=pd.DataFrame(final_values,columns=TARGETS);final.insert(0,'StudyInstanceUID',uids);final.to_csv(WORK/'submission.csv',index=False)
component_dir=WORK/'components';component_dir.mkdir()
for key,values in components.items():
 f=pd.DataFrame(values,columns=TARGETS);f.insert(0,'StudyInstanceUID',uids);f.to_csv(component_dir/(key+'.csv'),index=False)
audit=read(WORK/'canonical/PROFILE_AUDIT.json');raw_audit=read(WORK/'mirror_run/PROFILE_AUDIT.json');assert audit['status']==raw_audit['status']=='PASS'
for worker in raw_audit['workers']:assert worker['orientation_mode']==worker['orientation_state']=='raw_acquired' and worker['orientation_policy']=='explicit-raw-acquired-no-canonicalization-v1'
assert all(w['synthetic_forwards']==0 and len(w['model_loads'])==9 and all(m['strict_loads']==m['device_copies']==1 for m in w['model_loads'].values()) for w in audit['workers'])
events=[json.loads(s) for s in (RAW_BANK/'EVENTS.jsonl').read_text().splitlines()];builds=[e for e in events if e['kind']=='source_build'];base=[e for e in events if e['kind']=='canonical_input_build'];forwards=[e for e in events if e['kind']=='resident_forward']
assert len(builds)==len({e['path'] for e in builds}) and all(e['source_reads']==e['pixel_decodes']==1 and not e['decode_error'] for e in builds)
assert len(base)==len({(e['uid'],e['route']) for e in base}) and len(forwards)==9*len(uids)
assert not any(e.get('component')=='mirror' and e['kind']=='canonical_input_build' for e in events)
extras={r['uid']:sum(a['adaptive_extension']['windows'] for a in r['adaptive_extra']) for w in audit['workers'] for r in w['records']}
if SAMPLE_RUN:assert len(builds)==557 and [extras[u] for u in uids]==[24,12,0]
read_qa={'status':'PASS','unique_original_dicoms':len(builds),'original_source_reads':sum(e.get('source_reads',0) for e in events),'raw_pixel_arrays':sum(e.get('pixel_decodes',0) for e in events),'canonical_routes_prepared':len(base),'canonical_tensor_reuses':sum(e['routes'] for e in forwards if e['shared_canonical_input']),'dependency_installs':1,'checkpoint_source_reads':sum(r['source_reads'] for r in pack_qa.values()),'checkpoint_deserializations':sum(r['checkpoint_deserializations'] for r in pack_qa.values()),'canonical_GPU_processes':2,'raw_GPU_processes':2,'canonical_model_strict_loads_per_GPU':9,'synthetic_forwards':0,'adaptive_extra_windows':extras,'raw_bank_bytes':sum(p.stat().st_size for p in RAW_BANK.glob('*.zpk'))}
save(WORK/'READ_REUSE_QA.json',read_qa);shutil.copyfile(RAW_BANK/'EVENTS.jsonl',WORK/'SHARED_BANK_EVENTS.jsonl')
save(WORK/'COMPONENT_PARITY_QA.json',{'status':'PASS' if SAMPLE_RUN else 'NOT_PUBLIC_SAMPLE','models':parity,'fivefold':'PASS' if SAMPLE_RUN else 'NOT_PUBLIC_SAMPLE','outer_weights':{k:'1/6' for k in components},'effective_fold_weights':'1/30','mirror':'raw_acquired_no_v83','adaptive_extra':extras})
timing={'status':'PASS','entry_utc':ENTRY,'completed_utc':datetime.now(timezone.utc).isoformat(),'studies':len(uids),'notebook_entry_to_submission_seconds':time.perf_counter()-START,'canonical_loop_seconds':audit['loop_seconds'],'canonical_controller_seconds':audit['controller_seconds'],'raw_loop_seconds':raw_audit['loop_seconds'],'scope':'Actual Kaggle sample entry throughCSV andQA; includes bootstrap/weightpack/copy/processstartup; excludes provider allocation/scoring.'}
save(WORK/'RUNTIME_TIMING.json',timing)
save(WORK/'INFERENCE_RECEIPT.json',{'status':'PASS','models_count':10,'components_count':6,'sample_parity':'PASS' if SAMPLE_RUN else 'NOT_PUBLIC_SAMPLE','official_UIDs':uids,'release_sha256':sha(DS/'RELEASE_PACKET.json'),'submission_sha256':sha(WORK/'submission.csv'),'timing_sha256':sha(WORK/'RUNTIME_TIMING.json'),'read_reuse_qa_sha256':sha(WORK/'READ_REUSE_QA.json'),'parity_qa_sha256':sha(WORK/'COMPONENT_PARITY_QA.json')})
for p in RAW_BANK.glob('*.zpk'):p.unlink()
signal.alarm(0);phase('RESIDENT_COMPLETE',timing=timing,reads=read_qa)
