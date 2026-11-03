"""Build only three new fullfit-only canonical336 samples on CPU; immutable source bank untouched."""
from pathlib import Path
import ast,hashlib,json,os,sys,time
import numpy as np,pandas as pd,torch
Q=Path(__file__).resolve().parent;W=Q.parents[2];RUNTIME=Q/'test3_orientation/runtime';OUT=Q/'test3_cache'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,sort_keys=True,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x))+'\n')
assert not torch.cuda.is_available(),'StrictCPU builder';torch.set_num_threads(4)
policy=json.loads((W/'configs/fullfit_public_test3_vm_policy.json').read_text());assert sha(W/policy['source_result']['path'])==policy['source_result']['sha256']
for row in policy['artifacts']:assert sha(W/row['path'])==row['sha256']
manifest=json.loads((W/'artifacts/production/public_test3_vm_v1/INPUT_MANIFEST.json').read_text())
for row in manifest['csv_sources']+manifest['files']:
 p=W/row['path'];assert p.is_file() and sha(p)==row['sha256']
 if 'bytes' in row:assert p.stat().st_size==row['bytes']
source=W/'.codex/runs/rsna-v83-big-train-20260915/kaggle_cache_preparation/kernel_global_pixel_shards/build_global_pixel_cache.py';assert sha(source)=='935625ae385df67cba32d11aca08a3cce7f3ec27d42313be79edc0236e9c483a'
tree=ast.parse(source.read_text());names={'sha256_file','sha256_array','array_hashes','load_compact','dump_compact'};nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];assert len(nodes)==5
namespace={'Path':Path,'np':np,'hashlib':hashlib};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
sys.path[:0]=[str(RUNTIME),str(RUNTIME/'rsna_knee')]
from resolver import LateralityResolver
from raw_inputs import DecodedStudy
assert not OUT.exists(),'Isolated cache namespace only, no overwrite';OUT.mkdir();(OUT/'fullfit_only').mkdir();(OUT/'audits').mkdir()
resolver=LateralityResolver(root=RUNTIME,device='cpu');series=pd.read_csv(W/'data/sample_test/test_series.csv',dtype=str).fillna('');records=[];slots=[]
for case in policy['cases']:
 uid=case['study_uid'];rows=series.loc[series.StudyInstanceUID.eq(uid)].to_dict('records');assert len(rows)>0
 study=DecodedStudy(uid,rows,W/'data/sample_test/test_series');decision=study.side_inputs(resolver);assert not decision['bilateral'] and len(decision['routes'])==1
 inputs,audit=study.h1_inputs(decision,decision['routes'][0]);assert not audit['source_errors'];pix,mask,meta=inputs
 pixels_f=pix.detach().cpu().numpy();valid=np.rint(pixels_f*255).clip(0,255).astype(np.uint8);m=mask.detach().cpu().numpy().astype(bool);mt=meta.detach().cpu().numpy().astype(np.float32)
 assert valid.shape[1:]==(3,336,336) and len(valid)<=64 and m.all() and mt.shape==(len(valid),6) and np.isfinite(mt).all()
 pixels=np.zeros((64,3,336,336),np.uint8);masks=np.zeros(64,bool);metas=np.zeros((64,6),np.float32);pixels[:len(valid)]=valid;masks[:len(valid)]=True;metas[:len(valid)]=mt
 assert pixels.any();arrays={'pixels':pixels,'mask':masks,'meta':metas};key=hashlib.sha256(uid.encode()).hexdigest()[:24];dest=OUT/'fullfit_only'/f'{key}.npz'
 packed=namespace['dump_compact'](dest,arrays,{'study_uid':uid,'sample_id':uid,'side':decision['side'],'split':'fullfit_only','orientation_policy':'knee-laterality-v8.3','orientation_state':'canonical_L_v8_3'})
 counts=[0]*5;sids=['']*5
 for r in audit['selection']:counts[r['slot']]=r.get('selected_centers',0);sids[r['slot']]=r['series_uid']
 assert sum(counts)==int(masks.sum())
 record={'namespace':'fullfit_only','StudyInstanceUID':uid,'SampleID':uid,'side':decision['side'],'cache_file':str(dest.relative_to(OUT)),'tokens':int(masks.sum()),'padded_tokens':int((~masks).sum()),'bytes':packed['bytes'],'sha256':packed['sha256'],'decoded_source_frames':study.decoded,'source_errors':0,'orientation_policy':'knee-laterality-v8.3','orientation_state':'canonical_L_v8_3','array_sha256':packed['array_sha256']}
 records.append(record);slots.append({'SampleID':uid,'StudyInstanceUID':uid,'side':decision['side'],'tokens':int(masks.sum()),'decoded_source_frames':study.decoded,'source_errors':0,'bilateral':False,'status':'mapped','reason':'actualsource_selection_counts','**unused':'',**{f'count_{i}':counts[i] for i in range(5)},**{f'sid_{i}':sids[i] for i in range(5)}})
 save(OUT/'audits'/f"{case['case_id']}.json",{'decision':decision,'audit':audit,'record':record,'builder_array_roundtrip':packed,'canonical_adapter_calls':1})
 print(json.dumps({'case':case['case_id'],'status':'PASS','tokens':record['tokens'],'decoded':study.decoded,'side':decision['side'],'bytes':packed['bytes']}),flush=True)
pd.DataFrame([{k:v for k,v in r.items() if k!='array_sha256'} for r in records]).to_csv(OUT/'TEST3_MANIFEST.csv',index=False);pd.DataFrame(slots).drop(columns=['**unused']).to_csv(OUT/'TEST3_TOKEN_SLOT_MAP.csv',index=False)
receipt={'status':'PASS_TEST3_CANONICAL336_REAL_COMPACT_CACHE','samples':3,'source_pixels':336,'crop_mm':140,'metadata':6,'max_tokens':64,'orientation_policy':'knee-laterality-v8.3','orientation_state':'canonical_L_v8_3','canonical_adapter_calls_per_sample':1,'strict_CPU':True,'policy_sha256':sha(W/'configs/fullfit_public_test3_vm_policy.json'),'source_vm_sha256':policy['source_result']['sha256'],'builder_source_sha256':sha(source),'isolated_builder_sha256':sha(__file__),'source_inputs_manifest_sha256':sha(W/'artifacts/production/public_test3_vm_v1/INPUT_MANIFEST.json'),'records':records,'token_map_sha256':sha(OUT/'TEST3_TOKEN_SLOT_MAP.csv'),'cache_manifest_sha256':sha(OUT/'TEST3_MANIFEST.csv'),'utc_unix':time.time()};save(OUT/'BUILD_RECEIPT.json',receipt);print(receipt['status'])
