"""Pre-promotion consumer-only revision. Pixel, source, geometry and metadata bytes stay frozen."""
import sys,json,ast,hashlib,inspect,textwrap,time,tempfile,types
from pathlib import Path
from anatomy_job import ROOT,OwnedJob,write
from fm_training_reader import sha,need,verify_core
import fm_aux_chunk_adapter as a
import fm_aux_reference as original
import fm_loader_reference as ref
from build_anatomical_chunks import atomic_bytes
OUT=ROOT/'training_pixel_store'
def main(job):
 need(not (OUT/'PROMOTION_RECEIPT.json').exists(),'REFUSE_REVISION_AFTER_PROMOTION');need(not (OUT/'provenance/LOADER_TARGET_QA.json').exists(),'READER_QA_MUST_BE_RERUN_AFTER_REVISION');old=json.loads((OUT/'CORE_BINDING.json').read_text());oldsha=sha(OUT/'CORE_BINDING.json');proof=[]
 for name,expected in a.EXPECTED_REFERENCE_AST.items():
  obj=getattr(original,name,getattr(ref,name,None));raw=hashlib.sha256(ast.dump(ast.parse(textwrap.dedent(inspect.getsource(obj))).body[0],include_attributes=False).encode()).hexdigest();normalized=a._object_AST_sha(obj);need(normalized==expected,'CROSS_PYTHON_AST_STILL_DIFFERS:'+name);proof.append({'name':name,'raw_ast_sha256':raw,'canonical_ast_sha256':normalized,'expected_frozen_sha256':expected,'PASS':True})
 with tempfile.TemporaryDirectory(prefix='generic-AST-',dir=ROOT/'qa') as td:
  p=Path(td)/'identity.py';p.write_text('def identity[T](x: T) -> T:\n return x\n');mod=types.ModuleType('fm_generic_type_guard_QA');mod.__file__=str(p);sys.modules[mod.__name__]=mod;exec(compile(p.read_text(),str(p),'exec'),mod.__dict__)
  try:a._object_AST_sha(mod.identity)
  except RuntimeError as e:need(str(e)=='FM_GENERIC_TYPE_PARAMETERS_NOT_IN_FROZEN_REFERENCE','WRONG_GENERIC_TYPE_GUARD')
  else:raise RuntimeError('MEANINGFUL_TYPE_PARAMETERS_WERE_DROPPED')
 report={'state':'PASS','python_version':sys.version,'checks':proof,'nonempty_type_parameters_rejected':True,'change':'Only absent-versus-empty Python3.12 type_params metadata normalized; all other AST nodes retained','code_sha256':sha(ROOT/'fm_aux_chunk_adapter.py')};write(ROOT/'qa/CROSS_PYTHON_AST_PARITY.json',report)
 for n,h in old['files_sha256'].items():job.check();need(sha(OUT/n)==h,'OLD_FROZEN_CORE_CHANGED:'+n)
 changed=['fm_aux_chunk_adapter.py','fm_training_reader.py','qa_fm_consumer.py'];job.register(True);job.check();atomic_bytes(OUT/'provenance/CORE_BINDING.before_AST_schema_fix.json',(OUT/'CORE_BINDING.json').read_bytes())
 files=dict(old['files_sha256'])
 for n in changed:atomic_bytes(OUT/'code'/n,(ROOT/n).read_bytes());files['code/'+n]=sha(ROOT/n)
 new=dict(old,revision=2,previous_core_binding_sha256=oldsha,reason='Cross-Python empty type_params normalization; whole-volume negative QA; reviewed-negative-status guard. No pixels/geometry/ontology/sampler change.',files_sha256=files,created_unix=time.time());write(OUT/'CORE_BINDING.json',new);verify_core(OUT)
 revision={'state':'PASS_UNPROMOTED_CONSUMER_REVISION','old_core_binding_sha256':oldsha,'new_core_binding_sha256':sha(OUT/'CORE_BINDING.json'),'changed_code':{n:sha(ROOT/n) for n in changed},'cross_python_ast_parity_sha256':sha(ROOT/'qa/CROSS_PYTHON_AST_PARITY.json'),'all_other2354_bound_files_unchanged':len(files)-len(changed),'training_started':False,'no_pixels_metadata_source_or_geometry_changed':True,'created_unix':time.time()};write(ROOT/'qa/CONSUMER_CORE_REVISION.v2.json',revision);return revision
if __name__=='__main__':
 with OwnedJob('revise_unpromoted_consumer_cross_Python_AST',cpu=1,ram_gib=1,disk_gib=.03) as job:print(json.dumps(main(job),indent=2),flush=True)
