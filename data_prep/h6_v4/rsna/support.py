"""Portable cache-only utilities. No credentials, model training or hidden paths."""
from pathlib import Path
import os,json,time,hashlib,ast,sys,importlib.util,fcntl,contextlib
sys.dont_write_bytecode=True
PACKAGE=Path(__file__).resolve().parent
RECIPE='rsna-v4-v83-lps-chunks-v1'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for part in iter(lambda:f.read(8<<20),b''):h.update(part)
 return h.hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def dump(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.writing')
 with tmp.open('w')as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path);fd=os.open(path.parent,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def stat_id(path):
 s=Path(path).stat();return [s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_ino]
def key(sample):return hashlib.sha256(sample.encode()).hexdigest()[:32]
@contextlib.contextmanager
def lock(path):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a+')as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise RuntimeError('OWN_PROCESS_ALREADY_RUNNING:'+str(path))
  yield

def exact_function(path,name,scope):
 tree=ast.parse(Path(path).read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name==name)
 exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),scope);return scope[name]
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def verify_package():
 m=json.loads((PACKAGE/'HANDOFF_MANIFEST.json').read_text())
 for row in m['files']:
  rel=Path(row['path']);p=PACKAGE/rel
  if rel.is_absolute()or'..'in rel.parts or p.is_symlink()or not p.is_file():raise ValueError('UNSAFE_OR_MISSING_COMPONENT:'+str(rel))
  if p.stat().st_size!=row['bytes']or sha(p)!=row['sha256']:raise ValueError('COMPONENT_SHA_MISMATCH:'+str(rel))
 return sha(PACKAGE/'HANDOFF_MANIFEST.json')
