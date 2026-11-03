from remote import run
print(run("""import subprocess,sys,os
from pathlib import Path
p=Path('/workspace/knee_v4_cache_20261021/fastmri_v1')
assert (p/'OWNER.json').exists()
x=subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(p/'venv')],capture_output=True,text=True);assert x.returncode==0,x.stderr[-1000:]
x=subprocess.run([str(p/'venv/bin/python'),'-m','pip','install','--disable-pip-version-check','h5py==3.14.0'],capture_output=True,text=True)
print('install_return',x.returncode);print(x.stdout[-1500:]);print(x.stderr[-1000:])
""",120))
