from remote import run
print(run("""import subprocess,sys,os,json,time
from pathlib import Path
r=Path('/workspace/knee_v4_cache_20261021/fastmri_v1');env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
x=subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(r/'render_venv')],capture_output=True,text=True,env=env);assert x.returncode==0,x.stderr[-500:]
x=subprocess.run([str(r/'render_venv/bin/python'),'-m','pip','install','--disable-pip-version-check','numpy==2.2.6','opencv-python-headless==4.12.0.88','Pillow==11.0.0'],capture_output=True,text=True,env=env);print(x.stdout[-1600:]);print(x.stderr[-1000:]);assert x.returncode==0
print(subprocess.check_output([str(r/'render_venv/bin/python'),'-c','import numpy,cv2,PIL,json;print(json.dumps({"numpy":numpy.__version__,"opencv":cv2.__version__,"Pillow":PIL.__version__}))'],text=True,env=env))
""",180))
