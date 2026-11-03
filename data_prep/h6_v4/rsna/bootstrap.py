"""Create own Python3.11 environment; no system/site-packages modification."""
from pathlib import Path
import argparse,sys,venv,subprocess
p=Path(__file__).resolve().parent
if sys.version_info[:2]!=(3,11):raise SystemExit('Use Python3.11: original MONAI0.6 runtime is tested on3.11.')
a=argparse.ArgumentParser();a.add_argument('--venv',default=str(p.parent/'rsna_v4_venv'));args=a.parse_args();d=Path(args.venv).resolve()
if not(d/'pyvenv.cfg').exists():venv.EnvBuilder(with_pip=True).create(d)
exe=d/('Scripts/python.exe'if sys.platform=='win32'else'bin/python')
subprocess.run([str(exe),'-m','pip','install','-r',str(p/'requirements.txt')],check=True)
subprocess.run([str(exe),str(p/'test_runtime.py')],check=True)
print('READY_PYTHON='+str(exe))
