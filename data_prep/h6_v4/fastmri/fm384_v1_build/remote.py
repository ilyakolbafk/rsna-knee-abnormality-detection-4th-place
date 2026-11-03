import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
ARGV=json.loads((ROOT/'ssh_connection.json').read_text())['argv']
def run(code,timeout=60):
 r=subprocess.run(ARGV+['python3 -'],input=code,text=True,capture_output=True,timeout=timeout)
 if r.returncode: raise RuntimeError(r.stderr[-2000:])
 return r.stdout
if __name__=='__main__':
 print(run(Path(sys.argv[1]).read_text(),int(sys.argv[2]) if len(sys.argv)>2 else 60))
