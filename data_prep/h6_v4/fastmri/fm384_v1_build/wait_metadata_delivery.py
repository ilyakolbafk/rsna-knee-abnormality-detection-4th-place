"""One-run follower for this exact server job; no model work or notifications."""
import json,subprocess,time,os,fcntl,hashlib,sys
from pathlib import Path
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,v):(R/n).write_text(json.dumps(v,indent=2)+'\n')
def run(n):
 p=subprocess.run([sys.executable,str(R/n)],cwd=R,text=True,capture_output=True,timeout=180)
 if p.returncode:raise RuntimeError(n+': '+p.stderr[-1500:])
 return p.stdout
def main():
 with (R/'metadata_delivery.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  names=['final_readback.py','collect_artifacts.py','write_result.py'];binding={n:sha(R/n) for n in names};write('LOCAL_DELIVERY_BINDING.json',{'PID':os.getpid(),'files':binding,'started_unix':time.time()})
  failures=0
  while True:
   try:m=json.loads(run('monitor.py'));failures=0
   except Exception as e:
    failures+=1;write('LOCAL_DELIVERY_PROGRESS.json',{'phase':'waiting_control_connection','consecutive_failures':failures,'error_type':type(e).__name__,'unix':time.time()})
    if failures>=3:raise
    time.sleep(40);continue
   write('LOCAL_DELIVERY_PROGRESS.json',{'phase':'waiting_exact_server_job','server':m,'unix':time.time()})
   if m['phase']=='failed' or m['finisher_failure']:raise RuntimeError('SERVER_OWN_JOB_FAILURE_REQUIRES_DIAGNOSIS')
   if m['final'] and not m['source_live'] and not m['finisher_live']:
    assert m['phase']=='source_and_candidate_verified_anatomical_gate_unresolved'
    assert {n:sha(R/n) for n in names}==binding,'DELIVERY_CODE_CHANGED_RESTART_EXPLICITLY'
    for n in names:print(run(n),flush=True)
    write('LOCAL_DELIVERY_FINAL.json',{'state':'PASS_METADATA_DELIVERY_AND_RESULT_RENDER','source_and_pixel_store_location':'server56967269','medical_pixels_transferred':False,'candidate_training_ready':False,'anatomical_ML_gate':'UNRESOLVED','result':str(R/'RESULT.md'),'completed_unix':time.time()});return
   time.sleep(55)
if __name__=='__main__':
 try:main()
 except Exception as e:
  write('LOCAL_DELIVERY_FAILURE.json',{'type':type(e).__name__,'reason':str(e),'unix':time.time(),'PID':os.getpid()});raise
