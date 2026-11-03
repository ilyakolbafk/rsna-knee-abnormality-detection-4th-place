"""Compact read-only state, with identity-bound process liveness."""
from remote import run,ROOT
import json,time
out=json.loads(run('''import json,time,shutil
from pathlib import Path
r=Path('/workspace/knee_v4_cache_20261021/fastmri_v1')
def read(n):
 p=r/n;return json.loads(p.read_text()) if p.exists() else None
def live(v):
 if not v:return False
 p=Path('/proc/%s'%v.get('PID'))
 try:return p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]==str(v.get('ctime')) and '\\nState:\\tZ' not in p.joinpath('status').read_text()
 except FileNotFoundError:return False
s=read('STATE.json');f=read('FINISHER_LAUNCH.json');ad=json.loads((r.parent/'shared/admission.json').read_text());o=ad['tasks'].get('fastmri_v1',{})
v={'server_unix':time.time(),'state':s,'source_live':live(read('LAUNCH_SOURCE_CORPUS.json')),'finisher_live':live(f),'finisher_launch':f,'source_volumes':len(list((r/'source_rss').glob('file*/receipt.json'))),'candidate_volumes':len(list((r/'candidate_pixel_store/volume_receipts').glob('file*.json'))),'archive_complete':{sp:(r/'receipts'/('ARCHIVE_COMPLETE_'+sp+'.json')).exists() for sp in ['train','val']},'failure':read('FINISHER_FAILURE.json'),'final':read('FINAL_STATUS.json'),'progress':read('PROGRESS.json'),'free_bytes':shutil.disk_usage(r).free,'own_admission':o,'peer_cpu_threads':sum(v.get('cpu_threads',0) for k,v in ad['tasks'].items() if k!='fastmri_v1')};print(json.dumps(v))
'''))
(ROOT/'MONITOR_READBACK.json').write_text(json.dumps(out,indent=2)+'\n')
s=out['state'] or {};p=out.get('progress') or {}
result={'unix':out['server_unix'],'phase':s.get('phase'),'source_volumes':out['source_volumes'],'physical_slices':s.get('slices_committed'),'candidate_volumes':out['candidate_volumes'],'bytes_transferred_this_run':s.get('bytes_transferred_this_run'),'source_live':out['source_live'],'finisher_live':out['finisher_live'],'archive_complete':out['archive_complete'],'error':s.get('error_type'),'finisher_failure':out['failure'],'historical_average_source_ETA_seconds':round(p['ETA_source_seconds']) if p.get('ETA_source_seconds') else None,'admission_pressure':out['own_admission'].get('admission_pressure'),'final':out['final']['state'] if out['final'] else None}
result['source_ETA_seconds']=None;result['source_ETA_basis']='Recent 2-5min source bytes only; conditional on stable throughput, excludes verification/build'
if out['source_live'] and s.get('stream_total') and s.get('stream_bytes') is not None:
 remaining=s['stream_total']-s['stream_bytes']+(78016052604 if s.get('split')=='val' else 0);result['source_remaining_bytes']=remaining
 hp=ROOT/'monitor_history.jsonl';history=[json.loads(x) for x in hp.read_text().splitlines()] if hp.exists() else []
 points=[x for x in history if x.get('source_live') and x.get('bytes_transferred_this_run') is not None and 120<=out['server_unix']-x['unix']<=300 and x['bytes_transferred_this_run']<=s['bytes_transferred_this_run']]
 if points:
  prev=points[0];span=out['server_unix']-prev['unix'];rate=(s['bytes_transferred_this_run']-prev['bytes_transferred_this_run'])/span
  result.update(recent_rate_window_seconds=round(span),recent_transfer_Bps=round(rate),source_ETA_seconds=round(remaining/rate) if rate else None)
with (ROOT/'monitor_history.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
print(json.dumps(result))
