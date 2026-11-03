from remote import run,ROOT
import json
s=run("""from pathlib import Path
import json,time,subprocess,shutil
r=Path('/workspace/knee_v4_cache_20261021/fastmri_v1')
out={'server_unix':time.time(),'state':json.loads((r/'STATE.json').read_text()) if (r/'STATE.json').exists() else None,'volumes':len(list((r/'source_rss').glob('file*/receipt.json'))),'free_bytes':shutil.disk_usage(r).free,'logs':{p.name:p.read_text()[-2400:] for p in [r/'canary.log',r/'source.log'] if p.exists()},'admission':json.loads((r.parent/'shared/admission.json').read_text())}
if out['state'] and out['state'].get('PID'):
 p=Path('/proc/%s'%out['state']['PID']);out['process_live']=p.exists();out['actual_ctime']=p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19] if p.exists() else None
print(json.dumps(out))
""")
d=json.loads(s);(ROOT/'STATE_READBACK.json').write_text(json.dumps(d,indent=2)+'\n');s=d.get('state') or {}; a=d['admission']; print(json.dumps({'server_unix':d['server_unix'],'phase':s.get('phase'),'committed':d['volumes'],'slices':s.get('slices_committed'),'split':s.get('split'),'bytes_transferred_this_run':s.get('bytes_transferred_this_run'),'reused_source_rss_bytes':s.get('reused_source_rss_bytes'),'started_unix':s.get('started_unix'),'updated_unix':s.get('updated_unix'),'PID':s.get('PID'),'process_live':d.get('process_live'),'network_reconnects':s.get('network_reconnects'),'error':s.get('error'),'free_bytes':d['free_bytes'],'own_admission':a['tasks'].get('fastmri_v1'),'peer_cpu_threads':a['tasks'].get('oai_v4',{}).get('cpu_threads'),'last_source_log':d['logs'].get('source.log','').splitlines()[-2:]},indent=2))
