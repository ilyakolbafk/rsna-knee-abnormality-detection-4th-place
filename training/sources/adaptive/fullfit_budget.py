"""Expanded same-rental80/78/79 training guard; frozen old guards are never reused here."""
from pathlib import Path
import json,os,shutil,time
TASK=Path(__file__).resolve().parent.parent
CFG=json.loads((TASK/'QUEUE_CONFIG.json').read_text())
def budget_guard():
    b=CFG['budget'];now=time.time()
    assert os.uname().nodename==b['binding']['hostname']
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==b['binding']['boot_id']
    upper=b['baseline_upper_usd']+(now-b['baseline_unix'])/3600*b['hourly_usd']
    assert upper<b['stop_usd'] and now<b['hard_stop_unix']-120
    assert shutil.disk_usage(TASK).free>=b['disk_reserve_bytes']
    assert not any((TASK/n).exists() for n in ['STOP_INTENT.json','STOP_ACCEPTED.json','CONTROLLER_FAILURE.json','HOST_FAILURE.json'])
    for kind in ['BUDGET','FALLBACK']:
        g=json.loads((TASK/(kind+'_GUARD_STATE.json')).read_text())
        assert g['status']=='WATCHING' and g.get('reason') is None and 0<=now-g['utc_unix']<60
        p=g['identity'];raw=Path('/proc/'+str(p['pid'])+'/stat').read_text().split(') ',1)[1].split()
        assert raw[0] not in ['Z','X','T'] and raw[2]==str(p['pgid']) and raw[19]==p['start_ticks']
    return {**b,'budget_upper_usd':upper,'upper_spend_usd':upper}
