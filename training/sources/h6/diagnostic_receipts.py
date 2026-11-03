"""Immutable checkpoint and full-diagnostic receipt bindings, metadata cached."""
from common import read,sha
_digests={}


def bound_sha(path):
    stat=path.stat();key=(str(path),stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
    if key not in _digests:_digests[key]=sha(path)
    return _digests[key]


def expected_binding(root,arm,checkpoint,package):
    cfg=read(root/'CONFIG.json');ref=read(root/'assets/GOLD_DIAGNOSTIC_SOURCE.json')
    return {'package_sha256':package,'checkpoint_sha256':bound_sha(checkpoint),'inputs_sha256':bound_sha(root/'runtime/INPUTS.json'),
        'labels_sha256':ref['sha256'],'arm':arm,'config_sha256':bound_sha(root/'CONFIG.json'),'phys_mm':cfg['phys_mm'],'eval_stride':cfg['eval_stride']}


def receipts_complete(root,folder,binding):
    gp=folder/'GOLD58.json';fp=folder/'FM_HOLDOUT.json'
    for path in [gp,fp]:
        if path.exists() and read(path).get('binding')!=binding:raise ValueError('STALE_DIAGNOSTIC_RECEIPT:'+str(path))
    if not gp.exists() or not fp.exists():return False
    g=read(gp);f=read(fp);expected=set(read(root/'assets/GOLD_EXCLUDED_IDS.json'))
    if len(g.get('ids',[]))!=58 or set(g['ids'])!=expected or len(g.get('predictions',[]))!=58 or any(len(row)!=12 for row in g['predictions']) or len(g.get('targets',[]))!=12 or g.get('scope')!='REUSED_GOLD58_DESCRIPTIVE_NO_SELECTION':raise ValueError('INCOMPLETE_GOLD58_RECEIPT')
    if f.get('windows',0)<=0 or f.get('volumes',0)<=0 or len(f.get('window_auc',{}))!=11 or len(f.get('volume_auc',{}))!=11 or f.get('selection') is not False:raise ValueError('INCOMPLETE_FM_HOLDOUT_RECEIPT')
    return True
