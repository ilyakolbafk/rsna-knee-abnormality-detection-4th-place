"""Deterministic gates for the Native64 regularization cycle; no provider writes."""
from pathlib import Path
import json
import math
import pandas as pd
from .constants import TARGET_COLUMNS
from .epoch_stop import plateau_stop
from .pseudo_oof import evaluate_oof_frame
from .training_state import file_sha256


def verified_completion(folder, record, expected_ids):
    folder=Path(folder)
    marker=json.loads((folder/'COMPLETED.json').read_text())
    summary=json.loads((folder/'summary.json').read_text())
    history=json.loads((folder/'history.json').read_text())
    planned=record['max_epochs']
    policy=record.get('stop_policy')
    actual=len(history)
    stopped=plateau_stop(history,summary['best_epoch'],policy,planned)
    if actual != planned and not (actual==12 and stopped and record['fold']>=0):
        raise ValueError('Unapproved incomplete epoch boundary')
    if (marker['completed_epochs']!=actual or summary['completed_epochs']!=actual or
            summary['planned_epochs']!=planned or summary['status']!='complete' or
            marker['fold']!=record['fold'] or marker['training_identity_sha256']!=record['identity_sha256']):
        raise ValueError('Completion identity/boundary mismatch')
    if [x['epoch'] for x in history]!=list(range(actual)):
        raise ValueError('Noncontiguous history')
    required={'best.pt','history.json','summary.json','training_identity.json','resolved_config.yaml'}
    if record['fold']>=0:required|={'oof_predictions.csv','epoch_011_predictions.csv','weights_epoch_011.pt'}
    else:required|={'training_studies.csv'}
    for epoch in record['save_weight_epochs']:
        if epoch<=actual:required.add(f'weights_epoch_{epoch-1:03d}.pt')
    if not required<=set(marker['files']):raise ValueError('Missing required retained artifact')
    for name,r in marker['files'].items():
        p=folder/name
        if Path(name).name!=name or p.is_symlink() or not p.is_file() or p.stat().st_size!=r['bytes'] or file_sha256(p)!=r['sha256']:
            raise ValueError('Completion payload mismatch: '+name)
    roster=pd.read_csv(folder/('oof_predictions.csv' if record['fold']>=0 else 'training_studies.csv'),dtype={'StudyInstanceUID':str})
    if not roster.StudyInstanceUID.is_unique or set(roster.StudyInstanceUID)!=set(expected_ids):
        raise ValueError('Roster mismatch')
    return marker


def read_paired(left_path, right_path):
    cols=[f'pred__{t}' for t in TARGET_COLUMNS]
    a=pd.read_csv(left_path).sort_values('StudyInstanceUID').reset_index(drop=True)
    b=pd.read_csv(right_path).sort_values('StudyInstanceUID').reset_index(drop=True)
    pd.testing.assert_frame_equal(a.drop(columns=cols),b.drop(columns=cols))
    scores=[evaluate_oof_frame(f)['pseudo']['macro_auc'] for f in (a,b)]
    if any(x is None or not math.isfinite(x) for x in scores):raise ValueError('Nonfinite gate score')
    return dict(baseline=scores[0],candidate=scores[1],delta=scores[1]-scores[0])


def screen_gate(delta, seed_delta):
    if not all(math.isfinite(x) for x in (delta,seed_delta)):raise ValueError('Nonfinite screen gate')
    # One seed difference is only a practical guard, not an estimated variance.
    if delta>=.005 and delta>2*abs(seed_delta):return 'CONFIRM_FOLD0'
    if delta>=.002:return 'INCONCLUSIVE_NO_AUTOMATIC_EXPANSION'
    return 'STOP_NO_GAIN'


def confirmation_gate(delta3,delta0,seed_delta):
    if not all(math.isfinite(x) for x in (delta3,delta0,seed_delta)):raise ValueError('Nonfinite confirmation gate')
    return delta3>0 and delta0>0 and (delta3+delta0)/2>=max(.003,2*abs(seed_delta))


def fits_remaining(now, deadline, seconds, reserve=900):
    if not all(math.isfinite(x) for x in (now,deadline,seconds,reserve)) or seconds<0 or reserve<600:
        raise ValueError('Invalid budget forecast')
    return now+seconds+reserve<=deadline
