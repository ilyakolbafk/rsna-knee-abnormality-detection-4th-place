"""Stage 3 of FINAL ens: per-target weighted mean of percentile ranks.

This is the arithmetic of cell 3 of the scored notebook (ilyakolb/rsna-final-six-h6-ens-20261022 v2), unchanged:
  rank = per-column percentile rank over all scored studies (pandas, method='average', pct=True);
  a study with no H6 prediction (NaN) takes the Sixway rank in the H6 slot;
  columns with weight {S6: 1.0} keep the raw Sixway probability byte for byte.
Ranks depend on the whole cohort: never blend chunks of the test set independently.

CLI:
  python -m rsna_knee_final.blend --sixway sixway_core.csv --core6 h6_core6_raw.csv --vm h6_vm_raw.csv --out submission.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import PACKAGE_ROOT, TARGETS

READ = dict(dtype={'StudyInstanceUID': str}, float_precision='round_trip')
DEFAULT_WEIGHTS = PACKAGE_ROOT / 'configs' / 'blend_final_ens.json'


def load_weights(path=DEFAULT_WEIGHTS):
    weights = json.loads(Path(path).read_text())['weights']
    assert list(weights) == TARGETS and all(abs(sum(w.values()) - 1) < 1e-12 for w in weights.values())
    return weights


def pct_rank(frame):
    return frame[TARGETS].rank(axis=0, method='average', pct=True).to_numpy(float)


def blend(base, core6, vm, weights=None):
    """base/core6/vm: DataFrames [StudyInstanceUID, *TARGETS]; core6/vm are reordered to base order."""
    weights = weights or load_weights()
    uids = base.StudyInstanceUID.tolist()
    assert list(base.columns) == ['StudyInstanceUID', *TARGETS] and len(uids) == len(set(uids)) and len(uids) >= 2
    members = {}
    for name, frame in (('C6', core6), ('VM', vm)):
        assert sorted(frame.StudyInstanceUID) == sorted(uids), f'{name}: study set differs from Sixway'
        members[name] = frame.set_index('StudyInstanceUID').loc[uids].reset_index()
    rs = pct_rank(base)
    rc = pct_rank(members['C6']); rc = np.where(np.isnan(rc), rs, rc)
    rv = pct_rank(members['VM']); rv = np.where(np.isnan(rv), rs, rv)
    src = {'C6': rc, 'VM': rv, 'S6': rs}
    final = base[TARGETS].to_numpy(float).copy()
    for j, target in enumerate(TARGETS):
        if weights[target] == {'S6': 1.0}:
            continue
        final[:, j] = sum(w * src[s][:, j] for s, w in weights[target].items())
    assert np.isfinite(final).all() and ((final >= 0) & (final <= 1)).all()
    s6_only = [j for j, t in enumerate(TARGETS) if weights[t] == {'S6': 1.0}]
    assert np.array_equal(final[:, s6_only], base[TARGETS].to_numpy(float)[:, s6_only])
    out = pd.DataFrame(final, columns=TARGETS)
    out.insert(0, 'StudyInstanceUID', uids)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sixway', required=True)
    ap.add_argument('--core6', required=True)
    ap.add_argument('--vm', required=True)
    ap.add_argument('--weights', default=str(DEFAULT_WEIGHTS))
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = blend(pd.read_csv(a.sixway, **READ), pd.read_csv(a.core6, **READ), pd.read_csv(a.vm, **READ), load_weights(a.weights))
    out.to_csv(a.out, index=False)
    print(json.dumps({'studies': len(out), 'out': a.out}))


if __name__ == '__main__':
    main()
