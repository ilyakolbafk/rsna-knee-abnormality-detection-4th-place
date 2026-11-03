import numpy as np

def fm_auc(y, s):
    """ROC AUC (Mann-Whitney with average ranks for ties); nan when y has no positive or no negative"""
    y = np.asarray(y, bool)
    s = np.asarray(s, np.float64)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    avg = np.cumsum(cnt) - (cnt - 1) / 2.0
    r = avg[inv]
    return float((r[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))
