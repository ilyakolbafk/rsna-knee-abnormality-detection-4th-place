"""Unchanged established Gold metrics; no fitting or selection."""
import numpy as np
def auc(y,p):
    positive=y==1;negative=y==0
    if not positive.any() or not negative.any():return None
    return float(((p[positive,None]>p[None,negative]).sum()+.5*(p[positive,None]==p[None,negative]).sum())/(positive.sum()*negative.sum()))

def metrics(y,p,names):
    rows=[]
    for j,name in enumerate(names):
        v=np.clip(p[:,j],1e-7,1-1e-7);yy=y[:,j]
        rows.append({'target':name,'auc':auc(yy,v),'brier':float(((v-yy)**2).mean()),'bce':float(-(yy*np.log(v)+(1-yy)*np.log(1-v)).mean()),'positives':int((yy==1).sum()),'negatives':int((yy==0).sum())})
    values=[r['auc'] for r in rows if r['auc'] is not None]
    return {'targets':rows,'macro_auc':sum(values)/len(values) if values else None}
