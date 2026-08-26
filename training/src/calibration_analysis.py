from __future__ import annotations

import numpy as np


def confidence_bins(y_true, y_pred, probabilities, n_bins=10):
    confidence=np.max(probabilities,axis=1)
    edges=np.linspace(0,1,n_bins+1)
    rows=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(confidence>=lo)&(confidence<(hi if hi<1 else hi+1e-9))
        rows.append({"lower":float(lo),"upper":float(hi),"n":int(mask.sum()),
                     "mean_confidence":float(confidence[mask].mean()) if mask.any() else None,
                     "accuracy":float((np.asarray(y_true)[mask]==np.asarray(y_pred)[mask]).mean()) if mask.any() else None})
    return rows
