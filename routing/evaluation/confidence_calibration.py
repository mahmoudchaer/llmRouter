from __future__ import annotations
import numpy as np,pandas as pd

BINS=[0,.5,.6,.7,.8,.9,1.0000001]
LABELS=["<50%","50-60%","60-70%","70-80%","80-90%","90%+"]


def calibration_report(predictions: pd.DataFrame) -> pd.DataFrame:
    required={"true_tier","predicted_tier","confidence"}
    if not required<=set(predictions):raise ValueError(f"Missing columns: {sorted(required-set(predictions))}")
    df=predictions.copy();df["confidence_bucket"]=pd.cut(df.confidence,BINS,labels=LABELS,right=False,include_lowest=True)
    rows=[]
    for bucket,part in df.groupby("confidence_bucket",observed=False):
        y=part.true_tier.to_numpy();p=part.predicted_tier.to_numpy();delta=p-y
        rows.append({"confidence_bucket":str(bucket),"count":len(part),
                     "exact_accuracy":float((p==y).mean()) if len(part) else None,
                     "mae":float(np.abs(delta).mean()) if len(part) else None,
                     "under_routing":float((delta<0).mean()) if len(part) else None,
                     "severe_under_routing":float((delta<=-2).mean()) if len(part) else None,
                     "T4_to_T1_T2":int(((y==4)&(p<=2)).sum()),"T3_to_T1":int(((y==3)&(p==1)).sum())})
    return pd.DataFrame(rows)

