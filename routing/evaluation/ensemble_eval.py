from __future__ import annotations
import numpy as np,pandas as pd


def evaluate_tier_predictions(true_tier, predicted_tier) -> dict[str,float|int]:
    y=np.asarray(true_tier);p=np.asarray(predicted_tier);d=p-y
    return {"n":len(y),"exact_accuracy":float((p==y).mean()),"mae":float(np.abs(d).mean()),
            "within_one_tier":float((np.abs(d)<=1).mean()),"under_routing":float((d<0).mean()),
            "over_routing":float((d>0).mean()),"severe_under_routing":float((d<=-2).mean()),
            "T3_to_T1":int(((y==3)&(p==1)).sum()),"T3_to_T2":int(((y==3)&(p==2)).sum()),
            "T2_to_T1":int(((y==2)&(p==1)).sum())}


def compare_ensemble_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for column in ["tier_router_tier","llm_tier","hybrid_tier"]:
        rows.append({"method":column,**evaluate_tier_predictions(frame.true_tier,frame[column])})
    return pd.DataFrame(rows)
