from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np,pandas as pd

BINS=[0,.5,.6,.7,.8,.9,1.0000001]
LABELS=["<50%","50-60%","60-70%","70-80%","80-90%","90%+"]

@dataclass(frozen=True)
class CalibrationSafetyTarget:
    max_under_routing:float=.10
    max_severe_under_routing:float=.02
    min_examples:int=50


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
                     "T3_to_T1":int(((y==3)&(p==1)).sum()),"T3_to_T2":int(((y==3)&(p==2)).sum()),
                     "T2_to_T1":int(((y==2)&(p==1)).sum())})
    return pd.DataFrame(rows)

def cumulative_threshold_report(predictions:pd.DataFrame,thresholds=(.5,.6,.7,.8,.9)) -> pd.DataFrame:
    required={"true_tier","predicted_tier","confidence"}
    if not required<=set(predictions):raise ValueError(f"Missing columns: {sorted(required-set(predictions))}")
    rows=[]
    for threshold in thresholds:
        part=predictions[predictions.confidence>=threshold];y=part.true_tier.to_numpy();p=part.predicted_tier.to_numpy();delta=p-y
        rows.append({"threshold":threshold,"count":len(part),"coverage":len(part)/len(predictions),
                     "exact_accuracy":float((p==y).mean()) if len(part) else None,
                     "mae":float(np.abs(delta).mean()) if len(part) else None,
                     "under_routing":float((delta<0).mean()) if len(part) else None,
                     "severe_under_routing":float((delta<=-2).mean()) if len(part) else None,
                     "T3_to_T1":int(((y==3)&(p==1)).sum()),"T3_to_T2":int(((y==3)&(p==2)).sum()),
                     "T2_to_T1":int(((y==2)&(p==1)).sum())})
    return pd.DataFrame(rows)

def recommend_high_confidence_threshold(report:pd.DataFrame,target:CalibrationSafetyTarget=CalibrationSafetyTarget())->float|None:
    eligible=report[(report["count"]>=target.min_examples)&(report["under_routing"]<=target.max_under_routing)&(report["severe_under_routing"]<=target.max_severe_under_routing)]
    return None if eligible.empty else float(eligible.sort_values("threshold").iloc[0].threshold)

def build_calibration_artifact(predictions:pd.DataFrame,target:CalibrationSafetyTarget=CalibrationSafetyTarget())->dict:
    buckets=calibration_report(predictions);thresholds=cumulative_threshold_report(predictions)
    return {"version":"tier_router_v2_calibration_v1","tier_count":3,
            "safety_target":target.__dict__,"recommended_high_confidence":recommend_high_confidence_threshold(thresholds,target),
            "confidence_buckets":buckets.where(pd.notna(buckets),None).to_dict("records"),
            "cumulative_thresholds":thresholds.where(pd.notna(thresholds),None).to_dict("records")}

def save_calibration_artifact(predictions:pd.DataFrame,path:Path,target:CalibrationSafetyTarget=CalibrationSafetyTarget())->dict:
    artifact=build_calibration_artifact(predictions,target);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(artifact,indent=2)+"\n");return artifact
