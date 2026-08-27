import pandas as pd
from routing.evaluation.confidence_calibration import CalibrationSafetyTarget,build_calibration_artifact,calibration_report,cumulative_threshold_report


def test_calibration_reports_safety_metrics_by_bucket():
    df=pd.DataFrame({"true_tier":[3,3,2,1],"predicted_tier":[1,2,1,1],"confidence":[.95,.85,.75,.45]})
    report=calibration_report(df).set_index("confidence_bucket")
    assert report.loc["90%+","T3_to_T1"]==1
    assert report.loc["80-90%","T3_to_T2"]==1
    assert report.loc["70-80%","T2_to_T1"]==1

def test_calibration_artifact_uses_empirical_safety_target():
    df=pd.DataFrame({"true_tier":[1,2,3,3],"predicted_tier":[1,2,3,2],"confidence":[.95,.92,.91,.85]})
    report=cumulative_threshold_report(df)
    artifact=build_calibration_artifact(df,CalibrationSafetyTarget(max_under_routing=.01,max_severe_under_routing=0,min_examples=3))
    assert report.loc[report.threshold==.9,"count"].iloc[0]==3
    assert artifact["recommended_high_confidence"]==.9
