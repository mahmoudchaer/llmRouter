import pandas as pd
from routing.evaluation.confidence_calibration import calibration_report


def test_calibration_reports_safety_metrics_by_bucket():
    df=pd.DataFrame({"true_tier":[4,4,3,1],"predicted_tier":[1,2,1,1],"confidence":[.95,.85,.75,.45]})
    report=calibration_report(df).set_index("confidence_bucket")
    assert report.loc["90%+","T4_to_T1_T2"]==1
    assert report.loc["80-90%","T4_to_T1_T2"]==1
    assert report.loc["70-80%","T3_to_T1"]==1

