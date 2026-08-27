from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from routing.evaluation.confidence_calibration import calibration_report


def main():
    p=argparse.ArgumentParser();p.add_argument("predictions");p.add_argument("--output",required=True);args=p.parse_args()
    frame=pd.read_parquet(args.predictions) if args.predictions.endswith(".parquet") else pd.read_csv(args.predictions)
    if "confidence" not in frame and "probabilities" in frame:
        frame["confidence"]=frame.probabilities.map(lambda values:max(values))
    report=calibration_report(frame);Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    report.to_csv(args.output,index=False);print(report.to_string(index=False))
if __name__=="__main__":main()

