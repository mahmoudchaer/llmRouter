from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from routing.evaluation.ensemble_eval import evaluate_tier_predictions
from routing.mock_registry import build_mock_registry
from routing.models.factory import build_combined_llm_classifier
from routing.pipeline.router import LLMOnlyRuntimeRouter
from routing.schemas.request import CustomerPriceCeiling,HardRequirements,RoutingRequest
from routing.selection.model_selector import ModelSelector


def domain_metrics(frame:pd.DataFrame)->dict:
    from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score
    labels=["affective","code","instruction_following","knowledge","logic","math","tool_use"]
    return {"accuracy":float(accuracy_score(frame.true_domain,frame.predicted_domain)),
            "macro_f1":float(f1_score(frame.true_domain,frame.predicted_domain,labels=labels,average="macro",zero_division=0)),
            "per_domain":classification_report(frame.true_domain,frame.predicted_domain,labels=labels,output_dict=True,zero_division=0),
            "confusion_matrix":confusion_matrix(frame.true_domain,frame.predicted_domain,labels=labels).tolist(),"labels":labels}


def evaluate_rows(frame:pd.DataFrame)->dict:
    resolved=frame[frame.true_tier.notna()].copy()
    tier=evaluate_tier_predictions(resolved.true_tier.astype(int),resolved.predicted_tier.astype(int))
    baseline=frame.baseline_ceiling_cost.to_numpy();actual=frame.estimated_cost.to_numpy()
    valid=baseline>0
    return {"n":len(frame),"resolved_tier_n":len(resolved),"domain":domain_metrics(frame),"tier":tier,
            "latency":{"mean_seconds":float(frame.latency_seconds.mean()),"median_seconds":float(frame.latency_seconds.median()),
                       "p95_seconds":float(frame.latency_seconds.quantile(.95)),"throughput_requests_per_second":float(len(frame)/frame.latency_seconds.sum())},
            "selection":{"selected_model_counts":frame.selected_model.value_counts().to_dict(),
                         "capability_shortfall_count":int(frame.capability_shortfall.sum()),
                         "mean_estimated_cost":float(actual.mean()),
                         "mean_savings_fraction_vs_strongest_permitted":float(np.mean(1-actual[valid]/baseline[valid])) if valid.any() else None}}


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--subset",default="training/reports/domain_router_eval/subset.parquet")
    parser.add_argument("--labels",default="data_pipeline/output/three_tier/final_labeled_dataset_3tier.parquet")
    parser.add_argument("--config",default="routing/config/routing.yaml");parser.add_argument("--output-dir",default="routing/reports/llm_only_mvp")
    parser.add_argument("--max-prompts",type=int);parser.add_argument("--fresh",action="store_true");args=parser.parse_args()
    cfg=yaml.safe_load(open(args.config));subset=pd.read_parquet(args.subset);labels=pd.read_parquet(args.labels)[["prompt_id","tier","resolved"]]
    data=subset.merge(labels,on="prompt_id",how="left");data=data.iloc[:args.max_prompts] if args.max_prompts else data
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True);partial=output/"predictions.jsonl"
    if args.fresh and partial.exists():partial.unlink()
    completed={}
    if partial.exists():
        for line in partial.read_text().splitlines():
            if line.strip():row=json.loads(line);completed[str(row["prompt_id"])]=row
    classifier=build_combined_llm_classifier(cfg["mvp_llm_classifier"]);registry=build_mock_registry();selector=ModelSelector();router=LLMOnlyRuntimeRouter(classifier,selector,registry)
    ceiling=CustomerPriceCeiling(1,3)
    for i,row in enumerate(data.itertuples(),1):
        if str(row.prompt_id) in completed:continue
        context_tokens=max(1,len(classifier.tokenizer.encode(str(row.prompt),add_special_tokens=False)))
        request=RoutingRequest(str(row.prompt_id),str(row.prompt),ceiling,HardRequirements(context_tokens=context_tokens),expected_output_tokens=512)
        started=time.perf_counter();decision=router.route(request);latency=time.perf_counter()-started
        baseline=selector.select(registry,request,decision.domain,3)
        result={"prompt_id":str(row.prompt_id),"source_dataset":row.source_dataset,"true_domain":row.domain,
                "true_tier":None if pd.isna(row.tier) else int(row.tier),"predicted_domain":decision.domain,
                "predicted_tier":decision.audit["llm_tier_prediction"],"final_tier":decision.final_tier,
                "selected_model":decision.selected_model,"capability_shortfall":decision.audit["capability_shortfall"],
                "estimated_cost":decision.audit["estimated_request_cost"],"baseline_ceiling_cost":baseline.estimated_cost,
                "latency_seconds":latency,"chunks_classified":decision.audit["llm_chunks_classified"]}
        with partial.open("a") as handle:handle.write(json.dumps(result)+"\n")
        completed[result["prompt_id"]]=result
        if len(completed)%10==0:print(f"{len(completed)}/{len(data)}",flush=True)
    predictions=pd.DataFrame([completed[str(pid)] for pid in data.prompt_id]);predictions.to_parquet(output/"predictions.parquet",index=False)
    metrics=evaluate_rows(predictions);(output/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    (output/"route_config.json").write_text(json.dumps(cfg["mvp_llm_classifier"],indent=2)+"\n");print(json.dumps(metrics,indent=2))


if __name__=="__main__":main()
