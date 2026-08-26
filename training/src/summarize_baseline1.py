from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd

METRICS=["exact_accuracy","macro_f1","mean_absolute_tier_error","within_one_tier_accuracy",
         "under_routing_rate","over_routing_rate","severe_under_routing_rate","log_loss","multiclass_brier"]

def run():
    root=Path(__file__).resolve().parent.parent
    results=json.load(open(root/"reports"/"baseline1_all_results.json"))
    rows=[]; domain_rows=[]; danger=[]
    for r in results:
        d=r["domain_test"]
        domain_rows.append({"track":r["track"],"seed":r["seed"],"accuracy":d["accuracy"],
                            "macro_f1":d["classification_report"]["macro avg"]["f1-score"],"log_loss":d["log_loss"]})
        for variant,v in r["tier"].items():
            if v.get("not_evaluable"):continue
            row={"track":r["track"],"seed":r["seed"],"variant":variant,"n_test":v["n_test"]}
            row.update({m:v["test"][m] for m in METRICS});rows.append(row)
            danger.append({"track":r["track"],"seed":r["seed"],"variant":variant,**v["test"]["dangerous_errors"]})
    tier=pd.DataFrame(rows);domain=pd.DataFrame(domain_rows);pd.DataFrame(danger).to_csv(root/"reports"/"baseline1_dangerous_errors.csv",index=False)
    tier.to_csv(root/"reports"/"baseline1_tier_results.csv",index=False);domain.to_csv(root/"reports"/"baseline1_domain_results.csv",index=False)
    summary=[]
    for keys,g in tier.groupby(["track","variant"]):
        for m in METRICS:
            summary.append({"track":keys[0],"variant":keys[1],"metric":m,"mean":g[m].mean(),"std":g[m].std(ddof=1),
                            "minimum":g[m].min(),"maximum":g[m].max(),"worst_seed":int(g.loc[g[m].idxmin() if m in ('exact_accuracy','macro_f1','within_one_tier_accuracy') else g[m].idxmax(),"seed"])})
    pd.DataFrame(summary).to_csv(root/"reports"/"baseline1_seed_summary.csv",index=False)
    dsummary=domain.groupby("track").agg({m:["mean","std","min","max"] for m in ("accuracy","macro_f1","log_loss")})
    dsummary.to_csv(root/"reports"/"baseline1_domain_seed_summary.csv")
    # Per-domain tier metrics from saved probabilities.
    labels=pd.read_parquet(root.parent/"data_pipeline"/"output"/"final_labeled_dataset.parquet",columns=["prompt_id","component"])
    labels["semantic_domain"]=labels.component.replace({"logic_arc_agi":"logic","logic_classic":"logic"})
    per=[]
    for p in (root/"reports"/"predictions").glob("*.parquet"):
        stem=p.stem
        track="primary_source_isolated" if stem.startswith("primary_source_isolated") else "secondary_prompt_stratified"
        rest=stem[len(track)+1:]; seed=int(rest.split("_",1)[0]);variant=rest.split("_",1)[1]
        x=pd.read_parquet(p).merge(labels,on="prompt_id");delta=x.predicted_tier-x.true_tier
        for dom,part in x.assign(delta=delta).groupby("semantic_domain"):
            per.append({"track":track,"seed":seed,"variant":variant,"domain":dom,"n":len(part),
                        "accuracy":(part.delta==0).mean(),"mae":part.delta.abs().mean(),
                        "under_routing":(part.delta<0).mean(),"severe_under_routing":(part.delta<=-2).mean()})
    perdf=pd.DataFrame(per);perdf.to_csv(root/"reports"/"baseline1_tier_by_domain.csv",index=False)
    perdf.groupby(["track","variant","domain"]).agg({"n":"sum","accuracy":["mean","std","min"],"mae":["mean","std","max"],
        "under_routing":["mean","std","max"],"severe_under_routing":["mean","std","max"]}).to_csv(root/"reports"/"baseline1_tier_by_domain_summary.csv")
    print(domain.groupby("track")[["accuracy","macro_f1","log_loss"]].agg(["mean","std","min"]).to_string())
    print(tier.groupby(["track","variant"])[METRICS].agg(["mean","std","min","max"]).to_string())

if __name__=="__main__":run()
