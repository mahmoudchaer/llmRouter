from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,yaml
from sklearn.metrics import f1_score
from .xgb_models import deterministic_search_configs,fit_candidate,domain_specific_eligible
from .train_models import hybrid_oof_fold_ids


def load_cfg(path):return yaml.safe_load(open(path))


def audit(config_path):
    cfg=load_cfg(config_path);root=Path(config_path).resolve().parent.parent
    meta=json.load(open(root/cfg["cache"]["metadata_file"]));assert meta["fingerprint"]==cfg["cache"]["expected_fingerprint"]
    cache=np.load(root/cfg["cache"]["embedding_file"],mmap_mode="r")
    df=pd.read_parquet((root/cfg["data"]["labeled_dataset"]).resolve(),columns=["prompt_id","component","resolved","tier","source_dataset"])
    assert len(df)==25203 and len(cache["prompt_id"])==len(df)
    assert np.array_equal(cache["prompt_id"].astype(str),df.prompt_id.astype(str).to_numpy())
    df["domain"]=df.component.replace({"logic_arc_agi":"logic","logic_classic":"logic"})
    counts=pd.crosstab(df.loc[df.resolved,"domain"],df.loc[df.resolved,"tier"])
    report={"cache_fingerprint":meta["fingerprint"],"rows":len(df),"resolved":int(df.resolved.sum()),
            "search_configurations":len(deterministic_search_configs(cfg["xgboost"])),
            "resolved_by_domain":{d:{"total":int(row.sum()),"tiers":{str(int(k)):int(v) for k,v in row.items()}} for d,row in counts.iterrows()},
            "split_files":{str(s):{"primary":str(root/cfg["splits"]["primary_pattern"].format(seed=s)),
                                    "secondary":str(root/cfg["splits"]["secondary_pattern"].format(seed=s))} for s in cfg["splits"]["seeds"]}}
    out=root/"reports"/"baseline2_implementation_audit.json";out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    pd.DataFrame(deterministic_search_configs(cfg["xgboost"])).to_csv(root/"reports"/"baseline2_search_configurations.csv",index_label="configuration")
    eligibility=[]
    for seed in cfg["splits"]["seeds"]:
        assignment=json.load(open(root/cfg["splits"]["primary_pattern"].format(seed=seed)))["datasets"]
        train=df.source_dataset.map(assignment).eq("train") & df.resolved
        for domain,g in df[train].groupby("domain"):
            ok,dist=domain_specific_eligible((g.tier.to_numpy()-1).astype(int),cfg["domain_specific_tier"])
            eligibility.append({"seed":seed,"domain":domain,"eligible":ok,"n_train":len(g),**{f"tier_{k}":v for k,v in dist.items()}})
    pd.DataFrame(eligibility).fillna(0).to_csv(root/"reports"/"baseline2_domain_specific_eligibility.csv",index=False)


def smoke(config_path):
    cfg=load_cfg(config_path);root=Path(config_path).resolve().parent.parent
    cache=np.load(root/cfg["cache"]["embedding_file"]);X=cache["embedding"][:600].astype(np.float32)
    df=pd.read_parquet((root/cfg["data"]["labeled_dataset"]).resolve(),columns=["component","source_dataset","tier","resolved"]).iloc[:600]
    y=pd.Categorical(df.component.replace({"logic_arc_agi":"logic","logic_classic":"logic"})).codes
    tr=np.arange(0,450);va=np.arange(450,600);params=deterministic_search_configs(cfg["xgboost"])[0]
    # Synthetic target guarantees every class in train/validation for API validation.
    ys=np.arange(600)%4
    model=fit_candidate(X[tr],ys[tr],X[va],ys[va],params,cfg["xgboost"],smoke=True)
    p=model.predict_proba(X[va]);assert p.shape==(150,4) and np.allclose(p.sum(1),1,atol=1e-5)
    folds,methods=hybrid_oof_fold_ids(np.array(["tool"]*20+["code"]*20),np.array(["tau2"]*20+["c1"]*10+["c2"]*10),5)
    eligible,dist=domain_specific_eligible(np.tile(np.arange(4),150),cfg["domain_specific_tier"])
    result={"xgboost_version":__import__("xgboost").__version__,"probability_shape":list(p.shape),"eligible_synthetic":bool(eligible),
            "eligibility_distribution":dist,"hybrid_oof_methods":methods,"best_iteration":int(model.best_iteration)}
    (root/"reports"/"baseline2_smoke_test.json").write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("command",choices=["audit","smoke"]);ap.add_argument("--config",default="configs/baseline2.yaml");a=ap.parse_args()
    (audit if a.command=="audit" else smoke)(a.config)
