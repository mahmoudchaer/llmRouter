from __future__ import annotations

import argparse, json
from pathlib import Path
import pandas as pd
import yaml
from .build_grouped_splits import controlled_group_split, save_split, semantic_domain, prompt_stratified_split


def load_config(path):
    with open(path) as f: return yaml.safe_load(f)


def prepare(config_path: str):
    root=Path(config_path).resolve().parent.parent
    cfg=load_config(config_path)
    data_path=(root / cfg["data"]["labeled_dataset"]).resolve()
    cols=["prompt_id","prompt","source_dataset","domain","component","tier","resolved"]
    df=pd.read_parquet(data_path,columns=cols)
    assert len(df)==25203 and df.prompt_id.is_unique
    df["semantic_domain"]=semantic_domain(df)
    reports=root/"reports"; splits=root/"splits"
    reports.mkdir(exist_ok=True); splits.mkdir(exist_ok=True)
    audits=[]
    for seed in cfg["splits"]["seeds"]:
        assignment,score=controlled_group_split(df,seed,cfg["splits"]["search_iterations"])
        save_split(splits/f"grouped_split_{seed}.json",seed,assignment,score)
        row_split=df.source_dataset.map(assignment)
        audits.append({"seed":seed,"objective":score,"datasets":assignment,
                       "row_counts":row_split.value_counts().to_dict(),
                       "domain_counts":pd.crosstab(row_split,df.semantic_domain).to_dict(orient="index"),
                       "resolved_tier_counts":pd.crosstab(row_split[df.resolved],df.loc[df.resolved,"tier"]).to_dict(orient="index")})
        secondary=prompt_stratified_split(df,seed)
        pd.DataFrame({"prompt_id":df.prompt_id,"split":secondary}).to_parquet(
            splits/f"prompt_stratified_split_{seed}.parquet",index=False)
    source_counts=df.groupby("semantic_domain").source_dataset.nunique().to_dict()
    oof_policy={d:{"n_source_datasets":int(n),
                   "method":"source_grouped" if n>=2 else "stratified_prompt_fallback",
                   "primary_unseen_source_evaluable":bool(n>=2)} for d,n in source_counts.items()}
    schema={"semantic_dimension":3*cfg["embedding"]["dimension"],
            "structural_dimension":len(cfg["features"]["structural"]),
            "base_dimension":3*cfg["embedding"]["dimension"]+len(cfg["features"]["structural"]),
            "domain_conditioned_dimension":3*cfg["embedding"]["dimension"]+len(cfg["features"]["structural"])+7,
            "semantic_blocks":cfg["features"]["semantic_blocks"],
            "structural_features":cfg["features"]["structural"]}
    (reports/"baseline1_feature_schema.json").write_text(json.dumps(schema,indent=2))
    (reports/"baseline1_split_audit.json").write_text(json.dumps(audits,indent=2))
    (reports/"baseline1_oof_policy.json").write_text(json.dumps(oof_policy,indent=2,sort_keys=True))
    print(json.dumps({"rows":len(df),"resolved":int(df.resolved.sum()),"schema":schema},indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("command",choices=["prepare"]); ap.add_argument("--config",default="configs/baseline1.yaml")
    a=ap.parse_args()
    if a.command=="prepare": prepare(a.config)
