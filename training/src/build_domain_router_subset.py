from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd,yaml

def stable_key(seed,prompt_id):return hashlib.sha256(f"{seed}:{prompt_id}".encode()).hexdigest()
def main():
    cfg=yaml.safe_load(open("configs/domain_router_eval.yaml"));df=pd.read_parquet("../data_pipeline/output/final_labeled_dataset.parquet")[["prompt_id","prompt","source_dataset","domain","component"]].copy();df["domain"]=df.domain.replace({"logic_arc_agi":"logic","logic_classic":"logic"});chosen=[]
    for domain,g in df.groupby("domain"):
        target=cfg["samples_per_domain"];sources=sorted(g.source_dataset.unique());alloc={s:target//len(sources) for s in sources}
        for s in sources[:target%len(sources)]:alloc[s]+=1
        leftovers=[]
        for source,sg in g.groupby("source_dataset"):
            sg=sg.copy();sg["length_bin"]=pd.qcut(sg.prompt.astype(str).str.len().rank(method="first"),min(4,len(sg)),labels=False,duplicates="drop");sg["key"]=sg.prompt_id.map(lambda x:stable_key(cfg["seed"],x));parts=[]
            n=min(alloc[source],len(sg));bins=sorted(sg.length_bin.unique());base=n//len(bins);extra=n%len(bins)
            for i,b in enumerate(bins):parts.append(sg[sg.length_bin.eq(b)].sort_values("key").head(base+(i<extra)))
            pick=pd.concat(parts).head(n);chosen.append(pick);leftovers.append(sg[~sg.prompt_id.isin(pick.prompt_id)])
        current=sum(len(x) for x in chosen if len(x) and x.domain.iloc[0]==domain);need=target-current
        if need>0:chosen.append(pd.concat(leftovers).sort_values("key").head(need))
    out=pd.concat(chosen).drop_duplicates("prompt_id").sort_values(["domain","source_dataset","prompt_id"]);Path("reports/domain_router_eval").mkdir(parents=True,exist_ok=True);out.to_parquet("reports/domain_router_eval/subset.parquet",index=False);out[["prompt_id","domain","source_dataset"]].to_csv("reports/domain_router_eval/selected_prompt_ids.csv",index=False)
    audit={"seed":cfg["seed"],"n":len(out),"by_domain":out.domain.value_counts().sort_index().to_dict(),"by_source":out.source_dataset.value_counts().sort_index().to_dict()};Path("reports/domain_router_eval/subset_audit.json").write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
if __name__=="__main__":main()
