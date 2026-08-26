from __future__ import annotations

import json, time
from pathlib import Path
import joblib, numpy as np, pandas as pd, yaml
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from .build_grouped_splits import semantic_domain
from .train_models import make_logreg, oof_domain_probabilities
from .evaluate import domain_metrics, tier_metrics
from .calibration_analysis import confidence_bins


def load_features(root, cfg, df):
    files=list((root/"cache"/"embeddings").glob("qwen3_0.6b_*.npz"))
    if len(files)!=1: raise RuntimeError(f"Expected one official cache, found {files}")
    cache=np.load(files[0]); ids=cache["prompt_id"].astype(str); emb=cache["embedding"].astype(np.float32)
    if not np.array_equal(ids,df.prompt_id.astype(str).to_numpy()): raise RuntimeError("Embedding row alignment mismatch")
    norms=np.linalg.norm(emb,axis=1)
    if not np.allclose(norms,1,atol=2e-3): raise RuntimeError("Embedding normalization check failed")
    counts=pd.read_parquet(root/"cache"/"features"/"request_counts.parquet").set_index("prompt_id").loc[ids]
    structural=np.zeros((len(df),15),dtype=np.float32)
    structural[:,0]=np.log1p(counts.token_count.to_numpy())
    structural[:,1]=np.log1p(df.prompt.astype(str).str.len().to_numpy())
    structural[:,2]=np.log1p(1); structural[:,3]=np.log1p(counts.chunk_count.to_numpy())
    structural[:,5]=structural[:,0]; structural[:,12]=1
    semantic=np.concatenate([emb,np.zeros_like(emb),np.zeros_like(emb)],axis=1)
    return semantic,structural,files[0]


def split_mask(df,root,seed,track):
    if track=="primary_source_isolated":
        spec=json.load(open(root/"splits"/f"grouped_split_{seed}.json"))["datasets"]
        return df.source_dataset.map(spec)
    p=pd.read_parquet(root/"splits"/f"prompt_stratified_split_{seed}.parquet").set_index("prompt_id")
    return df.prompt_id.map(p.split)


def select_model(Xtr,ytr,Xv,yv,grid):
    rows=[]; best=None
    for C in grid["C"]:
        for cw in grid["class_weight"]:
            m=make_logreg(C,cw,grid["max_iter"]); t=time.perf_counter();m.fit(Xtr,ytr);pred=m.predict(Xv)
            row={"C":C,"class_weight":cw,"validation_macro_f1":f1_score(yv,pred,average="macro",zero_division=0),
                 "validation_accuracy":accuracy_score(yv,pred),"fit_seconds":time.perf_counter()-t}
            rows.append(row)
            key=(row["validation_macro_f1"],row["validation_accuracy"],-float(C),cw is None)
            if best is None or key>best[0]: best=(key,m,row)
    return best[1],best[2],rows


def evaluate_per_source(df,idx,y,pred,kind):
    rows=[]
    for source in sorted(df.iloc[idx].source_dataset.unique()):
        mask=df.iloc[idx].source_dataset.to_numpy()==source; yt=np.asarray(y)[mask]; yp=np.asarray(pred)[mask]
        row={"source_dataset":source,"n":int(mask.sum()),"accuracy":accuracy_score(yt,yp),
             "macro_f1":f1_score(yt,yp,average="macro",zero_division=0)}
        if kind=="tier":
            d=yp.astype(int)-yt.astype(int);row.update(mae=float(np.abs(d).mean()),under_routing=float((d<0).mean()),severe_under_routing=float((d<=-2).mean()))
        rows.append(row)
    return rows


def run_one(df,semantic,structural,root,cfg,seed,track):
    split=split_mask(df,root,seed,track); idx={k:np.flatnonzero(split.to_numpy()==k) for k in ("train","validation","test")}
    scaler=StandardScaler().fit(structural[idx["train"]]); scaled=scaler.transform(structural).astype(np.float32)
    X=np.concatenate([semantic,scaled],axis=1); domain=semantic_domain(df).to_numpy(); classes=sorted(set(domain))
    dm,dsel,dgrid=select_model(X[idx["train"]],domain[idx["train"]],X[idx["validation"]],domain[idx["validation"]],cfg["logistic_regression"])
    dprob={k:dm.predict_proba(X[v]) for k,v in idx.items()}; dpred={k:dm.predict(X[v]) for k,v in idx.items()}
    domain_test=domain_metrics(domain[idx["test"]],dpred["test"],dprob["test"],list(dm.classes_))
    domain_test["confidence_bins"]=confidence_bins(domain[idx["test"]],dpred["test"],dprob["test"])
    domain_test["per_source"]=evaluate_per_source(df,idx["test"],domain[idx["test"]],dpred["test"],"domain")
    train_idx=idx["train"]
    oof,methods=oof_domain_probabilities(X[train_idx],domain[train_idx],df.source_dataset.iloc[train_idx].to_numpy(),classes,
                                         cfg["splits"]["oof_folds"],dsel["C"],dsel["class_weight"])
    results={"seed":seed,"track":track,"split_counts":{k:len(v) for k,v in idx.items()},"domain_selection":dsel,
             "domain_grid":dgrid,"domain_test":domain_test,"oof_methods":methods,"tier":{}}
    resolved=df.resolved.to_numpy(); true_tier=df.tier.to_numpy()
    domain_col={c:i for i,c in enumerate(classes)}
    oracle=np.zeros((len(df),len(classes)),dtype=np.float32)
    for i,c in enumerate(domain): oracle[i,domain_col[c]]=1
    variants=("features_only","predicted_domain","oracle_domain")
    for variant in variants:
        tr=train_idx[resolved[train_idx]]; va=idx["validation"][resolved[idx["validation"]]]; te=idx["test"][resolved[idx["test"]]]
        if not len(te) or len(set(true_tier[tr]))<4:
            results["tier"][variant]={"not_evaluable":True,"n_test":len(te)};continue
        if variant=="features_only": Xt,Xv,Xe=X[tr],X[va],X[te]
        elif variant=="oracle_domain": Xt=np.c_[X[tr],oracle[tr]];Xv=np.c_[X[va],oracle[va]];Xe=np.c_[X[te],oracle[te]]
        else:
            train_pos={row:i for i,row in enumerate(train_idx)}
            Xt=np.c_[X[tr],oof[[train_pos[r] for r in tr]]]
            Xv=np.c_[X[va],dm.predict_proba(X[va])];Xe=np.c_[X[te],dm.predict_proba(X[te])]
        tm,tsel,tgrid=select_model(Xt,true_tier[tr].astype(int),Xv,true_tier[va].astype(int),cfg["logistic_regression"])
        prob=tm.predict_proba(Xe);pred=tm.predict(Xe);metrics=tier_metrics(true_tier[te],pred,prob)
        metrics["confidence_bins"]=confidence_bins(true_tier[te].astype(int),pred,prob)
        metrics["per_source"]=evaluate_per_source(df,te,true_tier[te].astype(int),pred,"tier")
        results["tier"][variant]={"selection":tsel,"grid":tgrid,"test":metrics,"n_train":len(tr),"n_validation":len(va),"n_test":len(te)}
        pred_dir=root/"reports"/"predictions";pred_dir.mkdir(parents=True,exist_ok=True)
        pd.DataFrame({"prompt_id":df.prompt_id.iloc[te],"true_tier":true_tier[te].astype(int),"predicted_tier":pred,
                      **{f"p_tier_{c}":prob[:,j] for j,c in enumerate(tm.classes_)}}).to_parquet(pred_dir/f"{track}_{seed}_{variant}.parquet",index=False)
        model_dir=root/"output"/track/f"seed_{seed}";model_dir.mkdir(parents=True,exist_ok=True)
        joblib.dump(tm,model_dir/f"tier_{variant}.joblib")
    model_dir=root/"output"/track/f"seed_{seed}";model_dir.mkdir(parents=True,exist_ok=True)
    joblib.dump({"model":dm,"structural_scaler":scaler,"domain_classes":classes},model_dir/"domain.joblib")
    return results


def run(config_path="configs/baseline1.yaml"):
    cfg=yaml.safe_load(open(config_path));root=Path(config_path).resolve().parent.parent
    cols=["prompt_id","prompt","source_dataset","component","tier","resolved"]
    df=pd.read_parquet((root/cfg["data"]["labeled_dataset"]).resolve(),columns=cols)
    semantic,structural,cache_file=load_features(root,cfg,df)
    all_results=[]
    for track in ("primary_source_isolated","secondary_prompt_stratified"):
        for seed in cfg["splits"]["seeds"]:
            print(f"RUN {track} seed={seed}",flush=True)
            result=run_one(df,semantic,structural,root,cfg,seed,track);all_results.append(result)
            out=root/"reports"/f"baseline1_{track}_{seed}.json";out.write_text(json.dumps(result,indent=2))
    (root/"reports"/"baseline1_all_results.json").write_text(json.dumps(all_results,indent=2))
    print("COMPLETE",len(all_results))


if __name__=="__main__":run()
