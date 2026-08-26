"""Full Baseline 2 runner. Deliberately not invoked during implementation review."""
from __future__ import annotations
import json
from pathlib import Path
import joblib,numpy as np,pandas as pd,yaml
from sklearn.preprocessing import StandardScaler
from .run_baseline1_experiments import load_features,split_mask,evaluate_per_source
from .build_grouped_splits import semantic_domain
from .evaluate import domain_metrics,tier_metrics
from .calibration_analysis import confidence_bins
from .xgb_models import deterministic_search_configs,select_xgb,xgb_oof_probabilities,domain_specific_eligible,fit_candidate


def gain_importance(model, feature_names):
    raw=model.get_booster().get_score(importance_type="gain")
    return pd.DataFrame([{"feature":feature_names[int(k[1:])],"gain":v} for k,v in raw.items()]).sort_values("gain",ascending=False)


def approximate_shap_importance(model,X,n,seed,names):
    import xgboost as xgb
    rng=np.random.default_rng(seed);take=rng.choice(len(X),size=min(n,len(X)),replace=False)
    contrib=model.get_booster().predict(xgb.DMatrix(X[take]),pred_contribs=True,approx_contribs=True)
    values=np.asarray(contrib)
    if values.ndim==3: values=np.abs(values[...,:-1]).mean(axis=(0,1))
    else: values=np.abs(values[:,:-1]).mean(axis=0)
    return pd.DataFrame({"feature":names,"mean_abs_approx_shap":values}).sort_values("mean_abs_approx_shap",ascending=False)


def feature_names():
    structural=["log1p_total_tokens","log1p_character_count","log1p_message_count","log1p_chunk_count",
    "log1p_system_tokens","log1p_user_tokens","log1p_history_tokens","log1p_retrieved_context_tokens",
    "log1p_tool_schema_tokens","log1p_tool_count","structured_output_flag","json_schema_requirement_flag",
    "modality_text_flag","modality_image_flag","modality_audio_flag"]
    return [f"task_embedding_{i}" for i in range(1024)]+[f"context_mean_delta_{i}" for i in range(1024)]+[f"context_relevant_delta_{i}" for i in range(1024)]+structural


def run_one(df,X,root,cfg,seed,track):
    split=split_mask(df,root,seed,track);idx={k:np.flatnonzero(split.to_numpy()==k) for k in ("train","validation","test")}
    scaler=StandardScaler().fit(X[idx["train"],-15:]);X=X.copy();X[:,-15:]=scaler.transform(X[:,-15:])
    domains=semantic_domain(df).to_numpy();classes=sorted(set(domains));enc={c:i for i,c in enumerate(classes)};dy=np.array([enc[x] for x in domains])
    configs=deterministic_search_configs(cfg["xgboost"])
    dm,dsel,dgrid=select_xgb(X[idx["train"]],dy[idx["train"]],X[idx["validation"]],dy[idx["validation"]],configs,cfg["xgboost"])
    dprob=dm.predict_proba(X[idx["test"]]);dpred=dm.predict(X[idx["test"]]);dtrue=dy[idx["test"]]
    dmetrics=domain_metrics(dtrue,dpred,dprob,list(range(7)));dmetrics["confidence_bins"]=confidence_bins(dtrue,dpred,dprob)
    dmetrics["per_source"]=evaluate_per_source(df,idx["test"],dtrue,dpred,"domain")
    oof,methods=xgb_oof_probabilities(X[idx["train"]],dy[idx["train"]],df.source_dataset.iloc[idx["train"]].to_numpy(),list(range(7)),dsel,cfg["xgboost"],cfg["splits"]["oof_folds"])
    resolved=df.resolved.to_numpy();tier=df.tier.to_numpy();tr=idx["train"][resolved[idx["train"]]];va=idx["validation"][resolved[idx["validation"]]];te=idx["test"][resolved[idx["test"]]]
    ty=tier.astype(float)-1;oracle=np.eye(7,dtype=np.float32)[dy]
    train_pos={r:i for i,r in enumerate(idx["train"])}
    variants={"features_only":(X[tr],X[va],X[te]),"predicted_domain":(np.c_[X[tr],oof[[train_pos[r] for r in tr]]],np.c_[X[va],dm.predict_proba(X[va])],np.c_[X[te],dm.predict_proba(X[te])]),
              "oracle_domain":(np.c_[X[tr],oracle[tr]],np.c_[X[va],oracle[va]],np.c_[X[te],oracle[te]])}
    source_audit={}
    for target_domain in ("knowledge","logic"):
        source_audit[target_domain]={}
        for part,rows in idx.items():
            sub=df.iloc[rows].copy();sub["semantic_domain"]=semantic_domain(sub);sub=sub[sub.semantic_domain==target_domain]
            source_audit[target_domain][part]={s:{"prompts":len(g),"resolved":int(g.resolved.sum()),
                "tier_distribution":{str(int(k)):int(v) for k,v in g.loc[g.resolved,"tier"].value_counts().sort_index().items()}}
                for s,g in sub.groupby("source_dataset")}
    result={"seed":seed,"track":track,"domain_selection":dsel,"domain_grid":dgrid,"domain_test":dmetrics,"oof_methods":methods,
            "knowledge_logic_split_audit":source_audit,"tier":{}}
    models={}
    for name,(Xt,Xv,Xe) in variants.items():
        model,sel,grid=select_xgb(Xt,ty[tr].astype(int),Xv,ty[va].astype(int),configs,cfg["xgboost"]);models[name]=model
        prob=model.predict_proba(Xe);pred=model.predict(Xe)+1;metrics=tier_metrics(tier[te].astype(int),pred,prob)
        metrics["confidence_bins"]=confidence_bins(tier[te].astype(int),pred,prob);metrics["per_source"]=evaluate_per_source(df,te,tier[te].astype(int),pred,"tier")
        result["tier"][name]={"selection":sel,"grid":grid,"test":metrics,"n_test":len(te)}
        pred_dir=root/"reports"/"baseline2_predictions";pred_dir.mkdir(parents=True,exist_ok=True)
        pd.DataFrame({"prompt_id":df.prompt_id.iloc[te],"true_tier":tier[te].astype(int),"predicted_tier":pred,
          **{f"p_tier_{i+1}":prob[:,i] for i in range(4)}}).to_parquet(pred_dir/f"{track}_{seed}_{name}.parquet",index=False)
    # True-domain routed diagnostic; ineligible domains use global features-only.
    base=models["features_only"];ds_pred=base.predict(X[te])+1;ds_prob=base.predict_proba(X[te]);eligibility={}
    for domain in classes:
        dtr=tr[domains[tr]==domain];dva=va[domains[va]==domain];dte_pos=np.flatnonzero(domains[te]==domain)
        eligible,distribution=domain_specific_eligible(ty[dtr].astype(int),cfg["domain_specific_tier"])
        eligibility[domain]={"eligible":eligible,"n_train":len(dtr),"tier_distribution":distribution}
        if eligible and len(dva) and len(dte_pos):
            model=fit_candidate(X[dtr],ty[dtr].astype(int),X[dva],ty[dva].astype(int),result["tier"]["features_only"]["selection"],cfg["xgboost"])
            ds_pred[dte_pos]=model.predict(X[te[dte_pos]])+1;ds_prob[dte_pos]=model.predict_proba(X[te[dte_pos]])
    result["tier"]["domain_specific_true_domain_routing"]={"eligibility":eligibility,"test":tier_metrics(tier[te].astype(int),ds_pred,ds_prob),"n_test":len(te)}
    imp=root/"reports"/"baseline2_feature_importance";imp.mkdir(parents=True,exist_ok=True)
    gain_importance(dm,feature_names()).to_csv(imp/f"{track}_{seed}_domain_gain.csv",index=False)
    gain_importance(models["features_only"],feature_names()).to_csv(imp/f"{track}_{seed}_tier_gain.csv",index=False)
    approximate_shap_importance(dm,X[idx["test"]],cfg["importance"]["shap_sample_rows"],cfg["importance"]["shap_seed"],feature_names()).to_csv(
        imp/f"{track}_{seed}_domain_approx_shap.csv",index=False)
    approximate_shap_importance(models["features_only"],X[te],cfg["importance"]["shap_sample_rows"],cfg["importance"]["shap_seed"],feature_names()).to_csv(
        imp/f"{track}_{seed}_tier_approx_shap.csv",index=False)
    outdir=root/"output"/"baseline2"/track/f"seed_{seed}";outdir.mkdir(parents=True,exist_ok=True);joblib.dump({"domain":dm,**models},outdir/"models.joblib")
    return result


def run(config_path="configs/baseline2.yaml"):
    cfg=yaml.safe_load(open(config_path));root=Path(config_path).resolve().parent.parent
    df=pd.read_parquet((root/cfg["data"]["labeled_dataset"]).resolve(),columns=["prompt_id","prompt","source_dataset","component","tier","resolved"])
    b1=yaml.safe_load(open(root/cfg["baseline1_config"]));semantic,structural,_=load_features(root,b1,df)
    # Trees do not require standardization; identical values are retained.
    X=np.concatenate([semantic,structural],axis=1).astype(np.float32);results=[]
    for track in ("primary_source_isolated","secondary_prompt_stratified"):
        for seed in cfg["splits"]["seeds"]:
            print("RUN",track,seed,flush=True);r=run_one(df,X,root,cfg,seed,track);results.append(r)
            (root/"reports"/f"baseline2_{track}_{seed}.json").write_text(json.dumps(r,indent=2))
    (root/"reports"/"baseline2_all_results.json").write_text(json.dumps(results,indent=2))

if __name__=="__main__":run()
