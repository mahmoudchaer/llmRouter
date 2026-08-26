from __future__ import annotations
from collections import Counter
import itertools, numpy as np


def deterministic_search_configs(cfg: dict) -> list[dict]:
    keys=["n_estimators","max_depth","learning_rate","subsample","colsample_bytree",
          "min_child_weight","reg_lambda","reg_alpha"]
    combos=list(itertools.product(*(cfg[k] for k in keys)))
    rng=np.random.default_rng(cfg["search_seed"])
    n=cfg["configurations"]//2
    chosen=rng.choice(len(combos),size=n,replace=False)
    result=[]
    for i in chosen:
        base=dict(zip(keys,combos[i]))
        for weighting in ("none","balanced"): result.append({**base,"weighting":weighting})
    return result


def balanced_weights(y):
    y=np.asarray(y);counts=Counter(y);n=len(y);k=len(counts)
    return np.asarray([n/(k*counts[v]) for v in y],dtype=np.float32)


def make_xgb(params: dict, n_classes: int, cfg: dict, smoke=False):
    from xgboost import XGBClassifier
    allowed={"n_estimators","max_depth","learning_rate","subsample","colsample_bytree",
             "min_child_weight","reg_lambda","reg_alpha"}
    p={k:v for k,v in params.items() if k in allowed}
    if smoke:p["n_estimators"]=min(20,p["n_estimators"])
    return XGBClassifier(**p,objective="multi:softprob",num_class=n_classes,
                         eval_metric=cfg["eval_metric"],tree_method=cfg["tree_method"],
                         device=cfg["device"],n_jobs=cfg["n_jobs"],random_state=cfg["search_seed"],
                         early_stopping_rounds=cfg["early_stopping_rounds"] if not smoke else 5)


def fit_candidate(Xtr,ytr,Xv,yv,params,cfg,smoke=False):
    model=make_xgb(params,len(np.unique(ytr)),cfg,smoke)
    weight=balanced_weights(ytr) if params["weighting"]=="balanced" else None
    model.fit(Xtr,ytr,sample_weight=weight,eval_set=[(Xv,yv)],verbose=False)
    return model


def select_xgb(Xtr,ytr,Xv,yv,configs,cfg):
    from sklearn.metrics import f1_score,accuracy_score
    rows=[];best=None
    for i,params in enumerate(configs):
        model=fit_candidate(Xtr,ytr,Xv,yv,params,cfg)
        pred=model.predict(Xv);row={"configuration":i,**params,"best_iteration":model.best_iteration,
            "validation_macro_f1":f1_score(yv,pred,average="macro",zero_division=0),"validation_accuracy":accuracy_score(yv,pred)}
        rows.append(row);key=(row["validation_macro_f1"],row["validation_accuracy"],-model.best_iteration)
        if best is None or key>best[0]:best=(key,model,row)
    return best[1],best[2],rows


def xgb_oof_probabilities(X,y,sources,classes,params,cfg,n_splits=5):
    from .train_models import hybrid_oof_fold_ids
    fold,methods=hybrid_oof_fold_ids(y,sources,n_splits=n_splits)
    out=np.zeros((len(y),len(classes)),dtype=np.float32)
    for k in range(n_splits):
        tr=np.flatnonzero(fold!=k);ho=np.flatnonzero(fold==k)
        model=fit_candidate(X[tr],y[tr],X[ho],y[ho],params,cfg)
        p=model.predict_proba(X[ho])
        for j,c in enumerate(model.classes_):out[ho,classes.index(c)]=p[:,j]
    if not np.allclose(out.sum(1),1,atol=1e-5):raise RuntimeError("XGBoost OOF coverage incomplete")
    return out,methods


def domain_specific_eligible(y, cfg):
    y=np.asarray(y);values,counts=np.unique(y,return_counts=True)
    eligible=(len(y)>=cfg["minimum_resolved_train_rows"] and
            (not cfg["require_all_four_tiers"] or set(values)=={0,1,2,3}) and
            counts.min()>=cfg["minimum_rows_per_tier"])
    return bool(eligible), {int(v+1):int(n) for v,n in zip(values,counts)}
