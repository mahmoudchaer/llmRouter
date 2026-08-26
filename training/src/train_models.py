from __future__ import annotations

import numpy as np
import pandas as pd


def make_logreg(C=1.0, class_weight=None, max_iter=2000):
    from sklearn.linear_model import LogisticRegression
    # sklearn >=1.7 infers multinomial for multiclass; lbfgs is deterministic here.
    return LogisticRegression(C=C, class_weight=class_weight, solver="lbfgs", max_iter=max_iter)


def hybrid_oof_fold_ids(y, sources, n_splits=5, seed=173):
    """Source-group classes with >=2 sources; prompt-fold singleton-source classes."""
    y=np.asarray(y); sources=np.asarray(sources); rng=np.random.default_rng(seed)
    fold=np.full(len(y),-1,dtype=int); methods={}
    for cls in sorted(set(y)):
        idx=np.flatnonzero(y==cls); unique=np.unique(sources[idx])
        if len(unique)>=2:
            methods[str(cls)]={"method":"source_grouped","n_sources":int(len(unique))}
            shuffled=rng.permutation(unique)
            source_fold={s:i % n_splits for i,s in enumerate(shuffled)}
            fold[idx]=[source_fold[s] for s in sources[idx]]
        else:
            methods[str(cls)]={"method":"stratified_prompt_fallback","n_sources":int(len(unique))}
            shuffled=rng.permutation(idx)
            fold[shuffled]=np.arange(len(shuffled)) % n_splits
    if (fold<0).any(): raise RuntimeError("Hybrid OOF assignment incomplete")
    # Every training complement must contain every class.
    for k in range(n_splits):
        if set(y[fold!=k]) != set(y): raise RuntimeError(f"Class absent from OOF training fold {k}")
    return fold, methods


def oof_domain_probabilities(X, y, groups, classes, n_splits=5, C=1.0, class_weight=None):
    out=np.zeros((len(y),len(classes)),dtype=np.float32)
    fold,methods=hybrid_oof_fold_ids(y,groups,n_splits=n_splits)
    for k in range(n_splits):
        train=np.flatnonzero(fold!=k); holdout=np.flatnonzero(fold==k)
        model=make_logreg(C,class_weight); model.fit(X[train],y[train])
        p=model.predict_proba(X[holdout])
        for j,c in enumerate(model.classes_): out[holdout,classes.index(c)]=p[:,j]
    if np.any(out.sum(axis=1)==0): raise RuntimeError("OOF coverage incomplete")
    return out, methods
