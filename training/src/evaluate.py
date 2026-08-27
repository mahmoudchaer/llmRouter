from __future__ import annotations

import numpy as np


def domain_metrics(y, pred, prob, labels):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
    return {"accuracy": accuracy_score(y, pred),
            "classification_report": classification_report(y, pred, labels=labels, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
            "log_loss": log_loss(y, prob, labels=labels)}


def tier_metrics(y, pred, prob):
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, log_loss
    y=np.asarray(y,dtype=int); pred=np.asarray(pred,dtype=int)
    delta=pred-y
    num_tiers=prob.shape[1];labels=list(range(1,num_tiers+1));onehot=np.eye(num_tiers)[y-1]
    dangerous={}
    for true_tier in range(2,num_tiers+1):
        for predicted_tier in range(1,true_tier):dangerous[f"T{true_tier}_to_T{predicted_tier}"]=int(((y==true_tier)&(pred==predicted_tier)).sum())
    return {"exact_accuracy": accuracy_score(y,pred), "macro_f1": f1_score(y,pred,labels=labels,average="macro",zero_division=0),
            "mean_absolute_tier_error": float(np.abs(delta).mean()),
            "within_one_tier_accuracy": float((np.abs(delta)<=1).mean()),
            "under_routing_rate": float((delta<0).mean()), "over_routing_rate": float((delta>0).mean()),
            "severe_under_routing_rate": float((delta<=-2).mean()),
            "dangerous_errors": dangerous,
            "confusion_matrix": confusion_matrix(y,pred,labels=labels).tolist(),
            "log_loss": log_loss(y,prob,labels=labels),
            "multiclass_brier": float(((prob-onehot)**2).sum(axis=1).mean())}
