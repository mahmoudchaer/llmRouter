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
    onehot=np.eye(4)[y-1]
    return {"exact_accuracy": accuracy_score(y,pred), "macro_f1": f1_score(y,pred,average="macro"),
            "mean_absolute_tier_error": float(np.abs(delta).mean()),
            "within_one_tier_accuracy": float((np.abs(delta)<=1).mean()),
            "under_routing_rate": float((delta<0).mean()), "over_routing_rate": float((delta>0).mean()),
            "severe_under_routing_rate": float((delta<=-2).mean()),
            "dangerous_errors": {"T4_to_T1":int(((y==4)&(pred==1)).sum()),
                                  "T4_to_T2":int(((y==4)&(pred==2)).sum()),
                                  "T3_to_T1":int(((y==3)&(pred==1)).sum())},
            "confusion_matrix": confusion_matrix(y,pred,labels=[1,2,3,4]).tolist(),
            "log_loss": log_loss(y,prob,labels=[1,2,3,4]),
            "multiclass_brier": float(((prob-onehot)**2).sum(axis=1).mean())}

