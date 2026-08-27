from __future__ import annotations
import pandas as pd
from routing.evaluation.ensemble_eval import evaluate_tier_predictions
from routing.models.small_llm import SmallLLMDomainClassifier,SmallLLMTierClassifier


def evaluate_llm(domain_classifier: SmallLLMDomainClassifier, tier_classifier: SmallLLMTierClassifier,
                 examples: pd.DataFrame) -> tuple[pd.DataFrame,dict]:
    rows=[]
    for row in examples.itertuples():
        domain=domain_classifier.classify_domain(str(row.prompt));tier=tier_classifier.classify_tier(str(row.prompt))
        rows.append({"prompt_id":row.prompt_id,"true_domain":row.domain,"true_tier":int(row.tier),
                     "predicted_domain":domain.domain,"predicted_tier":tier.tier,
                     "domain_chunks":domain.chunks_classified,"tier_chunks":tier.chunks_classified})
    result=pd.DataFrame(rows);metrics=evaluate_tier_predictions(result.true_tier,result.predicted_tier)
    metrics["domain_accuracy"]=float((result.true_domain==result.predicted_domain).mean())
    return result,metrics
