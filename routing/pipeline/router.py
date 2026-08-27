from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from routing.models.small_llm import SmallLLMDomainClassifier,SmallLLMTierClassifier
from routing.models.tier_router import DedicatedTierRouter
from routing.policies.tier_decision import TierDecisionPolicy
from routing.schemas.model import ModelRecord
from routing.schemas.request import RoutingRequest
from routing.schemas.routing_result import RoutingDecision
from routing.selection.model_selector import ModelSelector


class RuntimeRouter:
    def __init__(self, domain_classifier: SmallLLMDomainClassifier, llm_tier_classifier: SmallLLMTierClassifier,
                 tier_router: DedicatedTierRouter,
                 decision_policy: TierDecisionPolicy, selector: ModelSelector,
                 registry: list[ModelRecord]):
        self.domain_classifier,self.llm_tier_classifier=domain_classifier,llm_tier_classifier
        self.tier_router=tier_router
        self.decision_policy,self.selector,self.registry=decision_policy,selector,registry

    def route(self, request: RoutingRequest) -> RoutingDecision:
        with ThreadPoolExecutor(max_workers=3,thread_name_prefix="tarsiq-signal") as pool:
            domain_future=pool.submit(self.domain_classifier.classify_domain,request.task_text)
            llm_tier_future=pool.submit(self.llm_tier_classifier.classify_tier,request.task_text)
            tier_future=pool.submit(self.tier_router.predict,request)
            domain=domain_future.result();llm_tier=llm_tier_future.result();dedicated=tier_future.result()
        tier_decision=self.decision_policy.decide(dedicated,llm_tier)
        selection=self.selector.select(self.registry,request,domain.domain,tier_decision.tier)
        audit={
            "domain_llm_prediction":domain.domain,"llm_tier_prediction":llm_tier.tier,
            "tier_router_prediction":dedicated.tier,"tier_router_confidence":dedicated.confidence,
            "tier_router_probabilities":dedicated.probabilities,"tier_disagreement":tier_decision.disagreement,
            "decision_policy_used":tier_decision.policy_used,"uncapped_recommended_tier":tier_decision.tier,
            "final_tier":tier_decision.tier,
            "customer_input_price_ceiling":request.price_ceiling.max_input_price_per_1m,
            "customer_output_price_ceiling":request.price_ceiling.max_output_price_per_1m,
            "eligible_models_after_constraints":selection.compatible_model_ids,
            "capable_models":selection.capable_model_ids,"selected_model":selection.model.model_id,
            "estimated_request_cost":selection.estimated_cost,"domain_chunks_classified":domain.chunks_classified,
            "llm_tier_chunks_classified":llm_tier.chunks_classified,
        }
        return RoutingDecision(request.request_id,domain.domain,tier_decision.tier,selection.model.model_id,
                               selection.model.provider,tier_decision.policy_used,
                               f"{tier_decision.reason}; cheapest compatible model meeting capability tier",audit)
