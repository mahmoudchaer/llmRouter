import time
from routing.models.small_llm import SmallLLMDomainClassifier,SmallLLMTierClassifier
from routing.models.tier_router import DedicatedTierRouter
from routing.pipeline.router import RuntimeRouter
from routing.policies.tier_decision import TierDecisionPolicy
from routing.schemas.request import CustomerPriceCeiling,HardRequirements,RoutingRequest
from routing.schemas.routing_result import DomainPrediction,LLMTierEstimate,TierRouterPrediction
from routing.selection.model_selector import ModelSelector
from routing.mock_registry import build_mock_registry


class FakeDomain(SmallLLMDomainClassifier):
    def classify_domain(self,text):time.sleep(.08);return DomainPrediction("code")
class FakeLLMTier(SmallLLMTierClassifier):
    def classify_tier(self,text):time.sleep(.08);return LLMTierEstimate(2)
class FakeTier(DedicatedTierRouter):
    def predict(self,request):time.sleep(.08);return TierRouterPrediction(3,.55,{"T1":.05,"T2":.15,"T3":.55,"T4":.25})


def test_signals_run_in_parallel_and_audit_is_complete():
    router=RuntimeRouter(FakeDomain(),FakeLLMTier(),FakeTier(),TierDecisionPolicy(),ModelSelector(),build_mock_registry())
    req=RoutingRequest("x","debug this race",CustomerPriceCeiling(1,3),HardRequirements(context_tokens=1000))
    start=time.perf_counter();result=router.route(req);elapsed=time.perf_counter()-start
    assert elapsed<.15
    assert result.final_tier==3 and result.selected_model=="mock/strong"
    for key in ["domain_llm_prediction","llm_tier_prediction","tier_router_prediction",
                "tier_router_probabilities","decision_policy_used","eligible_models_after_constraints",
                "customer_input_price_ceiling","selected_model"]:
        assert key in result.audit
