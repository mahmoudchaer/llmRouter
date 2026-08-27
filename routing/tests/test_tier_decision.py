import pytest
from routing.policies.confidence import ConfidencePolicyConfig
from routing.policies.tier_decision import TierDecisionPolicy
from routing.schemas.routing_result import LLMTierEstimate,TierRouterPrediction


def dedicated(tier,confidence):
    probs={f"T{i}":0.0 for i in range(1,4)};probs[f"T{tier}"]=1.0
    return TierRouterPrediction(tier,confidence,probs)


def test_agreement_is_used_directly():
    result=TierDecisionPolicy().decide(dedicated(2,.9),LLMTierEstimate(2))
    assert result.tier==2 and result.policy_used=="agreement"


def test_low_confidence_biases_upward():
    result=TierDecisionPolicy().decide(dedicated(1,.55),LLMTierEstimate(3))
    assert result.tier==3 and "safe_upward" in result.policy_used


def test_default_does_not_downgrade_against_llm_even_at_high_confidence():
    result=TierDecisionPolicy().decide(dedicated(2,.94),LLMTierEstimate(3))
    assert result.tier==3


def test_calibrated_override_must_be_explicit():
    policy=TierDecisionPolicy(ConfidencePolicyConfig(.85,.60),allow_high_confidence_downward_disagreement=True)
    assert policy.decide(dedicated(2,.94),LLMTierEstimate(3)).tier==2
